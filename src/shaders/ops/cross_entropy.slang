module cross_entropy;

// Fused train-path cross-entropy for the lm_head (vocabulary projection):
// replaces the eager chain mm -> .float() -> log_softmax -> nll_loss (and its
// backward) that materializes fp32 logits [B*T, V] plus several fp32
// intermediates (~1.5 GB at B*T=2048, V=65536 — OOM on 4 GB). Here the only
// [rows, V] buffer is the fp16 GEMM output itself; everything else is per-row.
//
//   ce_lse_main  (forward): per row — exact max (order-free), then the sum of
//                exp(x - max) accumulated in the EXACT order of the backend's
//                reduce_tree kernel (reduce.cpp tree_reduce_vulkan: n_chunks /
//                chunk_size computed on the host with the same formulas, phase-0
//                scalar per-thread scan with WG stride + halving groupshared
//                tree per chunk + phase-1 fold of the partials). Bit-matching
//                the aten log_softmax path is the point: the backward
//                grad_logits then round to fp16 identically and the two grad
//                GEMMs (ordinary aten mm) produce bit-identical gradients.
//                Also gathers the target logit and writes
//                loss[r] = -((x32[t] - max) - log_s)   (0 for ignored rows),
//                the very expression chain of log_softmax + nll_loss.
//                Saves max[r] and log_s[r] (fp32) for the backward kernel.
//                One workgroup (WG = reduce_tree's tree_wg = 128) per row.
//   ce_shift_main (backward, pass 1): shifted = fl(cast32(x) - max) per element
//                into an fp32 row-chunk scratch — the FIRST rounding of the
//                aten log_softmax chain, isolated in its own kernel (a single
//                binary op, nothing to reassociate).
//   ce_grad_main  (backward, pass 2): lp = fl(shifted - log_s); p = exp(lp);
//                grad_logits[r,v] = p*g[r] - onehot*g[r] — the exact arithmetic
//                of the aten log_softmax_backward for the nll grad
//                (res = go - exp(out) * sum(go)) — written in place over the
//                saved logits buffer; ignored rows are zeroed. The caller then
//                runs the two FULL grad GEMMs (grad_h = grad_logits @ W,
//                grad_W = grad_logits.T @ h) with ordinary aten mm — full, not
//                chunked, so the fp32 accumulation order over K matches aten's.
//
// The two backward passes are separate kernels ON PURPOSE: the Metal compiler
// reassociates straight-line fp chains inside one kernel ((x - mx) - log_s
// became x - (mx + log_s), moving an ulp and flipping ~1e-6 of the fp16
// roundings); across a kernel boundary it cannot. The row-chunked fp32 scratch
// (default ~128 MB) is the only extra memory; the eager chain instead
// materializes fp32 logits [B*T, V] plus sub/exp/log/nll intermediates.
//
// WG is a compile-time constant: ce_lse needs reduce_tree's 128 (the halving
// tree and the WG-stride scan define the summation order), the pointwise
// kernels use 256.

__exported import core.dtype;

public struct CeLsePushConstants
{
    public uint64_t tensor_logits; // (rows, V) T, contiguous
    public uint64_t tensor_target; // (rows,) int64
    public uint64_t tensor_max;    // (rows,) float — out
    public uint64_t tensor_logs;   // (rows,) float — out (log of the sum of exp)
    public uint64_t tensor_loss;   // (rows,) float — out (0 for ignored rows)
    public uint V;
    public uint rows;
    public int ignore_index;
    public uint n_chunks;          // tree_reduce_vulkan phase structure, replicated
    public uint chunk_size;
}

groupshared float sm_m[128];
groupshared float sm_v[128];
groupshared float sm_p[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void ce_lse_main<T : ITorchVulkan, let WG : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform CeLsePushConstants pcs)
{
    uint r = wg.x;
    if (r >= pcs.rows) return;

    T* lin = (T*)pcs.tensor_logits + uint64_t(r) * pcs.V;

    // pass 1: the exact row max (a max is order-free; the aten reference takes
    // it with amax, whose value does not depend on the reduction order)
    float mx = -3.4e38;
    for (uint j = ti; j < pcs.V; j += WG)
        mx = max(mx, castValue<T, float>(lin[j]));
    sm_m[ti] = mx;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG >> 1; s > 0; s >>= 1)
    {
        if (ti < s) sm_m[ti] = max(sm_m[ti], sm_m[ti + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    mx = sm_m[0];

    // pass 2: sum of exp(x - max) chunk by chunk, each chunk scanned with the
    // WG stride and tree-folded exactly like reduce_tree phase 0 (fp32 input
    // takes its scalar branch)
    for (uint c = 0; c < pcs.n_chunks; c++)
    {
        uint base = c * pcs.chunk_size;
        uint stop = min(base + pcs.chunk_size, pcs.V);
        float acc = 0.0;
        for (uint j = base + ti; j < stop; j += WG)
            acc += exp(castValue<T, float>(lin[j]) - mx);
        sm_v[ti] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG >> 1; s > 0; s >>= 1)
        {
            if (ti < s) sm_v[ti] = sm_v[ti] + sm_v[ti + s];
            GroupMemoryBarrierWithGroupSync();
        }
        if (ti == 0) sm_p[c] = sm_v[0];
        GroupMemoryBarrierWithGroupSync();
    }

    // phase 1 fold of the n_chunks partials (reduce_tree phase 1: identity
    // start, direct assignment of the partials, same halving tree)
    float best = 0.0;
    if (ti < pcs.n_chunks) best = sm_p[ti];
    sm_v[ti] = best;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG >> 1; s > 0; s >>= 1)
    {
        if (ti < s) sm_v[ti] = sm_v[ti] + sm_v[ti + s];
        GroupMemoryBarrierWithGroupSync();
    }

    if (ti == 0)
    {
        float log_s = log(sm_v[0]);
        float* max_out = (float*)pcs.tensor_max;
        float* logs_out = (float*)pcs.tensor_logs;
        float* loss_out = (float*)pcs.tensor_loss;
        int64_t* tgt = (int64_t*)pcs.tensor_target;
        int64_t t = tgt[r];
        max_out[r] = mx;
        logs_out[r] = log_s;
        // log_softmax: lp = (x - max) - log(sum); nll: loss = -lp[target]
        loss_out[r] = (t == int64_t(pcs.ignore_index)) ? 0.0
            : -((castValue<T, float>(lin[uint64_t(t)]) - mx) - log_s);
    }
}

// Tile transpose [R, C] -> [C, R] (both row-major), fp16/fp32. Used to feed
// the lm_head GEMM its weight as a contiguous [H, V] matrix: the GEMM's
// transposed-B path (strides (1, H)) loses vectorized B loads and runs the
// same accumulation ~3x slower; the accumulation ORDER does not depend on the
// operand layout, so the transposed feed is bit-identical (verified against
// mm(h, weight.t())). The generic permute+clone copy path in the backend runs
// this at ~0.4 GB/s, hence the dedicated kernel: 32x32 fp16/fp32 tiles through
// padded groupshared (bank-conflict-free both phases).
public struct CeTransposePushConstants
{
    public uint64_t tensor_src; // (R, C) T row-major
    public uint64_t tensor_dst; // (C, R) T row-major — out
    public uint R;
    public uint C;
}

groupshared float sm_t[32 * 33]; // float-страйд 33 = 32 + 1 (анти-bank-conflict)

[shader("compute")]
[numthreads(32, 32, 1)]
void ce_transpose_main<T : ITorchVulkan>(
    uint3 tid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    uniform CeTransposePushConstants pcs)
{
    T* src = (T*)pcs.tensor_src;
    T* dst = (T*)pcs.tensor_dst;
    uint src_row = gid.y * 32 + tid.y;
    uint src_col = gid.x * 32 + tid.x;
    if (src_row < pcs.R && src_col < pcs.C)
        sm_t[tid.y * 33 + tid.x] = castValue<T, float>(src[uint64_t(src_row) * pcs.C + src_col]);
    GroupMemoryBarrierWithGroupSync();
    // храним в транспонированном порядке: поток (tid.y, tid.x) пишет
    // dst[gid.x*32 + tid.y, gid.y*32 + tid.x] — коалесированно по tid.x
    uint dst_row = gid.x * 32 + tid.y;
    uint dst_col = gid.y * 32 + tid.x;
    if (dst_row < pcs.C && dst_col < pcs.R)
        dst[uint64_t(dst_row) * pcs.R + dst_col] = castValue<float, T>(sm_t[tid.x * 33 + tid.y]);
}

public struct CeShiftPushConstants
{
    public uint64_t tensor_logits; // (chunk_rows, V) T — fp16/fp32 logits of the row chunk
    public uint64_t tensor_max;    // (rows_total,) float (row = row0 + local row)
    public uint64_t tensor_shift;  // (chunk_rows, V) float — out: fl(x32 - max)
    public uint V;
    public uint rows;              // rows in this chunk
    public uint row0;              // global row offset of the chunk
}

// The FIRST rounding of the aten log_softmax chain, isolated in its own kernel:
// shifted = fl(cast32(x) - max). A single binary op — nothing for the Metal
// compiler to reassociate, so the bits match aten's sub kernel exactly.
[shader("compute")]
[numthreads(256, 1, 1)]
void ce_shift_main<T : ITorchVulkan, let WG : int>(
    uint3 tid : SV_DispatchThreadID,
    uniform CeShiftPushConstants pcs)
{
    uint col = tid.x;
    uint r = tid.y;
    if (r >= pcs.rows || col >= pcs.V) return;

    T* xin = (T*)pcs.tensor_logits + (uint64_t(r) * pcs.V + col);
    float* mxp = (float*)pcs.tensor_max;
    float* sout = (float*)pcs.tensor_shift + (uint64_t(r) * pcs.V + col);
    sout[0] = castValue<T, float>(xin[0]) - mxp[pcs.row0 + r];
}

public struct CeGradPushConstants
{
    public uint64_t tensor_shift;  // (chunk_rows, V) float — fl(x32 - max) from ce_shift
    public uint64_t tensor_logs;   // (rows_total,) float
    public uint64_t tensor_target; // (rows_total,) int64
    public uint64_t tensor_g;      // (rows_total,) float — per-row grad scale
    public uint64_t tensor_out;    // (chunk_rows, V) T — out: grad_logits (the saved logits buffer)
    public uint V;
    public uint rows;
    public int ignore_index;
    public uint row0;
}

// The SECOND rounding + the rest of the aten backward, in its own kernel (the
// chain must not merge with ce_shift's sub — the Metal compiler reassociates
// straight-line fp chains inside one kernel, which moves an ulp and flips fp16
// roundings on ~1e-6 of the elements; across two kernels it cannot).
//   lp = fl(shifted - log_s);  p = exp(lp);
//   grad_logits = p*g - onehot*g   (log_softmax_backward for the nll grad,
//   res = go - exp(out)*sum(go), see the header comment) — rounded once to T.
[shader("compute")]
[numthreads(256, 1, 1)]
void ce_grad_main<T : ITorchVulkan, let WG : int>(
    uint3 tid : SV_DispatchThreadID,
    uniform CeGradPushConstants pcs)
{
    uint col = tid.x;
    uint r = tid.y;
    if (r >= pcs.rows || col >= pcs.V) return;

    float* sin = (float*)pcs.tensor_shift + (uint64_t(r) * pcs.V + col);
    float* lsp = (float*)pcs.tensor_logs;
    int64_t* tgt = (int64_t*)pcs.tensor_target;
    float* g = (float*)pcs.tensor_g;
    T* out = (T*)pcs.tensor_out + (uint64_t(r) * pcs.V + col);

    float gl;
    if (tgt[pcs.row0 + r] == int64_t(pcs.ignore_index))
    {
        gl = 0.0;
    }
    else
    {
        float lp = sin[0] - lsp[pcs.row0 + r];
        float pr = exp(lp);
        gl = pr * g[pcs.row0 + r] - ((uint64_t(tgt[pcs.row0 + r]) == col) ? g[pcs.row0 + r] : 0.0);
    }
    out[0] = castValue<float, T>(gl);
}
