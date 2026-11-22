module sdpa;

// Fused causal SDPA (training path) replacing the math decomposition
// (_scaled_dot_product_attention_math): bmm + materialized B*H*T*T scores +
// a dozen pointwise/reduction kernels fwd, and the autograd graph over the
// same ops bwd. Three kernels:
//
//   sdpa_fwd_main   QK^T (causal) + softmax + P*V in one kernel, one
//                   workgroup of WG threads per q row (b, h, i). Two passes,
//                   no score row in groupshared memory: pass 1 streams the
//                   thread's scores (j = ti, ti+WG, ...) keeping only the
//                   running max and denominator in registers (online rescale),
//                   pass 2 recomputes the same scores (K is L1/L2-hot), writes
//                   the normalized prob row and accumulates out[dim = ti]
//                   against a WG-wide chunk of weights exchanged through a
//                   128-float groupshared scratch. The old single-pass design
//                   kept the whole score row (up to 4096 floats = 16 KB) in
//                   groupshared memory, which capped occupancy at ~3 WG/CU on
//                   the 64 KB LDS of the 5300M and made the kernel
//                   latency-bound (~0.44 us per row); with the 512-byte
//                   scratch the same shape runs several times faster.
//                   Outputs: out (B, Hq, L, hd) and probs P (B, Hkv, qpg, L, S)
//                   fp16/fp32 — P is what the backward kernels consume
//                   (dS = P*(dP - delta), dV = P^T @ dO), so no logsumexp
//                   and no score recompute in backward.
//   sdpa_rowdot_main  delta[r] = sum_d dO[r, d] * out[r, d] (one thread per row)
//   sdpa_ds_main      dS = P * (dP - delta[row]) with the causal entries zeroed
//
// Semantics match the math path: the mask is j <= i (at::ones(L,S).tril(),
// top-left aligned, also for L != S), the effective scale folded into the
// scores is calculate_scale (1/sqrt(head_dim) or the user's value). GQA maps
// q head h to kv group g = h / qpg — the same per-head math the math path's
// repeat_interleave expansion produces. All accumulation fp32.
//
// Backward GEMMs (dP = dO @ V^T, dQ = dS @ K, dK = dS^T @ Q, dV = P^T @ dO)
// stay on the regular matmul kernels: with the grouped views
// (B, Hkv, qpg*L, {S | hd}) all four are B*Hkv-batched and need no expansion.

__exported import core.dtype;
import core.indexing;

static const uint HD_CAP = 128;

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

public struct SdpaFwdPushConstants
{
    public uint64_t tensor_q;     // (B, Hq, L, hd) contiguous
    public uint64_t tensor_k;     // (B, Hkv, S, hd) contiguous
    public uint64_t tensor_v;     // (B, Hkv, S, hd) contiguous
    public uint64_t tensor_out;   // (B, Hq, L, hd)
    public uint64_t tensor_probs; // (B, Hkv, qpg, L, S)
    public uint rows;             // B * Hq * L
    public uint hq;
    public uint hkv;
    public uint qpg;              // Hq / Hkv (1 when there is no GQA)
    public uint l_dim;            // L (<= SMEM_L_CAP)
    public uint s_dim;            // S (<= SMEM_L_CAP)
    public uint hd;               // (<= HD_CAP)
    public uint rows_x;           // row-groups per y slice (grid y-split)
    public float scale;           // effective multiplier on q.k^T
}

groupshared float smem_red[128];   // lane reduction scratch (max / den / weights)

[shader("compute")]
[numthreads(128, 1, 1)]
void sdpa_fwd_main<T : ITorchVulkan, let WG : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform SdpaFwdPushConstants pcs)
{
    uint row = wg.y * pcs.rows_x + wg.x;
    if (row >= pcs.rows) return;

    uint bl = pcs.hq * pcs.l_dim;
    uint b_idx = row / bl;
    uint rem = row - b_idx * bl;
    uint h = rem / pcs.l_dim;
    uint i = rem - h * pcs.l_dim;
    uint g = h / pcs.qpg;
    uint jmax = min(i + 1, pcs.s_dim); // causal: j <= i (and j < S)

    T* K = (T*)pcs.tensor_k + uint64_t(b_idx * pcs.hkv + g) * pcs.s_dim * pcs.hd;
    T* V = (T*)pcs.tensor_v + uint64_t(b_idx * pcs.hkv + g) * pcs.s_dim * pcs.hd;
    T* prob_row = (T*)pcs.tensor_probs
        + uint64_t((b_idx * pcs.hkv + g) * pcs.qpg + (h - g * pcs.qpg)) * pcs.l_dim * pcs.s_dim
        + uint64_t(i) * pcs.s_dim;

    // q адрес строки: значения читаются из L1 прямо в FMA-цепочках (все
    // потоки WG читают один адрес — broadcast), q НЕ держится в регистрах:
    // массив hd float на поток резал occupancy по VGPR сильнее, чем
    // groupshared скоры по LDS
    bool vec = (sizeof(T) == 2) && (pcs.hd % 8 == 0);
    uint wcnt = vec ? pcs.hd / 8 : 0;
    uint4* q4 = (uint4*)pcs.tensor_q;
    uint qbase4 = row * wcnt;

    // pass 1: stream the thread's scores (j = ti, ti+WG, ...), keeping only
    // the running max and denominator in registers — no score row in
    // groupshared memory, so the WG scratch stays at 128 floats
    float m_lane = -3.4e38;
    float den_lane = 0.0;
    for (uint j = ti; j < jmax; j += WG)
    {
        float acc;
        if (vec)
        {
            uint4* k4 = (uint4*)pcs.tensor_k;
            uint base4 = (b_idx * pcs.hkv + g) * pcs.s_dim * wcnt + j * wcnt;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            uint c = 0;
            for (; c + 3 < wcnt; c += 4)
            {
                uint4 w0 = k4[base4 + c + 0];
                uint4 w1 = k4[base4 + c + 1];
                uint4 w2 = k4[base4 + c + 2];
                uint4 w3 = k4[base4 + c + 3];
                uint4 p0 = q4[qbase4 + c + 0];
                uint4 p1 = q4[qbase4 + c + 1];
                uint4 p2 = q4[qbase4 + c + 2];
                uint4 p3 = q4[qbase4 + c + 3];
                float4 l0 = f16tof32(w0), h0 = f16tof32(w0 >> 16);
                float4 l1 = f16tof32(w1), h1 = f16tof32(w1 >> 16);
                float4 l2 = f16tof32(w2), h2 = f16tof32(w2 >> 16);
                float4 l3 = f16tof32(w3), h3 = f16tof32(w3 >> 16);
                float4 u0 = f16tof32(p0), v0 = f16tof32(p0 >> 16);
                float4 u1 = f16tof32(p1), v1 = f16tof32(p1 >> 16);
                float4 u2 = f16tof32(p2), v2 = f16tof32(p2 >> 16);
                float4 u3 = f16tof32(p3), v3 = f16tof32(p3 >> 16);
                a0 += u0.x * l0.x + v0.x * h0.x;
                a1 += u0.y * l0.y + v0.y * h0.y;
                a2 += u0.z * l0.z + v0.z * h0.z;
                a3 += u0.w * l0.w + v0.w * h0.w;
                a0 += u1.x * l1.x + v1.x * h1.x;
                a1 += u1.y * l1.y + v1.y * h1.y;
                a2 += u1.z * l1.z + v1.z * h1.z;
                a3 += u1.w * l1.w + v1.w * h1.w;
                a0 += u2.x * l2.x + v2.x * h2.x;
                a1 += u2.y * l2.y + v2.y * h2.y;
                a2 += u2.z * l2.z + v2.z * h2.z;
                a3 += u2.w * l2.w + v2.w * h2.w;
                a0 += u3.x * l3.x + v3.x * h3.x;
                a1 += u3.y * l3.y + v3.y * h3.y;
                a2 += u3.z * l3.z + v3.z * h3.z;
                a3 += u3.w * l3.w + v3.w * h3.w;
            }
            acc = (a0 + a1) + (a2 + a3);
            for (; c < wcnt; c++)
            {
                uint4 w = k4[base4 + c];
                uint4 p = q4[qbase4 + c];
                float4 lo = f16tof32(w), hi = f16tof32(w >> 16);
                float4 ql = f16tof32(p), qh = f16tof32(p >> 16);
                acc += ql.x * lo.x + qh.x * hi.x
                     + ql.y * lo.y + qh.y * hi.y
                     + ql.z * lo.z + qh.z * hi.z
                     + ql.w * lo.w + qh.w * hi.w;
            }
        }
        else
        {
            T* qrow = (T*)pcs.tensor_q + uint64_t(row) * pcs.hd;
            T* krow = K + uint64_t(j) * pcs.hd;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            uint d = 0;
            for (; d + 3 < pcs.hd; d += 4)
            {
                a0 += castValue<T, float>(qrow[d]) * castValue<T, float>(krow[d]);
                a1 += castValue<T, float>(qrow[d + 1]) * castValue<T, float>(krow[d + 1]);
                a2 += castValue<T, float>(qrow[d + 2]) * castValue<T, float>(krow[d + 2]);
                a3 += castValue<T, float>(qrow[d + 3]) * castValue<T, float>(krow[d + 3]);
            }
            acc = (a0 + a1) + (a2 + a3);
            for (; d < pcs.hd; d++)
                acc += castValue<T, float>(qrow[d]) * castValue<T, float>(krow[d]);
        }
        acc *= pcs.scale;
        // online softmax update of the lane's running (max, denominator)
        float m_new = max(m_lane, acc);
        den_lane = den_lane * exp(m_lane - m_new) + exp(acc - m_new);
        m_lane = m_new;
    }

    // group reduction of the lane (max, den) pairs into the row's (m, den):
    // каждый поток сканирует весь scratch сам (как в исходном кернеле) —
    // без serial-редукции одним потоком; m_glob/den_glob каждый поток
    // вычисляет локально, broadcast не нужен
    smem_red[ti] = m_lane;
    GroupMemoryBarrierWithGroupSync();
    float m_glob = -3.4e38;
    [ForceUnroll]
    for (uint t = 0; t < WG; t++) m_glob = max(m_glob, smem_red[t]);
    // барьер обязателен: ниже каждый поток ПЕРЕЗАПИСЫВАЕТ свой слот, а чужие
    // сканы m_glob могли ещё не дочитать его
    GroupMemoryBarrierWithGroupSync();
    // exp(m_lane - m_glob) <= 1; a lane with no scores (m_lane = -3.4e38,
    // den_lane = 0) contributes exp(0)*0 = 0
    smem_red[ti] = den_lane * exp(m_lane - m_glob);
    GroupMemoryBarrierWithGroupSync();
    float den_tot = 0;
    [ForceUnroll]
    for (uint t = 0; t < WG; t++) den_tot += smem_red[t];
    float inv_den = 1.0 / den_tot;
    // и перед перезаписью scratch весами chunk'а pass 2 — сканы den_tot тоже
    // могли не завершиться
    GroupMemoryBarrierWithGroupSync();

    // pass 2, chunked over the keys: recompute this thread's score (K is
    // L1/L2-hot from pass 1), write the normalized prob (zero past the causal
    // boundary — dV/dS read the full row), exchange the chunk's weights
    // through the scratch and accumulate out[dim = ti] against them
    float out_acc = 0.0;
    for (uint c0 = 0; c0 < pcs.s_dim; c0 += WG)
    {
        uint j = c0 + ti;
        float w = 0.0;
        if (j < jmax)
        {
            float s;
            if (vec)
            {
                uint4* k4 = (uint4*)pcs.tensor_k;
                uint base4 = (b_idx * pcs.hkv + g) * pcs.s_dim * wcnt + j * wcnt;
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                uint c = 0;
                for (; c + 3 < wcnt; c += 4)
                {
                    uint4 w0 = k4[base4 + c + 0];
                    uint4 w1 = k4[base4 + c + 1];
                    uint4 w2 = k4[base4 + c + 2];
                    uint4 w3 = k4[base4 + c + 3];
                    uint4 p0 = q4[qbase4 + c + 0];
                    uint4 p1 = q4[qbase4 + c + 1];
                    uint4 p2 = q4[qbase4 + c + 2];
                    uint4 p3 = q4[qbase4 + c + 3];
                    float4 l0 = f16tof32(w0), h0 = f16tof32(w0 >> 16);
                    float4 l1 = f16tof32(w1), h1 = f16tof32(w1 >> 16);
                    float4 l2 = f16tof32(w2), h2 = f16tof32(w2 >> 16);
                    float4 l3 = f16tof32(w3), h3 = f16tof32(w3 >> 16);
                    float4 u0 = f16tof32(p0), v0 = f16tof32(p0 >> 16);
                    float4 u1 = f16tof32(p1), v1 = f16tof32(p1 >> 16);
                    float4 u2 = f16tof32(p2), v2 = f16tof32(p2 >> 16);
                    float4 u3 = f16tof32(p3), v3 = f16tof32(p3 >> 16);
                    a0 += u0.x * l0.x + v0.x * h0.x;
                    a1 += u0.y * l0.y + v0.y * h0.y;
                    a2 += u0.z * l0.z + v0.z * h0.z;
                    a3 += u0.w * l0.w + v0.w * h0.w;
                    a0 += u1.x * l1.x + v1.x * h1.x;
                    a1 += u1.y * l1.y + v1.y * h1.y;
                    a2 += u1.z * l1.z + v1.z * h1.z;
                    a3 += u1.w * l1.w + v1.w * h1.w;
                    a0 += u2.x * l2.x + v2.x * h2.x;
                    a1 += u2.y * l2.y + v2.y * h2.y;
                    a2 += u2.z * l2.z + v2.z * h2.z;
                    a3 += u2.w * l2.w + v2.w * h2.w;
                    a0 += u3.x * l3.x + v3.x * h3.x;
                    a1 += u3.y * l3.y + v3.y * h3.y;
                    a2 += u3.z * l3.z + v3.z * h3.z;
                    a3 += u3.w * l3.w + v3.w * h3.w;
                }
                s = (a0 + a1) + (a2 + a3);
                for (; c < wcnt; c++)
                {
                    uint4 wk = k4[base4 + c];
                    uint4 pk = q4[qbase4 + c];
                    float4 lo = f16tof32(wk), hi = f16tof32(wk >> 16);
                    float4 ql = f16tof32(pk), qh = f16tof32(pk >> 16);
                    s += ql.x * lo.x + qh.x * hi.x
                       + ql.y * lo.y + qh.y * hi.y
                       + ql.z * lo.z + qh.z * hi.z
                       + ql.w * lo.w + qh.w * hi.w;
                }
            }
            else
            {
                T* qrow = (T*)pcs.tensor_q + uint64_t(row) * pcs.hd;
                T* krow = K + uint64_t(j) * pcs.hd;
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                uint d = 0;
                for (; d + 3 < pcs.hd; d += 4)
                {
                    a0 += castValue<T, float>(qrow[d]) * castValue<T, float>(krow[d]);
                    a1 += castValue<T, float>(qrow[d + 1]) * castValue<T, float>(krow[d + 1]);
                    a2 += castValue<T, float>(qrow[d + 2]) * castValue<T, float>(krow[d + 2]);
                    a3 += castValue<T, float>(qrow[d + 3]) * castValue<T, float>(krow[d + 3]);
                }
                s = (a0 + a1) + (a2 + a3);
                for (; d < pcs.hd; d++)
                    s += castValue<T, float>(qrow[d]) * castValue<T, float>(krow[d]);
            }
            w = exp(s * pcs.scale - m_glob);
        }
        if (j < pcs.s_dim) prob_row[j] = castValue<float, T>(w * inv_den);
        smem_red[ti] = w;
        GroupMemoryBarrierWithGroupSync();

        // thread ti owns output dim ti over this chunk's keys
        if (ti < pcs.hd && jmax > c0)
        {
            uint jstop = min(jmax - c0, WG);
            for (uint jj = 0; jj < jstop; jj++)
                out_acc += smem_red[jj] * castValue<T, float>(V[uint64_t(c0 + jj) * pcs.hd + ti]);
        }
        GroupMemoryBarrierWithGroupSync(); // smem_red is reused by the next chunk
    }

    if (ti < pcs.hd)
    {
        T* out_row = (T*)pcs.tensor_out + uint64_t(row) * pcs.hd;
        out_row[ti] = castValue<float, T>(out_acc * inv_den);
    }
}

// ---------------------------------------------------------------------------
// delta: delta[r] = sum_d dO[r, d] * out[r, d], one thread per row
// ---------------------------------------------------------------------------

public struct SdpaRowDotPushConstants
{
    public uint64_t tensor_a; // (rows, hd)
    public uint64_t tensor_b; // (rows, hd)
    public uint64_t tensor_out; // (rows) float
    public uint rows;
    public uint hd;
    public uint blocks_x; // row-groups per y slice (grid y-split)
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sdpa_rowdot_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform SdpaRowDotPushConstants pcs)
{
    uint row = (wg.y * pcs.blocks_x + wg.x) * 128 + ti;
    if (row >= pcs.rows) return;

    float acc = 0;
    if ((sizeof(T) == 2) && (pcs.hd % 8 == 0))
    {
        uint wcnt = pcs.hd / 8;
        uint4* a4 = (uint4*)pcs.tensor_a;
        uint4* b4 = (uint4*)pcs.tensor_b;
        uint base4 = row * wcnt;
        for (uint c = 0; c < wcnt; c++)
        {
            uint4 wa = a4[base4 + c];
            uint4 wb = b4[base4 + c];
            float4 la = f16tof32(wa), ha = f16tof32(wa >> 16);
            float4 lb = f16tof32(wb), hb = f16tof32(wb >> 16);
            acc += la.x * lb.x + ha.x * hb.x + la.y * lb.y + ha.y * hb.y
                 + la.z * lb.z + ha.z * hb.z + la.w * lb.w + ha.w * hb.w;
        }
    }
    else
    {
        T* arow = (T*)pcs.tensor_a + uint64_t(row) * pcs.hd;
        T* brow = (T*)pcs.tensor_b + uint64_t(row) * pcs.hd;
        for (uint d = 0; d < pcs.hd; d++)
            acc += castValue<T, float>(arow[d]) * castValue<T, float>(brow[d]);
    }
    float* out = (float*)pcs.tensor_out;
    out[row] = acc;
}

// ---------------------------------------------------------------------------
// dS = scale * P * (dP - delta[row]) with the causal entries zeroed.
// The forward folded the scale into the scores, so the chain back to q/k is
// ds_ij/dq_i = scale * k_j — one extra multiply here covers both dQ and dK
// (dV reads P directly and needs no scale).
// dP comes in the same dtype as P: the fp16 GEMM accumulates in fp32 and
// rounds only its output, so dP = dO @ V^T fp16 keeps the backward on fast
// fp16 GEMMs with no fp32 materialization of dO/V; the dS product itself is
// computed in fp32 here and stored fp16.
// ---------------------------------------------------------------------------

public struct SdpaDsPushConstants
{
    public uint64_t tensor_probs; // (B, Hkv, qpg, L, S) T
    public uint64_t tensor_dp;    // same layout, T
    public uint64_t tensor_delta; // (B * Hq * L) float, rows of probs
    public uint64_t tensor_out;   // dS, same layout as probs, T
    public uint numel;
    public uint s_dim;
    public uint l_dim;
    public uint blocks_x;         // element-blocks per y slice (grid y-split)
    public float scale;           // effective multiplier on q.k^T
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sdpa_ds_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform SdpaDsPushConstants pcs)
{
    // one element block of 8 per thread in the vector branch, one element in
    // the scalar one; blockIdx layout identical for both
    bool vec = (sizeof(T) == 2) && (pcs.s_dim % 8 == 0);
    uint per_thread = vec ? 8 : 1;
    uint blk = (wg.y * pcs.blocks_x + wg.x) * 128 + ti;
    uint64_t first = uint64_t(blk) * per_thread;
    if (first >= pcs.numel) return;

    if (vec)
    {
        uint row = uint(first / pcs.s_dim);
        uint j = uint(first - uint64_t(row) * pcs.s_dim);
        uint i = row % pcs.l_dim;
        uint4* p4 = (uint4*)pcs.tensor_probs;
        uint4* d4 = (uint4*)pcs.tensor_dp;
        uint4* o4 = (uint4*)pcs.tensor_out;
        float* delta = (float*)pcs.tensor_delta;
        uint word = uint(blk);
        uint4 wp = p4[word];
        uint4 wd = d4[word];
        float4 lp = f16tof32(wp), hp = f16tof32(wp >> 16);
        // dP в том же T, что и probs: те же пары lo/hi по словам
        float4 ld = f16tof32(wd), dd = f16tof32(wd >> 16);
        float dl = delta[row];
        float p0 = (j + 0 <= i) ? lp.x : 0.0;
        float p1 = (j + 1 <= i) ? hp.x : 0.0;
        float p2 = (j + 2 <= i) ? lp.y : 0.0;
        float p3 = (j + 3 <= i) ? hp.y : 0.0;
        float p4v = (j + 4 <= i) ? lp.z : 0.0;
        float p5 = (j + 5 <= i) ? hp.z : 0.0;
        float p6 = (j + 6 <= i) ? lp.w : 0.0;
        float p7 = (j + 7 <= i) ? hp.w : 0.0;
        // dP распаковывается так же, как probs: в слове чётный элемент в
        // младших битах (ld), нечётный — в старших (dd): j+0 -> ld.x,
        // j+1 -> dd.x, j+2 -> ld.y, ...
        float r0 = p0 * (ld.x - dl) * pcs.scale;
        float r1 = p1 * (dd.x - dl) * pcs.scale;
        float r2 = p2 * (ld.y - dl) * pcs.scale;
        float r3 = p3 * (dd.y - dl) * pcs.scale;
        float r4 = p4v * (ld.z - dl) * pcs.scale;
        float r5 = p5 * (dd.z - dl) * pcs.scale;
        float r6 = p6 * (ld.w - dl) * pcs.scale;
        float r7 = p7 * (dd.w - dl) * pcs.scale;
        uint words[4];
        words[0] = f32tof16(r0) | (f32tof16(r1) << 16);
        words[1] = f32tof16(r2) | (f32tof16(r3) << 16);
        words[2] = f32tof16(r4) | (f32tof16(r5) << 16);
        words[3] = f32tof16(r6) | (f32tof16(r7) << 16);
        o4[word] = uint4(words[0], words[1], words[2], words[3]);
    }
    else
    {
        uint idx = uint(first);
        uint row = idx / pcs.s_dim;
        uint j = idx - row * pcs.s_dim;
        uint i = row % pcs.l_dim;
        T* p = (T*)pcs.tensor_probs;
        T* dp = (T*)pcs.tensor_dp;
        T* out = (T*)pcs.tensor_out;
        float* delta = (float*)pcs.tensor_delta;
        float pv = (j <= i) ? castValue<T, float>(p[idx]) : 0.0;
        float dsv = pv * (castValue<T, float>(dp[idx]) - delta[row]) * pcs.scale;
        out[idx] = castValue<float, T>(dsv);
    }
}
