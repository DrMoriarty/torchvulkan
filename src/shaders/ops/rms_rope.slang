module rms_rope;

// Fused per-row RMSNorm + rotate-half RoPE (Qwen decode path): replaces the
// two-kernel sequence rms_norm -> rope_rotate_half on the qk block.
//
// x: (rows, hd) contiguous (the q|k block of the fused qkv GEMV output —
// rows 0..n_head-1 are q heads, rows n_head.. are k heads; the QK-norm
// weights are pre-merged into one (rows, hd) matrix with the attention
// scale folded into the q rows). cos/sin: (hd,) rows of the full tables
// (C = cat(c, c), S = cat(-s, s)).
//
// One workgroup of WG threads per row: phase 1 computes rrms in fp32 via a
// groupshared reduction and writes the normed value to groupshared scratch;
// phase 2 (after a barrier) applies rotate-half from the scratch:
//     out[i, j] = xn[j] * cos[j] + xn[pair(j)] * sin[j],
//     pair(j) = j < hd/2 ? j + hd/2 : j - hd/2.
//
// Requires hd <= WG (128 for Qwen3 head_dim). All math fp32. NB: variable
// names avoid MSL reserved words (`half`, `cos`, `sin`).

__exported import core.dtype;

public struct RmsRopePushConstants
{
    public uint64_t tensor_x;
    public uint64_t tensor_cos_tab;
    public uint64_t tensor_sin_tab;
    public uint64_t tensor_w;   // (rows, hd) RMS weights
    public uint64_t tensor_out;
    public float eps;
    public uint hd;
}

groupshared float smem_x[128];
groupshared float smem_red[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void rms_rope_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform RmsRopePushConstants pcs)
{
    uint row = wg.x;
    uint base = row * pcs.hd;
    T* xp = (T*)pcs.tensor_x;
    T* wp = (T*)pcs.tensor_w;
    T* out = (T*)pcs.tensor_out;

    float xi = 0.0;
    if (ti < pcs.hd)
        xi = castValue<T, float>(xp[base + ti]);

    // rrms строки: каждый поток комбинирует избыточно (как в attn_decode)
    smem_red[ti] = xi * xi;
    GroupMemoryBarrierWithGroupSync();
    float ss = 0.0;
    [ForceUnroll]
    for (uint i = 0; i < 128; i++)
        ss += smem_red[i];
    float rrms = rsqrt(ss / float(pcs.hd) + pcs.eps);

    float xn = xi * rrms * castValue<T, float>(wp[base + ti]);
    smem_x[ti] = xn;
    GroupMemoryBarrierWithGroupSync();

    if (ti >= pcs.hd) return;
    uint half_dim = pcs.hd / 2;
    uint pair = ti < half_dim ? ti + half_dim : ti - half_dim;
    T* cos_tab = (T*)pcs.tensor_cos_tab;
    T* sin_tab = (T*)pcs.tensor_sin_tab;
    float v = smem_x[ti] * castValue<T, float>(cos_tab[ti])
            + smem_x[pair] * castValue<T, float>(sin_tab[ti]);
    out[base + ti] = castValue<float, T>(v);
}
