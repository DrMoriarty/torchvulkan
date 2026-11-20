module rope_half;

// Fused rotate-half RoPE (Qwen/HF style) for precomputed full-size tables:
// for x (rows, hd) contiguous and cos/sin (hd,) rows of the full tables
// (C = cat(c, c), S = cat(-s, s)),
//
//     out[i, j] = x[i, j] * cos[j] + x[i, pair(j)] * sin[j],
//     pair(j) = j < hd/2 ? j + hd/2 : j - hd/2.
//
// Replaces the 4-kernel aten composition (mul, cat, mul, add). hd must be
// even; one thread per element, math in fp32. NB: member/variable names avoid
// MSL reserved words (`half`, `cos`, `sin`).

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint WG = 0;

public struct RopeHalfPushConstants
{
    public uint64_t tensor_x;
    public uint64_t tensor_cos_tab;
    public uint64_t tensor_sin_tab;
    public uint64_t tensor_out;
    public uint total; // rows * hd
    public uint hd;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void rope_half_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform RopeHalfPushConstants pcs)
{
    uint idx = gi.x;
    if (idx >= pcs.total) return;

    uint half_dim = pcs.hd / 2;
    uint j = idx % pcs.hd;
    uint pair_j = j < half_dim ? j + half_dim : j - half_dim;

    T* x = (T*)pcs.tensor_x;
    T* cos_tab = (T*)pcs.tensor_cos_tab;
    T* sin_tab = (T*)pcs.tensor_sin_tab;
    T* out = (T*)pcs.tensor_out;

    float v = castValue<T, float>(x[idx]) * castValue<T, float>(cos_tab[j])
            + castValue<T, float>(x[idx - j + pair_j]) * castValue<T, float>(sin_tab[j]);
    out[idx] = castValue<float, T>(v);
}
