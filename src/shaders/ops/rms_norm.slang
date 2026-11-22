module rms_norm;

// Fused RMSNorm: out[r, :] = x[r, :] * rsqrt(mean(x[r,:]^2) + eps) * w[:].
//
// Replaces the 9-kernel aten composition (cast fp32, mul, 2-stage mean, add,
// rsqrt, mul, cast back, weight mul) with one kernel: statistics and the
// weighted scale accumulate in fp32, one subgroup per row. Requires WG to be a
// multiple of the subgroup size and x contiguous (rows major, last dim =
// hidden).

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint WG = 0;
[[vk::constant_id(1)]] public const uint ROW_W = 0; // 1: w is (rows, hidden), per-row weights

public struct RmsNormPushConstants
{
    public uint64_t tensor_x;
    public uint64_t tensor_w;
    public uint64_t tensor_out;
    public uint rows;
    public uint hidden;
    public float eps;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void rms_norm_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform RmsNormPushConstants pcs)
{
    uint lanes = WaveGetLaneCount();
    uint wave = ti / lanes;
    uint lane = ti % lanes;
    uint waves = WG / lanes; // rows per workgroup

    uint r = wg.x * waves + wave;
    if (r >= pcs.rows) return;

    T* x = (T*)pcs.tensor_x + uint64_t(r) * pcs.hidden;
    T* w = (T*)pcs.tensor_w;
    T* out = (T*)pcs.tensor_out + uint64_t(r) * pcs.hidden;

    float acc = 0;
    for (uint j = lane; j < pcs.hidden; j += lanes)
        acc += castValue<T, float>(x[j]) * castValue<T, float>(x[j]);
    acc = WaveActiveSum(acc);

    float rrms = rsqrt(acc / float(pcs.hidden) + pcs.eps);
    uint w_row = (ROW_W != 0) ? r * pcs.hidden : 0;
    for (uint j = lane; j < pcs.hidden; j += lanes)
        out[j] = castValue<float, T>(castValue<T, float>(x[j]) * rrms * castValue<T, float>(w[w_row + j]));
}

// Train-path fused RMSNorm forward, saving the per-row rrms (fp32) for the
// backward kernel (rms_norm_bwd.slang). Replicates the transformers
// decomposition (Lfm2RMSNorm) op-for-op at the rounding level so train-path
// equivalence holds against eager autograd:
//   x32 = f32(x);  r = rsqrt(mean(x32^2) + eps)        (fp32 statistics)
//   rrms[r] = r                                        (saved for backward)
//   out[r,j] = round_T( f32(round_T(x32[r,j]*r)) * w[j] )
// (the eager chain materialises round_T(x32*r) via .to(input_dtype), then
// multiplies by the weight in T). w: (hidden,).
public struct RmsNormTrainPushConstants
{
    public uint64_t tensor_x;
    public uint64_t tensor_w;
    public uint64_t tensor_rrms; // (rows,) float
    public uint64_t tensor_out;
    public uint rows;
    public uint hidden;
    public float eps;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void rms_norm_train_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform RmsNormTrainPushConstants pcs)
{
    uint lanes = WaveGetLaneCount();
    uint wave = ti / lanes;
    uint lane = ti % lanes;
    uint waves = WG / lanes; // rows per workgroup

    uint r = wg.x * waves + wave;
    if (r >= pcs.rows) return;

    T* x = (T*)pcs.tensor_x + uint64_t(r) * pcs.hidden;
    T* w = (T*)pcs.tensor_w;
    float* rrms_out = (float*)pcs.tensor_rrms;
    T* out = (T*)pcs.tensor_out + uint64_t(r) * pcs.hidden;

    float acc = 0;
    for (uint j = lane; j < pcs.hidden; j += lanes)
        acc += castValue<T, float>(x[j]) * castValue<T, float>(x[j]);
    acc = WaveActiveSum(acc);

    float rr = rsqrt(acc / float(pcs.hidden) + pcs.eps);
    if (lane == 0) rrms_out[r] = rr;
    for (uint j = lane; j < pcs.hidden; j += lanes)
    {
        float xn = castValue<T, float>(x[j]) * rr;
        float xn16 = castValue<T, float>(castValue<float, T>(xn));
        out[j] = castValue<float, T>(xn16 * castValue<T, float>(w[j]));
    }
}

// Fused residual-add + RMSNorm with two outputs (decode hotpath): one kernel
// instead of aten add + rms_norm (two dispatches per decoder layer).
//   sum[r, :] = x1[r, :] + x2[r, :]                       (residual, fp16-rounded)
//   out[r, :] = fp16(sum)[r, :] * rsqrt(mean(sum^2)+eps) * w[:]
// Statistics in fp32; out normalizes the fp16-rounded sum so the result matches
// the split aten path (add writes fp16, rms reads it back) bit-close. w: (hidden,).
public struct RmsNormAddPushConstants
{
    public uint64_t tensor_x1;
    public uint64_t tensor_x2;
    public uint64_t tensor_w;
    public uint64_t tensor_sum;
    public uint64_t tensor_out;
    public uint rows;
    public uint hidden;
    public float eps;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void rms_norm_add_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform RmsNormAddPushConstants pcs)
{
    uint lanes = WaveGetLaneCount();
    uint wave = ti / lanes;
    uint lane = ti % lanes;
    uint waves = WG / lanes; // rows per workgroup

    uint r = wg.x * waves + wave;
    if (r >= pcs.rows) return;

    T* x1 = (T*)pcs.tensor_x1 + uint64_t(r) * pcs.hidden;
    T* x2 = (T*)pcs.tensor_x2 + uint64_t(r) * pcs.hidden;
    T* w = (T*)pcs.tensor_w;
    T* sum = (T*)pcs.tensor_sum + uint64_t(r) * pcs.hidden;
    T* out = (T*)pcs.tensor_out + uint64_t(r) * pcs.hidden;

    float acc = 0;
    for (uint j = lane; j < pcs.hidden; j += lanes)
    {
        float s = castValue<T, float>(x1[j]) + castValue<T, float>(x2[j]);
        T s16 = castValue<float, T>(s);
        sum[j] = s16;
        acc += castValue<T, float>(s16) * castValue<T, float>(s16);
    }
    acc = WaveActiveSum(acc);

    float rrms = rsqrt(acc / float(pcs.hidden) + pcs.eps);
    for (uint j = lane; j < pcs.hidden; j += lanes)
    {
        // повторно из x1/x2 (регистры, без чтения собственной записи в global):
        // то же fp16-округление суммы, что в sum[j]
        float s = castValue<T, float>(x1[j]) + castValue<T, float>(x2[j]);
        out[j] = castValue<float, T>(castValue<T, float>(castValue<float, T>(s)) * rrms * castValue<T, float>(w[j]));
    }
}
