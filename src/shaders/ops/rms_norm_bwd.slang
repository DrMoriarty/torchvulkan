module rms_norm_bwd;

// Train-path backward of the fused RMSNorm (rms_norm_train_main in
// rms_norm.slang), pair for the transformers decomposition (Lfm2RMSNorm):
//   x32 = f32(x); r = rsqrt(mean(x32^2)+eps); out = w * round_T(x32*r)
// The eager autograd graph rounds the mul-node products to T (fp16 in the
// training path), so the kernel replicates those roundings:
//   gxn   = round_T(g*w)                  (grad wrt the normed value)
//   dot_r = sum_j gxn_j * f32(x[r,j])     (fp32, groupshared reduction)
//   dx[r,j] = round_T(r*gxn - f32(x)*r^3*dot/n)
//   dwc[r,j] = round_T(g * round_T(f32(x)*r))  (per-row weight-grad term)
// rrms comes from the forward kernel (fp32, (rows,)) — recomputing it here
// would duplicate the reduction for no gain.
//
// grad_w is NOT folded here: the host reduces dwc over the rows with the
// ordinary aten sum — the very reduction the eager graph runs on the same
// (rows, hidden) fp16 term tensor, so grad_w matches the eager path bit-exact
// by construction (and the reduce kernel picks its own split/tree for long
// row counts instead of a serial per-column fold).

__exported import core.dtype;

public struct RmsNormBwdPushConstants
{
    public uint64_t tensor_g;    // grad_out (rows, hidden) T
    public uint64_t tensor_x;    // x (rows, hidden) T
    public uint64_t tensor_w;    // w (hidden,) T
    public uint64_t tensor_rrms; // (rows,) float
    public uint64_t tensor_dwc;  // (rows, hidden) T — per-row weight-grad terms
    public uint64_t tensor_dx;   // (rows, hidden) T
    public uint rows;
    public uint hidden;
}

// one workgroup of 128 threads per row (hidden arbitrary, strided loop);
// groupshared reduction, redundant per-thread dot like rms_rope.slang
groupshared float sm_red[128];

[shader("compute")]
[numthreads(128, 1, 1)]
void rms_norm_bwd_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform RmsNormBwdPushConstants pcs)
{
    uint row = wg.x;
    if (row >= pcs.rows) return;

    T* g = (T*)pcs.tensor_g + uint64_t(row) * pcs.hidden;
    T* x = (T*)pcs.tensor_x + uint64_t(row) * pcs.hidden;
    T* w = (T*)pcs.tensor_w;
    float* rrms_in = (float*)pcs.tensor_rrms;
    T* dwc = (T*)pcs.tensor_dwc + uint64_t(row) * pcs.hidden;
    T* dx = (T*)pcs.tensor_dx + uint64_t(row) * pcs.hidden;

    float r = rrms_in[row];

    float local = 0;
    for (uint j = ti; j < pcs.hidden; j += 128)
    {
        float g32 = castValue<T, float>(g[j]);
        float w32 = castValue<T, float>(w[j]);
        float gxn = castValue<T, float>(castValue<float, T>(g32 * w32));
        local += gxn * castValue<T, float>(x[j]);
    }
    sm_red[ti] = local;
    GroupMemoryBarrierWithGroupSync();
    float dot = 0;
    [ForceUnroll]
    for (uint i = 0; i < 128; i++)
        dot += sm_red[i];

    float r3 = r * r * r;
    float inv_n = 1.0 / float(pcs.hidden);
    for (uint j = ti; j < pcs.hidden; j += 128)
    {
        float g32 = castValue<T, float>(g[j]);
        float x32 = castValue<T, float>(x[j]);
        float w32 = castValue<T, float>(w[j]);
        float gxn = castValue<T, float>(castValue<float, T>(g32 * w32));
        float xn32 = x32 * r;
        float xn16 = castValue<T, float>(castValue<float, T>(xn32));
        float dxv = r * gxn - x32 * (r3 * dot * inv_n);
        dx[j] = castValue<float, T>(dxv);
        dwc[j] = castValue<float, T>(g32 * xn16);
    }
}
