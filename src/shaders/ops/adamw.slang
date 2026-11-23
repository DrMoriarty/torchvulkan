module adamw;

// Fused AdamW/Adam optimizer step (experiments/lfm2_train_profile.md,
// приоритет 5): one kernel per parameter instead of the ~6-op eager
// decomposition (lerp_/addcmul_/addcdiv_/div_/pow/mul chains) the aten
// _fused_adamw_ override ran before. One flat elementwise dispatch updates
// param, exp_avg (m) and exp_avg_sq (v) in place.
//
// Types are pairwise: T — the dtype of param/grad/m, U — the dtype of v.
// The compact-state layout for 4 GB VRAM is T = fp16 (m = zeros_like(p) of
// torch.optim.AdamW fused=True), U = fp32 for v: v = g^2 spans five orders
// below the fp16 normal range (g ~ 1e-4 -> v ~ 1e-8), where fp16 flushes to
// zero and denom = sqrt(v)/bc2sqrt + eps collapses to eps -> updates of
// ~lr*m/eps — orders of magnitude too large (observed as a loss spike on a
// 100-step finetune). m is linear in g and safe in fp16. Math runs in fp32
// registers; bias corrections (1 - beta1^step, sqrt(1 - beta2^step)) are
// computed on the host in fp64 and pushed — the kernel never sees the step.
//
// Semantics match the eager path in ops/optim.cpp (adam_step):
//   adam  (mode ORIGINAL): g = g + wd * p            (p before the update)
//   adamw (mode ADAMW):    p = p * (1 - lr * wd)     (decoupled weight decay)
//   m  = beta1 * m  + (1 - beta1) * g
//   v  = beta2 * v  + (1 - beta2) * g * g
//   p  = p - (lr / bc1) * m / (sqrt(v) / bc2sqrt + eps)
// maximize negates the gradient first (flags bit 1).

__exported import core.dtype;
import core.indexing;

// bit 0: AdamW (decoupled weight decay), bit 1: maximize (negate the gradient)
public struct AdamWPushConstants
{
    public uint64_t tensor_p;
    public uint64_t tensor_g;
    public uint64_t tensor_m;
    public uint64_t tensor_v;
    public uint numel;
    public uint flags;
    public float bc1;     // 1 - beta1^step
    public float bc2sqrt; // sqrt(1 - beta2^step)
    public float lr;
    public float beta1;
    public float beta2;
    public float weight_decay;
    public float eps;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void adamw_main<T : ITorchVulkan, U : ITorchVulkan, let WG : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform AdamWPushConstants pcs)
{
    uint i = flat_group(wg) * WG + ti;
    if (i >= pcs.numel) return;

    T* p = (T*)pcs.tensor_p;
    T* g = (T*)pcs.tensor_g;
    T* m = (T*)pcs.tensor_m;
    U* v = (U*)pcs.tensor_v;

    float pv = castValue<T, float>(p[i]);
    float gv = castValue<T, float>(g[i]);
    if ((pcs.flags & 2u) != 0u) gv = -gv;
    float mv = castValue<T, float>(m[i]);
    float vv = castValue<U, float>(v[i]);

    if (pcs.weight_decay != 0.0)
    {
        if ((pcs.flags & 1u) != 0u) pv = pv * (1.0 - pcs.lr * pcs.weight_decay);
        else gv = gv + pcs.weight_decay * pv;
    }

    mv = pcs.beta1 * mv + (1.0 - pcs.beta1) * gv;
    vv = pcs.beta2 * vv + (1.0 - pcs.beta2) * gv * gv;

    float denom = sqrt(vv) / pcs.bc2sqrt + pcs.eps;
    pv = pv - (pcs.lr / pcs.bc1) * (mv / denom);

    p[i] = castValue<float, T>(pv);
    m[i] = castValue<float, T>(mv);
    v[i] = castValue<float, U>(vv);
}
