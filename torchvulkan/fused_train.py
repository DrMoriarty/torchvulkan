"""Train-path fused kernels with autograd pairs (private torchvulkan ops).

The decode hotpath kernels (rms_norm_add, rms_rope, ...) have no backward; the
training path needs fwd+bwd pairs. This module wraps them in
torch.autograd.Function so a training script can patch RMSNorm-like modules
(Lfm2RMSNorm, nn.RMSNorm) without touching the rest of the model.

Fused RMSNorm train: 3 dispatches per fwd+bwd instead of ~18 aten kernels
(~450 dispatches/iter in LFM2.5-230M B2 T128, see
experiments/lfm2_train_profile.md). The rounding pattern replicates the
transformers decomposition (Lfm2RMSNorm: x32 = x.to(f32);
r = rsqrt(mean(x32^2)+eps); out = w * round_T(x32*r)) so gradients match the
eager path at fp16-rounding level.
"""
import torch

_rms_fwd_op = torch.ops.torchvulkan.rms_norm_train_fwd
_rms_bwd_op = torch.ops.torchvulkan.rms_norm_train_bwd
_ce_fwd_op = torch.ops.torchvulkan.ce_fwd
_ce_bwd_op = torch.ops.torchvulkan.ce_bwd


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        out, rrms = _rms_fwd_op(x, weight, eps)
        ctx.save_for_backward(x, weight, rrms)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rrms = ctx.saved_tensors
        grad_x, grad_w = _rms_bwd_op(grad_out, x, weight, rrms)
        return grad_x, grad_w, None


def rms_norm_train(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused train-path RMSNorm: out = w * x * rsqrt(mean(x^2, dim=-1) + eps).

    x: (..., hidden) fp16/fp32 vulkan tensor; weight: (hidden,) same dtype.
    Forward is 1 dispatch, backward 2; statistics in fp32, rounding pattern
    replicates the eager transformers decomposition (Lfm2RMSNorm).
    """
    return _FusedRMSNorm.apply(x, weight, eps)


_RMS_NORM_CLASS_NAMES = ("Lfm2RMSNorm", "RMSNorm")


def _module_eps(module) -> float:
    for attr in ("variance_epsilon", "eps"):
        if hasattr(module, attr):
            return float(getattr(module, attr))
    return 1e-6


def patch_rms_norm_modules(model: torch.nn.Module, eps: float = None) -> int:
    """Replace the forward of RMSNorm-like modules with the fused train kernel.

    Patches every instance whose class is named Lfm2RMSNorm / RMSNorm (the
    transformers decoder norms and the attention q/k norms share the class).
    `eps` overrides the module's own; by default it is taken from the module
    (variance_epsilon / eps). Returns the number of patched modules.
    """
    patched = 0
    for module in model.modules():
        if type(module).__name__ not in _RMS_NORM_CLASS_NAMES or not hasattr(module, "weight"):
            continue
        e = eps if eps is not None else _module_eps(module)

        def make_forward(mod, e):
            def forward(hidden_states):
                return _FusedRMSNorm.apply(hidden_states, mod.weight, e)
            return forward

        module.forward = make_forward(module, e)
        patched += 1
    return patched


def unpatch_rms_norm_modules(model: torch.nn.Module) -> int:
    """Remove instance-level forward overrides installed by patch_rms_norm_modules."""
    removed = 0
    for module in model.modules():
        forward = module.__dict__.get("forward")
        if forward is not None and getattr(forward, "__qualname__", "").startswith("make_forward.<locals>"):
            del module.__dict__["forward"]
            removed += 1
    return removed


class _FusedCrossEntropy(torch.autograd.Function):
    """Fused chunked-memory cross-entropy over the lm_head projection.

    forward:  logits = h @ weight.T      (single fp16 [N, V] buffer, the only
                                          [N, V] materialization — no fp32 copy)
              loss_r = -((x32[t] - max) - log(sum(exp)))  (fused kernel, per row;
                                          the sum replays reduce_tree's order, so
                                          the loss is bit-identical to aten CE)
    backward: grad_logits = (softmax - onehot) * g_row   (fused kernel, in place
                                          over the saved logits buffer)
              grad_h   = grad_logits @ weight
              grad_W   = grad_logits.T @ h          (with tied embeddings this
                                          is the embedding gradient)
    5 dispatches per fwd+bwd instead of ~15 aten ops; peak CE memory is one
    [N, V] activation-dtype buffer (~256 MB fp16 at N=2048, V=65536) instead of
    ~1.5 GB of fp32 logits + intermediates.
    """

    @staticmethod
    def forward(ctx, h, weight, target, ignore_index, reduction):
        h2d = h.reshape(-1, h.shape[-1])
        tgt = target.reshape(-1)
        loss_rows, logits, mx, log_s = _ce_fwd_op(h2d, weight, tgt, ignore_index)
        ctx.save_for_backward(h2d, weight, mx, log_s, tgt)
        ctx.logits = logits  # consumed in place by the backward kernel
        ctx.h_shape = h.shape
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        if reduction == "none":
            return loss_rows.view(target.shape)
        if reduction == "sum":
            return loss_rows.sum()
        # 'mean': aten divides the sum by the number of non-ignored targets
        valid = (tgt != ignore_index).sum()
        return loss_rows.sum() / valid

    @staticmethod
    def backward(ctx, grad_out):
        h2d, weight, mx, log_s, tgt = ctx.saved_tensors
        n = tgt.numel()
        if ctx.reduction == "none":
            grad_rows = grad_out.reshape(-1).float()
        else:
            g = grad_out
            if ctx.reduction == "mean":
                valid = (tgt != ctx.ignore_index).sum()
                g = g / valid
            # materialize the uniform per-row scales as one fp32 vector
            grad_rows = g.float().expand(n).contiguous()
        grad_h2d, grad_w = _ce_bwd_op(
            grad_rows, ctx.logits, mx, log_s, tgt, h2d, weight, ctx.ignore_index)
        return grad_h2d.reshape(ctx.h_shape), grad_w, None, None, None


def fused_cross_entropy(h: torch.Tensor, weight: torch.Tensor, target: torch.Tensor,
                        ignore_index: int = -100, reduction: str = "mean") -> torch.Tensor:
    """Fused train-path cross-entropy over the lm_head projection.

    h: (..., H) fp16/fp32 vulkan hidden states; weight: (V, H) lm_head /
    tied-embedding matrix (logits = h @ weight.T); target: (...,) int64 labels
    aligned with the leading dims of h (values >= V or == ignore_index are
    ignored — only ignore_index semantics of aten cross_entropy, no class
    weights / label smoothing). reduction: 'mean' | 'sum' | 'none'.

    Replaces `cross_entropy(h @ weight.T, target)` without materializing fp32
    logits: peak CE memory is one [N, V] buffer in the dtype of h.
    """
    return _FusedCrossEntropy.apply(h, weight, target, ignore_index, reduction)
