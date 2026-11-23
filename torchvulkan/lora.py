"""QLoRA-style LoRA adapters over weight-only int8/int4 quantized ``nn.Linear`` layers.

The base weights stay packed (int8 per-row / int4 group-64 asymmetric, format in
``torchvulkan.quantize``) and frozen; every wrapped linear runs

    y = quant_mm(x, Wq) + s * (x @ A^T) @ B^T,      s = alpha / r

with the trainable A/B adapters in fp16 (bf16 is unavailable on MoltenVK) and
B zero-initialised, so a freshly patched model reproduces the quantized model
exactly. Only the adapters (plus whatever else the user leaves trainable) get
gradients — the packed weights are plain non-parameter attributes and never
enter autograd.

The autograd pair ``_QuantLinear`` closes blocker #1 of the quant training path:
``weight_int8_mm`` / ``weight_int4_mm`` are registered on PrivateUse1 only (their
outputs have no grad_fn), so the train path goes through this Function instead:
forward is the existing op (M==1 quant-GEMV kernel; M>1 dequant+GEMM, or a plain
GEMM on an optional dequantized-weight cache), backward computes

    grad_x = grad_out @ deq(W)          (grad wrt the packed weights: none)

with ``deq(W)`` either taken from a per-layer fp16 cache (``dequant_cache=True``
— trades ~2 bytes/weight of VRAM for skipping the re-dequant) or recomputed
transiently per backward through ``torch.ops.torchvulkan.weight_dequant`` (the
default: zero persistent cost, per-layer dequant matrices are a few MB each).
CPU tensors follow the same math with plain aten ops (reference paths).

Typical use (transformers model on vulkan fp16):

    import torchvulkan
    from torchvulkan.lora import (patch_lora_linear, mark_only_lora_trainable,
                                  lora_parameters, AdamWFused16-compatible)

    patch_lora_linear(model, bits=4, group=64, r=16, alpha=32.0)
    mark_only_lora_trainable(model)
    opt = torchvulkan.optim.AdamWFused16(lora_parameters(model), lr=1e-4)
"""
import math

import torch

from .quantize import quantize_int4, quantize_int8, quantize_weight

_int8_mm_op = torch.ops.torchvulkan.weight_int8_mm
_int4_mm_op = torch.ops.torchvulkan.weight_int4_mm
_dequant_op = torch.ops.torchvulkan.weight_dequant


def dequantize_weight(w_packed, scales, zeros, is_int4, K, dtype=None):
    """Dequantize a packed weight-only matrix to ``dtype`` (kernel semantics).

    w_packed: (N, row_bytes) uint8; scales: (N, ngroups) fp32; zeros: (N, ngroups)
    fp32 for int4 (None for int8); K: the logical k-length (the packed rows may
    be padded beyond it); dtype: fp16/fp32 (default fp16). Vulkan tensors go
    through the private ``weight_dequant`` op, CPU tensors through the same
    nibble layout in plain aten (reference paths).
    """
    if w_packed.device.type == "vulkan":
        return _dequant_op(w_packed, scales, zeros, is_int4, K, dtype or torch.float16)
    wi = w_packed.detach().cpu().to(torch.int64)
    N, row_bytes = wi.shape
    sc = scales.detach().cpu()
    ngroups = sc.shape[1]
    if is_int4:
        q = torch.stack((wi & 0xF, (wi >> 4) & 0xF), dim=-1).view(N, row_bytes * 2)[:, :K]
        z = zeros.detach().cpu().double()
    else:
        q = wi - 128
        z = None
    group = K // ngroups
    centered = q.view(N, ngroups, group) - (z.unsqueeze(-1) if is_int4 else 0)
    deq = centered * sc.double().unsqueeze(-1)
    deq = deq.view(N, K)
    if dtype is not None:
        deq = deq.to(dtype)
    return deq


class _QuantLinear(torch.autograd.Function):
    """y = x @ deq(W)^T (+ bias) through the packed weight-only quant ops.

    forward:  M==1 rows -> the quant-GEMV kernel (streams the packed bytes);
              M>1 with a matching-dtype dequant cache -> plain GEMM on the cache
              (skips the per-call re-dequant of the M>1 op path);
              M>1 without a cache -> the private op (dequant+GEMM inside);
              CPU -> F.linear on a CPU-dequantized weight.
    backward: grad_x = grad_out @ deq(W)   (cache hit or transient weight_dequant)
              grad wrt the packed weights/scales/zeros: none (frozen base).
    """

    @staticmethod
    def forward(ctx, x, w_packed, scales, zeros, bias, dequant):
        K = x.shape[-1]
        rows = x.numel() // K
        N = w_packed.shape[0]
        ctx.x_shape = x.shape
        ctx.N = N
        ctx.dequant = dequant
        ctx.w_packed = w_packed
        ctx.scales = scales
        ctx.zeros = zeros
        ctx.bias = bias
        b = bias if bias is None or bias.dtype == x.dtype else bias.to(x.dtype)
        if dequant is not None and rows > 1 and x.device.type == "vulkan" \
                and dequant.dtype == x.dtype:
            out = torch.matmul(x.reshape(-1, K), dequant.t())
            if b is not None:
                out = out + b
            return out.reshape(*x.shape[:-1], N)
        if x.device.type != "vulkan":
            w = dequant if dequant is not None else dequantize_weight(
                w_packed, scales, zeros, zeros is not None, K, x.dtype)
            return torch.nn.functional.linear(x, w.to(x.dtype), b)
        if zeros is None:
            return _int8_mm_op(x, w_packed, scales, b)
        return _int4_mm_op(x, w_packed, scales, zeros, b)

    @staticmethod
    def backward(ctx, grad_out):
        K = ctx.x_shape[-1]
        deq = ctx.dequant
        if deq is None:
            deq = dequantize_weight(ctx.w_packed, ctx.scales, ctx.zeros,
                                    ctx.zeros is not None, K, grad_out.dtype)
        elif deq.dtype != grad_out.dtype:
            deq = deq.to(grad_out.dtype)
        g = grad_out.reshape(-1, ctx.N).contiguous()
        grad_x = torch.matmul(g, deq).reshape(ctx.x_shape) if ctx.needs_input_grad[0] else None
        grad_bias = g.sum(0) if ctx.needs_input_grad[4] else None
        return grad_x, None, None, None, grad_bias, None


def quantized_linear(x, w_packed, scales, zeros=None, bias=None, dequant=None):
    """Weight-only quantized matmul with a backward pass: y = x @ deq(W)^T (+ bias).

    x: (..., K) fp16/fp32; w_packed/scales/zeros: the packing of
    ``torchvulkan.quantize.quantize_weight`` (zeros only for int4); bias: (N,)
    in x.dtype (frozen — no gradient); dequant: optional (N, K) cache of the
    dequantized weight in x.dtype (speeds up M>1 forward and backward;
    recompute otherwise). The packed weights receive no gradient.
    """
    return _QuantLinear.apply(x, w_packed, scales, zeros, bias, dequant)


# ---------------------------------------------------------------- LoRA patcher

_LORA_ATTR = "_torchvulkan_lora_orig_forward"
_DEQ_CACHE_ATTR = "_lora_deq"

# the block-linear names of Qwen3-like decoder architectures (Llama/Qwen3/Mistral)
_DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj")


def _make_lora_forward(mod):
    packed, scales = mod.w_packed, mod.qscales
    zeros, bias = mod.qzeros, getattr(mod, "_lora_bias", None)
    A, B, scale = mod.lora_A, mod.lora_B, mod.lora_scaling
    K = mod.in_features
    is_int4 = zeros is not None

    def forward(x):
        deq = getattr(mod, _DEQ_CACHE_ATTR, None)
        if deq is None and mod.lora_dequant_cache:
            deq = dequantize_weight(packed, scales, zeros, is_int4, K, x.dtype)
            setattr(mod, _DEQ_CACHE_ATTR, deq)
        a = A if A.dtype == x.dtype else A.to(x.dtype)
        b = B if B.dtype == x.dtype else B.to(x.dtype)
        base = _QuantLinear.apply(x, packed, scales, zeros, bias, deq)
        delta = torch.matmul(torch.matmul(x, a.t()), b.t())
        return base + (scale * delta).to(base.dtype)

    return forward


def patch_lora_linear(model, bits=4, group=64, r=16, alpha=32.0,
                      targets=_DEFAULT_TARGETS, adapter_dtype=torch.float16,
                      dequant_cache=False) -> int:
    """Wrap every target ``nn.Linear`` of ``model`` with a frozen weight-only
    quantized base plus trainable LoRA adapters (in place, instance-level
    forward override like the fused_train.py patchers).

    The fp16/fp32 ``module.weight`` is quantized on CPU (deterministic, strict-
    safe) with ``quantize_weight(bits, group)``, kept as plain attributes
    (``w_packed``/``qscales``/``qzeros`` — never returned by state_dict() or
    moved by .to()), and the original weight/bias are released: the adapters
    train against the quantized base exactly like inference uses it. A/B get
    the PEFT init (A kaiming-uniform, B zeros); ``lora_scaling = alpha / r``.
    With ``dequant_cache=True`` each layer materializes its dequantized weight
    in the activation dtype on first use (~2 bytes per weight element of
    persistent VRAM; the 0.6B-model total is ~1.2 GB — watch the 4 GB budget;
    without the cache the backward recomputes it transiently per layer).

    Do NOT list a tied lm_head in ``targets`` (quantizing it would break the
    tie with the embedding). Returns the number of patched modules. Idempotent
    (already-patched modules are skipped).
    """
    assert bits in (4, 8)
    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or module.weight is None:
            continue
        if targets is not None and name.rsplit(".", 1)[-1] not in targets:
            continue
        if hasattr(module, _LORA_ATTR):
            continue

        w = module.weight.detach()
        dev = w.device
        if bits == 4:
            packed, scales, zeros = quantize_weight(w.float().cpu(), bits, group)
        else:
            packed, scales = quantize_weight(w.float().cpu(), bits, group)
            zeros = None
        module.w_packed = packed.to(dev)
        module.qscales = scales.to(dev)
        module.qzeros = None if zeros is None else zeros.to(dev)
        if module.bias is not None:
            module._lora_bias = module.bias.detach().to(device=dev, dtype=adapter_dtype)
            module.bias = None
        module.weight = None  # the fp16/fp32 base is released with the packing in place

        K, N = module.in_features, module.out_features
        A = torch.empty(r, K, dtype=adapter_dtype)
        torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        module.lora_A = torch.nn.Parameter(A.to(dev))
        module.lora_B = torch.nn.Parameter(
            torch.zeros(N, r, dtype=adapter_dtype, device=dev))
        module.lora_scaling = alpha / r
        module.lora_dequant_cache = dequant_cache

        orig = module.forward
        module.forward = _make_lora_forward(module)
        setattr(module, _LORA_ATTR, orig)
        patched += 1
    return patched


def unpatch_lora(model, restore_weight=True) -> int:
    """Remove the LoRA forward overrides (inverse of ``patch_lora_linear``).

    The adapters and packed state are deleted; with ``restore_weight=True`` and
    a materialized dequant cache the (dequantized) base weight is restored as a
    frozen Parameter — otherwise the module is left with ``weight=None`` and
    cannot run. Returns the number of restored modules.
    """
    restored = 0
    for module in model.modules():
        orig = getattr(module, _LORA_ATTR, None)
        if orig is None:
            continue
        module.forward = orig
        delattr(module, _LORA_ATTR)
        for attr in ("lora_A", "lora_B", "lora_scaling", "lora_dequant_cache",
                     "w_packed", "qscales", "qzeros", "_lora_bias"):
            if hasattr(module, attr):
                delattr(module, attr)
        deq = getattr(module, _DEQ_CACHE_ATTR, None)
        if hasattr(module, _DEQ_CACHE_ATTR):
            delattr(module, _DEQ_CACHE_ATTR)
        if restore_weight and deq is not None and module.weight is None:
            module.weight = torch.nn.Parameter(deq.detach().clone(), requires_grad=False)
        restored += 1
    return restored


def _is_lora_param(name):
    return ".lora_A" in name or ".lora_B" in name or name.startswith("lora_")


def lora_parameters(model):
    """The trainable adapter parameters (lora_A / lora_B of patched modules)."""
    return [p for n, p in model.named_parameters() if _is_lora_param(n)]


def lora_state_dict(model):
    """Adapter-only state dict (checkpoint just the deltas)."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if _is_lora_param(n)}


def mark_only_lora_trainable(model) -> int:
    """requires_grad_(False) on everything except the LoRA adapters (call after
    patching, before the optimizer is built). Returns the number of trainable
    adapter tensors."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if _is_lora_param(name):
            p.requires_grad_(True)
            n += 1
    return n
