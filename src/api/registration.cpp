#include <torch/extension.h>
#include <ATen/autocast_mode.h>
#include "api/ops/internal.h"
#include "api/ops/fusion.h"

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

// custom private ops (the torchvulkan analogue of ATen's _weight_int8pack_mm /
// _weight_int4pack_mm): weight-only quantized matmul, kernels in ops/quant.cpp;
// fused decoder-hotpath kernels, ops in ops/fused.cpp
TORCH_LIBRARY(torchvulkan, m) {
    m.def("weight_int8_mm(Tensor x, Tensor w_packed, Tensor scales, Tensor? bias=None, float? rms_eps=None, Tensor? rms_w=None, Tensor? acc=None, Tensor? x2=None) -> Tensor");
    m.def("weight_int4_mm(Tensor x, Tensor w_packed, Tensor scales, Tensor zeros, Tensor? bias=None, float? rms_eps=None, Tensor? rms_w=None, Tensor? acc=None, Tensor? x2=None) -> Tensor");
    // dequantize a packed weight-only matrix to fp16/fp32 (kernel semantics of
    // the M>1 path of weight_int*_mm); the train path (lora.py) uses it for
    // grad_x = grad_out @ deq(W) and optional per-layer dequant caches
    m.def("weight_dequant(Tensor w_packed, Tensor scales, Tensor? zeros, bool is_int4, int K, ScalarType out_dtype) -> Tensor");
    m.def("rms_norm(Tensor x, Tensor w, float eps) -> Tensor");
    m.def("rms_norm_add(Tensor x1, Tensor x2, Tensor w, float eps) -> (Tensor, Tensor)");
    // train-path fused RMSNorm (ops/fused.cpp): fwd saves the per-row rrms,
    // bwd consumes it — the autograd pair lives in torchvulkan/fused_train.py
    m.def("rms_norm_train_fwd(Tensor x, Tensor w, float eps) -> (Tensor, Tensor)");
    m.def("rms_norm_train_bwd(Tensor grad_out, Tensor x, Tensor w, Tensor rrms) -> (Tensor, Tensor)");
    // train-path fused cross-entropy (ops/ce.cpp): the autograd pair lives in
    // torchvulkan/fused_train.py; the only [rows, V] buffer is the fp16 GEMM
    // output (grad_logits overwrite it in place in bwd)
    m.def("ce_fwd(Tensor h, Tensor weight, Tensor target, int ignore_index=-100) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("ce_bwd(Tensor grad_rows, Tensor logits, Tensor max, Tensor log_s, Tensor target, Tensor h, Tensor weight, int ignore_index=-100) -> (Tensor, Tensor)");
    m.def("rope_rotate_half(Tensor x, Tensor cos, Tensor sin) -> Tensor");
    // train-path fused rotate-half RoPE (ops/fused.cpp): fwd и bwd — по одному
    // кернелу на q+k; autograd-пара lives in torchvulkan/fused_train.py
    m.def("rope_train_fwd(Tensor q, Tensor k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
    m.def("rope_train_bwd(Tensor grad_q, Tensor grad_k, Tensor cos, Tensor sin) -> (Tensor, Tensor)");
    // train-path fused depthwise causal conv1d of the LFM2 short-conv layers
    // (ops/dwconv.cpp): fwd consumes the in_proj output [B, T, 3C] in place;
    // the autograd pair lives in torchvulkan/fused_train.py
    m.def("dwconv_train_fwd(Tensor proj, Tensor w, Tensor? bias=None, int activation=0) -> (Tensor, Tensor)");
    m.def("dwconv_train_bwd(Tensor grad_out, Tensor proj, Tensor w, Tensor cache, int activation=0, bool with_bias=False) -> (Tensor, Tensor, Tensor)");
    m.def("rms_rope(Tensor x, Tensor cos, Tensor sin, Tensor w, float eps) -> Tensor");
    m.def("attn_decode(Tensor q, Tensor k_cache, Tensor v_cache, int cur, Tensor? k_cur=None, Tensor? v_cur=None, Tensor? qkn_w=None, Tensor? cos=None, Tensor? sin=None, float? eps=None) -> Tensor");
    m.def("silu_mul(Tensor g, Tensor u) -> Tensor");
    m.def("add_relu(Tensor a, Tensor b, float alpha=1.0, Tensor? pre=None) -> (Tensor, Tensor)");
    // fused causal SDPA internals (ops/sdpa.cpp): the private pieces behind the
    // aten::scaled_dot_product_attention AutogradPrivateUse1 override
    m.def("sdpa_fwd(Tensor q, Tensor k, Tensor v, float scale) -> (Tensor, Tensor)");
    m.def("sdpa_rowdot(Tensor a, Tensor b) -> Tensor");
    m.def("sdpa_ds(Tensor probs, Tensor dp, Tensor delta, int l, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchvulkan, PrivateUse1, m) {
    m.impl("weight_int8_mm", &torchvulkan::weight_int8_mm);
    m.impl("weight_int4_mm", &torchvulkan::weight_int4_mm);
    m.impl("weight_dequant", &torchvulkan::weight_dequant);
    m.impl("rms_norm", &torchvulkan::rms_norm_fused);
    m.impl("rms_norm_add", &torchvulkan::rms_norm_add);
    m.impl("rms_norm_train_fwd", &torchvulkan::rms_norm_train_fwd);
    m.impl("rms_norm_train_bwd", &torchvulkan::rms_norm_train_bwd);
    m.impl("ce_fwd", &torchvulkan::ce_fwd);
    m.impl("ce_bwd", &torchvulkan::ce_bwd);
    m.impl("rope_rotate_half", &torchvulkan::rope_rotate_half);
    m.impl("rope_train_fwd", &torchvulkan::rope_train_fwd);
    m.impl("rope_train_bwd", &torchvulkan::rope_train_bwd);
    m.impl("dwconv_train_fwd", &torchvulkan::dwconv_train_fwd);
    m.impl("dwconv_train_bwd", &torchvulkan::dwconv_train_bwd);
    m.impl("rms_rope", &torchvulkan::rms_rope);
    m.impl("attn_decode", &torchvulkan::attn_decode);
    m.impl("silu_mul", &torchvulkan::silu_mul_vulkan);
    m.impl("add_relu", &torchvulkan::add_relu);
    m.impl("sdpa_fwd", &torchvulkan::sdpa_causal_fwd);
    m.impl("sdpa_rowdot", &torchvulkan::sdpa_rowdot);
    m.impl("sdpa_ds", &torchvulkan::sdpa_ds);
}

// fused causal SDPA: the autograd-level override of the CompositeImplicitAutograd
// aten op (a backend key would never be reached) plus the PrivateUse1 override
// of the math kernel the composite falls into under no_grad; ops/sdpa.cpp
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("scaled_dot_product_attention", &torchvulkan::scaled_dot_product_attention_vulkan);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_scaled_dot_product_attention_math", &torchvulkan::scaled_dot_product_attention_math_vulkan);
}

namespace at::autocast {

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
#define _KERNEL_PRIVATEUSEONE_LOW_PRECISION_FP(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)
    AT_FORALL_LOWER_PRECISION_FP(_KERNEL_PRIVATEUSEONE_LOW_PRECISION_FP)
#define _KERNEL_PRIVATEUSEONE_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)
    AT_FORALL_FP32(_KERNEL_PRIVATEUSEONE_FP32)
#define _KERNEL_PRIVATEUSEONE_FP32_SET_OPT_DTYPE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)
    AT_FORALL_FP32_SET_OPT_DTYPE(_KERNEL_PRIVATEUSEONE_FP32_SET_OPT_DTYPE)
    AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)
#define _KERNEL_PRIVATEUSEONE_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)
    AT_FORALL_PROMOTE(_KERNEL_PRIVATEUSEONE_PROMOTE)
    KERNEL_PRIVATEUSEONE(linalg_matrix_sqrth, fp32)
}

}
