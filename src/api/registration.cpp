#include <torch/extension.h>
#include <ATen/autocast_mode.h>
#include "api/ops/internal.h"

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}

// custom private ops (the torchvulkan analogue of ATen's _weight_int8pack_mm /
// _weight_int4pack_mm): weight-only quantized matmul, kernels in ops/quant.cpp;
// fused decoder-hotpath kernels, ops in ops/fused.cpp
TORCH_LIBRARY(torchvulkan, m) {
    m.def("weight_int8_mm(Tensor x, Tensor w_packed, Tensor scales, Tensor? bias=None, float? rms_eps=None, Tensor? rms_w=None, Tensor? acc=None, Tensor? x2=None) -> Tensor");
    m.def("weight_int4_mm(Tensor x, Tensor w_packed, Tensor scales, Tensor zeros, Tensor? bias=None, float? rms_eps=None, Tensor? rms_w=None, Tensor? acc=None, Tensor? x2=None) -> Tensor");
    m.def("rms_norm(Tensor x, Tensor w, float eps) -> Tensor");
    m.def("rms_norm_add(Tensor x1, Tensor x2, Tensor w, float eps) -> (Tensor, Tensor)");
    m.def("rope_rotate_half(Tensor x, Tensor cos, Tensor sin) -> Tensor");
    m.def("rms_rope(Tensor x, Tensor cos, Tensor sin, Tensor w, float eps) -> Tensor");
    m.def("attn_decode(Tensor q, Tensor k_cache, Tensor v_cache, int cur, Tensor? k_cur=None, Tensor? v_cur=None, Tensor? qkn_w=None, Tensor? cos=None, Tensor? sin=None, float? eps=None) -> Tensor");
    m.def("silu_mul(Tensor g, Tensor u) -> Tensor");
}

TORCH_LIBRARY_IMPL(torchvulkan, PrivateUse1, m) {
    m.impl("weight_int8_mm", &torchvulkan::weight_int8_mm);
    m.impl("weight_int4_mm", &torchvulkan::weight_int4_mm);
    m.impl("rms_norm", &torchvulkan::rms_norm_fused);
    m.impl("rms_norm_add", &torchvulkan::rms_norm_add);
    m.impl("rope_rotate_half", &torchvulkan::rope_rotate_half);
    m.impl("rms_rope", &torchvulkan::rms_rope);
    m.impl("attn_decode", &torchvulkan::attn_decode);
    m.impl("silu_mul", &torchvulkan::silu_mul_vulkan);
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
