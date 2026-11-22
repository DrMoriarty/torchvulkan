#pragma once
#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>

namespace torchvulkan {

// ops/binary.cpp
at::Tensor add_vulkan(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor multiply_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_vulkan(const at::Tensor& self, const at::Tensor& other);
at::Tensor divide_scalar_vulkan(const at::Tensor& self, const at::Scalar& other);

struct ComplexParts {
    at::Tensor re;
    at::Tensor im;
};
ComplexParts complex_parts(const at::Tensor& z);
at::Tensor complex_from(const ComplexParts& z, at::ScalarType dtype);
ComplexParts complex_scalar_parts(const at::Scalar& s, const at::Tensor& like);
ComplexParts complex_mul(const ComplexParts& a, const ComplexParts& b);
ComplexParts complex_div(const ComplexParts& a, const ComplexParts& b);
at::Tensor complex_abs(const ComplexParts& z);
ComplexParts complex_square(const ComplexParts& z);
ComplexParts complex_exp(const ComplexParts& z);
ComplexParts complex_log(const ComplexParts& z);
ComplexParts complex_sqrt(const ComplexParts& z);
ComplexParts complex_sin(const ComplexParts& z);
ComplexParts complex_cos(const ComplexParts& z);
ComplexParts complex_tan(const ComplexParts& z);
ComplexParts complex_sinh(const ComplexParts& z);
ComplexParts complex_cosh(const ComplexParts& z);
ComplexParts complex_tanh(const ComplexParts& z);
ComplexParts complex_asin(const ComplexParts& z);
ComplexParts complex_acos(const ComplexParts& z);
ComplexParts complex_atan(const ComplexParts& z);
ComplexParts complex_asinh(const ComplexParts& z);
ComplexParts complex_acosh(const ComplexParts& z);
ComplexParts complex_atanh(const ComplexParts& z);
ComplexParts complex_expm1(const ComplexParts& z);
ComplexParts complex_log1p(const ComplexParts& z);
ComplexParts complex_logaddexp(const ComplexParts& x, const ComplexParts& y);

// ops/reduce.cpp
at::Tensor sum_vulkan(const at::Tensor& self, c10::optional<at::ScalarType> dtype);

// ops/quant.cpp
at::Tensor weight_int8_mm(const at::Tensor& x, const at::Tensor& w_packed, const at::Tensor& scales,
    const c10::optional<at::Tensor>& bias, const c10::optional<double>& rms_eps,
    const c10::optional<at::Tensor>& rms_w, const c10::optional<at::Tensor>& acc,
    const c10::optional<at::Tensor>& x2);
at::Tensor weight_int4_mm(const at::Tensor& x, const at::Tensor& w_packed, const at::Tensor& scales,
    const at::Tensor& zeros, const c10::optional<at::Tensor>& bias, const c10::optional<double>& rms_eps,
    const c10::optional<at::Tensor>& rms_w, const c10::optional<at::Tensor>& acc,
    const c10::optional<at::Tensor>& x2);

// ops/fused.cpp
at::Tensor rms_norm_fused(const at::Tensor& x, const at::Tensor& w, double eps);
std::tuple<at::Tensor, at::Tensor> rms_norm_add(const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& w, double eps);
std::tuple<at::Tensor, at::Tensor> rms_norm_train_fwd(const at::Tensor& x, const at::Tensor& w, double eps);
std::tuple<at::Tensor, at::Tensor> rms_norm_train_bwd(const at::Tensor& grad_out, const at::Tensor& x,
    const at::Tensor& w, const at::Tensor& rrms);
at::Tensor rope_rotate_half(const at::Tensor& x, const at::Tensor& cos_t, const at::Tensor& sin_t);
// ops/fused.cpp — fused train-path rotate-half RoPE (q и k одним диспатчем);
// autograd-пара lives in torchvulkan/fused_train.py
std::tuple<at::Tensor, at::Tensor> rope_train_fwd(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& cos_t, const at::Tensor& sin_t);
std::tuple<at::Tensor, at::Tensor> rope_train_bwd(const at::Tensor& grad_q, const at::Tensor& grad_k,
    const at::Tensor& cos_t, const at::Tensor& sin_t);
at::Tensor rms_rope(const at::Tensor& x, const at::Tensor& cos_t, const at::Tensor& sin_t,
    const at::Tensor& w, double eps);
at::Tensor attn_decode(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, int64_t cur,
    const c10::optional<at::Tensor>& k_cur, const c10::optional<at::Tensor>& v_cur,
    const c10::optional<at::Tensor>& qkn_w, const c10::optional<at::Tensor>& cos_t,
    const c10::optional<at::Tensor>& sin_t, const c10::optional<double>& eps);

// ops/sdpa.cpp — fused causal SDPA (training path, A5)
bool sdpa_route_ok(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& attn_mask, double dropout_p, bool is_causal,
    bool enable_gqa);
std::tuple<at::Tensor, at::Tensor> sdpa_causal_fwd(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, double scale);
at::Tensor sdpa_rowdot(const at::Tensor& a, const at::Tensor& b);
at::Tensor sdpa_ds(const at::Tensor& probs, const at::Tensor& dP, const at::Tensor& delta,
    int64_t l, double scale);
std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_causal_backward(const at::Tensor& grad_out,
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& out,
    const at::Tensor& probs, double scale);
at::Tensor scaled_dot_product_attention_vulkan(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const c10::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, c10::optional<double> scale, bool enable_gqa);
std::tuple<at::Tensor, at::Tensor> scaled_dot_product_attention_math_vulkan(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& attn_mask, double dropout_p, bool is_causal,
    const c10::optional<at::Tensor>& dropout_mask, c10::optional<double> scale, bool enable_gqa);

// ops/binary.cpp
at::Tensor silu_mul_vulkan(const at::Tensor& g, const at::Tensor& u);

// ops/ce.cpp — fused train-path cross-entropy (lm_head); the autograd pair
// lives in torchvulkan/fused_train.py
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ce_fwd(const at::Tensor& h,
    const at::Tensor& weight, const at::Tensor& target, int64_t ignore_index);
std::tuple<at::Tensor, at::Tensor> ce_bwd(const at::Tensor& grad_rows, const at::Tensor& logits,
    const at::Tensor& mx, const at::Tensor& log_s, const at::Tensor& target, const at::Tensor& h,
    const at::Tensor& weight, int64_t ignore_index);

// ops/unary.cpp
at::Tensor gelu_native_vulkan(const at::Tensor& self);

// ops/matmul.cpp — fused re-dispatch of a cancelled addmm with the dual-output
// gelu epilogue (eager training fusion, see ops/fusion.cpp)
at::Tensor addmm_gelu_fused(const at::Tensor& input, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Scalar& beta, const at::Scalar& alpha, const at::Tensor& pre);

// tensor/allocation.cpp
void resize_bytes_vulkan(c10::StorageImpl* storage, size_t nbytes);
const at::Tensor& resize_vulkan(const at::Tensor& self, c10::IntArrayRef size, c10::optional<at::MemoryFormat> memory_format);

// tensor/transfer.cpp
at::Tensor copy_vulkan(const at::Tensor& self, const at::Tensor& dst, bool non_blocking);

// tensor/cast.cpp
void dispatch_cast_shader(const at::Tensor& src, const at::Tensor& dst);

} // namespace torchvulkan
