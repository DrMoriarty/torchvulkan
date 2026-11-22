#pragma once
#include <ATen/ATen.h>
#include "vulkan/shader.h"

namespace torchvulkan {

// Eager one-op lookback fusion for the training hot paths (experiments/
// training_speedup.md Б1 + Б2). A producer op (aten add, addmm) records the
// span of pending shader ops it just queued plus its inputs; when the very next
// op on the same thread is a matching consumer (relu of the add's output, gelu
// of the addmm's output), the still-unsubmitted producer ops are cancelled and
// ONE fused kernel is dispatched instead — writing both the producer's output
// (the pre-activation, which the backward needs) and the activated result.
// Nothing outside the backend changes: `F.relu(out + shortcut)` and
// `nn.GELU()(nn.Linear(...)(x))` are matched natively.
//
// The mechanism is conservative: the consumer only fires when the producer's
// ops are still the tail of the pending queue (nothing was dispatched in
// between, not flushed yet, identity verified) and the shapes/dtypes fit the
// fused kernel; otherwise the ordinary separate kernels run. Kill switch:
// TORCHVULKAN_NO_FUSION=1 (A/B benchmarking).

// the span a producer captured from the shader manager
using FusionOpSpan = VulkanShaderManager::OpSpan;

// producers (call right after the op's shader dispatch succeeded)
void record_add_fusion(const at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
    const at::Scalar& alpha, const FusionOpSpan& span);
void record_addmm_fusion(const at::Tensor& out, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Tensor& bias, const at::Scalar& alpha, const at::Scalar& beta, const FusionOpSpan& span);

// consumers: return the activated result when fused (the producer's output
// tensor was filled in-place as the pre-activation), an undefined tensor when
// the caller must run its normal path
at::Tensor consume_add_relu(const at::Tensor& input);
at::Tensor consume_addmm_gelu(const at::Tensor& input);

// fused add + relu kernel: pre = a + alpha*b, out = relu(pre); fp16/fp32,
// equal shapes, contiguous, 16-byte-aligned. Returns (pre, out); `pre_dst`
// optionally receives the pre-activation in an existing buffer (the fusion
// path passes the add's already-returned output tensor).
std::tuple<at::Tensor, at::Tensor> add_relu_fused(const at::Tensor& a, const at::Tensor& b,
    const at::Scalar& alpha, const at::Tensor& pre_dst = {});

// the private-op wrapper: torch.ops.torchvulkan.add_relu(a, b, alpha, pre=None) -> (pre, out)
std::tuple<at::Tensor, at::Tensor> add_relu(const at::Tensor& a, const at::Tensor& b, double alpha,
    const c10::optional<at::Tensor>& pre = c10::nullopt);

// how many times each eager fusion actually fired since process start
std::tuple<uint64_t, uint64_t> fusion_stats();

} // namespace torchvulkan
