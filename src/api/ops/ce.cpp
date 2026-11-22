#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"

// Fused train-path cross-entropy for the lm_head projection (the analogue of
// the aten chain mm -> float() -> cross_entropy and its backward). The eager
// chain materializes fp32 logits [B*T, V] plus several fp32 intermediates
// (~1.5 GB at B*T=2048, V=65536 — OOM on 4 GB VRAM); here the only [rows, V]
// buffer is the fp16 GEMM output itself, everything else is per-row fp32.
//
// forward (ce_fwd):
//   logits = h @ W.T                       — one aten mm, [rows, V] in the
//                                            dtype of h (the saved buffer)
//   ce_lse kernel                          — per-row max + sum(exp) accumulated
//                                            in the exact order of the backend's
//                                            reduce_tree kernel (so the loss and
//                                            the softmax probs are bit-identical
//                                            to the aten log_softmax path) +
//                                            target-logit gather + per-row loss
// backward (ce_bwd):
//   ce_grad kernel                         — grad_logits = p*g - onehot*g,
//                                            p = exp((x32 - max) - log_s): the
//                                            exact arithmetic of the aten
//                                            log_softmax_backward for the nll
//                                            grad; written in place over the
//                                            saved logits
//   grad_h   = grad_logits @ W             — aten mm, [rows, V] @ [V, H]
//   grad_W   = grad_logits.T @ h           — aten mm, [V, rows] @ [rows, H]
//                                            (with tied embeddings this IS the
//                                            embedding gradient)
// 5 dispatches per fwd+bwd instead of ~15, no fp32 copy of [rows, V].
//
// The reduction (mean/sum/none) and the ignore_index bookkeeping of the loss
// live in the Python autograd pair (torchvulkan/fused_train.py): it passes
// per-row grad scales (already divided by the valid count for 'mean') and
// reduces the per-row loss.

namespace {

// ce_lse must match reduce_tree's workgroup (the halving tree and the
// WG-stride scan define the summation order); the pointwise kernels use 256.
constexpr uint32_t CE_LSE_WG = 128;
constexpr uint32_t CE_GRAD_WG = 256;
// fp32 scratch budget for the row-chunked backward (~128 MB)
constexpr uint64_t CE_SCRATCH_BYTES = 128ull << 20;

void check_ce_dtype(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.is_privateuseone(), "torchvulkan [ERROR]: ", name, " must be a vulkan tensor, got ", t.device(), ".");
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat,
        "torchvulkan [ERROR]: ", name, " must be fp16 or fp32, got ", t.scalar_type(), ".");
}

// The phase structure of tree_reduce_vulkan (src/api/ops/reduce.cpp) for
// sum over the last dim of (num_lines, V): the ce_lse kernel replays exactly
// this chunking so the fp32 sum bits match the aten log_softmax path.
void tree_chunk_params(int64_t num_lines, int64_t V, int64_t& n_chunks, int64_t& chunk_size)
{
    constexpr int64_t chunk_target = 256;  // элементов на чанк в фазе 0
    constexpr int64_t grid_target = 4096;  // WG фазы 0 суммарно
    constexpr int64_t tree_wg = 128;
    if (V >= 2048) {
        n_chunks = std::min<int64_t>((V + chunk_target - 1) / chunk_target,
                                     (grid_target + num_lines - 1) / num_lines);
        n_chunks = std::min<int64_t>(std::max<int64_t>(n_chunks, 1), tree_wg);
        chunk_size = (V + n_chunks - 1) / n_chunks;
        chunk_size = (chunk_size + 15) / 16 * 16;
        n_chunks = (V + chunk_size - 1) / chunk_size;
    } else {
        n_chunks = 1;
        chunk_size = V;
    }
}

} // namespace

namespace torchvulkan {

// Forward: (per-row loss fp32 [rows], logits [rows, V] (saved for backward),
// per-row max fp32 [rows], per-row log(sum(exp)) fp32 [rows]).
// h: (..., H); weight: (V, H) — the lm_head / tied-embedding matrix,
// logits = h @ weight.T; target: (...,) int64 aligned with the rows of h.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ce_fwd(const at::Tensor& h_,
    const at::Tensor& weight_, const at::Tensor& target_, int64_t ignore_index)
{
    at::AutoGradMode guard(false);
    check_ce_dtype(h_, "h");
    check_ce_dtype(weight_, "weight");
    TORCH_CHECK(h_.scalar_type() == weight_.scalar_type(),
        "torchvulkan [ERROR]: ce_fwd dtype mismatch: h (", h_.scalar_type(), ") vs weight (", weight_.scalar_type(), ").");
    TORCH_CHECK(target_.is_privateuseone() && target_.scalar_type() == at::kLong,
        "torchvulkan [ERROR]: ce_fwd target must be a vulkan int64 tensor, got ",
        target_.device(), " ", target_.scalar_type(), ".");

    at::Tensor h = h_.contiguous();
    at::Tensor weight = weight_.contiguous();
    at::Tensor target = target_.contiguous();

    const int64_t H = h.size(-1);
    TORCH_CHECK(H >= 1, "torchvulkan [ERROR]: ce_fwd needs a non-empty last dimension.");
    const int64_t N = h.numel() / H;
    const int64_t V = weight.size(0);
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == H,
        "torchvulkan [ERROR]: ce_fwd weight must be (V, H) = (V, ", H, "), got ", weight.sizes(), ".");
    TORCH_CHECK(target.numel() == N,
        "torchvulkan [ERROR]: ce_fwd target (", target.sizes(), ") must have one entry per row (", N, ").");

    at::Tensor h2d = h.view({N, H});
    DeviceContext* device = get_device_context(h2d);
    TORCH_CHECK(CE_LSE_WG <= device->properties.limits.maxComputeWorkGroupInvocations,
        "torchvulkan [ERROR]: ce_fwd workgroup of ", CE_LSE_WG, " exceeds the device limit.");
    TORCH_CHECK(1024 <= device->properties.limits.maxComputeWorkGroupInvocations,
        "torchvulkan [ERROR]: ce_fwd transpose workgroup of 1024 exceeds the device limit.");
    // the single [N, V] materialization: fp16/fp32 GEMM output, reused as the
    // grad_logits buffer in backward.
    // Бит-равно и до ~3x быстрее на широких словарях: у GEMM-кернела
    // transposed-B (weight.t(), страйды (1, H)) падает векторизация загрузок B
    // (~0.3 TFLOP/s при N>=1024, V=65536); пре-транспонирование веса в
    // row-major [H, V] включает vec-aligned contiguous-B путь (~0.95 TFLOP/s).
    // Порядок накопления K у вариантов одинаков — значения бит-в-бит равны
    // mm(h, weight.t()) (проверено в test_cross_entropy_fused и изолированно).
    // Родной permute+clone для такого транспонирования даёт ~0.4 ГБ/с, поэтому
    // свой тайловый кернел (CE_TRANSPOSE, ~3-5 мс на 128 МБ fp16).
    at::Tensor logits;
    if (N >= 256) {
        at::Tensor w_t = at::empty({H, V}, weight.options());
        {
            SpecializationBuilder spd{}; // WG — compile-time константа (config.py)
            PushConstantBuilder pcs{};
            pcs.push(get_tensor_address(weight));
            pcs.push(get_output_address(w_t));
            pcs.push(static_cast<uint32_t>(V)); // R
            pcs.push(static_cast<uint32_t>(H)); // C

            device->shader_manager->dispatchShader(
                ShaderKey{torchvulkan::Kernel::CE_TRANSPOSE, weight.scalar_type()},
                device->default_subgroup_size,
                spd.build(),
                pcs.build(),
                /* read = */ {weight},
                /* write = */ {w_t},
                (static_cast<uint32_t>(H) + 31) / 32, (static_cast<uint32_t>(V) + 31) / 32, 1
            );
        }
        logits = at::matmul(h2d, w_t);
    } else {
        logits = at::matmul(h2d, weight.t());
    }
    at::Tensor mx = at::empty({N}, h.options().dtype(at::kFloat));
    at::Tensor log_s = at::empty({N}, h.options().dtype(at::kFloat));
    at::Tensor loss = at::empty({N}, h.options().dtype(at::kFloat));
    if (N == 0 || V == 0) return {loss, logits, mx, log_s};

    TORCH_CHECK(N <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: ce_fwd row count ", N, " exceeds the device workgroup limit.");
    TORCH_CHECK(N <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: ce_fwd row count ", N, " exceeds the device workgroup limit.");
    // the exact-order replay assumes the reference sum takes the tree path
    // (the gates of tree_reduce_vulkan: contiguous fp32 input, last-dim reduce,
    // lines fit the phase-0 grid, V >= 32)
    TORCH_CHECK(N <= (int64_t)(32768ULL < device->properties.limits.maxComputeWorkGroupCount[0]
            ? 32768ULL : device->properties.limits.maxComputeWorkGroupCount[0])
            * (int64_t)device->properties.limits.maxComputeWorkGroupCount[1]
        && V >= 32,
        "torchvulkan [ERROR]: ce_fwd shape (rows ", N, ", V ", V, ") is outside the exact-order tree path.");
    materialize_cow(h2d);
    materialize_cow(weight);
    materialize_cow(target);

    int64_t n_chunks = 0, chunk_size = 0;
    tree_chunk_params(N, V, n_chunks, chunk_size);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(logits));
    pcs.push(get_tensor_address(target));
    pcs.push(get_output_address(mx));
    pcs.push(get_output_address(log_s));
    pcs.push(get_output_address(loss));
    pcs.push(static_cast<uint32_t>(V));
    pcs.push(static_cast<uint32_t>(N));
    pcs.push(static_cast<int32_t>(ignore_index));
    pcs.push(static_cast<uint32_t>(n_chunks));
    pcs.push(static_cast<uint32_t>(chunk_size));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::CE_LSE, logits.scalar_type()},
        device->default_subgroup_size,
        SpecializationBuilder{}.build(),
        pcs.build(),
        /* read = */ {logits, target},
        /* write = */ {mx, log_s, loss},
        static_cast<uint32_t>(N), 1, 1
    );

    return {loss, logits, mx, log_s};
}

// Backward: (grad_h [..., H], grad_weight [V, H]) for ce_fwd above.
// grad_rows: (rows,) fp32 per-row gradient scale (the reduction — division by
// the valid count for 'mean' — applied by the caller); logits: the buffer
// saved by ce_fwd, overwritten in place with grad_logits; mx/log_s: the
// per-row max and log(sum(exp)) saved by ce_fwd.
std::tuple<at::Tensor, at::Tensor> ce_bwd(const at::Tensor& grad_rows_, const at::Tensor& logits_,
    const at::Tensor& mx_, const at::Tensor& log_s_, const at::Tensor& target_,
    const at::Tensor& h_, const at::Tensor& weight_, int64_t ignore_index)
{
    at::AutoGradMode guard(false);
    check_ce_dtype(h_, "h");
    check_ce_dtype(weight_, "weight");
    check_ce_dtype(logits_, "logits");
    TORCH_CHECK(h_.scalar_type() == weight_.scalar_type() && h_.scalar_type() == logits_.scalar_type(),
        "torchvulkan [ERROR]: ce_bwd dtype mismatch: h (", h_.scalar_type(), "), weight (",
        weight_.scalar_type(), "), logits (", logits_.scalar_type(), ").");
    TORCH_CHECK(grad_rows_.is_privateuseone() && grad_rows_.scalar_type() == at::kFloat,
        "torchvulkan [ERROR]: ce_bwd grad_rows must be a vulkan fp32 tensor, got ",
        grad_rows_.device(), " ", grad_rows_.scalar_type(), ".");
    TORCH_CHECK(mx_.is_privateuseone() && mx_.scalar_type() == at::kFloat &&
        log_s_.is_privateuseone() && log_s_.scalar_type() == at::kFloat,
        "torchvulkan [ERROR]: ce_bwd max/log_s must be vulkan fp32 tensors, got ",
        mx_.device(), " ", mx_.scalar_type(), ", ", log_s_.device(), " ", log_s_.scalar_type(), ".");
    TORCH_CHECK(target_.is_privateuseone() && target_.scalar_type() == at::kLong,
        "torchvulkan [ERROR]: ce_bwd target must be a vulkan int64 tensor, got ",
        target_.device(), " ", target_.scalar_type(), ".");

    at::Tensor h = h_.contiguous();
    at::Tensor weight = weight_.contiguous();
    // overwritten in place with grad_logits; a non-contiguous saved buffer
    // would be copied here (the copy becomes the scratch)
    at::Tensor logits = logits_.contiguous();
    at::Tensor mx = mx_.contiguous();
    at::Tensor log_s = log_s_.contiguous();
    at::Tensor target = target_.contiguous();
    at::Tensor grad_rows = grad_rows_.contiguous();

    const int64_t H = h.size(-1);
    const int64_t N = h.numel() / H;
    const int64_t V = weight.size(0);
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == H,
        "torchvulkan [ERROR]: ce_bwd weight must be (V, H), got ", weight.sizes(), ".");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == N && logits.size(1) == V,
        "torchvulkan [ERROR]: ce_bwd logits (", logits.sizes(), ") must be (", N, ", ", V, ").");
    TORCH_CHECK(mx.numel() == N && log_s.numel() == N && target.numel() == N && grad_rows.numel() == N,
        "torchvulkan [ERROR]: ce_bwd max/log_s/target/grad_rows must have ", N, " entries each.");

    at::Tensor h2d = h.view({N, H});
    at::Tensor grad_h = at::empty({N, H}, h.options());
    at::Tensor grad_w = at::empty({V, H}, weight.options());
    if (N == 0 || V == 0) return {grad_h, grad_w};

    DeviceContext* device = get_device_context(h2d);
    TORCH_CHECK(CE_GRAD_WG <= device->properties.limits.maxComputeWorkGroupInvocations,
        "torchvulkan [ERROR]: ce_bwd workgroup of ", CE_GRAD_WG, " exceeds the device limit.");
    TORCH_CHECK(N <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[1],
        "torchvulkan [ERROR]: ce_bwd row count ", N, " exceeds the device workgroup count limit.");
    materialize_cow(logits);
    materialize_cow(mx);
    materialize_cow(log_s);
    materialize_cow(target);
    materialize_cow(grad_rows);

    // The two backward passes run row-chunked through an fp32 scratch
    // (~CE_SCRATCH_BYTES): pass 1 (ce_shift) writes fl(x32 - max), pass 2
    // (ce_grad) consumes it — the two roundings of the aten lp chain must not
    // merge inside one kernel (the Metal compiler reassociates them, see the
    // shader header). The grad GEMMs afterwards are FULL (not chunked), so the
    // fp32 accumulation order over K matches the aten single-mm backward.
    int64_t chunk = std::min<int64_t>(N, std::max<int64_t>(1, (int64_t)(CE_SCRATCH_BYTES / 4) / std::max<int64_t>(V, 1)));
    at::Tensor scratch = at::empty({chunk, V}, h.options().dtype(at::kFloat));

    for (int64_t r0 = 0; r0 < N; r0 += chunk)
    {
        const int64_t rows = std::min(chunk, N - r0);
        at::Tensor logits_c = logits.narrow(0, r0, rows);
        at::Tensor shift_c = scratch.narrow(0, 0, rows);
        const uint32_t gx = (static_cast<uint32_t>(V) + CE_GRAD_WG - 1) / CE_GRAD_WG;

        {
            SpecializationBuilder spd{}; // WG — compile-time константа (config.py)
            PushConstantBuilder pcs{};
            pcs.push(get_tensor_address(logits_c));
            pcs.push(get_tensor_address(mx));
            pcs.push(get_output_address(shift_c));
            pcs.push(static_cast<uint32_t>(V));
            pcs.push(static_cast<uint32_t>(rows));
            pcs.push(static_cast<uint32_t>(r0));

            device->shader_manager->dispatchShader(
                ShaderKey{torchvulkan::Kernel::CE_SHIFT, logits.scalar_type()},
                device->default_subgroup_size,
                spd.build(),
                pcs.build(),
                /* read = */ {logits_c, mx},
                /* write = */ {shift_c},
                gx, static_cast<uint32_t>(rows), 1
            );
        }

        {
            SpecializationBuilder spd{};
            PushConstantBuilder pcs{};
            pcs.push(get_tensor_address(shift_c));
            pcs.push(get_tensor_address(log_s));
            pcs.push(get_tensor_address(target));
            pcs.push(get_tensor_address(grad_rows));
            pcs.push(get_output_address(logits_c));
            pcs.push(static_cast<uint32_t>(V));
            pcs.push(static_cast<uint32_t>(rows));
            pcs.push(static_cast<int32_t>(ignore_index));
            pcs.push(static_cast<uint32_t>(r0));

            device->shader_manager->dispatchShader(
                ShaderKey{torchvulkan::Kernel::CE_GRAD, logits.scalar_type()},
                device->default_subgroup_size,
                spd.build(),
                pcs.build(),
                /* read = */ {shift_c, log_s, target, grad_rows},
                /* write = */ {logits_c},
                gx, static_cast<uint32_t>(rows), 1
            );
        }
    }

    // the two grad GEMMs on the ordinary matmul path; grad_logits.t() rides
    // the transposed-A support of the GEMM kernel
    grad_h = at::matmul(logits, weight);       // [N, V] @ [V, H]
    grad_w = at::matmul(logits.t(), h2d);      // [V, N] @ [N, H]

    return {grad_h, grad_w};
}

} // namespace torchvulkan
