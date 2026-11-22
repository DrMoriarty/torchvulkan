#include "fusion.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

#include <cstdlib>
#include <atomic>
#include <tuple>
#include <utility>

// Eager one-op lookback fusion: see fusion.h for the design. The registry is a
// single thread_local slot — the producer/consumer pair must dispatch back to
// back on one thread (autograd runs forward ops eagerly in dispatch order), so
// one pending producer is enough, and any other op in between invalidates the
// slot via the shader manager's span check.

namespace {

// observability: how many times each fusion actually fired (consumers can
// silently fall back when a flush slipped in between, so the counts are the
// only reliable trigger evidence)
std::atomic<uint64_t> add_relu_fusions{0};
std::atomic<uint64_t> addmm_gelu_fusions{0};

struct PendingProducer
{
    enum class Kind : uint32_t { NONE = 0, ADD = 1, ADDMM = 2 };

    Kind kind = Kind::NONE;
    torchvulkan::FusionOpSpan span{};
    at::Tensor out;          // the producer's output — the future pre-activation buffer
    at::Tensor a, b, bias;   // add: a, b (alpha) / addmm: mat1, mat2 (bias)
    at::Scalar alpha, beta;
};

PendingProducer& pending_producer()
{
    static thread_local PendingProducer pending;
    return pending;
}

bool fusion_enabled()
{
    static const bool enabled = std::getenv("TORCHVULKAN_NO_FUSION") == nullptr;
    return enabled;
}

bool fusable_elementwise(const at::Tensor& t)
{
    c10::ScalarType dtype = t.scalar_type();
    return (dtype == at::kHalf || dtype == at::kFloat) && t.is_contiguous();
}

bool aligned_16(const at::Tensor& t)
{
    return (get_tensor_address(t) & 15) == 0;
}

// the record-time validation for both producers: the fused kernels are
// contiguous elementwise paths, so anything else keeps the separate ops.
// Everything checked here is re-guaranteed at consume time — a consumer must
// never cancel the producer's ops and THEN discover it can't run the fused
// kernel (the cancelled work would be lost).
bool fusable_producer(const at::Tensor& out, const torchvulkan::FusionOpSpan& span)
{
    if (!fusion_enabled() || span.count == 0) return false;
    if (!out.defined() || out.nbytes() == 0) return false;
    if (!fusable_elementwise(out)) return false;
    DeviceContext* device = get_device_context(out);
    if (!is_dtype_supported(device, out.scalar_type())) return false;
    // grid indices fit 32 bits with room for the vector width
    if (out.numel() > (int64_t{1} << 31)) return false;
    if (!aligned_16(out)) return false;
    return true;
}

} // namespace

namespace torchvulkan {

void record_add_fusion(const at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
    const at::Scalar& alpha, const FusionOpSpan& span)
{
    if (!fusable_producer(out, span)) return;
    if (!fusable_elementwise(a) || !fusable_elementwise(b)) return;
    if (!a.sizes().equals(b.sizes())) return; // the fused kernel has no broadcast path
    if (!aligned_16(a) || !aligned_16(b)) return;

    PendingProducer& p = pending_producer();
    p.kind = PendingProducer::Kind::ADD;
    p.span = span;
    p.out = out;
    p.a = a;
    p.b = b;
    p.bias = at::Tensor();
    p.alpha = alpha;
    p.beta = at::Scalar(0.0);
}

void record_addmm_fusion(const at::Tensor& out, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Tensor& bias, const at::Scalar& alpha, const at::Scalar& beta, const FusionOpSpan& span)
{
    if (!fusable_producer(out, span)) return;
    // mat1/mat2 strides are free: the GEMM kernels take them through the push
    // constants (the linear pattern is mat2 = weight.t(), non-contiguous)

    PendingProducer& p = pending_producer();
    p.kind = PendingProducer::Kind::ADDMM;
    p.span = span;
    p.out = out;
    p.a = mat1;
    p.b = mat2;
    p.bias = bias;
    p.alpha = alpha;
    p.beta = beta;
}

at::Tensor consume_add_relu(const at::Tensor& input)
{
    PendingProducer& p = pending_producer();
    if (p.kind != PendingProducer::Kind::ADD) return at::Tensor();
    // exact tensor identity: only the add's own output fuses; the slot is
    // consumed either way (any following dispatch invalidates it)
    bool matches = input.defined() &&
        input.unsafeGetTensorImpl() == p.out.unsafeGetTensorImpl();
    PendingProducer taken;
    taken.kind = p.kind;
    taken.span = p.span;
    taken.out = p.out;
    taken.a = p.a;
    taken.b = p.b;
    taken.alpha = p.alpha;
    p = PendingProducer{};
    if (!matches) return at::Tensor();

    // all validity checks happened at record time; the only remaining failure
    // mode is a flush that slipped in between — then the separate kernels run
    DeviceContext* device = get_device_context(input);
    if (!device->shader_manager->cancelPendingOps(taken.span)) return at::Tensor();

    at::Tensor pre, out;
    std::tie(pre, out) = add_relu_fused(taken.a, taken.b, taken.alpha, taken.out);
    add_relu_fusions.fetch_add(1, std::memory_order_relaxed);
    return out;
}

at::Tensor consume_addmm_gelu(const at::Tensor& input)
{
    PendingProducer& p = pending_producer();
    if (p.kind != PendingProducer::Kind::ADDMM) return at::Tensor();
    bool matches = input.defined() &&
        input.unsafeGetTensorImpl() == p.out.unsafeGetTensorImpl();
    PendingProducer taken;
    taken.kind = p.kind;
    taken.span = p.span;
    taken.out = p.out;
    taken.a = p.a;
    taken.b = p.b;
    taken.bias = p.bias;
    taken.alpha = p.alpha;
    taken.beta = p.beta;
    p = PendingProducer{};
    if (!matches) return at::Tensor();

    DeviceContext* device = get_device_context(input);
    if (!device->shader_manager->cancelPendingOps(taken.span)) return at::Tensor();

    at::Tensor out = addmm_gelu_fused(taken.bias, taken.a, taken.b, taken.beta, taken.alpha, taken.out);
    addmm_gelu_fusions.fetch_add(1, std::memory_order_relaxed);
    return out;
}

std::tuple<uint64_t, uint64_t> fusion_stats()
{
    return {add_relu_fusions.load(std::memory_order_relaxed), addmm_gelu_fusions.load(std::memory_order_relaxed)};
}

// fused add(+alpha*b) + relu: pre = a + alpha*b (accumulated in fp32 for half),
// out = relu(pre) — relu is monotone and the fp32->fp16 rounding too, so
// relu(pre) and round(relu(sum)) are bit-identical; the activation is computed
// from the fp32 sum directly.
std::tuple<at::Tensor, at::Tensor> add_relu_fused(const at::Tensor& a_, const at::Tensor& b_,
    const at::Scalar& alpha, const at::Tensor& pre_dst)
{
    at::Tensor a = a_.contiguous();
    at::Tensor b = b_.contiguous();
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.sizes().equals(b.sizes()),
        "torchvulkan [ERROR]: add_relu expects equal-shape same-dtype tensors, got ",
        a.sizes(), " / ", b.sizes(), ".");
    c10::ScalarType dtype = a.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kFloat,
        "torchvulkan [ERROR]: add_relu is only implemented for fp16/fp32, got ", dtype, ".");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
        "torchvulkan [ERROR]: add_relu needs contiguous inputs.");

    at::Tensor pre = pre_dst.defined() ? pre_dst : at::empty_like(a);
    TORCH_CHECK(pre.is_contiguous() && pre.sizes().equals(a.sizes()) && pre.scalar_type() == dtype,
        "torchvulkan [ERROR]: add_relu pre-activation destination must be a contiguous ",
        dtype, " tensor of the same shape.");
    at::Tensor out = at::empty_like(pre);
    materialize_cow(pre);
    materialize_cow(out);
    int64_t numel = a.numel();
    if (numel == 0) return {pre, out};

    DeviceContext* device = get_device_context(a);
    if (!is_dtype_supported(device, dtype) || !fits_32bit_indexing({a, b, pre, out}) ||
        (get_tensor_address(a) & 15) || (get_tensor_address(b) & 15) ||
        (get_tensor_address(pre) & 15) || (get_tensor_address(out) & 15)) {
        TORCHVULKAN_FALLBACK("add_relu on ", dtype, " can't run on the Vulkan device.");
        at::Tensor sum = alpha.toDouble() == 1.0 ? at::add(a, b) : at::add(a, b, alpha);
        pre.copy_(sum);
        out.copy_(at::relu(sum));
        return {pre, out};
    }

    // vector width mirrors the binary kernel's wide fp16 variant (8 halfs per
    // uint4) and the fp32 4-float variant
    uint32_t vec = dtype == at::kHalf ? 8 : 4;
    uint32_t workgroup = 256;
    // the binary kernel rounds the alpha of an fp16 add through the dtype before
    // the arithmetic (wraps_scalar_with_dtype) — reproduce it exactly
    at::Scalar pushed_alpha = alpha;
    if (dtype == at::kHalf && alpha.isFloatingPoint())
        pushed_alpha = at::Scalar(static_cast<float>(c10::Half(static_cast<float>(alpha.toDouble()))));

    uint32_t has_alpha = pushed_alpha.toDouble() != 1.0 ? 1u : 0u;

    SpecializationBuilder spd{};
    spd.push(vec)
       .push(has_alpha)
       .push(workgroup);

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(a))
       .push(get_tensor_address(b))
       .push(get_output_address(pre))
       .push(get_output_address(out))
       .push(static_cast<uint64_t>(numel))
       .push(static_cast<float>(pushed_alpha.toDouble()));

    uint64_t threads = (static_cast<uint64_t>(numel) + vec - 1) / vec;
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::ADD_RELU, dtype},
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {a, b},
        /* write = */ {pre, out},
        get_group_count(threads, workgroup), 1, 1
    );

    return {pre, out};
}

// the private-op wrapper (torch.ops.torchvulkan.add_relu)
std::tuple<at::Tensor, at::Tensor> add_relu(const at::Tensor& a, const at::Tensor& b, double alpha,
    const c10::optional<at::Tensor>& pre)
{
    return add_relu_fused(a, b, at::Scalar(alpha), pre.value_or(at::Tensor()));
}

} // namespace torchvulkan
