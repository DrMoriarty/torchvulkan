#include <torch/extension.h>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

// Fused train-path depthwise causal conv1d of the LFM2 short-conv layers
// (src/shaders/ops/dwconv_train.slang, experiments/lfm2_train_profile.md
// приоритет 4). One dispatch forward over the in_proj output [B, T, 3C]
// (gate/mix/x chunks read in place — no transpose copies, no chunk, no B*x /
// C* muls) and two backward dispatches + the host aten fold of the wgrad
// partials, instead of the ~6-10-dispatch eager chain per layer per phase.

namespace torchvulkan {

namespace {

constexpr uint32_t DWCONV_WG = 256; // must match the constants=(256,) in scripts/config.py
constexpr int64_t DWCONV_MAX_K = 4;

void check_dwconv_dtype(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.is_privateuseone(), "torchvulkan [ERROR]: ", name, " must be a vulkan tensor, got ", t.device(), ".");
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat,
        "torchvulkan [ERROR]: ", name, " must be fp16 or fp32, got ", t.scalar_type(), ".");
}

struct DwConvPlan
{
    at::Tensor proj, w, bias;
    int64_t batch, seq, chan, kern;
};

// Shared checks of fwd/bwd: proj (B, T, 3C) contiguous, w (C, k) k <= 4,
// bias (C)?; everything in one dtype.
DwConvPlan plan_dwconv(const at::Tensor& proj, const at::Tensor& w,
    const c10::optional<at::Tensor>& bias, const char* who)
{
    check_dwconv_dtype(proj, "proj");
    check_dwconv_dtype(w, "w");
    TORCH_CHECK(w.scalar_type() == proj.scalar_type(),
        "torchvulkan [ERROR]: dwconv weight dtype (", w.scalar_type(), ") must match proj (", proj.scalar_type(), ").");
    TORCH_CHECK(proj.dim() == 3 && proj.size(2) % 3 == 0,
        "torchvulkan [ERROR]: dwconv ", who, " proj must be (B, T, 3C), got ", proj.sizes(), ".");
    TORCH_CHECK(proj.is_contiguous(), "torchvulkan [ERROR]: dwconv ", who, " proj must be contiguous.");
    TORCH_CHECK(w.dim() == 2 && w.is_contiguous(),
        "torchvulkan [ERROR]: dwconv weight must be a contiguous (C, k) matrix, got ", w.sizes(), ".");
    const int64_t chan = proj.size(2) / 3;
    const int64_t kern = w.size(1);
    TORCH_CHECK(w.size(0) == chan, "torchvulkan [ERROR]: dwconv weight rows (", w.size(0),
        ") must match the channel count (", chan, ").");
    TORCH_CHECK(kern >= 1 && kern <= DWCONV_MAX_K,
        "torchvulkan [ERROR]: dwconv supports kernel sizes 1..", DWCONV_MAX_K, ", got ", kern, ".");
    at::Tensor bc;
    if (bias.has_value() && bias->defined())
    {
        check_dwconv_dtype(*bias, "bias");
        TORCH_CHECK(bias->scalar_type() == proj.scalar_type(),
            "torchvulkan [ERROR]: dwconv bias dtype (", bias->scalar_type(), ") must match proj (", proj.scalar_type(), ").");
        TORCH_CHECK(bias->dim() == 1 && bias->numel() == chan,
            "torchvulkan [ERROR]: dwconv bias must be (", chan, "), got ", bias->sizes(), ".");
        bc = bias->contiguous();
    }
    return DwConvPlan{proj, w, bc, proj.size(0), proj.size(1), chan, kern};
}

} // namespace

// Forward of the fused layer: out = C_chunk * act(depthwise_causal_conv(gate*x)).
// proj: (B, T, 3C) contiguous — the in_proj output, chunks [gate | mix | x];
// w: (C, k) contiguous; act: 0 none, 1 silu. Returns (out, conv_cache) — the
// cache keeps the rounded conv output (pre-activation) for the backward.
std::tuple<at::Tensor, at::Tensor> dwconv_train_fwd(const at::Tensor& proj, const at::Tensor& w,
    const c10::optional<at::Tensor>& bias, int64_t act)
{
    TORCH_CHECK(act == 0 || act == 1, "torchvulkan [ERROR]: dwconv activation must be 0 (none) or 1 (silu), got ", act, ".");
    DwConvPlan p = plan_dwconv(proj, w, bias, "fwd");
    const int64_t numel = p.batch * p.seq * p.chan;
    at::Tensor out = at::empty({p.batch, p.seq, p.chan}, proj.options());
    at::Tensor cache = at::empty({p.batch, p.seq, p.chan}, proj.options());
    if (numel == 0) return {out, cache};

    DeviceContext* device = get_device_context(p.proj);
    TORCH_CHECK((numel + DWCONV_WG - 1) / DWCONV_WG <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: dwconv fwd element count ", numel, " exceeds the workgroup count limit.");

    materialize_cow(p.proj);
    materialize_cow(p.w);
    if (p.bias.defined()) materialize_cow(p.bias);

    SpecializationBuilder spd{}; // WG = 256 задан compile-time константой в config.py
    SpecializationArgs specialization = spd.build();

    // порядок пушей = порядку полей DwConvFwdPushConstants (см. шейдер)
    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(p.proj));
    pcs.push(get_tensor_address(p.w));
    pcs.push(p.bias.defined() ? get_tensor_address(p.bias) : uint64_t(0));
    pcs.push(get_output_address(cache));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<uint32_t>(numel));
    pcs.push(static_cast<uint32_t>(p.seq));
    pcs.push(static_cast<uint32_t>(p.chan));
    pcs.push(static_cast<uint32_t>(p.kern));
    pcs.push(static_cast<uint32_t>(act));

    std::vector<at::Tensor> reads = {p.proj, p.w};
    if (p.bias.defined()) reads.push_back(p.bias);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::DWCONV_TRAIN_FWD, p.proj.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ reads,
        /* write = */ {cache, out},
        (static_cast<uint32_t>(numel) + DWCONV_WG - 1) / DWCONV_WG, 1, 1
    );

    return {out, cache};
}

// Backward of dwconv_train_fwd: (grad_proj, grad_w, grad_bias).
// grad_proj (B, T, 3C) — grads of the gate/mix/x chunks for the in_proj
// backward; grad_w (C, k) / grad_bias (C) — fp32 per-batch partials folded by
// one aten sum; grad_bias undefined unless with_bias (the forward had a bias).
std::tuple<at::Tensor, at::Tensor, at::Tensor> dwconv_train_bwd(const at::Tensor& grad_out,
    const at::Tensor& proj, const at::Tensor& w, const at::Tensor& cache, int64_t act, bool with_bias)
{
    TORCH_CHECK(act == 0 || act == 1, "torchvulkan [ERROR]: dwconv activation must be 0 (none) or 1 (silu), got ", act, ".");
    DwConvPlan p = plan_dwconv(proj, w, c10::nullopt, "bwd");
    check_dwconv_dtype(grad_out, "grad_out");
    check_dwconv_dtype(cache, "conv_cache");
    TORCH_CHECK(grad_out.scalar_type() == proj.scalar_type() && cache.scalar_type() == proj.scalar_type(),
        "torchvulkan [ERROR]: dwconv bwd dtype mismatch: grad_out (", grad_out.scalar_type(),
        "), conv_cache (", cache.scalar_type(), "), proj (", proj.scalar_type(), ").");
    TORCH_CHECK(grad_out.dim() == 3 && grad_out.size(0) == p.batch && grad_out.size(1) == p.seq
            && grad_out.size(2) == p.chan,
        "torchvulkan [ERROR]: dwconv bwd grad_out must be (B, T, C) = (", p.batch, ", ", p.seq, ", ", p.chan,
        "), got ", grad_out.sizes(), ".");
    TORCH_CHECK(cache.sizes() == grad_out.sizes(),
        "torchvulkan [ERROR]: dwconv bwd conv_cache ", cache.sizes(), " must match grad_out ", grad_out.sizes(), ".");
    const int64_t numel = p.batch * p.seq * p.chan;
    at::Tensor dproj = at::empty_like(proj);
    if (numel == 0)
        return {dproj, at::Tensor(), at::Tensor()};

    DeviceContext* device = get_device_context(p.proj);
    at::Tensor gc = grad_out.contiguous();
    at::Tensor cc = cache.contiguous();

    // --- kernel 1: grads of the three proj chunks (one thread per element) ---
    {
        TORCH_CHECK((numel + DWCONV_WG - 1) / DWCONV_WG <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[0],
            "torchvulkan [ERROR]: dwconv bwd element count ", numel, " exceeds the workgroup count limit.");
        materialize_cow(gc);
        materialize_cow(p.proj);
        materialize_cow(p.w);
        materialize_cow(cc);

        SpecializationBuilder spd{};
        SpecializationArgs specialization = spd.build();

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(gc));
        pcs.push(get_tensor_address(p.proj));
        pcs.push(get_tensor_address(p.w));
        pcs.push(get_tensor_address(cc));
        pcs.push(get_output_address(dproj));
        pcs.push(static_cast<uint32_t>(numel));
        pcs.push(static_cast<uint32_t>(p.seq));
        pcs.push(static_cast<uint32_t>(p.chan));
        pcs.push(static_cast<uint32_t>(p.kern));
        pcs.push(static_cast<uint32_t>(act));

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::DWCONV_TRAIN_BWD, p.proj.scalar_type()},
            device->default_subgroup_size,
            specialization,
            pcs.build(),
            /* read = */ {gc, p.proj, p.w, cc},
            /* write = */ {dproj},
            (static_cast<uint32_t>(numel) + DWCONV_WG - 1) / DWCONV_WG, 1, 1
        );
    }

    // --- kernel 2: per-(b, c) fp32 partials of grad_w / grad_bias, then one
    // aten sum over the batch (fold of (B, C, k+1) -> (C, k+1)) --------------
    at::Tensor grad_w, grad_bias;
    {
        const int64_t rows = p.batch * p.chan;
        at::Tensor partial = at::empty({p.batch, p.chan, p.kern + 1}, p.proj.options().dtype(at::kFloat));

        SpecializationBuilder spd{};
        SpecializationArgs specialization = spd.build();

        PushConstantBuilder pcs{};
        pcs.push(get_tensor_address(gc));
        pcs.push(get_tensor_address(p.proj));
        pcs.push(get_tensor_address(cc));
        pcs.push(get_output_address(partial));
        pcs.push(static_cast<uint32_t>(rows));
        pcs.push(static_cast<uint32_t>(p.seq));
        pcs.push(static_cast<uint32_t>(p.chan));
        pcs.push(static_cast<uint32_t>(p.kern));
        pcs.push(static_cast<uint32_t>(act));

        device->shader_manager->dispatchShader(
            ShaderKey{torchvulkan::Kernel::DWCONV_WGRAD, p.proj.scalar_type()},
            device->default_subgroup_size,
            specialization,
            pcs.build(),
            /* read = */ {gc, p.proj, cc},
            /* write = */ {partial},
            (static_cast<uint32_t>(rows) + DWCONV_WG - 1) / DWCONV_WG, 1, 1
        );

        at::Tensor folded = partial.sum(0); // (C, k+1) fp32
        grad_w = folded.narrow(1, 0, p.kern).contiguous().to(p.proj.scalar_type());
        if (with_bias)
            grad_bias = folded.narrow(1, p.kern, 1).squeeze(1).contiguous().to(p.proj.scalar_type());
    }

    return {dproj, grad_w, grad_bias};
}

} // namespace torchvulkan
