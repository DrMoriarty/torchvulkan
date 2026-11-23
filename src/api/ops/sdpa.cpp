#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <ATen/ops/_scaled_dot_product_attention_math_native.h>
#include <cmath>
#include <cstdlib>
#include <optional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"

// Fused SDPA for the training path (A5 in experiments/training_speedup.md).
// The math decomposition materializes B*H*T*T fp32 scores and runs softmax and
// the whole backward as dozens of pointwise/reduction kernels through the
// dispatcher; here the forward is one kernel (sdpa_fwd_main: QK^T + softmax +
// P*V, one workgroup per R-row tile, scores streamed, P materialized) and the
// backward is two private kernels (delta rowdot, dS = P*(dP - delta)) plus
// four regular matmul dispatches (dP, dQ, dK, dV) on grouped views that keep
// GQA batched at B*Hkv without expanding K/V. A fully fused backward (FA2-
// style, per-thread rows/columns, no dP materialization) was implemented and
// benchmarked during A5 and lost to these four GEMM dispatches on the 5300M
// (~1.5 vs ~1.1 ms of GPU time at B8/H8/T128/hd64 — see §9), so the GEMM
// composition stayed.
//
// Routing: aten::scaled_dot_product_attention is overridden under
// AutogradPrivateUse1 (the op itself is CompositeImplicitAutograd, so a backend
// key would never be consulted); unsupported cases fall back to the native
// composite, whose math path behaves exactly as before. Covered: causal and
// non-causal, additive fp32/fp16 and bool masks broadcastable to (L, S), any
// L/S (the former 4096 cap was historical — the two-pass kernel keeps only a
// 128-float scratch; the 32-bit guard on the materialized probs is what
// actually bounds the shapes).

namespace torchvulkan {

namespace {

constexpr uint32_t SDPA_WG = 128;
// q-строк на workgroup в fwd/bwd (row-tiling, generic R в sdpa.slang;
// константа в config.py должна совпадать)
constexpr int64_t SDPA_ROWS_R = 4;
constexpr int64_t SDPA_HD_CAP = 128;

bool sdpa_env_disabled()
{
    static const bool disabled = std::getenv("TORCHVULKAN_NO_FUSED_SDPA") != nullptr;
    return disabled;
}

uint32_t sdpa_grid_rows(int64_t rows, const DeviceContext* device)
{
    const uint32_t limit = device->properties.limits.maxComputeWorkGroupCount[0];
    return static_cast<uint32_t>(std::min<int64_t>(rows, limit));
}

// strides (elements) for reading `mask` as a 2D broadcast over (L, S), or
// nullopt when the shape can't be served by the kernels: only masks whose
// last dim indexes S, the one before it L and every leading dim is 1 (i.e.
// anything at::broadcastable to (L, S) without folding the batch/head dims).
// Size-1 dims index with stride 0; broadcast views (stride 0 on a sized dim)
// and arbitrary inner strides pass through unchanged.
static std::optional<std::pair<int64_t, int64_t>> sdpa_mask_strides(
    const at::Tensor& mask, int64_t l, int64_t s)
{
    const auto sizes = mask.sizes();
    const auto strides = mask.strides();
    const int64_t nd = sizes.size();
    if (nd < 1 || sizes[nd - 1] == 0) return std::nullopt;
    if (sizes[nd - 1] != 1 && sizes[nd - 1] != s) return std::nullopt;
    const int64_t s1 = sizes[nd - 1] == 1 ? 0 : strides[nd - 1];
    int64_t s0 = 0;
    if (nd >= 2)
    {
        if (sizes[nd - 2] != 1 && sizes[nd - 2] != l) return std::nullopt;
        s0 = sizes[nd - 2] == 1 ? 0 : strides[nd - 2];
        for (int64_t d = 0; d < nd - 2; d++)
            if (sizes[d] != 1) return std::nullopt;
    }
    return std::make_pair(s0, s1);
}

// the kernels read an additive fp32 mask; bool masks become 0 / -inf exactly
// like the math path's convert_boolean_attn_mask, other dtypes are cast
static at::Tensor sdpa_prepare_mask(const at::Tensor& attn_mask)
{
    if (attn_mask.scalar_type() == at::kBool)
    {
        return at::masked_fill(
            at::zeros(attn_mask.sizes(), attn_mask.options().dtype(at::kFloat)),
            attn_mask.logical_not(), -std::numeric_limits<float>::infinity());
    }
    return attn_mask.scalar_type() == at::kFloat ? attn_mask : attn_mask.to(at::kFloat);
}

} // namespace

// whether the fused path can run this call; everything else keeps the
// native composite (math) semantics
bool sdpa_route_ok(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& attn_mask, double dropout_p, bool is_causal,
    bool enable_gqa)
{
    if (sdpa_env_disabled()) return false;
    if (dropout_p != 0.0) return false;
    if (!q.is_privateuseone() || !k.is_privateuseone() || !v.is_privateuseone()) return false;
    if (q.scalar_type() != at::kHalf && q.scalar_type() != at::kFloat) return false;
    if (k.scalar_type() != q.scalar_type() || v.scalar_type() != q.scalar_type()) return false;
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) return false;
    const int64_t hd = q.size(3);
    if (k.size(3) != hd || v.size(3) != hd) return false; // fused path needs E == Ev
    if (hd < 1 || hd > SDPA_HD_CAP) return false;
    const int64_t hq = q.size(1), hkv = k.size(1);
    if (q.size(0) != k.size(0) || q.size(0) != v.size(0)) return false;
    if (k.size(2) != v.size(2)) return false;
    if (hkv < 1 || hq < 1 || hq % hkv != 0) return false;
    if (!enable_gqa && hq != hkv) return false;
    const int64_t l = q.size(2), s = k.size(2);
    if (l < 1 || s < 1) return false;
    // torch's own contract (validate + math kernel TORCH_CHECK): explicit mask
    // and is_causal are mutually exclusive — route out so the native composite
    // raises the same error
    if (is_causal && attn_mask.has_value()) return false;
    if (attn_mask.has_value())
    {
        const at::Tensor& mask = *attn_mask;
        if (!mask.is_privateuseone() || mask.device() != q.device()) return false;
        // bool -> additive on the host; the kernel reads fp32 only. Same dtype
        // set the composite's validate_sdpa_input accepts (bool / float /
        // query dtype)
        if (mask.scalar_type() != at::kBool && mask.scalar_type() != at::kFloat
            && mask.scalar_type() != q.scalar_type()) return false;
        // math path would differentiate through an additive mask; rare — keep
        // it on the composite
        if (mask.requires_grad() || mask.numel() == 0) return false;
        if (!fits_32bit_indexing({mask})) return false;
        if (!sdpa_mask_strides(mask, l, s).has_value()) return false;
    }
    // контiguity не требуется: sdpa_causal_fwd/bwd копируют в contiguous
    // (типичный кейс llm — v после permute), копия бесплатна для уже
    // непрерывных входов
    if (q.numel() == 0 || k.numel() == 0 || v.numel() == 0) return false;
    DeviceContext* device = get_device_context(q);
    if (q.scalar_type() == at::kHalf && !device->support_float16) return false;
    // everything the kernels index must fit 31-bit element counts (probs is the
    // largest buffer)
    const int64_t b = q.size(0);
    if (!fits_32bit_indexing({q, k, v})
        || b * hq * l * s > int64_t(std::numeric_limits<int32_t>::max())) return false;
    return true;
}

// out = softmax(scale * q k^T + mask) v with the optional causal tril; also
// materializes the probability row P (causal and masked entries zero) for the
// backward
std::tuple<at::Tensor, at::Tensor> sdpa_causal_fwd(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const c10::optional<at::Tensor>& attn_mask, bool is_causal,
    double scale)
{
    const int64_t b = q.size(0), hq = q.size(1), l = q.size(2), hd = q.size(3);
    const int64_t hkv = k.size(1), s = k.size(2);
    const int64_t qpg = hq / hkv;

    at::Tensor out = at::empty(q.sizes(), q.options());
    at::Tensor probs = at::empty({b, hkv, qpg, l, s}, q.options());
    const int64_t rows = b * hq * l;
    if (rows == 0 || s == 0) return {out, probs};

    at::Tensor qc = q.contiguous();
    at::Tensor kc = k.contiguous();
    at::Tensor vc = v.contiguous();
    at::Tensor mask;
    uint64_t mask_addr = 0;
    uint32_t mask_s0 = 0, mask_s1 = 0, has_mask = 0;
    if (attn_mask.has_value() && attn_mask->defined())
    {
        mask = sdpa_prepare_mask(*attn_mask);
        auto strides = sdpa_mask_strides(mask, l, s);
        TORCH_CHECK(strides.has_value(),
            "torchvulkan [ERROR]: sdpa_fwd attn_mask of shape ", mask.sizes(),
            " is not broadcastable to (L, S) = (", l, ", ", s, ").");
        mask_s0 = static_cast<uint32_t>(strides->first);
        mask_s1 = static_cast<uint32_t>(strides->second);
        has_mask = 1;
    }

    DeviceContext* device = get_device_context(qc);
    // row-tiling: один WG обрабатывает SDPA_ROWS_R подряд идущих строк
    // одной головы (tph тайлов на голову), сетка по тайлам
    const int64_t tph = (l + SDPA_ROWS_R - 1) / SDPA_ROWS_R;
    const int64_t tiles = b * hq * tph;
    const uint32_t rows_x = sdpa_grid_rows(tiles, device);
    const uint32_t grid_y = static_cast<uint32_t>((tiles + rows_x - 1) / rows_x);
    TORCH_CHECK(int64_t(rows_x) * grid_y >= tiles && grid_y <= device->properties.limits.maxComputeWorkGroupCount[1],
        "torchvulkan [ERROR]: sdpa_fwd tile count ", tiles, " exceeds the device workgroup limit.");

    materialize_cow(qc);
    materialize_cow(kc);
    materialize_cow(vc);
    if (has_mask) materialize_cow(mask);

    SpecializationBuilder spd{}; // WG задан compile-time константой в config.py
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(qc));
    pcs.push(get_tensor_address(kc));
    pcs.push(get_tensor_address(vc));
    pcs.push(get_output_address(out));
    pcs.push(get_output_address(probs));
    pcs.push(static_cast<uint32_t>(rows));
    pcs.push(static_cast<uint32_t>(hq));
    pcs.push(static_cast<uint32_t>(hkv));
    pcs.push(static_cast<uint32_t>(qpg));
    pcs.push(static_cast<uint32_t>(l));
    pcs.push(static_cast<uint32_t>(s));
    pcs.push(static_cast<uint32_t>(hd));
    pcs.push(rows_x);
    pcs.push(static_cast<uint32_t>(tph));
    pcs.push(static_cast<float>(scale));
    pcs.push(has_mask ? get_tensor_address(mask) : uint64_t(0));
    pcs.push(mask_s0);
    pcs.push(mask_s1);
    pcs.push(has_mask);
    pcs.push(is_causal ? uint32_t(1) : uint32_t(0));

    std::vector<at::Tensor> read = {qc, kc, vc};
    if (has_mask) read.push_back(mask);
    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SDPA_FWD, qc.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        read,
        /* write = */ {out, probs},
        rows_x, grid_y, 1
    );

    return {out, probs};
}

// delta[r] = sum_d a[r, d] * b[r, d] in fp32; a/b: (rows, hd) contiguous
at::Tensor sdpa_rowdot(const at::Tensor& a, const at::Tensor& b)
{
    TORCH_CHECK(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type(),
        "torchvulkan [ERROR]: sdpa_rowdot operands must match, got ", a.sizes(),
        " / ", b.sizes(), ".");
    const int64_t rows = a.size(0), hd = a.size(1);
    at::Tensor out = at::empty({rows}, a.options().dtype(at::kFloat));
    if (rows == 0 || hd == 0) return out;

    at::Tensor ac = a.contiguous();
    at::Tensor bc = b.contiguous();

    DeviceContext* device = get_device_context(ac);
    const int64_t blocks = (rows + SDPA_WG - 1) / SDPA_WG;
    const uint32_t blocks_x = sdpa_grid_rows(blocks, device);
    const uint32_t grid_y = static_cast<uint32_t>((blocks + blocks_x - 1) / blocks_x);

    materialize_cow(ac);
    materialize_cow(bc);

    SpecializationBuilder spd{};
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(ac));
    pcs.push(get_tensor_address(bc));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<uint32_t>(rows));
    pcs.push(static_cast<uint32_t>(hd));
    pcs.push(blocks_x);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SDPA_ROWDOT, ac.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {ac, bc},
        /* write = */ {out},
        blocks_x, grid_y, 1
    );

    return out;
}

// dS = scale * P * (dP - delta[row]); causal entries (j > i) zeroed when the
// forward ran causal (masked entries are already exactly 0 in P — exp(-inf) —
// so the mask needs no handling here); probs/dP/dS share the (..., L, S)
// layout, delta has one fp32 value per row. The scale is the scores
// multiplier folded in the forward: the chain back to q/k is ds_ij/dq_i =
// scale * k_j, so one multiply here fixes both dQ and dK.
at::Tensor sdpa_ds(const at::Tensor& probs, const at::Tensor& dP, const at::Tensor& delta,
    int64_t l, bool is_causal, double scale)
{
    TORCH_CHECK(probs.sizes() == dP.sizes(),
        "torchvulkan [ERROR]: sdpa_ds probs (", probs.sizes(), ") and dP (", dP.sizes(), ") must match.");
    TORCH_CHECK(probs.is_contiguous() && dP.is_contiguous(),
        "torchvulkan [ERROR]: sdpa_ds expects contiguous probs/dP.");
    TORCH_CHECK(dP.scalar_type() == probs.scalar_type(),
        "torchvulkan [ERROR]: sdpa_ds dP dtype must match probs, got ", dP.scalar_type(),
        " vs ", probs.scalar_type(), ".");
    const int64_t numel = probs.numel();
    const int64_t s = probs.size(-1);
    TORCH_CHECK(delta.numel() * s == numel,
        "torchvulkan [ERROR]: sdpa_ds delta (", delta.numel(), ") must have one value per probs row.");
    at::Tensor out = at::empty_like(probs);
    if (numel == 0) return out;

    at::Tensor pc = probs.contiguous();
    at::Tensor dc = dP.contiguous();
    at::Tensor delc = delta.contiguous();

    DeviceContext* device = get_device_context(pc);
    const uint32_t limit0 = device->properties.limits.maxComputeWorkGroupCount[0];
    const int64_t total_blocks = (numel + SDPA_WG - 1) / SDPA_WG;
    const uint32_t blocks_x = static_cast<uint32_t>(std::min<int64_t>(total_blocks, limit0));
    const uint32_t grid_y = static_cast<uint32_t>((total_blocks + blocks_x - 1) / blocks_x);
    TORCH_CHECK(grid_y <= device->properties.limits.maxComputeWorkGroupCount[1],
        "torchvulkan [ERROR]: sdpa_ds element count ", numel, " exceeds the device workgroup limit.");

    materialize_cow(pc);
    materialize_cow(dc);
    materialize_cow(delc);

    SpecializationBuilder spd{};
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(pc));
    pcs.push(get_tensor_address(dc));
    pcs.push(get_tensor_address(delc));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<uint32_t>(numel));
    pcs.push(static_cast<uint32_t>(s));
    pcs.push(static_cast<uint32_t>(l));
    pcs.push(blocks_x);
    pcs.push(static_cast<float>(scale));
    pcs.push(is_causal ? uint32_t(1) : uint32_t(0));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::SDPA_DS, pc.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {pc, dc, delc},
        /* write = */ {out},
        blocks_x, grid_y, 1
    );

    return out;
}

// the fused backward: (dQ, dK, dV) from grad_out and the forward's saved state
std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_causal_backward(const at::Tensor& grad_out,
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& out,
    const at::Tensor& probs, bool is_causal, double scale)
{
    const int64_t b = q.size(0), hq = q.size(1), l = q.size(2), hd = q.size(3);
    const int64_t hkv = k.size(1), s = k.size(2);
    const int64_t qpg = hq / hkv;

    at::Tensor dO = grad_out.contiguous();
    // входы могли прийти неконтигуальными (route их пропускает, fwd копирует)
    at::Tensor qc = q.contiguous();
    at::Tensor kc = k.contiguous();
    at::Tensor vc = v.contiguous();
    // grouped views: the q/head dim folds into the row dim, so every GEMM is
    // batched (B, Hkv) and K/V need no per-head expansion
    at::Tensor dO_g = dO.view({b, hkv, qpg * l, hd});
    at::Tensor q_g = qc.view({b, hkv, qpg * l, hd});
    at::Tensor p_g = probs.contiguous().view({b, hkv, qpg * l, s});

    // delta_i = dO_i . out_i (= sum_j P_ij * dP_ij, the softmax backward term)
    at::Tensor delta = sdpa_rowdot(dO.view({b * hq * l, hd}), out.view({b * hq * l, hd}));

    // dP stays in the activations dtype: the fp16 GEMM accumulates in fp32 and
    // rounds only its output, so no fp32 materialization of dO/V and no extra
    // cast dispatches; the dS product itself is fp32 inside sdpa_ds
    at::Tensor dP = at::matmul(dO_g, vc.transpose(-2, -1));
    at::Tensor dS = sdpa_ds(p_g, dP, delta, l, is_causal, scale); // (b, hkv, qpg*l, s)

    at::Tensor dQ = at::matmul(dS, kc);                         // (b, hkv, qpg*l, hd)
    at::Tensor dK = at::matmul(dS.transpose(-2, -1), q_g);      // (b, hkv, s, hd)
    at::Tensor dV = at::matmul(p_g.transpose(-2, -1), dO_g);    // (b, hkv, s, hd)

    return {dQ.view(q.sizes()), dK, dV};
}

namespace {

// the attn_mask can't travel through Function::apply: an undefined tensor
// input trips InputMetadata's device() (custom_function.h builds input_info_
// for every tensor arg, defined or not), so the entries park the raw mask in
// a thread-local slot and the autograd forward — which runs synchronously on
// the same thread inside apply — picks it up. The mask itself needs no
// gradient (route_ok refuses requires_grad masks) and no saved versioning:
// the backward consumes only the materialized P, which already carries the
// masking as exact zeros.
thread_local c10::optional<at::Tensor> tls_sdpa_mask;
struct TlsSdpaMaskGuard
{
    c10::optional<at::Tensor> saved;
    explicit TlsSdpaMaskGuard(c10::optional<at::Tensor> mask)
        : saved(std::move(tls_sdpa_mask))
    {
        tls_sdpa_mask = std::move(mask);
    }
    ~TlsSdpaMaskGuard() { tls_sdpa_mask = std::move(saved); }
};

// one autograd node for both entry points; F.scaled_dot_product_attention
// exposes only the output, the math override also surfaces probs (no gradient
// through it — same as calling into a fused backend kernel). The forward
// returns a variable_list because Function::apply's to_optional supports only
// single tensors and lists, not tuples.
struct SdpaAutogradFunction : public torch::autograd::Function<SdpaAutogradFunction>
{
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx,
        const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool is_causal,
        double scale)
    {
        auto fwd = sdpa_causal_fwd(q, k, v, tls_sdpa_mask, is_causal, scale);
        ctx->save_for_backward({q, k, v, std::get<0>(fwd), std::get<1>(fwd)});
        ctx->saved_data["causal"] = is_causal;
        ctx->saved_data["scale"] = scale;
        return {std::get<0>(fwd), std::get<1>(fwd)};
    }

    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_out)
    {
        auto saved = ctx->get_saved_variables();
        const bool is_causal = ctx->saved_data["causal"].toBool();
        const double scale = ctx->saved_data["scale"].toDouble();
        auto grads = sdpa_causal_backward(grad_out[0], saved[0], saved[1], saved[2], saved[3],
            saved[4], is_causal, scale);
        // один градиент на каждый вход forward: q, k, v, is_causal, scale
        return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads), at::Tensor(),
            at::Tensor()};
    }
};

} // namespace

// the AutogradPrivateUse1 kernel of aten::scaled_dot_product_attention
at::Tensor scaled_dot_product_attention_vulkan(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const c10::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, c10::optional<double> scale, bool enable_gqa)
{
    if (!sdpa_route_ok(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa))
    {
        return at::native::scaled_dot_product_attention(q, k, v, attn_mask, dropout_p,
            is_causal, scale, enable_gqa);
    }
    const double effective = scale.has_value()
        ? *scale
        : 1.0 / std::sqrt(static_cast<double>(q.size(3)));
    TlsSdpaMaskGuard guard(attn_mask);
    return SdpaAutogradFunction::apply(q, k, v, is_causal, effective)[0];
}

namespace {

// the math override sees the composite's pre-processed args: bool masks are
// already converted to float, but K/V are NOT expanded (the math kernel does
// the GQA expansion itself), so the native GQA path is kept
bool sdpa_math_route_ok(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& attn_mask, double dropout_p,
    const c10::optional<at::Tensor>& dropout_mask, bool is_causal, bool enable_gqa)
{
    if (dropout_mask.has_value() || dropout_p != 0.0) return false;
    if (!q.is_privateuseone() || !k.is_privateuseone() || !v.is_privateuseone()) return false;
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) return false;
    const int64_t hq = q.size(1), hkv = k.size(1);
    if (hkv < 1 || hq < 1 || hq % hkv != 0) return false;
    if (!enable_gqa && hq != hkv) return false;
    return sdpa_route_ok(q, k, v, attn_mask, dropout_p, is_causal, enable_gqa);
}

} // namespace

// зеркальная композиция aten::native::_scaled_dot_product_attention_math из
// обычных aten-операторов. Нужна потому, что после PrivateUse1-перекрытия
// native-вызов стал для автограда opaque-оператором (autograd-not-implemented
// fallback молча обнуляет градиенты); здесь каждый оператор трассируется как
// раньше. Семантика 1:1 с math-кернелом torch 2.15 (upcast fp16→fp32 при
// выключенном allowFP16BF16ReductionMathSDP, scale в оба сомножителя, маска
// 0/-inf, _safe_softmax, GQA через repeat_interleave).
std::tuple<at::Tensor, at::Tensor> sdpa_math_fallback(const at::Tensor& q_, const at::Tensor& k_,
    const at::Tensor& v_, const c10::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, const c10::optional<at::Tensor>& dropout_mask, c10::optional<double> scale,
    bool enable_gqa)
{
    auto& ctx = at::globalContext();
    const auto origin_dtype = q_.scalar_type();
    const bool upcast = !ctx.allowFP16BF16ReductionMathSDP() && origin_dtype == at::kHalf;
    const at::Tensor q = upcast ? q_.to(at::kFloat) : q_;
    const at::Tensor k = upcast ? k_.to(at::kFloat) : k_;
    const at::Tensor v = upcast ? v_.to(at::kFloat) : v_;

    const bool is_negative_scaling = scale.has_value() && *scale < 0.0;
    const double effective = std::abs(scale.value_or(1.0 / std::sqrt(static_cast<double>(q.size(-1)))));
    const double scaling_factor = std::sqrt(effective);
    at::Tensor query = q * (is_negative_scaling ? -scaling_factor : scaling_factor);

    at::Tensor mask = attn_mask.has_value() ? *attn_mask : at::Tensor();
    if (is_causal)
    {
        auto tri = at::ones({q.size(-2), k.size(-2)}, q.options().dtype(at::kBool)).tril();
        // convert_boolean_attn_mask: 0 где разрешено, -inf где запрещено
        mask = at::masked_fill(at::zeros_like(tri, q.options()), tri.logical_not(),
            -std::numeric_limits<float>::infinity());
    }

    const int64_t qpg = enable_gqa && q.size(1) % k.size(1) == 0 ? q.size(1) / k.size(1) : 1;
    const at::Tensor key_expanded = qpg > 1 ? k.repeat_interleave(qpg, 1) : k;
    const at::Tensor value_expanded = qpg > 1 ? v.repeat_interleave(qpg, 1) : v;

    at::Tensor attn = at::matmul(query, key_expanded.transpose(-2, -1) * scaling_factor);
    if (mask.defined()) attn = attn + mask;
    attn = at::_safe_softmax(attn, -1);
    if (dropout_p > 0.0)
    {
        if (dropout_mask.has_value())
        {
            attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
            attn = at::matmul(attn, value_expanded * (1.0 / (1.0 - dropout_p)));
        }
        else
        {
            attn = at::dropout(attn, dropout_p, true);
            attn = at::matmul(attn, value_expanded);
        }
    }
    else
    {
        attn = at::matmul(attn, value_expanded);
    }
    return {attn.to(origin_dtype), attn.to(origin_dtype)};
}

// the PrivateUse1 kernel of aten::_scaled_dot_product_attention_math: this is
// where the composite lands under no_grad (the AutogradPrivateUse1 key of the
// outer op is skipped when GradMode is off), so the fused forward covers the
// inference path too
std::tuple<at::Tensor, at::Tensor> scaled_dot_product_attention_math_vulkan(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& attn_mask, double dropout_p, bool is_causal,
    const c10::optional<at::Tensor>& dropout_mask, c10::optional<double> scale, bool enable_gqa)
{
    if (!sdpa_math_route_ok(q, k, v, attn_mask, dropout_p, dropout_mask, is_causal, enable_gqa))
    {
        return sdpa_math_fallback(q, k, v, attn_mask, dropout_p, is_causal, dropout_mask, scale,
            enable_gqa);
    }
    const double effective = scale.has_value()
        ? *scale
        : 1.0 / std::sqrt(static_cast<double>(q.size(3)));
    const bool needs_grad = at::GradMode::is_enabled()
        && (q.requires_grad() || k.requires_grad() || v.requires_grad());
    if (!needs_grad)
    {
        auto fwd = sdpa_causal_fwd(q, k, v, attn_mask, is_causal, effective);
        // контракт math-кернела: ОБА выхода — attn (второй — та же матрица до
        // каста dtype; без upcast это тот же тензор), не вероятности
        return {std::get<0>(fwd), std::get<0>(fwd)};
    }
    auto outs = [&]
    {
        TlsSdpaMaskGuard guard(attn_mask);
        return SdpaAutogradFunction::apply(q, k, v, is_causal, effective);
    }();
    return {outs[0], outs[0]};
}

} // namespace torchvulkan
