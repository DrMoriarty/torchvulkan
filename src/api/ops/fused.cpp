#include <torch/extension.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"

// Fused decoder-hotpath kernels: RMSNorm (src/shaders/ops/rms_norm.slang) and
// rotate-half RoPE (src/shaders/ops/rope_half.slang). Each replaces a 9- /
// 4-kernel aten composition that costs ~35 us per shader in dispatch
// (Metal pipeline switch + DAG-wave barrier per kernel on MoltenVK), which
// dominates small-batch LLM decode. Private torch.ops, the analogue of the
// quantized matmul entry points.

namespace {

constexpr uint32_t RMS_WG = 128;   // multiple of both 32- and 64-lane subgroups
constexpr uint32_t ROPE_WG = 256;
constexpr uint32_t RMS_ROPE_WG = 128; // one workgroup per row, hd <= WG
constexpr uint32_t ATTN_L_CAP = 4096; // groupshared score scratch (see attn_decode.slang)

void check_fused_dtype(const at::Tensor& t, const char* name)
{
    TORCH_CHECK(t.is_privateuseone(), "torchvulkan [ERROR]: ", name, " must be a vulkan tensor, got ", t.device(), ".");
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat,
        "torchvulkan [ERROR]: ", name, " must be fp16 or fp32, got ", t.scalar_type(), ".");
}

} // namespace

namespace torchvulkan {

// out = x * rsqrt(mean(x^2, dim=-1) + eps) * w, statistics in fp32,
// output in the dtype of x. x: (..., hidden) contiguous; w: (hidden,) shared
// across rows or (rows, hidden) per-row weights.
at::Tensor rms_norm_fused(const at::Tensor& x, const at::Tensor& w, double eps)
{
    check_fused_dtype(x, "x");
    check_fused_dtype(w, "w");
    TORCH_CHECK(w.scalar_type() == x.scalar_type(), "torchvulkan [ERROR]: rms_norm weight dtype (",
        w.scalar_type(), ") must match the input dtype (", x.scalar_type(), ").");
    TORCH_CHECK(w.dim() == 1 || w.dim() == 2, "torchvulkan [ERROR]: rms_norm weight must be 1D or 2D, got ", w.sizes(), ".");
    const int64_t hidden = x.size(-1);
    TORCH_CHECK(hidden >= 1, "torchvulkan [ERROR]: rms_norm needs a non-empty last dimension.");
    TORCH_CHECK(w.size(-1) == hidden, "torchvulkan [ERROR]: rms_norm weight last dimension (", w.size(-1),
        ") must match the input last dimension (", hidden, ").");

    at::Tensor xc = x.contiguous();
    at::Tensor wc = w.contiguous();
    const int64_t rows = hidden > 0 ? xc.numel() / hidden : 0;
    // ROW_W: per-row weights (rows, hidden) — например, слитые q/k norm-веса
    const bool row_w = w.dim() == 2;
    if (row_w)
    {
        TORCH_CHECK(w.size(0) == rows && w.size(1) == hidden,
            "torchvulkan [ERROR]: 2D rms_norm weight must be (rows, hidden) = (", rows, ", ", hidden,
            "), got ", w.sizes(), ".");
    }
    at::Tensor out = at::empty_like(xc);
    if (rows == 0) return out;

    DeviceContext* device = get_device_context(xc);
    const uint32_t lanes = device->default_subgroup_size;
    const uint32_t wg = (RMS_WG % lanes == 0) ? RMS_WG : lanes;
    const uint32_t waves = wg / lanes; // rows per workgroup
    TORCH_CHECK(waves >= 1 && (rows + waves - 1) / waves <= device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: rms_norm row count ", rows, " exceeds the device workgroup limit.");

    materialize_cow(xc);
    materialize_cow(wc);

    SpecializationBuilder spd{};
    spd.push(wg);
    spd.push(row_w ? 1u : 0u);
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(xc));
    pcs.push(get_tensor_address(wc));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<uint32_t>(rows));
    pcs.push(static_cast<uint32_t>(hidden));
    pcs.push(static_cast<float>(eps));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::RMS_NORM, xc.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {xc, wc},
        /* write = */ {out},
        (static_cast<uint32_t>(rows) + waves - 1) / waves, 1, 1
    );

    return out;
}

// Fused residual-add + RMSNorm with two outputs (decode hotpath): one kernel
// instead of aten add + rms_norm (two dispatches per decoder layer).
//   sum = x1 + x2                                  (residual carried forward)
//   out = sum * rsqrt(mean(sum^2, dim=-1) + eps) * w
// x1/x2: (..., hidden) contiguous fp16/fp32; w: (hidden,). Statistics in fp32.
std::tuple<at::Tensor, at::Tensor> rms_norm_add(const at::Tensor& x1, const at::Tensor& x2,
    const at::Tensor& w, double eps)
{
    check_fused_dtype(x1, "x1");
    check_fused_dtype(x2, "x2");
    check_fused_dtype(w, "w");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type() && w.scalar_type() == x1.scalar_type(),
        "torchvulkan [ERROR]: rms_norm_add dtype mismatch: x1 (", x1.scalar_type(), "), x2 (",
        x2.scalar_type(), "), w (", w.scalar_type(), ").");
    TORCH_CHECK(x2.sizes() == x1.sizes(),
        "torchvulkan [ERROR]: rms_norm_add x2 (", x2.sizes(), ") must match x1 (", x1.sizes(), ").");
    TORCH_CHECK(w.dim() == 1, "torchvulkan [ERROR]: rms_norm_add weight must be 1D, got ", w.sizes(), ".");
    const int64_t hidden = x1.size(-1);
    TORCH_CHECK(hidden >= 1, "torchvulkan [ERROR]: rms_norm_add needs a non-empty last dimension.");
    TORCH_CHECK(w.numel() == hidden, "torchvulkan [ERROR]: rms_norm_add weight (", w.sizes(),
        ") must match the last dimension (", hidden, ").");

    at::Tensor x1c = x1.contiguous();
    at::Tensor x2c = x2.contiguous();
    at::Tensor wc = w.contiguous();
    const int64_t rows = hidden > 0 ? x1c.numel() / hidden : 0;
    at::Tensor sum = at::empty_like(x1c);
    at::Tensor out = at::empty_like(x1c);
    if (rows == 0) return {sum, out};

    DeviceContext* device = get_device_context(x1c);
    const uint32_t lanes = device->default_subgroup_size;
    const uint32_t wg = (RMS_WG % lanes == 0) ? RMS_WG : lanes;
    const uint32_t waves = wg / lanes; // rows per workgroup
    TORCH_CHECK(waves >= 1 && (rows + waves - 1) / waves <= device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: rms_norm_add row count ", rows, " exceeds the device workgroup limit.");

    materialize_cow(x1c);
    materialize_cow(x2c);
    materialize_cow(wc);

    SpecializationBuilder spd{};
    spd.push(wg);
    spd.push(0u); // ROW_W unused in the add-entry
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(x1c));
    pcs.push(get_tensor_address(x2c));
    pcs.push(get_tensor_address(wc));
    pcs.push(get_output_address(sum));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<uint32_t>(rows));
    pcs.push(static_cast<uint32_t>(hidden));
    pcs.push(static_cast<float>(eps));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::RMS_NORM_ADD, x1c.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {x1c, x2c, wc},
        /* write = */ {sum, out},
        (static_cast<uint32_t>(rows) + waves - 1) / waves, 1, 1
    );

    return {sum, out};
}

// Fused rotate-half RoPE with precomputed full-size tables: x (..., hd)
// contiguous, cos/sin (hd,) rows of the tables (C = cat(c, c), S = cat(-s, s));
// out[i, j] = x[i, j] * cos[j] + x[i, pair(j)] * sin[j] in fp32 math.
at::Tensor rope_rotate_half(const at::Tensor& x, const at::Tensor& cos_t, const at::Tensor& sin_t)
{
    check_fused_dtype(x, "x");
    check_fused_dtype(cos_t, "cos");
    check_fused_dtype(sin_t, "sin");
    const int64_t hd = x.size(-1);
    TORCH_CHECK(hd >= 2 && hd % 2 == 0, "torchvulkan [ERROR]: rope_rotate_half needs an even last dimension, got ", hd, ".");
    TORCH_CHECK(cos_t.dim() == 1 && sin_t.dim() == 1 && cos_t.numel() == hd && sin_t.numel() == hd,
        "torchvulkan [ERROR]: rope_rotate_half expects cos/sin rows of ", hd, " elements, got ",
        cos_t.sizes(), " and ", sin_t.sizes(), ".");
    TORCH_CHECK(cos_t.scalar_type() == x.scalar_type() && sin_t.scalar_type() == x.scalar_type(),
        "torchvulkan [ERROR]: rope_rotate_half cos/sin dtype must match the input dtype.");

    at::Tensor xc = x.contiguous();
    at::Tensor cc = cos_t.contiguous();
    at::Tensor sc = sin_t.contiguous();
    const int64_t total = xc.numel();
    at::Tensor out = at::empty_like(xc);
    if (total == 0) return out;

    DeviceContext* device = get_device_context(xc);
    materialize_cow(xc);
    materialize_cow(cc);
    materialize_cow(sc);

    SpecializationBuilder spd{};
    spd.push(ROPE_WG);
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(xc));
    pcs.push(get_tensor_address(cc));
    pcs.push(get_tensor_address(sc));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<uint32_t>(total));
    pcs.push(static_cast<uint32_t>(hd));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::ROPE_HALF, xc.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {xc, cc, sc},
        /* write = */ {out},
        (static_cast<uint32_t>(total) + ROPE_WG - 1) / ROPE_WG, 1, 1
    );

    return out;
}

// Fused per-row RMSNorm + rotate-half RoPE (decode QK stage): replaces the
// rms_norm -> rope_rotate_half pair with one kernel. x: (rows, hd) contiguous;
// w: (rows, hd) per-row QK-norm weights (attention scale pre-folded into the
// q rows upstream); cos/sin: (hd,) rows of the full tables. One workgroup of
// RMS_ROPE_WG threads per row, hd <= RMS_ROPE_WG.
at::Tensor rms_rope(const at::Tensor& x, const at::Tensor& cos_t, const at::Tensor& sin_t,
    const at::Tensor& w, double eps)
{
    check_fused_dtype(x, "x");
    check_fused_dtype(cos_t, "cos");
    check_fused_dtype(sin_t, "sin");
    check_fused_dtype(w, "w");
    const int64_t hd = x.size(-1);
    const int64_t rows = x.numel() / hd;
    TORCH_CHECK(hd >= 2 && hd % 2 == 0, "torchvulkan [ERROR]: rms_rope needs an even last dimension, got ", hd, ".");
    TORCH_CHECK(hd <= RMS_ROPE_WG, "torchvulkan [ERROR]: rms_rope supports hd <= ", RMS_ROPE_WG, ", got ", hd, ".");
    TORCH_CHECK(cos_t.numel() == hd && sin_t.numel() == hd,
        "torchvulkan [ERROR]: rms_rope expects cos/sin rows of ", hd, " elements, got ",
        cos_t.sizes(), " and ", sin_t.sizes(), ".");
    TORCH_CHECK(cos_t.scalar_type() == x.scalar_type() && sin_t.scalar_type() == x.scalar_type()
        && w.scalar_type() == x.scalar_type(),
        "torchvulkan [ERROR]: rms_rope dtype mismatch.");
    TORCH_CHECK(w.numel() == rows * hd,
        "torchvulkan [ERROR]: rms_rope weight must have ", rows * hd, " elements, got ", w.numel(), ".");

    at::Tensor xc = x.contiguous();
    at::Tensor cc = cos_t.contiguous();
    at::Tensor sc = sin_t.contiguous();
    at::Tensor wc = w.contiguous();
    at::Tensor out = at::empty_like(xc);
    if (rows == 0) return out;

    DeviceContext* device = get_device_context(xc);
    TORCH_CHECK(RMS_ROPE_WG <= device->properties.limits.maxComputeWorkGroupInvocations,
        "torchvulkan [ERROR]: rms_rope workgroup exceeds the device limit.");
    TORCH_CHECK(rows <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: rms_rope row count ", rows, " exceeds the workgroup count limit.");

    materialize_cow(xc);
    materialize_cow(cc);
    materialize_cow(sc);
    materialize_cow(wc);

    SpecializationBuilder spd{}; // WG задан compile-time в шейдере (numthreads(128))
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(xc));
    pcs.push(get_tensor_address(cc));
    pcs.push(get_tensor_address(sc));
    pcs.push(get_tensor_address(wc));
    pcs.push(get_output_address(out));
    pcs.push(static_cast<float>(eps));
    pcs.push(static_cast<uint32_t>(hd));

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::RMS_ROPE, xc.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ {xc, cc, sc, wc},
        /* write = */ {out},
        static_cast<uint32_t>(rows), 1, 1
    );

    return out;
}

// Fused decode attention (T == 1, GQA): out[n_kv, q_per_kv, hd] =
// softmax(q[h] · K[g, :L] / 1) @ V[g, :L] with the scale already folded into q
// upstream. q: (n_kv, q_per_kv, hd) contiguous; k_cache/v_cache: the full
// (1, n_kv, max_len, hd) fp16 caches, only :cur+1 positions are read.
at::Tensor attn_decode(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, int64_t cur,
    const c10::optional<at::Tensor>& k_cur, const c10::optional<at::Tensor>& v_cur,
    const c10::optional<at::Tensor>& qkn_w, const c10::optional<at::Tensor>& cos_t,
    const c10::optional<at::Tensor>& sin_t, const c10::optional<double>& eps)
{
    check_fused_dtype(q, "q");
    check_fused_dtype(k_cache, "k_cache");
    check_fused_dtype(v_cache, "v_cache");
    TORCH_CHECK(k_cache.scalar_type() == q.scalar_type() && v_cache.scalar_type() == q.scalar_type(),
        "torchvulkan [ERROR]: attn_decode cache dtype must match q.");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
        "torchvulkan [ERROR]: attn_decode caches must be (1, n_kv, max_len, hd), got ",
        k_cache.sizes(), " and ", v_cache.sizes(), ".");
    const int64_t n_kv = k_cache.size(1);
    const int64_t hd = k_cache.size(3);

    // два режима: (a) q — roped (n_kv, q_per_kv, hd); (b) rms-фуз — q — RAW
    // qk-блок (n_head + n_kv, hd), нормировка и rotate-half в кернеле
    const bool fuse = qkn_w.has_value();
    TORCH_CHECK(fuse == cos_t.has_value() && fuse == sin_t.has_value() && fuse == eps.has_value(),
        "torchvulkan [ERROR]: attn_decode rms fusion requires all of qkn_w/cos/sin/eps (or none).");
    int64_t rows, q_per_kv;
    if (fuse)
    {
        check_fused_dtype(*qkn_w, "qkn_w");
        check_fused_dtype(*cos_t, "cos");
        check_fused_dtype(*sin_t, "sin");
        TORCH_CHECK(q.dim() == 2, "torchvulkan [ERROR]: attn_decode rms fusion expects q as the raw "
            "qk block (n_head + n_kv, hd), got ", q.sizes(), ".");
        const int64_t qk_rows = q.size(0);
        TORCH_CHECK(qk_rows > n_kv && (qk_rows - n_kv) % n_kv == 0,
            "torchvulkan [ERROR]: attn_decode rms fusion qk block must have (q_per_kv + 1) * n_kv rows, got ", qk_rows, ".");
        q_per_kv = (qk_rows - n_kv) / n_kv;
        rows = n_kv * q_per_kv;
        TORCH_CHECK(q.size(1) == hd, "torchvulkan [ERROR]: attn_decode rms fusion qk block hd must be ", hd, ".");
        TORCH_CHECK(qkn_w->numel() == qk_rows * hd && qkn_w->scalar_type() == q.scalar_type(),
            "torchvulkan [ERROR]: attn_decode qkn_w must be (", qk_rows, ", ", hd, ") in the q dtype.");
        TORCH_CHECK(cos_t->numel() == hd && sin_t->numel() == hd
            && cos_t->scalar_type() == q.scalar_type() && sin_t->scalar_type() == q.scalar_type(),
            "torchvulkan [ERROR]: attn_decode cos/sin must be (", hd, ") rows in the q dtype.");
    }
    else
    {
        TORCH_CHECK(q.dim() == 3, "torchvulkan [ERROR]: attn_decode expects q (n_kv, q_per_kv, hd), got ", q.sizes(), ".");
        TORCH_CHECK(q.size(0) == n_kv && q.size(2) == hd,
            "torchvulkan [ERROR]: attn_decode q shape ", q.sizes(), " does not match the cache.");
        q_per_kv = q.size(1);
        rows = n_kv * q_per_kv;
    }
    TORCH_CHECK(cur >= 0 && cur < k_cache.size(2), "torchvulkan [ERROR]: attn_decode cur ", cur,
        " out of the cache range [0, ", k_cache.size(2), ").");
    const int64_t l_len = cur + 1;
    TORCH_CHECK(l_len <= ATTN_L_CAP, "torchvulkan [ERROR]: attn_decode supports at most ", ATTN_L_CAP,
        " cache positions, got ", l_len, ".");
    TORCH_CHECK(hd <= 8 * 32, "torchvulkan [ERROR]: attn_decode supports hd <= 256, got ", hd, ".");

    at::Tensor qc = q.contiguous();
    at::Tensor kc = k_cache.contiguous();
    at::Tensor vc = v_cache.contiguous();
    // k_cur/v_cur — строки (n_kv, hd) текущей позиции: в rms-режиме v_cur читается
    // отсюда (k берётся RAW из qk-блока), иначе читаются обе; вызывающему не нужны
    // copy_ в кэш — текущая позиция пишется в кэш самим кернелом
    at::Tensor kcc, vcc;
    if (k_cur.has_value() && v_cur.has_value())
    {
        TORCH_CHECK(k_cur->scalar_type() == q.scalar_type() && v_cur->scalar_type() == q.scalar_type(),
            "torchvulkan [ERROR]: attn_decode k_cur/v_cur dtype must match q.");
        TORCH_CHECK(k_cur->numel() == n_kv * hd && v_cur->numel() == n_kv * hd,
            "torchvulkan [ERROR]: attn_decode k_cur/v_cur must have ", n_kv * hd, " elements.");
        kcc = k_cur->reshape({n_kv, hd}).contiguous();
        vcc = v_cur->reshape({n_kv, hd}).contiguous();
    }
    at::Tensor wc, cc, sc;
    if (fuse)
    {
        wc = qkn_w->contiguous();
        cc = cos_t->contiguous();
        sc = sin_t->contiguous();
    }
    at::Tensor out = at::empty({n_kv, q_per_kv, hd}, qc.options());
    if (rows == 0 || hd == 0) return out;

    DeviceContext* device = get_device_context(qc);
    // one output dim per thread in phase B; workgroup-wide reductions go
    // through groupshared memory, so the subgroup size doesn't matter
    const uint32_t wg = 128;
    TORCH_CHECK(hd <= wg, "torchvulkan [ERROR]: attn_decode supports hd <= ", wg, ", got ", hd, ".");
    TORCH_CHECK(wg <= device->properties.limits.maxComputeWorkGroupInvocations,
        "torchvulkan [ERROR]: attn_decode workgroup of ", wg, " exceeds the device limit.");

    materialize_cow(qc);
    materialize_cow(kc);
    materialize_cow(vc);

    SpecializationBuilder spd{}; // WG задан compile-time константой в config.py
    SpecializationArgs specialization = spd.build();

    // порядок пушей = порядку полей AttnDecodePushConstants (см. шейдер):
    // 6 указателей, kv_stride/hd/l_len/rows/q_per_kv/rms_flag, 3 указателя, eps
    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(qc));
    pcs.push(get_tensor_address(kc));
    pcs.push(get_tensor_address(vc));
    pcs.push(get_output_address(out));
    pcs.push(kcc.defined() ? get_tensor_address(kcc) : uint64_t(0));
    pcs.push(vcc.defined() ? get_tensor_address(vcc) : uint64_t(0));
    pcs.push(static_cast<uint32_t>(kc.stride(1))); // elements: max_len * hd
    pcs.push(static_cast<uint32_t>(hd));
    pcs.push(static_cast<uint32_t>(l_len));
    pcs.push(static_cast<uint32_t>(rows));
    pcs.push(static_cast<uint32_t>(q_per_kv));
    pcs.push(fuse ? 1u : 0u);
    pcs.push(wc.defined() ? get_tensor_address(wc) : uint64_t(0));
    pcs.push(cc.defined() ? get_tensor_address(cc) : uint64_t(0));
    pcs.push(sc.defined() ? get_tensor_address(sc) : uint64_t(0));
    pcs.push(static_cast<float>(eps.value_or(0.0)));

    std::vector<at::Tensor> reads = {qc, kc, vc};
    if (kcc.defined()) { reads.push_back(kcc); reads.push_back(vcc); }
    if (wc.defined()) { reads.push_back(wc); reads.push_back(cc); reads.push_back(sc); }
    // кэш пишется (текущая позиция) кернелом при заданных k_cur/v_cur —
    // регистрируем и как запись для корректных зависимостей DAG
    std::vector<at::Tensor> writes = {out};
    if (kcc.defined()) { writes.push_back(kc); writes.push_back(vc); }

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::ATTN_DECODE, qc.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ reads,
        /* write = */ writes,
        static_cast<uint32_t>(rows), 1, 1
    );

    return out;
}

} // namespace torchvulkan
