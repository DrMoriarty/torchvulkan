#include <torch/extension.h>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"

// Weight-only quantized matmul, the torchvulkan analogue of ATen's
// _weight_int8pack_mm / _weight_int4pack_mm. Activations fp16/fp32, weights
// int8 (one byte per k, zero folded into the offset: value = (q - 128) * scale),
// int4 (two nibbles per byte, low nibble = even k: value = (q - zero) * scale)
// or NF4 (the same nibble layout, the nibble is an index into the 16-entry
// NormalFloat4 codebook of QLoRA: value = NF4_LUT[q] * absmax-scale),
// accumulation fp32, output in the activation dtype. Rows may be padded (the
// last dimension of w_packed may exceed K / 2 or K), padding is never read.
//
// M == 1 (decode) runs a dedicated GEMV kernel (src/shaders/ops/quant_gemv.slang)
// that dequantizes in registers while streaming the weights: the weight bytes
// are the only real traffic, so the 4-bit formats decode at roughly half the
// fp16 weight bandwidth and int8 at ~2/3 of it. Larger M dequantizes to the
// activation dtype and runs the regular fp GEMM (prefill is compute-bound anyway).

namespace {

constexpr uint32_t GEMV_WG = 128;

// weight-only formats: the `format` argument of weight_dequant; the dedicated
// weight_*_mm ops fix it per op (int8 = 0, int4 = 1, nf4 = 2)
constexpr int QUANT_FMT_INT8 = 0;
constexpr int QUANT_FMT_INT4 = 1;
constexpr int QUANT_FMT_NF4 = 2;

// NormalFloat4 codebook (QLoRA, Dettmers et al. 2023 — the published bitsandbytes
// constants); 16 quantiles of the standard normal distribution, |max| normalized
// to 1, strictly ascending, index 7 is exactly 0.0
constexpr float kNF4Codebook[16] = {
    -1.0f,    -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f,  0.0f,
     0.0796f,  0.1609f,  0.2461f,  0.3379f,  0.4407f,  0.5626f,  0.7230f,  1.0f
};

// k-values one lane covers per aligned iteration (one 32-bit weight load)
uint32_t gemv_per_lane(int fmt) { return fmt == QUANT_FMT_INT8 ? 4 : 8; }

// the NF4 dequant table as a tiny device buffer, cached per device: the GEMV
// kernel fetches dequant values with L1-resident indexed loads (a WaveReadLaneAt
// shuffle per nibble measured ~1.44x int4 GEMV time on the 5300M — the readlane
// burns ALU slots; a per-nibble buffer load ~1.26x; the byte-pair table below
// closes on int4). Entry b holds both codes of the weight byte b as ready fp32
// values: uint2 = (code of the low nibble = even k, code of the high nibble).
at::Tensor nf4_lut_buffer(DeviceContext* device, const at::TensorOptions& options)
{
    static std::mutex mu;
    static std::unordered_map<DeviceContext*, at::Tensor> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(device);
    if (it != cache.end()) return it->second;

    uint32_t bits[16];
    for (int i = 0; i < 16; i++)
    {
        float f = kNF4Codebook[i];
        std::memcpy(&bits[i], &f, 4);
    }
    std::vector<int64_t> table(256);
    for (int b = 0; b < 256; b++)
        table[b] = static_cast<int64_t>(
            static_cast<uint64_t>(bits[b & 15]) |
            (static_cast<uint64_t>(bits[b >> 4]) << 32));

    at::Tensor lut = at::empty({256}, options.dtype(at::kLong)).copy_(
        at::tensor(at::ArrayRef<int64_t>(table.data(), 256),
                   at::TensorOptions().dtype(at::kLong)));
    cache.emplace(device, lut);
    return lut;
}

at::Tensor dequantize_packed(
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& zeros,
    int fmt,
    int64_t K,
    at::ScalarType out_dtype)
{
    const int64_t N = w_packed.size(0);
    const int64_t row_bytes = w_packed.size(1);
    const int64_t ngroups = scales.size(1);
    const int64_t group = K / ngroups;

    at::Tensor w = w_packed.to(at::kInt);
    at::Tensor vals;
    if (fmt == QUANT_FMT_INT8) {
        vals = at::sub(w, 128).narrow(1, 0, K);
    } else {
        at::Tensor lo = at::bitwise_and(w, 0xF);
        at::Tensor hi = at::bitwise_and(at::bitwise_right_shift(w, 4), 0xF);
        // byte j of the row holds k = 2*j (low nibble) and k = 2*j + 1 (high nibble)
        at::Tensor q = at::stack({lo, hi}, -1).reshape({N, row_bytes * 2}).narrow(1, 0, K);
        if (fmt == QUANT_FMT_NF4) {
            // the nibble is a codebook index: value = NF4_LUT[q] * scale
            at::Tensor lut = at::tensor(at::ArrayRef<float>(kNF4Codebook, 16),
                w_packed.options().dtype(at::kFloat));
            vals = lut.index_select(0, q.reshape(-1)).view({N, K});
        } else {
            vals = q;
        }
    }

    // per-group scales (and int4 zero points): (N, ngroups) -> broadcast over k
    at::Tensor centered = vals.view({N, ngroups, group});
    if (fmt == QUANT_FMT_INT4) centered = centered - zeros->unsqueeze(-1);
    at::Tensor deq = centered * scales.unsqueeze(-1);
    return deq.view({N, K}).to(out_dtype).contiguous();
}

void check_quant_args(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& zeros,
    const std::optional<at::Tensor>& bias,
    int fmt)
{
    TORCH_CHECK(fmt >= QUANT_FMT_INT8 && fmt <= QUANT_FMT_NF4,
        "torchvulkan [ERROR]: unknown weight-only quant format ", fmt, ".");
    TORCH_CHECK(x.is_privateuseone(), "torchvulkan [ERROR]: weight_*_mm expects vulkan activations, got ", x.device(), ".");
    TORCH_CHECK(w_packed.is_privateuseone() && w_packed.scalar_type() == at::kByte && w_packed.dim() == 2,
        "torchvulkan [ERROR]: packed weights must be a 2D uint8 vulkan tensor, got dtype ", w_packed.dtype(),
        ", shape ", w_packed.sizes(), ".");
    TORCH_CHECK(scales.is_privateuseone() && scales.scalar_type() == at::kFloat && scales.dim() == 2,
        "torchvulkan [ERROR]: scales must be a 2D float32 vulkan tensor (N, ngroups), got dtype ", scales.dtype(),
        ", shape ", scales.sizes(), ".");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat,
        "torchvulkan [ERROR]: quantized matmul activations must be fp16 or fp32, got ", x.dtype(), ".");
    TORCH_CHECK(x.dim() >= 1, "torchvulkan [ERROR]: activations must have at least 1 dimension.");
    const int64_t N = w_packed.size(0);
    const int64_t K = x.size(-1);
    TORCH_CHECK(scales.size(0) == N, "torchvulkan [ERROR]: scales rows (", scales.size(0), ") must match weight rows (", N, ").");
    TORCH_CHECK(K % scales.size(1) == 0,
        "torchvulkan [ERROR]: K = ", K, " must divide into the ", scales.size(1), " scale groups along k.");
    if (fmt == QUANT_FMT_INT4) {
        TORCH_CHECK(zeros.has_value() && zeros->is_privateuseone() && zeros->scalar_type() == at::kFloat &&
            zeros->dim() == 2 && zeros->sizes() == scales.sizes(),
            "torchvulkan [ERROR]: int4 requires a zeros tensor with the same (N, ngroups) shape as scales.");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_privateuseone() && bias->scalar_type() == x.scalar_type() && bias->numel() == N,
            "torchvulkan [ERROR]: bias must be a vulkan tensor of ", N, " ", x.dtype(), " elements.");
    }
}

// the quantized GEMV kernel: one subgroup per output column, weights streamed
// along k and dequantized in registers (see quant_gemv.slang)
at::Tensor dispatch_quant_gemv(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& zeros,
    const std::optional<at::Tensor>& bias,
    int fmt,
    const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w,
    const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    DeviceContext* device = get_device_context(x);
    const c10::ScalarType x_dtype = x.scalar_type();
    const int64_t N = w_packed.size(0);
    const int64_t K = x.size(-1);
    const int64_t ngroups = scales.size(1);
    const int64_t group = K / ngroups;
    const int64_t B = x.dim() >= 2 ? x.size(0) : 1; // inner dims (except the last) are 1 on the GEMV path

    at::Tensor out = at::empty({B, N}, x.options());

    if (N == 0 || B == 0) return out;

    TORCH_CHECK(GEMV_WG <= device->properties.limits.maxComputeWorkGroupInvocations,
        "torchvulkan [ERROR]: the quantized GEMV workgroup of ", GEMV_WG, " exceeds the device limit.");
    TORCH_CHECK(B <= (int64_t)device->properties.limits.maxComputeWorkGroupCount[2],
        "torchvulkan [ERROR]: quantized GEMV batch ", B, " exceeds the device's workgroup count limit.");

    const uint32_t lanes = device->default_subgroup_size;
    const uint32_t per_lane = gemv_per_lane(fmt);
    const uint32_t step = lanes * per_lane;
    const uint32_t row_stride = static_cast<uint32_t>(w_packed.stride(0)); // bytes: uint8 rows

    auto aligned_16b = [](const at::Tensor& t) { return get_tensor_address(t) % 16 == 0; };
    // NF4 dequantizes through a cached byte-pair codebook table in device memory
    at::Tensor nf4_lut;
    if (fmt == QUANT_FMT_NF4)
    {
        nf4_lut = nf4_lut_buffer(device, x.options());
        materialize_cow(nf4_lut);
    }
    materialize_cow(x);
    materialize_cow(w_packed);
    materialize_cow(scales);
    if (zeros.has_value()) materialize_cow(*zeros);
    if (bias.has_value()) materialize_cow(*bias);
    const uint32_t isAligned =
        (GEMV_WG % lanes == 0) &&
        (K % step == 0) &&                    // whole k-range in full subgroup-wide steps
        (group % per_lane == 0) &&            // every per-lane chunk inside one scale group
        (row_stride % 16 == 0) &&             // 16-byte-aligned weight rows for the wide loads
        w_packed.is_contiguous() && scales.is_contiguous() && x.is_contiguous() &&
        aligned_16b(w_packed) && aligned_16b(x) &&
        (get_tensor_address(scales) % 4 == 0) &&
        (!zeros.has_value() || (zeros->is_contiguous() && get_tensor_address(*zeros) % 4 == 0)) &&
        (!bias.has_value() || (bias->is_contiguous() && get_tensor_address(*bias) % 2 == 0)) ? 1 : 0;
    const uint32_t hasBias = bias.has_value() ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(GEMV_WG)
       .push(isAligned)
       .push(hasBias);
    SpecializationArgs specialization = spd.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(x));
    pcs.push(get_tensor_address(w_packed));
    pcs.push(get_output_address(out));
    pcs.push(get_tensor_address(scales));
    pcs.push(zeros.has_value() ? get_tensor_address(*zeros) : (uint64_t)0);
    pcs.push(hasBias ? get_tensor_address(*bias) : (uint64_t)0);
    pcs.push(rms_w.has_value() ? get_tensor_address(*rms_w) : (uint64_t)0);
    pcs.push(static_cast<uint32_t>(N));
    pcs.push(static_cast<uint32_t>(K));
    pcs.push(row_stride);
    pcs.push(static_cast<uint32_t>(ngroups));
    pcs.push(static_cast<float>(rms_eps.value_or(0.0)));
    pcs.push(rms_eps.has_value() ? 1u : 0u);
    pcs.push(acc.has_value() ? get_tensor_address(*acc) : (uint64_t)0);
    pcs.push(x2.has_value() ? get_tensor_address(*x2) : (uint64_t)0);
    pcs.push(acc.has_value() ? 1u : 0u);
    pcs.push(x2.has_value() ? 1u : 0u);
    pcs.push(fmt == QUANT_FMT_NF4 ? get_tensor_address(nf4_lut) : (uint64_t)0);

    std::vector<at::Tensor> readTensors = {x, w_packed, scales};
    if (zeros.has_value()) readTensors.push_back(*zeros);
    if (hasBias) readTensors.push_back(*bias);
    if (rms_w.has_value()) readTensors.push_back(*rms_w);
    if (acc.has_value()) readTensors.push_back(*acc);
    if (x2.has_value()) readTensors.push_back(*x2);
    if (fmt == QUANT_FMT_NF4) readTensors.push_back(nf4_lut);

    const torchvulkan::Kernel kernel = fmt == QUANT_FMT_INT4 ? torchvulkan::Kernel::QUANT_GEMV_W4
                                     : fmt == QUANT_FMT_NF4 ? torchvulkan::Kernel::QUANT_GEMV_NF4
                                     : torchvulkan::Kernel::QUANT_GEMV_W8;
    // aligned path: each subgroup computes one output column
    const uint32_t columns_per_wg = isAligned ? GEMV_WG / std::max(lanes, (uint32_t)1) : GEMV_WG;

    device->shader_manager->dispatchShader(
        ShaderKey{kernel, x_dtype},
        device->default_subgroup_size,
        specialization,
        pcs.build(),
        /* read = */ readTensors,
        /* write = */ {out},
        (N + columns_per_wg - 1) / columns_per_wg, 1, static_cast<uint32_t>(B)
    );

    return out;
}

at::Tensor weight_quant_mm(
    const at::Tensor& x_,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& zeros_,
    const std::optional<at::Tensor>& bias_,
    int fmt,
    const std::optional<double>& rms_eps_,
    const std::optional<at::Tensor>& rms_w_,
    const std::optional<at::Tensor>& acc_,
    const std::optional<at::Tensor>& x2_)
{
    check_quant_args(x_, w_packed, scales, zeros_, bias_, fmt);

    at::Tensor x = x_.contiguous();
    at::Tensor w_packed_c = w_packed.contiguous();
    at::Tensor scales_c = scales.contiguous();
    std::optional<at::Tensor> zeros;
    if (zeros_.has_value()) zeros = zeros_->contiguous();
    std::optional<at::Tensor> bias;
    if (bias_.has_value()) bias = bias_->contiguous();

    const int64_t N = w_packed_c.size(0);
    const int64_t K = x.size(-1);

    // rows beyond the first (M of them) go through dequantize + regular GEMM;
    // the GEMV kernel serves one activation row per batch entry
    const int64_t inner = std::max<int64_t>(x.numel() / K / (x.dim() >= 2 ? x.size(0) : 1), 1);

    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end() - 1);
    out_shape.push_back(N);

    if (inner != 1) {
        TORCH_CHECK(!rms_eps_.has_value(), "torchvulkan [ERROR]: rms_eps fusion requires M == 1 activations.");
        TORCH_CHECK(!acc_.has_value(), "torchvulkan [ERROR]: acc fusion requires M == 1 activations.");
        TORCH_CHECK(!x2_.has_value(), "torchvulkan [ERROR]: x2 fusion requires M == 1 activations.");
        at::Tensor deq = dequantize_packed(w_packed_c, scales_c, zeros, fmt, K, x.scalar_type());
        at::Tensor x2d = x.reshape({-1, K});
        at::Tensor out = at::matmul(x2d, deq.t()).reshape(out_shape);
        // the GEMV path folds the bias into the kernel; the GEMM path adds it here
        if (bias.has_value()) out = out + *bias;
        return out;
    }

    std::optional<at::Tensor> acc;
    if (acc_.has_value())
    {
        const int64_t B = x.dim() >= 2 ? x.size(0) : 1;
        TORCH_CHECK(acc_->scalar_type() == x.scalar_type(),
            "torchvulkan [ERROR]: acc dtype must match activations.");
        TORCH_CHECK(acc_->numel() == B * N,
            "torchvulkan [ERROR]: acc must have ", B * N, " elements.");
        acc = acc_->contiguous();
    }

    std::optional<at::Tensor> x2;
    if (x2_.has_value())
    {
        TORCH_CHECK(!rms_eps_.has_value(),
            "torchvulkan [ERROR]: x2 (SwiGLU) and rms fusion are mutually exclusive.");
        TORCH_CHECK(x2_->scalar_type() == x.scalar_type(),
            "torchvulkan [ERROR]: x2 dtype must match activations.");
        TORCH_CHECK(x2_->numel() == x.numel(),
            "torchvulkan [ERROR]: x2 must have the same element count as activations.");
        x2 = x2_->contiguous();
    }

    at::Tensor result = dispatch_quant_gemv(x, w_packed_c, scales_c, zeros, bias, fmt, rms_eps_, rms_w_, acc, x2_);
    return result.reshape(out_shape);
}

at::Tensor weight_int8_mm_vulkan(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& bias,
    const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w,
    const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    return weight_quant_mm(x, w_packed, scales, std::nullopt, bias, QUANT_FMT_INT8, rms_eps, rms_w, acc, x2);
}

at::Tensor weight_int4_mm_vulkan(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& zeros,
    const std::optional<at::Tensor>& bias,
    const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w,
    const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    return weight_quant_mm(x, w_packed, scales, zeros, bias, QUANT_FMT_INT4, rms_eps, rms_w, acc, x2);
}

at::Tensor weight_nf4_mm_vulkan(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& bias,
    const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w,
    const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    return weight_quant_mm(x, w_packed, scales, std::nullopt, bias, QUANT_FMT_NF4, rms_eps, rms_w, acc, x2);
}

void check_dequant_args(
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const std::optional<at::Tensor>& zeros,
    int fmt,
    int64_t K,
    at::ScalarType out_dtype)
{
    TORCH_CHECK(fmt >= QUANT_FMT_INT8 && fmt <= QUANT_FMT_NF4,
        "torchvulkan [ERROR]: unknown weight-only quant format ", fmt, ".");
    TORCH_CHECK(w_packed.is_privateuseone() && w_packed.scalar_type() == at::kByte && w_packed.dim() == 2,
        "torchvulkan [ERROR]: weight_dequant expects a 2D uint8 vulkan packed weight, got dtype ",
        w_packed.dtype(), ", shape ", w_packed.sizes(), ".");
    TORCH_CHECK(scales.is_privateuseone() && scales.scalar_type() == at::kFloat && scales.dim() == 2,
        "torchvulkan [ERROR]: weight_dequant scales must be a 2D float32 vulkan tensor (N, ngroups), got dtype ",
        scales.dtype(), ", shape ", scales.sizes(), ".");
    const int64_t N = w_packed.size(0);
    const int64_t ngroups = scales.size(1);
    TORCH_CHECK(scales.size(0) == N, "torchvulkan [ERROR]: scales rows (", scales.size(0),
        ") must match weight rows (", N, ").");
    TORCH_CHECK(K > 0 && K % ngroups == 0,
        "torchvulkan [ERROR]: K = ", K, " must be positive and divide into the ", ngroups, " scale groups.");
    const int64_t k_capacity = w_packed.size(1) * (fmt == QUANT_FMT_INT8 ? 1 : 2);
    TORCH_CHECK(K <= k_capacity,
        "torchvulkan [ERROR]: K = ", K, " exceeds the packed capacity of ", k_capacity, " k-values.");
    if (fmt == QUANT_FMT_INT4) {
        TORCH_CHECK(zeros.has_value() && zeros->is_privateuseone() && zeros->scalar_type() == at::kFloat &&
            zeros->dim() == 2 && zeros->sizes() == scales.sizes(),
            "torchvulkan [ERROR]: int4 dequantization requires a zeros tensor with the same (N, ngroups) shape as scales.");
    }
    TORCH_CHECK(out_dtype == at::kHalf || out_dtype == at::kFloat,
        "torchvulkan [ERROR]: weight_dequant out_dtype must be fp16 or fp32, got ", out_dtype, ".");
}

} // namespace

// definitions live here so the registration in api/registration.cpp can bind them;
// the ops are defined on the torchvulkan library namespace (custom, like ATen's
// _weight_int8pack_mm) and dispatched on PrivateUse1

namespace torchvulkan {

// host-side escape hatch over the internal dequantize_packed: the LoRA train
// path (torchvulkan/lora.py) needs the dequantized fp16/fp32 weight for
// grad_x = grad_out @ deq(W) — either to materialize a per-layer backward
// cache or to recompute it transiently. Same kernel semantics as the M>1 path
// of weight_*_mm (the composite is identical aten ops on vulkan).
// `format`: 0 = int8, 1 = int4, 2 = nf4 (legacy bool callers pass 0/1).
at::Tensor weight_dequant(const at::Tensor& w_packed_, const at::Tensor& scales_,
    const std::optional<at::Tensor>& zeros_, int64_t format, int64_t K, at::ScalarType out_dtype)
{
    const int fmt = static_cast<int>(format);
    check_dequant_args(w_packed_, scales_, zeros_, fmt, K, out_dtype);
    return dequantize_packed(w_packed_.contiguous(), scales_.contiguous(),
        zeros_.has_value() ? std::optional<at::Tensor>(zeros_->contiguous()) : std::nullopt,
        fmt, K, out_dtype);
}

at::Tensor weight_int8_mm(const at::Tensor& x, const at::Tensor& w_packed, const at::Tensor& scales,
    const std::optional<at::Tensor>& bias, const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w, const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    return weight_int8_mm_vulkan(x, w_packed, scales, bias, rms_eps, rms_w, acc, x2);
}

at::Tensor weight_int4_mm(const at::Tensor& x, const at::Tensor& w_packed, const at::Tensor& scales,
    const at::Tensor& zeros, const std::optional<at::Tensor>& bias, const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w, const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    return weight_int4_mm_vulkan(x, w_packed, scales, zeros, bias, rms_eps, rms_w, acc, x2);
}

at::Tensor weight_nf4_mm(const at::Tensor& x, const at::Tensor& w_packed, const at::Tensor& scales,
    const std::optional<at::Tensor>& bias, const std::optional<double>& rms_eps,
    const std::optional<at::Tensor>& rms_w, const std::optional<at::Tensor>& acc,
    const std::optional<at::Tensor>& x2)
{
    return weight_nf4_mm_vulkan(x, w_packed, scales, bias, rms_eps, rms_w, acc, x2);
}

} // namespace torchvulkan
