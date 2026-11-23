module quant_gemv;

// Weight-only quantized GEMV (matmul with M == 1): activations fp16/fp32,
// weights int8 (one byte per k), int4 (two nibbles per byte) or NF4 (the same
// nibble layout, nibble = index into the NormalFloat4 codebook), dequantized in
// registers while streaming. Reuses the matmul_gemv pattern: one subgroup per
// output column, weight reads coalesce along the row's contiguous k-stride,
// the reduction goes through WaveActiveSum, accumulation in fp32.
//
// PACK is the compile-time payload tag (scripts/config.py): 8 = int8 byte,
// 4 = int4 nibble with a per-group zero point, 2 = NF4 nibble with a codebook
// lookup (same 2-nibbles-per-byte layout as int4, low nibble = even k).
//
// Weight layout: rows of N outputs, quantized along k. Each row is padded so
// its byte stride (row_stride) is a multiple of 16 - padding nibbles/bytes are
// never read (loops stop at K). Scales and zeros are fp32 (N, ngroups),
// group = K / ngroups k-elements per group; ngroups == 1 means one scale per
// output row. int8 is symmetric: q = value / scale + 128 in [0, 255], the zero
// is folded into the constant offset and the zeros tensor is not read. int4 is
// asymmetric: value = (q - zero) * scale, q in [0, 15], low nibble = even k.
// NF4 (QLoRA, Dettmers et al. 2023): value = nf4_code(q) * scale with the
// per-block absmax as the scale, no zero point (code index 7 is exactly 0.0).
//
// Aligned path: lane covers PER_LANE consecutive k-values from one 32-bit
// load (4 bytes for int8, 8 nibbles for int4/NF4), so a warp instruction reads
// lanes * 4 contiguous bytes and every per-lane chunk sits inside one scale
// group (group % PER_LANE == 0). Fallback path: one thread per output column,
// scalar loads, per-element scale lookup.

__exported import core.dtype;

// NF4 codebook: 16 quantiles of the standard normal distribution (QLoRA,
// Dettmers et al. 2023 — the published bitsandbytes constants), |max| normalized
// to 1, strictly ascending, index 7 is exactly 0.0. The aligned path dequantizes
// through a byte-pair table in device memory (tensor_lut push constant, built
// and cached per device on the host, see quant.cpp); this static array serves
// the fallback path. Alternatives measured worse on the 5300M: a per-nibble
// WaveReadLaneAt shuffle ~1.44x int4 GEMV time (the readlane eats ~half the ALU
// bandwidth and pulls the kernel out of memory-bound), a 15-select decode tree
// ~1.46x, a per-nibble device-buffer load ~1.26x.
static const float NF4_LUT[16] = {
    -1.0,    -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0,
     0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0
};

float nf4_code(uint q)
{
    return NF4_LUT[q];
}

[[vk::constant_id(0)]] public const uint WG = 0;
[[vk::constant_id(1)]] public const uint IS_ALIGNED = 1;
[[vk::constant_id(2)]] public const uint HAS_BIAS = 0;

// silu(x) = x * sigmoid(x); fp32-математика (fast-math exp совпадает с aten
// в пределах допусков — проверено test_quant/test_fused)
float silu_v(float v)
{
    return v / (1.0 + exp(-v));
}

public struct QuantGemvPushConstants
{
    public uint64_t tensor_x;      // activations (B, 1, K), T, contiguous
    public uint64_t tensor_w;      // packed weights, uint8
    public uint64_t tensor_out;    // (B, 1, N), T
    public uint64_t tensor_scales; // (N, ngroups), float
    public uint64_t tensor_zeros;  // (N, ngroups), float, int4 only
    public uint64_t tensor_bias;   // (N,), T, optional
    public uint64_t tensor_rms_w;  // (K,), T: RMSNorm weight applied to activations, 0 = absent
    public uint N;
    public uint K;
    public uint row_stride;        // packed weight row stride in bytes
    public uint ngroups;
    public float rms_eps;          // fused RMSNorm: rrms = rsqrt(mean(x^2) + eps)
    public uint rms_flag;          // 1: activations scaled by the in-kernel rrms
    public uint64_t tensor_acc;    // (B, N), T: residual added to the result, 0 = absent
    public uint64_t tensor_x2;     // (B, K), T: SwiGLU-пара u; акт = silu(x)*u
    public uint acc_flag;          // 1: out = acc + GEMV result
    public uint x2_flag;           // 1: активации = silu(x[k]) * x2[k] (down-proj SwiGLU)
    public uint64_t tensor_lut;    // (256,) uint64, NF4 byte-pair dequant table, 0 = absent
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void quant_gemv_main<T : ITorchVulkan, let PACK : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform QuantGemvPushConstants pcs)
{
    const uint PER_LANE = PACK == 8 ? 4 : 8; // k-values per lane per aligned iteration
    const float OFFSET = PACK == 8 ? 128.0 : 0.0; // int8 folds its zero here; int4 keeps it in tensor_zeros; NF4 needs none

    // the batch rides the grid's z dimension
    uint b = wg.z;
    T* x = (T*)pcs.tensor_x + uint64_t(b) * pcs.K;
    T* out = (T*)pcs.tensor_out + uint64_t(b) * pcs.N;
    uint8_t* w = (uint8_t*)pcs.tensor_w;
    T* rms_w = (T*)pcs.tensor_rms_w;
    uint* w32 = (uint*)pcs.tensor_w; // u32 view of the same rows (no pointer bitcast)
    float* scales = (float*)pcs.tensor_scales;
    float* zeros = (float*)pcs.tensor_zeros;
    T* bias = (T*)pcs.tensor_bias;
    T* acc_in = (T*)pcs.tensor_acc;
    T* x2 = (T*)pcs.tensor_x2;
    // NF4 dequant table: entry per weight byte, uint2 = (code of the low nibble,
    // code of the high nibble) as ready fp32 values — one L1-resident u64 load
    // per byte, zero unpack ALU
    uint2* lut2 = (uint2*)pcs.tensor_lut;

    uint group = pcs.K / pcs.ngroups; // ngroups == 1 -> group == K, k/group == 0

    if (IS_ALIGNED == 1)
    {
        uint lanes = WaveGetLaneCount();
        uint wave = ti / lanes;
        uint lane = ti % lanes;
        uint waves = WG / lanes; // output columns per workgroup

        uint n = wg.x * waves + wave;
        if (n >= pcs.N)
        {
            return;
        }

        uint step = lanes * PER_LANE;

        // fused RMSNorm: rrms по активациям считает сам кернел (второй проход
        // по x — он L1-горячий), вес нормы применяется к активациям поэлементно
        float rrms = 1.0;
        if (pcs.rms_flag == 1)
        {
            float ss = 0;
            for (uint k0 = 0; k0 < pcs.K; k0 += step)
            {
                uint k = k0 + lane * PER_LANE;
                [unroll]
                for (uint j = 0; j < PER_LANE; j++)
                {
                    float xv = castValue<T, float>(x[k + j]);
                    ss += xv * xv;
                }
            }
            ss = WaveActiveSum(ss);
            rrms = rsqrt(ss / float(pcs.K) + pcs.rms_eps);
        }

        // all math in fp32: Accum for both fp16 and fp32 activations
        float acc = 0;
        for (uint k0 = 0; k0 < pcs.K; k0 += step)
        {
            uint k = k0 + lane * PER_LANE;
            uint g = n * pcs.ngroups + k / group;
            float sc = scales[g];
            float z = PACK == 4 ? zeros[g] : OFFSET;
            // row stride is a multiple of 16 bytes, every per-lane chunk 4-byte aligned:
            // the 32-bit weight word is indexed directly in the u32 view of the buffer
            uint word = n * (pcs.row_stride >> 2) + (PACK == 8 ? (k >> 2) : (k >> 3));
            uint wb = w32[word];

            // the whole per-lane chunk sits in one scale group
            float dot = 0;
            if (PACK == 2)
            {
                // NF4: one u64 table load per weight byte, both dequantized codes
                // ready in the uint2 (low nibble = even k -> .x, high -> .y). The
                // 2 KB table stays hot in L1 and the load rides the LSU pipe —
                // a per-nibble WaveReadLaneAt shuffle here measured ~1.44x int4
                // GEMV time (readlane burns ALU slots), a per-nibble buffer load
                // ~1.26x, this byte-pair form closes on int4.
                [unroll]
                for (uint jb = 0; jb < 4; jb++)
                {
                    uint bw = (wb >> (8 * jb)) & 0xFF;
                    uint2 pair = lut2[bw];
                    uint k0e = k + 2 * jb;
                    float av0, av1;
                    if (pcs.x2_flag == 1)
                    {
                        av0 = silu_v(castValue<T, float>(x[k0e])) * castValue<T, float>(x2[uint64_t(b) * pcs.K + k0e]);
                        av1 = silu_v(castValue<T, float>(x[k0e + 1])) * castValue<T, float>(x2[uint64_t(b) * pcs.K + k0e + 1]);
                    }
                    else
                    {
                        float coef0 = rrms;
                        float coef1 = rrms;
                        if (pcs.tensor_rms_w != 0)
                        {
                            coef0 *= castValue<T, float>(rms_w[k0e]);
                            coef1 *= castValue<T, float>(rms_w[k0e + 1]);
                        }
                        av0 = castValue<T, float>(x[k0e]) * coef0;
                        av1 = castValue<T, float>(x[k0e + 1]) * coef1;
                    }
                    dot += asfloat(pair.x) * av0 + asfloat(pair.y) * av1;
                }
            }
            else
            {
                [unroll]
                for (uint j = 0; j < PER_LANE; j++)
                {
                    uint q = PACK == 8 ? ((wb >> (8 * j)) & 0xFF) : ((wb >> (4 * j)) & 0xF);
                    // int8/int4 shift the integer code by the zero point
                    float wv = (float(q) - z);
                    float coef = rrms;
                    if (pcs.tensor_rms_w != 0)
                        coef *= castValue<T, float>(rms_w[k + j]);
                    float av;
                    if (pcs.x2_flag == 1)
                        av = silu_v(castValue<T, float>(x[k + j])) * castValue<T, float>(x2[uint64_t(b) * pcs.K + k + j]);
                    else
                        av = castValue<T, float>(x[k + j]) * coef;
                    dot += wv * av;
                }
            }
            acc += dot * sc;
        }
        acc = WaveActiveSum(acc);

        if (lane == 0)
        {
            if (HAS_BIAS == 1) acc += castValue<T, float>(bias[n]);
            if (pcs.acc_flag == 1) acc += castValue<T, float>(acc_in[uint64_t(b) * pcs.N + n]);
            out[n] = castValue<float, T>(acc);
        }
    }
    else
    {
        // arbitrary K/group/row alignment: one thread per output column, scalar loads
        uint n = wg.x * WG + ti;
        if (n >= pcs.N) return;

        uint8_t* wrow = w + uint64_t(n) * pcs.row_stride;

        float rrms = 1.0;
        if (pcs.rms_flag == 1)
        {
            float ss = 0;
            for (uint k = 0; k < pcs.K; k++)
            {
                float xv = castValue<T, float>(x[k]);
                ss += xv * xv;
            }
            rrms = rsqrt(ss / float(pcs.K) + pcs.rms_eps);
        }

        float acc = 0;
        for (uint k = 0; k < pcs.K; k++)
        {
            uint q;
            if (PACK == 8)
            {
                q = wrow[k];
            }
            else
            {
                uint qb = wrow[k >> 1];
                q = (k & 1) != 0 ? ((qb >> 4) & 0xF) : (qb & 0xF);
            }
            uint g = n * pcs.ngroups + k / group;
            float z = PACK == 4 ? zeros[g] : OFFSET;
            float wv = (PACK == 2) ? nf4_code(q) : (float(q) - z);
            float av;
            if (pcs.x2_flag == 1)
                av = silu_v(castValue<T, float>(x[k])) * castValue<T, float>(x2[uint64_t(b) * pcs.K + k]);
            else
            {
                float coef = rrms;
                if (pcs.tensor_rms_w != 0)
                    coef *= castValue<T, float>(rms_w[k]);
                av = castValue<T, float>(x[k]) * coef;
            }
            acc += wv * scales[g] * av;
        }

        if (HAS_BIAS == 1) acc += castValue<T, float>(bias[n]);
        if (pcs.acc_flag == 1) acc += castValue<T, float>(acc_in[uint64_t(b) * pcs.N + n]);
        out[n] = castValue<float, T>(acc);
    }
}
