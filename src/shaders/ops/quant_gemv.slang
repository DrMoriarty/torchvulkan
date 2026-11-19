module quant_gemv;

// Weight-only quantized GEMV (matmul with M == 1): activations fp16/fp32,
// weights int8 (one byte per k) or int4 (two nibbles per byte), dequantized in
// registers while streaming. Reuses the matmul_gemv pattern: one subgroup per
// output column, weight reads coalesce along the row's contiguous k-stride,
// the reduction goes through WaveActiveSum, accumulation in fp32.
//
// Weight layout: rows of N outputs, quantized along k. Each row is padded so
// its byte stride (row_stride) is a multiple of 16 - padding nibbles/bytes are
// never read (loops stop at K). Scales and zeros are fp32 (N, ngroups),
// group = K / ngroups k-elements per group; ngroups == 1 means one scale per
// output row. int8 is symmetric: q = value / scale + 128 in [0, 255], the zero
// is folded into the constant offset and the zeros tensor is not read. int4 is
// asymmetric: value = (q - zero) * scale, q in [0, 15], low nibble = even k.
//
// Aligned path: lane covers PER_LANE consecutive k-values from one 32-bit
// load (4 bytes for int8, 8 nibbles for int4), so a warp instruction reads
// lanes * 4 contiguous bytes and every per-lane chunk sits inside one scale
// group (group % PER_LANE == 0). Fallback path: one thread per output column,
// scalar loads, per-element scale lookup.

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint WG = 0;
[[vk::constant_id(1)]] public const uint IS_ALIGNED = 1;
[[vk::constant_id(2)]] public const uint HAS_BIAS = 0;

public struct QuantGemvPushConstants
{
    public uint64_t tensor_x;      // activations (B, 1, K), T, contiguous
    public uint64_t tensor_w;      // packed weights, uint8
    public uint64_t tensor_out;    // (B, 1, N), T
    public uint64_t tensor_scales; // (N, ngroups), float
    public uint64_t tensor_zeros;  // (N, ngroups), float, int4 only
    public uint64_t tensor_bias;   // (N,), T, optional
    public uint N;
    public uint K;
    public uint row_stride;        // packed weight row stride in bytes
    public uint ngroups;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void quant_gemv_main<T : ITorchVulkan, let PACK : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform QuantGemvPushConstants pcs)
{
    const uint PER_LANE = PACK == 4 ? 8 : 4; // k-values per lane per aligned iteration
    const float OFFSET = PACK == 4 ? 0.0 : 128.0; // int4 keeps its zero in tensor_zeros

    // the batch rides the grid's z dimension
    uint b = wg.z;
    T* x = (T*)pcs.tensor_x + uint64_t(b) * pcs.K;
    T* out = (T*)pcs.tensor_out + uint64_t(b) * pcs.N;
    uint8_t* w = (uint8_t*)pcs.tensor_w;
    uint* w32 = (uint*)pcs.tensor_w; // u32 view of the same rows (no pointer bitcast)
    float* scales = (float*)pcs.tensor_scales;
    float* zeros = (float*)pcs.tensor_zeros;
    T* bias = (T*)pcs.tensor_bias;

    uint group = pcs.K / pcs.ngroups; // ngroups == 1 -> group == K, k/group == 0

    if (IS_ALIGNED == 1)
    {
        uint lanes = WaveGetLaneCount();
        uint wave = ti / lanes;
        uint lane = ti % lanes;
        uint waves = WG / lanes; // output columns per workgroup

        uint n = wg.x * waves + wave;
        if (n >= pcs.N) return;

        uint step = lanes * PER_LANE;

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
            uint word = n * (pcs.row_stride >> 2) + (PACK == 4 ? (k >> 3) : (k >> 2));
            uint wb = w32[word];

            // the whole per-lane chunk sits in one scale group
            float dot = 0;
            [unroll]
            for (uint j = 0; j < PER_LANE; j++)
            {
                uint q = PACK == 4 ? ((wb >> (4 * j)) & 0xF) : ((wb >> (8 * j)) & 0xFF);
                dot += (float(q) - z) * castValue<T, float>(x[k + j]);
            }
            acc += dot * sc;
        }
        acc = WaveActiveSum(acc);

        if (lane == 0)
        {
            if (HAS_BIAS == 1) acc += castValue<T, float>(bias[n]);
            out[n] = castValue<float, T>(acc);
        }
    }
    else
    {
        // arbitrary K/group/row alignment: one thread per output column, scalar loads
        uint n = wg.x * WG + ti;
        if (n >= pcs.N) return;

        uint8_t* wrow = w + uint64_t(n) * pcs.row_stride;

        float acc = 0;
        for (uint k = 0; k < pcs.K; k++)
        {
            uint q;
            if (PACK == 4)
            {
                uint qb = wrow[k >> 1];
                q = (k & 1) != 0 ? ((qb >> 4) & 0xF) : (qb & 0xF);
            }
            else
            {
                q = wrow[k];
            }
            uint g = n * pcs.ngroups + k / group;
            float z = PACK == 4 ? zeros[g] : OFFSET;
            acc += (float(q) - z) * scales[g] * castValue<T, float>(x[k]);
        }

        if (HAS_BIAS == 1) acc += castValue<T, float>(bias[n]);
        out[n] = castValue<float, T>(acc);
    }
}
