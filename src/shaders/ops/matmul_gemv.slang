module matmul_gemv;

// Dedicated GEMV kernel (matmul with M == 1). The general GEMM kernels stage
// tiles through shared memory, which costs two barriers per tile and
// per-element loads - for a single output row the weights dominate, so the
// fastest pattern is a straight streaming dot product.
//
// Aligned path: one subgroup per output column. Lane i covers WV consecutive
// k-elements, so a subgroup instruction reads lanes*WV contiguous elements
// (512 bytes for fp16 with 32 lanes) and the reduction goes through
// WaveActiveSum. Fallback path: one thread per output column, scalar loads.

__exported import core.dtype;
__exported import ops.unary_ops;

[[vk::constant_id(0)]] public const uint WG = 0;
[[vk::constant_id(1)]] public const uint IS_ALIGNED = 1;
[[vk::constant_id(2)]] public const uint HAS_BIAS = 0;
[[vk::constant_id(3)]] public const uint ACTIVATION = 0;

// same layout as MatmulPushConstants in ops/matmul_simd.slang (the host builds
// one push-constant block for both kernels)
public struct TensorStrides
{
    public uint batch;
    public uint row;
    public uint col;
    public uint padding;
}

public struct GemvPushConstants<T>
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t tensor_bias;
    public TensorStrides strides_A;
    public TensorStrides strides_B;
    public TensorStrides strides_C;
    public TensorStrides strides_bias;
    public uint M;
    public uint N;
    public uint K;
    public uint padding;
    public T alpha;
    public T beta;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void matmul_gemv_main<T : ITorchVulkan, let WV : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform GemvPushConstants<T.Accum> pcs)
{
    typealias Acc = T.Accum;

    // the batch rides the grid's z dimension
    uint b = wg.z;
    T* a = (T*)pcs.tensor_a + uint64_t(b) * pcs.strides_A.batch;
    T* b_ = (T*)pcs.tensor_b + uint64_t(b) * pcs.strides_B.batch;
    T* out = (T*)pcs.tensor_out + uint64_t(b) * pcs.strides_C.batch;
    T* bias = (T*)pcs.tensor_bias + uint64_t(b) * pcs.strides_bias.batch;

    if (IS_ALIGNED == 1)
    {
        uint lanes = WaveGetLaneCount();
        uint wave = ti / lanes;
        uint lane = ti % lanes;
        uint waves = WG / lanes; // output columns per workgroup

        uint n = wg.x * waves + wave;
        if (n >= pcs.N) return;

        uint bcol = pcs.strides_B.col;
        uint step = lanes;
        Acc acc = Acc(0);
        for (uint k0 = 0; k0 < pcs.K; k0 += step)
        {
            uint k = k0 + lane;
            Acc w = b_[k + n * bcol].toAccum();
            Acc x = a[k].toAccum();
            acc += w * x;
        }
        acc = WaveActiveSum(acc);

        if (lane == 0)
        {
            Acc value = acc * pcs.alpha;
            if (HAS_BIAS == 1) value = value + bias[n * pcs.strides_bias.col].toAccum() * pcs.beta;
            switch (ACTIVATION)
            {
            case 1: value = ReluOp<Acc>.eval(value); break;
            case 2: value = GeluOp<Acc>.eval(value); break;
            default: break;
            }
            out[n * pcs.strides_C.col] = T.fromAccum(value);
        }
    }
    else
    {
        // arbitrary strides: one thread per output column, scalar loads
        uint n = wg.x * WG + ti;
        if (n >= pcs.N) return;

        Acc acc = Acc(0);
        for (uint k = 0; k < pcs.K; ++k)
        {
            Acc w = b_[k * pcs.strides_B.row + n * pcs.strides_B.col].toAccum();
            Acc x = a[k * pcs.strides_A.col].toAccum();
            acc += w * x;
        }

        Acc value = acc * pcs.alpha;
        if (HAS_BIAS == 1) value = value + bias[n * pcs.strides_bias.col].toAccum() * pcs.beta;
        switch (ACTIVATION)
        {
        case 1: value = ReluOp<Acc>.eval(value); break;
        case 2: value = GeluOp<Acc>.eval(value); break;
        default: break;
        }
        out[n * pcs.strides_C.col] = T.fromAccum(value);
    }
}
