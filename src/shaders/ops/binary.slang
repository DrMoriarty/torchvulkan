module binary;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint OP_TYPE = 0;
[[vk::constant_id(1)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(2)]] public const uint USE_SCALAR = 0;
[[vk::constant_id(3)]] public const uint NDIM = 0;
[[vk::constant_id(4)]] public const uint WG_x = 0;
[[vk::constant_id(5)]] public const uint HAS_ALPHA = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_a;
    public uint64_t strides_b;
    public uint64_t strides_out;
}

public struct BinaryPushConstants<T>
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
    public T alpha;
    public T scalar_value;
}

public interface IBinaryOp<T>
{
    static T eval(T a, T b, T alpha);
}

public struct AddOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return (HAS_ALPHA != 0) ? T.ext_add(a, T.ext_mul(alpha, b)) : T.ext_add(a, b); }
}

public struct SubOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return (HAS_ALPHA != 0) ? T.ext_sub(a, T.ext_mul(alpha, b)) : T.ext_sub(a, b); }
}

public struct RSubOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return (HAS_ALPHA != 0) ? T.ext_sub(b, T.ext_mul(alpha, a)) : T.ext_sub(b, a); }
}

public struct MulOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_mul(a, b); }
}

// fused silu(a) * b for the SwiGLU MLP (decode hotpath): one kernel instead of
// unary silu + binary mul
public struct SiluMulOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_mul(T.ext_mul(a, T.ext_sigmoid(a)), b); }
}

public struct DivOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_div(a, b); }
}

public struct PowOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_pow(a, b); }
}

public struct RPowOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_pow(b, a); }
}

public struct MaxOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_max(a, b); }
}

public struct MinOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_min(a, b); }
}

public struct ThresholdBackwardOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_threshold_backward(a, b, alpha); }
}

public struct FmaxOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_fmax(a, b); }
}

public struct FminOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_fmin(a, b); }
}

public struct FmodOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_fmod(a, b); }
}

public struct RemainderOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_remainder(a, b); }
}

public struct Atan2Op<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_atan2(a, b); }
}

public struct HypotOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_hypot(a, b); }
}

public struct XlogyOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_xlogy(a, b); }
}

public struct LogaddexpOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_logaddexp(a, b); }
}

public struct Logaddexp2Op<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_logaddexp2(a, b); }
}

public struct FloorDivideOp<T : ITorchVulkan> : IBinaryOp<T>
{
    public static T eval(T a, T b, T alpha) { return T.ext_floor_divide(a, b); }
}

public void baseBinaryKernel<T : ITorchVulkan, Op : IBinaryOp<T>, let VEC_SIZE : int>(
    uint3 gi,
    uint64_t address_a,
    uint64_t address_b,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel,
    T alpha,
    T scalar_value)
{
    uint idx = flat_thread(gi, WG_x);

    if (IS_CONTIG != 0)
    {
        if (idx * VEC_SIZE >= numel) return;

        // wide path (VEC_SIZE == 8, 2-byte T, host-checked 16-byte-aligned
        // addresses): one uint4 = 8 halfs per load/store instead of VEC_SIZE
        // scalar 2-byte accesses, which is what caps pointwise kernels at
        // ~25-55 GB/s on MoltenVK (the attn_decode.slang pattern). Arithmetic
        // stays per-element in T; the f32 round trip of f16tof32/castValue is
        // exact for values that are halfs to begin with.
        if (VEC_SIZE == 8 && sizeof(T) == 2)
        {
            uint64_t base = idx * VEC_SIZE;
            uint64_t remainder = numel - base;

            if (remainder >= VEC_SIZE)
            {
                uint4 wa = ((uint4*)address_a)[idx];
                float4 alo = f16tof32(wa);
                float4 ahi = f16tof32(wa >> 16); // in each word: even dim low, odd dim high
                T va[VEC_SIZE];
                va[0] = castValue<float, T>(alo.x); va[1] = castValue<float, T>(ahi.x);
                va[2] = castValue<float, T>(alo.y); va[3] = castValue<float, T>(ahi.y);
                va[4] = castValue<float, T>(alo.z); va[5] = castValue<float, T>(ahi.z);
                va[6] = castValue<float, T>(alo.w); va[7] = castValue<float, T>(ahi.w);

                T vb[VEC_SIZE];
                if (USE_SCALAR == 0)
                {
                    uint4 wb = ((uint4*)address_b)[idx];
                    float4 blo = f16tof32(wb);
                    float4 bhi = f16tof32(wb >> 16);
                    vb[0] = castValue<float, T>(blo.x); vb[1] = castValue<float, T>(bhi.x);
                    vb[2] = castValue<float, T>(blo.y); vb[3] = castValue<float, T>(bhi.y);
                    vb[4] = castValue<float, T>(blo.z); vb[5] = castValue<float, T>(bhi.z);
                    vb[6] = castValue<float, T>(blo.w); vb[7] = castValue<float, T>(bhi.w);
                }
                else
                {
                    [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) vb[k] = scalar_value;
                }

                uint words[4] = { 0, 0, 0, 0 };
                [ForceUnroll]
                for (int k = 0; k < VEC_SIZE; k++)
                {
                    T r = Op.eval(va[k], vb[k], alpha);
                    uint bits = f32tof16(castValue<T, float>(r));
                    if ((k & 1) == 0) words[k >> 1] = bits;
                    else words[k >> 1] |= bits << 16;
                }
                ((uint4*)address_out)[idx] = uint4(words[0], words[1], words[2], words[3]);
            }
            else
            {
                // tail (< 8 elements left): scalar
                T* tensor_a = (T*)address_a;
                T* tensor_out = (T*)address_out;
                for (uint i = 0; i < remainder; i++)
                {
                    T vb = (USE_SCALAR != 0) ? scalar_value : ((T*)address_b)[base + i];
                    tensor_out[base + i] = Op.eval(tensor_a[base + i], vb, alpha);
                }
            }
            return;
        }

        T* tensor_a = (T*)address_a;
        T* tensor_out = (T*)address_out;

        T va[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) va[k] = tensor_a[idx * VEC_SIZE + k];

        T vb[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) vb[k] = scalar_value;

        if (USE_SCALAR == 0)
        {
            T* tensor_b = (T*)address_b;
            [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) vb[k] = tensor_b[idx * VEC_SIZE + k];
        }

        T res[VEC_SIZE];
        [ForceUnroll] for (int k = 0; k < VEC_SIZE; k++) res[k] = Op.eval(va[k], vb[k], alpha);

        uint64_t remainder = numel - (idx * VEC_SIZE);
        uint64_t count = (remainder >= VEC_SIZE) ? VEC_SIZE : remainder;
        for (uint i = 0; i < count; i++) tensor_out[idx * VEC_SIZE + i] = res[i];

        return;
    }

    if (idx >= numel) return;

    T* tensor_a = (T*)address_a;
    T* tensor_out = (T*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_a = (uint*)md.strides_a;
    uint* strides_b = (uint*)md.strides_b;
    uint* strides_out = (uint*)md.strides_out;

    uint idx_a = 0;
    uint idx_b = 0;
    uint idx_out = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        idx_a += coord * strides_a[i];
        idx_b += coord * strides_b[i];
        idx_out += coord * strides_out[i];
    }

    T v1 = tensor_a[idx_a];
    T v2 = scalar_value;
    if (USE_SCALAR == 0)
    {
        T* tensor_b = (T*)address_b;
        v2 = tensor_b[idx_b];
    }

    tensor_out[idx_out] = Op.eval(v1, v2, alpha);
}

public void dispatchBinary<T : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi, BinaryPushConstants<T> pcs)
{
    switch (OP_TYPE)
    {
    case 0:  baseBinaryKernel<T, AddOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 1:  baseBinaryKernel<T, SubOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 2:  baseBinaryKernel<T, RSubOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 3:  baseBinaryKernel<T, MulOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 4:  baseBinaryKernel<T, DivOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 5:  baseBinaryKernel<T, MaxOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 6:  baseBinaryKernel<T, MinOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 7:  baseBinaryKernel<T, PowOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 8:  baseBinaryKernel<T, RPowOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 9:  baseBinaryKernel<T, Atan2Op<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 10: baseBinaryKernel<T, ThresholdBackwardOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 11: baseBinaryKernel<T, FmaxOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 12: baseBinaryKernel<T, FminOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 13: baseBinaryKernel<T, FmodOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 14: baseBinaryKernel<T, RemainderOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 15: baseBinaryKernel<T, HypotOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 16: baseBinaryKernel<T, XlogyOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 17: baseBinaryKernel<T, LogaddexpOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 18: baseBinaryKernel<T, Logaddexp2Op<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 19: baseBinaryKernel<T, FloorDivideOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    case 20: baseBinaryKernel<T, SiluMulOp<T>, VEC_SIZE>(gi, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel, pcs.alpha, pcs.scalar_value); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void binary_main<T : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform BinaryPushConstants<T> pcs)
{
    dispatchBinary<T, VEC_SIZE>(gi, pcs);
}