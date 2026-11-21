module where;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint IS_CONTIG = 0;
[[vk::constant_id(1)]] public const uint NDIM = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_cond;
    public uint64_t strides_a;
    public uint64_t strides_b;
}

public struct WherePushConstants
{
    public uint64_t tensor_cond;
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
}

public void whereKernel<T : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi,
    uint64_t address_cond,
    uint64_t address_a,
    uint64_t address_b,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel)
{
    uint idx = flat_thread(gi, WG_x);

    // wide contiguous path (VEC_SIZE == 8, 2-byte T, host-checked 16-byte-
    // aligned a/b/out and 8-byte-aligned cond): 8 elements per thread, the
    // condition bytes in one uint64_t and each half tensor in one uint4,
    // instead of 3*8 scalar 1-2-byte accesses; see binary.slang for notes.
    if (IS_CONTIG != 0 && VEC_SIZE == 8 && sizeof(T) == 2)
    {
        uint64_t base = idx * VEC_SIZE;
        if (base >= numel) return;
        uint64_t remainder = numel - base;

        if (remainder >= VEC_SIZE)
        {
            uint64_t cw = ((uint64_t*)address_cond)[idx];
            uint4 wa = ((uint4*)address_a)[idx];
            uint4 wb = ((uint4*)address_b)[idx];
            float4 alo = f16tof32(wa);
            float4 ahi = f16tof32(wa >> 16); // in each word: even dim low, odd dim high
            float4 blo = f16tof32(wb);
            float4 bhi = f16tof32(wb >> 16);
            T va[VEC_SIZE];
            va[0] = castValue<float, T>(alo.x); va[1] = castValue<float, T>(ahi.x);
            va[2] = castValue<float, T>(alo.y); va[3] = castValue<float, T>(ahi.y);
            va[4] = castValue<float, T>(alo.z); va[5] = castValue<float, T>(ahi.z);
            va[6] = castValue<float, T>(alo.w); va[7] = castValue<float, T>(ahi.w);
            T vb[VEC_SIZE];
            vb[0] = castValue<float, T>(blo.x); vb[1] = castValue<float, T>(bhi.x);
            vb[2] = castValue<float, T>(blo.y); vb[3] = castValue<float, T>(bhi.y);
            vb[4] = castValue<float, T>(blo.z); vb[5] = castValue<float, T>(bhi.z);
            vb[6] = castValue<float, T>(blo.w); vb[7] = castValue<float, T>(bhi.w);

            uint words[4] = { 0, 0, 0, 0 };
            [ForceUnroll]
            for (int k = 0; k < VEC_SIZE; k++)
            {
                uint8_t cbyte = uint8_t((cw >> (8 * k)) & uint64_t(0xFF));
                T r = (cbyte != uint8_t(0)) ? va[k] : vb[k];
                uint bits = f32tof16(castValue<T, float>(r));
                if ((k & 1) == 0) words[k >> 1] = bits;
                else words[k >> 1] |= bits << 16;
            }
            ((uint4*)address_out)[idx] = uint4(words[0], words[1], words[2], words[3]);
        }
        else
        {
            // tail (< 8 elements left): scalar
            uint8_t* tensor_cond = (uint8_t*)address_cond;
            T* tensor_a = (T*)address_a;
            T* tensor_b = (T*)address_b;
            T* tensor_out = (T*)address_out;
            for (uint i = 0; i < remainder; i++)
                tensor_out[base + i] = (tensor_cond[base + i] != uint8_t(0)) ? tensor_a[base + i] : tensor_b[base + i];
        }
        return;
    }

    if (idx >= numel) return;

    uint8_t* tensor_cond = (uint8_t*)address_cond;
    T* tensor_a = (T*)address_a;
    T* tensor_b = (T*)address_b;
    T* tensor_out = (T*)address_out;

    uint idx_c = idx;
    uint idx_a = idx;
    uint idx_b = idx;

    if (IS_CONTIG == 0)
    {
        Metadata md = *(Metadata*)address_metadata;
        uint* strides_cond = (uint*)md.strides_cond;
        uint* strides_a = (uint*)md.strides_a;
        uint* strides_b = (uint*)md.strides_b;

        idx_c = 0;
        idx_a = 0;
        idx_b = 0;

        var walker = IndexWalker(md.sizes, NDIM, idx);
        for (uint i = 0; i < NDIM; i++)
        {
            uint coord = walker.next(i);
            idx_c += coord * strides_cond[i];
            idx_a += coord * strides_a[i];
            idx_b += coord * strides_b[i];
        }
    }

    tensor_out[idx] = (tensor_cond[idx_c] != uint8_t(0)) ? tensor_a[idx_a] : tensor_b[idx_b];
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void where_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform WherePushConstants pcs)
{
    whereKernel<T, 1>(gi, pcs.tensor_cond, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel);
}

// wide fp16 variant: VEC_SIZE comes in as 8 (config.py 'wide_vec'), the host
// routes only fully contiguous, aligned shapes here
[shader("compute")]
[numthreads(WG_x, 1, 1)]
void where_main_w<T : ITorchVulkan, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform WherePushConstants pcs)
{
    whereKernel<T, VEC_SIZE>(gi, pcs.tensor_cond, pcs.tensor_a, pcs.tensor_b, pcs.tensor_out, pcs.metadata, pcs.numel);
}
