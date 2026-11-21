module copy;

import core.indexing;

[[vk::constant_id(0)]] public const uint NDIM = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_in;
    public uint64_t strides_out;
}

public struct CopyPushConstants
{
    public uint64_t tensor_in;
    public uint64_t tensor_out;
    public uint64_t metadata;
    public uint64_t numel;
}

public void copyKernel<T>(
    uint3 gi,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel)
{
    uint idx = flat_thread(gi, WG_x);
    if (idx >= numel) return;

    T* tensor_in = (T*)address_in;
    T* tensor_out = (T*)address_out;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;
    uint* strides_out = (uint*)md.strides_out;

    uint idx_in = 0;
    uint idx_out = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        idx_in += coord * strides_in[i];
        idx_out += coord * strides_out[i];
    }

    tensor_out[idx_out] = tensor_in[idx_in];
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void copy_main<T>(
    uint3 gi : SV_DispatchThreadID,
    uniform CopyPushConstants pcs)
{
    copyKernel<T>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel);
}

// wide strided copy (2-byte width variants, i.e. fp16/bf16/int16 tensors): one
// uint4 = 8 elements per load/store along a contiguous innermost dimension,
// instead of scalar 2-byte accesses. The host routes only shapes where the
// innermost size is a multiple of 8, its stride is 1 on both sides, every outer
// stride is a multiple of 8 (so each vector's element base stays 16-byte
// aligned) and both base addresses are 16-byte aligned.
public void copyKernelW<T>(
    uint3 gi,
    uint64_t address_in,
    uint64_t address_out,
    uint64_t address_metadata,
    uint64_t numel)
{
    uint idx = flat_thread(gi, WG_x);
    uint64_t vectors = numel / 8;
    if (idx >= vectors) return;

    Metadata md = *(Metadata*)address_metadata;
    uint* strides_in = (uint*)md.strides_in;
    uint* strides_out = (uint*)md.strides_out;

    uint64_t off_in = 0;
    uint64_t off_out = 0;
    uint inner = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx * 8);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        if (i + 1 == NDIM)
        {
            inner = coord; // innermost stride is 1 on both sides (host-checked)
        }
        else
        {
            off_in += (uint64_t)coord * strides_in[i];
            off_out += (uint64_t)coord * strides_out[i];
        }
    }

    uint64_t base_in = off_in + inner;
    uint64_t base_out = off_out + inner;
    uint4 w = ((uint4*)address_in)[base_in / 8];
    ((uint4*)address_out)[base_out / 8] = w;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void copy_main_w<T>(
    uint3 gi : SV_DispatchThreadID,
    uniform CopyPushConstants pcs)
{
    copyKernelW<T>(gi, pcs.tensor_in, pcs.tensor_out, pcs.metadata, pcs.numel);
}
