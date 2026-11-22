module add_relu;

// Fused residual add + ReLU (training_speedup.md Б2): one kernel instead of
// aten add + relu, writing BOTH the pre-activation (the residual sum, which
// relu's backward needs) and the activated result. Only the resnet's
// `F.relu(out + shortcut)` shape is supported: equal shapes, contiguous,
// 16-byte-aligned fp16/fp32 tensors (host-checked, see api/ops/fusion.cpp).

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint VEC = 1;       // 8: fp16 uint4 pairs, 4: fp32
[[vk::constant_id(1)]] public const uint HAS_ALPHA = 0;
[[vk::constant_id(2)]] public const uint WG_x = 256;

public struct AddReluPushConstants
{
    public uint64_t tensor_a;
    public uint64_t tensor_b;
    public uint64_t tensor_pre;
    public uint64_t tensor_out;
    public uint64_t numel;
    public float alpha;
}

// scalar path, one element at a time (vector tails): the sum is accumulated in
// fp32 for half, pre rounds to T like a standalone store, relu applies to the
// fp32 sum — rounding and relu are both monotone, so relu(pre) and
// round(relu(sum)) are bit-identical
void addReluScalar<T : ITorchVulkan>(uint64_t base, uint64_t count, AddReluPushConstants pcs)
{
    T* tensor_a = (T*)pcs.tensor_a;
    T* tensor_b = (T*)pcs.tensor_b;
    T* tensor_pre = (T*)pcs.tensor_pre;
    T* tensor_out = (T*)pcs.tensor_out;

    for (uint64_t k = 0; k < count; k++)
    {
        float av = castValue<T, float>(tensor_a[base + k]);
        float bv = castValue<T, float>(tensor_b[base + k]);
        float s = (HAS_ALPHA != 0) ? av + pcs.alpha * bv : av + bv;
        tensor_pre[base + k] = castValue<float, T>(s);
        tensor_out[base + k] = castValue<float, T>(max(s, 0.0));
    }
}

// wide fp16 path: one uint4 = 8 halfs per load/store (see unary.slang for the
// pattern notes; in each word the even dim is the low half, the odd the high)
void addReluWide16<T : ITorchVulkan>(uint idx, AddReluPushConstants pcs)
{
    uint64_t base = idx * 8;
    uint64_t remainder = pcs.numel - base;
    if (base >= pcs.numel) return;

    if (remainder >= 8)
    {
        uint4 wa = ((uint4*)pcs.tensor_a)[idx];
        uint4 wb = ((uint4*)pcs.tensor_b)[idx];
        float4 alo = f16tof32(wa);
        float4 ahi = f16tof32(wa >> 16);
        float4 blo = f16tof32(wb);
        float4 bhi = f16tof32(wb >> 16);

        float sums[8];
        sums[0] = alo.x + blo.x; sums[1] = ahi.x + bhi.x;
        sums[2] = alo.y + blo.y; sums[3] = ahi.y + bhi.y;
        sums[4] = alo.z + blo.z; sums[5] = ahi.z + bhi.z;
        sums[6] = alo.w + blo.w; sums[7] = ahi.w + bhi.w;
        if (HAS_ALPHA != 0)
        {
            sums[0] = alo.x + pcs.alpha * blo.x; sums[1] = ahi.x + pcs.alpha * bhi.x;
            sums[2] = alo.y + pcs.alpha * blo.y; sums[3] = ahi.y + pcs.alpha * bhi.y;
            sums[4] = alo.z + pcs.alpha * blo.z; sums[5] = ahi.z + pcs.alpha * bhi.z;
            sums[6] = alo.w + pcs.alpha * blo.w; sums[7] = ahi.w + pcs.alpha * bhi.w;
        }

        uint words_pre[4] = { 0, 0, 0, 0 };
        uint words_out[4] = { 0, 0, 0, 0 };
        [ForceUnroll]
        for (int k = 0; k < 8; k++)
        {
            uint pre_bits = f32tof16(sums[k]);
            uint out_bits = f32tof16(max(sums[k], 0.0));
            if ((k & 1) == 0)
            {
                words_pre[k >> 1] = pre_bits;
                words_out[k >> 1] = out_bits;
            }
            else
            {
                words_pre[k >> 1] |= pre_bits << 16;
                words_out[k >> 1] |= out_bits << 16;
            }
        }
        ((uint4*)pcs.tensor_pre)[idx] = uint4(words_pre[0], words_pre[1], words_pre[2], words_pre[3]);
        ((uint4*)pcs.tensor_out)[idx] = uint4(words_out[0], words_out[1], words_out[2], words_out[3]);
    }
    else
    {
        addReluScalar<T>(base, remainder, pcs); // tail: scalar per element
    }
}

void addReluWide32<T : ITorchVulkan>(uint idx, AddReluPushConstants pcs)
{
    uint64_t base = idx * 4;
    uint64_t remainder = pcs.numel - base;
    if (base >= pcs.numel) return;

    if (remainder >= 4)
    {
        float4 a = ((float4*)pcs.tensor_a)[idx];
        float4 b = ((float4*)pcs.tensor_b)[idx];
        float4 s = (HAS_ALPHA != 0) ? a + pcs.alpha * b : a + b;
        ((float4*)pcs.tensor_pre)[idx] = s;
        ((float4*)pcs.tensor_out)[idx] = max(s, float4(0.0, 0.0, 0.0, 0.0));
    }
    else
    {
        addReluScalar<T>(base, remainder, pcs); // tail: scalar per element
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_relu_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform AddReluPushConstants pcs)
{
    uint idx = flat_thread(gi, WG_x);

    if (VEC == 8) addReluWide16<T>(idx, pcs);
    else addReluWide32<T>(idx, pcs);
}
