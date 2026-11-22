module batch_norm;

// Fused native batch_norm for the training path (A4, experiments/training_speedup.md):
// the aten composition (fp32 materialisation, three sequential multi-dim
// reductions, a pointwise var/apply chain — 15-20 kernels per fwd, the same
// again in the decomposed backward) is replaced by three kernels per call.
//
//   bn_partials_main   phase 0: one workgroup per (channel, chunk) accumulates
//                      fp32 sums over the channel's elements — fwd statistics
//                      (sum, sumsq) or backward sums (sum_dy, sum_dy*xhat);
//   bn_finalize_main   one thread per channel folds the chunks: fwd computes
//                      mean/rstd, writes the fp32 saves and updates the running
//                      stats; backward emits grad_bias/grad_weight (= the raw
//                      sums) in the parameter dtype;
//   bn_apply_main      elementwise pass: fwd out = (x-mean)*rstd*w + b,
//                      backward grad_input = w*rstd*(dy - mean_dy - xhat*mean_dy_xhat)
//                      (train) or dy*w*rstd (eval).
//
// Memory layout: input (N0, C, S...) contiguous NCHW-ish — channel c holds N0
// contiguous blocks of S elements. The virtual element v of a channel maps to
// flat (seg*C + c)*S + off, seg = v / S. fp16 streams are read uint4-wide
// (8 halfs) when every 16-element group stays inside one block (S % 16 == 0)
// and the chunk grid is 16-element aligned (host-guaranteed chunk_size);
// otherwise, and for fp32, the scalar coalesced walk runs.
//
// Accumulation is fp32 everywhere; var = E[x^2] - mean^2 (clamped at 0), the
// same fp32-accumulation class as the aten path. Save_mean/save_rstd are fp32
// for every input dtype, matching native_batch_norm's contract.

import core.indexing;
__exported import core.dtype;

static const uint WG = 128;

// MODE: 0 = fwd statistics, 1 = backward sums
public struct BnPartialsPushConstants
{
    public uint64_t tensor_x;    // input (mode 0) / x, kept for xhat (mode 1)
    public uint64_t tensor_y;    // mode 1: grad_out; unused (0) in mode 0
    public uint64_t tensor_mean; // mode 1: save_mean, fp32 (C)
    public uint64_t tensor_rstd; // mode 1: save_invstd, fp32 (C)
    public uint64_t tensor_part; // out: fp32 [C, n_chunks, 2] interleaved pairs
    public uint C;
    public uint S;
    public uint NHW;             // elements per channel (N0 * S)
    public uint n_chunks;
    public uint chunk_size;
    public uint mode;
}

groupshared float sm0[WG];
groupshared float sm1[WG];

// 8 fp16 from one uint4: in word w[k] the even element sits in the low bits,
// the odd one in the high (the reduce_tree/attn_decode pattern)
void accumulateHalf8(uint4 w, uint mode, float mean_c, float rstd_c, inout float acc0, inout float acc1)
{
    float4 lo = f16tof32(w);
    float4 hi = f16tof32(w >> 16);
    float e0 = lo.x, e1 = hi.x, e2 = lo.y, e3 = hi.y;
    float e4 = lo.z, e5 = hi.z, e6 = lo.w, e7 = hi.w;
    if (mode == 0)
    {
        acc0 += e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc1 += e0 * e0 + e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4 + e5 * e5 + e6 * e6 + e7 * e7;
    }
    else
    {
        float4 xh0 = float4((e0 - mean_c) * rstd_c, (e1 - mean_c) * rstd_c, (e2 - mean_c) * rstd_c, (e3 - mean_c) * rstd_c);
        float4 xh1 = float4((e4 - mean_c) * rstd_c, (e5 - mean_c) * rstd_c, (e6 - mean_c) * rstd_c, (e7 - mean_c) * rstd_c);
        float4 d0 = float4(e0, e1, e2, e3);
        float4 d1 = float4(e4, e5, e6, e7);
        acc0 += d0.x + d0.y + d0.z + d0.w + d1.x + d1.y + d1.z + d1.w;
        acc1 += dot(d0, xh0) + dot(d1, xh1);
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void bn_partials_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform BnPartialsPushConstants pcs)
{
    uint gid = flat_group(wg);
    uint c = gid / pcs.n_chunks;
    uint chunk = gid - c * pcs.n_chunks;
    if (c >= pcs.C) return; // whole workgroup leaves, before the barriers

    uint base = chunk * pcs.chunk_size;
    uint stop = min(base + pcs.chunk_size, pcs.NHW);
    float mean_c = 0.0;
    float rstd_c = 0.0;
    if (pcs.mode == 1)
    {
        mean_c = ((float*)pcs.tensor_mean)[c];
        rstd_c = ((float*)pcs.tensor_rstd)[c];
    }

    float acc0 = 0.0;
    float acc1 = 0.0;
    // vector body: fp16 with 16-element-aligned blocks and chunks
    bool can_vec = (sizeof(T) == 2) && (pcs.S % 16 == 0) && (pcs.chunk_size % 16 == 0);

    uint seg = base / pcs.S;
    uint v = base;
    while (v < stop)
    {
        uint seg_stop = min(stop, (seg + 1) * pcs.S);
        uint64_t seg_base = (uint64_t(seg) * pcs.C + c) * pcs.S; // element offset of this (seg, c) block
        uint off = v - seg * pcs.S;
        uint off_end = seg_stop - seg * pcs.S;
        T* xp = (T*)pcs.tensor_x + seg_base;

        if (can_vec)
        {
            uint len = off_end - off;
            uint vec_len = len & ~(WG * 8 - 1u);
            if (len >= WG * 8)
            {
                uint4* vp = (uint4*)(pcs.tensor_x + (seg_base + off) * sizeof(T));
                uint4* yv = (uint4*)(pcs.tensor_y + (seg_base + off) * sizeof(T)); // separate ConvertUToPtr
                for (uint s = 0; s * (WG * 8) < vec_len; s++)
                {
                    uint4 w0 = vp[s * WG + ti];
                    if (pcs.mode == 1)
                    {
                        uint4 w1 = yv[s * WG + ti];
                        // interleave the two streams elementwise
                        float4 lo0 = f16tof32(w0);
                        float4 hi0 = f16tof32(w0 >> 16);
                        float4 lo1 = f16tof32(w1);
                        float4 hi1 = f16tof32(w1 >> 16);
                        float d[8] = { lo1.x, hi1.x, lo1.y, hi1.y, lo1.z, hi1.z, lo1.w, hi1.w };
                        float e[8] = { lo0.x, hi0.x, lo0.y, hi0.y, lo0.z, hi0.z, lo0.w, hi0.w };
                        [unroll]
                        for (uint k = 0; k < 8; k++)
                        {
                            acc0 += d[k];
                            acc1 += d[k] * ((e[k] - mean_c) * rstd_c);
                        }
                    }
                    else
                        accumulateHalf8(w0, 0, 0.0, 0.0, acc0, acc1);
                }
            }
            // scalar tail of the vector body
            for (uint j = off + vec_len + ti; j < off_end; j += WG)
            {
                float e = castValue<T, float>(xp[j]);
                if (pcs.mode == 0) { acc0 += e; acc1 += e * e; }
                else
                {
                    float d = castValue<T, float>(((T*)pcs.tensor_y)[seg_base + j]);
                    acc0 += d;
                    acc1 += d * ((e - mean_c) * rstd_c);
                }
            }
        }
        else
        {
            for (uint j = off + ti; j < off_end; j += WG)
            {
                float e = castValue<T, float>(xp[j]);
                if (pcs.mode == 0) { acc0 += e; acc1 += e * e; }
                else
                {
                    float d = castValue<T, float>(((T*)pcs.tensor_y)[seg_base + j]);
                    acc0 += d;
                    acc1 += d * ((e - mean_c) * rstd_c);
                }
            }
        }

        v = seg_stop;
        seg++;
    }

    sm0[ti] = acc0;
    sm1[ti] = acc1;
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint s = WG >> 1; s > 0; s >>= 1)
    {
        if (ti < s)
        {
            sm0[ti] += sm0[ti + s];
            sm1[ti] += sm1[ti + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (ti == 0)
    {
        float* part = (float*)pcs.tensor_part;
        part[uint64_t(gid) * 2 + 0] = sm0[0];
        part[uint64_t(gid) * 2 + 1] = sm1[0];
    }
}

// MODE: 0 = fwd (fold -> mean/rstd, saves, running stats), 1 = backward
// (fold -> grad_bias/grad_weight in the parameter dtype T)
public struct BnFinalizePushConstants
{
    public uint64_t tensor_part;  // fp32 [C, n_chunks, 2]
    public uint64_t tensor_save0; // mode 0: save_mean fp32 (C) / mode 1: grad_bias (T)
    public uint64_t tensor_save1; // mode 0: save_rstd fp32 (C) / mode 1: grad_weight (T)
    public uint64_t tensor_rm;    // mode 0: running_mean (T) or 0
    public uint64_t tensor_rv;    // mode 0: running_var (T) or 0
    public uint C;
    public uint n_chunks;
    public uint NHW;
    public float momentum;
    public float eps;
    public float unbias;          // N/(N-1), 0 when N <= 1
    public uint mode;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void bn_finalize_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform BnFinalizePushConstants pcs)
{
    uint c = wg.x * WG + ti;
    if (c >= pcs.C) return;

    float* part = (float*)pcs.tensor_part;
    float s = 0.0;
    float ss = 0.0;
    uint64_t row = uint64_t(c) * pcs.n_chunks * 2;
    for (uint k = 0; k < pcs.n_chunks; k++)
    {
        s += part[row + 2 * k + 0];
        ss += part[row + 2 * k + 1];
    }

    if (pcs.mode == 0)
    {
        float n = float(pcs.NHW);
        float mean = s / n;
        float var = max(ss / n - mean * mean, 0.0);
        ((float*)pcs.tensor_save0)[c] = mean;
        ((float*)pcs.tensor_save1)[c] = rsqrt(var + pcs.eps);
        if (pcs.tensor_rm != 0)
        {
            T* rm = (T*)pcs.tensor_rm;
            rm[c] = castValue<float, T>(castValue<T, float>(rm[c]) * (1.0 - pcs.momentum) + mean * pcs.momentum);
        }
        if (pcs.tensor_rv != 0)
        {
            T* rv = (T*)pcs.tensor_rv;
            rv[c] = castValue<float, T>(castValue<T, float>(rv[c]) * (1.0 - pcs.momentum) + var * pcs.unbias * pcs.momentum);
        }
    }
    else
    {
        ((T*)pcs.tensor_save0)[c] = castValue<float, T>(s);
        ((T*)pcs.tensor_save1)[c] = castValue<float, T>(ss);
    }
}

// MODE 0: fwd apply, out = (x - mean[c]) * rstd[c] * w[c] + b[c]
// MODE 1: backward grad_input (train) =
//             w[c]*rstd[c]*(dy - sum_dy[c]/n - xhat*sum_dy_xhat[c]/n),
//         eval (flags bit 2 clear): dy * w[c] * rstd[c]
//
// One workgroup per (channel, chunk) — the channel params load once per WG
// and the flat index needs no divisions (the (seg, c) block offset is derived
// per segment, as in bn_partials_main). Vector body: fp16 8 elements and
// fp32 4 elements per thread per step via uint4, when every vector group
// stays inside one (seg, c) block (S % width == 0) and the chunk grid is
// aligned (host-guaranteed chunk_size); otherwise the scalar walk.
public struct BnApplyPushConstants
{
    public uint64_t tensor_x;    // mode 0: input / mode 1: input (xhat source)
    public uint64_t tensor_dy;   // mode 1: grad_out; unused (0) in mode 0
    public uint64_t tensor_mean; // fp32 (C)
    public uint64_t tensor_rstd; // fp32 (C)
    public uint64_t tensor_w;    // (C) in the input dtype, or 0
    public uint64_t tensor_b;    // (C) in the input dtype, or 0
    public uint64_t tensor_sdy;  // mode 1 train: sum_dy fp32 (C), else 0
    public uint64_t tensor_sdyx; // mode 1 train: sum_dy_xhat fp32 (C), else 0
    public uint64_t tensor_out;
    public uint S;
    public uint C;
    public uint NHW;
    public uint n_chunks;
    public uint chunk_size;
    public uint flags;           // bit0 weight, bit1 bias, bit2 train
    public float inv_n;          // 1 / (N0 * S)
    public uint mode;            // 0 = fwd apply, 1 = grad_input
}

float applyScalar(float e, float d, uint mode, bool train, float mean, float rstd, float w, float b, float mdy, float mdyx)
{
    if (mode == 0) return (e - mean) * rstd * w + b;
    if (train)
    {
        float xhat = (e - mean) * rstd;
        return w * rstd * (d - mdy - xhat * mdyx);
    }
    return d * w * rstd;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void bn_apply_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform BnApplyPushConstants pcs)
{
    bool has_w = (pcs.flags & 1u) != 0;
    bool has_b = (pcs.flags & 2u) != 0;
    bool train = (pcs.flags & 4u) != 0;

    uint gid = flat_group(wg);
    uint c = gid / pcs.n_chunks;
    uint chunk = gid - c * pcs.n_chunks;
    if (c >= pcs.C) return;

    uint base = chunk * pcs.chunk_size;
    uint stop = min(base + pcs.chunk_size, pcs.NHW);

    float mean = ((float*)pcs.tensor_mean)[c];
    float rstd = ((float*)pcs.tensor_rstd)[c];
    float w = has_w ? castValue<T, float>(((T*)pcs.tensor_w)[c]) : 1.0;
    float b = has_b ? castValue<T, float>(((T*)pcs.tensor_b)[c]) : 0.0;
    float mdy = 0.0;
    float mdyx = 0.0;
    if (pcs.mode == 1 && train)
    {
        mdy = ((float*)pcs.tensor_sdy)[c] * pcs.inv_n;
        mdyx = ((float*)pcs.tensor_sdyx)[c] * pcs.inv_n;
    }

    // vector width in elements: 8 for fp16, 4 for fp32 (one uint4)
    uint vw = (sizeof(T) == 2) ? 8u : 4u;
    bool can_vec = (pcs.S % vw == 0) && (pcs.chunk_size % vw == 0);

    uint seg = base / pcs.S;
    uint v = base;
    while (v < stop)
    {
        uint seg_stop = min(stop, (seg + 1) * pcs.S);
        uint64_t seg_base = (uint64_t(seg) * pcs.C + c) * pcs.S; // element offset of this (seg, c) block
        uint off = v - seg * pcs.S;
        uint off_end = seg_stop - seg * pcs.S;
        T* xp = (T*)pcs.tensor_x + seg_base;
        T* op = (T*)pcs.tensor_out + seg_base;

        if (can_vec)
        {
            uint len = off_end - off;
            uint vec_len = len & ~(WG * vw - 1u);
            if (len >= WG * vw)
            {
                uint4* xv = (uint4*)(pcs.tensor_x + (seg_base + off) * sizeof(T));
                uint4* yv = pcs.mode == 1 ? (uint4*)(pcs.tensor_dy + (seg_base + off) * sizeof(T)) : (uint4*)0;
                uint4* ov = (uint4*)(pcs.tensor_out + (seg_base + off) * sizeof(T));
                for (uint s = 0; s * (WG * vw) < vec_len; s++)
                {
                    uint step = s * WG + ti;
                    uint4 wx = xv[step];
                    if (sizeof(T) == 2)
                    {
                        float r[8];
                        float4 lo = f16tof32(wx);
                        float4 hi = f16tof32(wx >> 16);
                        float e[8] = { lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, lo.w, hi.w };
                        if (pcs.mode == 0)
                        {
                            [unroll]
                            for (uint k = 0; k < 8; k++)
                                r[k] = (e[k] - mean) * rstd * w + b;
                        }
                        else
                        {
                            uint4 wy = yv[step];
                            float4 dlo = f16tof32(wy);
                            float4 dhi = f16tof32(wy >> 16);
                            float d[8] = { dlo.x, dhi.x, dlo.y, dhi.y, dlo.z, dhi.z, dlo.w, dhi.w };
                            if (train)
                            {
                                [unroll]
                                for (uint k = 0; k < 8; k++)
                                {
                                    float xhat = (e[k] - mean) * rstd;
                                    r[k] = w * rstd * (d[k] - mdy - xhat * mdyx);
                                }
                            }
                            else
                            {
                                [unroll]
                                for (uint k = 0; k < 8; k++)
                                    r[k] = d[k] * w * rstd;
                            }
                        }
                        // pack pairs: even element in the low bits, odd in the high
                        uint words[4] = { 0, 0, 0, 0 };
                        [ForceUnroll]
                        for (uint k = 0; k < 8; k++)
                        {
                            T rounded = castValue<float, T>(r[k]);
                            uint bits = f32tof16(castValue<T, float>(rounded));
                            if ((k & 1) == 0) words[k >> 1] = bits;
                            else words[k >> 1] |= bits << 16;
                        }
                        ov[step] = uint4(words[0], words[1], words[2], words[3]);
                    }
                    else
                    {
                        float4 rr;
                        float4 e = asfloat(wx);
                        if (pcs.mode == 0)
                            rr = (e - mean) * rstd * w + b;
                        else
                        {
                            float4 d = asfloat(yv[step]);
                            if (train)
                            {
                                float4 xhat = (e - mean) * rstd;
                                rr = w * rstd * (d - mdy - xhat * mdyx);
                            }
                            else
                                rr = d * w * rstd;
                        }
                        ov[step] = asuint(rr);
                    }
                }
            }
            // scalar tail of the vector body
            for (uint j = off + vec_len + ti; j < off_end; j += WG)
                op[j] = castValue<float, T>(applyScalar(castValue<T, float>(xp[j]),
                    pcs.mode == 1 ? castValue<T, float>(((T*)pcs.tensor_dy)[seg_base + j]) : 0.0,
                    pcs.mode, train, mean, rstd, w, b, mdy, mdyx));
        }
        else
        {
            for (uint j = off + ti; j < off_end; j += WG)
                op[j] = castValue<float, T>(applyScalar(castValue<T, float>(xp[j]),
                    pcs.mode == 1 ? castValue<T, float>(((T*)pcs.tensor_dy)[seg_base + j]) : 0.0,
                    pcs.mode, train, mean, rstd, w, b, mdy, mdyx));
        }

        v = seg_stop;
        seg++;
    }
}
