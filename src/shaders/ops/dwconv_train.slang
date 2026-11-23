module dwconv_train;

// Fused train-path depthwise causal conv1d for the LFM2 short-conv layers
// (experiments/lfm2_train_profile.md, приоритет 4). Replaces the per-layer
// eager chain of Lfm2ShortConv prefill
//
//   BCx = in_proj(h).transpose(-1, -2)          [B, 3C, T] copy
//   B, C, x = BCx.chunk(3, dim=-2)
//   m = B * x                                   [B, C, T]
//   h = F.conv1d(m, w[:, None], bias, padding=k-1, groups=C)[:, :, :T]
//   y = (C * h).transpose(-1, -2).contiguous()  [B, T, C] copy
//
// with ONE kernel over the [B, T, C] layout — the in_proj output is consumed
// in place (no transposed copies, no chunk, no B*x / C* muls):
//
//   fwd: proj (B, T, 3C) contiguous, w (C, k), bias (C)?
//        -> conv_cache (B, T, C) pre-activation conv output,
//           out       (B, T, C) = C_chunk * act(conv)
//   bwd: grad_out (B, T, C) + saved proj/w/cache
//        -> grad_proj (B, T, 3C) (grads of the gate/mix/x chunks in place),
//           grad_w via dwconv_wgrad_main partials + the host aten sum
//
// Causal window: out[t] = sum_j w[j] * m[t + j - (k - 1)], zero-padded left,
// sliced to T — exactly F.conv1d(padding=k-1, groups=C)[:, :, :T].
//
// Rounding: the eager graph stores every mul/add result in T (fp16 in the
// train path), so the kernels replicate those roundings with the integer RNE
// (round16 below — a register half-cast or the f32tof16/f16tof32 round-trip
// folds away on Metal, see roundToStorage in binary.slang / rope_train.slang):
// m = round_T(B*x), h = round_T(conv + bias), out = round_T(Cc * act(h));
// all window sums in fp32. conv is cached fp16 — grad_C = round_T(g * h16)
// reads it back, grad_m replicates the eager conv-backward output rounding.
// The optional activation (0 = none, 1 = silu; LFM2 trains with none) is
// applied to the conv output before the C-mul, its backward in fp32.
//
// One thread per output element (fwd, bwd) or per (batch, channel) row
// (wgrad, k weight accumulators + one bias accumulator in fp32), no shared
// memory. NB: variable names avoid MSL reserved words (`half`).

__exported import core.dtype;

// round-to-nearest-even of an fp32 value onto the fp16 grid, in pure integer
// arithmetic — cannot be folded by the compiler (see rope_train.slang for the
// full algorithm note; fp16 subnormals round onto the m*2^-24 grid).
float round16(float v)
{
    uint b = asuint(v);
    uint sgn = b & 0x80000000u;
    uint exp32 = (b >> 23) & 0xFFu;
    uint man = b & 0x7FFFFFu;
    if (exp32 == 0xFFu || (exp32 == 0u && man == 0u))
        return v;                       // NaN/Inf/±0
    int e = int(exp32) - 127;
    uint hb;
    if (e > 15)
    {
        hb = 0x7C00u;                   // overflow → fp16 Inf
    }
    else if (e >= -14)
    {
        uint r = man + 0x0FFFu + ((man >> 13) & 1u);
        uint e16 = uint(e + 15);
        if ((r & 0x800000u) != 0u) { r = 0u; e16 += 1u; } // carry into exponent
        hb = (e16 << 10) | (r >> 13);
    }
    else
    {
        int shift = -e - 1;             // right shift of the 24-bit significand
        if (shift > 24)
            hb = 0u;
        else
        {
            uint m = 0x800000u | man;
            uint keep = m >> shift;
            uint rem = m & ((1u << shift) - 1u);
            uint halfs = 1u << (shift - 1);
            if (rem > halfs || (rem == halfs && (keep & 1u) != 0u))
                keep += 1u;
            hb = keep;                  // 1024 == 0x400 — уже биты мин. нормали fp16
        }
    }
    uint e16 = hb >> 10;
    uint m16 = hb & 0x3FFu;
    if (e16 == 0u)
        return (sgn != 0u ? -1.0 : 1.0) * float(m16) * asfloat((127u - 24u) << 23);
    return asfloat(sgn | ((e16 + 112u) << 23) | (m16 << 13));
}

// round_T: fp16 through the integer RNE, fp32 identity
float roundT<T : ITorchVulkan>(float v)
{
    if (sizeof(T) == 2)
        return round16(v);
    return v;
}

float act_apply(float v, uint act)
{
    if (act == 1u)
        return v / (1.0 + exp(-v));     // silu
    return v;
}

// act'(v); for act == 0 the derivative of the identity is 1
float act_grad(float v, uint act)
{
    if (act == 1u)
    {
        float s = 1.0 / (1.0 + exp(-v));
        return s * (1.0 + v * (1.0 - s));
    }
    return 1.0;
}

public struct DwConvFwdPushConstants
{
    public uint64_t tensor_proj;  // (B, T, 3C) T contiguous: [gate B | mix C | x]
    public uint64_t tensor_w;     // (C, k) T
    public uint64_t tensor_bias;  // (C,) T or 0
    public uint64_t tensor_cache; // (B, T, C) T — conv output pre-activation
    public uint64_t tensor_out;   // (B, T, C) T
    public uint numel;            // B * T * C
    public uint seq;              // T
    public uint chan;             // C
    public uint kern;             // k, 1..4
    public uint act;              // 0 none, 1 silu
}

// fwd, one thread per (b, t, c):
//   m[s]  = round_T(B[s] * x[s])
//   h[t]  = round_T(bias[c] + sum_j w[c,j] * m[t + j - (k-1)])   (s >= 0)
//   out[t]= round_T(Cc[t] * act(h[t])), cache[t] = h[t]
[shader("compute")]
[numthreads(WG, 1, 1)]
void dwconv_fwd_main<T : ITorchVulkan, let WG : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform DwConvFwdPushConstants pcs)
{
    if (gi.x >= pcs.numel) return;
    uint c = gi.x % pcs.chan;
    uint bt = gi.x / pcs.chan;
    uint b = bt / pcs.seq;
    uint t = bt % pcs.seq;
    uint pc3 = pcs.chan * 3u;

    T* proj = (T*)pcs.tensor_proj;
    T* w = (T*)pcs.tensor_w;
    uint64_t rowb = uint64_t(b) * pcs.seq * pc3;   // (b, 0, 0) in proj

    float acc = pcs.tensor_bias != 0u
        ? castValue<T, float>(((T*)pcs.tensor_bias)[c])
        : 0.0;
    for (uint j = 0; j < pcs.kern; j++)
    {
        int s = int(t) + int(j) - int(pcs.kern) + 1;
        if (s >= 0)
        {
            uint64_t off = rowb + uint64_t(s) * pc3 + c;
            // m = round_T(B * x) — the eager mul's fp16 store
            float m = roundT<T>(
                castValue<T, float>(proj[off])
                * castValue<T, float>(proj[off + 2u * pcs.chan]));
            acc += castValue<T, float>(w[uint64_t(c) * pcs.kern + j]) * m;
        }
    }
    // h = round_T(conv + bias) — the eager conv output's store
    float h16 = roundT<T>(acc);

    T* cache = (T*)pcs.tensor_cache;
    T* out = (T*)pcs.tensor_out;
    float a16 = roundT<T>(act_apply(h16, pcs.act));
    float cc = castValue<T, float>(proj[rowb + uint64_t(t) * pc3 + pcs.chan + c]);
    cache[gi.x] = castValue<float, T>(h16);
    out[gi.x] = castValue<float, T>(roundT<T>(cc * a16));
}

public struct DwConvBwdPushConstants
{
    public uint64_t tensor_g;     // grad_out (B, T, C) T
    public uint64_t tensor_proj;  // (B, T, 3C) T
    public uint64_t tensor_w;     // (C, k) T
    public uint64_t tensor_cache; // (B, T, C) T — conv output pre-activation
    public uint64_t tensor_dproj; // (B, T, 3C) T — grads of the three chunks
    public uint numel;            // B * T * C
    public uint seq;              // T
    public uint chan;             // C
    public uint kern;             // k, 1..4
    public uint act;              // 0 none, 1 silu
}

// bwd, one thread per (b, t, c):
//   gc[s]  = round_T(g[s] * Cc[s]) * act'(h[s])     (grad wrt the conv output)
//   gm[t]  = sum_j w[c,j] * gc[t + (k-1) - j]       (transposed conv, s >= t)
//   grad_B = round_T(gm * x), grad_x = round_T(gm * B)
//   grad_C = round_T(g * h)                          (h from the cache)
[shader("compute")]
[numthreads(WG, 1, 1)]
void dwconv_bwd_main<T : ITorchVulkan, let WG : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform DwConvBwdPushConstants pcs)
{
    if (gi.x >= pcs.numel) return;
    uint c = gi.x % pcs.chan;
    uint bt = gi.x / pcs.chan;
    uint b = bt / pcs.seq;
    uint t = bt % pcs.seq;
    uint pc3 = pcs.chan * 3u;

    T* g = (T*)pcs.tensor_g;
    T* proj = (T*)pcs.tensor_proj;
    T* w = (T*)pcs.tensor_w;
    T* cache = (T*)pcs.tensor_cache;
    T* dp = (T*)pcs.tensor_dproj;
    uint64_t rowb = uint64_t(b) * pcs.seq * pc3;
    uint64_t rowg = uint64_t(b) * pcs.seq * pcs.chan;

    float gm = 0.0;
    for (uint j = 0; j < pcs.kern; j++)
    {
        uint tp = t + (pcs.kern - 1u) - j;
        if (tp < pcs.seq)
        {
            float gc = roundT<T>(
                castValue<T, float>(g[rowg + uint64_t(tp) * pcs.chan + c])
                * castValue<T, float>(proj[rowb + uint64_t(tp) * pc3 + pcs.chan + c]));
            gc *= act_grad(castValue<T, float>(cache[rowg + uint64_t(tp) * pcs.chan + c]), pcs.act);
            gm += castValue<T, float>(w[uint64_t(c) * pcs.kern + j]) * gc;
        }
    }
    uint64_t here = rowb + uint64_t(t) * pc3 + c;
    float b32 = castValue<T, float>(proj[here]);
    float x32 = castValue<T, float>(proj[here + 2u * pcs.chan]);
    float g16 = roundT<T>(gm);
    dp[here] = castValue<float, T>(roundT<T>(g16 * x32));                 // grad gate B
    dp[here + 2u * pcs.chan] = castValue<float, T>(roundT<T>(g16 * b32)); // grad x
    dp[here + pcs.chan] = castValue<float, T>(roundT<T>(                  // grad mix C
        castValue<T, float>(g[gi.x]) * castValue<T, float>(cache[gi.x])));
}

public struct DwConvWgradPushConstants
{
    public uint64_t tensor_g;     // grad_out (B, T, C) T
    public uint64_t tensor_proj;  // (B, T, 3C) T
    public uint64_t tensor_cache; // (B, T, C) T — conv output pre-activation
    public uint64_t tensor_part;  // (B, C, k+1) fp32 partials, slot k = grad_bias
    public uint numel_bc;         // B * C
    public uint seq;              // T
    public uint chan;             // C
    public uint kern;             // k, 1..4
    public uint act;              // 0 none, 1 silu
}

// grad wrt w / bias partials, one thread per (b, c): k weight accumulators +
// one bias accumulator over t in fp32, partial [B, C, k+1] folded over b by
// the host aten sum.
[shader("compute")]
[numthreads(WG, 1, 1)]
void dwconv_wgrad_main<T : ITorchVulkan, let WG : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform DwConvWgradPushConstants pcs)
{
    if (gi.x >= pcs.numel_bc) return;
    uint c = gi.x % pcs.chan;
    uint b = gi.x / pcs.chan;
    uint pc3 = pcs.chan * 3u;

    T* g = (T*)pcs.tensor_g;
    T* proj = (T*)pcs.tensor_proj;
    T* cache = (T*)pcs.tensor_cache;
    float* part = (float*)pcs.tensor_part;
    uint64_t rowg = uint64_t(b) * pcs.seq * pcs.chan + c;
    uint64_t rowb = uint64_t(b) * pcs.seq * pc3 + c;

    float accw[4];
    [ForceUnroll]
    for (uint j = 0; j < 4; j++)
        accw[j] = 0.0;
    float accb = 0.0;

    for (uint t = 0; t < pcs.seq; t++)
    {
        float h = castValue<T, float>(cache[rowg + uint64_t(t) * pcs.chan]);
        float gc = roundT<T>(
            castValue<T, float>(g[rowg + uint64_t(t) * pcs.chan])
            * castValue<T, float>(proj[rowb + uint64_t(t) * pc3 + pcs.chan]));
        gc *= act_grad(h, pcs.act);
        accb += gc;
        for (uint j = 0; j < pcs.kern; j++)
        {
            int s = int(t) + int(j) - int(pcs.kern) + 1;
            if (s >= 0)
            {
                uint64_t off = rowb + uint64_t(s) * pc3;
                float m = roundT<T>(
                    castValue<T, float>(proj[off])
                    * castValue<T, float>(proj[off + 2u * pcs.chan]));
                accw[j] += gc * m;
            }
        }
    }
    float* mine = part + uint64_t(gi.x) * (pcs.kern + 1u);
    for (uint j = 0; j < pcs.kern; j++)
        mine[j] = accw[j];
    mine[pcs.kern] = accb;
}
