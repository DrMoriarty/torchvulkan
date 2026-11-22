module rope_train;

// Fused train-path rotate-half RoPE (experiments/lfm2_train_profile.md,
// приоритет 3): replaces the per-tensor eager chain of apply_rotary_pos_emb
// (2 slice + neg + cat + 2 mul + add ≈ 7 kernels fwd + ~8-10 bwd per q and k,
// per attention layer) with ONE kernel for BOTH q and k per phase.
//
//   fwd: (q, k, cos, sin) -> (q_embed, k_embed)
//   bwd: (grad_q_embed, grad_k_embed, cos, sin) -> (grad_q, grad_k)
//
// rotate-half is a signed permutation P (rh[j] = -x[j+hd/2] for j < hd/2,
// +x[j-hd/2] otherwise; skew-symmetric, P^T = -P), so the composite is
// linear:
//   y[j]   = x[j]*cos[j] + P(x)[j]*sin[j]
//   grad_x = g*cos - P(g)*sin     (sin evaluated at the pair index in general;
//                                  for the real cat(s, s) tables sin[pair]==sin[j])
// — the backward is the same kernel with the sign of the pair term flipped,
// no saved activations beyond the (small) cos/sin tables.
//
// Layouts (transformers apply_rotary_pos_emb): q (B, n_q, T, hd), k
// (B, n_kv, T, hd) — strides are passed explicitly because the attention
// transposes (1, 2); cos/sin (B, T, hd) contiguous rows of the full-size
// tables. Row -> position: b = row / (n_heads*T), t = row % T, table row
// = b*T + t. pair(j) = j < hd/2 ? j + hd/2 : j - hd/2.
//
// The eager graph rounds every mul/add to T (fp16 in the training path), so
// the kernel replicates those roundings: term1 = round_T(x*cos),
// term2 = round_T(±P(x)*sin), out = round_T(term1 + term2) — bit-exact
// against the eager chain in fp16 (neg/cat are exact) and bitwise-equal
// arithmetic order in fp32. All intermediate math in fp32. NB: the fp16
// rounding must be an INTEGER bit operation (RNE of the fp32 bits into the
// top half) — both a register half-cast and the f32tof16/f16tof32 bit
// round-trip fold away on Metal (see roundToStorage in binary.slang). The
// backward reads sin at the PAIR index: grad_x[m] = g[m]*cos[m] ±
// g[pair(m)]*sin[pair(m)] with the sign +1 for m < hd/2 (general tables; for
// the real cat(s, s) tables sin[pair] == sin[m]).
//
// One thread per output element, math per row independent — no shared memory.
// NB: variable names avoid MSL reserved words (`half`, `cos`, `sin`).

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint WG = 0;

public struct RopeTrainPushConstants
{
    public uint64_t tensor_a;   // fwd: q, bwd: grad_q_embed (B, n_q, T, hd) T
    public uint64_t tensor_b;   // fwd: k, bwd: grad_k_embed (B, n_kv, T, hd) T
    public uint64_t tensor_cos; // (B, T, hd) T
    public uint64_t tensor_sin; // (B, T, hd) T
    public uint64_t tensor_oa;  // fwd: q_embed, bwd: grad_q (B, n_q, T, hd) T
    public uint64_t tensor_ob;  // fwd: k_embed, bwd: grad_k (B, n_kv, T, hd) T
    public uint hd;
    public uint seq;            // T
    public uint n_q;            // heads of tensor_a
    public uint n_kv;           // heads of tensor_b
    public uint rows_q;         // B * n_q * T (rows of the (…, hd) matrix)
    public uint rows_k;         // B * n_kv * T
    public uint sa_b;           // strides of tensor_a in elements (last = 1)
    public uint sa_h;
    public uint sa_t;
    public uint sb_b;           // strides of tensor_b in elements (last = 1)
    public uint sb_h;
    public uint sb_t;
    public uint tab_b;          // table rows per batch entry (seq), 0 = broadcast
}

// round-to-nearest-even of an fp32 value onto the fp16 grid, in pure integer
// arithmetic — cannot be folded by the compiler, unlike half casts or the
// f32tof16/f16tof32 bit round-trip. The result equals the fp16-rounded value
// as an fp32 number. Full algorithm: normals round the 23-bit mantissa to 10
// bits, fp16 SUBNORMALS (products of the rope operands regularly land below
// 2^-14) round onto the m*2^-24 grid — the uniform 13-bit drop is wrong
// there. NaN/Inf pass through; overflow (|v| > 65504) saturates to Inf.
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
        // normal fp16: RNE of the 23-bit mantissa to 10 bits
        uint r = man + 0x0FFFu + ((man >> 13) & 1u);
        uint e16 = uint(e + 15);
        if ((r & 0x800000u) != 0u) { r = 0u; e16 += 1u; } // carry into exponent
        hb = (e16 << 10) | (r >> 13);
    }
    else
    {
        // fp16 subnormal: value = m * 2^-24, m = RNE((0x800000|man) * 2^(e+1))
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

// fwd: BACKWARD = 0; bwd: BACKWARD = 1 (a plain argument — the literal at the
// call site is constant-folded per specialization)
void rope_train_body<T : ITorchVulkan>(uint3 gi, uniform RopeTrainPushConstants pcs, bool backward)
{
    uint total_q = pcs.rows_q * pcs.hd;
    uint total = total_q + pcs.rows_k * pcs.hd;
    uint idx = gi.x;
    if (idx >= total) return;

    bool is_q = idx < total_q;
    uint row = is_q ? idx / pcs.hd : (idx - total_q) / pcs.hd;
    uint j = idx % pcs.hd;

    uint heads = is_q ? pcs.n_q : pcs.n_kv;
    uint b = row / (heads * pcs.seq);
    uint rem = row % (heads * pcs.seq);
    uint h = rem / pcs.seq;
    uint t = rem % pcs.seq;

    T* src = is_q ? (T*)pcs.tensor_a : (T*)pcs.tensor_b;
    T* dst = is_q ? (T*)pcs.tensor_oa : (T*)pcs.tensor_ob;
    uint sb = is_q ? pcs.sa_b : pcs.sb_b;
    uint sh = is_q ? pcs.sa_h : pcs.sb_h;
    uint st = is_q ? pcs.sa_t : pcs.sb_t;
    uint64_t base = uint64_t(b) * sb + uint64_t(h) * sh + uint64_t(t) * st;

    T* cos_tab = (T*)pcs.tensor_cos;
    T* sin_tab = (T*)pcs.tensor_sin;
    uint64_t tab_off = (uint64_t(b) * pcs.tab_b + t) * pcs.hd + j;

    uint half_dim = pcs.hd / 2;
    uint pair_j = j < half_dim ? j + half_dim : j - half_dim;

    float c = castValue<T, float>(cos_tab[tab_off]);
    float s = castValue<T, float>(sin_tab[tab_off]);

    float v0 = castValue<T, float>(src[base + j]);
    float sgn = (j < half_dim) ? -1.0 : 1.0; // rotate_half sign at j

    // round_T for the eager per-op stores: fp16 through the bit conversions
    // (a register half-cast folds away on Metal), fp32 identity
    float r1f = v0 * c;
    float r2f;
    if (!backward)
    {
        float rh = sgn * castValue<T, float>(src[base + pair_j]);
        r2f = rh * s;                       // sin at j
    }
    else
    {
        float gp = castValue<T, float>(src[base + pair_j]);
        float sp = castValue<T, float>(sin_tab[tab_off - j + pair_j]); // sin at pair(j)
        r2f = -sgn * gp * sp;               // grad: the sign pattern flips
    }
    if (sizeof(T) == 2)
    {
        r1f = round16(r1f);
        r2f = round16(r2f);
    }
    float out_v = r1f + r2f;
    if (sizeof(T) == 2)
        out_v = round16(out_v);
    dst[is_q ? idx : idx - total_q] = castValue<float, T>(out_v);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void rope_train_fwd_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform RopeTrainPushConstants pcs)
{
    rope_train_body<T>(gi, pcs, false);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void rope_train_bwd_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform RopeTrainPushConstants pcs)
{
    rope_train_body<T>(gi, pcs, true);
}
