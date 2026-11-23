module sdpa;

// Fused causal SDPA (training path) replacing the math decomposition
// (_scaled_dot_product_attention_math): bmm + materialized B*H*T*T scores +
// a dozen pointwise/reduction kernels fwd, and the autograd graph over the
// same ops bwd. Three kernels:
//
//   sdpa_fwd_main   QK^T (+ additive mask) + causal-masked softmax + P*V in
//                   one kernel, one workgroup of WG threads per R q-rows
//                   (b, h, row-tile).
//                   Two passes, no score row in groupshared memory: pass 1
//                   streams the thread's scores (j = ti, ti+WG, ...) keeping
//                   only the running max and denominator in registers (online
//                   rescale), pass 2 recomputes the same scores (K is L1/L2-
//                   hot), writes the normalized prob row and accumulates
//                   out[dim = ti] against a WG-wide chunk of weights
//                   exchanged through a small groupshared scratch. The main
//                   range of pass 2 is unrolled x4 (four independent V loads
//                   in flight) — the serial load->FMA chain was the dominant
//                   cost on MoltenVK (~0.35 us per R-row tile -> ~0.14).
//                   Outputs: out (B, Hq, L, hd) and probs P (B, Hkv, qpg, L, S)
//                   fp16/fp32 — P is what the backward kernels consume
//                   (dS = P*(dP - delta), dV = P^T @ dO), so no logsumexp
//                   and no score recompute in backward.
//   sdpa_rowdot_main  delta[r] = sum_d dO[r, d] * out[r, d] (one thread per row)
//   sdpa_ds_main      dS = P * (dP - delta[row]); causal entries zeroed when
//                     the forward was causal (masked entries are already 0 in P)
//
// The backward's four GEMMs (dP = dO @ V^T, dQ = dS @ K, dK = dS^T @ Q,
// dV = P^T @ dO) stay on the regular matmul kernels: hand-rolled fused
// backward kernels (FA2-style, per-thread rows/columns) were tried and lost
// to the GEMMs on the 5300M at these shapes — the GEMM kernels win on
// occupancy and load scheduling even for skinny (128, 64) batches.
//
// Semantics match the math path: with causal the mask is j <= i
// (at::ones(L,S).tril(), top-left aligned, also for L != S); an additive mask
// (fp32, 0 / -inf after the bool conversion on the host) is added to the
// scores before the softmax, with any strides broadcast over (L, S). The
// effective scale folded into the scores is calculate_scale (1/sqrt(head_dim)
// or the user's value). GQA maps q head h to kv group g = h / qpg — the same
// per-head math the math path's repeat_interleave expansion produces. All
// accumulation fp32; a fully masked row gives zeros everywhere (the math
// path's _safe_softmax contract), the denominator guard below.

__exported import core.dtype;
import core.indexing;

static const uint HD_CAP = 128;

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

public struct SdpaFwdPushConstants
{
    public uint64_t tensor_q;     // (B, Hq, L, hd) contiguous
    public uint64_t tensor_k;     // (B, Hkv, S, hd) contiguous
    public uint64_t tensor_v;     // (B, Hkv, S, hd) contiguous
    public uint64_t tensor_out;   // (B, Hq, L, hd)
    public uint64_t tensor_probs; // (B, Hkv, qpg, L, S)
    public uint rows;             // B * Hq * L
    public uint hq;
    public uint hkv;
    public uint qpg;              // Hq / Hkv (1 when there is no GQA)
    public uint l_dim;            // L (<= SMEM_L_CAP)
    public uint s_dim;            // S (<= SMEM_L_CAP)
    public uint hd;               // (<= HD_CAP)
    public uint rows_x;           // row-groups per y slice (grid y-split)
    public uint tph;              // tiles per head: ceil(L / R)
    public float scale;           // effective multiplier on q.k^T
    public uint64_t tensor_mask;  // additive fp32 mask, read via the strides
                                  // below; 0 when has_mask == 0
    public uint mask_stride0;     // elements, per L index (0 for broadcast)
    public uint mask_stride1;     // elements, per S index (0 for broadcast)
    public uint has_mask;
    public uint causal;           // 1 = top-left aligned tril j <= i
}

// lane reduction scratch (max / den / weights), R строк тайла рядами:
// smem_red[r * WG + t]; R-цепочки сканов чередуются, чтобы латентность
// скана не росла с числом строк. Размер под R=8 (максимум инстансов).
groupshared float smem_red[8 * 128];

[shader("compute")]
[numthreads(128, 1, 1)]
void sdpa_fwd_main<T : ITorchVulkan, let WG : int, let R : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform SdpaFwdPushConstants pcs)
{
    // один WG обрабатывает R подряд идущих q-строк одного (b, h): K/V и
    // групповые сканы/барьеры амортизируются на R строк, V в phase B
    // читается один раз на все R строк
    uint tile = wg.y * pcs.rows_x + wg.x;
    uint head = tile / pcs.tph;
    uint i0 = (tile - head * pcs.tph) * R;
    if (i0 >= pcs.l_dim) return;
    uint b_idx = head / pcs.hq;
    uint h = head - b_idx * pcs.hq;
    uint g = h / pcs.qpg;

    T* K = (T*)pcs.tensor_k + (b_idx * pcs.hkv + g) * pcs.s_dim * pcs.hd;
    T* V = (T*)pcs.tensor_v + (b_idx * pcs.hkv + g) * pcs.s_dim * pcs.hd;
    T* prob_base = (T*)pcs.tensor_probs
        + ((b_idx * pcs.hkv + g) * pcs.qpg + (h - g * pcs.qpg)) * pcs.l_dim * pcs.s_dim;
    float* M = (float*)pcs.tensor_mask;

    bool vec = (sizeof(T) == 2) && (pcs.hd % 8 == 0);
    uint wcnt = vec ? pcs.hd / 8 : 0;
    uint4* k4 = (uint4*)pcs.tensor_k;
    uint kbase4 = (b_idx * pcs.hkv + g) * pcs.s_dim * wcnt;

    // q-строки не живут в регистрах (массив hd float на строку резал
    // occupancy по VGPR сильнее groupshared скоров по LDS) — читаются из L1
    uint4* q4 = (uint4*)pcs.tensor_q;
    uint qbase4 = head * pcs.l_dim * wcnt + i0 * wcnt; // строка r: + r * wcnt

    // строка r активна, пока i0 + r < l_dim; при causal верхняя граница j —
    // каузальная, без causal считаются все j < S
    float m_lane[R];
    float den_lane[R];
    uint jmax_hi = (pcs.causal != 0) ? min(i0 + R, pcs.s_dim) : pcs.s_dim;
    [ForceUnroll]
    for (uint r = 0; r < R; r++)
    {
        m_lane[r] = -3.4e38;
        den_lane[r] = 0.0;
    }

    // ---- pass 1: скоры потока (j = ti, ti+WG, ...) для всех R строк,
    // K-слово загружается один раз и скалярно умножается на R q-строк;
    // running (max, den) каждой строки — в регистрах (online-рескейл)
    float acc0[R], acc1[R], acc2[R], acc3[R];
    for (uint j = ti; j < jmax_hi; j += WG)
    {
        [ForceUnroll]
        for (uint r = 0; r < R; r++)
        {
            acc0[r] = 0.0; acc1[r] = 0.0; acc2[r] = 0.0; acc3[r] = 0.0;
        }
        if (vec)
        {
            uint kb = kbase4 + j * wcnt;
            for (uint c = 0; c < wcnt; c++)
            {
                uint4 wk = k4[kb + c];
                float4 kl = f16tof32(wk), kh = f16tof32(wk >> 16);
                [ForceUnroll]
                for (uint r = 0; r < R; r++)
                {
                    uint4 pq = q4[qbase4 + r * wcnt + c];
                    float4 ql = f16tof32(pq), qh = f16tof32(pq >> 16);
                    acc0[r] += ql.x * kl.x + qh.x * kh.x;
                    acc1[r] += ql.y * kl.y + qh.y * kh.y;
                    acc2[r] += ql.z * kl.z + qh.z * kh.z;
                    acc3[r] += ql.w * kl.w + qh.w * kh.w;
                }
            }
        }
        else
        {
            T* krow = K + j * pcs.hd;
            [ForceUnroll]
            for (uint r = 0; r < R; r++)
            {
                T* qrow = (T*)pcs.tensor_q
                    + (head * pcs.l_dim + i0 + r) * pcs.hd;
                for (uint d = 0; d < pcs.hd; d++)
                {
                    float kv = castValue<T, float>(krow[d]);
                    if (d % 4 == 0) acc0[r] += castValue<T, float>(qrow[d]) * kv;
                    if (d % 4 == 1) acc1[r] += castValue<T, float>(qrow[d]) * kv;
                    if (d % 4 == 2) acc2[r] += castValue<T, float>(qrow[d]) * kv;
                    if (d % 4 == 3) acc3[r] += castValue<T, float>(qrow[d]) * kv;
                }
            }
        }
        [ForceUnroll]
        for (uint r = 0; r < R; r++)
        {
            uint im = i0 + r;
            if (im < pcs.l_dim && j < pcs.s_dim && (pcs.causal == 0 || j <= im))
            {
                // additive mask (0 / -inf / пользовательский bias) к скору до
                // softmax — семантика math-пути; -inf даёт w = exp(-inf) = 0
                float mv = (pcs.has_mask != 0)
                    ? M[im * pcs.mask_stride0 + j * pcs.mask_stride1] : 0.0;
                float s = ((acc0[r] + acc1[r]) + (acc2[r] + acc3[r])) * pcs.scale + mv;
                float m_new = max(m_lane[r], s);
                den_lane[r] = den_lane[r] * exp(m_lane[r] - m_new) + exp(s - m_new);
                m_lane[r] = m_new;
            }
        }
    }

    // ---- групповая редукция (max, den) по строкам: R независимых цепочек
    // чередуются в одном скане — латентность скана не умножается на R
    [ForceUnroll]
    for (uint r = 0; r < R; r++) smem_red[r * WG + ti] = m_lane[r];
    GroupMemoryBarrierWithGroupSync();
    float m_glob[R];
    [ForceUnroll]
    for (uint r = 0; r < R; r++) m_glob[r] = -3.4e38;
    for (uint t = 0; t < WG; t++)
    {
        [ForceUnroll]
        for (uint r = 0; r < R; r++) m_glob[r] = max(m_glob[r], smem_red[r * WG + t]);
    }
    // барьер до перезаписи scratch — чужие сканы могли не дочитать
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint r = 0; r < R; r++)
    {
        // exp(m_lane - m_glob) <= 1; пустая строка даёт exp(0)*0 = 0
        smem_red[r * WG + ti] = den_lane[r] * exp(m_lane[r] - m_glob[r]);
    }
    GroupMemoryBarrierWithGroupSync();
    float inv_den[R];
    [ForceUnroll]
    for (uint r = 0; r < R; r++) inv_den[r] = 0.0;
    for (uint t = 0; t < WG; t++)
    {
        [ForceUnroll]
        for (uint r = 0; r < R; r++) inv_den[r] += smem_red[r * WG + t];
    }
    // гвард _safe_softmax: строка, замаскированная целиком, даёт den = 0 —
    // нулевой inv_den записывает нули в P и out (а не NaN от 1/0)
    [ForceUnroll]
    for (uint r = 0; r < R; r++) inv_den[r] = (inv_den[r] > 0.0) ? 1.0 / inv_den[r] : 0.0;
    // и перед весами chunk'а pass 2 — сканы den тоже могли не завершиться
    GroupMemoryBarrierWithGroupSync();

    // ---- pass 2, чанки по WG ключей: пересчёт скоров (K горячий в L1/L2),
    // запись нормированных вероятностей (ноль за каузальной границей) и
    // накопление out[dim = ti] против весов чанка: V читается ОДИН раз на
    // все R строк
    float acc0o[R], acc1o[R];
    [ForceUnroll]
    for (uint r = 0; r < R; r++)
    {
        acc0o[r] = 0.0;
        acc1o[r] = 0.0;
    }
    for (uint c0 = 0; c0 < pcs.s_dim; c0 += WG)
    {
        uint j = c0 + ti;
        float w[R];
        [ForceUnroll]
        for (uint r = 0; r < R; r++) w[r] = 0.0;
        if (j < jmax_hi)
        {
            [ForceUnroll]
            for (uint r = 0; r < R; r++) { acc0[r] = 0.0; acc1[r] = 0.0; acc2[r] = 0.0; acc3[r] = 0.0; }
            if (vec)
            {
                uint kb = kbase4 + j * wcnt;
                for (uint c = 0; c < wcnt; c++)
                {
                    uint4 wk = k4[kb + c];
                    float4 kl = f16tof32(wk), kh = f16tof32(wk >> 16);
                    [ForceUnroll]
                    for (uint r = 0; r < R; r++)
                    {
                        uint4 pq = q4[qbase4 + r * wcnt + c];
                        float4 ql = f16tof32(pq), qh = f16tof32(pq >> 16);
                        acc0[r] += ql.x * kl.x + qh.x * kh.x;
                        acc1[r] += ql.y * kl.y + qh.y * kh.y;
                        acc2[r] += ql.z * kl.z + qh.z * kh.z;
                        acc3[r] += ql.w * kl.w + qh.w * kh.w;
                    }
                }
            }
            else
            {
                T* krow = K + j * pcs.hd;
                [ForceUnroll]
                for (uint r = 0; r < R; r++)
                {
                    T* qrow = (T*)pcs.tensor_q
                        + (head * pcs.l_dim + i0 + r) * pcs.hd;
                    for (uint d = 0; d < pcs.hd; d++)
                    {
                        float kv = castValue<T, float>(krow[d]);
                        if (d % 4 == 0) acc0[r] += castValue<T, float>(qrow[d]) * kv;
                        if (d % 4 == 1) acc1[r] += castValue<T, float>(qrow[d]) * kv;
                        if (d % 4 == 2) acc2[r] += castValue<T, float>(qrow[d]) * kv;
                        if (d % 4 == 3) acc3[r] += castValue<T, float>(qrow[d]) * kv;
                    }
                }
            }
            [ForceUnroll]
            for (uint r = 0; r < R; r++)
            {
                uint im = i0 + r;
                if (im < pcs.l_dim && j < pcs.s_dim && (pcs.causal == 0 || j <= im))
                {
                    float mv = (pcs.has_mask != 0)
                        ? M[im * pcs.mask_stride0 + j * pcs.mask_stride1] : 0.0;
                    float s = ((acc0[r] + acc1[r]) + (acc2[r] + acc3[r])) * pcs.scale + mv;
                    w[r] = exp(s - m_glob[r]);
                }
            }
        }
        [ForceUnroll]
        for (uint r = 0; r < R; r++)
        {
            if (i0 + r < pcs.l_dim && j < pcs.s_dim)
            {
                T* pr = prob_base + (i0 + r) * pcs.s_dim;
                pr[j] = castValue<float, T>(w[r] * inv_den[r]);
            }
            smem_red[r * WG + ti] = w[r];
        }
        GroupMemoryBarrierWithGroupSync();

        // thread ti владеет выходным измерением ti; V[j][ti] одна на R строк.
        // Основной диапазон jj < jstop_0 свободен от пер-строчных гвардов
        // (jstop растёт с r — строка 0 самая короткая), чтобы загрузки V
        // конвейеризовались; хвост [jstop_0, jstop_max) — с гвардами.
        // Основной диапазон раскручен ×4: четыре независимых загрузки V в
        // полёте скрывают латентность L1 (серьяная цепочка load->FMA на
        // MoltenVK не софт-пайплайнится компилятором)
        if (ti < pcs.hd)
        {
            // видимая граница строки при causal (без causal — вся S); строка 0
            // тайла — самая короткая
            uint jstop_0 = ((pcs.causal != 0) ? min(i0 + 1, pcs.s_dim) : pcs.s_dim) - c0;
            // последняя активная строка тайла: тайл может пересекать границу l_dim
            uint i_last = min(i0 + R, pcs.l_dim) - 1;
            uint jstop_max = ((pcs.causal != 0) ? min(i_last + 1, pcs.s_dim) : pcs.s_dim) - c0;
            uint chunk = min(pcs.s_dim - c0, (uint)WG);
            uint main_stop = min(jstop_0, chunk);
            uint main4 = main_stop & ~3u;
            uint jj = 0;
            for (; jj < main4; jj += 4)
            {
                float v0 = castValue<T, float>(V[(c0 + jj) * pcs.hd + ti]);
                float v1 = castValue<T, float>(V[(c0 + jj + 1) * pcs.hd + ti]);
                float v2 = castValue<T, float>(V[(c0 + jj + 2) * pcs.hd + ti]);
                float v3 = castValue<T, float>(V[(c0 + jj + 3) * pcs.hd + ti]);
                [ForceUnroll]
                for (uint r = 0; r < R; r++)
                {
                    acc0o[r] += smem_red[r * WG + jj] * v0
                        + smem_red[r * WG + jj + 2] * v2;
                    acc1o[r] += smem_red[r * WG + jj + 1] * v1
                        + smem_red[r * WG + jj + 3] * v3;
                }
            }
            for (; jj + 1 < main_stop; jj += 2)
            {
                float v0 = castValue<T, float>(V[(c0 + jj) * pcs.hd + ti]);
                float v1 = castValue<T, float>(V[(c0 + jj + 1) * pcs.hd + ti]);
                [ForceUnroll]
                for (uint r = 0; r < R; r++)
                {
                    acc0o[r] += smem_red[r * WG + jj] * v0;
                    acc1o[r] += smem_red[r * WG + jj + 1] * v1;
                }
            }
            if (main_stop & 1) // нечётный остаток основного диапазона
            {
                uint jj = main_stop - 1;
                float v0 = castValue<T, float>(V[(c0 + jj) * pcs.hd + ti]);
                [ForceUnroll]
                for (uint r = 0; r < R; r++) acc0o[r] += smem_red[r * WG + jj] * v0;
            }
            for (uint jj = main_stop; jj < min(jstop_max, chunk); jj++)
            {
                float v0 = castValue<T, float>(V[(c0 + jj) * pcs.hd + ti]);
                [ForceUnroll]
                for (uint r = 0; r < R; r++)
                {
                    uint jstop = (i0 + r < pcs.l_dim)
                        ? (((pcs.causal != 0) ? min(i0 + r + 1, pcs.s_dim) : pcs.s_dim) - c0) : 0;
                    if (jj < jstop) acc0o[r] += smem_red[r * WG + jj] * v0;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync(); // smem_red переиспользуется следующим чанком
    }

    if (ti < pcs.hd)
    {
        [ForceUnroll]
        for (uint r = 0; r < R; r++)
        {
            if (i0 + r < pcs.l_dim)
            {
                T* out_row = (T*)pcs.tensor_out
                    + (head * pcs.l_dim + i0 + r) * pcs.hd;
                out_row[ti] = castValue<float, T>((acc0o[r] + acc1o[r]) * inv_den[r]);
            }
        }
    }
}

// delta: delta[r] = sum_d dO[r, d] * out[r, d], one thread per row
// ---------------------------------------------------------------------------

public struct SdpaRowDotPushConstants
{
    public uint64_t tensor_a; // (rows, hd)
    public uint64_t tensor_b; // (rows, hd)
    public uint64_t tensor_out; // (rows) float
    public uint rows;
    public uint hd;
    public uint blocks_x; // row-groups per y slice (grid y-split)
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sdpa_rowdot_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform SdpaRowDotPushConstants pcs)
{
    uint row = (wg.y * pcs.blocks_x + wg.x) * 128 + ti;
    if (row >= pcs.rows) return;

    float acc = 0;
    if ((sizeof(T) == 2) && (pcs.hd % 8 == 0))
    {
        uint wcnt = pcs.hd / 8;
        uint4* a4 = (uint4*)pcs.tensor_a;
        uint4* b4 = (uint4*)pcs.tensor_b;
        uint base4 = row * wcnt;
        for (uint c = 0; c < wcnt; c++)
        {
            uint4 wa = a4[base4 + c];
            uint4 wb = b4[base4 + c];
            float4 la = f16tof32(wa), ha = f16tof32(wa >> 16);
            float4 lb = f16tof32(wb), hb = f16tof32(wb >> 16);
            acc += la.x * lb.x + ha.x * hb.x + la.y * lb.y + ha.y * hb.y
                 + la.z * lb.z + ha.z * hb.z + la.w * lb.w + ha.w * hb.w;
        }
    }
    else
    {
        T* arow = (T*)pcs.tensor_a + uint64_t(row) * pcs.hd;
        T* brow = (T*)pcs.tensor_b + uint64_t(row) * pcs.hd;
        for (uint d = 0; d < pcs.hd; d++)
            acc += castValue<T, float>(arow[d]) * castValue<T, float>(brow[d]);
    }
    float* out = (float*)pcs.tensor_out;
    out[row] = acc;
}

// ---------------------------------------------------------------------------
// dS = scale * P * (dP - delta[row]), with the causal entries zeroed when the
// forward ran causal (the mask itself needs no handling here: masked entries
// have P == 0 exactly — exp(-inf) — so their dS is 0 through the product).
// The forward folded the scale into the scores, so the chain back to q/k is
// ds_ij/dq_i = scale * k_j — one extra multiply here covers both dQ and dK
// (dV reads P directly and needs no scale).
// dP comes in the same dtype as P: the fp16 GEMM accumulates in fp32 and
// rounds only its output, so dP = dO @ V^T fp16 keeps the backward on fast
// fp16 GEMMs with no fp32 materialization of dO/V; the dS product itself is
// computed in fp32 here and stored fp16.
// ---------------------------------------------------------------------------

public struct SdpaDsPushConstants
{
    public uint64_t tensor_probs; // (B, Hkv, qpg, L, S) T
    public uint64_t tensor_dp;    // same layout, T
    public uint64_t tensor_delta; // (B * Hq * L) float, rows of probs
    public uint64_t tensor_out;   // dS, same layout as probs, T
    public uint numel;
    public uint s_dim;
    public uint l_dim;
    public uint blocks_x;         // element-blocks per y slice (grid y-split)
    public float scale;           // effective multiplier on q.k^T
    public uint causal;           // 1 = zero dS past the diagonal (j > i)
}

[shader("compute")]
[numthreads(128, 1, 1)]
void sdpa_ds_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform SdpaDsPushConstants pcs)
{
    // one element block of 8 per thread in the vector branch, one element in
    // the scalar one; blockIdx layout identical for both
    bool vec = (sizeof(T) == 2) && (pcs.s_dim % 8 == 0);
    uint per_thread = vec ? 8 : 1;
    uint blk = (wg.y * pcs.blocks_x + wg.x) * 128 + ti;
    uint64_t first = uint64_t(blk) * per_thread;
    if (first >= pcs.numel) return;

    if (vec)
    {
        uint row = uint(first / pcs.s_dim);
        uint j = uint(first - uint64_t(row) * pcs.s_dim);
        uint i = row % pcs.l_dim;
        uint4* p4 = (uint4*)pcs.tensor_probs;
        uint4* d4 = (uint4*)pcs.tensor_dp;
        uint4* o4 = (uint4*)pcs.tensor_out;
        float* delta = (float*)pcs.tensor_delta;
        uint word = uint(blk);
        uint4 wp = p4[word];
        uint4 wd = d4[word];
        float4 lp = f16tof32(wp), hp = f16tof32(wp >> 16);
        // dP в том же T, что и probs: те же пары lo/hi по словам
        float4 ld = f16tof32(wd), dd = f16tof32(wd >> 16);
        float dl = delta[row];
        bool c0m = pcs.causal != 0;
        float p0 = (!c0m || j + 0 <= i) ? lp.x : 0.0;
        float p1 = (!c0m || j + 1 <= i) ? hp.x : 0.0;
        float p2 = (!c0m || j + 2 <= i) ? lp.y : 0.0;
        float p3 = (!c0m || j + 3 <= i) ? hp.y : 0.0;
        float p4v = (!c0m || j + 4 <= i) ? lp.z : 0.0;
        float p5 = (!c0m || j + 5 <= i) ? hp.z : 0.0;
        float p6 = (!c0m || j + 6 <= i) ? lp.w : 0.0;
        float p7 = (!c0m || j + 7 <= i) ? hp.w : 0.0;
        // dP распаковывается так же, как probs: в слове чётный элемент в
        // младших битах (ld), нечётный — в старших (dd): j+0 -> ld.x,
        // j+1 -> dd.x, j+2 -> ld.y, ...
        float r0 = p0 * (ld.x - dl) * pcs.scale;
        float r1 = p1 * (dd.x - dl) * pcs.scale;
        float r2 = p2 * (ld.y - dl) * pcs.scale;
        float r3 = p3 * (dd.y - dl) * pcs.scale;
        float r4 = p4v * (ld.z - dl) * pcs.scale;
        float r5 = p5 * (dd.z - dl) * pcs.scale;
        float r6 = p6 * (ld.w - dl) * pcs.scale;
        float r7 = p7 * (dd.w - dl) * pcs.scale;
        uint words[4];
        words[0] = f32tof16(r0) | (f32tof16(r1) << 16);
        words[1] = f32tof16(r2) | (f32tof16(r3) << 16);
        words[2] = f32tof16(r4) | (f32tof16(r5) << 16);
        words[3] = f32tof16(r6) | (f32tof16(r7) << 16);
        o4[word] = uint4(words[0], words[1], words[2], words[3]);
    }
    else
    {
        uint idx = uint(first);
        uint row = idx / pcs.s_dim;
        uint j = idx - row * pcs.s_dim;
        uint i = row % pcs.l_dim;
        T* p = (T*)pcs.tensor_probs;
        T* dp = (T*)pcs.tensor_dp;
        T* out = (T*)pcs.tensor_out;
        float* delta = (float*)pcs.tensor_delta;
        float pv = (pcs.causal == 0 || j <= i) ? castValue<T, float>(p[idx]) : 0.0;
        float dsv = pv * (castValue<T, float>(dp[idx]) - delta[row]) * pcs.scale;
        out[idx] = castValue<float, T>(dsv);
    }
}
