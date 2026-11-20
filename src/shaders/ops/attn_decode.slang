module attn_decode;

// Fused decode attention (T == 1) with GQA: QK^T + softmax + P·V in one
// kernel, replacing the ~9-kernel aten composition (bmm, amax/sub/exp/sum/div
// softmax chain, bmm) whose per-kernel dispatch+barrier cost dominates small
// batch decode on MoltenVK.
//
// Mapping: one workgroup of WG threads per output row (kv_head g, q_head j in
// group); requires hd <= WG (one output dim per thread in phase B).
//   Phase A: thread ti walks positions t = ti, ti+WG, ...; q is hoisted into
//     registers once, the hd-dot runs as four independent FMA chains (latency
//     hiding), the score goes to groupshared scratch (fp32).
//   Softmax: row max and denominator combined through smem_red (every thread
//     combines redundantly — 2x128 smem reads), then threads overwrite
//     smem_scores with p_t = exp(s_t - m)/den.
//   Phase B: thread ti owns output dim ti: acc = sum_t p_t * V[g, t, ti];
//     four independent accumulator chains over t, consecutive threads read
//     consecutive V elements (coalesced).
//
// The current position (t == l_len - 1) may be read from k_cur/v_cur
// (rows (n_kv, hd), the just-produced roped k and raw v) instead of the
// caches, so the caller does not need cache write kernels at all.
//
// All math in fp32; exp arguments are <= 0, so no overflow. Requires
// L <= SMEM_L_CAP, hd <= 128 and q contiguous [rows, hd] with
// rows = n_kv * q_per_kv, K/V caches contiguous [n_kv, max_len, hd]
// (kv_stride = max_len * hd).

__exported import core.dtype;

// WG: compile-time workgroup size (128 via config.py constants)
static const uint SMEM_L_CAP = 4096;
static const uint HD_CAP = 128;

public struct AttnDecodePushConstants
{
    public uint64_t tensor_q;
    public uint64_t tensor_k;
    public uint64_t tensor_v;
    public uint64_t tensor_out;
    public uint64_t tensor_k_cur; // (n_kv, hd) roped k of the current position, 0 = absent
    public uint64_t tensor_v_cur; // (n_kv, hd) v of the current position, 0 = absent
    public uint kv_stride;  // elements between kv heads (max_len * hd)
    public uint hd;
    public uint l_len;      // cur + 1 valid cache positions
    public uint rows;       // n_kv * q_per_kv
    public uint q_per_kv;   // 2 for Qwen3 GQA
    public uint rms_flag;   // 1: q/k читаются RAW из qk-блока (rows + n_kv строк) и
                            // нормируются/вращаются прямо в кернеле (фуз rms_rope)
    public uint64_t tensor_qkn_w; // (rows + n_kv, hd) QK-norm веса; scale вмножен в q-строки
    public uint64_t tensor_cos;   // (hd,) строка полной таблицы C = cat(c, c)
    public uint64_t tensor_sin;   // (hd,) строка полной таблицы S = cat(-s, s)
    public float rms_eps;
}

groupshared float smem_scores[SMEM_L_CAP];
groupshared float smem_red[128];

[shader("compute")]
[numthreads(WG, 1, 1)]
void attn_decode_main<T : ITorchVulkan, let WG : int>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform AttnDecodePushConstants pcs)
{
    uint r = wg.x;
    if (r >= pcs.rows) return;
    uint g = r / pcs.q_per_kv;

    T* qk = (T*)pcs.tensor_q; // rms_flag: RAW qk-блок (rows + n_kv строк: 0..rows-1 q, rows.. k)
    T* K = (T*)pcs.tensor_k + uint64_t(g) * pcs.kv_stride;
    T* V = (T*)pcs.tensor_v + uint64_t(g) * pcs.kv_stride;
    T* out = (T*)pcs.tensor_out + uint64_t(r) * pcs.hd;
    T* k_cur = (T*)pcs.tensor_k_cur + uint64_t(g) * pcs.hd;
    T* v_cur = (T*)pcs.tensor_v_cur + uint64_t(g) * pcs.hd;
    bool has_cur = pcs.tensor_k_cur != 0 && pcs.tensor_v_cur != 0;
    uint l_last = pcs.l_len - 1;
    uint k_row = pcs.rows + g; // строка RAW-k этой kv-группы в qk-блоке

    T* cos_tab = (T*)pcs.tensor_cos;
    T* sin_tab = (T*)pcs.tensor_sin;
    T* wtab = (T*)pcs.tensor_qkn_w;

    // позиция t: текущая — из k_cur/v_cur (не записана в кэш), прочие — кэш;
    // условие инлайнится в точках доступа (Slang не даёт вложенных функций)

    // ---- QK-norm + RoPE фуз (rms_flag): rrms строки q/k — избыточно в каждом
    // потоке (строка 256 Б, L1); scale уже вмножен в q-строки веса
    float rrms_k = 1.0;
    if (pcs.rms_flag == 1)
    {
        T* krow = qk + uint64_t(k_row) * pcs.hd;
        float ss = 0.0;
        for (uint d = 0; d < pcs.hd; d++)
        {
            float kv = castValue<T, float>(krow[d]);
            ss += kv * kv;
        }
        rrms_k = rsqrt(ss / float(pcs.hd) + pcs.rms_eps);
    }

    // персистентность кэша: roped k текущей позиции пишется прямо здесь, чтобы
    // следующий шаг декода прочитал её из кэша (потоки обеих строк (g, j=0,1)
    // пишут одни и те же значения — бенignный дубликат)
    if (has_cur && ti < pcs.hd)
    {
        if (pcs.rms_flag == 1)
        {
            // k_cur — RAW k (строка k_row); roping на лету, в кэш — roped
            uint half_dim = pcs.hd / 2;
            uint pair = ti < half_dim ? ti + half_dim : ti - half_dim;
            T* krow = qk + uint64_t(k_row) * pcs.hd;
            float xn_t = castValue<T, float>(krow[ti]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + ti]);
            float xn_p = castValue<T, float>(krow[pair]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + pair]);
            float roped = xn_t * castValue<T, float>(cos_tab[ti]) + xn_p * castValue<T, float>(sin_tab[ti]);
            K[uint64_t(l_last) * pcs.hd + ti] = castValue<float, T>(roped);
        }
        else
        {
            // k_cur/v_cur уже смещены на строку g (см. выше) — двойное смещение
            // здесь записывало бы в кэш строку 2g (а для g >= n_kv/2 — за пределами)
            K[uint64_t(l_last) * pcs.hd + ti] = k_cur[ti];
        }
        V[uint64_t(l_last) * pcs.hd + ti] = v_cur[ti];
    }

    // q в регистрах: одна загрузка на поток; при rms_flag — нормировка и
    // rotate-half на лету (избыточно в каждом потоке, строка 256 Б)
    float qreg[HD_CAP];
    if (pcs.rms_flag == 1)
    {
        T* qrow = qk + uint64_t(r) * pcs.hd;
        float ss = 0.0;
        for (uint d = 0; d < pcs.hd; d++)
        {
            float qv = castValue<T, float>(qrow[d]);
            ss += qv * qv;
        }
        float rrms_q = rsqrt(ss / float(pcs.hd) + pcs.rms_eps);
        uint half_dim = pcs.hd / 2;
        [ForceUnroll]
        for (uint d = 0; d < HD_CAP; d++)
        {
            if (d >= pcs.hd) { qreg[d] = 0.0; continue; }
            uint pair = d < half_dim ? d + half_dim : d - half_dim;
            float xn_d = castValue<T, float>(qrow[d]) * rrms_q * castValue<T, float>(wtab[uint64_t(r) * pcs.hd + d]);
            float xn_p = castValue<T, float>(qrow[pair]) * rrms_q * castValue<T, float>(wtab[uint64_t(r) * pcs.hd + pair]);
            qreg[d] = xn_d * castValue<T, float>(cos_tab[d]) + xn_p * castValue<T, float>(sin_tab[d]);
        }
    }
    else if (pcs.hd == HD_CAP)
    {
        // векторная загрузка q (fp16): 16 чтений uint4 вместо 128 скалярных;
        // uint4 = 8 halfs (пары low/high в каждом слове), f16tof32 — распаковка.
        // Ветка инстанцируется только для F16 (см. config.py).
        uint4* q4 = (uint4*)pcs.tensor_q; // отдельный ConvertUToPtr, без биткаста указателей
        uint base4 = uint64_t(r) * pcs.hd / 8;
        [ForceUnroll]
        for (uint c = 0; c < HD_CAP / 8; c++)
        {
            uint4 w = q4[base4 + c];
            float4 lo = f16tof32(w);
            float4 hi = f16tof32(w >> 16);
            // в слове w.k: чётный dim в младших битах, нечётный — в старших
            qreg[c * 8 + 0] = lo.x; qreg[c * 8 + 1] = hi.x;
            qreg[c * 8 + 2] = lo.y; qreg[c * 8 + 3] = hi.y;
            qreg[c * 8 + 4] = lo.z; qreg[c * 8 + 5] = hi.z;
            qreg[c * 8 + 6] = lo.w; qreg[c * 8 + 7] = hi.w;
        }
    }
    else
    {
        T* qrow = (T*)pcs.tensor_q + uint64_t(r) * pcs.hd;
        [ForceUnroll]
        for (uint d = 0; d < HD_CAP; d++)
            qreg[d] = d < pcs.hd ? castValue<T, float>(qrow[d]) : 0.0;
    }

    // phase A: scores q[r]·K[g, t] для позиций потока, fp32,
    // четыре независимых FMA-цепочки против latency
    float m_lane = -3.4e38;
    // векторное view K-кэша для горячей ветки (hd == HD_CAP, fp16):
    // uint4-загрузки вместо скалярных — латентность памяти главный лимит
    // phase A (16 WG на весь кернел, TLP нет)
    uint4* K4 = (uint4*)pcs.tensor_k;
    for (uint t = ti; t < pcs.l_len; t += WG)
    {
        bool is_cur = has_cur && t == l_last;
        if (pcs.rms_flag == 0 && !is_cur && pcs.hd == HD_CAP && pcs.kv_stride % 8 == 0)
        {
            // горячая ветка (fp16): строка K = 16 uint4 (8 halfs каждый) вместо
            // 128 скалярных загрузок; латентность памяти — главный лимит phase A
            // (16 WG на весь кернел, TLP нет), 4 независимые цепочки по 4 uint4
            uint base4 = uint64_t(g) * pcs.kv_stride / 8 + t * (HD_CAP / 8);
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            [ForceUnroll]
            for (uint c = 0; c < HD_CAP / 32; c++)
            {
                uint4 w0 = K4[base4 + c * 4 + 0];
                uint4 w1 = K4[base4 + c * 4 + 1];
                uint4 w2 = K4[base4 + c * 4 + 2];
                uint4 w3 = K4[base4 + c * 4 + 3];
                float4 l0 = f16tof32(w0), h0 = f16tof32(w0 >> 16);
                float4 l1 = f16tof32(w1), h1 = f16tof32(w1 >> 16);
                float4 l2 = f16tof32(w2), h2 = f16tof32(w2 >> 16);
                float4 l3 = f16tof32(w3), h3 = f16tof32(w3 >> 16);
                // в слове w.k: чётный dim в младших битах, нечётный — в старших
                uint e = c * 32;
                a0 += qreg[e + 0] * l0.x + qreg[e + 1] * h0.x + qreg[e + 2] * l0.y + qreg[e + 3] * h0.y
                    + qreg[e + 4] * l0.z + qreg[e + 5] * h0.z + qreg[e + 6] * l0.w + qreg[e + 7] * h0.w;
                a1 += qreg[e + 8] * l1.x + qreg[e + 9] * h1.x + qreg[e + 10] * l1.y + qreg[e + 11] * h1.y
                    + qreg[e + 12] * l1.z + qreg[e + 13] * h1.z + qreg[e + 14] * l1.w + qreg[e + 15] * h1.w;
                a2 += qreg[e + 16] * l2.x + qreg[e + 17] * h2.x + qreg[e + 18] * l2.y + qreg[e + 19] * h2.y
                    + qreg[e + 20] * l2.z + qreg[e + 21] * h2.z + qreg[e + 22] * l2.w + qreg[e + 23] * h2.w;
                a3 += qreg[e + 24] * l3.x + qreg[e + 25] * h3.x + qreg[e + 26] * l3.y + qreg[e + 27] * h3.y
                    + qreg[e + 28] * l3.z + qreg[e + 29] * h3.z + qreg[e + 30] * l3.w + qreg[e + 31] * h3.w;
            }
            float acc = (a0 + a1) + (a2 + a3);
            smem_scores[t] = acc;
            m_lane = max(m_lane, acc);
            continue;
        }
        T* kp = is_cur ? k_cur : K + uint64_t(t) * pcs.hd;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        uint d = 0;
        if (is_cur && pcs.rms_flag == 1)
        {
            // roped k текущей позиции на лету (та же математика, что и запись в кэш)
            T* krow = qk + uint64_t(k_row) * pcs.hd;
            uint half_dim = pcs.hd / 2;
            for (; d + 3 < pcs.hd; d += 4)
            {
                float kv0, kv1, kv2, kv3;
                uint p0 = d < half_dim ? d + half_dim : d - half_dim;
                uint p1 = d + 1 < half_dim ? d + 1 + half_dim : d + 1 - half_dim;
                uint p2 = d + 2 < half_dim ? d + 2 + half_dim : d + 2 - half_dim;
                uint p3 = d + 3 < half_dim ? d + 3 + half_dim : d + 3 - half_dim;
                kv0 = castValue<T, float>(krow[d]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + d]) * castValue<T, float>(cos_tab[d])
                    + castValue<T, float>(krow[p0]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + p0]) * castValue<T, float>(sin_tab[d]);
                kv1 = castValue<T, float>(krow[d + 1]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + d + 1]) * castValue<T, float>(cos_tab[d + 1])
                    + castValue<T, float>(krow[p1]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + p1]) * castValue<T, float>(sin_tab[d + 1]);
                kv2 = castValue<T, float>(krow[d + 2]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + d + 2]) * castValue<T, float>(cos_tab[d + 2])
                    + castValue<T, float>(krow[p2]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + p2]) * castValue<T, float>(sin_tab[d + 2]);
                kv3 = castValue<T, float>(krow[d + 3]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + d + 3]) * castValue<T, float>(cos_tab[d + 3])
                    + castValue<T, float>(krow[p3]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + p3]) * castValue<T, float>(sin_tab[d + 3]);
                a0 += qreg[d] * kv0;
                a1 += qreg[d + 1] * kv1;
                a2 += qreg[d + 2] * kv2;
                a3 += qreg[d + 3] * kv3;
            }
        }
        for (; d + 3 < pcs.hd; d += 4)
        {
            a0 += qreg[d]     * castValue<T, float>(kp[d]);
            a1 += qreg[d + 1] * castValue<T, float>(kp[d + 1]);
            a2 += qreg[d + 2] * castValue<T, float>(kp[d + 2]);
            a3 += qreg[d + 3] * castValue<T, float>(kp[d + 3]);
        }
        float acc = (a0 + a1) + (a2 + a3);
        for (; d < pcs.hd; d++)
        {
            float kv;
            if (is_cur && pcs.rms_flag == 1)
            {
                T* krow = qk + uint64_t(k_row) * pcs.hd;
                uint half_dim = pcs.hd / 2;
                uint pair = d < half_dim ? d + half_dim : d - half_dim;
                kv = castValue<T, float>(krow[d]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + d]) * castValue<T, float>(cos_tab[d])
                   + castValue<T, float>(krow[pair]) * rrms_k * castValue<T, float>(wtab[uint64_t(k_row) * pcs.hd + pair]) * castValue<T, float>(sin_tab[d]);
            }
            else
                kv = castValue<T, float>(kp[d]);
            acc += qreg[d] * kv;
        }
        smem_scores[t] = acc;
        m_lane = max(m_lane, acc);
    }
    smem_red[ti] = m_lane;
    GroupMemoryBarrierWithGroupSync();

    // row max по поточечным максимумам (каждый поток комбинирует избыточно)
    float m_glob = -3.4e38;
    [ForceUnroll]
    for (uint i = 0; i < WG; i++) m_glob = max(m_glob, smem_red[i]);

    // знаменатель, затем p_t заменяет s_t в общем скратче
    float den = 0;
    for (uint t = ti; t < pcs.l_len; t += WG)
        den += exp(smem_scores[t] - m_glob);
    smem_red[ti] = den;
    GroupMemoryBarrierWithGroupSync();
    float den_tot = 0;
    [ForceUnroll]
    for (uint i = 0; i < WG; i++) den_tot += smem_red[i];
    float inv_den = 1.0 / den_tot;

    for (uint t = ti; t < pcs.l_len; t += WG)
        smem_scores[t] = exp(smem_scores[t] - m_glob) * inv_den;
    GroupMemoryBarrierWithGroupSync();

    // phase B: поток владеет выходным измерением ti, идёт по всем позициям,
    // четыре независимых accumulator-цепочки
    float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    uint t = 0;
    for (; t + 3 < pcs.l_len; t += 4)
    {
        T* vp0 = (has_cur && t == l_last) ? v_cur : V + uint64_t(t) * pcs.hd;
        T* vp1 = (has_cur && t + 1 == l_last) ? v_cur : V + uint64_t(t + 1) * pcs.hd;
        T* vp2 = (has_cur && t + 2 == l_last) ? v_cur : V + uint64_t(t + 2) * pcs.hd;
        T* vp3 = (has_cur && t + 3 == l_last) ? v_cur : V + uint64_t(t + 3) * pcs.hd;
        float v0 = castValue<T, float>(vp0[ti]);
        float v1 = castValue<T, float>(vp1[ti]);
        float v2 = castValue<T, float>(vp2[ti]);
        float v3 = castValue<T, float>(vp3[ti]);
        b0 += smem_scores[t] * v0;
        b1 += smem_scores[t + 1] * v1;
        b2 += smem_scores[t + 2] * v2;
        b3 += smem_scores[t + 3] * v3;
    }
    float acc = (b0 + b1) + (b2 + b3);
    for (; t < pcs.l_len; t++)
    {
        T* vp = (has_cur && t == l_last) ? v_cur : V + uint64_t(t) * pcs.hd;
        acc += smem_scores[t] * castValue<T, float>(vp[ti]);
    }
    out[ti] = castValue<float, T>(acc);
}
