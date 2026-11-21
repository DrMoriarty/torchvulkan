module reduce_tree;

// Two-phase tree reduction for long reduce dims (sum/mean/prod/amax/amin on
// contiguous fp16/fp32 with few output lines: full-tensor sum, batch_norm
// train statistics, wgrad chains). The base reduce kernel splits the dim into
// per-THREAD chunks: each thread scans its chunk sequentially and warp lanes
// land megabytes apart (no coalescing, ~4 GB/s). Here phase 0 gives each
// (line, chunk) a whole workgroup of WG threads that scan the chunk with WG
// stride — fully coalesced, uint4-wide where alignment allows — and tree-
// reduces through groupshared memory; phase 1 combines the n_chunks <= WG
// partials of each line (one workgroup per line, same tree).
//
// Accumulation is float (fp32) for every op and every input dtype: the host
// routes fp16 sum/prod here directly (the materialised fp32 copy of the base
// path costs more than the reduction) and fp32 tensors as-is. Output and
// partials are fp32; casting to the output dtype is the caller's job.
//
// Semantics match the base reduce ops: sum/prod propagate NaN/Inf through
// +/*. amax/amin need NaN-wins (float ext_max), and the ONLY reliable NaN
// test under fast-math is the bit test — which the compiler folds away on
// any value produced by fp ARITHMETIC (nnan lets it prove "not NaN"). So the
// ext combine, like arg_reduce_tree's argCombine, never computes a new fp
// value: it only ASSIGNS one of its operands (loads/identity keep provenance
// clean, and assigned loads stay clean), making the bit tests survive.
// The per-thread accumulator starts at the op's identity, so chunk interiors
// never need padding; the summation ORDER differs from the sequential base
// kernel (pairwise tree), which moves an fp32 sum by its last ulps — the same
// rounding-order class of difference the CPU's vectorized sums already show.

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG = 0;
[[vk::constant_id(1)]] public const uint OP_TYPE = 0; // 0 sum, 1 amax, 2 amin, 3 prod

public struct ReduceTreePushConstants
{
    public uint64_t tensor_in;   // (num_lines, reduce_size) T, каждая строка непрерывна
    public uint64_t tensor_part; // (num_lines, n_chunks) float — частичные (фаза 0)
    public uint64_t tensor_out;  // (num_lines) float — результат (фаза 1)
    public uint num_lines;
    public uint n_chunks;
    public uint chunk_size;
    public uint reduce_size;
    public uint phase;           // 0 = частичные, 1 = финал
}

groupshared float sm_val[128];

float treeIdentity()
{
    if (OP_TYPE == 0) return 0.0;                   // sum
    if (OP_TYPE == 3) return 1.0;                   // prod
    if (OP_TYPE == 1) return asfloat(0xFF800000u);  // amax: -inf
    return asfloat(0x7F800000u);                    // amin: +inf
}

float treeCombine(float a, float b)
{
    if (OP_TYPE == 0) return a + b;
    if (OP_TYPE == 3) return a * b;
    return a; // unreachable для sum/prod-специализаций (см. treeExtCombine)
}

// NaN-тест по битам: выживает только на значениях, не прошедших fp-арифметику
bool treeIsNaN(float v)
{
    return (asuint(v) & 0x7FFFFFFFu) > 0x7F800000u;
}

// amax/amin: NaN побеждает (семантика float ext_max), победитель ПРИСВАИВАЕТСЯ
// (не вычисляется) — см. комментарий выше про provenance и fast-math
void treeExtCombine(inout float b, float v)
{
    bool vnan = treeIsNaN(v);
    bool bnan = treeIsNaN(b);
    bool take;
    if (vnan) take = true;
    else if (bnan) take = false;
    else take = (OP_TYPE == 1) ? (v > b) : (v < b);
    if (take) b = v;
}

// 16 fp16 из пары uint4: в слове w.k чётный элемент в младших битах, нечётный —
// в старших (паттерн attn_decode.slang)
void combineHalf16(uint4 w, uint4 w2, inout float acc)
{
    float4 lo = f16tof32(w);
    float4 hi = f16tof32(w >> 16);
    float4 lo2 = f16tof32(w2);
    float4 hi2 = f16tof32(w2 >> 16);
    acc = treeCombine(acc, lo.x);
    acc = treeCombine(acc, hi.x);
    acc = treeCombine(acc, lo.y);
    acc = treeCombine(acc, hi.y);
    acc = treeCombine(acc, lo.z);
    acc = treeCombine(acc, hi.z);
    acc = treeCombine(acc, lo.w);
    acc = treeCombine(acc, hi.w);
    acc = treeCombine(acc, lo2.x);
    acc = treeCombine(acc, hi2.x);
    acc = treeCombine(acc, lo2.y);
    acc = treeCombine(acc, hi2.y);
    acc = treeCombine(acc, lo2.z);
    acc = treeCombine(acc, hi2.z);
    acc = treeCombine(acc, lo2.w);
    acc = treeCombine(acc, hi2.w);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void reduce_tree_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform ReduceTreePushConstants pcs)
{
    float best = treeIdentity();
    // flat_group: y-разбиение сетки, когда строк-чанков больше 65535
    uint gid = flat_group(wg);

    if (pcs.phase == 0)
    {
        uint line = gid / pcs.n_chunks;
        uint chunk = gid - line * pcs.n_chunks;
        if (line >= pcs.num_lines) return; // строка одинакова для всего WG — выходим целиком
        uint base = chunk * pcs.chunk_size;
        uint stop = min(base + pcs.chunk_size, pcs.reduce_size);
        uint len = stop - base;
        uint64_t line_addr = pcs.tensor_in + uint64_t(line) * pcs.reduce_size * sizeof(T);

        if (OP_TYPE == 0 || OP_TYPE == 3)
        {
            // sum/prod: векторное тело — 2 uint4 (16 fp16) на поток за шаг,
            // соседние потоки читают соседние 32 Б; хвост — скалярно.
            // Требует 32-байтового выравнивания начала чанка (хост даёт
            // chunk_size, кратный 32 элементам, и 32-байтовой базой буфера).
            // векторизация только для fp16 (распаковка f16tof32); для fp32
            // скалярный проход и так коалесцирован
            bool can_vec = (sizeof(T) == 2) && (len >= WG * 16)
                && (line_addr % 32 == 0) && (base % 16 == 0);
            if (can_vec)
            {
                // базовый uint4-указатель чанка + абсолютный word-индекс
                // (паттерн attn_decode.slang: указатель из u64, без биткаста
                // указателей); шаг итерации = WG*16 элементов, поток ti берёт
                // слова s*WG*2 + 2ti и +1 — свои 16 элементов подряд
                uint vec_len = len & ~(WG * 16 - 1u);
                uint64_t chunk_addr = line_addr + uint64_t(base) * sizeof(T);
                uint4* v = (uint4*)chunk_addr;
                T* xin = (T*)line_addr;
                for (uint s = 0; s * (WG * 16) < vec_len; s++)
                {
                    uint4 w0 = v[s * WG * 2 + ti * 2 + 0];
                    uint4 w1 = v[s * WG * 2 + ti * 2 + 1];
                    combineHalf16(w0, w1, best);
                }
                for (uint j = base + vec_len + ti; j < stop; j += WG)
                    best = treeCombine(best, castValue<T, float>(xin[j]));
            }
            else
            {
                T* xin = (T*)line_addr;
                for (uint j = base + ti; j < stop; j += WG)
                    best = treeCombine(best, castValue<T, float>(xin[j]));
            }
        }
        else
        {
            // amax/amin: скалярный проход (сравнения дёшевы, векторизация
            // усложняет NaN-семантику без выигрыша на практике)
            T* xin = (T*)line_addr;
            for (uint j = base + ti; j < stop; j += WG)
            {
                float e = castValue<T, float>(xin[j]);
                if (OP_TYPE == 1 || OP_TYPE == 2) treeExtCombine(best, e);
                else best = treeCombine(best, e);
            }
        }
        sm_val[ti] = best;
    }
    else
    {
        // фаза 1: n_chunks <= WG частичных строки; сетка фазы 1 <= 16384 WG
        // (1D), но гард всё равно нужен — весь WG выходит целиком, до барьера
        uint line = wg.x;
        if (line >= pcs.num_lines) return;
        if (ti < pcs.n_chunks)
        {
            float* pin = (float*)pcs.tensor_part + uint64_t(line) * pcs.n_chunks;
            float p = pin[ti];
            if (OP_TYPE == 1 || OP_TYPE == 2) treeExtCombine(best, p);
            else best = treeCombine(best, p);
        }
        sm_val[ti] = best;
    }
    GroupMemoryBarrierWithGroupSync();

    // дерево по WG парам (WG — специализируемая константа: 128 в чанченном
    // режиме и фазе 1, pow2(rs) в прямом; без ForceUnroll — специз-константа
    // не разворачивается)
    for (uint s = WG >> 1; s > 0; s >>= 1)
    {
        if (ti < s)
        {
            if (OP_TYPE == 1 || OP_TYPE == 2) treeExtCombine(sm_val[ti], sm_val[ti + s]);
            else sm_val[ti] = treeCombine(sm_val[ti], sm_val[ti + s]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (ti == 0)
    {
        if (pcs.phase == 0)
        {
            // индекс = плоский gid строки-чанка (при y-разбиении wg.x одинок)
            float* pout = (float*)pcs.tensor_part + uint64_t(gid);
            pout[0] = sm_val[0];
        }
        else
        {
            float* out = (float*)pcs.tensor_out + uint64_t(wg.x);
            out[0] = sm_val[0];
        }
    }
}
