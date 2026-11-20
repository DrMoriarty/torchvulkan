module arg_reduce_tree;

// Fast argmax/argmin for long reduce dimensions (vocab-size argmax in LLM
// decode: [1, 151936] fp16). The base arg_reduce kernel scans reduce_size
// sequentially with ONE thread (~0.2 us/element -> ~30 ms for the vocab) —
// here the scan is a two-phase tree reduction:
//   phase 0: one workgroup per (line, chunk), chunk = ceil(reduce_size /
//            n_chunks); threads scan the chunk with WG stride, the per-thread
//            (best, idx) pairs combine through groupshared memory;
//   phase 1: one workgroup per line reduces the n_chunks partial (val, idx).
// Semantics match arg_reduce: strict gt/lt (first maximum wins ties), NaN is
// the maximum (aten semantics). Floating T only (comparisons in fp32; fp16
// values compare exactly). WG is fixed at 128 by the host.

__exported import core.dtype;

[[vk::constant_id(0)]] public const uint WG = 0;
[[vk::constant_id(1)]] public const uint OP_TYPE = 0; // 0 = max (argmax), 1 = min (argmin)

public struct ArgReduceTreePushConstants
{
    public uint64_t tensor_in;   // (num_lines, reduce_size) T, contiguous
    public uint64_t tensor_val;  // (num_lines, n_chunks) T — частичные значения (фаза 0)
    public uint64_t tensor_idx;  // (num_lines, n_chunks) int64 — глобальные индексы (фаза 0)
    public uint64_t tensor_out;  // (num_lines) int64 — результат (фаза 1)
    public uint num_lines;
    public uint n_chunks;
    public uint chunk_size;
    public uint reduce_size;
    public uint phase;           // 0 = частичные (val/idx), 1 = финальный индекс
}

groupshared float sm_val[128];
groupshared uint sm_idx[128];

// combine (v, j) into running (b, bi) with aten tie/NaN semantics
void argCombine(inout float b, inout uint bi, float v, uint j)
{
    // NaN-тест через биты: компилятор выбрасывает v != v (fast-math)
    bool vnan = (asuint(v) & 0x7FFFFFFFu) > 0x7F800000u;
    bool bnan = (asuint(b) & 0x7FFFFFFFu) > 0x7F800000u;
    bool take;
    if (vnan && bnan) take = j < bi;
    else if (vnan) take = true;           // NaN — максимум/минимум (aten)
    else if (bnan) take = false;
    else if (v == b) take = j < bi;       // первый экстремум
    else take = (OP_TYPE == 0) ? (v > b) : (v < b);
    if (take) { b = v; bi = j; }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void arg_reduce_tree_main<T : ITorchVulkan>(
    uint ti : SV_GroupIndex,
    uint3 wg : SV_GroupID,
    uniform ArgReduceTreePushConstants pcs)
{
    float best = (OP_TYPE == 0) ? -3.4e38 : 3.4e38;
    uint bidx = 0;

    if (pcs.phase == 0)
    {
        uint gid = wg.x;
        uint line = gid / pcs.n_chunks;
        uint chunk = gid - line * pcs.n_chunks;
        if (line >= pcs.num_lines) return;
        uint start = chunk * pcs.chunk_size;
        uint stop = min(start + pcs.chunk_size, pcs.reduce_size);
        T* xin = (T*)pcs.tensor_in + uint64_t(line) * pcs.reduce_size;
        for (uint j = start + ti; j < stop; j += WG)
        {
            float v = castValue<T, float>(xin[j]);
            argCombine(best, bidx, v, j);
        }
        sm_val[ti] = best;
        sm_idx[ti] = bidx;
    }
    else
    {
        // фаза 1: сводим n_chunks частичных (значение T, индекс int64);
        // без раннего return (барьер ниже обязаны пройти все потоки WG)
        uint line = wg.x;
        if (ti < pcs.n_chunks)
        {
            T* vin = (T*)pcs.tensor_val + uint64_t(line) * pcs.n_chunks;
            int64_t* iin = (int64_t*)pcs.tensor_idx + uint64_t(line) * pcs.n_chunks;
            float v = castValue<T, float>(vin[ti]);
            uint j = uint(iin[ti]);
            argCombine(best, bidx, v, j);
        }
        sm_val[ti] = best;
        sm_idx[ti] = bidx;
    }
    GroupMemoryBarrierWithGroupSync();

    // дерево по 128 парам (WG хоста фиксирован 128)
    [ForceUnroll]
    for (uint s = 64; s > 0; s >>= 1)
    {
        if (ti < s)
            argCombine(sm_val[ti], sm_idx[ti], sm_val[ti + s], sm_idx[ti + s]);
        GroupMemoryBarrierWithGroupSync();
    }

    if (ti == 0)
    {
        if (pcs.phase == 0)
        {
            T* vout = (T*)pcs.tensor_val + uint64_t(wg.x);
            int64_t* iout = (int64_t*)pcs.tensor_idx + uint64_t(wg.x);
            vout[0] = castValue<float, T>(sm_val[0]);
            iout[0] = int64_t(sm_idx[0]);
        }
        else
        {
            int64_t* out = (int64_t*)pcs.tensor_out + uint64_t(wg.x);
            out[0] = int64_t(sm_idx[0]);
        }
    }
}
