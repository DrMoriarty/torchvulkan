"""Weight-only int8/int4/NF4 packing — the canonical (library-side) format implementation.

The layout matches the quant-GEMV kernels (src/shaders/ops/quant_gemv.slang) and the
private matmuls ``torch.ops.torchvulkan.weight_int8_mm`` / ``weight_int4_mm`` /
``weight_nf4_mm``:

- int8: one byte per k value, symmetric per-output-row scale,
  value = (q - 128) * scale, q in [0, 255] (the -128 offset folded into zero);
- int4: two nibbles per byte (low nibble = even k), asymmetric per-block scale+zero
  along K (``group`` k values per group), value = (q - zero) * scale, q in [0, 15];
- nf4 (NormalFloat4, QLoRA — Dettmers et al. 2023): the same packed nibble layout as
  int4, but the nibble is an index into the 16-entry normal-quantile codebook
  (``NF4_LUT``) and the per-block scale is the block absmax: value = NF4_LUT[q] * scale.
  LLM weights are ~normal, so fixed normal quantiles minimize the quantization error
  at the same 4-bit footprint (no zero point — NF4_LUT[7] == 0.0);
- int4/nf4 rows are zero-padded to a multiple of 16 bytes (wide kernel loads; never read).

Moved from experiments/quantize.py (which now re-exports these and keeps the CLI);
torchvulkan/tests/test_llm_inference.py carries a documented value-copy because the
test suite must not depend on experiments/.
"""
import torch

ROW_ALIGN = 16  # int4/nf4 row padding in bytes (uint32-aligned rows x4)

# NormalFloat4 codebook (bitsandbytes nf4): 16 quantiles of N(0, 1) at the
# (i + 0.5) / 16 midpoints of the CDF, normalized by the extreme quantile; strictly
# ascending, index 7 is exactly 0.0. Values are the published bnb constants.
NF4_LUT = (-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
           0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0)


def quantize_int8(w):
    """(N, K) -> (uint8 (N, K), float32 scales (N, 1)): symmetric per-row scale."""
    wf = w.float()
    amax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scales = amax / 127.0
    q = torch.round(wf / scales).clamp_(-127, 127) + 128
    return q.to(torch.uint8).contiguous(), scales.contiguous()


def quantize_int4(w, group=64):
    """(N, K) -> (uint8 (N, row_bytes), float32 scales (N, ngroups), float32 zeros (N, ngroups)).

    Asymmetric per-block quantization along K: ngroups = ceil(K/group), blocks are
    zero-padded (zero nibbles q=zero dequantize to 0); rows are padded to a
    multiple of ROW_ALIGN bytes.
    """
    wf = w.float()
    N, K = wf.shape
    Kpad = (K + group - 1) // group * group
    if Kpad != K:
        wf = torch.nn.functional.pad(wf, (0, Kpad - K))
    ngroups = Kpad // group

    blocks = wf.view(N, ngroups, group)
    wmin = blocks.amin(dim=-1, keepdim=True)
    wmax = blocks.amax(dim=-1, keepdim=True)
    scales = ((wmax - wmin) / 15.0).clamp_min(1e-12)
    zeros = torch.round(-wmin / scales).clamp_(0, 15)
    q = (torch.round(blocks / scales) + zeros).clamp_(0, 15).view(N, Kpad).to(torch.uint8)

    lo, hi = q[:, 0::2], q[:, 1::2]
    packed = lo | (hi << 4)  # low nibble = even k
    row_bytes = packed.shape[1]
    row_pad = (-row_bytes) % ROW_ALIGN
    if row_pad:
        packed = torch.nn.functional.pad(packed, (0, row_pad))
    return (packed.contiguous(), scales.view(N, ngroups).contiguous(),
            zeros.view(N, ngroups).contiguous())


def quantize_nf4(w, group=64):
    """(N, K) -> (uint8 (N, row_bytes), float32 scales (N, ngroups)).

    NF4 (NormalFloat4) weight-only quantization: same packed layout as
    :func:`quantize_int4` (two nibbles per byte, low nibble = even k, blocks of
    ``group`` k values, rows padded to ROW_ALIGN bytes), but the nibble is the
    index of the nearest NF4_LUT entry and the per-block scale is the block
    absmax (value = NF4_LUT[q] * scale). Encoding is O(1) per value: the 16
    midpoints of the sorted codebook partition the value axis into 16
    nearest-neighbour bins, so a searchsorted against the 15 boundaries maps
    w / absmax straight to its code. Padding nibbles are 7 (NF4_LUT[7] == 0).
    """
    wf = w.float()
    N, K = wf.shape
    Kpad = (K + group - 1) // group * group
    if Kpad != K:
        wf = torch.nn.functional.pad(wf, (0, Kpad - K))
    ngroups = Kpad // group

    blocks = wf.view(N, ngroups, group)
    scales = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    v = (blocks / scales).reshape(-1).double()
    lut = torch.tensor(NF4_LUT, dtype=torch.float64)
    mids = (lut[1:] + lut[:-1]) * 0.5  # 15 nearest-neighbour bin boundaries
    q = torch.searchsorted(mids, v).clamp_(0, 15).reshape(N, ngroups, group)
    q = q.view(N, Kpad).to(torch.uint8)

    lo, hi = q[:, 0::2], q[:, 1::2]
    packed = lo | (hi << 4)  # low nibble = even k
    row_bytes = packed.shape[1]
    row_pad = (-row_bytes) % ROW_ALIGN
    if row_pad:
        packed = torch.nn.functional.pad(packed, (0, row_pad))
    return packed.contiguous(), scales.view(N, ngroups).contiguous()


def quantize_weight(w, bits, group=64):
    """int4 -> (packed, scales, zeros); int8 / 'nf4' -> (packed, scales)."""
    if bits == "nf4":
        return quantize_nf4(w, group)
    return quantize_int4(w, group) if bits == 4 else quantize_int8(w)
