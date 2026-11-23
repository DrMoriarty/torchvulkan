"""End-to-end LLM inference test: a tiny randomly-initialised decoder-only transformer
(the parameterisable ``GPT`` from ``examples/llm.py`` — pre-RMSNorm, RoPE, GQA, SwiGLU,
tied embeddings) run through the full inference cycle on both CPU and Vulkan:

  * prefill of the whole prompt, logits compared against the CPU reference;
  * incremental greedy decode with an explicit KV cache (append via ``cat``-free slice
    ``copy_``, per-position RoPE slices), tokens compared bit-for-bit against a CPU
    reference that recomputes the full sequence every step;
  * ``multinomial`` sampling only as a smoke check (RNG streams differ across devices);
  * wall-clock prefill/decode timings and a per-stage breakdown printed to stdout
    (visible with ``pytest -s``).

No real checkpoints are loaded — the model is small enough (~115k params) that the
whole test runs in well under a minute.

Strict mode is enabled *before* torchvulkan is imported so a silent CPU fallback can't
mask backend gaps (repository rule, see AGENTS.md). ``TORCHVULKAN_TEST_DEVICE`` overrides
the Vulkan device index (e.g. ``TORCHVULKAN_TEST_DEVICE=1`` or ``TORCHVULKAN_TEST_DEVICE=vulkan:1``)
for machines where device 0 is not the target discrete GPU.

The quantization part (``test_llm_weight_quant*``) runs the same cycle with every block
linear replaced by a weight-only quant layer driven by the backend's private matmuls
``torch.ops.torchvulkan.weight_int8_mm`` / ``weight_int4_mm`` — a single quant mode for the
whole run, so prefill exercises the M>1 dequant+GEMM path and decode the M==1 quant-GEMV
path of the kernels (operator-level accuracy vs fp64 lives in experiments/test_quant.py).
"""

import importlib.util
import os
import pathlib
import time

os.environ.setdefault("TORCHVULKAN_STRICT", "1")

import pytest
import torch

import torchvulkan

# import examples/llm.py by path (the examples/ directory is not a package)
_LLM_PATH = pathlib.Path(__file__).resolve().parents[1] / "examples" / "llm.py"
_spec = importlib.util.spec_from_file_location("llm_example", _LLM_PATH)
_llm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_llm)
GPT = _llm.GPT

# tiny Qwen3-shaped config: 4 query heads of 16 dims, 2 KV heads (GQA group of 2), 2 layers
CFG = dict(vocab_size=256, block_size=64, n_layer=2, n_embd=64, head_dim=16, kv_groups=2)
PROMPT_LEN = 8
GEN_STEPS = 16
MAX_LEN = PROMPT_LEN + GEN_STEPS
# prefill logits agree to ~3e-7 in fp32; the tolerance leaves two orders of magnitude of slack
LOGIT_RTOL = 1e-4
LOGIT_ATOL = 1e-4
TOTAL_TIME_BUDGET_S = 60.0  # the only perf assertion: a rough ceiling on the whole test


def vulkan_device() -> torch.device:
    """The Vulkan device to test on: the backend default, or ``TORCHVULKAN_TEST_DEVICE``
    (an index like ``1`` or a device string like ``vulkan:1``). Skips if unavailable."""
    if not torchvulkan.is_available():
        pytest.skip("torchvulkan is not available")
    spec = os.environ.get("TORCHVULKAN_TEST_DEVICE", "").strip()
    if spec:
        index = int(spec) if spec.isdigit() else torch.device(spec).index
        if index is None:
            index = 0
    else:
        index = torchvulkan.current_device()
    if index >= torchvulkan.device_count():
        pytest.skip(f"TORCHVULKAN_TEST_DEVICE={spec}: only {torchvulkan.device_count()} device(s)")
    torchvulkan.set_device(index)
    return torch.device("vulkan", index)


class CachedGPT(GPT):
    """``GPT`` with an explicit KV cache: prefill the whole prompt once, then decode one
    token at a time, appending K/V to a pre-allocated cache and attending over it with
    grouped-query 5D matmuls (the pattern proven on this backend in experiments/vulkan_infer.py)."""

    def init_cache(self, batch: int, max_len: int, device: torch.device, dtype: torch.dtype = torch.float32):
        attn = self.blocks[0].attn
        shape = (batch, attn.n_kv_head, max_len, attn.head_dim)
        kc = [torch.empty(shape, dtype=dtype, device=device) for _ in self.blocks]
        vc = [torch.empty(shape, dtype=dtype, device=device) for _ in self.blocks]
        return kc, vc

    def _causal_mask(self, T: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Additive causal mask [1, 1, T, T]: 0 on and below the diagonal, most-negative
        value of ``dtype`` above it (the pattern from experiments/vulkan_infer.py)."""
        m = torch.full((T, T), torch.finfo(dtype).min).triu(1)
        return m.view(1, 1, T, T).to(device=device, dtype=dtype)

    @torch.no_grad()
    def forward_cache(self, idx, cur, kc, vc, mask=None, timings=None):
        """``idx`` is [B, T] with ``cur`` tokens already in the cache; returns logits for all
        T positions ([B, T, vocab]). ``timings`` (optional dict) accumulates wall-clock seconds
        per stage (embedding / qkv / attention / mlp / lm_head), synchronising at every
        boundary — only pass it on a throwaway pass, the syncs dominate small steps."""
        B, T = idx.shape
        dev = idx.device
        sync = torchvulkan.synchronize if dev.type == "vulkan" else (lambda: None)
        t_last = time.perf_counter()

        def tick(stage):
            nonlocal t_last
            if timings is None:
                return
            sync()
            now = time.perf_counter()
            timings[stage] = timings.get(stage, 0.0) + (now - t_last)
            t_last = time.perf_counter()

        cos, sin = self.cos[cur:cur + T], self.sin[cur:cur + T]
        x = self.tok_emb(idx)
        tick("embedding")
        L = cur + T
        for i, block in enumerate(self.blocks):
            attn = block.attn
            h = block.attn_norm(x)
            q = attn.q(h).view(B, T, attn.n_head, attn.head_dim).transpose(1, 2)
            k, v = attn.kv(h).view(B, T, 2, attn.n_kv_head, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k = _llm.apply_rope(q, cos, sin), _llm.apply_rope(k, cos, sin)
            tick("qkv")
            kc[i][:, :, cur:cur + T].copy_(k)
            vc[i][:, :, cur:cur + T].copy_(v)
            K = kc[i].narrow(2, 0, L)
            V = vc[i].narrow(2, 0, L)
            groups = attn.n_head // attn.n_kv_head
            scores = torch.matmul(q.reshape(B, attn.n_kv_head, groups, T, attn.head_dim),
                                  K.reshape(B, attn.n_kv_head, 1, L, attn.head_dim).transpose(-2, -1))
            scores = scores * (attn.head_dim ** -0.5)
            if mask is not None:
                scores = scores + mask
            o = torch.matmul(scores.softmax(-1), V.reshape(B, attn.n_kv_head, 1, L, attn.head_dim))
            o = o.reshape(B, attn.n_head, T, attn.head_dim).transpose(1, 2).reshape(B, T, -1)
            x = x + attn.out(o)
            tick("attention")
            x = x + block.mlp(block.mlp_norm(x))
            tick("mlp")
        x = self.head(self.norm(x))
        tick("lm_head")
        return x

    @torch.no_grad()
    def generate_greedy(self, idx, steps: int, timings=None, dtype=None):
        """Greedy decode of ``steps`` new tokens: prefill (with a causal mask) then one-token
        steps against the cache. Returns (tokens [1, steps], prefill_s, decode_step_times).
        ``dtype`` selects the cache/mask dtype; defaults to the model's parameter dtype."""
        dev = idx.device
        sync = torchvulkan.synchronize if dev.type == "vulkan" else (lambda: None)
        if dtype is None:
            dtype = self.tok_emb.weight.dtype
        kc, vc = self.init_cache(idx.shape[0], idx.shape[1] + steps, dev, dtype)

        mask = self._causal_mask(idx.shape[1], dev, dtype) if idx.shape[1] > 1 else None

        sync()
        t0 = time.perf_counter()
        logits = self.forward_cache(idx, 0, kc, vc, mask=mask, timings=timings)[:, -1]
        sync()
        prefill_s = time.perf_counter() - t0

        tokens = [logits.argmax(-1, keepdim=True)]
        step_times = []
        cur = idx.shape[1]
        for _ in range(steps - 1):
            sync()
            t1 = time.perf_counter()
            logits = self.forward_cache(tokens[-1], cur, kc, vc, timings=timings)[:, -1]
            sync()
            step_times.append(time.perf_counter() - t1)
            cur += 1
            tokens.append(logits.argmax(-1, keepdim=True))
        return torch.cat(tokens, dim=1), prefill_s, step_times


@torch.no_grad()
def greedy_reference(model, idx, steps: int) -> torch.Tensor:
    """CPU reference: no cache, the full sequence is re-run through the plain GPT forward
    every step (a tiny model makes that cheaper than maintaining a second cache path)."""
    seq = idx
    logits = model(seq)[:, -1]
    tokens = [logits.argmax(-1, keepdim=True)]
    for _ in range(steps - 1):
        seq = torch.cat((seq, tokens[-1]), dim=1)
        logits = model(seq)[:, -1]
        tokens.append(logits.argmax(-1, keepdim=True))
    return torch.cat(tokens, dim=1)


def make_models(device: torch.device):
    """Two instances of the same tiny GPT with identical weights: the reference on CPU and
    the model under test on Vulkan (weights shared via load_state_dict, not just the seed)."""
    torch.manual_seed(0)
    reference = GPT(**CFG).to("cpu").eval()
    model = CachedGPT(**CFG)
    model.load_state_dict(reference.state_dict())
    model = model.to(device).eval()
    return reference, model


def make_prompt() -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, CFG["vocab_size"], (1, PROMPT_LEN), generator=g)


def test_llm_full_cycle():
    t_test = time.perf_counter()
    device = vulkan_device()
    print(f"\nvulkan device: {torchvulkan.get_device_name(device)} (index {device.index})")

    reference, model = make_models(device)
    n_params = sum(p.numel() for p in reference.parameters())
    prompt = make_prompt().to(device)
    prompt_cpu = prompt.to("cpu")

    # warmup: the first Vulkan call compiles shaders/pipelines (~0.5 s), so time only after it
    warm_tokens, _, _ = model.generate_greedy(prompt[:, :2], 2)
    assert warm_tokens.shape == (1, 2)

    # --- correctness: prefill logits -----------------------------------------------------------
    logits_ref = reference(prompt_cpu)[:, -1, :]
    prefill_kc, prefill_vc = model.init_cache(1, PROMPT_LEN, device)
    prefill_mask = model._causal_mask(PROMPT_LEN, device)
    logits_vk = model.forward_cache(prompt, 0, prefill_kc, prefill_vc, mask=prefill_mask)[:, -1, :]
    logits_vk = logits_vk.to("cpu")
    max_diff = (logits_vk - logits_ref).abs().max().item()
    print(f"prefill logits: max|vulkan-cpu| = {max_diff:.3e}")
    torch.testing.assert_close(logits_vk, logits_ref, rtol=LOGIT_RTOL, atol=LOGIT_ATOL)

    # --- correctness: greedy decode with KV cache, bit-for-bit vs the CPU reference ------------
    vk_tokens, prefill_s, step_times = model.generate_greedy(prompt, GEN_STEPS)
    ref_tokens = greedy_reference(reference, prompt_cpu, GEN_STEPS)
    print(f"greedy tokens ({GEN_STEPS} steps): "
          f"{'MATCH' if torch.equal(vk_tokens.to('cpu'), ref_tokens) else 'MISMATCH'}")
    print(f"  tokens: {vk_tokens[0].tolist()}")
    torch.testing.assert_close(vk_tokens.to("cpu"), ref_tokens)  # exact: bit-for-bit

    # --- timings (report only) ------------------------------------------------------------------
    timings = {}
    model.generate_greedy(prompt, GEN_STEPS, timings=timings)
    mean_step = sum(step_times) / len(step_times)
    print(f"timings (fp32, after warmup):")
    print(f"  prefill ({PROMPT_LEN} tokens): {prefill_s * 1e3:8.2f} ms  "
          f"({PROMPT_LEN / prefill_s:6.1f} tok/s)")
    print(f"  decode step mean over {len(step_times)} steps: {mean_step * 1e3:8.2f} ms  "
          f"({1 / mean_step:6.1f} tok/s)")
    print(f"  per-stage breakdown of one instrumented generation "
          f"(includes per-stage sync overhead):")
    print(f"    {'stage':<12}{'total ms':>10}{'per decode step (us)':>24}")
    for stage in ("embedding", "qkv", "attention", "mlp", "lm_head"):
        total = timings.get(stage, 0.0)
        # (GEN_STEPS - 1) instrumented decode steps after the instrumented prefill
        print(f"    {stage:<12}{total * 1e3:>10.2f}{total * 1e6 / (GEN_STEPS - 1):>24.1f}")

    # --- smoke: multinomial sampling runs (RNG differs across devices, so values are not checked)
    sampled = model.generate(prompt, 4)
    assert sampled.shape == (1, PROMPT_LEN + 4)
    assert ((sampled[:, :PROMPT_LEN] == prompt).all()), "prompt tokens must be preserved"
    assert ((sampled[:, PROMPT_LEN:] >= 0) & (sampled[:, PROMPT_LEN:] < CFG["vocab_size"])).all()

    elapsed = time.perf_counter() - t_test
    print(f"  total test wall clock: {elapsed:.1f} s")
    assert elapsed < TOTAL_TIME_BUDGET_S, f"test took {elapsed:.1f} s, budget is {TOTAL_TIME_BUDGET_S:.0f} s"


def test_llm_cpu_vs_vulkan_weights_identical():
    """Guard for the weight-sharing setup itself: state_dict transfer must be exact."""
    device = vulkan_device()
    reference, model = make_models(device)
    for (name, ref_p), (_, vk_p) in zip(reference.state_dict().items(), model.state_dict().items()):
        assert name.split(".")[-1] != "cos" and name.split(".")[-1] != "sin"  # non-persistent buffers
        assert torch.equal(ref_p, vk_p.to("cpu")), f"weight mismatch after transfer: {name}"


# =============================================================================================
# Weight-only quantization (int8 / int4 / nf4)
# =============================================================================================
# The same tiny-GPT cycle with every block linear running through the backend's private
# weight-only quant matmuls (torch.ops.torchvulkan.weight_int8_mm / weight_int4_mm /
# weight_nf4_mm; kernels
# in src/api/ops/quant.cpp, shaders in src/shaders/ops/quant_gemv.slang). A single quant
# mode for the whole run: prefill (M > 1) goes through the kernel's dequant+GEMM path,
# decode (M == 1) through the quant-GEMV path.
#
# torchvulkan/tests/ must not depend on experiments/, so the packing functions below are a
# minimal value-copy of experiments/quantize.py — that file (and its docstrings) stays the
# format source of truth:
#   * int8: one byte per k value, symmetric per-output-row scale, value = (q - 128) * scale,
#     q in [0, 255] (the -128 offset folded into zero);
#   * int4: two nibbles per byte (low nibble = even k), asymmetric per-block scale+zero
#     along K (``group`` k values per group), value = (q - zero) * scale, q in [0, 15];
#   * nf4 (NormalFloat4, QLoRA — Dettmers et al. 2023): the same nibble layout as int4,
#     the nibble is an index into the 16-entry normal-quantile codebook NF4_LUT,
#     value = NF4_LUT[q] * scale with the per-block absmax as the scale (no zero point);
#   * int4/nf4 rows are zero-padded to a multiple of 16 bytes (wide kernel loads; never read).

ROW_ALIGN = 16  # experiments/quantize.py:ROW_ALIGN — int4/nf4 row padding in bytes

# NormalFloat4 codebook (value-copy of torchvulkan/quantize.py:NF4_LUT)
NF4_LUT = (-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
           0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0)


def quantize_int8(w):
    """Value-copy of experiments/quantize.py:quantize_int8 (format source of truth):
    (N, K) -> (uint8 (N, K), float32 scales (N, 1)), symmetric per-row scale."""
    wf = w.float()
    amax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scales = amax / 127.0
    q = torch.round(wf / scales).clamp_(-127, 127) + 128
    return q.to(torch.uint8).contiguous(), scales.contiguous()


def quantize_int4(w, group=64):
    """Value-copy of experiments/quantize.py:quantize_int4 (format source of truth):
    (N, K) -> (uint8 (N, row_bytes), float32 scales (N, ngroups), float32 zeros (N, ngroups)),
    asymmetric per-block quantization along K, k-padded to a group multiple and rows
    padded to ROW_ALIGN."""
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
    row_pad = (-packed.shape[1]) % ROW_ALIGN
    if row_pad:
        packed = torch.nn.functional.pad(packed, (0, row_pad))
    return (packed.contiguous(), scales.view(N, ngroups).contiguous(),
            zeros.view(N, ngroups).contiguous())


def quantize_nf4(w, group=64):
    """Value-copy of torchvulkan/quantize.py:quantize_nf4 (format source of truth):
    (N, K) -> (uint8 (N, row_bytes), float32 scales (N, ngroups)), the same packed
    layout as int4 but the nibble is the nearest NF4_LUT index and the per-block
    scale is the block absmax; encoded by searchsorted against the 15 codebook
    midpoints (O(1) nearest-neighbour binning per value)."""
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
    mids = (lut[1:] + lut[:-1]) * 0.5
    q = torch.searchsorted(mids, v).clamp_(0, 15).reshape(N, ngroups, group)
    q = q.view(N, Kpad).to(torch.uint8)

    lo, hi = q[:, 0::2], q[:, 1::2]
    packed = lo | (hi << 4)
    row_pad = (-packed.shape[1]) % ROW_ALIGN
    if row_pad:
        packed = torch.nn.functional.pad(packed, (0, row_pad))
    return packed.contiguous(), scales.view(N, ngroups).contiguous()


class QuantLinear(torch.nn.Module):
    """Drop-in replacement for an ``nn.Linear`` whose forward dispatches to the backend's
    private weight-only quant matmul on Vulkan; on CPU (the reference path) it dequantizes
    and falls back to a plain fp32 matmul — the kernel semantics per
    experiments/test_quant.py:reference().

    Activations come in ``act_dtype`` (fp16 or fp32), accumulation is fp32 inside the
    kernel and the output follows the activation dtype. The packed bytes and fp32
    scales/zeros are kept as *plain attributes* (not buffers/parameters) so
    ``Module.to()`` never re-casts or re-moves them; weights are quantized on the fly
    from the fp32 initialisation of the layer being replaced."""

    def __init__(self, linear: torch.nn.Linear, bits, group: int = 64,
                 device="cpu", act_dtype: torch.dtype = torch.float32):
        super().__init__()
        assert bits in (4, 8, "nf4")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.group = group
        self.act_dtype = act_dtype
        if linear.bias is not None:
            self.bias = torch.nn.Parameter(
                linear.bias.detach().to(device=device, dtype=act_dtype))
        else:
            self.bias = None
        if bits == 8:
            packed, scales = quantize_int8(linear.weight.detach())
            zeros = None
        elif bits == "nf4":
            packed, scales = quantize_nf4(linear.weight.detach(), group)
            zeros = None
        else:
            packed, scales, zeros = quantize_int4(linear.weight.detach(), group)
        self.w_packed = packed.to(device)
        self.qscales = scales.to(device)
        self.qzeros = None if zeros is None else zeros.to(device)

    def dequantized_weight(self) -> torch.Tensor:
        """Kernel-semantics dequantization to fp32 on CPU (the reference path)."""
        wi = self.w_packed.detach().cpu().to(torch.int64)
        sc = self.qscales.detach().cpu()
        if self.bits == 8:
            return (wi - 128).float() * sc
        ngroups = sc.shape[1]
        q = torch.stack((wi & 0xF, (wi >> 4) & 0xF), dim=-1).view(wi.shape[0], -1)
        q = q[:, :ngroups * self.group]
        if self.bits == "nf4":
            lut = torch.tensor(NF4_LUT, dtype=torch.float64)
            deq = (lut[q.reshape(-1)].view(-1, ngroups, self.group)
                   * sc.double().unsqueeze(-1)).view(wi.shape[0], -1)
            return deq[:, :self.in_features].float()
        z = self.qzeros.detach().cpu()
        deq = (q.view(-1, ngroups, self.group) - z.unsqueeze(-1)) * sc.unsqueeze(-1)
        return deq.view(wi.shape[0], -1)[:, :self.in_features]

    def forward(self, x):
        if x.device.type != "vulkan":
            w = self.dequantized_weight().to(x.dtype)
            return torch.nn.functional.linear(x, w, self.bias)
        if self.bits == 8:
            return torch.ops.torchvulkan.weight_int8_mm(x, self.w_packed, self.qscales, self.bias)
        if self.bits == "nf4":
            return torch.ops.torchvulkan.weight_nf4_mm(x, self.w_packed, self.qscales, self.bias)
        return torch.ops.torchvulkan.weight_int4_mm(x, self.w_packed, self.qscales,
                                                    self.qzeros, self.bias)


def quantize_block_linears(model, bits, group=64, device="cpu", act_dtype=torch.float32):
    """Swap every linear inside ``model.blocks`` for a QuantLinear quantized on the fly
    from the fp32 initialisation. Embedding, RMSNorm and the tied lm_head stay in
    floating point (the scheme of experiments/quantize.py:quantize_state_dict)."""
    for block in model.blocks:
        block.attn.q = QuantLinear(block.attn.q, bits, group, device, act_dtype)
        block.attn.kv = QuantLinear(block.attn.kv, bits, group, device, act_dtype)
        block.attn.out = QuantLinear(block.attn.out, bits, group, device, act_dtype)
        block.mlp.gate_up = QuantLinear(block.mlp.gate_up, bits, group, device, act_dtype)
        block.mlp.down = QuantLinear(block.mlp.down, bits, group, device, act_dtype)
    return model


def make_quant_models(device: torch.device, bits, act_dtype: torch.dtype = torch.float32):
    """Three views of the same tiny GPT (identical weights via load_state_dict):
    the unquantized fp32 CPU reference, the quantized model on CPU (dequant reference
    path of QuantLinear) and the Vulkan model under test with the same packed weights."""
    torch.manual_seed(0)
    fp32_ref = GPT(**CFG).to("cpu").eval()
    cpu_quant = CachedGPT(**CFG)
    cpu_quant.load_state_dict(fp32_ref.state_dict())
    quantize_block_linears(cpu_quant, bits, device="cpu")
    cpu_quant.eval()
    vk_quant = CachedGPT(**CFG)
    vk_quant.load_state_dict(fp32_ref.state_dict())
    quantize_block_linears(vk_quant, bits, device=device, act_dtype=act_dtype)
    vk_quant = vk_quant.to(device=device, dtype=act_dtype).eval()
    return fp32_ref, cpu_quant, vk_quant


def time_generate(model, prompt, steps: int = GEN_STEPS):
    """Wall-clock prefill and mean decode step of one greedy generation (uninstrumented)."""
    _, prefill_s, step_times = model.generate_greedy(prompt, steps)
    return prefill_s, sum(step_times) / len(step_times)


# Logit tolerances vs the CPU dequant reference: this is kernel error only (the same packed
# weights on both sides; experiments/test_quant.py measures <= 5e-4 relative at operator
# level), loosened for the fp16-activation variants where activation rounding adds up.
QUANT_TOL = {
    8: dict(rtol=2e-3, atol=2e-3),
    4: dict(rtol=5e-3, atol=5e-3),
    "nf4": dict(rtol=5e-3, atol=5e-3),
}
QUANT_TOL_FP16_ACT = {
    8: dict(rtol=5e-3, atol=5e-3),
    4: dict(rtol=1e-2, atol=1e-2),
    "nf4": dict(rtol=1e-2, atol=1e-2),
}


def _assert_finite(t, what):
    assert torch.isfinite(t).all(), f"{what}: non-finite values"


@pytest.mark.parametrize("bits", [8, 4, "nf4"], ids=["int8", "int4", "nf4"])
def test_llm_weight_quant(bits):
    """End-to-end quantized cycle in fp32 activations: prefill logits against the CPU
    dequant reference (kernel-error tolerance), greedy decode with KV cache against the
    same reference, run-to-run determinism, all values finite; timings vs fp32 printed."""
    t_test = time.perf_counter()
    device = vulkan_device()
    print(f"\n[{bits}] vulkan device: {torchvulkan.get_device_name(device)} (index {device.index})")

    fp32_ref, cpu_quant, vk_quant = make_quant_models(device, bits)
    prompt = make_prompt().to(device)
    prompt_cpu = prompt.to("cpu")

    # warmup: the first Vulkan call compiles shaders/pipelines (~0.5 s), so time only after it
    warm_tokens, _, _ = vk_quant.generate_greedy(prompt[:, :2], 2)
    assert warm_tokens.shape == (1, 2)

    # --- correctness: prefill logits -----------------------------------------------------------
    logits_deq = cpu_quant(prompt_cpu)[:, -1, :]
    logits_fp32 = fp32_ref(prompt_cpu)[:, -1, :]
    prefill_kc, prefill_vc = vk_quant.init_cache(1, PROMPT_LEN, device)
    logits_vk = vk_quant.forward_cache(prompt, 0, prefill_kc, prefill_vc,
                                       mask=vk_quant._causal_mask(PROMPT_LEN, device))[:, -1, :]
    logits_vk = logits_vk.to("cpu")
    _assert_finite(logits_vk, "prefill logits")
    err_deq = (logits_vk - logits_deq).abs().max().item()
    err_fp32 = (logits_vk - logits_fp32).abs().max().item()
    print(f"prefill logits: max|vulkan-quant - cpu-dequant| = {err_deq:.3e} (asserted), "
          f"max|vulkan-quant - fp32-unquant| = {err_fp32:.3e} (quantization noise, reported)")
    torch.testing.assert_close(logits_vk, logits_deq, **QUANT_TOL[bits])

    # --- correctness: greedy decode with KV cache ------------------------------------------------
    vk_tokens, prefill_s, step_times = vk_quant.generate_greedy(prompt, GEN_STEPS)
    _assert_finite(vk_tokens, "decode tokens")
    deq_tokens = greedy_reference(cpu_quant, prompt_cpu, GEN_STEPS)
    fp32_tokens = greedy_reference(fp32_ref, prompt_cpu, GEN_STEPS)
    match_deq = torch.equal(vk_tokens.cpu(), deq_tokens)
    match_fp32 = torch.equal(vk_tokens.cpu(), fp32_tokens)
    print(f"greedy tokens ({GEN_STEPS} steps) vs cpu-dequant reference: "
          f"{'MATCH' if match_deq else 'MISMATCH'}")
    print(f"greedy tokens vs fp32-unquantized reference: "
          f"{'MATCH' if match_fp32 else 'MISMATCH (quantization noise on a near-tie)'}")
    print(f"  tokens: {vk_tokens[0].tolist()}")
    # the exact criterion: the Vulkan quant path implements the dequant semantics
    torch.testing.assert_close(vk_tokens.cpu(), deq_tokens)
    # deterministic: the same inputs must give the same tokens on a second run
    vk_tokens_2, _, _ = vk_quant.generate_greedy(prompt, GEN_STEPS)
    assert torch.equal(vk_tokens, vk_tokens_2), "quantized generation must be deterministic"

    # --- timings (report only): quant M==1 GEMV / M>1 GEMM paths vs the fp32 baseline ------------
    fp32_vk = make_models(device)[1]
    fp32_vk.generate_greedy(prompt[:, :2], 2)  # warmup
    fp32_prefill, fp32_step = time_generate(fp32_vk, prompt)
    mean_step = sum(step_times) / len(step_times)
    print(f"timings (fp32 activations, after warmup; {GEN_STEPS} decode steps):")
    print(f"  {'mode':<6}{'prefill ms':>12}{'prefill tok/s':>16}"
          f"{'decode ms/step':>18}{'decode tok/s':>16}")
    for name, (pf, st) in ((str(bits), (prefill_s, mean_step)), ("fp32", (fp32_prefill, fp32_step))):
        print(f"  {name:<6}{pf * 1e3:>12.2f}{PROMPT_LEN / pf:>16.1f}"
              f"{st * 1e3:>18.2f}{1 / st:>16.1f}")

    elapsed = time.perf_counter() - t_test
    print(f"  total test wall clock: {elapsed:.1f} s")
    assert elapsed < TOTAL_TIME_BUDGET_S, f"test took {elapsed:.1f} s, budget is {TOTAL_TIME_BUDGET_S:.0f} s"


@pytest.mark.parametrize("bits", [8, 4, "nf4"], ids=["int8", "int4", "nf4"])
def test_llm_weight_quant_fp16_activations(bits):
    """fp16 activations exercise the f16 template variants of the quant kernels
    (quant_gemv_w{8,4}/nf4 f16): prefill + a few cached decode steps, logits against the
    fp32 CPU dequant reference with the fp16 activation rounding folded into the tolerance."""
    device = vulkan_device()
    print(f"\n[{bits} fp16-act] vulkan device: {torchvulkan.get_device_name(device)}")
    _, cpu_quant, vk_quant = make_quant_models(device, bits, act_dtype=torch.float16)
    prompt = make_prompt().to(device)

    vk_quant.generate_greedy(prompt[:, :2], 2)  # warmup

    prefill_kc, prefill_vc = vk_quant.init_cache(1, PROMPT_LEN, device, dtype=torch.float16)
    logits_vk = vk_quant.forward_cache(prompt, 0, prefill_kc, prefill_vc,
                                       mask=vk_quant._causal_mask(PROMPT_LEN, device, torch.float16)
                                       )[:, -1, :].to("cpu").float()
    logits_ref = cpu_quant(prompt.cpu())[:, -1, :]
    _assert_finite(logits_vk, "prefill logits (fp16 activations)")
    max_diff = (logits_vk - logits_ref).abs().max().item()
    print(f"prefill logits: max|vulkan-fp16-quant - cpu-dequant-fp32| = {max_diff:.3e}")
    torch.testing.assert_close(logits_vk, logits_ref, **QUANT_TOL_FP16_ACT[bits])

    tokens, _, _ = vk_quant.generate_greedy(prompt, 4)
    assert tokens.shape == (1, 4)  # generate_greedy returns only the generated tokens
    _assert_finite(tokens, "decode tokens (fp16 activations)")
    tokens_2, _, _ = vk_quant.generate_greedy(prompt, 4)
    assert torch.equal(tokens, tokens_2), "fp16-activation generation must be deterministic"
