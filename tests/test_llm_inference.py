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

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Additive causal mask [1, 1, T, T]: 0 on and below the diagonal, most-negative
        fp32 above it (the pattern from experiments/vulkan_infer.py)."""
        m = torch.full((T, T), torch.finfo(torch.float32).min).triu(1)
        return m.view(1, 1, T, T).to(device)

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
    def generate_greedy(self, idx, steps: int, timings=None):
        """Greedy decode of ``steps`` new tokens: prefill (with a causal mask) then one-token
        steps against the cache. Returns (tokens [1, steps], prefill_s, decode_step_times)."""
        dev = idx.device
        sync = torchvulkan.synchronize if dev.type == "vulkan" else (lambda: None)
        kc, vc = self.init_cache(idx.shape[0], idx.shape[1] + steps, dev)

        mask = self._causal_mask(idx.shape[1], dev) if idx.shape[1] > 1 else None

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
