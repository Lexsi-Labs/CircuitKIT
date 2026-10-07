# Tiny Aya (Cohere2)

CircuitKIT ships a TransformerLens **compatibility port** that teaches
TransformerLens 3.8 how to load Cohere's tiny-aya checkpoints
(`model_type="cohere2"`) so the discovery pipeline can run on them. Upstream
TransformerLens 3.8 does not know about `cohere2`; the port adds it
additively, without forking or pinning a different TL version.

!!! note "Now a multi-architecture seam"
    Tiny-aya was the first model this port supported, and the seam it
    established (`circuitkit.backends._tl_compat`) has since grown to also
    cover **Command R7B** (same `cohere2` architecture as tiny-aya),
    **Aya Expanse 8B** and **32B** (`cohere1`), and **SmolLM3-3B** (a new
    Llama-family port) — all discovery/evaluation/intervention-capable
    through the same seam, except Aya Expanse 32B, which is registered,
    parity-validated (truncated depth), and validated for pruning/
    quantization but not discovery/evaluation/weight-steering — those are
    blocked by a confirmed transformer-lens multi-GPU bug, not left undone
    for time (see [Experimental Models](experimental-models.md) for what
    was and wasn't validated and why).

!!! warning "Experimental & gated"
    tiny-aya is an **experimental** discovery target. The checkpoints are
    **gated** — you need an `HF_TOKEN` with access to the `CohereLabs/tiny-aya-*`
    repositories — and results should only be trusted once the real-weight
    **parity gate** passes on your machine (see [Tests](#tests)).

---

## Supported checkpoints

The port registers five tiny-aya repositories as known TransformerLens models:

- `CohereLabs/tiny-aya-base`
- `CohereLabs/tiny-aya-global`
- `CohereLabs/tiny-aya-earth`
- `CohereLabs/tiny-aya-fire`
- `CohereLabs/tiny-aya-water`

The `*-GGUF` variants are **not** registered: GGUF is llama.cpp's quantized
single-file format, which `AutoModelForCausalLM.from_pretrained` (what
TransformerLens uses internally) cannot load.

---

## Architecture at a glance

| Property | Value |
|---|---|
| Layers | 36 |
| `d_model` | 2048 |
| Query / KV heads | 16 / 4 (GQA) |
| `d_head` | 128 |
| `d_mlp` | 11008 (gated SiLU) |
| Vocab | 262144 |
| Positional encoding | GPT-J–style interleaved RoPE (`rope_theta=50000`, full-head) |
| Attention | per-layer **sliding-window (4096)** vs **full attention**; full-attention layers use **NoPE** |
| Block topology | **parallel** attention + MLP off one `input_layernorm` |
| Normalization | LayerNorm (`eps=1e-5`) |
| `logit_scale` | 1.0 (fold is a no-op for tiny-aya) |
| Embeddings | tied (`W_E` == `W_U`) |
| Tokenizer | `CohereTokenizer` (BOS id 2, `add_bos_token=true`, `add_eos_token=false`) |

---

## The `_tl_compat` seam

All of the port lives in `circuitkit.backends._tl_compat`:

- **`_tl_compat/__init__.py`** — an idempotent `apply_patches()` guarded to
  TransformerLens **3.8** (with an `importlib.metadata` fallback for installs
  where `transformer_lens.__version__` is empty).
- **`_tl_compat/cohere.py`** — the port itself: registers the repo IDs, wraps
  `convert_hf_model_config` and `get_pretrained_state_dict`, and monkeypatches
  `AbstractAttention.apply_rotary` for NoPE.

The patch is applied **at import time** from `circuitkit/backends/__init__.py`:

```python
from . import _tl_compat
_tl_compat.apply_patches()
```

Every discovery backend imports from `circuitkit.backends` before touching
TransformerLens, so the patch is always live before any
`HookedTransformer.from_pretrained()` call.

### Config mapping (`convert_hf_model_config`)

Each mapping below is traceable to HuggingFace's `transformers.models.cohere2`
source, not assumption:

| HF / spec | TL `HookedTransformerConfig` |
|---|---|
| `rope_gptj` (interleaved pairs) | `rotary_adjacent_pairs=True` |
| `rotary_pct=1.0` | `rotary_dim = d_head` (full head rotated) |
| `rope_theta=50000` | `rotary_base=50000` |
| `sliding_window=4096` + `layer_types` | `use_local_attn=True`, `window_size=4096`, `attn_types` (`full_attention`→`global`, else `local`) |
| parallel block, single `input_layernorm` | `parallel_attn_mlp=True` |
| `num_key_value_heads` | `n_key_value_heads` (GQA) |
| LayerNorm | `normalization_type="LN"` |

### Weight conversion (`get_pretrained_state_dict`)

- **GQA**: K/V are stored in TL's compact `_W_K` / `_W_V` slots, sliced by
  `n_key_value_heads`; TL's `GroupedQueryAttention` ungroups them at runtime.
- **Parallel block**: Cohere2's single `input_layernorm` feeds both branches,
  so it is duplicated into TL's `ln1` **and** `ln2` (tied) — the trick TL
  already uses for GPT-J.
- **Tied embeddings**: HF's own `tie_weights()` already made `lm_head.weight ==
  embed_tokens.weight`, so `W_E` / `W_U` are read from their own tensors.
- **`logit_scale`**: HF applies `logits = lm_head(hidden) * logit_scale`; since
  `lm_head` has no bias this is a pure linear fold into `W_U` (a no-op at
  `logit_scale=1.0`).

### NoPE on full-attention layers

Cohere2's `Cohere2Attention.forward` only rotates q/k when
`self.sliding_window is not None` — the periodic **full-attention** layers get
**no positional embedding at all**. The port reproduces this by patching
`AbstractAttention.apply_rotary` to return its input unchanged for cohere2
`global` layers, while `local` layers rotate normally.

---

## Running discovery

```bash
export HF_TOKEN=hf_...   # account with access to CohereLabs/tiny-aya-*
```

```python
import circuitkit as ck

model = ck.load_model("CohereLabs/tiny-aya-base")   # loads via the TL port
circuit = ck.discover(model, task="greater_than", algorithm="eap-ig")
```

### Task caveat — single-token answers

The single-answer-position discovery metrics (EAP/EAP-IG and the IOI /
greater-than / SVA logit-difference metrics) assume each answer is **one
token**. On Cohere's 262k-token vocabulary a string that is a single token on
GPT-2's 50k BPE can split. CircuitKIT guards this:

- **`greater_than`** builds its own single-token operand pool against the
  loaded tokenizer, so it is always valid — the recommended smoke task.
- **`ioi`** fails fast (or warns) if the tokenizer splits its name pool; if that
  fires, prefer `greater_than` or a custom single-token task.

See the [tokenization guard](#tests) for details.

---

## Tests

Offline unit tests (no gated weights, no network, no GPU) run in ordinary CI:

- `tests/backends/tl_compat/test_cohere.py` — registration, config/weight
  conversion, NoPE patch, tiny random-init end-to-end.
- `tests/backends/tl_compat/test_cohere_gqa.py` — runtime GQA / parallel-block
  gate on a tiny random-init model.
- `tests/tasks/test_cohere_tokenization.py` — the single-token answer guard.

Real-weight gates are **opt-in** (they download a 3.35B gated checkpoint):

```bash
CIRCUITKIT_RUN_TINY_AYA=1 HF_TOKEN=... \
    python -m pytest tests/backends/tl_compat/test_cohere_parity.py \
                     tests/regression/test_cohere_discovery.py -v
```

- `test_cohere_parity.py` — **the gate**: TL-vs-HF next-token KL `< 1e-4` on
  short prompts and past the 4096-token sliding-window boundary. This is the
  only test that numerically catches a wrong RoPE interleaving, LayerNorm fold,
  embedding tie, or `logit_scale` fold.
- `test_cohere_discovery.py` — end-to-end `discover_circuit` on `greater_than`
  for EAP, EAP-IG, CD-T, and IBCircuit; asserts finite, non-degenerate scores.

!!! note "LayerNorm-fold fallback"
    If the parity gate fails only slightly (LayerNorm-fold accumulation on a
    3.35B model), load the TL model with `from_pretrained_no_processing` /
    `fold_ln=False` and re-compare before suspecting the port.

---

## See also

- [Experimental Models](experimental-models.md) — Command R7B, Aya Expanse
  8B / 32B, and SmolLM3-3B, added through the same `_tl_compat` seam.
- [Architecture Registry](architecture-registry.md) — the `cohere` family entry
  and `EXPERIMENTAL_FAMILIES`.
- `circuitkit.backends._tl_compat` — the port source.
