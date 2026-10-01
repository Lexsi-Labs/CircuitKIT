# Experimental Models: Command R7B, Aya Expanse 8B, SmolLM3-3B

CircuitKIT's TransformerLens **compatibility port**
(`circuitkit.backends._tl_compat`) started with [Tiny Aya](tiny-aya.md)
(`cohere2`) and has since grown into a **multi-architecture seam**: one
internal registry (`_tl_compat/registry.py`) that lets several HF
architectures share the same `convert_hf_model_config` /
`get_pretrained_state_dict` dispatch and a per-architecture, per-layer
rotary (NoPE) policy. Three more model families now ride that seam:

| Model | HF architecture | `model_type` | Port module |
|---|---|---|---|
| [Tiny Aya](tiny-aya.md) | `Cohere2ForCausalLM` | `cohere2` | `_tl_compat/cohere.py` |
| **Command R7B** | `Cohere2ForCausalLM` | `cohere2` | `_tl_compat/cohere.py` |
| **Aya Expanse 8B** | `CohereForCausalLM` | `cohere` | `_tl_compat/cohere.py` |
| **SmolLM3-3B** | `SmolLM3ForCausalLM` | `smollm3` | `_tl_compat/smollm3.py` |

All four are **experimental** discovery/evaluation/intervention targets:
TransformerLens 3.8.0 has no native support for any of `cohere`, `cohere2`, or
`smollm3`; the port adds it additively, at import time, with every patch
falling through to stock TL for any other architecture. Command R7B and Aya
Expanse are **not gated on the Hub the way tiny-aya is**, but both are
flagged `gated=auto` — an account needs to accept Cohere's license
click-through once before the safetensors are downloadable, using the same
`HF_TOKEN` / cached-login auth path as tiny-aya. SmolLM3-3B is a fully public
repository — no `HF_TOKEN` needed.

!!! warning "Experimental"
    These are experimental discovery targets. Results should only be
    trusted once the real-weight **parity gate** passes on your machine (see
    [Tests](#tests) below).

!!! note "Local checkpoint folders"
    TransformerLens 3.8 uses `llama` and `gemma` substrings in a local checkpoint
    path for loader selection before it reads `config.json`. CircuitKIT rejects
    ambiguous local paths with a rename/symlink suggestion; use a neutral folder
    name such as `./checkpoint`. Local SmolLM3 checkpoint folders are not
    supported by the current port; load `HuggingFaceTB/SmolLM3-3B` by Hub ID.

## Gemma-4 and Sarvam-MoE: discovery-only TransformerLens ports

The Gemma-4 and Sarvam-MoE patches provide a **discovery-only** TransformerLens
path. They do not establish support for CircuitKIT evaluation, pruning,
quantization, steering or interventions; those surfaces are not validated for
these models. Treat successful loading as experimental, not as full model
support.

| Checkpoint | Scope and requirements |
|---|---|
| `google/gemma-4-31B-it` | Text-side discovery only; the multimodal vision tower is not exposed through the TransformerLens port. At bf16, weights alone are about 58 GiB; use an 80 GB-class GPU or larger to leave room for activations. |
| `sarvamai/sarvam-30b` | TransformerLens discovery only. Sarvam ships its modeling code in the repository rather than in `transformers`, so it needs `trust_remote_code`: `load_model(..., trust_remote_code=True)`, `trust_remote_code: true` under `model:` in a YAML/dict config, or `circuitkit discover --trust-remote-code`. This executes code from the model repository, so it is off by default and never inferred. At bf16, weights alone are about 56 GiB; use an 80 GB-class GPU or larger to leave room for activations. |

---

## Supported checkpoints

- `CohereLabs/c4ai-command-r7b-12-2024` (Command R7B, `cohere2`)
- `CohereLabs/aya-expanse-8b` (Aya Expanse 8B, `cohere1`/`cohere`)
- `HuggingFaceTB/SmolLM3-3B` (SmolLM3-3B, `smollm3`)

---

## Architecture at a glance

| Property | Command R7B | Aya Expanse 8B | SmolLM3-3B |
|---|---|---|---|
| HF architecture | `Cohere2ForCausalLM` | `CohereForCausalLM` | `SmolLM3ForCausalLM` |
| Params | ~7B | ~8B | ~3B |
| `d_model` / layers | 4096 / 32 | 4096 / 32 | 2048 / 36 |
| Query / KV heads | 32 / 8 (GQA) | 32 / 8 (GQA) | 16 / 4 (GQA) |
| `d_head` | 128 | 128 | 128 |
| `d_mlp` (gated SiLU) | 14336 | 14336 | 11008 |
| Normalization | LayerNorm (bias-less) | LayerNorm (bias-less) | **RMSNorm** |
| Block topology | **parallel** attn+MLP, one input LN | **parallel** attn+MLP, one input LN | **sequential** (standard Llama) |
| RoPE style | GPT-J interleaved (`rotary_adjacent_pairs=True`) | interleaved (`rotary_adjacent_pairs=True`) | **standard Llama**, non-interleaved (`rotary_adjacent_pairs=False`) |
| `rope_theta` | 50000 | 10000 | 5000000 |
| Attention pattern | **sliding-window (4096) ⨯ full**, interleaved | all full attention | all full attention |
| NoPE (no rotary) layers | full-attention layers | **none** — rotary everywhere | **per-layer**, driven by config's `no_rope_layers` (every 4th layer on the real checkpoint) |
| `logit_scale` | 0.25 (real fold) | 0.125 (real fold) | none (no post-multiply at all) |
| Tied embeddings | yes | yes | yes |
| Vocab | 256000 | 256000 | 128256 |
| Max position embeddings (real ckpt) | 132096 (preserved) | 8192 | 65536 (preserved) |
| Gated on the Hub? | `gated=auto` (license click-through) | `gated=auto` (license click-through) | fully public |

For reference, [Tiny Aya](tiny-aya.md) is 36 layers / `d_model=2048` / 16:4
GQA heads / `d_mlp=11008`, the same `cohere2` shape as Command R7B but with
`logit_scale=1.0` (a no-op fold) and a smaller, ~3.35B-parameter checkpoint.

---

## The `_tl_compat` seam

All four models' support lives in `circuitkit.backends._tl_compat`, applied
at import time from `circuitkit/backends/__init__.py`:

```python
from . import _tl_compat
_tl_compat.apply_patches()
```

- **`_tl_compat/registry.py`** — the shared multi-arch seam: an `ArchPort`
  dataclass (`architecture`, `model_names`, `config_converter`,
  `weight_converter`, `should_skip_rotary`) registered per HF architecture,
  resolved by model name or by architecture at call time. `apply_rotary` is
  patched **once** per `AbstractAttention` subclass and dispatches to
  whichever port's NoPE policy applies — a policy takes `(cfg, layer_id,
  layer_attn_type)` so it can key off either the layer index (SmolLM3's
  per-layer mask) or TL's `local`/`global` attn-type string (cohere2's
  sliding-window split).
- **`_tl_compat/cohere.py`** — Tiny Aya, Command R7B (`cohere2`), **and**
  Aya Expanse (`cohere1`/`CohereForCausalLM`). cohere1 and cohere2 share the
  exact same weight converter (`convert_cohere2_weights`); only the config
  converter and NoPE policy differ.
- **`_tl_compat/smollm3.py`** — SmolLM3-3B, a structurally distinct
  Llama-family port (RMSNorm, sequential block, standard rotary, per-layer
  NoPE) with its own weight converter.

### Config mapping (`convert_hf_model_config`)

**Command R7B** — identical mapping to Tiny Aya (same `cohere2` converter);
the only difference is `logit_scale=0.25` instead of `1.0`:

| HF / spec | TL `HookedTransformerConfig` |
|---|---|
| `rope_gptj` (interleaved pairs) | `rotary_adjacent_pairs=True` |
| `rope_theta=50000` | `rotary_base=50000` |
| `sliding_window=4096` + `layer_types` | `use_local_attn=True`, `window_size=4096`, `attn_types` |
| parallel block, single `input_layernorm` | `parallel_attn_mlp=True` |
| `num_key_value_heads` | `n_key_value_heads` (GQA) |
| LayerNorm | `normalization_type="LN"` |
| `logit_scale=0.25` | folded into `unembed.W_U` at weight-conversion time |

**Aya Expanse 8B** — a *simplification* of the cohere2 mapping above, via
its own `_convert_cohere1_config` (cohere1's HF config has no
`sliding_window`/`layer_types` attributes at all, and doesn't always set an
explicit `head_dim`, so this is a separate function rather than a branch):

| HF / spec | TL `HookedTransformerConfig` |
|---|---|
| `rotate_half` interleaved (confirmed from `modeling_cohere.py` source) | `rotary_adjacent_pairs=True` |
| `rope_theta=10000` | `rotary_base=10000` |
| no `sliding_window` / `layer_types` at all | `use_local_attn=False`, no `attn_types` — rotary and full attention on every layer |
| parallel block, single `input_layernorm` | `parallel_attn_mlp=True` |
| `head_dim` not always set | `getattr(hf_config, "head_dim", None) or (hidden_size // n_heads)` |
| `logit_scale=0.125` | folded into `unembed.W_U` |
| `use_qk_norm=True` (unused today) | raises `NotImplementedError` — the weight converter has no q/k-norm conversion path |

**SmolLM3-3B** — a new Llama-family mapping, via `_convert_smollm3_config`:

| HF / spec | TL `HookedTransformerConfig` |
|---|---|
| RMSNorm | `normalization_type="RMS"`, `final_rms=True`, `eps=rms_norm_eps` |
| sequential decoder block | `parallel_attn_mlp=False` |
| standard (non-interleaved) Llama rotary (`rotate_half` splits first/second half) | `rotary_adjacent_pairs=False` |
| `rope_theta=5000000` | `rotary_base=5000000` |
| `no_rope_layers` list (`1`=apply rotary, `0`=skip) | per-layer NoPE policy, see below |
| gated SiLU MLP | `gated_mlp=True`, `act_fn="silu"` |
| no logit scale | `unembed.W_U` is an unscaled transpose of `lm_head.weight` |
| `head_dim` not always set | same `getattr(..., None) or (hidden_size // n_heads)` fallback as cohere1 |
| `attention_bias` / `mlp_bias` / `use_sliding_window` = `True` | each raises `NotImplementedError` — none true on the real checkpoint, but the converter would otherwise silently produce a wrong model |

### Context length and memory

The ports preserve each checkpoint's full `max_position_embeddings` in
TransformerLens 3.8. That release keeps an empty causal-mask buffer and builds
the mask for the active input length at forward time; it does not allocate a
dense `n_ctx × n_ctx` mask during model construction. Longer input sequences
still require more attention memory, so choose sequence length and batch size
for the available device rather than relying on an artificial 8192-token cap.

### Weight conversion

Command R7B and Aya Expanse share `convert_cohere2_weights` (see
[Tiny Aya](tiny-aya.md#weight-conversion-get_pretrained_state_dict) for the
GQA / parallel-block / tied-embedding details, which apply unchanged). The
one thing that varies per checkpoint is the `logit_scale` fold:
`unembed.W_U = lm_head.weight.T * logit_scale` — `0.25` for Command R7B,
`0.125` for Aya Expanse, `1.0` (a no-op) for Tiny Aya.

SmolLM3-3B has its own `convert_smollm3_weights`, mirroring TL's own
`convert_llama_weights`: RMSNorm `.w`-only (no bias tensors anywhere — unlike
the cohere family's explicit zero-bias trick, RMSNorm has no bias term to
satisfy in the first place), `_W_K`/`_W_V` GQA slots, gated-SiLU
`W_gate`/`W_in`/`W_out`, tied `W_E`/`W_U` with **no** scale fold.

All zero-tensor constructions (biases, LN zero-biases where applicable) are
built on `next(model.parameters()).device` rather than a bare
`torch.zeros(...)` (which silently defaults to CPU) — this matters once the
source HF model is loaded explicitly onto a GPU, as the real-weight parity
gates do.

### NoPE

- **Command R7B** (same policy as Tiny Aya): `Cohere2Attention.forward` only
  rotates q/k when `self.sliding_window is not None`, so periodic
  full-attention ("global") layers get **no** positional embedding at all.
- **Aya Expanse**: `CohereAttention.forward` calls `apply_rotary_pos_emb`
  unconditionally on every layer — there is no per-layer branch in this
  architecture at all, so the registered NoPE policy always returns `False`.
- **SmolLM3-3B**: a genuinely **per-layer** policy. HF's `no_rope_layers`
  list means truthy (`1`) = apply rotary, falsy (`0`) = skip (confirmed from
  `SmolLM3Attention.__init__`'s `self.use_rope = config.no_rope_layers[layer_idx]`).
  Because `HookedTransformerConfig` is a plain dataclass with no "extra
  data" slot, the resolved per-layer list can't be threaded through the
  config dict the way cohere2 reuses TL's own `attn_types` field for its
  local/global split — it is instead stashed in a small module-level cache
  keyed by `official_model_name` (the same string already written to
  `cfg.tokenizer_name`), and the registered rotary policy looks it up from
  there at call time.

---

## Running discovery

```bash
export HF_TOKEN=hf_...   # Command R7B / Aya Expanse only (license click-through); not needed for SmolLM3
```

```python
import circuitkit as ck

model = ck.load_model("CohereLabs/c4ai-command-r7b-12-2024")   # or aya-expanse-8b / SmolLM3-3B
circuit = ck.discover(model, task="greater_than", algorithm="eap-ig")
```

### Task caveat — single-token answers

Same caveat as [Tiny Aya](tiny-aya.md#task-caveat-single-token-answers): the
single-answer-position discovery metrics assume each answer is one token.
`greater_than` builds its own single-token operand pool against the loaded
tokenizer and is the recommended smoke task for all three models. `ioi`'s
99-name pool was verified to stay single-token under Command R7B's 256k
vocab (a larger vocab than GPT-2's 50k only shrinks the risk of a name
splitting, never grows it); it is used as the ACDC opt-in task below since
ACDC has no built-in `greater_than` support.

---

## Tests

Offline unit tests (no network, no GPU, no gated weights) run in ordinary CI:

- `tests/backends/tl_compat/test_command_r7b.py` — registration, config
  mapping (including the `logit_scale=0.25` fold and the preserved context length).
- `tests/backends/tl_compat/test_aya_expanse.py` — registration, the cohere1
  config converter (no sliding window / `attn_types`, `head_dim` fallback,
  `use_qk_norm` guard), rotary-everywhere NoPE policy.
- `tests/backends/tl_compat/test_smollm3.py` (41 cases) — registration,
  config/weight conversion, **per-layer NoPE** (`TestSmolLM3PerLayerNoPE`,
  parametrized across NoPE and non-NoPE layer indices) and
  **RMSNorm-vs-LayerNorm** (`TestSmolLM3RmsNormNotLayerNorm`) assertions.
- `tests/apply/test_architecture_registry.py` — offline `arch_registry`
  resolution/mock coverage for the `cohere` and `smollm3` families
  (`MockCohereModel`, `MockSmolLM3Model`).

Real-weight gates are **opt-in**, one env var per family, and are `slow`.
Each family has four gates: **parity** (the numerical proof), **discovery**
(end-to-end circuit finding), **evaluation** (faithfulness), and
**interventions** (pruning/quantization/steering module resolution) —
~20 Group F test files across all four models in this port.

**Command R7B** (`CIRCUITKIT_RUN_COMMAND_R7B=1`, needs `HF_TOKEN`):

```bash
CIRCUITKIT_RUN_COMMAND_R7B=1 HF_TOKEN=... python -m pytest \
    tests/backends/tl_compat/test_command_r7b_parity.py \
    tests/regression/test_command_r7b_discovery.py \
    tests/regression/test_command_r7b_evaluation.py \
    tests/regression/test_command_r7b_interventions.py -v

# ACDC only (separate opt-in — see "ACDC is excluded" below):
CIRCUITKIT_RUN_COMMAND_R7B=1 CIRCUITKIT_RUN_COMMAND_R7B_ACDC=1 HF_TOKEN=... \
    python -m pytest tests/regression/test_command_r7b_discovery.py -v -k acdc
```

**Aya Expanse 8B** (`CIRCUITKIT_RUN_AYA_EXPANSE=1`, needs `HF_TOKEN`):

```bash
CIRCUITKIT_RUN_AYA_EXPANSE=1 HF_TOKEN=... python -m pytest \
    tests/backends/tl_compat/test_aya_expanse_parity.py \
    tests/regression/test_aya_expanse_discovery.py \
    tests/regression/test_aya_expanse_evaluation.py \
    tests/regression/test_aya_expanse_interventions.py -v

# ACDC only:
CIRCUITKIT_RUN_AYA_EXPANSE=1 CIRCUITKIT_RUN_AYA_EXPANSE_ACDC=1 HF_TOKEN=... \
    python -m pytest tests/regression/test_aya_expanse_discovery.py -v -k acdc
```

**SmolLM3-3B** (`CIRCUITKIT_RUN_SMOLLM3=1`, public repo, no `HF_TOKEN`
needed):

```bash
CIRCUITKIT_RUN_SMOLLM3=1 python -m pytest \
    tests/backends/tl_compat/test_smollm3_parity.py \
    tests/regression/test_smollm3_discovery.py \
    tests/regression/test_smollm3_evaluation.py \
    tests/regression/test_smollm3_interventions.py -v

# ACDC only:
CIRCUITKIT_RUN_SMOLLM3=1 CIRCUITKIT_RUN_SMOLLM3_ACDC=1 \
    python -m pytest tests/regression/test_smollm3_discovery.py -v -k acdc
```

(Tiny Aya's own gates use `CIRCUITKIT_RUN_TINY_AYA=1` — see
[Tiny Aya's Tests section](tiny-aya.md#tests).)

**What each gate proves:**

- **Parity** (`test_*_parity.py`) — TL-vs-HF next-token KL divergence
  `< 1e-4` on short prompts (and past the 4096-token sliding-window
  boundary for Command R7B). This is the only test that numerically catches
  a wrong RoPE interleaving choice, LayerNorm fold, embedding tie, or
  `logit_scale` fold. Observed on real weights: Aya Expanse KL ≈ 1e-7–4e-7
  (four orders of magnitude under the gate); SmolLM3-3B KL ≈ 2.9e-8–1.1e-7
  (four to five orders of magnitude under the gate), with argmax agreement
  and the exact per-layer NoPE mask (`{3, 7, 11, ..., 35}` skip rotary) both
  confirmed directly on the loaded model. These figures are from manual
  real-weight GPU runs; CI does not reproduce this parity measurement.
- **Discovery** (`test_*_discovery.py`) — end-to-end `discover_circuit` on
  `greater_than` for 5 of the 6 stable algorithms (`eap`, `eap-ig`,
  `eap-gp`, `ibcircuit`, `cdt`); asserts finite, non-degenerate node scores.
- **Evaluation** (`test_*_evaluation.py`) — `circuitkit.api.evaluate_circuit`
  faithfulness (patching + ablation pillars) on a `cdt`-discovered
  `greater_than` circuit, loading the model once and passing the same
  handle to both discovery and evaluation (avoiding a second 7–8B load).
  `evaluate_circuit` unconditionally sets the same qkv-flag activations
  (`use_attn_result`/`use_split_qkv_input`/`use_hook_mlp_in`) that only the
  EAP-family algorithms need during discovery, regardless of which algorithm
  produced the circuit being evaluated — at smoke-test batch/example counts
  this ran cleanly on GPU for all four models, but a future maintainer
  raising those counts on a 7–8B model may need the same `CUDA_VISIBLE_DEVICES=""`
  CPU fallback discovery needed.
- **Interventions** (`test_*_interventions.py`) — pruning score extraction,
  quantization target-module resolution, and weight-steering setup, all on
  real weights. See [Architecture Registry](architecture-registry.md) for
  exactly what was and wasn't run.

### ACDC is excluded from the standard gate

`acdc` (the sixth stable algorithm) needs the same per-head qkv-flag
activation blow-up as the EAP family. CircuitKIT's own OOM preflight
correctly refuses this on a single 47 GB GPU for a 7–8B model; the only
fallback is CPU, where a single ACDC run — already scoped to one `tao` value
instead of the default 8-combination grid — took over 2.5 hours on a 7B
model without finishing. That is not a practical regression gate, so each
model's ACDC test is kept (it does work — verified to launch, wire up `ioi`,
and run) but skipped by default behind a second, separate opt-in
(`CIRCUITKIT_RUN_<FAMILY>_ACDC=1`) so it doesn't silently cost hours on
every real-weight run. This exclusion is standing policy for future
large-model discovery regression suites, not specific to these three models.

### GPU memory reality

None of CircuitKIT's discovery algorithms currently shard activations across
multiple GPUs. For a 7–8B model, the qkv-flag algorithms (`eap`, `eap-ig`,
`eap-gp`, `acdc`) and `ibcircuit`'s own fixed-batch training step both
exceed a single 47 GB GPU's headroom and need to run on CPU
(`CUDA_VISIBLE_DEVICES=""`) — slower (tens of minutes), but correct, and the
repo's own OOM preflight refuses the GPU attempt loudly rather than crashing
mid-forward. `cdt` needs neither. SmolLM3-3B (~6 GB bf16) is small enough
that its entire discovery gate ran on GPU with no CPU fallback.

---

## See also

- [Tiny Aya (Cohere2)](tiny-aya.md) — the original `cohere2` port these three
  models extend; its Tests section documents `CIRCUITKIT_RUN_TINY_AYA`.
- [Architecture Registry](architecture-registry.md) — the `cohere` family
  (now covering Tiny Aya, Command R7B, and Aya Expanse) and the new
  `smollm3` family, plus what the intervention validation actually
  covered per model.
- `circuitkit.backends._tl_compat` — the port source
  (`registry.py`, `cohere.py`, `smollm3.py`).
