<!-- circuitkit-logo -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../../docs/assets/circuitkit-logo-white.png">
    <img src="../../../docs/assets/circuitkit-logo-black.png" width="200" alt="CircuitKIT">
  </picture>
</p>

# TransformerLens patch series

Adds `HookedTransformer` support for Gemma-4 (`google/gemma-4-31B-it`),
Sarvam-MoE (`sarvamai/sarvam-30b`) and Cohere / Cohere2 (Aya Expanse, Tiny Aya) to
**transformer-lens 3.8.0**, and fixes two loading bugs (local checkpoints, GPT-NeoX / Pythia).
It replaces a locally maintained TransformerLens fork with a small, versioned patch
series that is applied in memory and checked against upstream source.

Ministral-3 needs no patch. Upstream 3.8.0 supports `Ministral3ForCausalLM` through
`TransformerBridge` only; neither upstream nor the fork has a `HookedTransformer` path
for it.

| Patch | Changes | In the fork |
|---|---|---|
| `0001-moe-dtype-and-devices-fixes` | `MoE` linears honour `cfg.dtype`; drop an O(n²) no-op loop in `move_to_and_update_config` | yes |
| `0002-gemma4` | per-layer-type attention geometry, unscaled `v_norm`, `convert_gemma4_weights`, registry | yes |
| `0003-sarvam-moe` | sigmoid router with expert bias, shared expert, dense first layer, `convert_sarvam_moe_weights`, registry | yes |
| `0004-gemma4-hf-parity` | proportional RoPE on global layers, final logit softcap 30, `layer_scalar` | **no, new** |
| `0005-memory-no-weight-copies` | opt-in via `CIRCUITKIT_TL_MEMORY_PATCH=1`: `einsum` for `W_O`; no `.T.contiguous()` copies in `GatedMLP`/`Unembed` | yes |
| `0006-cohere` | `CohereForCausalLM` / `Cohere2ForCausalLM` weight conversion and model names; config conversion stays in `_tl_compat/cohere.py` | **no, new** |
| `0007-local-checkpoint-config` | a local checkpoint's config and model class come from its `config.json`, never from substrings of its path; marker `LOCAL_CHECKPOINTS_FROM_CONFIG` | **no, new** |
| `0008-neox-lm-head` | `convert_neox_weights` reads the unembedding from `lm_head` when `embed_out` is absent (GPT-NeoX / Pythia on current transformers) | **no, new** |

`0007` is needed because TransformerLens 3.8 chose the architecture, and for
29 config branches the whole config, from substrings of the model name: a
local fine-tune in `meta-llama-Llama-3.2-1B_sft/` loaded with base Llama-3.2-1B's
hard-coded config and never read its own `config.json`. For a local directory the
branch chain now sees an empty name, so only architecture-keyed branches match
(Llama: the generic `LlamaForCausalLM` branch); Hub names are unchanged. Gemma
1-4 configs exist only for Hub names, so `load_model` refuses local Gemma
checkpoints. TransformerLens 4 loads every model by its `config.json`, so the
patch retires with the series.

`0001`-`0003` reproduce the vendored fork byte for byte. `0005` is not applied by
default because it changes bf16 accumulation order and logits; set
`CIRCUITKIT_TL_MEMORY_PATCH=1` before importing CircuitKIT to opt into its VRAM
reduction. The import hook rejects an import order where `transformer_lens` was
already loaded; restart and import `circuitkit` first. Without `0004`, a
tiny random Gemma-4 disagrees with Hugging Face on 12.5% of top-1 tokens; with it,
max |Δlogit| is 7e-5. `0006` needs no new TransformerLens feature: Cohere uses the
existing parallel-block and interleaved-RoPE flags, and Cohere2's full-attention layers
(no RoPE) reuse `0004`'s `proportional_rotary_factor`, set to 0. Tiny random models of
both match Hugging Face to 1e-7. The config is read from `config.json`, so local
checkpoints of either architecture load too; models with `use_qk_norm` or
`attention_bias` (some Command R releases) raise `NotImplementedError`.
`circuitkit.backends._tl_compat` (Tiny Aya, Command R7B, Aya Expanse, SmolLM3) runs on
top of this series. It registers the Hub names, builds configs, and converts weights;
the Cohere configuration is maintained in `_tl_compat/cohere.py` rather than duplicated
in patch `0006`. Both Hub and local-checkpoint paths are checked against Hugging Face
logits. TransformerLens 3.8 builds causal masks for the active sequence length, so the
ports preserve each checkpoint's full `max_position_embeddings` rather than capping
`n_ctx` at 8192.

`0008` backports the `HookedTransformer` half of an upstream fix released in
TransformerLens 3.9.0. transformers renamed `GPTNeoXForCausalLM.embed_out` to `lm_head`
(upstream gives the version as 5.13 in one place and 5.14 in another), and 3.8.0 reads
only `embed_out`, so without `0008` every GPT-NeoX / Pythia checkpoint, Hub or local,
fails with `AttributeError` on the transformers versions CircuitKIT requires. The patch
is `hasattr`-based, so it does not depend on the exact version. TransformerLens 4 loads
models by their `config.json` and the patch retires with the series.

## How it is applied

`import circuitkit` puts an import hook on `sys.meta_path`
(`circuitkit/_tl_patches/__init__.py`). When a patched `transformer_lens` module is
imported, the hook reads the installed 3.8.0 source, applies the series in memory
(checking every context line) and executes the result. Nothing in site-packages
changes, and `pip install circuitkit` needs no extra step. `transformer-lens` is
pinned to `==3.8.0` for this reason. If `transformer_lens` was imported before
`circuitkit`, the hook is not installed and a warning says so. An environment patched
on disk by the old `scripts/apply_tl_patches.py` (it carries `_circuitkit_patches.txt`)
is used as is.

Load these models without weight processing (`fold_ln=False`,
`center_writing_weights=False`, `center_unembed=False`, `fold_value_biases=False`),
which is what `circuitkit.load_model` does by default. That is the only mode the ports
were validated in; weight processing knows nothing about `v_norm`, `layer_scalar`,
expert bias or the shared expert.

## Tests

```bash
pytest tests/backends/tl_patches -W ignore::DeprecationWarning
```

`test_patch_hook.py` checks that the hook serves exactly what `git apply` of the enabled
patch series writes to disk.

## Moving to a new transformer-lens version

1. Unpack the new wheel into a scratch git repo and commit it.
2. `git am` this series, resolving any conflicts.
3. `git format-patch` the result back into this directory.
4. Update `TL_VERSION` in `__init__.py` here and the `transformer-lens` pin in
   `pyproject.toml`.
5. Drop `0008` once the target version is 3.9.0 or later (the fix is upstream there).
6. Re-run the tests.
