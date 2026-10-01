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
**transformer-lens 3.8.0**. It replaces the
fork vendored at `c-delta-theta-multilingual-safety-audit@09092026:vendor/transformer_lens`
(commit `dc0caa76`, ported from `CircuitKit_old@gemma4-cdelta-theta`).

Ministral-3 needs no patch. Upstream 3.8.0 supports `Ministral3ForCausalLM` through
`TransformerBridge` only; neither upstream nor the fork has a `HookedTransformer` path
for it.

| Patch | Changes | In the fork |
|---|---|---|
| `0001-moe-dtype-and-devices-fixes` | `MoE` linears honour `cfg.dtype`; drop an O(n²) no-op loop in `move_to_and_update_config` | yes |
| `0002-gemma4` | per-layer-type attention geometry, unscaled `v_norm`, `convert_gemma4_weights`, registry | yes |
| `0003-sarvam-moe` | sigmoid router with expert bias, shared expert, dense first layer, `convert_sarvam_moe_weights`, registry | yes |
| `0004-gemma4-hf-parity` | proportional RoPE on global layers, final logit softcap 30, `layer_scalar` | **no, new** |
| `0005-memory-no-weight-copies` | `einsum` for `W_O`; no `.T.contiguous()` copies in `GatedMLP`/`Unembed` | yes |
| `0006-cohere` | `CohereForCausalLM` / `Cohere2ForCausalLM` config + `convert_cohere_weights`, registry (`CohereLabs/aya-expanse-*`, `CohereLabs/tiny-aya-*`) | **no, new** |

`0001`-`0003` + `0005` reproduce the vendored fork byte for byte. Without `0004`, a
tiny random Gemma-4 disagrees with Hugging Face on 12.5% of top-1 tokens; with it,
max |Δlogit| is 7e-5. `0006` needs no new TransformerLens feature: Cohere uses the
existing parallel-block and interleaved-RoPE flags, and Cohere2's full-attention layers
(no RoPE) reuse `0004`'s `proportional_rotary_factor`, set to 0. Tiny random models of
both match Hugging Face to 1e-7. The config is read from `config.json`, so local
checkpoints of either architecture load too; models with `use_qk_norm` or
`attention_bias` (some Command R releases) raise `NotImplementedError`.
`circuitkit.backends._tl_compat` (Tiny Aya, Command R7B, Aya Expanse, SmolLM3) runs on
top of this series. For the Hub names it registers it builds the config itself (with
`n_ctx` capped at 8192), and it converts the weights of every Cohere / Cohere2 model;
`0006` supplies the config for local directories and unlisted repo ids. Both paths are
checked against Hugging Face logits. `0005` changes bf16 accumulation order for every model, so
logits are no longer bit-identical to HF. Delete it to keep upstream numerics.

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
which is what `circuitkit.load_model` does by default. That is the only mode the ports were validated in; weight processing knows nothing
about `v_norm`, `layer_scalar`, expert bias or the shared expert.

## Tests

```bash
pytest tests/backends/tl_patches -W ignore::DeprecationWarning
```

`test_patch_hook.py` checks that the hook serves exactly what `git apply` of the series
writes to disk.

## Moving to a new transformer-lens version

1. Unpack the new wheel into a scratch git repo and commit it.
2. `git am` this series, resolving any conflicts.
3. `git format-patch` the result back into this directory.
4. Update `TL_VERSION` in `__init__.py` here and the `transformer-lens` pin in
   `pyproject.toml`.
5. Re-run the tests.
