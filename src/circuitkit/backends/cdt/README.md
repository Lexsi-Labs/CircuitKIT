# cdt

Backend for CD-T (Contextual Decomposition for Transformers), a gradient-free
forward-pass method that attributes predictions to source nodes by propagating a
`(rel, irrel)` split through every transformer component.

## Key modules

- `propagation.py` — `cd_propagate_tl` (and helpers): full CD-T propagation over a
  TransformerLens model using TL's unified weight API, so it works across GPT-2,
  GPT-J, Llama 1/2/3, Mistral, Gemma, Qwen2, Falcon, Pythia, etc.
- `adapter.py` — `run_cdt_discovery`: runs the CD-T importance metric on the
  TL checkpoint already loaded by `discover_circuit` (full-propagation by default,
  with a simplified TL-cache fallback).

## Subpackages

- `pyfunctions/` — the vendored CD-T reference implementation (BERT/GPT wrappers,
  core decomposition ops, ablation and source→target routines).

## Public API / entry points

`__init__.py` re-exports the raw `pyfunctions` modules `core`, `basic`, `wrappers`
(via `__all__`) for direct use outside the pipeline.

## How it fits

Dispatched from `api.discover_circuit` (`cdt`, research tier) via
`adapter.run_cdt_discovery`. Note: RoPE handling and the gated-MLP split are
approximations and unvalidated (see registry comment in `backends/__init__.py`).
