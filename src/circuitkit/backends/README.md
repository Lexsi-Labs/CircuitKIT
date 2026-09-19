# backends

Circuit-discovery algorithm backends (EAP, ACDC, CD-T, IBCircuit) plus the
canonical registry of every algorithm name CircuitKit knows about.

## Key modules

- `__init__.py` — the `ALGORITHMS` registry: single source of truth mapping each
  algorithm to its `(category, stability)` pair and derived helper views.

## Public API / entry points

`__init__.py` exposes:

- `ALGORITHMS` — dict of `name → (category, stability)`; `category ∈ {discovery,
  pruning, quantization}`, `stability ∈ {stable, experimental, research}`.
- `STABILITY`, `STABLE_ALGORITHMS`, `EXPERIMENTAL_ALGORITHMS`,
  `RESEARCH_ALGORITHMS` — derived stability views.
- `DISCOVERY_ALGORITHMS`, `PRUNING_ALGORITHMS`, `QUANTIZATION_ALGORITHMS`,
  `SUPPORTED_ALGORITHMS` — derived category views.
- `is_stable`, `is_experimental`, `is_research`, `category_of`,
  `default_algorithm` — predicates/lookups. `DEFAULT_ALGORITHM = "eap-ig"`.

## Stability tiers

- **stable** — tested across the GPT-2, Llama, Gemma, and Qwen families: `eap`,
  `eap-ig`, `eap-gp`, `acdc` (node-only by construction, slow), `ibcircuit`
  (memory ceiling on multi-billion-parameter models at aggressive settings),
  `cdt` (clean-only; approximate scores on RoPE models), plus pruning/quantization
  selectors.
- **experimental** — works on IOI, may fail on larger models: none currently.
- **research** — implemented but unvalidated outside GPT-2 IOI: EAP research
  variants (`eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`,
  `relp`, `peap`, `eap-ifr`).

## How it fits

`circuitkit.utils.exceptions` derives its validation registries from this
`__init__.py` — do not maintain a second copy. Each subpackage is one discovery
backend, dispatched from `api.discover_circuit`.
