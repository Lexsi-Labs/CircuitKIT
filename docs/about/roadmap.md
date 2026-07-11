# Roadmap

---

## v1.1 — Planned

### Steering and Editing at Scale

- **Production-grade `CircuitWeightSteering`** — validation on Llama-3 and Gemma-2 at 7B+ scale
- **Attention target selection for ROME** — currently only MLP targets are supported; v1.1 adds head-level editing targets
- **`HealingEvaluator` production hardening** — recovery metrics validated on benchmark splits

### Evaluation Improvements

- **Pillar 6 (Generalization) production validation** — full scale sweep to move it from "preliminary" to validated
- **Intervention Reliability as standard pillar** — promote from optional auxiliary to the canonical numbered framework

### Algorithm Expansions

- **IBCircuit OOM fix** — memory-efficient implementation for >3B parameter models
- **CD-T validation** — validate the RoPE handling and gated-MLP split, which are currently approximations, to move CD-T off the research tier
- **New stable algorithm candidate** — RelP promotion from research tier pending validation

### Infrastructure

- **HuggingFace Hub integration** — push circuit artifacts to HF Hub; `ck.load_scores("org/model-circuit")`
- **Async pipeline execution** — non-blocking multi-step pipeline via asyncio
- **Circuit visualization improvements** — interactive HTML graphs with filtering by layer and score threshold

---

## v1.2 and Beyond

- **Multi-model circuit comparison** — compare circuits across model families for the same task
- **Automated circuit regression testing** — CI hook to detect circuit drift across model versions
- **Cross-language support** — extend to non-English tasks with multilingual evaluation

---

## Not Planned

- **PyTorch 1.x support** — minimum is 2.0 and will stay that way
- **Python 3.9 support** — minimum is 3.10 for match/case and improved type annotations
- **Non-TransformerLens discovery** — the discovery stage will remain TL-based in the v1.x line

---

## Feature Requests

Open a GitHub issue with the label `enhancement`. PRs welcome — see the [Contributing Guide](../contributing/setup.md) for how to add a new algorithm.
