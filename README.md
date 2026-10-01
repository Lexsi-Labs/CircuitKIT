<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/circuitkit-logo-white.png">
    <img src="docs/assets/circuitkit-logo-black.png" width="360" alt="CircuitKIT">
  </picture>
</p>

<p align="center">
  <b>Discover, evaluate, and intervene on circuits in transformer models.</b><br>
  One call takes a model + task to a discovered circuit, a 6-pillar faithfulness score, and a pruned HuggingFace checkpoint.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-LSAL%20v1.2-blue.svg" alt="License: LSAL v1.2 (source-available)"></a>
  <a href="https://circuitkit.lexsi.ai/"><img src="https://img.shields.io/badge/docs-mkdocs%20material-FF4B0A.svg" alt="Docs"></a>
</p>

---

CircuitKIT is a framework for mechanistic interpretability. Given a model and a task, it discovers the circuit driving that behaviour, evaluates how faithful it is, and lets you act on it (prune, quantize, edit, steer, or fine-tune), then export a reloadable HuggingFace checkpoint.

**No GPU required for the quickstart** — GPT-2 runs on CPU in a few minutes.

## Quick start

```bash
# CPU-only, no GPU needed:
pip install -e .

# For benchmarking, add: pip install -e ".[benchmarks]"
```

```python
from circuitkit import Pipeline

pipe = Pipeline("gpt2", task="ioi")
pipe.discover(algorithm="eap-ig", sparsity=0.3)
pipe.evaluate()
pipe.prune()
pipe.export("./checkpoint")
```

`load_model`, `Pipeline` and the `api` entry points also take a local checkpoint
directory or any Hub repo id (e.g. a SafeTune or AlignTune output); the architecture is
read from its `config.json`. Aya Vision and North Micro Vision are vision-language
models and are rejected with an error: TransformerLens has no vision tower.

### Single environment

CircuitKIT runs on transformer-lens 3.8.0 and transformers 5. Import
`circuitkit` before `transformer_lens`, and a plain `pip install circuitkit` is
enough.

```python
import circuitkit as ck

model = ck.load_model("CohereLabs/aya-expanse-8b")  # or "CohereLabs/tiny-aya-global" (gated), or a local dir
circuit = ck.discover(model, "ioi", n_examples=32)
ck.export_checkpoint(model, circuit, "./aya-pruned")
```

`ck.export_checkpoint` writes the model's current weights, so ROME / MEMIT edits and
weight steering are in the checkpoint, plus a `lexsi_provenance.json`. The
`*_scores.json` files also carry `safety_units` / `layer_suggestions`, which SafeTune's
CircuitKIT adapter reads.

Also works as a [CLI](https://circuitkit.lexsi.ai/cli/overview/) and [YAML config](https://circuitkit.lexsi.ai/cli/yaml-config/).

## What is a circuit?

A **circuit** is the minimal set of attention heads and MLP layers in a transformer that drives a specific behaviour. Most interp tooling stops at "here is a subgraph with attribution scores." CircuitKIT goes further: it prunes (or quantizes) the model down to that subgraph, exports a reloadable HuggingFace checkpoint, and measures how faithful the pruned model stays. Because the circuit is task-specific, this produces a task-specialized checkpoint — not a general-purpose compressed model.

## What you can do

| Capability | What it means |
|---|---|
| **Discover** | 6 stable algorithms (EAP, EAP-IG, EAP-GP, ACDC, IBCircuit, CD-T), tested across the GPT-2, Llama, Gemma, and Qwen families, plus 7 research ones |
| **Evaluate** | 6-pillar faithfulness: causal patching, ablation, stability, robustness, baselines, generalization |
| **Prune** | Structural weight pruning down to the circuit |
| **Quantize** | Circuit-aware mixed-precision quantization (3/4-bit + protect tiers) |
| **Edit** | ROME / MEMIT knowledge editing at circuit-identified components |
| **Steer** | Activation steering at inference (no retraining) |
| **Fine-tune** | Circuit-restricted LoRA — only circuit components update |
| **Benchmark** | lm-evaluation-harness integration for compressed checkpoints |

## Supported models

Production support covers Llama-3, Gemma/Gemma-3, Qwen, Mistral, Phi, Falcon,
and GPT-2 (see [Architecture Registry](https://circuitkit.lexsi.ai/advanced/architecture-registry/)).
CircuitKIT also supports four cohere-family and Llama-family models that
TransformerLens doesn't natively support, across all three surfaces —
discovery, evaluation, and interventions:

| Model | Arch | HF repo |
|---|---|---|
| Tiny Aya | `cohere2` | `CohereLabs/tiny-aya-*` (gated) |
| Command R7B | `cohere2` | `CohereLabs/c4ai-command-r7b-12-2024` |
| Aya Expanse 8B | `cohere1` | `CohereLabs/aya-expanse-8b` |
| SmolLM3-3B | `smollm3` | `HuggingFaceTB/SmolLM3-3B` |

All four support discovery (5/6 stable algorithms plus parity), faithfulness
evaluation, and interventions (pruning, quant resolution, steering).

Two caveats on that support. `acdc` is excluded from the standard gate as
impractically slow on 3–8B models — its test lives behind an opt-in flag — so
the five validated on real weights are `eap`, `eap-ig`, `eap-gp`, `ibcircuit`,
`cdt`. And "quant resolution" means target-module resolution confirmed on real
weights; the actual `optimum-quanto`/`llmcompressor` compression call is not
exercised in every environment, since those are optional dependencies.

Gemma-4 and Sarvam-MoE have separate **experimental, discovery-only**
TransformerLens ports; this is not a claim of end-to-end CircuitKIT support.
Their hardware and Sarvam loading requirements are documented in
[Experimental Models](docs/advanced/experimental-models.md#gemma-4-and-sarvam-moe-discovery-only-transformerlens-ports).

All four are **experimental** — see [Tiny Aya](https://circuitkit.lexsi.ai/advanced/tiny-aya/)
and [Experimental Models](https://circuitkit.lexsi.ai/advanced/experimental-models/)
for architecture details, config mapping, and the full opt-in test matrix.
Aya Expanse and Tiny Aya also load from local checkpoint directories and
unlisted Hub repo ids.

## Why CircuitKIT?

| Instead of stitching together… | …CircuitKIT gives you |
|---|---|
| A separate repo per discovery algorithm, plus a pruning script and lm-eval-harness — wired together by hand | One `Pipeline`: discover → evaluate → prune → export → benchmark |
| One-off data formats per tool | Standard circuit artifact + HuggingFace checkpoint |
| GPT-2-only tooling | Llama-3, Gemma, Qwen — with GQA, RoPE, chat templates |
| One faithfulness score | 6-pillar evaluation suite |

## Next steps

| | |
|---|---|
| **[Getting Started](https://circuitkit.lexsi.ai/getting-started/)** | Install, quickstart, core concepts |
| **[User Guide](https://circuitkit.lexsi.ai/guides/)** | Pipeline, custom data, evaluation, selectors, tasks |
| **[Algorithms](https://circuitkit.lexsi.ai/algorithms/overview/)** | EAP, ACDC, IBCircuit, CD-T — with stability tiers |
| **[Applications](https://circuitkit.lexsi.ai/user-guide/applications/)** | Pruning, quantization, editing, steering, fine-tuning |
| **[Examples](https://circuitkit.lexsi.ai/examples/overview/)** | Runnable scripts and notebooks (all CPU-friendly) |
| **[API Reference](https://circuitkit.lexsi.ai/api-reference/overview/)** | Full API and CLI reference |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Citation

```bibtex
@software{circuitkit2026,
  title  = {CircuitKIT: Circuit Discovery, Evaluation, and Application Toolkit
            for Mechanistic Interpretability},
  author = {Seth, Pratinav and Gosalia, Hem and Kasliwal, Aditya
            and Sankarapu, Vinay Kumar},
  year   = {2026},
  url    = {https://github.com/Lexsi-Labs/circuitkit}
}
```

## License

Lexsi Labs Source Available License (LSAL) v1.2: free for academic research and teaching on MIT-like terms; use by any organization requires written acknowledgement or permission (Section 1A); commercial use requires a separate license; responsible-use conditions apply. See [LICENSE.md](LICENSE.md).
