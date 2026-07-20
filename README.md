<p align="center">
  <img src="docs/assets/circuitkit-mark-v3.svg" width="140" alt="CircuitKit">
</p>

<h1 align="center">CircuitKit</h1>

<p align="center">
  <b>Discover, evaluate, and intervene on circuits in transformer models.</b><br>
  One call takes a model + task to a discovered circuit, a 6-pillar faithfulness score, and a intervened HuggingFace checkpoint.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch 2.0+"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-LSAL%20v1.1-blue.svg" alt="License: LSAL v1.1 (source-available)"></a>
  <a href="https://lexsi-labs.github.io/CircuitKIT/"><img src="https://img.shields.io/badge/docs-mkdocs%20material-EC5A2C.svg" alt="Docs"></a>
</p>

---

CircuitKit is a framework for mechanistic interpretability. Given a model and a task, it discovers the circuit driving that behaviour, evaluates how faithful it is, and lets you act on it (prune, quantize, edit, steer, or fine-tune), then export a reloadable HuggingFace checkpoint.

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

Also works as a [CLI](https://lexsi-labs.github.io/circuitkit/cli/overview/) and [YAML config](https://lexsi-labs.github.io/CircuitKIT/cli/yaml-config/).

## What is a circuit?

A **circuit** is the minimal set of attention heads and MLP layers in a transformer that drives a specific behaviour. Most interp tooling stops at "here is a subgraph with attribution scores." CircuitKit goes further: it prunes (or quantizes) the model down to that subgraph, exports a reloadable HuggingFace checkpoint, and measures how faithful the pruned model stays. Because the circuit is task-specific, this produces a task-specialized checkpoint — not a general-purpose compressed model.

## What you can do

| Capability | What it means |
|---|---|
| **Discover** | 13 algorithms across maturity tiers — 2 stable (EAP, EAP-IG), 2 experimental (ACDC, IBCircuit), 9 research |
| **Evaluate** | 6-pillar faithfulness: causal patching, ablation, stability, robustness, baselines, generalization |
| **Prune** | Structural weight pruning down to the circuit |
| **Quantize** | Circuit-aware mixed-precision quantization (3/4-bit + protect tiers) |
| **Edit** | ROME / MEMIT knowledge editing at circuit-identified components |
| **Steer** | Activation steering at inference (no retraining) |
| **Fine-tune** | Circuit-restricted LoRA — only circuit components update |
| **Benchmark** | lm-evaluation-harness integration for compressed checkpoints |

## Why CircuitKit?

| Instead of stitching together… | …CircuitKit gives you |
|---|---|
| A separate repo per discovery algorithm, plus a pruning script and lm-eval-harness — wired together by hand | One `Pipeline`: discover → evaluate → prune → export → benchmark |
| One-off data formats per tool | Standard circuit artifact + HuggingFace checkpoint |
| GPT-2-only tooling | Llama-3, Gemma, Qwen — with GQA, RoPE, chat templates |
| One faithfulness score | 6-pillar evaluation suite |

## Next steps

| | |
|---|---|
| **[Getting Started](https://lexsi-labs.github.io/CircuitKIT/getting-started/)** | Install, quickstart, core concepts |
| **[User Guide](https://lexsi-labs.github.io/CircuitKIT/guides/)** | Pipeline, custom data, evaluation, selectors, tasks |
| **[Algorithms](https://lexsi-labs.github.io/CircuitKIT/algorithms/overview/)** | EAP, ACDC, IBCircuit, CD-T — with stability tiers |
| **[Applications](https://lexsi-labs.github.io/CircuitKIT/applications/)** | Pruning, quantization, editing, steering, fine-tuning |
| **[Examples](https://lexsi-labs.github.io/CircuitKIT/examples/overview/)** | Runnable scripts and notebooks (all CPU-friendly) |
| **[API Reference](https://lexsi-labs.github.io/CircuitKIT/api-reference/overview/)** | Full API and CLI reference |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

## Citation

```bibtex
@software{circuitkit2026,
  title  = {CircuitKit: Circuit Discovery, Evaluation, and Application Toolkit
            for Mechanistic Interpretability},
  author = {Seth, Pratinav and Gosalia, Hem and Kasliwal, Aditya
            and Sankarapu, Vinay Kumar},
  year   = {2026},
  version = {1.0.0},
  url    = {https://github.com/Lexsi-Labs/CircuitKIT}
}
```

## License

Lexsi Labs Source Available License (LSAL) v1.1: free for research, evaluation, education, and audit; commercial use requires a separate license; responsible-use conditions apply. See [LICENSE.md](LICENSE.md).
