# EAP Family

**Edge Attribution Patching** (EAP) and its variants power most of CircuitKit's discovery. Both Stable-tier algorithms, `eap` and `eap-ig`, belong to this family, alongside the Research-tier `eap-ig-activations` and `eap-clean-corrupted` and several other Research variants.

---

## Core Idea

EAP frames circuit discovery as an attribution problem. Given a model $M$, a clean input $x$, a corrupted input $x^*$, and a scalar metric $f$ (e.g., logit-difference):

1. Run both forward passes: $M(x)$ and $M(x^*)$
2. For each edge $(u, v)$ in the computation graph, compute the attribution score:
   $$\text{attr}(u \to v) = \frac{\partial f}{\partial \text{act}(u \to v)} \cdot \Delta\text{act}(u \to v)$$
   where $\Delta\text{act} = \text{act}_{clean} - \text{act}_{corrupt}$
3. Rank edges (or aggregate to nodes) by attribution score
4. Keep the top-K nodes as the circuit

EAP-IG extends this with Integrated Gradients — instead of using the gradient at a single point, it averages the gradient along a path from the corrupted activation to the clean activation:

$$\text{attr}_{IG}(u \to v) = (\text{act}_{clean} - \text{act}_{corrupt}) \cdot \int_0^1 \frac{\partial f}{\partial \text{act}_\alpha} d\alpha$$

approximated with `ig_steps` steps (default: 5).

---

## The Two Stable Variants

### `eap-ig` — Default

```python
discover_circuit({
    "model": {"name": "gpt2"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "level": "node",
        "data_params": {"num_examples": 128},
        "ig_steps": 5,           # integration steps (default; increase for accuracy)
    },
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

**Why EAP-IG is the default:**
- More accurate than vanilla EAP (IG reduces noise in gradient estimates)
- Marginally more expensive (~`ig_steps` × the cost of vanilla EAP)
- Validated across GPT-2 through Llama-3.2-3B, Gemma-4B, Qwen2.5-1.5B

**Memory guidance:** With `ig_steps=5`, peak GPU memory is ~1.5× the model size. For 3B models, this means ~9–12 GB. See [Memory Optimization](../advanced/memory-optimization.md) for reduction strategies.

---

### `eap` — Fast Baseline

```python
"discovery": {"algorithm": "eap", "task": "ioi", "level": "node", ...}
```

Vanilla EAP — one gradient step at the clean activation point. ~30% faster than EAP-IG, but gradient estimates are noisier. Use when discovery speed is the bottleneck and you have many models to process.

---

## Research-Tier EAP Variants

These are implemented for algorithm comparison studies. Only validated on GPT-2 IOI — do not use for standard circuit discovery:

| Algorithm | Key idea |
|-----------|---------|
| `eap-ig-activations` | Integrated Gradients over **node activations** (the output of each component) rather than edge activations |
| `eap-clean-corrupted` | Uses both clean and corrupted forward passes in the gradient computation |
| `eap-exact` | Exact leave-one-out patching; used as reference baseline |
| `atp-gd` | Attribution Patching with GradDrop (Kramár et al. 2024) |
| `eap-gp` | EAP-GP / GradPath — adaptive integration path instead of straight-line IG (Zhang et al. 2025) |
| `relp` | Relevance Patching — LRP-ε-style gradient re-routing (Rezaei Jafari et al. 2025) |
| `peap` | Position-aware EAP — retains the position dimension (Haklay et al. 2025) |
| `eap-ifr` | Information Flow Routes — proximity scores from a single clean forward pass (Ferrando et al. 2024) |

```python
"discovery": {"algorithm": "eap-ig-activations", "task": "ioi", "level": "node", ...}
"discovery": {"algorithm": "eap-clean-corrupted", "task": "ioi", "level": "node", ...}
```

---

## Tuning EAP-IG

### `ig_steps`

The number of integration steps controls the accuracy vs. speed tradeoff:

| `ig_steps` | Accuracy | Runtime (GPT-2, 128 examples) |
|-----------|---------|-------------------------------|
| 1 | Low (= vanilla EAP) | ~45s |
| 5 | Good (default) | ~3.5 min |
| 10 | High | ~7 min |
| 20 | Very high | ~14 min |

For research-grade circuit discovery, use `ig_steps=10`. For iteration, `ig_steps=3` is often sufficient.

### `n_examples`

More examples → more stable attribution scores. Practical minimums:

| Use case | Minimum `n_examples` |
|----------|---------------------|
| Quick iteration | 32 |
| Standard experiment | 128 |
| Publication-quality | 256–512 |

### `batch_size`

Reduce if you get OOM. Attribution runs one batch at a time, so `batch_size=1` is always safe (just slow).

---

## EAP-IG on Instruction-Tuned Models

For instruction-tuned models, set `chat_template_mode`:

```python
discover_circuit({
    "model": {"name": "meta-llama/Llama-3.2-1B-Instruct"},
    "discovery": {
        "algorithm": "eap-ig",
        "task": "ioi",
        "chat_template_mode": "on",   # wrap in chat template
        "level": "node",
        "data_params": {"num_examples": 128},
    },
    ...
})
```

Or use a downstream-behavior task with `chat_template_mode="auto"` (the default for `boolq`, `mmlu`, etc.):

```python
"discovery": {"algorithm": "eap-ig", "task": "boolq", ...}
# chat_template_mode is auto-resolved based on the model
```

---

## Next Steps

- [ACDC](acdc.md) — greedy edge-pruning alternative
- [IBCircuit](ibcircuit.md) — information-bottleneck approach
- [Stability Tiers](stability-tiers.md) — tier definitions
- [Advanced: Memory Optimization](../advanced/memory-optimization.md) — EAP-IG memory tuning

## References

- Zhang, F. & Nanda, N. (2023). "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods." *ICLR 2024*. [arXiv:2309.16042](https://arxiv.org/abs/2309.16042)
- Hanna, M., Pezzelle, S. & Belinkov, Y. (2024). "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms." (EAP-IG) [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)
- Rezaei Jafari, F., Eberle, O., Khakzar, A. & Nanda, N. (2025). "RelP: Faithful and Efficient Circuit Discovery in Language Models via Relevance Patching." [arXiv:2508.21258](https://arxiv.org/abs/2508.21258)
- Nanda, N., Bloom, J., & others. (2023). "TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models." [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
