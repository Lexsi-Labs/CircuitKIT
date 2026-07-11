# ACDC

**Automatic Circuit Discovery** (ACDC) discovers circuits by greedy edge removal rather than gradient attribution. It starts with the full model and removes edges that do not meet an importance threshold, stopping when no further removals are possible.

**Stability tier:**  Experimental — validated on GPT-2 IOI; may fail or slow on larger models.

---

## How It Works

ACDC operates on the **edge graph** of the computation:

1. Start with all edges included (full model)
2. For each edge, compute: how much does removing it change the metric?
3. Remove edges whose contribution is below the current tao (threshold) value
4. Repeat until no edges can be removed
5. The remaining edges define the circuit

This greedy approach tends to produce **smaller and sparser** circuits than EAP-based methods, which score edges globally and take the top-K. ACDC is more conservative — it only removes an edge if it can demonstrate the edge is unimportant.

---

## Usage

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "gpt2", "precision": "float32"},
    "discovery": {
        "algorithm": "acdc",
        "task": "ioi",
        "level": "node",
        "data_params": {"num_examples": 64},
        # Optional: scope the tao (threshold) grid ACDC sweeps.
        # Omit these to use the backend defaults.
        "tao_bases": [1, 3, 5, 7, 9],
        "tao_exps": [-5, -4, -3, -2],
        "faithfulness_target": "kl_div",  # or "mse"
    },
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

!!! warning "ACDC emits a UserWarning"
    Since ACDC is Experimental-tier, `discover_circuit` emits a `UserWarning` when you request it. This is expected behavior.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tao_bases` | `[1, 3, 5, 7, 9]` | Bases for the tao (threshold) grid ACDC sweeps. |
| `tao_exps` | `[-5, -4, -3, -2]` | Exponents for the tao grid; each edge is scored by the smallest tao at which it is pruned. |
| `faithfulness_target` | `"kl_div"` | Metric ACDC optimizes the circuit for; `"kl_div"` or `"mse"`. |
| `num_examples` | `128` | Example count under `data_params`, used to build the batch ACDC patches over. |

ACDC does not use a single threshold — it sweeps a grid of tao values built from the product of `tao_bases` and `tao_exps` (defaults give 20 sweeps). Narrow the grid to trim runtime:

1. Keep the defaults for a full sweep.
2. To go faster, pass fewer values, e.g. `tao_bases=[1]` and `tao_exps=[-3]` for a single tao.
3. Smaller taos are stricter (prune more); larger taos keep more edges.

---

## ACDC vs. EAP-IG

| Aspect | ACDC | EAP-IG |
|--------|------|--------|
| Approach | Greedy removal | Global attribution |
| Circuit size | Smaller (minimal) | Larger (top-K by score) |
| Runtime | Slow (O(edges × iterations)) | Fast (O(n_examples × ig_steps)) |
| Model size limit | GPT-2 scale practical | Validated to 4B params |
| Tier |  Experimental |  Stable |

For most use cases, start with EAP-IG and compare ACDC only if you need a minimal circuit or are specifically studying algorithm differences.

---

## When to Use ACDC

- You need the **smallest possible circuit** (ACDC is more conservative)
- You're working at **GPT-2 scale** and have time for a slower search
- You're **comparing algorithm outputs** in a multi-method study

---

## Known Limitations

- **Slow** for large models or large datasets — each iteration requires multiple forward passes
- **GPT-2 validated only** — may produce degenerate results or fail on GQA architectures
- **Greedy local optima** — ACDC can get stuck removing edges that are locally unimportant but jointly important

---

## Next Steps

- [EAP Family](eap.md) — faster, validated alternative
- [IBCircuit](ibcircuit.md) — information-theoretic alternative
- [Stability Tiers](stability-tiers.md) — understanding tier constraints

## References

- Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability." *NeurIPS 2023*. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
