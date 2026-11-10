# Pillar 1: Causal Patching

Causal patching is the **fastest** faithfulness test and the most commonly reported metric. It answers the question: *if we patch back only the circuit nodes to their clean activation values (leaving all other nodes at their corrupted values), does the model's behavior recover?*

---

## The Measurement

**Setup:**
1. Run the model on the corrupted input $x^*$ — all activations are now "corrupted"
2. Patch circuit nodes back to their clean activation values (from $x$)
3. Measure the logit-difference metric $f$ at the output

If the circuit is faithful, the metric after patching should be close to the full-model clean metric:

$$\text{patching score} = \frac{f(\text{corrupted} + \text{circuit nodes patched}) - f(x^*)}{f(x) - f(x^*)}$$

A score of 1.0 means the circuit alone can fully recover the clean behavior. A score of 0.0 means patching back the circuit has no effect.

---

## Running Pillar 1

```python
# Via Pipeline
pipe.evaluate(pillars=["patching"], n_examples=256)
print(pipe.report.patching_score)

# Via evaluate_circuit
results = evaluate_circuit({
    ...,
    "eval": {"pillars": ["patching"]},
})
print(results.patching_score)  # Pillar 1 patching score
```

Low-level:

```python
from circuitkit.evaluation.evaluate import evaluate_graph

# Apply circuit to graph
graph.apply_topn(n=30, level="node")  # keep top 30 nodes (default level="edge")

raw_metric = evaluate_graph(
    model,
    graph,
    dataloader,
    metrics=[metric_fn],
    intervention="patching",   # the patching intervention
    quiet=True,
)
```

!!! note "`evaluate_graph` returns the raw metric, not the normalized score"
    `evaluate_graph` gives you the raw per-sample metric under the patching intervention — **not** the 0-1 faithfulness ratio. The normalization against the clean and corrupt baselines happens inside `Pillar1_CausalPatching.run`, which is what populates `report.patching_score`. If you call `evaluate_graph` directly, you still need to normalize yourself: `(raw_metric - corrupt_baseline) / (clean_baseline - corrupt_baseline)`.

---

## The raw ratio, and scores outside 0 to 1

`report.patching_score` and `report.ablation_score` are the ratio clipped to the range 0 to 1.
The unclipped value is kept next to them:

```python
report = ck.faithfulness(model, circuit, "ioi")
report.patching_score        # clipped to [0, 1]
report.patching_raw_ratio    # (circuit - corrupt) / (clean - corrupt), not clipped
report.ablation_raw_ratio    # same, for Pillar 2
report.summary()             # includes pillar_1_patching_raw_ratio and pillar_2_ablation_raw_ratio
```

The raw ratio is also in `report.metadata["patching_raw_ratio"]` and
`report.metadata["ablation_raw_ratio"]`, which is where the JSON report stores it, and each
pillar's result dict has `raw_ratio` and an `out_of_range` flag.

A raw ratio can leave the range in two ways:

- **Above 1**: the circuit scores higher on the metric than the full model. Removing components
  can raise a logit difference, so this does not mean the circuit is "more than faithful".
- **Below 0**: the circuit scores lower than the corrupt baseline. Zero ablation often does this,
  because zeroed activations push the model somewhere neither the clean nor the corrupt run goes.

When that happens the pillar logs a warning, and the printed report says so:

```text
PILLAR 2: Ablation Faithfulness
  Score: 0.0000  (OUT OF RANGE: raw ratio -0.7333 is below 0: the circuit scores lower than the corrupt baseline, and the score is floored at 0)
```

Read the raw ratio whenever a score is exactly 0.0 or 1.0. Two circuits that both clip to the same
end can be far apart, and a clipped 1.0 is not evidence that a pruned checkpoint still does the
task. The values are `None` for reports that did not record a ratio.

### A control circuit to compare against

A faithfulness score means more next to a deliberately bad circuit of the same size. Reverse the
ranking, so the components the algorithm scored highest are the ones removed:

```python
from circuitkit.circuit import Circuit

top = max(abs(v) for v in circuit.scores.values())
control = Circuit(circuit.nodes, {k: top - abs(v) for k, v in circuit.scores.items()},
                  task="ioi", algorithm="eap", level="node")
bad = ck.faithfulness(model, control, "ioi")
```

Measured on GPT-2, `ioi`, `eap`, 32 examples, seed 0, CPU. Raw ratios; the clipped score is in
brackets where it differs:

| Circuit | Pruned | Pillar 1 patching | Pillar 2 zero ablation |
|---|---|:---:|:---:|
| discovered | 30% | 0.931 | -0.155 (0.0) |
| reversed control | 30% | -0.015 (0.0) | -0.733 (0.0) |
| discovered | 80% | 0.834 | -0.308 (0.0) |
| reversed control | 80% | -0.010 (0.0) | -0.329 (0.0) |

Patching separates the discovered circuit from the control by a wide margin. Zero ablation does
not on this model and task: both circuits fall below the corrupt baseline, and at 80% pruned the
two raw ratios are nearly equal. Do not use the zero-ablation score alone to accept a GPT-2 `ioi`
circuit; use patching and the control comparison. Mean ablation was not measured here.

### Where the `ioi` and `double_io` prompts end

Both tasks score the logit difference at the last prompt token, the one that predicts the name
("... gave a drink to"). The cached prompt text stops there. Before this was corrected the prompt
kept the answer name at its end (for `ioi` on every row, with a second BOS token at the start; for
`double_io` on every row shorter than the longest, followed by padding tokens written out as
text), so the metric was read at the name itself. On GPT-2, 32 prompts, seed 0, the mean logit
difference is:

| Task | At the name (before) | At the predicting position (now) |
|---|:---:|:---:|
| `ioi` | 1.06 | 3.87 |
| `double_io` | 1.51 | 2.21 |

Every `ioi` and `double_io` score measured on the paired-text path (the EAP family and CD-T, and
`ck.faithfulness` on those tasks) before the correction is on the old position and is not
comparable with new ones. ACDC and IBCircuit already used the predicting position. Finetuning
data built from the discovery cache also changes: it used to append the answer to a prompt that
already ended with it.

Cache files are now named `ioi_prompt_<model>_<n>_seed<seed>.csv` and
`double_io_prompt_<model>_<n>_seed<seed>.csv`. Files written under the old names
(`ioi_<model>_...`, `double_io_<model>_...`) are no longer read and can be deleted.

---

## Interpreting the Score

| Score | Interpretation |
|-------|---------------|
| ≥ 0.85 | **Strong.** The circuit accounts for ≥ 85% of the behavior. |
| 0.70 – 0.85 | **Moderate.** The circuit captures the main computation but misses some components. |
| 0.50 – 0.70 | **Weak.** A meaningful portion of the behavior is outside the circuit. |
| < 0.50 | **Poor.** The circuit does not explain the behavior well. |

**Typical values in the CircuitKIT audit** (the two IOI rows were measured before the scored
position was corrected, see above, and have not been re-measured):

| Model | Task | Algorithm | Pillar 1 Score |
|-------|------|-----------|:-----------:|
| GPT-2 | IOI | `eap-ig` | 0.83 |
| GPT-2 | SVA | `eap-ig` | 0.79 |
| Llama-1B | MMLU | `eap-ig` | 0.71 |
| Gemma-2B | IOI | `eap-ig` | 0.76 |

---

## What Causal Patching Tests (and Does Not Test)

**Tests:**
- Whether the circuit nodes are sufficient to reproduce the behavior (sufficiency)
- The circuit's ability to carry the relevant information

**Does NOT test:**
- Circuit *sufficiency* under ablation of out-of-circuit nodes (that's Pillar 2: Ablation)
- Whether the circuit is stable across seeds (Pillar 3)
- Whether the circuit is better than a random selection of the same size (Pillar 5)

A circuit can pass Pillar 1 and fail Pillar 2 if there are other components outside the circuit that *also* carry the information: the model has redundancy.

---

## Next Steps

- [Ablation (Pillar 2)](ablation.md) — testing necessity
- [Baselines (Pillar 5)](baselines.md) — comparing to random selection
- [Framework Overview](framework.md) — all 6 pillars
