# Pillar 6: Generalization

Generalization tests whether a circuit discovered for one task also explains a related task. If the circuit represents a genuine computational mechanism (e.g., "name-mover heads"), it should transfer partially to tasks that share that mechanism.

!!! warning "Preliminary status"
    Pillar 6 is implemented but **not yet validated at scale**. It has not been run on safety datasets or in a production sweep. Treat its scores as preliminary. This pillar is automatically skipped if `target_task` is not specified.

---

## The Measurement

1. Discover a circuit on the **source task** (e.g., IOI)
2. Evaluate the same circuit (via Pillar 1 patching) on the **source task** to get its `source_score`
3. Evaluate the same circuit on a **target task** (e.g., SVA) to get its `target_score`
4. Report the **transfer ratio** = `target_score / source_score`

If the circuit generalizes, its target-task score stays close to its source-task score, so the transfer ratio approaches 1, even though the circuit was discovered for a different task.

---

## Running Pillar 6

Pillar 6 **requires** `target_task` to be specified:

```python
# Via Pipeline
pipe.evaluate(
    pillars=["generalization"],
    target_task="sva",         # required — the task to generalize to
    n_examples=256,
)
print(pipe.report.generalization)
# {
#   "source_task": "ioi",
#   "target_task": "sva",
#   "source_score": 0.83,             # circuit's patching score on the source task
#   "target_score": 0.61,             # circuit's patching score on the target task
#   "transfer_ratio": 0.73,           # target_score / source_score
#   "transfer_delta": 0.22,           # source_score - target_score
#   "relative_transfer_drop": 0.27,   # transfer_delta / source_score
# }
```

If `target_task` is not set, Pillar 6 is silently skipped.

---

## Choosing Target Tasks

Transfer is most meaningful between tasks that share computational mechanisms:

| Source task | Related target tasks | Shared mechanism |
|-------------|---------------------|-----------------|
| `ioi` | `sva`, `gender_bias`, `double_io` | Name-mover / indirect-object heads |
| `boolq` | `truthfulqa`, `mmlu` | Factual QA reasoning |
| `capital_country` | `hypernymy`, `ioi` | Lookup / relation heads |
| `winogrande` | `ioi`, `sva` | Coreference resolution |

---

## Interpreting the Results

| Transfer Ratio | Interpretation |
|:---:|----------------|
| ≥ 0.70 | **Good transfer** — circuit captures a task-general mechanism |
| 0.50 – 0.70 | **Partial transfer** — some overlap but task-specific components also matter |
| < 0.50 | **Poor transfer** — the circuit is task-specific |

**Transfer ratio** = `target_score / source_score`. A high transfer ratio means the circuit's patching score on the target task is close to its score on the source task.

**Note:** Even partial transfer (0.5 – 0.7) is meaningful — it indicates the algorithm found components that are not purely task-specific.

---

## Low-Level API

At the low level, `run_full_faithfulness` needs **both** `target_task_spec` **and** `target_dataloader`. Passing only the spec makes Pillar 6 log a warning and skip — `report.generalization` comes back `None`. (The high-level `pipe.evaluate(target_task=...)` path builds the target dataloader for you, which is why the Pipeline snippet above only needs `target_task`.)

```python
from circuitkit.evaluation.full import run_full_faithfulness
from circuitkit.tasks import get_task

target_spec = get_task("sva")
# Build the target dataloader yourself — omitting it silently skips Pillar 6.
target_dataloader = target_spec.build_dataloader(model, discovery_cfg, device)

report = run_full_faithfulness(
    model, graph, source_task_spec, discovery_cfg,
    pillars=["generalization"],
    target_task_spec=target_spec,
    target_dataloader=target_dataloader,
)
print(report.generalization)
```

---

## Next Steps

- [Downstream Benchmarking](benchmarking.md) — lm-eval integration for extrinsic evaluation
- [Framework Overview](framework.md) — all 6 pillars
- [Causal Patching (Pillar 1)](causal-patching.md) — the core patching measurement used in transfer
