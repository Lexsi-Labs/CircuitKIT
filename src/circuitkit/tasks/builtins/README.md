# builtins

The task specifications that ship with CircuitKit. They are registered via `tasks/bootstrap.py`.

## Key modules

- `ioi.py` — `IOITaskSpec`: Indirect Object Identification via entity-swap corruption (thin `GenericTaskSpec` wrapper).
- `ioi_acdc.py` — IOI variant using ACDC data generation with caching.
- `ioi_legacy.py` — deprecated historical IOI implementation, kept for backwards compatibility.
- `double_io.py` — DoubleIO: generalization companion to IOI where both S and IO appear twice.
- `greater_than.py` — `GreaterThanTaskSpec`: numeric greater-than comparison with meaning-altering corruption.
- `sva.py` — `SVATaskSpec`: subject-verb agreement (singular/plural).
- `capital_country.py` — `CapitalCountryTaskSpec`: capital-country factual knowledge.
- `hypernymy.py` — `HypernymyTaskSpec`: hypernymy word relations.
- `gender_bias.py` — `GenderBiasTaskSpec`: gender bias detection.
- `boolq.py` — `BoolQTaskSpec`: BoolQ yes/no reading comprehension via label-swap corruption.
- `winogrande.py` — `WinoGrandeTaskSpec`: WinoGrande cloze task, suffix log-likelihood scoring.
- `winogrande_mc.py` — `WinoGrandeMCTaskSpec`: chat-templatable multiple-choice WinoGrande variant.
- `mmlu.py` — `MMLUTaskSpec`: MMLU with question-stem-replacement corruption.
- `wmdp.py` — `WMDPTaskSpec`: MMLU-style task on WMDP configs (bio/chem/cyber).
- `truthfulqa.py` — `TruthfulQATaskSpec`: TruthfulQA multiple-choice.
- `glue.py` — `GLUETaskSpec`: GLUE tasks (MRPC, QQP, SST-2, RTE, CoLA).
- `gsm8k.py` — `GSM8KTaskSpec`: open-ended math generation circuit discovery.
- `ifeval.py` — `IFEvalTaskSpec`: instruction-following eval prompts (mainly collateral evaluation).

## Public API / entry points

Re-exported task-spec classes: `IOITaskSpec`, `IOITaskSpecLegacy`, `SVATaskSpec`, `GenderBiasTaskSpec`, `CapitalCountryTaskSpec`, `HypernymyTaskSpec`, `GreaterThanTaskSpec`, `MMLUTaskSpec`, `GLUETaskSpec`, `BoolQTaskSpec`, `WinoGrandeTaskSpec`, `WinoGrandeMCTaskSpec`, `TruthfulQATaskSpec`, `IFEvalTaskSpec`, `GSM8KTaskSpec`.

## How it fits

Each module defines a concrete `TaskSpec` (usually a thin `GenericTaskSpec` wrapper) that `bootstrap._bootstrap_builtin_tasks()` registers into the `tasks/registry`, making the task available by name to discovery and evaluation.
