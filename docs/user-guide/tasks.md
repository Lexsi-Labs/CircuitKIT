# Built-in tasks

A task packages a dataset, a metric, and (for paired tasks) a way to build the corrupt half into one registered spec. CircuitKit ships 16 of them. List what's registered:

```python
from circuitkit import list_tasks
print(list_tasks())
```

!!! note "Import from `circuitkit`, not `circuitkit.tasks`"
    `circuitkit.tasks.list_tasks` exists too, but it skips the bootstrap that registers the built-ins. Calling it directly returns an incomplete list unless something else already ran the bootstrap. Import `list_tasks`, `get_task`, and `register_task` from the `circuitkit` package root.

## The 16 tasks

The first seven are synthetic diagnostic tasks — short, template-generated English sentences with a clean single-token answer. The rest wrap real benchmark datasets. "Discovery" is the metric the task uses internally; the loader-exposed metrics (`logit_diff`, `kl`, `accuracy`) are what you select for a custom YAML task, not these.

| Task | What it tests | Discovery data | Discovery metric | Chat template default |
|---|---|---|---|---|
| `ioi` | Indirect object identification (name resolution) | Paired + clean-only | logit_diff | off |
| `greater_than` | Ordinal number comparison | Paired + clean-only | logit_diff | off |
| `sva` | Subject–verb agreement | Paired + clean-only | probability diff | off |
| `hypernymy` | Word-relation (is-a) prediction | Paired + clean-only | probability diff | off |
| `gender_bias` | Gendered-pronoun prediction | Paired + clean-only | logit_diff | off |
| `capital_country` | Country → capital factual recall | Paired + clean-only | logit_diff | off |
| `double_io` | IOI with a second indirect object | Paired + clean-only | logit_diff | off |
| `mmlu` | 4-choice knowledge QA | Paired + clean-only | logit_diff | auto |
| `glue` | Text classification (MRPC/QQP/SST-2/RTE/CoLA) | Paired + clean-only | logit_diff | auto |
| `wmdp` | Hazardous-knowledge multiple choice | Paired + clean-only | logit_diff | auto |
| `boolq` | Boolean (yes/no) reading QA | Paired + clean-only | logit_diff | auto |
| `winogrande` | Cloze commonsense (fill the blank) | Paired + clean-only | suffix log-likelihood | off |
| `winogrande_mc` | WinoGrande as A/B multiple choice | Paired + clean-only | logit_diff | auto |
| `truthfulqa` | Truthful vs. plausible-false answers | Paired + clean-only | logit_diff | auto |
| `gsm8k` | Grade-school math word problems | Paired + clean-only | answer-span NLL | auto |
| `ifeval` | Instruction following | **No discovery** | — | auto |

A few things worth reading off this table.

**Diagnostic tasks default to `chat_template_mode: off`.** `ioi`, `sva`, `greater_than`, and the rest are minimal English sentences, not chat prompts. Wrapping them in a chat template would misalign the tokens. The benchmark-wrapping tasks default to `auto` because they're meant to run as real downstream behaviors on whatever model you load.

**Two tasks don't use `logit_diff`.** `winogrande`'s disambiguating word comes after the blank, so a last-position logit difference measures chance; it scores the log-likelihood of the whole suffix span instead. `gsm8k` is open-ended generation, scored by a differentiable NLL over the answer span. Both are still EAP-compatible.

**`ifeval` is not a discovery task.** It's registered for collateral evaluation only. Calling `discover_circuit` with `task: ifeval` raises a `ValueError`. Use it as a downstream benchmark, not a discovery target.

**"Clean-only" means IBCircuit and CD-T also work.** Every discovery-capable task here builds a paired dataloader for the EAP family and ACDC, and also supports the clean-only path (IBCircuit / CD-T). You choose which by the `algorithm` you pass.

## Loading a task

```python
from circuitkit import get_task

spec = get_task("ioi")
print(spec.name)               # "ioi"
print(spec.chat_template_mode) # "off"
```

Task specs load lazily — the underlying data downloads or generates on first use, not at import.

## Chat template policy

Every task carries a `chat_template_mode` that decides whether prompts get wrapped in the model's chat template before tokenization.

| Mode | Behavior |
|---|---|
| `auto` | Wrap only if the tokenizer ships a `chat_template` (i.e. the model is instruction-tuned) |
| `on` | Always wrap |
| `off` | Always use raw text |

Instruction-tuned models (Llama-Instruct, Gemma-IT, Qwen-Instruct) expect chat-formatted prompts, so downstream tasks default to `auto`. Diagnostic tasks default to `off`.

Override per run:

```python
discover_circuit({
    "discovery": {"task": "ioi", "chat_template_mode": "on", ...},
    ...
})
```

See [Chat templates](../advanced/chat-templates.md) for the full resolution rules.

## Next steps

- [Bring your own data](custom-data.md) — CSV, JSONL, and HuggingFace via YAML
- [Corruption strategies](data-corruption.md) — how the corrupt half is generated
- [YAML configuration](../cli/yaml-config.md) — the full task-YAML schema reference
