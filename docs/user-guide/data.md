# Data

Circuit discovery runs on data. Before you point CircuitKit at a model, you need to know what kind of data each algorithm expects — because they don't all expect the same thing.

This page explains the data model. The rest of the section covers the [built-in tasks](tasks.md) that ship ready to run, how to [bring your own data](custom-data.md), and how the [corrupt half of a pair gets generated](data-corruption.md) when you don't supply it yourself.

## What discovery needs

CircuitKit's algorithms split into two camps by their data requirement.

**EAP-family algorithms (and ACDC) need a contrastive pair.** They compare two forward passes: a clean prompt and a corrupt one. The clean prompt produces the behavior you care about; the corrupt prompt is a minimally different version that produces a different answer. Attribution comes from the difference between the two runs — patch corrupt activations into the clean forward pass, measure how much the metric moves, and you learn which components carried the signal. No contrast, no signal.

**IBCircuit and CD-T need only the clean prompt.** They find the circuit from the clean forward pass alone (IBCircuit injects its own information-bottleneck noise; CD-T works from clean activations directly). You still supply an answer so the metric has something to score, but there is no corrupt partner to construct.

| Algorithm | Data requirement |
|---|---|
| EAP family (`eap`, `eap-ig`, …) | Paired: clean + corrupt |
| ACDC | Paired: clean + corrupt |
| IBCircuit | Clean only |
| CD-T | Clean only |

If you're running an EAP-family algorithm on clean-only data with no corruption strategy, discovery has nothing to attribute against. CircuitKit warns loudly when this happens (see [Corruption](data-corruption.md#when-the-pair-carries-no-signal)) rather than silently returning a meaningless circuit.

## What makes a dataset contrastive

A dataset is contrastive when a clean example has a natural counterfactual: change one thing and the correct answer flips.

| Task | Clean | Corrupt | Contrastive? |
|---|---|---|---|
| IOI | "When Mary and John went to the store, John gave a drink to" → " Mary" | "…Susan gave a drink to" → different name | Yes — swap the subject name |
| BoolQ | passage + question → " yes" | same template, paired with a " no" example | Yes — flip the label |
| MMLU | "Q: … Answer:" → correct letter | length-matched stem, correct answer at a different letter | Yes — the answer letter differs |
| WinoGrande | sentence + correct fill | same sentence, wrong fill | Yes, but the cue is *after* the blank (see below) |
| GSM8K | word problem → answer digits | operand-swapped problem → different digits | Yes — the final answer differs |
| Plain language modeling | "the cat sat on the" | — | No natural counterfactual |

If your data has correct answers but no wrong ones, it is not contrastive, and EAP-family discovery has no pair to work with. Either add explicit corrupt columns, apply a corruption strategy, or use a clean-only algorithm.

## The metric drives it

The metric is what discovery optimizes, and it decides what shape your answers need to take. CircuitKit's YAML loader exposes three:

| Metric | When to use | Contrastive? | Differentiable? |
|---|---|---|---|
| `logit_diff` | Single-token answers with a correct/incorrect pair | Yes — needs both tokens | Yes |
| `kl` | Multi-token answers; compares the full output distribution | No pair required | Yes |
| `accuracy` | Reporting only | — | **No** |

`logit_diff` is the default and the workhorse. It reads the correct token and the incorrect token off each example and scores the gap between them at the answer position. That is why contrastive tasks work with it out of the box — the corrupt example supplies the incorrect token.

`kl` compares the clean and patched output distributions instead of two specific tokens, so it fits multi-token answers where "the correct next token" is not well defined.

`accuracy` is non-differentiable. You can report it, but EAP-family discovery cannot use it — the attribution backward pass needs a gradient. If you set `metric: accuracy` on a discovery run, CircuitKit warns and you should switch to `logit_diff` or `kl`.

!!! note "These three are the framework-exposed metrics"
    A built-in task may compute its own internal metric that is tuned to its data — `winogrande` uses a suffix log-likelihood, `gsm8k` uses an answer-span NLL. Those are still differentiable and EAP-compatible, but they are chosen inside the task, not selected through the loader's `metric:` key. When you write your own task YAML, `logit_diff`, `kl`, and `accuracy` are your options.

### Not every contrastive task uses `logit_diff`

A dataset can be fully contrastive and still need a different metric. `winogrande` is the clearest case. It is contrastive — correct fill versus wrong fill — but the disambiguating word sits *after* the blank, so a single last-position logit difference measures roughly chance accuracy. Its task spec scores the log-likelihood of the whole suffix span under each filling instead. The lesson: the discovery metric is chosen per task, not applied uniformly.

## Two ways to bring data

CircuitKit has two entry points for custom data, and they use different keys. Don't mix them up.

- **Task YAML** (`circuitkit discover-yaml --task-yaml task.yaml`) builds a `GenericTaskSpec` from a `name` / `source` / `schema` file. The schema maps your column names onto CircuitKit's fields. This is covered in [Bring your own data](custom-data.md).
- **Inline `data` config** (a `data:` block inside a `discover_circuit(config)` dict or YAML) uses `data.type` of `template`, `auto`, or `clean_only`, with `clean_prompt` / `corrupt_prompt` / `clean_answer` / `corrupt_answer` template keys. This is covered in [Bring your own data](custom-data.md#inline-data-config).

## Where to go next

- [Built-in tasks](tasks.md) — the 16 tasks that ship ready to run
- [Bring your own data](custom-data.md) — CSV, JSONL, and HuggingFace via YAML
- [Corruption strategies](data-corruption.md) — generating the corrupt half of a pair
