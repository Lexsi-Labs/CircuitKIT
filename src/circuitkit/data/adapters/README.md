# adapters

Dataset-shape adapters that convert a raw dataset of a known shape into a
`NormalizedDataset[ContrastiveRecord]`, independent of any corruption strategy.

## Key modules

- `base.py` — `DataAdapter` abstract base + registry (`ADAPTER_REGISTRY`,
  `register_adapter`, `get_adapter`, `list_adapters`); each adapter declares a
  `shape`, a `fits()` sniff-check, and an `adapt()` normalizer.
- `pairwise.py` — natively-paired datasets (CrowS-Pairs, StereoSet
  intersentence) with `sent_more` / `sent_less` columns; also hosts shared
  `_iter_rows` / `_peek_columns` helpers.
- `mcq.py` — multiple-choice datasets (MMLU, ARC, HellaSwag, BBQ,
  CommonSenseQA, …); lays out choices and predicts the answer letter.
- `instruction.py` — single-turn instruction data (Alpaca / Dolly / OASST):
  `{instruction, input?, output}` with no native pair.
- `conversational.py` — ShareGPT / OpenAI multi-turn chat; one
  (user-turn, assistant-reply) pair per record.
- `math.py` — math word problems (GSM8K / MATH / AQuA / GSM-Plus); extracts the
  final numeric answer and keeps the solution trace in meta.
- `code.py` — code-generation datasets (HumanEval / MBPP / BigCodeBench);
  signature + docstring as prompt, first solution token as answer.
- `forget_retain.py` — unlearning splits (TOFU / MUSE / WMDP) tagged
  `meta['split'] = 'forget' | 'retain'`.
- `safety_prompt.py` — harm-prompt datasets (AdvBench / HarmBench / Sorry-Bench /
  JailbreakBench / TruthfulQA); pairs refusal vs compliance answer tokens.

## Public API / entry points

From `__init__.py` `__all__`: `DataAdapter`, `register_adapter`, `get_adapter`,
`list_adapters`, `ADAPTER_REGISTRY`.

## How it fits

Adapters are the first stage of the data pipeline: `auto_detect` selects one by
shape, and its output feeds the `corruption/` strategies and worthiness checks.
