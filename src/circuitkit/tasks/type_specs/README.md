<!-- circuitkit-logo -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../../../docs/assets/circuitkit-logo-white.png">
    <img src="../../../../docs/assets/circuitkit-logo-black.png" width="200" alt="CircuitKIT">
  </picture>
</p>

# type_specs

Reusable per-task-type base specs that subclass `GenericTaskSpec`. Each one pairs a contrastive-pair strategy with a metric, so there is no `task_type` branching.

## Key modules

- `qa_spec.py` — `QASpec` (binary/extractive/simple QA); answer-flip or entity-swap corruption, logit-difference metric.
- `mcq_spec.py` — `MCQSpec`: multiple-choice (MMLU, TruthfulQA MC); correct/incorrect choice-swap, logit-difference metric.
- `classification_spec.py` — `ClassificationSpec`: GLUE-style classification; label-flip corruption, logit-difference metric.
- `generation_spec.py` — `GenerationSpec`: open generation / instruction following; instruction-swap corruption, NLL metric (no contrastive pair).
- `summarization_spec.py` — `SummarizationSpec`: summarization; article entity-swap corruption, NLL over the summary span.
- `translation_spec.py` — `TranslationSpec`: translation; source entity-swap corruption, NLL over the target span.

## Public API / entry points

`QASpec`, `MCQSpec`, `ClassificationSpec`, `GenerationSpec`, `SummarizationSpec`, `TranslationSpec`.

## How it fits

The auto-schema detector and `hf_factory` select one of these specs for an arbitrary HuggingFace dataset based on its detected task type, giving each dataset the correct corruption strategy and metric without task-specific code.
