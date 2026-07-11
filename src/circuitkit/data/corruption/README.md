# corruption

Corruption strategies that turn a clean `ContrastiveRecord` into a
contrastive (clean, corrupt) pair suitable for attribution-based discovery.

## Key modules

- `base.py` — `CorruptionStrategy` base + registry (`STRATEGY_REGISTRY`,
  `register_strategy`, `get_strategy`, `list_strategies`), `CorruptionResult`,
  and the `LengthContract` enum (PRESERVE / EXTEND / SHRINK / UNKNOWN / NATIVE).
- `entity_swap.py` — swap a salient noun/entity (spaCy or a built-in name pool).
- `token_swap.py` — length-preserving content-token replacement (Zhang & Nanda STR).
- `resample.py` — pair each clean prompt with another clean prompt (resample ablation).
- `mcq_choice_swap.py` — swap two MCQ choices' contents, keeping letter labels.
- `final_answer_swap.py` — perturb the final math answer (operand or answer-only mode).
- `operand_swap.py` — swap arithmetic operands on shared templates (Stolfo et al. 2023).
- `math_step_corrupt.py` — corrupt one intermediate CoT equation.
- `logical_negation.py` — insert/remove `not` to flip logical sense.
- `profession_swap.py` — swap gender-stereotyped professions (Vig et al. 2020).
- `benign_rewrite.py` — symbolic harmful→benign rewrite via a keyword map.
- `code_syntax_corrupt.py` — corrupt a code signature/docstring (rename, operator flip).
- `instruction_swap.py` — swap directive verbs for instruction-following tasks;
  also exports `audit_instruction_swap_degeneracy`.
- `llm_counterfactual.py` — last-resort counterfactual via a small instruction-tuned LM.
- `template.py` — `TemplateStrategy`: build pairs from placeholder templates
  (explicit / auto_peer modes).
- `template_utils.py` — pure placeholder-parsing / alignment helpers used by
  `template.py` and `data/template.py`.

## Public API / entry points

From `__init__.py` `__all__`: `CorruptionStrategy`, `CorruptionResult`,
`register_strategy`, `get_strategy`, `list_strategies`, `STRATEGY_REGISTRY`,
`InstructionSwap`, `audit_instruction_swap_degeneracy`.

## How it fits

Strategies are the second data stage, running after an adapter: they fill in
`corrupt_prompt` / `corrupt_answer`, and their `length_contract` is checked by
`data.worthiness` so gradient-based discovery sees token-aligned pairs.
