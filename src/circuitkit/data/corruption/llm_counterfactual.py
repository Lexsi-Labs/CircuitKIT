"""llm_counterfactual — use a small instruction-tuned LLM to generate
the corrupt counterfactual prompt.

The strategy uses a frozen `transformers` causal LM (e.g. Qwen2.5-0.5B,
GPT-2-medium-instruct) to rewrite the clean prompt into a "minimally
different but functionally inverted" counterfactual. Token-alignment
is best-effort: the LLM is prompted to keep the same length, but if
its output drifts the worthiness validator will flag the result.

Length contract: UNKNOWN (depends on the generator's outputs).

This is the strategy of last resort when no symbolic strategy
(entity_swap, token_swap, mcq_choice_swap, ...) applies. It works on
arbitrary user data including instruction-tuning pairs, code, and math.

Reference: Induce-then-Contrast Decoding ([arxiv:2312.15710]),
ParaFusion (arxiv:2404.12010), and the HF Self-Instruct line.
"""

from __future__ import annotations

from typing import Any, Optional

from ..normalized import ContrastiveRecord
from .base import CorruptionResult, CorruptionStrategy, LengthContract, register_strategy

_DEFAULT_PROMPT_TEMPLATE = """\
Rewrite the following input by changing the meaning to something
factually different, but keeping the same overall length and structure.
Output ONLY the rewritten text, no explanation.

Original: {prompt}
Rewritten:"""


@register_strategy("llm_counterfactual")
class LLMCounterfactual(CorruptionStrategy):
    description = (
        "Use a small instruction-tuned LLM to rewrite the clean prompt "
        "into a counterfactual. Pass the generator via "
        "kwargs['generator']: an object with .generate(prompt) -> str. "
        "The 'corrupt_answer' is taken from kwargs['corrupt_answer'] or "
        "left equal to clean_answer (caller's responsibility to flip)."
    )
    length_contract = LengthContract.UNKNOWN

    def corrupt(
        self,
        record: ContrastiveRecord,
        *,
        generator: Optional[Any] = None,
        prompt_template: str = _DEFAULT_PROMPT_TEMPLATE,
        corrupt_answer: Optional[str] = None,
        max_new_tokens: int = 64,
        **_unused: Any,
    ) -> CorruptionResult:
        if generator is None:
            return CorruptionResult(
                None,
                None,
                notes=(
                    "llm_counterfactual requires a generator. Pass "
                    "kwargs['generator']: a callable with "
                    ".generate(prompt, max_new_tokens=...) -> str, "
                    "or use HFLLMCounterfactual which wraps a HF model."
                ),
                succeeded=False,
            )
        prompt = prompt_template.format(prompt=record.clean_prompt)
        try:
            output = generator.generate(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            return CorruptionResult(
                None,
                None,
                notes=f"generator failed: {type(e).__name__}: {e}",
                succeeded=False,
            )
        new_prompt = (output or "").strip()
        if not new_prompt or new_prompt == record.clean_prompt:
            return CorruptionResult(
                None,
                None,
                notes="generator returned empty/identical output",
                succeeded=False,
            )
        return CorruptionResult(
            corrupt_prompt=new_prompt,
            corrupt_answer=corrupt_answer or record.clean_answer,
            notes=(
                "LLM rewrote prompt; corrupt_answer ="
                + (" caller-provided" if corrupt_answer else " same as clean")
            ),
            succeeded=True,
        )


# ---------------------------------------------------------------------------
# Convenience generator: wrap an HF causal LM
# ---------------------------------------------------------------------------


class HFGenerator:
    """Thin wrapper around an HF causal LM exposing a ``.generate(prompt)`` method.

    Example:
        from circuitkit.data.corruption.llm_counterfactual import HFGenerator, LLMCounterfactual
        gen = HFGenerator(model_name='Qwen/Qwen2.5-0.5B-Instruct',
                          device='auto')
        strategy = LLMCounterfactual()
        record_paired = strategy.apply(record, generator=gen)

    Loads the model lazily on first call so import-time is cheap.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
            )
            .to(self.device)
            .eval()
        )

    def generate(self, prompt: str, *, max_new_tokens: int = 64) -> str:
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


__all__ = ["LLMCounterfactual", "HFGenerator"]
