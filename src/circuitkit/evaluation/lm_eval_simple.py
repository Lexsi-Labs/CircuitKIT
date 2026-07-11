import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

# Suppress lm-eval warnings before importing
warnings.filterwarnings("ignore", message=".*pretrained.*model kwarg is not of type.*")
warnings.filterwarnings("ignore", message=".*Passed an already-initialized model.*")
warnings.filterwarnings("ignore", message=".*Overwriting default num_fewshot.*")
logging.getLogger("lm_eval").setLevel(logging.ERROR)
logging.getLogger("lm_eval.models.huggingface").setLevel(logging.ERROR)
logging.getLogger("lm_eval.evaluator").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except Exception:  # pragma: no cover - optional dependency
    evaluator = None  # type: ignore
    HFLM = None  # type: ignore


# Reuse hook builders from the existing harness wrapper to avoid duplication
from .lm_harness import (  # type: ignore  # noqa: E402 - import after intentional pre-import setup
    _build_neuron_hooks,
    _build_node_hooks,
)


def build_hooks_from_artifact(
    pruned_artifact: Union[List[str], Dict[str, Any]]
) -> List[Tuple[str, Any]]:
    """Build forward hooks list from a pruning artifact (nodes list or neuron dict)."""
    if isinstance(pruned_artifact, list):
        return _build_node_hooks(pruned_artifact)
    if isinstance(pruned_artifact, dict):
        mlp = pruned_artifact.get("mlp", {})
        attn = pruned_artifact.get("attn", {})
        return _build_neuron_hooks(mlp, attn)
    raise TypeError(f"Unsupported pruning artifact type: {type(pruned_artifact)}")


def evaluate_lm_eval(
    lens_model: HookedTransformer,
    tasks: List[str],
    *,
    hooks: Optional[List[Tuple[str, Any]]] = None,
    verbosity: str = "WARNING",
    apply_chat_template: Union[bool, str] = "auto",
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate a HookedTransformer on lm-eval tasks via a lightweight HF-like adapter.

    Set hooks to evaluate a pruned view of the model.
    Returns the raw lm-eval results dict.

    Args:
        apply_chat_template: Wrap each task prompt in the model's chat template
            before scoring. Instruction-tuned models score badly on raw task
            text (MMLU/BoolQ/GSM8K/...), so this must be on for them. ``"auto"``
            (default) enables it iff the model's tokenizer ships a
            ``chat_template``; pass ``True``/``False`` to force it.
    """
    if evaluator is None or HFLM is None:
        raise ImportError("lm-eval is not installed. Please install `lm-eval`.")

    class HFLikeModelAdapter(nn.Module):
        """Adapts HookedTransformer to match the HuggingFace interface expected by lm-eval."""

        def __init__(
            self,
            model: HookedTransformer,
            hooks_for_forward: Optional[List[Tuple[str, Any]]] = None,
        ):
            super().__init__()
            self.model = model
            self.tokenizer = model.tokenizer
            # config needs to exist; name is sufficient for many call sites
            try:
                from transformers import AutoConfig

                self.config = AutoConfig.from_pretrained(model.cfg.tokenizer_name)
            except Exception:
                self.config = type("Cfg", (), {"model_type": "causal-lm"})()
            # Ensure device is a proper torch.device for callers that use `.device.type`
            self.device = torch.device(str(model.cfg.device))
            # Best-effort dtype exposure for some integrations
            try:
                self.dtype = next(self.model.parameters()).dtype
            except Exception:
                self.dtype = torch.float32
            self.tie_weights = lambda: self  # no-op, some callers expect this attr
            self._hooks = hooks_for_forward or []

        def forward(self, input_ids=None, attention_mask=None, **forward_kwargs):
            # Ensure tensors are on the right device
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            if self._hooks:
                with self.model.hooks(fwd_hooks=self._hooks):
                    output_logits = self.model(
                        input_ids, attention_mask=attention_mask, **forward_kwargs
                    )
            else:
                output_logits = self.model(
                    input_ids, attention_mask=attention_mask, **forward_kwargs
                )

            # Return an object with a .logits attribute
            class _Out:
                pass

            out = _Out()
            out.logits = output_logits if isinstance(output_logits, torch.Tensor) else output_logits
            return out

        def to(self, *args, **kwargs):
            self.model.to(*args, **kwargs)
            try:
                self.device = next(self.model.parameters()).device
            except Exception:
                self.device = torch.device(str(self.model.cfg.device))
            return self

        def eval(self):
            self.model.eval()
            return self

        def train(self, mode: bool = True):
            self.model.train(mode)
            return self

        def generate(
            self,
            input_ids=None,
            attention_mask=None,
            *,
            max_new_tokens: int = 64,
            do_sample: bool = False,
            temperature: float = 1.0,
            eos_token_id=None,
            pad_token_id=None,
            **gen_kwargs,
        ):
            # Align tensors to device
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # TransformerLens generate API doesn't require attention_mask; pass only supported args
            def _run_generate():
                return self.model.generate(
                    input_ids,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                )

            with torch.no_grad():
                if self._hooks:
                    with self.model.hooks(fwd_hooks=self._hooks):
                        out_tokens = _run_generate()
                else:
                    out_tokens = _run_generate()
            return out_tokens

    model_adapter = HFLikeModelAdapter(lens_model, hooks_for_forward=hooks)
    warnings.filterwarnings("ignore", message="Failed to get model SHA for")

    # Map a few common kwargs to the names lm-eval expects
    mapped_kwargs = {}
    if "fewshot" in kwargs and "num_fewshot" not in kwargs:
        mapped_kwargs["num_fewshot"] = kwargs.get("fewshot")
    if "limit" in kwargs:
        mapped_kwargs["limit"] = kwargs.get("limit")
    if "task_manager" in kwargs:
        mapped_kwargs["task_manager"] = kwargs.get("task_manager")

    hflm = HFLM(pretrained=model_adapter, tokenizer=model_adapter.tokenizer)

    # Instruction-tuned models must receive prompts wrapped in their chat
    # template — raw task text (MMLU/BoolQ/GSM8K/...) confuses them. "auto"
    # triggers iff the model's tokenizer ships a chat_template.
    if apply_chat_template == "auto":
        chat_tmpl = getattr(model_adapter.tokenizer, "chat_template", None)
        apply_chat_template = chat_tmpl is not None
    if apply_chat_template:
        logger.info("Instruction-tuned model detected; applying chat template.")

    import inspect

    _sig = inspect.signature(evaluator.simple_evaluate)
    if "apply_chat_template" in _sig.parameters:
        mapped_kwargs["apply_chat_template"] = apply_chat_template
        if "fewshot_as_multiturn" in _sig.parameters:
            _fewshot = mapped_kwargs.get("num_fewshot") or 0
            mapped_kwargs["fewshot_as_multiturn"] = bool(apply_chat_template) and _fewshot > 0
    elif apply_chat_template:
        logger.warning(
            "Installed lm-eval has no apply_chat_template support; upgrade "
            "lm-eval to score instruction-tuned models correctly."
        )

    # Newer lm-eval versions removed the top-level `verbosity` kwarg in
    # `simple_evaluate` (it now reads from a global lm_eval logger config).
    # Inspect the signature and only forward `verbosity` when it's accepted,
    # so this call stays portable across lm-eval versions.
    if "verbosity" in _sig.parameters:
        mapped_kwargs["verbosity"] = verbosity
    results = evaluator.simple_evaluate(
        model=hflm,
        tasks=tasks,
        **mapped_kwargs,
    )
    return results
