# FILE: circuitkit/evaluation/weight_based_eval.py

import warnings
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
from transformer_lens import HookedTransformer


import logging

logger = logging.getLogger(__name__)

try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except Exception:  # pragma: no cover - optional dependency
    evaluator = None  # type: ignore
    HFLM = None  # type: ignore

from circuitkit.applications.pruning.weight_pruner import create_weight_pruned_model


def evaluate_lm_eval_weight_based(
    lens_model: HookedTransformer,
    tasks: List[str],
    *,
    pruned_artifact: Union[List[str], Dict[str, Any]],
    verbosity: str = "WARNING",
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate a HookedTransformer on lm-eval tasks using weight-based pruning.

    This approach creates a new model with pruned weights instead of using hooks.
    This is more efficient and doesn't require use_attn_result=True.

    Args:
        lens_model: The original HookedTransformer model
        tasks: List of task names to evaluate
        pruned_artifact: Either a list of node names or a dict with MLP/attention neuron info
        verbosity: Verbosity level for lm-eval
        **kwargs: Additional arguments passed to lm-eval

    Returns:
        Dictionary containing evaluation results
    """
    if evaluator is None or HFLM is None:
        raise ImportError("lm-eval is not installed. Please install `lm-eval`.")

    class HFLikeModelAdapter(nn.Module):
        """Adapts HookedTransformer to match the HuggingFace interface expected by lm-eval."""

        def __init__(self, model: HookedTransformer):
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

        def forward(self, input_ids=None, attention_mask=None, **forward_kwargs):
            # Ensure tensors are on the right device
            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            output_logits = self.model(input_ids, attention_mask=attention_mask, **forward_kwargs)

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
                out_tokens = _run_generate()
            return out_tokens

    # Create pruned model
    logger.info("Creating weight-pruned model...")
    pruned_model = create_weight_pruned_model(lens_model, pruned_artifact)

    # Create adapter for the pruned model
    model_adapter = HFLikeModelAdapter(pruned_model)
    warnings.filterwarnings("ignore", message="Failed to get model SHA for")

    # Map a few common kwargs to the names lm-eval expects
    mapped_kwargs = {}
    if "fewshot" in kwargs and "num_fewshot" not in kwargs:
        mapped_kwargs["num_fewshot"] = kwargs.get("fewshot")
    if "limit" in kwargs:
        mapped_kwargs["limit"] = kwargs.get("limit")
    if "task_manager" in kwargs:
        mapped_kwargs["task_manager"] = kwargs.get("task_manager")
    if "confirm_run_unsafe_code" in kwargs:
        mapped_kwargs["confirm_run_unsafe_code"] = kwargs.get("confirm_run_unsafe_code")

    results = evaluator.simple_evaluate(
        model=HFLM(pretrained=model_adapter, tokenizer=model_adapter.tokenizer),
        tasks=tasks,
        verbosity=verbosity,
        **mapped_kwargs,
    )

    # `pruned_model` / `model_adapter` are local to this call and not part of
    # `results` - simple_evaluate() has already consumed them by this point.
    # Drop our references explicitly so this function's peak GPU usage
    # doesn't linger into whatever the caller does next (e.g.
    # compare_original_vs_pruned_weight_based()'s second call, which builds
    # another full pruned copy). `lens_model` (the caller's original model)
    # is untouched here - create_weight_pruned_model() returns a new object,
    # `lens_model` itself was never assigned to either of these names.
    del pruned_model, model_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def compare_original_vs_pruned_weight_based(
    model: HookedTransformer,
    pruned_artifact: Union[List[str], Dict[str, Any]],
    tasks: List[str],
    **kwargs,
) -> Dict[str, Any]:
    """Compare original model vs weight-pruned model on lm-eval tasks.

    Each call to evaluate_lm_eval_weight_based() below builds its own full
    copy of `model` (see create_weight_pruned_model) for the duration of
    that call. Since the two calls are sequential and neither needs the
    other's copy, we force collection and clear the CUDA cache between them
    so the second copy's peak memory doesn't have to coexist with whatever
    is left over from the first. `model` itself (the caller's original) is
    never touched here or in evaluate_lm_eval_weight_based - only the local
    pruned copies and the returned result dicts are involved.

    Args:
        model: The original HookedTransformer model
        pruned_artifact: Either a list of node names or a dict with MLP/attention neuron info
        tasks: List of task names to evaluate
        **kwargs: Additional arguments passed to lm-eval

    Returns:
        Dictionary with 'original' and 'pruned' results
    """
    logger.info("Evaluating original model...")
    original_results = evaluate_lm_eval_weight_based(
        model, tasks, pruned_artifact=[], **kwargs  # Empty artifact = no pruning
    )

    # evaluate_lm_eval_weight_based() already drops its own internal
    # pruned_model/model_adapter references before returning (see that
    # function), but force a collection pass here too before building the
    # second copy below, rather than relying on whenever the allocator next
    # decides to reclaim it.
    import gc


    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Evaluating weight-pruned model...")
    pruned_results = evaluate_lm_eval_weight_based(
        model, tasks, pruned_artifact=pruned_artifact, **kwargs
    )

    return {"original": original_results, "pruned": pruned_results}
