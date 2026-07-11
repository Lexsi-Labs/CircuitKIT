import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch as t

logger = logging.getLogger(__name__)

try:
    from lm_eval import evaluator
    from lm_eval.api.model import LM

    try:
        # Optional newer APIs
        from lm_eval import tasks as lm_tasks  # type: ignore
    except Exception:
        lm_tasks = None  # type: ignore
    try:
        from lm_eval.evaluator import TaskManager  # type: ignore
    except Exception:
        TaskManager = None  # type: ignore
    try:
        import lm_eval as _lm_eval_mod  # type: ignore

        _LMEVAL_VERSION = getattr(_lm_eval_mod, "__version__", "unknown")
    except Exception:
        _LMEVAL_VERSION = "unknown"
except Exception:  # pragma: no cover - optional dependency
    LM = object  # type: ignore
    evaluator = None  # type: ignore

from transformer_lens import HookedTransformer

from circuitkit.applications.pruning.neuron_pruner import zero_attn_neuron_hook, zero_neuron_hook
from circuitkit.applications.pruning.node_pruner import zero_head_hook, zero_mlp_hook


def _build_node_hooks(nodes_to_prune: List[str]) -> List[Tuple[str, Any]]:
    hooks: List[Tuple[str, Any]] = []
    import re
    from functools import partial

    for node_name in nodes_to_prune:
        attn_match = re.match(r"A(\d+)\.(\d+)", node_name)
        if attn_match:
            layer_idx, head_idx = int(attn_match.group(1)), int(attn_match.group(2))
            hooks.append(
                (
                    f"blocks.{layer_idx}.attn.hook_result",
                    partial(zero_head_hook, head_index=head_idx),
                )
            )
            continue

        mlp_match = re.match(r"MLP (\d+)", node_name)
        if mlp_match:
            layer_idx = int(mlp_match.group(1))
            hooks.append((f"blocks.{layer_idx}.hook_mlp_out", zero_mlp_hook))

    return hooks


def _build_neuron_hooks(
    pruned_mlp_neurons: Dict[int, List[int]],
    pruned_attn_neurons: Dict[Tuple[int, int], List[int]],
) -> List[Tuple[str, Any]]:
    hooks: List[Tuple[str, Any]] = []
    from functools import partial

    for layer_idx, neuron_indices in pruned_mlp_neurons.items():
        hook_point = f"blocks.{layer_idx}.hook_mlp_out"
        hooks.append(
            (
                hook_point,
                (
                    t.jit.script(partial(zero_neuron_hook, neuron_indices=neuron_indices))
                    if False
                    else partial(zero_neuron_hook, neuron_indices=neuron_indices)
                ),
            )
        )

    for (layer_idx, head_idx), neuron_indices in pruned_attn_neurons.items():
        hook_point = f"blocks.{layer_idx}.attn.hook_result"
        hooks.append(
            (
                hook_point,
                partial(zero_attn_neuron_hook, head_index=head_idx, neuron_indices=neuron_indices),
            )
        )

    return hooks


class HookedTransformerLM(LM):
    """LM Evaluation Harness-compatible wrapper around a TransformerLens HookedTransformer.

    This wrapper optionally applies pruning hooks on forward passes so we can evaluate
    the pruned behavior on standard benchmarks.
    """

    def __init__(
        self,
        model: HookedTransformer,
        hooks: Optional[List[Tuple[str, Any]]] = None,
        max_gen_toks_default: int = 64,
    ) -> None:
        if evaluator is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "lm-eval is not installed. Please install `lm-eval` to use harness evaluation."
            )
        self.model = model
        self.hooks = hooks or []
        self.max_gen_toks_default = max_gen_toks_default
        # Use underlying HF tokenizer exposed by TransformerLens
        self.tokenizer = model.tokenizer
        # Minimal distributed attributes expected by the harness
        self._rank = 0
        self._world_size = 1

    @property
    def eot_token_id(self) -> Optional[int]:  # type: ignore[override]
        # Harness uses this for chat models; for causal LMs, EOS is fine
        return getattr(self.tokenizer, "eos_token_id", None)

    def _device(self) -> t.device:
        return t.device(self.model.cfg.device)

    # Harness may query these for distributed eval; keep single-process defaults
    @property
    def rank(self) -> int:  # type: ignore[override]
        return self._rank

    @property
    def world_size(self) -> int:  # type: ignore[override]
        return self._world_size

    @property
    def max_batch_size(self) -> int:  # type: ignore[override]
        # Conservative default for large models; overridden by harness if needed
        return 1

    # --- Tokenization API ---
    def tok_encode(self, string: str, add_special_tokens: bool = False) -> List[int]:  # type: ignore[override]
        ids = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        return list(map(int, ids))

    def tok_decode(self, tokens: Sequence[int]) -> str:  # type: ignore[override]
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True)

    # --- Core scoring helpers ---
    def _forward_logits(self, input_ids: t.Tensor) -> t.Tensor:
        with t.no_grad():
            if self.hooks:
                with self.model.hooks(fwd_hooks=self.hooks):
                    logits = self.model(input_ids)
            else:
                logits = self.model(input_ids)
        return logits

    def _compute_loglikelihood(
        self, context_ids: List[int], continuation_ids: List[int]
    ) -> Tuple[float, bool]:
        # Build full sequence ids
        full_ids = context_ids + continuation_ids
        if len(full_ids) < 2:
            return 0.0, True

        input_ids = t.tensor([full_ids], dtype=t.long, device=self._device())
        logits = self._forward_logits(input_ids)[:, :-1, :]  # shift for next token prediction
        labels = input_ids[:, 1:]

        ctx_len = len(context_ids)
        cont_len = len(continuation_ids)
        start = max(ctx_len - 1, 0)  # prediction for first cont token at position ctx_len
        end = start + cont_len

        # Select logits for continuation positions
        cont_logits = logits[:, start:end, :]
        cont_labels = labels[:, start:end]
        log_probs = t.nn.functional.log_softmax(cont_logits, dim=-1)
        token_logprobs = t.gather(log_probs, 2, cont_labels.unsqueeze(-1)).squeeze(-1)
        sum_logprob = float(token_logprobs.sum().item())

        # Greedy check
        greedy_tokens = cont_logits.argmax(dim=-1)
        is_greedy = bool(t.equal(greedy_tokens, cont_labels))
        return sum_logprob, is_greedy

    # --- LM API ---
    def loglikelihood(self, requests):  # type: ignore[override]
        outputs: List[Tuple[float, bool]] = []
        for req in requests:
            # Support both tuple and Instance formats
            if hasattr(req, "args"):
                args = getattr(req, "args")
                context, continuation = args[0], args[1]
            else:
                context, continuation = req  # type: ignore[misc]
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)
            sum_logprob, is_greedy = self._compute_loglikelihood(ctx_ids, cont_ids)
            outputs.append((sum_logprob, is_greedy))
        return outputs

    def loglikelihood_rolling(self, requests):  # type: ignore[override]
        # Simple rolling loglikelihood: sum NLL over full string
        outputs: List[float] = []
        for req in requests:
            if hasattr(req, "args"):
                text = getattr(req, "args")[0]
            else:
                text = req  # type: ignore[assignment]
            ids = self.tok_encode(text)
            if len(ids) < 2:
                outputs.append(0.0)
                continue
            input_ids = t.tensor([ids], dtype=t.long, device=self._device())
            logits = self._forward_logits(input_ids)[:, :-1, :]
            labels = input_ids[:, 1:]
            log_probs = t.nn.functional.log_softmax(logits, dim=-1)
            token_logprobs = t.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
            outputs.append(float(token_logprobs.sum().item()))
        return outputs

    def greedy_until(self, requests):  # type: ignore[override]
        return self.generate_until(requests)

    def generate_until(self, requests):  # type: ignore[override]
        generations: List[str] = []
        for req in requests:
            # Accept both tuple and Instance formats
            if hasattr(req, "args"):
                args = getattr(req, "args")
                # Common patterns: (context, until) or just (context,)
                if len(args) >= 2:
                    context, until = args[0], args[1]
                else:
                    context = args[0]
                    until = getattr(req, "kwargs", {}).get("until", [])
                gen_kwargs = getattr(req, "kwargs", {})
            else:
                if len(req) == 3:  # type: ignore[truthy-bool]
                    context, until, gen_kwargs = req  # type: ignore[misc]
                elif len(req) == 2:
                    context, until = req  # type: ignore[misc]
                    gen_kwargs = {}
                else:
                    context = req[0]  # type: ignore[index]
                    until, gen_kwargs = [], {}

            max_new_tokens = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks_default))
            input_ids = t.tensor([self.tok_encode(context)], dtype=t.long, device=self._device())
            with t.no_grad():
                if self.hooks:
                    with self.model.hooks(fwd_hooks=self.hooks):
                        out = self.model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                        )
                else:
                    out = self.model.generate(
                        input_ids, max_new_tokens=max_new_tokens, do_sample=False
                    )
            gen = self.tok_decode(out[0].tolist()[input_ids.shape[1] :])
            # Apply until stopping strings if provided
            if until:
                cut_points = [gen.find(u) for u in until if u in gen]
                if cut_points:
                    gen = gen[: min(c for c in cut_points if c >= 0)]
            generations.append(gen)
        return generations


def evaluate_lm_harness_on_tasks(  # noqa: C901 - complex function, refactor out of scope for lint pass
    model: HookedTransformer,
    pruned_artifact: Union[List[str], Dict[str, Any]],
    task_names: List[str],
    *,
    fewshot: int = 0,
    limit: Optional[int] = None,
    max_gen_toks: int = 64,
    confirm_run_unsafe_code: bool = False,
    sequential_per_task: bool = True,
    apply_chat_template: Union[bool, str] = "auto",
) -> Dict[str, Any]:
    """Run lm-evaluation-harness on a baseline and a pruned view of `model`.

    Returns a merged results dict with keys namespaced under 'original' and 'pruned'.

    Args:
        apply_chat_template: Wrap each task prompt in the model's chat template
            before scoring. Instruction-tuned models score badly on raw task
            text (MMLU/BoolQ/GSM8K/...), so this must be on for them. ``"auto"``
            (default) enables it iff the model's tokenizer ships a
            ``chat_template``; pass ``True``/``False`` to force it. The SAME
            resolved value is used for both the original and pruned runs so the
            comparison stays consistent.
    """
    if evaluator is None:  # pragma: no cover
        raise ImportError("lm-eval is not installed. Install `lm-eval` to run benchmarks.")

    # Build hooks from artifact
    if isinstance(pruned_artifact, list):
        hooks = _build_node_hooks(pruned_artifact)
    elif isinstance(pruned_artifact, dict):
        mlp = pruned_artifact.get("mlp", {})
        attn = pruned_artifact.get("attn", {})
        hooks = _build_neuron_hooks(mlp, attn)
    else:
        raise TypeError(f"Unsupported pruning artifact type: {type(pruned_artifact)}")

    # Instantiate LM wrappers
    lm_original = HookedTransformerLM(model=model, hooks=None, max_gen_toks_default=max_gen_toks)
    lm_pruned = HookedTransformerLM(model=model, hooks=hooks, max_gen_toks_default=max_gen_toks)

    # Resolve the chat-template setting ONCE so the original and pruned runs use
    # the identical value — instruction-tuned models score badly on raw task
    # text (MMLU/BoolQ/GSM8K/...), so this must be on for them. "auto" triggers
    # iff the model's tokenizer ships a chat_template.
    if apply_chat_template == "auto":
        chat_tmpl = getattr(getattr(model, "tokenizer", None), "chat_template", None)
        apply_chat_template = chat_tmpl is not None
    if apply_chat_template:
        logger.info("[lm-harness] instruction-tuned model detected; applying chat template.")

    import inspect

    _sig = inspect.signature(evaluator.simple_evaluate)
    _chat_kwargs: Dict[str, Any] = {}
    if "apply_chat_template" in _sig.parameters:
        _chat_kwargs["apply_chat_template"] = apply_chat_template
        if "fewshot_as_multiturn" in _sig.parameters:
            _chat_kwargs["fewshot_as_multiturn"] = bool(apply_chat_template) and fewshot > 0
    elif apply_chat_template:
        logger.info(
            "[lm-harness] WARNING: installed lm-eval has no apply_chat_template "
            "support; upgrade lm-eval to score instruction-tuned models correctly."
        )

    # --- Debug instrumentation ---
    try:
        logger.info(
            f"[lm-harness] version={_LMEVAL_VERSION}, TaskManager={TaskManager is not None}, tasks_module={lm_tasks is not None}"
        )
    except Exception:
        pass

    def _merge_results(acc: Optional[Dict[str, Any]], part: Dict[str, Any]) -> Dict[str, Any]:
        if acc is None:
            # Shallow copy to avoid mutation of part by caller
            return {k: (v.copy() if isinstance(v, dict) else v) for k, v in part.items()}
        # Merge dict-like result containers expected from lm-eval
        for key in part.keys():
            if key in {"results", "versions", "configs", "groups"} and isinstance(part[key], dict):
                acc.setdefault(key, {})
                acc[key].update(part[key])  # type: ignore[index]
            else:
                # Keep the first seen config/meta if present; otherwise set
                acc.setdefault(key, part[key])
        return acc

    # Prepare optional TaskManager / task dict for newer harness versions
    task_manager = None
    tasks_arg: Any = task_names
    try:
        if TaskManager is not None:
            task_manager = TaskManager()  # type: ignore[call-arg]
        if lm_tasks is not None and hasattr(lm_tasks, "get_task_dict"):
            # Convert list of names into a task dict for better compatibility
            tasks_arg = lm_tasks.get_task_dict(task_names, benchmark=False)  # type: ignore[attr-defined]
    except Exception:
        # Fall back silently if API not available
        task_manager = None
        tasks_arg = task_names

    # Debug: show how tasks were constructed
    try:
        if isinstance(tasks_arg, dict):
            logger.info(f"[lm-harness] constructed task dict with keys={list(tasks_arg.keys())}")
        else:
            logger.info(f"[lm-harness] using task list={tasks_arg}")
    except Exception:
        pass

    if sequential_per_task:
        # Run tasks one-by-one so only a single progress bar is shown per dataset.
        original_metrics: Optional[Dict[str, Any]] = None
        pruned_metrics: Optional[Dict[str, Any]] = None

        for task in task_names:
            tasks_single: Any = task
            # Build per-task dict if API is present
            try:
                if lm_tasks is not None and hasattr(lm_tasks, "get_task_dict"):
                    tasks_single = lm_tasks.get_task_dict([task], benchmark=False)  # type: ignore[attr-defined]
            except Exception:
                tasks_single = task
            try:
                logger.info(f"[lm-harness] evaluating task={task} (sequential)")
            except Exception:
                pass
            # Evaluate original
            orig_part = evaluator.simple_evaluate(
                model=lm_original,
                tasks=tasks_single,
                num_fewshot=fewshot,
                limit=limit,
                confirm_run_unsafe_code=confirm_run_unsafe_code,
                task_manager=task_manager,
                **_chat_kwargs,
            )
            try:
                res_keys = (
                    list(orig_part.keys()) if isinstance(orig_part, dict) else type(orig_part)
                )
                inner_keys = (
                    list(orig_part.get("results", {}).keys())
                    if isinstance(orig_part, dict) and isinstance(orig_part.get("results"), dict)
                    else []
                )
                logger.info(
                    f"[lm-harness] original part keys={res_keys}, results keys sample={inner_keys[:5]}"
                )
            except Exception:
                pass
            original_metrics = _merge_results(original_metrics, orig_part)

            # Evaluate pruned
            pruned_part = evaluator.simple_evaluate(
                model=lm_pruned,
                tasks=tasks_single,
                num_fewshot=fewshot,
                limit=limit,
                confirm_run_unsafe_code=confirm_run_unsafe_code,
                task_manager=task_manager,
                **_chat_kwargs,
            )
            try:
                res_keys = (
                    list(pruned_part.keys()) if isinstance(pruned_part, dict) else type(pruned_part)
                )
                inner_keys = (
                    list(pruned_part.get("results", {}).keys())
                    if isinstance(pruned_part, dict)
                    and isinstance(pruned_part.get("results"), dict)
                    else []
                )
                logger.info(
                    f"[lm-harness] pruned part keys={res_keys}, results keys sample={inner_keys[:5]}"
                )
            except Exception:
                pass
            pruned_metrics = _merge_results(pruned_metrics, pruned_part)
    else:
        # Fallback to single call (may show multiple progress bars concurrently).
        original_metrics = evaluator.simple_evaluate(
            model=lm_original,
            tasks=tasks_arg,
            num_fewshot=fewshot,
            limit=limit,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
            task_manager=task_manager,
            **_chat_kwargs,
        )
        try:
            res_keys = (
                list(original_metrics.keys())
                if isinstance(original_metrics, dict)
                else type(original_metrics)
            )
            inner_keys = (
                list(original_metrics.get("results", {}).keys())
                if isinstance(original_metrics, dict)
                and isinstance(original_metrics.get("results"), dict)
                else []
            )
            logger.info(
                f"[lm-harness] original(all) keys={res_keys}, results keys sample={inner_keys[:5]}"
            )
        except Exception:
            pass
        pruned_metrics = evaluator.simple_evaluate(
            model=lm_pruned,
            tasks=tasks_arg,
            num_fewshot=fewshot,
            limit=limit,
            confirm_run_unsafe_code=confirm_run_unsafe_code,
            task_manager=task_manager,
            **_chat_kwargs,
        )
        try:
            res_keys = (
                list(pruned_metrics.keys())
                if isinstance(pruned_metrics, dict)
                else type(pruned_metrics)
            )
            inner_keys = (
                list(pruned_metrics.get("results", {}).keys())
                if isinstance(pruned_metrics, dict)
                and isinstance(pruned_metrics.get("results"), dict)
                else []
            )
            logger.info(f"[lm-harness] pruned(all) keys={res_keys}, results keys sample={inner_keys[:5]}")
        except Exception:
            pass

    return {
        "original": original_metrics,
        "pruned": pruned_metrics,
    }


def summarize_lm_eval_results(  # noqa: C901 - complex function, refactor out of scope for lint pass
    results: Dict[str, Any], allowed_tasks: Optional[Union[List[str], set]] = None
) -> Dict[str, Any]:
    """Reduce lm-eval results to minimal leaderboard-style aggregates per parent task.

    - Supports grouped tasks like "mmlu" and multi-part tasks like "truthfulqa" by aggregating
      over subtasks with a weighted average using 'n' when available.
    - Keeps only common leaderboard keys: acc, acc_norm, exact_match, ppl, pass@1, mc1, mc2, f1, em, n.
    """
    metrics_root: Dict[str, Any]
    if isinstance(results, dict) and "results" in results and isinstance(results["results"], dict):
        metrics_root = results["results"]
    else:
        metrics_root = results if isinstance(results, dict) else {}

    try:
        logger.info(
            f"[lm-harness] summarize: top-level keys={list(results.keys()) if isinstance(results, dict) else type(results)}"
        )
        logger.info(
            f"[lm-harness] summarize: results.keys={list(metrics_root.keys()) if isinstance(metrics_root, dict) else 'NA'}"
        )
    except Exception:
        pass

    # Optional per-task sample counts available at top level in newer harness
    n_samples_map: Dict[str, Any] = {}
    if isinstance(results, dict) and isinstance(results.get("n-samples"), dict):
        n_samples_map = results.get("n-samples", {})  # type: ignore[assignment]

    parent_tasks: List[str]
    if allowed_tasks is not None:
        parent_tasks = list(allowed_tasks)
    else:
        # Infer parents from top-level keys by stripping common suffixes for mmlu and truthfulqa
        parent_tasks = list({k.split("_")[0] for k in metrics_root.keys()})

    keep_keys = {"acc", "acc_norm", "exact_match", "ppl", "pass@1", "mc1", "mc2", "f1", "em", "n"}

    def extract_numeric(value: Any) -> Optional[float]:
        # Newer lm-eval may return metric objects like {"value": 0.51, "stderr": 0.01}
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            v = value.get("value")
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def get_metric_mapping(m: Dict[str, Any]) -> Dict[str, Any]:
        # Some versions place metrics under a nested 'metrics' key
        if isinstance(m, dict) and "metrics" in m and isinstance(m["metrics"], dict):
            return m["metrics"]
        return m

    def get_count(m: Dict[str, Any], task_name: Optional[str] = None) -> float:
        # Prefer 'n' if present; otherwise try common alternatives
        for k in ("n", "num_samples", "num_examples"):
            if k in m:
                num = extract_numeric(m[k]) if not isinstance(m[k], (int, float)) else float(m[k])
                if num is not None:
                    return float(num)
        # Also check under nested metrics
        mm = get_metric_mapping(m)
        for k in ("n", "num_samples", "num_examples"):
            if k in mm:
                num = (
                    extract_numeric(mm[k]) if not isinstance(mm[k], (int, float)) else float(mm[k])
                )
                if num is not None:
                    return float(num)
        # Fallback to top-level n-samples map when available
        if task_name is not None and task_name in n_samples_map:
            try:
                val = n_samples_map[task_name]
                return float(val)
            except Exception:
                return 0.0
        return 0.0

    def find_metric_value(mm: Dict[str, Any], short_key: str) -> Optional[float]:
        # Direct key
        if short_key in mm:
            return extract_numeric(mm[short_key])
        # Variants like 'acc,none' or 'acc_norm,none'
        for k, v in mm.items():
            if isinstance(k, str) and k.split(",")[0] == short_key:
                val = extract_numeric(v)
                if val is not None:
                    return val
        return None

    def is_child_of(parent: str, name: str) -> bool:
        if name == parent:
            return True
        if name.startswith(parent + "_"):
            return True
        # Common variants
        if parent == "truthfulqa" and (
            name.startswith("truthfulqa_") or name in {"truthfulqa_mc", "truthfulqa_gen"}
        ):
            return True
        return False

    summary: Dict[str, Any] = {}
    for parent in parent_tasks:
        # Direct parent metrics if present
        if parent in metrics_root and isinstance(metrics_root[parent], dict):
            task_metrics_raw = metrics_root[parent]
            task_metrics = get_metric_mapping(task_metrics_raw)
            try:
                logger.info(f"[lm-harness] summarize[{parent}]: metric keys={list(task_metrics.keys())}")
            except Exception:
                pass
            pruned: Dict[str, float] = {}
            for k in keep_keys:
                val = find_metric_value(task_metrics, k)
                if val is not None:
                    pruned[k] = val
            summary[parent] = pruned
            continue

        # Otherwise aggregate children (handles e.g., mmlu_* and truthfulqa_* variants)
        child_items = [
            (name, m)
            for name, m in metrics_root.items()
            if is_child_of(parent, name) and isinstance(m, dict)
        ]
        if not child_items:
            # Fallback: if parent exists but is not a dict, or metrics are nested, try shallow extraction
            m = metrics_root.get(parent)
            if isinstance(m, dict):
                pruned: Dict[str, float] = {}
                for k in keep_keys:
                    if k in m:
                        num = extract_numeric(m[k])
                        if num is not None:
                            pruned[k] = num
                summary[parent] = pruned
            else:
                summary[parent] = {}
            continue

        # Weighted aggregate by 'n' where available; fallback to simple mean
        agg: Dict[str, float] = {}
        total_n = 0.0
        for name, m in child_items:
            n = get_count(m, task_name=name)
            total_n += n
        for key in keep_keys - {"n"}:
            vals = []
            for name, m in child_items:
                mm = get_metric_mapping(m)
                num = find_metric_value(mm, key)
                if num is not None:
                    vals.append((num, get_count(m, task_name=name)))
            if not vals:
                continue
            if total_n > 0:
                agg[key] = sum(v * n for v, n in vals) / total_n
            else:
                agg[key] = sum(v for v, _ in vals) / len(vals)
        agg["n"] = (
            total_n if total_n > 0 else sum(get_count(m, task_name=name) for name, m in child_items)
        )
        summary[parent] = {k: agg[k] for k in keep_keys if k in agg}

    return summary
