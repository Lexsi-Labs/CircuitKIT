"""Post-application benchmarking: export an intervened model as a HuggingFace
checkpoint, then run lm-evaluation-harness on it.

This is the "after applications we export the checkpoint and bench on lm-eval"
step.  It turns a pruned / quantized model into a standard HF checkpoint whose
intervened weights are actually persisted to disk, and runs ``lm_eval`` on that
checkpoint with the stock ``HFLM`` (``--model hf`` equivalent) so the numbers
come from a real, reloadable artifact — not from forward-hooks on a live model.

Two entry points:

* :func:`export_and_benchmark` — one-shot: export + verify + lm-eval.
* :func:`compare_base_vs_intervened` — runs the same tasks on the unmodified
  model and on the intervened checkpoint, returns a side-by-side delta table.

Supported interventions
-----------------------
* **pruning** — a TransformerLens ``HookedTransformer`` plus a pruning artifact
  (node list ``["A0.0", "MLP 1", ...]`` or neuron dict).  The artifact is
  applied directly to the original HF weights via
  :func:`circuitkit.evaluation.hf_checkpoint.save_pruned_checkpoint`.
* **quantization** — an already-quantized HF ``AutoModelForCausalLM`` (e.g. the
  output of :func:`circuitkit.applications.quantization.circuit_quantize`),
  persisted via
  :func:`circuitkit.evaluation.hf_checkpoint.save_quantized_checkpoint`.
"""

from __future__ import annotations
from circuitkit.utils.device import get_device, empty_cache

import logging
import os
from typing import Any, Dict, List, Optional, Union

from .hf_checkpoint import (
    is_quantized_checkpoint,
    save_pruned_checkpoint,
    save_quantized_checkpoint,
)

logger = logging.getLogger("circuitkit.evaluation.checkpoint_benchmark")

# Sensible standard tasks for a quick post-intervention sanity benchmark.
DEFAULT_TASKS: List[str] = ["boolq", "winogrande", "lambada_openai"]

# lm-eval metric keys to surface, in priority order, per task.
_METRIC_KEYS = ["acc_norm,none", "acc,none", "perplexity,none", "exact_match,none"]

# Metric direction per lm-eval metric key: +1 = higher is better (accuracy /
# exact_match), -1 = lower is better (perplexity / word_perplexity / loss).
# Used so a base-vs-pruned delta is computed/labelled in the right direction.
_METRIC_DIRECTION: Dict[str, int] = {
    "acc_norm,none": +1,
    "acc,none": +1,
    "exact_match,none": +1,
    "perplexity,none": -1,
    "word_perplexity,none": -1,
    "byte_perplexity,none": -1,
    "bits_per_byte,none": -1,
}


def _metric_direction(metric_key: Optional[str]) -> int:
    """+1 if higher-is-better for this metric, -1 if lower-is-better."""
    if metric_key is None:
        return +1
    if metric_key in _METRIC_DIRECTION:
        return _METRIC_DIRECTION[metric_key]
    # Heuristic fallback for unrecognised keys.
    low = metric_key.lower()
    if "perplexity" in low or "loss" in low or "bits_per_byte" in low:
        return -1
    return +1


# Cloze / loglikelihood tasks scored on a continuous raw-text stream. They have
# no user/assistant turn structure, so they are never chat-templated even when
# the rest of the suite is — templating a cloze task is a category error and
# would mismatch raw-text circuit discovery (see tasks/_chat.py).
_CLOZE_RAW_TASKS = {
    "winogrande",
    "lambada_openai",
    "lambada",
    "lambada_standard",
    "wikitext",
}


def _extract_metric(task_results: Dict[str, Any]) -> Dict[str, float]:
    """Pull the leaderboard-style metrics out of one lm-eval task block."""
    out: Dict[str, float] = {}
    for k, v in task_results.items():
        if not isinstance(k, str) or k.endswith("_stderr,none"):
            continue
        if isinstance(v, (int, float)) and v == v:  # finite, not NaN
            out[k] = round(float(v), 4)
    return out


def _primary_score(metrics: Dict[str, float]) -> Optional[float]:
    """Return the single headline number for a task (acc_norm > acc > ...)."""
    score, _ = _primary_score_keyed(metrics)
    return score


def _primary_score_keyed(
    metrics: Dict[str, float],
) -> tuple[Optional[float], Optional[str]]:
    """Like :func:`_primary_score` but also returns the metric *key*, so the
    caller can look up its direction (higher- vs lower-is-better)."""
    for k in _METRIC_KEYS:
        if k in metrics:
            return metrics[k], k
    # Fall back to any non-stderr numeric metric.
    for k, v in metrics.items():
        if not k.endswith("_stderr,none"):
            return v, k
    return None, None


def verify_checkpoint_weights(
    checkpoint_path: str,
    *,
    expect_zeros_in: Optional[List[str]] = None,
    expect_zero_slices: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Reload a checkpoint from disk and report on its weights.

    Args:
        checkpoint_path: Directory written by ``save_*_checkpoint``.
        expect_zeros_in: Optional list of state-dict key substrings that the
            intervention should have driven *fully* to zero (e.g. an MLP node).
        expect_zero_slices: Optional list of per-slice expectations, each a
            dict ``{"key": str, "axis": 0|1|None, "start": int, "stop": int}``.
            ``axis=None`` means the whole tensor; otherwise only the
            ``[start:stop]`` slice along ``axis`` must be zero. This is how an
            attention-head node's specific ``o_proj`` sub-slice is checked
            (which substring matching alone cannot do).

    Returns:
        Dict with ``n_params``, ``n_zero``, ``zero_fraction`` and, when
        expectations are given, a ``pruning_verified`` bool plus a
        ``slice_checks`` list of per-expectation pass/fail records.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    sd = model.state_dict()
    n_params = sum(v.numel() for v in sd.values())
    n_zero = sum(int((v == 0).sum().item()) for v in sd.values())
    info: Dict[str, Any] = {
        "n_params": n_params,
        "n_zero": n_zero,
        "zero_fraction": round(n_zero / max(n_params, 1), 4),
        "reloaded_ok": True,
    }

    verified = True
    have_expectations = bool(expect_zeros_in) or bool(expect_zero_slices)

    for sub in expect_zeros_in or []:
        matched = [k for k in sd if sub in k]
        if not matched or any(float(sd[k].abs().sum()) != 0.0 for k in matched):
            verified = False

    slice_checks: List[Dict[str, Any]] = []
    for exp in expect_zero_slices or []:
        key = exp["key"]
        axis = exp.get("axis")
        rec: Dict[str, Any] = {
            "key": key,
            "axis": axis,
            "start": exp.get("start"),
            "stop": exp.get("stop"),
            "node": exp.get("node"),
        }
        if key not in sd:
            rec["status"] = "MISSING_KEY"
            verified = False
            slice_checks.append(rec)
            continue
        w = sd[key]
        if axis is None:
            sub_t = w
        elif axis == 0:
            sub_t = w[exp["start"] : exp["stop"], ...]
        elif axis == 1:
            sub_t = w[:, exp["start"] : exp["stop"]]
        else:
            raise ValueError(
                f"Unsupported slice axis {axis!r} in pruning artifact. "
                f"'axis' must be 0 (row slice), 1 (column slice), or None "
                f"(whole tensor). Fix the artifact's slice spec."
            )
        is_zero = float(sub_t.abs().sum()) == 0.0
        rec["status"] = "ZERO" if is_zero else "NONZERO"
        rec["abs_sum"] = float(sub_t.abs().sum())
        if not is_zero:
            verified = False
        slice_checks.append(rec)

    if have_expectations:
        info["pruning_verified"] = verified
        if slice_checks:
            info["slice_checks"] = slice_checks
    return info


def _is_compressed_tensors_unsupported_by_vllm(checkpoint_path: str) -> bool:
    """True iff ``checkpoint_path`` is a compressed-tensors checkpoint whose
    weight bit-width the installed vLLM build cannot serve.

    vLLM's compressed-tensors loader only accepts the weight bit-widths in its
    ``WNA16_SUPPORTED_BITS`` map (older builds: ``{4, 8}`` — no 3-bit). For an
    unsupported width vLLM raises ``NotImplementedError`` deep inside model
    load; detecting it up front lets ``run_lm_eval`` fall back to the HF
    backend cleanly. Returns ``False`` for any non-compressed-tensors
    checkpoint, or when the bit-width cannot be determined (let vLLM try).
    """
    import json as _json
    import os as _os

    from .hf_checkpoint import is_compressed_tensors_checkpoint

    if not is_compressed_tensors_checkpoint(checkpoint_path):
        return False
    cfg_path = _os.path.join(checkpoint_path, "config.json")
    try:
        cfg = _json.loads(open(cfg_path, "r").read())
    except (OSError, ValueError):
        return False
    qc = cfg.get("quantization_config", {})
    bits = set()
    for grp in (qc.get("config_groups") or {}).values():
        w = (grp or {}).get("weights") or {}
        if "num_bits" in w:
            bits.add(int(w["num_bits"]))
    if not bits:
        return False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
            WNA16_SUPPORTED_BITS,
        )
    except Exception:  # noqa: BLE001 - vLLM not importable -> let caller decide
        return False
    return any(b not in set(WNA16_SUPPORTED_BITS) for b in bits)


def _is_multimodal_checkpoint(checkpoint_path: str) -> bool:
    """True iff ``checkpoint_path``'s config declares a multimodal model.

    Multimodal models (e.g. Gemma 3, whose ``AutoModelForCausalLM`` class is
    ``Gemma3ForConditionalGeneration``) carry a ``vision_config`` block. vLLM
    loads and warms up the vision tower even for a text-only benchmark, which
    on some CUDA builds aborts with ``CUDA error: unspecified launch failure``
    inside the SigLIP encoder. Detecting it up front lets ``run_lm_eval`` fall
    back to the HF backend — which never invokes the vision tower for a
    text-only forward — so the cell still yields a finite row.
    """
    import json as _json
    import os as _os

    cfg_path = _os.path.join(checkpoint_path, "config.json")
    try:
        cfg = _json.loads(open(cfg_path, "r").read())
    except (OSError, ValueError):
        return False
    if "vision_config" in cfg:
        return True
    archs = cfg.get("architectures") or []
    return any("ConditionalGeneration" in a for a in archs)


def run_lm_eval(
    checkpoint_path: str,
    tasks: List[str],
    *,
    tokenizer: Optional[str] = None,
    limit: Optional[int] = None,
    fewshot: int = 0,
    batch_size: Union[int, str] = "auto",
    backend: str = "hf",
    device: str = "auto",
    dtype: str = "float32",
    apply_chat_template: Union[bool, str] = "auto",
    gpu_memory_utilization: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Run lm-evaluation-harness on a saved HF checkpoint — in a subprocess.

    The benchmark backends (notably the vLLM backend) do **not** release their
    GPU memory when ``run_lm_eval`` returns: neither lm-eval's ``VLLM`` wrapper
    nor a plain ``HFLM`` exposes a teardown hook, vLLM's v1 engine keeps the
    model + KV cache in a child ``EngineCore`` process, and the CUDA allocator
    caches reserved blocks. In a long-lived harness that runs many cells in one
    process this leaks ~tens of GiB *per cell* and OOMs a later cell.

    The robust fix — used here — is to run the whole benchmark in a fresh child
    **process**: when that process exits the OS reclaims *all* of its GPU
    memory (CUDA context, allocator cache, and any vLLM ``EngineCore``
    grandchild), so no leak can survive across cells. The child returns the
    ``{task: {metric: value}}`` dict over a pipe; the numbers are byte-identical
    to running in-process (same checkpoint, same lm-eval call).

    Set ``CK_LM_EVAL_INPROC=1`` to force the legacy in-process path (used
    internally inside the child, and available for debugging) — note that the
    in-process path leaks GPU memory and must not be used in a multi-cell loop.

    Args:
        checkpoint_path: Directory containing ``config.json`` + weights.
        tasks: lm-eval task names (e.g. ``["boolq", "winogrande"]``).
        tokenizer: HF tokenizer id; defaults to the checkpoint itself (the
            ``save_*`` helpers write a tokenizer into the directory).
        limit: Cap examples per task (use for quick smoke tests).
        fewshot: Few-shot example count.
        batch_size: lm-eval batch size (``"auto"`` recommended).
        backend: ``"hf"`` (HFLM) or ``"vllm"`` (vLLM backend, if installed).
        device: Torch device for the ``hf`` backend.
        dtype: Model dtype string for the ``hf`` backend.
        apply_chat_template: Wrap each task prompt in the model's chat template
            before scoring. Instruction-tuned checkpoints score badly on raw
            task text (MMLU/BoolQ/GSM8K/...), so this must be on for them.
            ``"auto"`` (default) enables it iff the checkpoint's tokenizer
            ships a ``chat_template``; pass ``True``/``False`` to force it.
        gpu_memory_utilization: vLLM-backend only. Fraction of GPU memory vLLM
            may claim on startup. ``None`` (default) uses vLLM's own default
            (0.9). Pass a lower value when the calling process still holds GPU
            tensors, so vLLM leaves headroom instead of OOMing at warmup.

    Returns:
        ``{task: {metric: value}}`` — only finite leaderboard metrics.
    """
    kwargs = dict(
        tokenizer=tokenizer, limit=limit, fewshot=fewshot,
        batch_size=batch_size, backend=backend, device=device, dtype=dtype,
        apply_chat_template=apply_chat_template,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    # Default path: run the benchmark in a child process so the OS reclaims
    # ALL of its GPU memory on exit (see the docstring). CK_LM_EVAL_INPROC=1
    # forces the legacy in-process path — it is set inside the child itself
    # (so the child does not recurse) and is available for debugging.
    if not os.environ.get("CK_LM_EVAL_INPROC"):
        return _run_lm_eval_subprocess(checkpoint_path, tasks, kwargs)
    return _run_lm_eval_inproc(checkpoint_path, tasks, **kwargs)


def _run_lm_eval_subprocess(
    checkpoint_path: str,
    tasks: List[str],
    kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Run :func:`_run_lm_eval_inproc` in a child process and return its result.

    The child runs this module as a script (``python -m`` style), does the
    benchmark, prints the ``{task: {metric: value}}`` dict as JSON between two
    sentinels on stdout, and exits — at which point the OS frees every byte of
    its GPU memory. The child inherits the parent's environment (HF cache /
    tokens / CUDA device) plus ``CK_LM_EVAL_INPROC=1`` so it takes the
    in-process path instead of recursing.
    """
    import json
    import subprocess
    import sys

    payload = json.dumps(
        {"checkpoint_path": checkpoint_path, "tasks": tasks, "kwargs": kwargs}
    )
    child_env = dict(os.environ)
    child_env["CK_LM_EVAL_INPROC"] = "1"
    # Run this module via ``-m`` so its package-relative imports
    # (``from .hf_checkpoint import ...``) resolve; __main__ below reads the
    # payload on stdin and prints the JSON result between the sentinels.
    proc = subprocess.run(
        [sys.executable, "-m", "circuitkit.evaluation.checkpoint_benchmark"],
        input=payload,
        capture_output=True,
        text=True,
        env=child_env,
    )
    # Surface the child's logs so a failing benchmark is still diagnosable.
    if proc.stderr:
        for line in proc.stderr.rstrip().splitlines():
            logger.info("[lm_eval child] %s", line)
    if proc.returncode != 0:
        raise RuntimeError(
            f"lm-eval subprocess exited {proc.returncode}; see '[lm_eval "
            f"child]' log lines above for the cause."
        )
    marker_a, marker_b = _RESULT_SENTINELS
    out = proc.stdout
    if marker_a not in out or marker_b not in out:
        raise RuntimeError(
            "lm-eval subprocess produced no result block; see '[lm_eval "
            "child]' log lines above."
        )
    blob = out.split(marker_a, 1)[1].split(marker_b, 1)[0]
    return json.loads(blob)


# Sentinels delimiting the JSON result the child prints on stdout — chosen so
# they cannot collide with lm-eval / vLLM progress output.
_RESULT_SENTINELS = ("<<<CK_LM_EVAL_RESULT>>>", "<<<END_CK_LM_EVAL_RESULT>>>")


def _run_lm_eval_inproc(
    checkpoint_path: str,
    tasks: List[str],
    *,
    tokenizer: Optional[str] = None,
    limit: Optional[int] = None,
    fewshot: int = 0,
    batch_size: Union[int, str] = "auto",
    backend: str = "hf",
    device: str = "auto",
    dtype: str = "float32",
    apply_chat_template: Union[bool, str] = "auto",
    gpu_memory_utilization: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """In-process lm-eval body (see :func:`run_lm_eval` for the public API).

    LEAKS GPU memory — the benchmark backends never release it. Call only from
    inside the :func:`_run_lm_eval_subprocess` child, never in a multi-cell
    loop. ``run_lm_eval`` dispatches here only when ``CK_LM_EVAL_INPROC`` is set.
    """
    device = get_device(device)
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("lm-eval is required: pip install lm-eval") from e

    tok = tokenizer or checkpoint_path

    # A quanto-quantized checkpoint cannot be reloaded by HFLM's plain
    # AutoModelForCausalLM.from_pretrained — that path ignores the quanto
    # quantization map and would load a broken / fp model. Reload it the
    # quanto way and hand HFLM the already-built model object.
    #
    # NOTE: ``is_quantized_checkpoint`` detects ONLY the optimum-quanto format
    # (quanto_qmap.json). A compressed-tensors checkpoint (llmcompressor
    # backend) is HF-native and vLLM-native: it carries a ``quantization_config``
    # in config.json and reloads via plain ``from_pretrained``. It deliberately
    # falls through to the plain branch below — vLLM serves it directly, with
    # NO dequantization step (unlike quanto).
    quantized = is_quantized_checkpoint(checkpoint_path)

    # A compressed-tensors checkpoint is vLLM-native — but only for the
    # weight bit-widths the *installed* vLLM build supports (its WNA16 scheme
    # map). e.g. a true-3-bit checkpoint loads fine via plain HF
    # ``from_pretrained`` but an older vLLM whose WNA16 map is {4, 8} raises
    # "No compressed-tensors compatible scheme was found". When that is the
    # case we transparently fall back to the HF backend (which serves the
    # checkpoint correctly) so the cell still yields a finite row.
    if backend == "vllm" and _is_compressed_tensors_unsupported_by_vllm(
        checkpoint_path
    ):
        logger.warning(
            "compressed-tensors checkpoint %s uses a weight bit-width the "
            "installed vLLM build does not support; falling back to the HF "
            "backend for this benchmark.",
            checkpoint_path,
        )
        backend = "hf"

    # A multimodal checkpoint (Gemma 3, …) keeps a vision tower in its config;
    # vLLM warms it up even for a text-only benchmark and can abort inside the
    # SigLIP encoder ("CUDA error: unspecified launch failure"). The HF backend
    # never runs the vision tower for a text-only forward — route to it.
    if backend == "vllm" and _is_multimodal_checkpoint(checkpoint_path):
        logger.warning(
            "multimodal checkpoint %s carries a vision tower vLLM cannot "
            "reliably warm up; falling back to the HF backend for this "
            "benchmark.",
            checkpoint_path,
        )
        backend = "hf"

    if backend == "vllm":
        # vLLM has no optimum-quanto loader (it supports GPTQ / AWQ /
        # bitsandbytes / fp8 / compressed-tensors / gguf / ...). A quanto
        # checkpoint therefore cannot be served directly. Instead of failing,
        # export a *dequantized* fp checkpoint: vLLM then runs the exact
        # quantized weights (the quantization error is already baked in), just
        # not in a packed int kernel — eval numbers are identical.
        vllm_checkpoint = checkpoint_path
        if quantized:
            from .hf_checkpoint import export_dequantized_checkpoint

            vllm_checkpoint = checkpoint_path.rstrip("/") + "__dequantized_for_vllm"
            logger.info(
                "Quantized checkpoint detected; vLLM cannot serve optimum-quanto. "
                "Exporting dequantized fp checkpoint to %s for the vLLM backend "
                "(weights = exact quantized values upcast to fp; eval numbers "
                "match the quanto checkpoint).",
                vllm_checkpoint,
            )
            export_dequantized_checkpoint(checkpoint_path, vllm_checkpoint, overwrite=True)
            # The dequantized checkpoint carries the tokenizer too; prefer it
            # unless the caller pinned an explicit tokenizer id.
            if tokenizer is None:
                tok = vllm_checkpoint
        from lm_eval.models.vllm_causallms import VLLM  # type: ignore

        # vLLM grabs `gpu_memory_utilization` of the device on startup (default
        # 0.9). When the caller still has tensors resident — e.g. a harness that
        # ran gradient-based discovery in-process moments earlier — 0.9 collides
        # with that residue and OOMs at sampler warmup. A caller that knows it
        # holds VRAM can pass a lower fraction to leave headroom.
        vllm_kwargs: Dict[str, Any] = {}
        if gpu_memory_utilization is not None:
            vllm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        lm = VLLM(pretrained=vllm_checkpoint, tokenizer=tok, dtype=dtype,
                  **vllm_kwargs)
    elif quantized:
        from .hf_checkpoint import load_quantized_checkpoint

        logger.info("Quantized checkpoint detected; reloading via quanto.")
        qmodel = load_quantized_checkpoint(checkpoint_path)
        qmodel.to(device)
        lm = HFLM(
            pretrained=qmodel,
            tokenizer=tok,
            batch_size=batch_size,
        )
    else:
        lm = HFLM(
            pretrained=checkpoint_path,
            tokenizer=tok,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
        )

    # Instruction-tuned checkpoints must receive prompts wrapped in their chat
    # template — raw task text (MMLU/BoolQ/GSM8K/...) confuses them. "auto"
    # triggers iff the checkpoint's tokenizer ships a chat_template.
    if apply_chat_template == "auto":
        tok_obj = getattr(lm, "tokenizer", None)
        chat_tmpl = getattr(tok_obj, "chat_template", None)
        if chat_tmpl is None and tok_obj is None:
            # vLLM (and some backends) don't expose ``.tokenizer`` the way HFLM
            # does — load the checkpoint tokenizer directly to read its
            # chat_template so "auto" isn't silently disabled for instruct models.
            try:
                from transformers import AutoTokenizer

                chat_tmpl = getattr(AutoTokenizer.from_pretrained(tok), "chat_template", None)
            except Exception:  # noqa: BLE001 - probe only; fall back to off
                chat_tmpl = None
        apply_chat_template = chat_tmpl is not None
    if apply_chat_template:
        logger.info("Instruction-tuned checkpoint detected; applying chat template.")

    import inspect

    _sig = inspect.signature(evaluator.simple_evaluate)
    _supports_ct = "apply_chat_template" in _sig.parameters

    def _evaluate(task_subset: List[str], apply_ct: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = dict(
            model=lm,
            tasks=task_subset,
            num_fewshot=fewshot,
            limit=limit,
            bootstrap_iters=0,  # skip stderr bootstrap — faster, deterministic
        )
        if _supports_ct:
            kw["apply_chat_template"] = apply_ct
            if "fewshot_as_multiturn" in _sig.parameters:
                kw["fewshot_as_multiturn"] = apply_ct and fewshot > 0
        elif apply_ct:
            logger.warning(
                "Installed lm-eval has no apply_chat_template support; upgrade "
                "lm-eval to score instruction-tuned checkpoints correctly."
            )
        return evaluator.simple_evaluate(**kw)

    # When chat-templating is on, cloze/loglikelihood tasks still run raw so
    # discovery (raw) and eval stay consistent — split the suite and run each
    # group with its own setting.
    if apply_chat_template:
        raw = [t for t in tasks if t in _CLOZE_RAW_TASKS]
        templated = [t for t in tasks if t not in _CLOZE_RAW_TASKS]
        groups = [g for g in ((templated, True), (raw, False)) if g[0]]
        if raw:
            logger.info("Tasks kept raw (cloze — never chat-templated): %s", raw)
    else:
        groups = [(tasks, False)]

    out: Dict[str, Dict[str, float]] = {}
    for task_subset, apply_ct in groups:
        results = _evaluate(task_subset, apply_ct)
        for task in task_subset:
            block = results.get("results", {}).get(task, {})
            out[task] = _extract_metric(block)
    return out


def _verify_pruned_checkpoint(
    model: Any,
    pruned_artifact: Union[List[str], Dict[str, Any]],
    checkpoint_path: str,
) -> Dict[str, Any]:
    """Verify a pruned checkpoint, asserting the *specific* per-head o_proj
    sub-slices and full MLP tensors are zero (not just substring matches)."""
    from .hf_checkpoint import _resolve_arch, expected_zero_keys

    expect_zeros_in: List[str] = []
    expect_slices: List[Dict[str, Any]] = []

    if isinstance(pruned_artifact, list) and pruned_artifact:
        arch_cfg = _resolve_arch(model.cfg.original_architecture, model.cfg.model_name)
        d_head = int(model.cfg.d_head)
        for exp in expected_zero_keys(pruned_artifact, arch_cfg):
            if exp["kind"] == "o_proj":
                # Per-head o_proj sub-slice: derive start/stop from d_head.
                head = int(exp["node"].split(".")[1])
                expect_slices.append(
                    {
                        "key": exp["key"],
                        "axis": exp["axis"],
                        "start": head * d_head,
                        "stop": head * d_head + d_head,
                        "node": exp["node"],
                    }
                )
            else:  # whole MLP tensor
                expect_slices.append(
                    {
                        "key": exp["key"],
                        "axis": None,
                        "start": None,
                        "stop": None,
                        "node": exp["node"],
                    }
                )

    return verify_checkpoint_weights(
        checkpoint_path,
        expect_zeros_in=expect_zeros_in or None,
        expect_zero_slices=expect_slices or None,
    )


def export_and_benchmark(
    model: Any,
    *,
    intervention: str = "pruning",
    pruned_artifact: Optional[Union[List[str], Dict[str, Any]]] = None,
    checkpoint_path: str,
    tasks: Optional[List[str]] = None,
    tokenizer: Optional[str] = None,
    limit: Optional[int] = None,
    fewshot: int = 0,
    backend: str = "hf",
    device: str = "auto",
    dtype: str = "float32",
    apply_chat_template: Union[bool, str] = "auto",
    overwrite: bool = True,
    keep_checkpoint: bool = True,
) -> Dict[str, Any]:
    """Export an intervened model as a HF checkpoint and benchmark it on lm-eval.

    Args:
        model: For ``intervention="pruning"`` a TransformerLens
            ``HookedTransformer``; for ``intervention="quantization"`` an
            already-quantized HF ``AutoModelForCausalLM``.
        intervention: ``"pruning"`` or ``"quantization"``.
        pruned_artifact: Required for pruning — node list or neuron dict.
        checkpoint_path: Where to write the HF checkpoint.
        tasks: lm-eval tasks (defaults to :data:`DEFAULT_TASKS`).
        tokenizer: HF tokenizer id; defaults to the checkpoint directory.
        limit / fewshot / backend / device / dtype: passed to :func:`run_lm_eval`.
        overwrite: Overwrite ``checkpoint_path`` if it exists.
        keep_checkpoint: If False, delete the checkpoint after benchmarking.

    Returns:
        Dict with ``checkpoint_path``, ``verification`` and ``benchmark``.
    """
    tasks = tasks or DEFAULT_TASKS

    if intervention == "pruning":
        if pruned_artifact is None:
            raise ValueError(
                "intervention='pruning' requires a 'pruned_artifact' argument, "
                "but none was provided. Pass the artifact returned by "
                "circuitkit.prune(...), or use intervention='quantization'."
            )
        save_pruned_checkpoint(model, pruned_artifact, checkpoint_path, overwrite=overwrite)
        # Verify the specific per-head o_proj sub-slices and full MLP tensors.
        verification = _verify_pruned_checkpoint(model, pruned_artifact, checkpoint_path)
    elif intervention == "quantization":
        save_quantized_checkpoint(
            model, checkpoint_path, tokenizer_name=tokenizer, overwrite=overwrite
        )
        verification = verify_checkpoint_weights(checkpoint_path)
    else:
        raise ValueError(
            f"Unknown intervention {intervention!r}. "
            f"Set 'intervention' to one of: 'pruning', 'quantization'."
        )

    logger.info("Checkpoint exported to %s (%s)", checkpoint_path, verification)

    benchmark = run_lm_eval(
        checkpoint_path,
        tasks,
        tokenizer=tokenizer,
        limit=limit,
        fewshot=fewshot,
        backend=backend,
        device=device,
        dtype=dtype,
        apply_chat_template=apply_chat_template,
    )

    if not keep_checkpoint:
        import shutil

        shutil.rmtree(checkpoint_path, ignore_errors=True)

    return {
        "checkpoint_path": checkpoint_path if keep_checkpoint else None,
        "intervention": intervention,
        "verification": verification,
        "benchmark": benchmark,
    }


def compare_base_vs_intervened(
    model: Any,
    pruned_artifact: Union[List[str], Dict[str, Any]],
    *,
    work_dir: str,
    tasks: Optional[List[str]] = None,
    tokenizer: Optional[str] = None,
    limit: Optional[int] = None,
    fewshot: int = 0,
    backend: str = "hf",
    device: str = "auto",
    dtype: str = "float32",
    apply_chat_template: Union[bool, str] = "auto",
    keep_checkpoints: bool = False,
) -> Dict[str, Any]:
    """Benchmark a base model and its pruned variant, return a delta table.

    Exports two checkpoints (an empty-artifact "base" and the pruned one) so the
    comparison is apples-to-apples — both go through the identical
    TransformerLens→HF export path.

    The pruned checkpoint's weights are verified (per-head ``o_proj`` slices and
    full MLP tensors asserted zero) before benchmarking.

    Each ``delta`` is *improvement-signed*: positive always means the pruned
    model is better, even for lower-is-better metrics (perplexity). The raw
    metric and its direction are also recorded per task.

    Returns:
        Dict ``{task: {"base": x, "pruned": y, "delta": ..., "metric": k,
        "direction": ±1, "raw_delta": p-b}, ...}`` plus the raw per-task
        metric dicts and the pruned-checkpoint ``verification`` record.
    """
    tasks = tasks or DEFAULT_TASKS
    base_dir = os.path.join(work_dir, "base")
    pruned_dir = os.path.join(work_dir, "pruned")

    save_pruned_checkpoint(model, [], base_dir, overwrite=True)
    save_pruned_checkpoint(model, pruned_artifact, pruned_dir, overwrite=True)
    verification = _verify_pruned_checkpoint(model, pruned_artifact, pruned_dir)
    logger.info("Pruned checkpoint verification: %s", verification)

    base_metrics = run_lm_eval(
        base_dir,
        tasks,
        tokenizer=tokenizer,
        limit=limit,
        fewshot=fewshot,
        backend=backend,
        device=device,
        dtype=dtype,
        apply_chat_template=apply_chat_template,
    )
    pruned_metrics = run_lm_eval(
        pruned_dir,
        tasks,
        tokenizer=tokenizer,
        limit=limit,
        fewshot=fewshot,
        backend=backend,
        device=device,
        dtype=dtype,
        apply_chat_template=apply_chat_template,
    )

    table: Dict[str, Any] = {}
    for task in tasks:
        b, b_key = _primary_score_keyed(base_metrics.get(task, {}))
        p, p_key = _primary_score_keyed(pruned_metrics.get(task, {}))
        # Direction comes from whichever headline metric is present.
        direction = _metric_direction(p_key or b_key)
        if b is not None and p is not None:
            raw_delta = round(p - b, 4)
            # Improvement-signed delta: + always means "pruned is better",
            # so for perplexity (lower-is-better) the sign is flipped.
            delta = round(direction * (p - b), 4)
        else:
            raw_delta = None
            delta = None
        table[task] = {
            "base": b,
            "pruned": p,
            "delta": delta,
            "raw_delta": raw_delta,
            "metric": p_key or b_key,
            "direction": direction,
            "delta_meaning": (
                "higher=pruned better"
                if direction == +1
                else "lower-is-better metric; delta sign-flipped so +=pruned better"
            ),
        }

    if not keep_checkpoints:
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)
        shutil.rmtree(pruned_dir, ignore_errors=True)

    return {
        "summary": table,
        "base_metrics": base_metrics,
        "pruned_metrics": pruned_metrics,
        "verification": verification,
    }


# --------------------------------------------------------------------------- #
# Subprocess entry point                                                       #
# --------------------------------------------------------------------------- #
# When this module is run as a script it IS the child spawned by
# ``_run_lm_eval_subprocess``: it reads a JSON job ({checkpoint_path, tasks,
# kwargs}) from stdin, runs the in-process benchmark, and prints the
# {task: {metric: value}} result on stdout between the sentinels. The process
# then exits and the OS reclaims every byte of its GPU memory — which is the
# whole point: no benchmark VRAM can leak back into the parent harness.
if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    import json as _json
    import sys as _sys


    _job = _json.loads(_sys.stdin.read())
    # CK_LM_EVAL_INPROC is set by the parent, so this calls the in-process body
    # directly (no recursion back into the subprocess dispatcher).
    _result = run_lm_eval(_job["checkpoint_path"], _job["tasks"], **_job["kwargs"])
    _a, _b = _RESULT_SENTINELS
    _sys.stdout.write(_a + _json.dumps(_result) + _b + "\n")
    _sys.stdout.flush()
