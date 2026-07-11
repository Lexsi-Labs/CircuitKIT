"""General-text calibration data for intervention-native selectors.

Wanda, GPTQ and AWQ all derive their importance signal from *activation
statistics* collected over a calibration corpus. In their source papers that
corpus is **general text** (Wanda: C4; GPTQ: C4/WikiText2; AWQ: Pile/general
text) — explicitly NOT the downstream task being evaluated. Using the task's
own EAP dataloader would make these selectors task-aware in a way the original
algorithms are not, which is exactly the conformance gap the audit flagged.

This module provides a single helper, :func:`wikitext_calibration_batches`,
that reproduces the standard WikiText-2 calibration protocol used by the Wanda
and GPTQ reference repos: concatenate the train split, tokenize once, and slice
into fixed-length non-overlapping windows.

The circuit-discovery selectors (EAP, etc.) are intentionally left untouched —
they are *supposed* to calibrate on the task.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_CACHE: dict = {}


def _resolve_seqlen(model, seqlen):
    """Pick a calibration window length.

    Default = min(model context length, 2048). GPT-2 has n_ctx=1024 so it
    naturally resolves to 1024, matching the Wanda/GPTQ WikiText2 protocol.
    """
    if seqlen is not None:
        return int(seqlen)
    model_ctx = None
    cfg = getattr(model, "cfg", None)
    if cfg is not None:
        model_ctx = getattr(cfg, "n_ctx", None)
    if model_ctx is None:
        model_ctx = 2048
    return int(min(model_ctx, 2048))


def _load_wikitext2_train_text():
    """Return the WikiText-2-raw-v1 train split as a list of text rows.

    Primary path is ``datasets.load_dataset``. That path fingerprints via dill,
    which can raise ``RuntimeError: RLock objects should only be shared ...``
    when process-global state is polluted (e.g. across a long test session).
    On any failure we fall back to fetching the split's parquet file directly
    with ``huggingface_hub`` + ``pyarrow`` — no datasets fingerprinting, fully
    robust. Real (fresh-process) runs take the fast primary path.
    """
    try:
        import datasets as _ds
        from datasets import load_dataset

        _caching_was_enabled = _ds.is_caching_enabled()
        _ds.disable_caching()
        try:
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", keep_in_memory=True)
        finally:
            if _caching_was_enabled:
                _ds.enable_caching()
        return list(data["text"])
    except Exception as exc:  # noqa: BLE001 - robust fallback below
        logger.warning(
            "datasets.load_dataset failed for WikiText-2 (%s); "
            "falling back to direct parquet fetch.",
            exc,
        )
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files

        repo = "Salesforce/wikitext"
        files = [
            f
            for f in list_repo_files(repo, repo_type="dataset")
            if f.endswith(".parquet") and "wikitext-2-raw-v1" in f and "train" in f
        ]
        if not files:
            raise RuntimeError(
                "Could not locate the WikiText-2-raw-v1 train parquet on "
                f"the '{repo}' dataset repo."
            ) from exc
        text_rows = []
        for fname in sorted(files):
            path = hf_hub_download(repo, fname, repo_type="dataset")
            text_rows.extend(pq.read_table(path).column("text").to_pylist())
        return text_rows


def wikitext_calibration_batches(model, n_samples: int = 128, seqlen=None):
    """Return ``n_samples`` general-text calibration token windows.

    Reproduces the Wanda / GPTQ WikiText-2 calibration protocol: load the
    ``wikitext`` ``wikitext-2-raw-v1`` ``train`` split, concatenate every
    document into one stream, tokenize the stream once, then slice it into
    ``n_samples`` non-overlapping windows of length ``seqlen``.

    Args:
        model: a HookedTransformer (or anything exposing ``.to_tokens`` and a
            ``.cfg.n_ctx``). Used for tokenization and device placement.
        n_samples: number of calibration windows to return.
        seqlen: window length. Defaults to ``min(model n_ctx, 2048)`` — i.e.
            1024 for GPT-2.

    Returns:
        list[Tensor]: ``n_samples`` tensors of shape ``[1, seqlen]`` (long),
        placed on the model's device, ready to feed to ``run_with_cache``.
    """
    seqlen = _resolve_seqlen(model, seqlen)
    device = next(model.parameters()).device

    # Cache by a STABLE model identity, not id(model): CPython reuses id()s
    # after garbage collection, so id()-keyed entries would hand a freshly
    # built model the calibration tokens of a freed, possibly differently
    # tokenized model. Key on model_name; skip caching if it is unavailable.
    model_name = getattr(getattr(model, "cfg", None), "model_name", None)
    cache_key = (model_name, n_samples, seqlen) if model_name else None
    if cache_key is not None and cache_key in _CACHE:
        return [b.to(device) for b in _CACHE[cache_key]]

    logger.info(
        "Loading WikiText-2 (wikitext-2-raw-v1/train) calibration data: " "n_samples=%d seqlen=%d",
        n_samples,
        seqlen,
    )
    full_text = "\n\n".join(_load_wikitext2_train_text())

    # Tokenize the concatenated stream exactly once. prepend_bos=False so the
    # windows tile a single continuous stream (Wanda/GPTQ do not inject BOS
    # between windows of the WikiText2 calibration stream). truncate=False is
    # essential: to_tokens otherwise truncates to the model's context length,
    # which would cap the whole calibration stream at ~n_ctx tokens — far too
    # short to slice into n_samples windows.
    all_ids = model.to_tokens(full_text, prepend_bos=False, truncate=False)
    all_ids = all_ids.reshape(-1)

    needed = n_samples * seqlen
    if all_ids.numel() < needed:
        usable = all_ids.numel() // seqlen
        if usable == 0:
            raise RuntimeError(
                f"WikiText-2 stream ({all_ids.numel()} tokens) too short for a "
                f"single window of seqlen={seqlen}."
            )
        logger.warning(
            "WikiText-2 stream yields only %d windows of seqlen=%d " "(requested %d); using %d.",
            usable,
            seqlen,
            n_samples,
            usable,
        )
        n_samples = usable
        needed = n_samples * seqlen

    windows = all_ids[:needed].reshape(n_samples, seqlen).contiguous()
    batches = [windows[i : i + 1].to(device) for i in range(n_samples)]

    if cache_key is not None:
        _CACHE[cache_key] = batches
    return batches
