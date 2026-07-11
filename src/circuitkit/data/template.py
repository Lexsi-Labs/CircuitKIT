"""template_normalize — one-call template-based dataset construction.

Usage:
    from circuitkit.data.template import template_normalize
    ds = template_normalize("my_pairs.csv", template_spec={...})
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .corruption.template_utils import (
    build_pair_from_templates,
    check_answer_discriminative,
    check_token_alignment,
    detect_peer_columns,
    pad_question_region,
    parse_placeholders,
    resolve_template,
    validate_placeholders_against_columns,
)
from .normalized import ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset


def template_normalize(
    raw: Any,
    template_spec: Dict[str, str],
    *,
    pairing_mode: str = "explicit",
    align_strategy: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    pad_region_end: Optional[str] = None,
    max_records: Optional[int] = None,
    name: Optional[str] = None,
    source: Optional[str] = None,
) -> NormalizedDataset:
    """Build a fully-paired NormalizedDataset from raw data + template spec.

    Args:
        raw:            CSV path (str), pandas DataFrame, or list of dicts.
        template_spec:  Dict with keys clean_prompt, corrupt_prompt,
                        clean_answer, corrupt_answer.
        pairing_mode:   "explicit" — all placeholders (incl. other_*) must
                        exist as CSV columns; "auto_peer" — other_* values
                        are taken from a randomly-selected peer row.
        align_strategy: One of "filter", "pad_question", or "none". Defaults to
                        "filter" when a tokenizer is given, else "none".
                        Controls how token-length misaligned pairs are handled.
                        "filter" drops misaligned pairs. "pad_question" pads
                        neutral tokens into the corrupt prompt's question region
                        (before pad_region_end) to reach the clean length.
                        "none" skips alignment enforcement entirely.
        tokenizer:      HuggingFace-compatible tokenizer. Required for
                        "filter" and "pad_question" strategies; optional for
                        "none".
        pad_region_end: Boundary string marking the end of the paddable region
                        in the corrupt prompt. Required when align_strategy is
                        "pad_question". E.g. "Answer:".
        max_records:    Optional cap on output record count.
        name:           Dataset name (defaults to "template").
        source:         Provenance string.

    Returns:
        NormalizedDataset with shape=TEMPLATE. Under "filter" and
        "pad_question" strategies, all records are token-aligned and have
        pre-computed discriminative labels in record.meta["_precomputed_labels"].
        Dataset-level alignment stats are stored in ds.meta["_alignment"].

    Raises:
        ValueError: if align_strategy is "filter" or "pad_question" and no
                    tokenizer is provided; if align_strategy is "pad_question"
                    and pad_region_end is not provided; or if no rows are found.
        RuntimeError: if all records are dropped after alignment passes.
        TypeError: if raw is not a supported type.
    """
    # Default: align when a tokenizer is available (better EAP signal), else
    # skip alignment. Avoids the footgun where the simplest call —
    # template_normalize(csv, spec) with no tokenizer — would otherwise raise.
    if align_strategy is None:
        align_strategy = "filter" if tokenizer is not None else "none"

    # ── Guard: tokenizer requirements ────────────────────────────────────────
    if align_strategy in ("filter", "pad_question") and tokenizer is None:
        raise ValueError(
            f"align_strategy={align_strategy!r} requires a tokenizer. "
            f"Pass tokenizer=model.tokenizer or use align_strategy='none'."
        )
    if align_strategy == "pad_question" and not pad_region_end:
        raise ValueError(
            "align_strategy='pad_question' requires pad_region_end to be set. "
            "E.g. pad_region_end='Answer:'."
        )
    if align_strategy not in ("filter", "pad_question", "none"):
        raise ValueError(
            f"align_strategy must be 'filter', 'pad_question', or 'none'; "
            f"got {align_strategy!r}."
        )

    # ── 1. Load raw data ─────────────────────────────────────────────────────
    rows: List[Dict[str, Any]]
    if isinstance(raw, str):
        import pandas as pd
        rows = pd.read_csv(raw).to_dict("records")
    elif hasattr(raw, "to_dict"):
        rows = raw.to_dict("records")
    elif isinstance(raw, list):
        rows = raw
    else:
        raise TypeError(f"Unsupported raw type: {type(raw)!r}")

    if not rows:
        raise ValueError("No rows found in raw data.")

    columns = list(rows[0].keys())

    # ── 2. Validate templates and detect peer structure ───────────────────────
    all_placeholders: List[str] = []
    for tmpl in template_spec.values():
        all_placeholders.extend(parse_placeholders(tmpl))
    unique_placeholders = list(set(all_placeholders))

    if pairing_mode == "explicit":
        missing = validate_placeholders_against_columns(template_spec, columns)
        if missing:
            raise ValueError(
                f"Template placeholder(s) {missing!r} not found in CSV columns: {columns}. "
                f"Add the missing column(s) or use pairing_mode='auto_peer'."
            )
        peer_map: Dict[str, str] = {}
    else:  # auto_peer
        peer_map, direct = detect_peer_columns(unique_placeholders)
        missing_direct = [p for p in direct if p not in columns]
        if missing_direct:
            raise ValueError(
                f"Template placeholder(s) {missing_direct!r} not found in CSV columns: {columns}."
            )

    # ── 3. Build records ──────────────────────────────────────────────────────

    if max_records is not None:
        rows = rows[:max_records]

    records = []
    for i, row in enumerate(rows):
        row_str = {k: str(v) for k, v in row.items()}

        if pairing_mode == "explicit":
            clean_values = row_str
            corrupt_values = row_str
        else:
            peer_idx = random.choice(
                [j for j in range(len(rows)) if j != i] or [i]
            )
            peer_row = {k: str(v) for k, v in rows[peer_idx].items()}
            corrupt_values = dict(row_str)
            for other_key, base_key in peer_map.items():
                corrupt_values[other_key] = peer_row.get(base_key, "")
            clean_values = row_str

        clean_prompt, corrupt_prompt, clean_answer, corrupt_answer = build_pair_from_templates(
            template_spec, clean_values, corrupt_values
        )

        records.append(
            ContrastiveRecord(
                record_id=f"{i:06d}",
                clean_prompt=clean_prompt,
                corrupt_prompt=corrupt_prompt,
                clean_answer=clean_answer,
                corrupt_answer=corrupt_answer,
                contrast_source=ContrastSource.GENERATED,
                target_field="template_answer",
                meta={"_template_values": row_str, "_corruption": "template"},
            )
        )

    # ── 4. Alignment passes ───────────────────────────────────────────────────
    total_input = len(records)
    dropped_nondiscriminative = 0
    dropped_misaligned = 0
    dropped_pad_failed = 0
    answer_prefix_absorbed = 0

    if align_strategy == "none":
        kept_records = records
    else:
        # Pass A — Answer discrimination + shared prefix absorption
        after_pass_a: List[ContrastiveRecord] = []
        for r in records:
            result = check_answer_discriminative(
                r.clean_prompt,
                r.clean_answer,
                r.corrupt_prompt,
                r.corrupt_answer,
                tokenizer,
            )
            if not result.discriminative:
                dropped_nondiscriminative += 1
                continue

            # Absorb shared prefix tokens into the prompt strings in-place.
            # ContrastiveRecord is a mutable dataclass — direct assignment is safe.
            if result.shared_prefix_len > 0:
                r.clean_prompt = result.adjusted_clean_prompt
                r.corrupt_prompt = result.adjusted_corrupt_prompt
                answer_prefix_absorbed += 1

            r.meta["_precomputed_labels"] = {
                "clean_label_id": result.clean_label_id,
                "corrupt_label_id": result.corrupt_label_id,
            }
            after_pass_a.append(r)

        # Pass B — Token alignment
        kept_records = []
        for r in after_pass_a:
            align = check_token_alignment(r.clean_prompt, r.corrupt_prompt, tokenizer)
            if align.aligned:
                kept_records.append(r)
                continue

            if align_strategy == "filter":
                dropped_misaligned += 1
                continue

            # pad_question: corrupt is shorter than clean → try padding.
            # If corrupt is already longer, padding cannot help → misaligned drop.
            if align.diff > 0:
                dropped_misaligned += 1
                continue

            try:
                padded, exact = pad_question_region(
                    r.corrupt_prompt,
                    target_len=align.clean_len,
                    tokenizer=tokenizer,
                    pad_boundary=pad_region_end,
                )
            except ValueError:
                # pad_boundary not found in this record's corrupt prompt.
                dropped_pad_failed += 1
                continue

            if not exact:
                # Could not reach target_len within max_iterations.
                dropped_pad_failed += 1
                continue

            r.corrupt_prompt = padded
            kept_records.append(r)

        dropped_misaligned = (
            dropped_misaligned if align_strategy == "filter"
            else 0
        )

    kept = len(kept_records)

    if kept == 0:
        raise RuntimeError(
            f"All {total_input} records were dropped during alignment "
            f"(strategy={align_strategy!r}).\n"
            f"  dropped_nondiscriminative : {dropped_nondiscriminative}\n"
            f"  dropped_misaligned        : {dropped_misaligned}\n"
            f"  dropped_pad_failed        : {dropped_pad_failed}\n"
            f"Suggestions:\n"
            f"  - Check that clean_answer and corrupt_answer differ in meaning.\n"
            f"  - For 'pad_question', verify pad_region_end appears in every corrupt prompt.\n"
            f"  - Use align_strategy='none' to skip alignment (recommended_metric: kl_divergence)."
        )

    alignment_meta: Dict[str, Any] = {
        "align_strategy": align_strategy,
        "total_input": total_input,
        "kept": kept,
        "dropped_nondiscriminative": dropped_nondiscriminative,
        "dropped_misaligned": dropped_misaligned,
        "dropped_pad_failed": dropped_pad_failed,
        "answer_prefix_absorbed": answer_prefix_absorbed,
        "recommended_pair_padding_side": "left",
        "recommended_metric": (
            "kl_divergence" if align_strategy == "none" else "logit_diff"
        ),
    }

    ds = NormalizedDataset(
        name=name or "template",
        shape=DatasetShape.TEMPLATE,
        records=kept_records,
        source=source or (str(raw) if isinstance(raw, str) else "raw"),
        meta={
            "template_spec": template_spec,
            "pairing_mode": pairing_mode,
            "_alignment": alignment_meta,
        },
    )
    return ds

def clean_only_from_template(
    raw: Any,
    template_spec: Dict[str, str],
    *,
    max_records: Optional[int] = None,
    name: Optional[str] = None,
    source: Optional[str] = None,
) -> NormalizedDataset:
    """Extract only the clean side from a template CSV for IBCircuit / CD-T.

    Renders ``clean_prompt`` and optionally ``clean_answer`` from the template
    spec against each CSV row.  No corrupt side is produced; records carry
    ``corrupt_prompt=None``.  Use when the algorithm does not require
    contrastive pairs (ibcircuit, cdt).

    Args:
        raw:           CSV path (str), pandas DataFrame, or list of dicts.
        template_spec: Dict containing at minimum ``clean_prompt``.
                       ``clean_answer`` is optional — omit or set to ``""``
                       for CD-T which ignores answers.
        max_records:   Optional cap on output record count.
        name:          Dataset name. Defaults to the source path stem.
        source:        Provenance string stored in NormalizedDataset.source.

    Returns:
        NormalizedDataset with shape=CLEAN_ONLY and all records unpaired.

    Raises:
        ValueError: If ``clean_prompt`` key is missing from template_spec,
                    if its placeholders are absent from the CSV columns,
                    or if raw is an unsupported type.
    """
    if "clean_prompt" not in template_spec:
        raise ValueError(
            "template_spec must contain 'clean_prompt'. "
            f"Got keys: {sorted(template_spec.keys())}"
        )

    # ── 1. Load rows ──────────────────────────────────────────────────────────
    if isinstance(raw, str):
        import pandas as pd
        rows = pd.read_csv(raw).to_dict("records")
    elif hasattr(raw, "to_dict"):
        rows = raw.to_dict("records")
    elif isinstance(raw, list):
        rows = raw
    else:
        raise TypeError(f"Unsupported raw type: {type(raw)!r}")

    if not rows:
        raise ValueError("clean_only_from_template received an empty dataset.")

    columns = list(rows[0].keys())

    # ── 2. Validate clean-side placeholders exist in CSV ─────────────────────
    clean_spec = {k: v for k, v in template_spec.items() if k in ("clean_prompt", "clean_answer")}
    missing = validate_placeholders_against_columns(clean_spec, columns)
    if missing:
        raise ValueError(
            f"Template placeholder(s) {missing!r} not found in CSV columns: {columns}."
        )

    # ── 3. Build records ──────────────────────────────────────────────────────
    if max_records is not None:
        rows = rows[:max_records]

    records = []
    for i, row in enumerate(rows):
        row_str = {k: str(v) for k, v in row.items()}
        clean_prompt = resolve_template(template_spec["clean_prompt"], row_str)
        clean_answer = (
            resolve_template(template_spec["clean_answer"], row_str)
            if template_spec.get("clean_answer")
            else ""
        )
        records.append(
            ContrastiveRecord(
                record_id=f"{i:06d}",
                clean_prompt=clean_prompt,
                clean_answer=clean_answer,
                corrupt_prompt=None,
                corrupt_answer=None,
                contrast_source=ContrastSource.NOT_PAIRED_YET,
                meta={"_template_values": row_str},
            )
        )

    return NormalizedDataset(
        name=name or (str(raw) if isinstance(raw, str) else "template_clean_only"),
        shape=DatasetShape.CLEAN_ONLY,
        records=records,
        source=source or (str(raw) if isinstance(raw, str) else "raw"),
    )
