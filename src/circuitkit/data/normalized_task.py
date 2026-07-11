"""NormalizedTaskSpec — bridge a paired NormalizedDataset into the
existing TaskSpec interface so ``discover_circuit`` can consume it.

Workflow:

    raw  →  Adapter  →  NormalizedDataset  →  CorruptionStrategy
           (paired records)  →  NormalizedTaskSpec  →  discover_circuit

The bridge tokenises the clean_answer / corrupt_answer of each paired
record (using the model's tokenizer) to produce
``correct_idx`` / ``incorrect_idx`` token IDs in the EAP CSV format,
then hands the existing EAPDiscoveryDataset path the CSV.

Supports EAP / EAP-IG / ACDC discovery algorithms (any algorithm that
consumes the EAP-CSV interface).
"""

from __future__ import annotations

import csv
import logging
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .normalized import NormalizedDataset
from ..tasks._algorithm_families import ACDC_FAMILY, CDT_FAMILY, EAP_FAMILY, IB_FAMILY

logger = logging.getLogger(__name__)


def _assert_pairs_contrastive(
    n_identical: int, n_total: int, discovery_cfg: Dict[str, Any], context: str
) -> None:
    """Fail loud (or warn) when clean/corrupt pairs are identical.

    Mirrors ``GenericTaskSpec._check_corruption_effectiveness`` for the
    normalized-data path: ``fully_paired`` only means the corrupt half is
    *present*, not that it differs from the clean half, so identical pairs
    (zero contrastive signal) must be caught before EAP discovery runs.
    """
    if n_total == 0 or n_identical == 0:
        return
    frac = n_identical / n_total
    msg = (
        f"{n_identical}/{n_total} ({frac:.0%}) records have a corrupt prompt "
        f"identical to their clean prompt — {context}. EAP discovery has no "
        f"contrastive signal for these examples."
    )
    if n_identical == n_total and not discovery_cfg.get("allow_degenerate_corruption", False):
        raise ValueError(
            msg + " Every pair is identical, so the discovered circuit would be "
            "meaningless. Provide genuinely contrastive corrupt prompts, or set "
            "discovery config 'allow_degenerate_corruption'=True to run anyway."
        )
    logger.warning(msg)


class NormalizedTaskSpec:
    """TaskSpec implementation backed by a NormalizedDataset.

    Plug into ``circuitkit.api.discover_circuit`` by registering the
    instance with ``register_task(my_normalized_task_spec)`` and passing
    its name in the discovery config.
    """

    pair_padding_side = "left"

    def __init__(
        self,
        ds: NormalizedDataset,
        *,
        name: Optional[str] = None,
        cache_dir: str = "./cache/normalized",
    ):
        self._is_paired = ds.fully_paired
        self.ds = ds
        self.name = name or f"normalized:{ds.name}"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Instance-level padding side: start from class default, then let
        # dataset alignment meta override it if present.
        self.pair_padding_side = self.__class__.pair_padding_side
        recommended = ds.meta.get("_alignment", {}).get("recommended_pair_padding_side")
        if recommended in ("left", "right"):
            self.pair_padding_side = recommended

        # Honour the alignment pipeline's recommendation when present.
        # Overrides the class-level default of "left".
        recommended = ds.meta.get("_alignment", {}).get("recommended_pair_padding_side")
        if recommended in ("left", "right"):
            self.pair_padding_side = recommended

    # ------ TaskSpec protocol ----------------------------------------

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:

        algo = str(discovery_cfg.get("algorithm", "")).lower()
        allowed = EAP_FAMILY | ACDC_FAMILY | CDT_FAMILY | IB_FAMILY
        if algo not in allowed:
            raise ValueError(
                f"NormalizedTaskSpec supports algorithms {sorted(allowed)}; " f"got {algo!r}."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ):
        """Build the right dataloader for the requested algorithm.

        For EAP / ACDC / CD-T families: yields ``(clean_str, corrupt_str,
        label)`` tuples via the EAP-CSV path.

        For IBCircuit: yields dict batches with the keys IBCircuit
        expects: ``{tokens, labels, answer_positions}``. tokens is
        the clean_prompt + clean_answer token sequence; the
        answer_position is the index of the first answer token.
        """
        if model is None:
            raise ValueError("NormalizedTaskSpec.build_dataloader needs the model")

        algo = str(discovery_cfg.get("algorithm", "")).lower()
        batch_size = int(discovery_cfg.get("batch_size", 1))

        if algo in IB_FAMILY:
            data_params = discovery_cfg.get("data_params", {}) or {}
            num_examples = (
                data_params.get("num_examples") or discovery_cfg.get("num_examples") or 32
            )
            return self._build_ibcircuit_dataloader(
                model,
                batch_size,
                device,
                max_records=int(num_examples),
            )

        if algo in CDT_FAMILY:
            # CD-T only uses the clean prompt; serve a minimal EAP-format
            # dataloader with empty-string corrupted and dummy labels.
            return self._build_cdt_clean_only_dataloader(model, batch_size)

        if not self.ds.fully_paired:
            paired = self.ds.n_paired
            raise ValueError(
                f"Algorithm '{algo}' requires fully-paired data; got "
                f"{paired}/{len(self.ds)} paired records. "
                f"Apply a CorruptionStrategy first, or use ibcircuit/cdt "
                f"with clean-only data."
            )

        # fully_paired only means the corrupt half is present, not that it
        # differs from the clean half. Catch degenerate (identical) pairs before
        # discovery runs on a zero-signal contrast.
        n_identical = sum(
            1 for r in self.ds.records if r.corrupt_prompt == r.clean_prompt
        )
        _assert_pairs_contrastive(
            n_identical, len(self.ds), discovery_cfg,
            context="the corrupt prompt equals the clean prompt",
        )

        data_params = discovery_cfg.get("data_params", {})
        seed = data_params.get("seed", discovery_cfg.get("seed", 42))
        return self._build_eap_dataloader(model, batch_size, seed)

    def _build_eap_dataloader(self, model, batch_size: int, seed: int = 42):
        from .eap_dataset import EAPDiscoveryDataset

        align_tag = self.ds.meta.get("_alignment", {}).get("align_strategy", "unk")
        cache_path = (
            self.cache_dir
            / f"{_safe_name(self.name)}_{len(self.ds)}_{align_tag}_seed{seed}.csv"
        )
        if not cache_path.exists():
            self._materialise_eap_csv(model, cache_path, seed)
        else:
            logger.info(f"Reusing cached EAP CSV: {cache_path}")
        ds = EAPDiscoveryDataset(str(cache_path))
        return ds.to_dataloader(
            batch_size=batch_size,
            pair_padding_side=self.pair_padding_side,
        )

    def _build_ibcircuit_dataloader(
        self, model, batch_size: int, device: str, max_records: int = 32
    ):
        """IBCircuit-format dataloader: yields ONE batch containing up to
        ``max_records`` records, mirroring the built-in tasks (BoolQ /
        MMLU). IBCircuit only reads the first batch of the loader and
        computes ``std_mean(activation, dim=0)`` across the batch
        dimension — with batch_size=1 that std is undefined / NaN. We
        bypass the caller's batch_size and stuff every record into a
        single batch, capped at ``max_records`` to keep the
        [N, max_seq_len, vocab] logits tensor manageable. The cap comes
        from the caller's ``data_params.num_examples`` (default 32).

        Uses ``model.to_tokens(text, prepend_bos=True)`` so the BOS
        token is present (built-in IBCircuit dataloaders all do this);
        answer_position is the index of the LAST real prompt token, so
        ``logits[answer_position]`` predicts the answer.
        """
        import torch

        tokenizer = model.tokenizer

        # Probe for the leading-whitespace token — needed as a fallback for
        # tokenizers that emit a bare space token before short answers.
        try:
            ws_probe = tokenizer.encode(" ", add_special_tokens=False)
            ws_token_id = ws_probe[0] if len(ws_probe) == 1 else None
        except Exception:  # noqa: BLE001
            ws_token_id = None

        token_lists = []
        label_ids = []
        skipped = 0
        for r in self.ds.records:
            if max_records and len(token_lists) >= max_records:
                break
            try:
                toks = model.to_tokens(r.clean_prompt, prepend_bos=True).squeeze(0).cpu()
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            if toks.numel() == 0:
                skipped += 1
                continue

            precomputed = r.meta.get("_precomputed_labels")
            if precomputed:
                ans_token = precomputed["clean_label_id"]
            else:
                # Joint encoding: tokenize prompt+answer together so the
                # tokenizer sees the same context as the model at inference
                # time. This resolves leading-space ambiguity (e.g. "No" vs
                # " No" after "Assistant:") in a model-invariant way.
                prompt_ids_solo = tokenizer.encode(
                    r.clean_prompt, add_special_tokens=False
                )
                full_ids = tokenizer.encode(
                    r.clean_prompt + r.clean_answer, add_special_tokens=False
                )
                boundary_clean = (
                    len(full_ids) > len(prompt_ids_solo)
                    and full_ids[: len(prompt_ids_solo)] == prompt_ids_solo
                )
                if boundary_clean:
                    first_cont = int(full_ids[len(prompt_ids_solo)])
                    # If the first continuation token is a bare space token,
                    # skip it and use the next token (same logic as standalone
                    # fallback), so both two-token and merged-space tokenizers
                    # converge on the actual answer token.
                    if (
                        ws_token_id is not None
                        and first_cont == ws_token_id
                        and len(full_ids) > len(prompt_ids_solo) + 1
                    ):
                        ans_token = int(full_ids[len(prompt_ids_solo) + 1])
                    else:
                        ans_token = first_cont
                else:
                    # Boundary re-tokenized (prompt+answer collapsed a token
                    # across the join); fall back to standalone encoding with
                    # leading-whitespace-token skip.
                    ans_ids = tokenizer.encode(r.clean_answer, add_special_tokens=False)
                    if not ans_ids:
                        skipped += 1
                        continue
                    ans_token = ans_ids[0]
                    if (
                        ws_token_id is not None
                        and ans_token == ws_token_id
                        and len(ans_ids) > 1
                    ):
                        ans_token = ans_ids[1]

            token_lists.append(toks)
            label_ids.append(ans_token)

        if not token_lists:
            raise RuntimeError(
                f"NormalizedTaskSpec[{self.name}] produced 0 IBCircuit "
                f"records (skipped {skipped})."
            )

        # Cap records the same way the apps caller would (data_params.num_examples
        # would otherwise be the EAP-tuple cap; here we just respect the call-site
        # batch_size when it looks like a per-batch chunk and use it as a soft cap
        # only when the dataset is large).
        # For now: use everything; the caller controls breadth via
        # data_params.num_examples in the NormalizedDataset itself.

        # Right-pad to uniform length with pad / eos.
        max_len = max(t.shape[0] for t in token_lists)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        padded = []
        for t in token_lists:
            gap = max_len - t.shape[0]
            if gap > 0:
                t = torch.cat([t, torch.full((gap,), pad_id, dtype=torch.long)])
            padded.append(t)

        tokens = torch.stack(padded).to(device)
        labels = torch.tensor(label_ids, dtype=torch.long, device=device)
        # answer_position = last real (pre-pad) prompt token: logits[p]
        # at that index predicts the answer.
        answer_positions = torch.tensor(
            [t.shape[0] - 1 for t in token_lists],
            dtype=torch.long,
            device=device,
        )

        batch = {
            "tokens": tokens,
            "labels": labels,
            "answer_positions": answer_positions,
        }

        class _SingleBatchLoader:
            def __init__(self, b):
                self.batch = b

            def __iter__(self):
                yield self.batch

            def __len__(self):
                return 1

        return _SingleBatchLoader(batch)
    
    def _build_cdt_clean_only_dataloader(self, model, batch_size: int):
        """CD-T dataloader for clean-only data.

        CD-T destructures ``(clean, corrupted, labels)`` from the loader but
        only consumes ``clean``. We serve a standard EAP-format CSV loader
        with an empty-string corrupted column and a dummy label (0) so the
        CD-T backend can iterate without modification.
        """
        cache_path = self.cache_dir / f"{_safe_name(self.name)}_{len(self.ds)}_cdt_clean.csv"
        if not cache_path.exists():
            rows = [
                {
                    "clean": r.clean_prompt,
                    "corrupted": "",
                    "correct_idx": 0,
                    "incorrect_idx": 0,
                }
                for r in self.ds.records
            ]
            with open(cache_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"],
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            logger.info(f"Wrote CD-T clean-only CSV: {cache_path} ({len(rows)} rows)")
        else:
            logger.info(f"Reusing cached CD-T CSV: {cache_path}")

        from .eap_dataset import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(str(cache_path))
        return ds.to_dataloader(
            batch_size=batch_size,
            pair_padding_side=self.pair_padding_side,
        )

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the metric function.

        ``logit_diff`` (default): single-token clean vs corrupt logit
        difference. Standard EAP / AtP / RelP metric.

        ``kl_divergence``: multi-token-friendly KL between the model's
        full distribution at the last token and the reference
        (clean_logits) distribution. Use when answers are multi-token
        and first-subword truncation loses information.
        """
        from ..api import _eap_kl_divergence, _eap_logit_diff

        if metric_type == "kl_divergence":
            return partial(_eap_kl_divergence, mean=True)
        if metric_type != "logit_diff":
            raise ValueError(
                f"NormalizedTaskSpec supports metric_type='logit_diff' "
                f"only; got {metric_type!r}."
            )
        return partial(_eap_logit_diff, loss=True, mean=True)

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task": self.name,
            "data_source": self.ds.source,
            "shape": self.ds.shape.value,
            "n_records": len(self.ds),
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
        }

    # ------ Internals -------------------------------------------------

    def _materialise_eap_csv(self, model, out_path: Path, seed: int = 42) -> None:
        """Convert the NormalizedDataset to the CSV format EAPDiscoveryDataset wants.

        ``seed`` controls the order records are written to the CSV. The
        EAP DataLoader consumes the CSV in file order (it does not shuffle),
        so this ordering is the only lever that makes per-seed stability
        runs see genuinely different batch composition — without it, every
        seed would produce byte-identical data and fabricate perfect
        stability. The dataset *contents* are unchanged; only the order is.

        Several silent-failure modes are filtered here:

        * Empty answer tokenisation (skipped, logged).
        * The "leading-space token" failure: some BPE tokenisers (Llama,
          Mistral, Gemma) emit a separate leading-whitespace token for
          " 18", giving ``encode(" 18") = [' ', '18']``. Naively taking
          ``[0]`` returns the space-token id, which is identical for
          " 18" and " 27" — degenerate metric. We strip a leading-only
          whitespace-token and use the next token as the answer. If the
          stripped sequence is still empty, the record is dropped.
        * Records where ``correct_idx == incorrect_idx`` after the
          tokenisation (e.g., both answers tokenise to the same first
          subword): dropped, as the EAP gradient on these is identically
          zero.
        * Records with ``_precomputed_labels`` in meta: labels taken directly
          from the alignment pipeline's joint-tokenization result; the empty
          and same-target checks are skipped for these records since
          ``check_answer_discriminative`` already guaranteed divergence.
        * Records where ``clean_prompt == corrupt_prompt``: dropped,
          since EAP's activation_difference would be zero.
        """
        tokenizer = model.tokenizer
        rows = []
        skipped_empty = 0
        skipped_same_target = 0
        skipped_same_prompt = 0

        # Detect the "leading whitespace token" once per tokenizer.
        try:
            ws_probe = tokenizer.encode(" ", add_special_tokens=False)
            ws_token_id = ws_probe[0] if len(ws_probe) == 1 else None
        except Exception:  # noqa: BLE001
            ws_token_id = None

        def _first_meaningful_token(s: str):
            ids = tokenizer.encode(s, add_special_tokens=False)
            if not ids:
                return None
            # If the first token is the bare-space token, skip it and use
            # the next token as the answer head.
            if ws_token_id is not None and ids[0] == ws_token_id and len(ids) > 1:
                return ids[1]
            return ids[0]

        # Seeded record ordering: different seeds yield different batch
        # composition downstream (the DataLoader itself does not shuffle).
        records = list(self.ds.records)
        random.Random(seed).shuffle(records)

        for r in records:
            if r.clean_prompt == r.corrupt_prompt:
                skipped_same_prompt += 1
                continue

            # Use pre-computed discriminative labels when the alignment
            # pipeline has already found the correct divergence token via
            # joint tokenization (D5). Falls back to standalone tokenization
            # for datasets that bypassed the alignment pass.
            precomputed = r.meta.get("_precomputed_labels")
            if precomputed:
                correct_idx = precomputed["clean_label_id"]
                incorrect_idx = precomputed["corrupt_label_id"]
            else:
                correct_idx = _first_meaningful_token(r.clean_answer)
                incorrect_idx = _first_meaningful_token(r.corrupt_answer)
                if correct_idx is None or incorrect_idx is None:
                    skipped_empty += 1
                    continue
                if correct_idx == incorrect_idx:
                    skipped_same_target += 1
                    continue

            rows.append(
                {
                    "clean": r.clean_prompt,
                    "corrupted": r.corrupt_prompt,
                    "correct_idx": correct_idx,
                    "incorrect_idx": incorrect_idx,
                }
            )
        if not rows:
            raise RuntimeError(
                f"All {len(self.ds)} records were filtered out: "
                f"{skipped_empty} empty / {skipped_same_target} same target / "
                f"{skipped_same_prompt} same prompt. Check the tokenizer-vs-data "
                f"compatibility and consider switching corruption strategy."
            )
        if skipped_empty or skipped_same_target or skipped_same_prompt:
            logger.warning(
                f"NormalizedTaskSpec dropped {skipped_empty} empty + "
                f"{skipped_same_target} same-target + {skipped_same_prompt} "
                f"same-prompt records out of {len(self.ds)}"
            )
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info(f"Wrote EAP CSV: {out_path} ({len(rows)} rows)")


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def validate_token_alignment(task_spec: "NormalizedTaskSpec", model=None) -> dict:
    """Audit token alignment for all records in task_spec.

    Reports counts and fractions of records that would be dropped by EAP's
    same-prompt filter, have empty prompts, or have answer tokens that
    tokenize to more than one token.

    Args:
        task_spec: A NormalizedTaskSpec instance.
        model: Optional HookedTransformer. If provided, multi-token answer
               detection uses actual tokenization; otherwise falls back to
               whitespace splitting.

    Returns:
        dict with keys:
            total, same_prompt, same_prompt_frac,
            empty_prompt, empty_prompt_frac,
            multi_token_answer, multi_token_answer_frac,
            dropped_frac (same_prompt + empty combined),
            records_ok
    """
    records = getattr(getattr(task_spec, "ds", None), "records", None)
    if records is None:
        return {"error": "task_spec has no .ds.records"}

    total = len(records)
    same_prompt = 0
    empty_prompt = 0
    multi_token_answer = 0

    for r in records:
        cp = getattr(r, "clean_prompt", "")
        pp = getattr(r, "corrupt_prompt", "")
        ca = getattr(r, "clean_answer", "")

        if not cp:
            empty_prompt += 1
            continue
        if cp == pp:
            same_prompt += 1
        if ca:
            if model is not None:
                try:
                    toks = model.to_tokens(ca, prepend_bos=False)
                    if toks.shape[1] > 1:
                        multi_token_answer += 1
                except Exception:
                    pass
            else:
                if len(ca.split()) > 1:
                    multi_token_answer += 1

    dropped = same_prompt + empty_prompt
    return {
        "total": total,
        "same_prompt": same_prompt,
        "same_prompt_frac": round(same_prompt / total, 4) if total else 0,
        "empty_prompt": empty_prompt,
        "empty_prompt_frac": round(empty_prompt / total, 4) if total else 0,
        "multi_token_answer": multi_token_answer,
        "multi_token_answer_frac": round(multi_token_answer / total, 4) if total else 0,
        "dropped_frac": round(dropped / total, 4) if total else 0,
        "records_ok": total - dropped,
    }


__all__ = ["NormalizedTaskSpec", "validate_token_alignment"]
