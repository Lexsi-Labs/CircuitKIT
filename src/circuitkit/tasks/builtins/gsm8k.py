"""
GSM8K Task Specification — open-ended generation circuit discovery.

GSM8K (Cobbe et al. 2021, ``openai/gsm8k`` config ``"main"``) is a grade-school
math word-problem benchmark. Unlike CircuitKit's classification / MCQ tasks
(which use a logit-difference metric over A/B/C/D tokens), GSM8K is an
*open-ended generation* task: the model must produce a numeric final answer.

This TaskSpec wires GSM8K into EAP / EAP-IG circuit discovery:

* **Prompt format.** Each clean prompt is the word problem plus its
  chain-of-thought reasoning trace (GSM8K's annotated solution, including the
  ``<<a op b=R>>`` calculator steps), ending with ``"The answer is"``. The
  model must predict the final numeric answer.

* **Contrastive corruption.** The ``final_answer_swap`` strategy perturbs the
  operand of the *final* calculator step so the computed final answer changes
  (e.g. ``<<48+24=72>>`` -> ``<<48+25=73>>``). Both the prompt and the target
  answer are edited, so clean and corrupt prompts genuinely differ — a valid
  contrast for activation-patching attribution — and the correct answer
  differs, giving the differentiable NLL metric real signal.

* **Discovery metric.** A differentiable negative-log-likelihood (cross-entropy)
  on the answer span, reusing ``perplexity_loss_span`` from
  ``backends/eap/metrics.py``. Lower loss = circuit more faithfully reproduces
  the correct final answer. This is the open-ended-generation analogue of the
  logit-diff metric used by classification tasks.

Supports EAP and EAP-IG. ACDC / IBCircuit are intentionally not supported.
"""

from __future__ import annotations

import random as _random
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ...data.eap_dataset import EAPDiscoveryDataset
from ...backends.eap.metrics import perplexity_loss_span
from ...data.corruption.final_answer_swap import FinalAnswerSwap
from ...data.normalized import ContrastiveRecord, ContrastSource
from ...utils.logging import get_logger
from .._algorithm_families import CDT_FAMILY, EAP_FAMILY, IB_FAMILY, is_eap_family, unsupported_algorithm_message
from .._chat import (
    resolve_chat_template,
    resolve_chat_template_from_tokenizer,
    to_tokens,
    wrap_prompt,
    wrap_prompt_with_tokenizer,
)

logger = get_logger("task.gsm8k")

#: Text placed at the start of the assistant turn — keeps the numeric answer
#: the immediate next token so the single-token NLL metric stays valid.
_ANSWER_TAIL = "\nThe answer is"


_GSM_FINAL = re.compile(r"####\s*(-?[\d,]+)")
_CALC_STEP = re.compile(r"<<[^>]*>>")


def _extract_final_answer(answer_text: str) -> Optional[str]:
    """Pull the final numeric answer out of a GSM8K ``answer`` field."""
    m = _GSM_FINAL.search(answer_text or "")
    if not m:
        return None
    return m.group(1).replace(",", "").strip()


def _reasoning_trace(answer_text: str) -> str:
    """Return the chain-of-thought reasoning, dropping the ``#### N`` line."""
    body = _GSM_FINAL.split(answer_text or "")[0]
    return body.strip()


def _format_prompt(question: str, reasoning: str) -> str:
    """Build a clean GSM8K prompt ending right before the numeric answer.

    The reasoning trace (with its ``<<a op b=R>>`` calculator annotations) is
    kept verbatim — the final calculator step telegraphs the answer, which is
    exactly what makes this a tractable open-ended-generation circuit.
    """
    return f"{question.strip()}\n{reasoning}{_ANSWER_TAIL}"


def _split_answer_tail(prompt: str) -> Tuple[str, str]:
    """Split a GSM8K prompt into (user_text, assistant_prefix).

    The assistant prefix is the answer-eliciting tail (``"\\nThe answer is"``);
    the user text is the question + reasoning that precedes it. Falls back to an
    empty prefix if the prompt does not carry the expected tail.
    """
    if prompt.endswith(_ANSWER_TAIL):
        return prompt[: -len(_ANSWER_TAIL)], _ANSWER_TAIL
    return prompt, ""


def _align_answer_pair(
    model,
    clean_prompt: str,
    clean_answer: str,
    corrupt_prompt: str,
    corrupt_answer: str,
) -> Optional[Tuple[str, str, int, int]]:
    """Align a contrastive GSM8K pair onto its first *discriminative* token.

    Tokenizing the bare answer string (``model.to_tokens(" 3450")``) is unsafe
    as a next-token label: tokenizers such as Llama-3's split a leading space
    into its own whitespace token, so the literal first token of every numeric
    answer collapses to that space — the clean/corrupt labels coincide and the
    single-token NLL metric loses all signal.

    Instead each side's *answer continuation* is recovered by tokenizing the
    full prompt+answer and slicing off the prompt's own tokens. The two
    continuations are then walked to the first index where they diverge; any
    shared leading answer tokens (e.g. a standalone space) are appended back
    onto each prompt. This keeps the clean/corrupt *body* difference intact
    (the operand-swap corruption) while guaranteeing the token predicted at
    the EAP backend's ``input_length - 1`` position genuinely differs between
    clean and corrupt — valid for any tokenizer.

    Returns ``(clean_text, corrupt_text, correct_idx, incorrect_idx)`` or
    ``None`` when no usable differing continuation token exists.
    """
    tok = model.tokenizer
    # The prompt strings already carry BOS (chat template) or get it added by
    # the EAP backend; only relative alignment matters here, so no specials.
    clean_p = tok.encode(clean_prompt, add_special_tokens=False)
    corr_p = tok.encode(corrupt_prompt, add_special_tokens=False)
    clean_full = tok.encode(clean_prompt + clean_answer, add_special_tokens=False)
    corr_full = tok.encode(corrupt_prompt + corrupt_answer, add_special_tokens=False)

    # The answer continuation is whatever joint tokenization adds after the
    # prompt's own tokens. Require a clean suffix (no cross-boundary merge);
    # if the boundary token merged, this pair is not safely alignable.
    if clean_full[: len(clean_p)] != clean_p or corr_full[: len(corr_p)] != corr_p:
        return None
    clean_cont = clean_full[len(clean_p) :]
    corr_cont = corr_full[len(corr_p) :]

    # Walk the two continuations to the first index where they differ. Shared
    # leading tokens (a standalone space, a common digit prefix) are appended
    # to each prompt so the discriminative token lands at input_length.
    n = min(len(clean_cont), len(corr_cont))
    d = None
    for i in range(n):
        if clean_cont[i] != corr_cont[i]:
            d = i
            break
    if d is None:
        return None

    clean_text = tok.decode(clean_p + clean_cont[:d])
    corrupt_text = tok.decode(corr_p + corr_cont[:d])
    return clean_text, corrupt_text, clean_cont[d], corr_cont[d]


class GSM8KTaskSpec:
    """GSM8K open-ended-generation task for EAP / EAP-IG circuit discovery."""

    name = "gsm8k"
    task_type = "generation"
    pair_padding_side = "left"
    # Downstream-behavior task: wrap discovery prompts in the model's chat
    # template iff the model is instruction-tuned ("auto"). Frozen into metadata.
    chat_template_mode: str = "auto"

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------
    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate GSM8K-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message("GSM8K", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY)
            )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"GSM8K ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        level = discovery_cfg.get("level")
        if level not in ("node", "neuron"):
            raise ValueError(
                f"GSM8K discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"GSM8K has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------
    def build_dataloader(self, model, discovery_cfg: Dict[str, Any], device: str):
        """Build a DataLoader for GSM8K (EAP / EAP-IG / CD-T / IBCircuit)."""
        if model is None:
            raise ValueError("GSM8K task requires a model for tokenization.")

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)

        if not (is_eap_family(algorithm) or algorithm == "cdt"):
            raise ValueError(
                unsupported_algorithm_message("GSM8K", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY)
            )

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)
        logger.debug(
            f"gsm8k EAP/CDT dataloader  pair_padding_side='{side}'  "
            f"batch_size={batch_size}  data={data_path}  templated={apply}"
        )
        return EAPDiscoveryDataset(data_path).to_dataloader(
            batch_size, pair_padding_side=side, templated=apply
        )
        
    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """Build fixed-batch DataLoader for IBCircuit. Mirrors BoolQ/TruthfulQA."""
        import torch as t

        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        df = pd.read_csv(str(data_path))
        clean_texts = df["clean"].tolist()
        correct_idxs = df["correct_idx"].tolist()

        # Tokenize each prompt individually. The CSV's clean strings are
        # already chat-template-wrapped when apply=True, so route through the
        # BOS-correct helper to avoid a double BOS.
        token_lists = [
            to_tokens(model, text, templated=apply).squeeze(0).cpu() for text in clean_texts
        ]
        # answer_position = last real token index (prompt ends right before answer)
        answer_positions_list = [toks.shape[0] - 1 for toks in token_lists]

        # Right-pad to uniform length
        max_len = max(toks.shape[0] for toks in token_lists)
        pad_id = getattr(model.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(model.tokenizer, "eos_token_id", None) or 0

        padded = []
        for toks in token_lists:
            gap = max_len - toks.shape[0]
            if gap > 0:
                toks = t.cat([toks, t.full((gap,), pad_id, dtype=t.long)])
            padded.append(toks)

        tokens = t.stack(padded).to(device)
        labels = t.tensor(correct_idxs, dtype=t.long, device=device)
        answer_positions = t.tensor(answer_positions_list, dtype=t.long, device=device)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "answer_positions": answer_positions,
        }
        logger.debug(
            f"[DEBUG PADDING] gsm8k IBCircuit  within-batch=right-padded  "
            f"max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Path:
        """Return the path to a cached GSM8K contrastive CSV, building it if absent."""
        data_params = discovery_cfg.get("data_params", {})
        n_samples = data_params.get("num_examples", data_params.get("n_samples", 16))
        seed = data_params.get("seed", 42)
        split = data_params.get("split", "train")
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/gsm8k"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve chat-template handling once; prompts written into the CSV are
        # wrapped with this `apply` value, and it is encoded into the cache file
        # name so a templated and a raw run never share a stale CSV.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        tmpl_tag = "tmpl" if apply else "raw"
        data_path = cache_dir / f"gsm8k_{model_name}_{split}_{n_samples}_seed{seed}_{tmpl_tag}.csv"

        if not data_path.exists():
            self._generate_gsm8k_csv(
                n_samples=n_samples,
                output_path=data_path,
                seed=seed,
                split=split,
                model=model,
                apply=apply,
            )
        return data_path

    # ------------------------------------------------------------------
    # Contrastive CSV generation
    # ------------------------------------------------------------------
    @staticmethod
    def _build_records(n_samples: int, seed: int, split: str) -> List[ContrastiveRecord]:
        """Load GSM8K and build clean ContrastiveRecords (no corruption yet)."""
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
        indices = list(range(len(ds)))
        rng = _random.Random(seed)
        rng.shuffle(indices)

        records: List[ContrastiveRecord] = []
        for idx in indices:
            if len(records) >= n_samples:
                break
            ex = ds[int(idx)]
            question = ex.get("question", "")
            answer_text = ex.get("answer", "")
            final = _extract_final_answer(answer_text)
            reasoning = _reasoning_trace(answer_text)
            if not question or final is None or not reasoning:
                continue
            # The corruption needs a calculator step in the final reasoning line.
            if not _CALC_STEP.search(reasoning):
                continue
            clean_prompt = _format_prompt(question, reasoning)
            records.append(
                ContrastiveRecord(
                    record_id=f"gsm8k-{idx}",
                    clean_prompt=clean_prompt,
                    clean_answer=" " + final,
                    corrupt_prompt=None,
                    corrupt_answer=None,
                    target_field="answer",
                    contrast_source=ContrastSource.GENERATED,
                    meta={"solution_text": clean_prompt, "final_answer": final},
                )
            )
        return records

    @classmethod
    def _generate_gsm8k_csv(
        cls,
        n_samples: int,
        output_path: Path,
        seed: int,
        split: str,
        model,
        apply: bool = False,
    ) -> pd.DataFrame:
        """Build a ``final_answer_swap``-corrupted GSM8K CSV.

        CSV columns: ``clean``, ``corrupted``, ``correct_idx``, ``incorrect_idx``.
        ``correct_idx`` / ``incorrect_idx`` are the first *discriminative*
        continuation-token IDs of the clean / corrupt answers, computed by
        :func:`_align_answer_pair` (it tokenizes prompt+answer jointly so the
        labels are the real next tokens and genuinely differ on any tokenizer
        — Llama-3 splits a leading answer space into its own token, so the
        naive ``to_tokens(" 3450")[0]`` collapses every numeric answer to that
        space). The differentiable NLL metric scores the model on producing
        the correct token at the ``input_length - 1`` position.

        When ``apply`` is True the clean / corrupt prompts are wrapped in the
        model's chat template at finalization time (before being written to the
        CSV). Both sides of a pair get the IDENTICAL answer-eliciting tail as the
        assistant prefix, so the template's prefix/suffix is identical for both
        and clean/corrupt stay token-aligned. ``apply=False`` is byte-identical
        to the legacy raw-text behavior.
        """
        # Over-sample so corruption / tokenization failures still leave n_samples.
        records = cls._build_records(
            n_samples=max(n_samples * 4, n_samples + 16), seed=seed, split=split
        )
        strategy = FinalAnswerSwap()
        rng = _random.Random(seed)

        rows: List[Dict[str, Any]] = []
        for rec in records:
            if len(rows) >= n_samples:
                break
            corrupted = strategy.apply(rec, rng=rng)
            if corrupted.corrupt_prompt is None or corrupted.corrupt_answer is None:
                continue
            # A meaningful contrast: prompt AND answer must actually differ.
            if corrupted.corrupt_prompt == rec.clean_prompt:
                continue
            if corrupted.corrupt_answer.strip() == rec.clean_answer.strip():
                continue

            # Wrap at finalization time. Split each prompt's answer-eliciting
            # tail off as the assistant prefix; clean and corrupt use the same
            # tail, so the chat template adds an identical prefix/suffix to
            # both and the EAP token alignment is preserved.
            clean_user, clean_tail = _split_answer_tail(rec.clean_prompt)
            corrupt_user, corrupt_tail = _split_answer_tail(corrupted.corrupt_prompt)
            clean_wrapped = wrap_prompt(model, clean_user, clean_tail, apply=apply)
            corrupt_wrapped = wrap_prompt(model, corrupt_user, corrupt_tail, apply=apply)

            # Align the pair onto its first genuinely differing token. This
            # tokenizes prompt+answer jointly, so the label is the real
            # next-token continuation (not a tokenizer-dependent standalone
            # token) and is guaranteed discriminative for any tokenizer.
            try:
                aligned = _align_answer_pair(
                    model,
                    clean_wrapped,
                    rec.clean_answer,
                    corrupt_wrapped,
                    corrupted.corrupt_answer,
                )
            except Exception:
                continue
            if aligned is None:
                # No differing continuation token (e.g. first tokens coincide):
                # skip — no signal for the single-token NLL metric.
                continue
            clean_text, corrupt_text, clean_ans_tok, corrupt_ans_tok = aligned

            rows.append(
                {
                    "clean": clean_text,
                    "corrupted": corrupt_text,
                    "correct_idx": clean_ans_tok,
                    "incorrect_idx": corrupt_ans_tok,
                }
            )

        if not rows:
            raise RuntimeError(
                "GSM8K CSV generation produced no valid contrastive pairs. "
                "Every candidate failed corruption / tokenization filtering."
            )

        rng.shuffle(rows)
        df = pd.DataFrame(rows)
        df.to_csv(str(output_path), index=False)
        logger.info(f"Saved {len(df)} GSM8K contrastive examples to {output_path}")
        return df

    # ------------------------------------------------------------------
    # Finetuning data (collateral)
    # ------------------------------------------------------------------
    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """Return (clean_texts, query_strings) GSM8K pairs for finetuning.

        When the finetuning model is instruction-tuned (and the task's resolved
        ``chat_template_mode`` is not ``"off"``) each prompt is wrapped in the
        model's chat template, splitting off the ``"\\nThe answer is"`` tail as
        the assistant prefix exactly as discovery does, so circuit-tuning trains
        on the discovery prompt distribution. For base models / ``"off"`` the
        prompt text is byte-identical to the legacy raw-text behavior.
        """
        cfg = discovery_cfg or {}
        split = cfg.get("data_params", {}).get("split", "train")
        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model); a discovery_cfg override wins.
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)
        records = self._build_records(n_samples=n_examples, seed=seed, split=split)
        clean_texts: List[str] = []
        query_strings: List[str] = []
        for rec in records:
            # Split off the answer-eliciting tail as the assistant prefix and
            # wrap (no-op when apply is False — byte-identical to the raw path).
            user_text, tail = _split_answer_tail(rec.clean_prompt)
            prompt = wrap_prompt_with_tokenizer(tokenizer, user_text, tail, apply=apply)
            clean_texts.append(prompt + rec.clean_answer)
            query_strings.append(prompt)
        return clean_texts, query_strings

    # ------------------------------------------------------------------
    # Metric
    # ------------------------------------------------------------------
    def metric_fn(self, metric_type: str = "nll") -> Callable:
        """Return the differentiable NLL metric for GSM8K circuit discovery.

        ``metric_type`` is accepted for API parity with other TaskSpecs; GSM8K
        always uses the answer-span negative-log-likelihood (cross-entropy),
        which is the open-ended-generation analogue of logit-diff.
        """
        return partial(self._nll_metric, loss=True, mean=True)

    @staticmethod
    def _nll_metric(
        logits, clean_logits, input_length, labels, mean: bool = True, loss: bool = True
    ):
        """Differentiable NLL on the answer span.

        Thin adapter over ``perplexity_loss_span`` (from
        ``backends/eap/metrics.py``). The EAP collate path supplies single-token
        ``labels`` of shape ``[batch, 2]`` (``[correct, incorrect]``) and no
        explicit ``answer_spans``; ``perplexity_span`` then defaults the span to
        the ``input_length - 1`` position and reads ``labels[:, :1]`` as the
        target token. Returns a scalar cross-entropy loss that EAP / EAP-IG
        minimise — lower loss = circuit reproduces the correct final answer.
        """

        if labels.ndim == 1:
            labels = labels.unsqueeze(-1)
        # perplexity_span treats labels[:, :span_len]; with a 1-token span it
        # uses labels[:, :1] = the correct answer token. Slice defensively.
        target = labels[:, :1]
        result = perplexity_loss_span(
            logits,
            clean_logits,
            input_length,
            target,
            answer_spans=None,
            mean=mean,
        )
        if not loss:
            result = -result
        return result

    # ------------------------------------------------------------------
    # Artifact metadata
    # ------------------------------------------------------------------
    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for GSM8K circuit artifacts."""
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        return {
            "task": "gsm8k",
            "data_source": "openai/gsm8k:main",
            "task_type": "generation",
            "metric": "answer_span_nll",
            "corruption": "final_answer_swap",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "num_examples": discovery_cfg.get("data_params", {}).get("num_examples", 16),
            "chat_template_mode": mode,
        }
