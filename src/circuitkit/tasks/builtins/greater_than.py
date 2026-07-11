"""
Greater-Than Task Specification

Thin wrapper on GenericTaskSpec for numeric comparison (greater-than) task.

The task:
- Generates prompts like: "Which number is greater: 7 or 5? Answer:"
- Builds a *meaning-altering* corrupt prompt: one operand is replaced so the
  larger (answer) number flips to the other slot. This mirrors the
  number-contrastive corruption used by SVA (singular<->plural subject) and
  the entity-swap corruption used by IOI -- the corrupt prompt has a genuinely
  different correct answer, which is what gives EAP/EAP-IG real signal.
- Predicts correct vs incorrect number tokens
- Computes probability difference or logit difference as metric

Corruption design (label-flipping contrast)
--------------------------------------------
Three distinct single-token operands ``lo < mid < hi`` are drawn per row
(all single-token, prompt token-length aligned). The corruption replaces the
*answer* operand so the larger (correct) number changes between clean and
corrupt, keeping the non-answer operand fixed:

    clean      : "Which number is greater: {lo} or {hi}? Answer:"  -> hi
    corrupted  : "Which number is greater: {lo} or {mid}? Answer:" -> mid

``correct_idx``  = the CLEAN prompt's true answer  (``hi``)
``incorrect_idx``= the CORRUPT prompt's true answer (``mid``)

This is the exact contract SVA uses (``correct_idx`` = clean subject's verb,
``incorrect_idx`` = the verb the corrupt subject would take). EAP runs the
metric on the clean forward pass with corrupt activations patched in; because
the corrupt prompt's true answer (``mid``) is a *different* token, the patched
activations push the clean run toward ``mid`` and away from ``hi`` -- which
degrades ``P(hi) - P(mid)`` and produces real gradient signal.

The old corruption merely swapped operand order (``hi or lo`` -> ``lo or hi``):
the SAME two numbers, so ``hi`` was the answer either way and the metric was
flat (circuit scores ~100x weaker than IOI/SVA). Half the rows here put the
answer in the *first* operand slot so corruption is not positionally
degenerate.
"""

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import torch as t

from ...data.eap_dataset import EAPDiscoveryDataset
from ...utils.logging import get_logger
from .._algorithm_families import (
    CDT_FAMILY,
    EAP_FAMILY,
    IB_FAMILY,
    is_eap_family,
    unsupported_algorithm_message,
)
from .._chat import resolve_chat_template
from ..specs import _find_task_cache, _load_finetuning_data_from_csv

logger = get_logger("task.greater_than")


class GreaterThanTaskSpec:
    """
    Greater-Than task using simplified wrapper approach.

    Replaces ~324 lines of hardcoded logic with direct CSV caching.
    """

    name = "greater_than"
    pair_padding_side = "right"
    # diagnostic minimal-pair task -- discovered raw, per the circuit-discovery literature
    chat_template_mode: str = "off"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate Greater-Than-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "Greater-Than", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"Greater-Than discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"Greater-Than has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """Build DataLoader for Greater-Than task."""
        if model is None:
            raise ValueError("Greater-Than task requires model for tokenizer.")

        # Greater-Than pairs are token-aligned minimal pairs with answer token
        # IDs precomputed against raw token positions, and the EAP dataloader
        # tokenizes the cached CSV internally -- there is no single
        # prompt-string finalization point this task spec can wrap without a
        # fragile rewrite. Default "off" keeps everything raw; an explicit
        # override fails loudly rather than being silently ignored.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)
        if apply:
            raise NotImplementedError(
                f"{self.name}: this diagnostic task is discovered on raw "
                f"(non-chat-templated) prompts and does not support a "
                f"chat_template_mode override that enables templating. "
                f"Remove the 'chat_template_mode' key from the discovery "
                f"config or set it to 'off'."
            )

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)
        elif is_eap_family(algorithm) or algorithm == "cdt":
            return self._build_eap_dataloader(discovery_cfg, device, model)
        else:
            raise ValueError(
                unsupported_algorithm_message(
                    "Greater-Than", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Path:
        """Return path to cached Greater-Than CSV, generating it if absent."""
        data_params = discovery_cfg.get("data_params", {})
        n_samples = data_params.get("num_examples", 128)
        seed = data_params.get("seed", 42)
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/greater_than"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        data_path = cache_dir / f"greater_than_{model_name}_{n_samples}_seed{seed}.csv"

        if not data_path.exists():
            self._synthesize_csv(model, str(data_path), n_samples, seed)
        return data_path

    def _synthesize_csv(self, model, out_path: str, num_samples: int, seed: int) -> None:
        """Synthesize Greater-Than CSV with *label-flipping* clean/corrupt pairs.

        The corruption replaces the answer operand so the larger (correct)
        number becomes a different number than in the clean prompt -- a genuine
        meaning-altering contrast, analogous to SVA's singular<->plural subject
        swap. ``correct_idx`` is the clean answer, ``incorrect_idx`` is the
        corrupt prompt's true answer. Patching corrupt activations into the
        clean run therefore pushes the prediction toward a different token,
        which is what produces real EAP/EAP-IG gradient signal.
        """
        import csv
        import random

        random.seed(seed)

        from ...utils.token_utils import TokenIDGenerator

        token_gen = TokenIDGenerator(model)

        # Use the same number-prefix form ('' or ' ') the tokenizer treats as
        # single-token so the answer token id matches the in-prompt token.
        prefix = self._number_prefix(model)

        def answer_token_id(ans: str) -> int:
            return token_gen.get_token_id(prefix + ans, prepend_space=False)

        valid_numbers = self._get_single_token_numbers(model)
        if len(valid_numbers) < 3:
            raise ValueError(
                "Greater-Than label-flipping corruption needs at least 3 "
                f"single-token numbers; only {len(valid_numbers)} available."
            )

        def prompt(x: int, y: int) -> str:
            return f"Which number is greater: {x} or {y}? Answer:"

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["clean", "corrupted", "correct_idx", "incorrect_idx"]
            )
            writer.writeheader()
            for i in range(num_samples):
                # Three distinct single-token numbers, sorted: lo < mid < hi.
                lo, mid, hi = sorted(random.sample(valid_numbers, 3))

                # Half the rows put the answer in the SECOND operand slot,
                # half in the FIRST, so the corruption is not a fixed-position
                # artefact. In both branches the corrupt prompt replaces the
                # ANSWER operand (hi -> mid), leaving the non-answer operand
                # 'lo' fixed, so the true correct answer flips hi -> mid.
                if i % 2 == 0:
                    # Answer in second slot.
                    #   clean  : lo vs hi  -> hi
                    #   corrupt: lo vs mid -> mid
                    clean = prompt(lo, hi)
                    corrupted = prompt(lo, mid)
                else:
                    # Answer in first slot.
                    #   clean  : hi vs lo  -> hi
                    #   corrupt: mid vs lo -> mid
                    clean = prompt(hi, lo)
                    corrupted = prompt(mid, lo)

                # EAP attribution runs the performance metric on the *clean*
                # forward pass while patching in *corrupt* activations
                # (see backends/eap/attribute.py). Mirroring SVA's contract:
                #   correct_idx   = the CLEAN prompt's true answer  (hi)
                #   incorrect_idx = the CORRUPT prompt's true answer (mid)
                # The corrupt prompt's answer is a DIFFERENT number, so the
                # patched activations push the clean run toward 'mid' and away
                # from 'hi' -- degrading P(hi) - P(mid) and producing real
                # gradients. The old code reused the same two numbers in
                # swapped order, so the answer never changed and the metric
                # was flat (~100x weaker scores than IOI/SVA).
                writer.writerow(
                    {
                        "clean": clean,
                        "corrupted": corrupted,
                        "correct_idx": answer_token_id(str(hi)),
                        "incorrect_idx": answer_token_id(str(mid)),
                    }
                )

    def _number_prefix(self, model) -> str:
        """Return ' ' or '' depending on which form yields single-token numbers.

        GPT-2-style BPE tokenizers encode ' 42' as one token; Llama-3 / many
        SentencePiece tokenizers instead encode '42' (no leading space) as one
        token while ' 42' splits. Pick whichever form is single-token so the
        Greater-Than task works on non-GPT-2 models too.
        """
        tokenizer = model.tokenizer
        spaced = sum(
            1
            for n in range(0, 1000)
            if len(tokenizer.encode(f" {n}", add_special_tokens=False)) == 1
        )
        unspaced = sum(
            1
            for n in range(0, 1000)
            if len(tokenizer.encode(f"{n}", add_special_tokens=False)) == 1
        )
        return " " if spaced >= unspaced else ""

    def _get_single_token_numbers(self, model) -> list:
        """Build pool of single-token numbers for this model."""
        tokenizer = model.tokenizer
        prefix = self._number_prefix(model)
        candidates = [
            n
            for n in range(0, 10000)
            if len(tokenizer.encode(f"{prefix}{n}", add_special_tokens=False)) == 1
        ]
        if len(candidates) < 2:
            raise ValueError(
                "Tokenizer yielded fewer than 2 single-token numbers. "
                "Cannot generate Greater-Than data."
            )
        ref_a, ref_b = candidates[0], candidates[1]
        ref_len = len(
            tokenizer.encode(
                f"Which number is greater: {ref_a} or {ref_b}? Answer:", add_special_tokens=False
            )
        )
        verified = [
            n
            for n in candidates
            if (
                len(
                    tokenizer.encode(
                        f"Which number is greater: {n} or {ref_b}? Answer:",
                        add_special_tokens=False,
                    )
                )
                == ref_len
                and len(
                    tokenizer.encode(
                        f"Which number is greater: {ref_a} or {n}? Answer:",
                        add_special_tokens=False,
                    )
                )
                == ref_len
            )
        ]
        if len(verified) < 2:
            raise ValueError(
                "After in-context verification, fewer than 2 valid numbers remain. "
                "Cannot generate Greater-Than data."
            )
        return verified

    def _build_eap_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> "DataLoader":
        """Build EAP/EAP-IG DataLoader."""
        data_path = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        logger.debug(
            f"[DEBUG PADDING] greater_than EAP dataloader  pair_padding_side='{side}'  batch_size={batch_size}"
        )
        return EAPDiscoveryDataset(data_path).to_dataloader(batch_size, pair_padding_side=side)

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """Build fixed-batch DataLoader for IBCircuit."""
        import pandas as pd
        import torch

        logger = get_logger("task.greater_than")

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        df = pd.read_csv(str(data_path))
        clean_texts = df["clean"].tolist()
        correct_idxs = df["correct_idx"].tolist()

        # Tokenize each prompt individually
        token_lists = [
            model.to_tokens(text, prepend_bos=True).squeeze(0).cpu() for text in clean_texts
        ]

        # answer_position = last real token index
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
                toks = torch.cat([toks, torch.full((gap,), pad_id, dtype=torch.long)])
            padded.append(toks)

        tokens = torch.stack(padded).to(device)
        labels = torch.tensor(correct_idxs, dtype=torch.long, device=device)
        answer_positions = torch.tensor(answer_positions_list, dtype=torch.long, device=device)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "answer_positions": answer_positions,
        }

        logger.debug(
            f"[DEBUG PADDING] greater_than IBCircuit  within-batch=right-padded  max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            """Yields one fixed batch repeatedly."""

            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """Load Greater-Than finetuning data from discovery cache."""
        cfg = discovery_cfg or {}
        model_name_safe = model_name.replace("/", "_")
        cache_dir = Path(cfg.get("cache_dir", "./cache/greater_than"))

        cache_path = _find_task_cache(cache_dir, self.name, model_name_safe)
        if cache_path is None:
            raise FileNotFoundError(
                f"No '{self.name}' cache found for model '{model_name}' "
                f"in {cache_dir}.\n"
                f"Run discover_circuit with task='{self.name}' first."
            )
        return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return metric function for circuit discovery.

        The default is ``logit_diff`` (the answer-vs-corrupt-answer logit gap),
        which mirrors IOI. ``logit_diff`` is used because softmax probabilities
        of specific number tokens are tiny (~1e-4) for GPT-2 -- so a
        ``prob_diff`` metric saturates near zero and yields ~100x weaker EAP
        gradients than IOI/SVA even with a correct label-flipping corruption.
        The unbounded logit difference does not saturate and swings strongly
        between clean (correct answer wins) and corrupt (corrupt answer wins),
        giving the attribution real signal. ``prob_diff`` remains available by
        name for callers that explicitly want it.
        """
        if metric_type in ("logit_diff", "ib_circuit"):
            return partial(self._logit_diff, loss=True, mean=True)
        elif metric_type == "prob_diff":
            return partial(self._prob_diff, loss=True, mean=True)
        elif metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._logit_diff, loss=True, mean=True)

    @staticmethod
    def _prob_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """Probability difference: P(greater) - P(smaller)."""
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)
        probs = t.softmax(logits[idx, input_length - 1], dim=-1)
        target_probs = t.gather(probs, -1, labels.to(logits.device))
        results = target_probs[:, 0] - target_probs[:, 1]
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """Logit difference: logit(greater) - logit(smaller)."""
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)
        logits_last = logits[idx, input_length - 1]
        results = logits_last.gather(-1, labels[:, 0:1].to(logits.device)) - logits_last.gather(
            -1, labels[:, 1:2].to(logits.device)
        )
        results = results.squeeze(-1)
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """KL divergence between patched and clean distributions."""
        import torch.nn.functional as F

        return F.kl_div(
            F.log_softmax(logits, dim=-1), F.softmax(clean_logits, dim=-1), reduction="batchmean"
        )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for Greater-Than artifacts."""
        return {
            "task": "greater_than",
            "data_source": "built_in_generation",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "num_samples": discovery_cfg.get("data_params", {}).get("num_examples", 128),
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }
