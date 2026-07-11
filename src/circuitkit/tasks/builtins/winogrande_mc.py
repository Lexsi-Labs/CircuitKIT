"""
WinoGrande-MC (multiple-choice) Task Specification

Chat-templatable multiple-choice variant of the WinoGrande commonsense-reasoning
task (winogrande, Sakaguchi et al. 2019).

----------------------------------------------------------------------
Why a separate task from ``winogrande`` -- READ THIS
----------------------------------------------------------------------
The built-in ``winogrande`` task is a *cloze* task: it fills the blank in
place and scores a multi-token suffix log-likelihood. A cloze prompt has no
user/assistant turn structure, so that task is ``chat_template_mode = "off"``
and cannot be evaluated on instruction-tuned models the way they are actually
used.

``winogrande_mc`` is a DIFFERENT task. It reformulates each WinoGrande item as
an explicit multiple-choice comprehension QUESTION::

    Sentence: "The trophy doesn't fit in the suitcase because _ is too large."
    Question: which word fills the blank?
    A) trophy
    B) suitcase
    Answer:

Because this has a real question -> answer structure, it CAN be wrapped in a
model's chat template (``chat_template_mode = "auto"``) and is scored by a
single-token logit-difference metric on the " A" / " B" letter tokens --
exactly like BoolQ and MMLU.

Supports EAP and EAP-IG circuit discovery (and IBCircuit) via an option-swap
corruption: the corrupted prompt presents the two options A/B swapped, so the
correct answer letter flips from " A" to " B" (or vice versa). The swap touches
only the two option-word spans, which keeps clean vs corrupted a minimal,
token-aligned change and the logit-diff metric well defined.
"""

import random
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import pandas as pd
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
from .._chat import resolve_chat_template, to_tokens, wrap_prompt
from ..specs import _find_task_cache, _load_finetuning_data_from_csv

logger = get_logger("task.winogrande_mc")

#: Text placed at the start of the assistant turn -- keeps the answer letter the
#: immediate next token so the single-token logit-diff metric stays valid.
_ASSISTANT_PREFIX = "Answer:"


def _format_body(sentence: str, option_a: str, option_b: str) -> str:
    """The WinoGrande-MC prompt body that belongs in the user turn.

    Presents the sentence (with its blank ``_``) and the two lettered options.
    Does NOT include the trailing answer-eliciting tail -- that is the
    assistant-turn prefix and is added by :func:`wrap_prompt`.
    """
    return (
        f'Sentence: "{sentence}"\n'
        f"Question: which word fills the blank?\n"
        f"A) {option_a}\n"
        f"B) {option_b}\n"
    )


def _format_prompt(sentence: str, option_a: str, option_b: str) -> str:
    """Standard WinoGrande-MC raw prompt (ends at "Answer:")."""
    return _format_body(sentence, option_a, option_b) + _ASSISTANT_PREFIX


class WinoGrandeMCTaskSpec:
    """WinoGrande multiple-choice task using the TaskSpec wrapper approach."""

    name = "winogrande_mc"
    pair_padding_side = "left"
    # Downstream-behavior MC task: this variant has a real question/answer turn
    # structure, so wrap discovery prompts in the model's chat template iff the
    # model is instruction-tuned ("auto"). Frozen into artifact metadata.
    chat_template_mode: str = "auto"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate WinoGrande-MC-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "WinoGrande-MC", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"WinoGrande-MC discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"WinoGrande-MC ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"WinoGrande-MC has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """Build EAP/EAP-IG/IBCircuit DataLoader for WinoGrande-MC."""
        if model is None:
            raise ValueError("WinoGrande-MC task requires model for tokenizer.")

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)

        if not is_eap_family(algorithm) and algorithm != "cdt":
            raise ValueError(
                unsupported_algorithm_message(
                    "WinoGrande-MC", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        data_path, apply = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        logger.debug(
            f"[DEBUG PADDING] winogrande_mc EAP dataloader  pair_padding_side='{side}'  "
            f"batch_size={batch_size}  templated={apply}"
        )
        # `apply` is the resolved chat-template boolean: when True the cached CSV
        # holds chat-templated text (which already carries its own BOS), so the
        # EAP backend must tokenize with prepend_bos=False to avoid a double BOS.
        return EAPDiscoveryDataset(data_path).to_dataloader(
            batch_size, pair_padding_side=side, templated=apply
        )

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """Build fixed-batch DataLoader for IBCircuit. Mirrors BoolQ."""
        import torch as t

        data_path, apply = self._get_or_generate_csv(discovery_cfg, model)
        df = pd.read_csv(str(data_path))
        clean_texts = df["clean"].tolist()
        correct_idxs = df["correct_idx"].tolist()

        # The CSV's clean strings are already chat-template-wrapped when
        # apply=True, so route through the BOS-correct helper.
        token_lists = [
            to_tokens(model, text, templated=apply).squeeze(0).cpu() for text in clean_texts
        ]
        answer_positions_list = [toks.shape[0] - 1 for toks in token_lists]

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
            f"[DEBUG PADDING] winogrande_mc IBCircuit  within-batch=right-padded  "
            f"max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, "
            f"{answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Tuple[Path, bool]:
        """Return (path, apply) for the cached WinoGrande-MC CSV, generating it if absent.

        ``apply`` is the resolved chat-template boolean -- prompts in the CSV are
        chat-template-wrapped iff it is True, so callers must tokenize the
        CSV's strings BOS-correctly for that same boolean.
        """
        data_params = discovery_cfg.get("data_params", {})
        n_samples = data_params.get("num_examples", data_params.get("n_samples", 128))
        seed = data_params.get("seed", 42)
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/winogrande_mc"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve chat-template handling once; prompts written into the CSV are
        # wrapped with this `apply` value, and it is encoded into the cache file
        # name so a templated and a raw run never share a stale CSV.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        tmpl_tag = "tmpl" if apply else "raw"
        data_path = cache_dir / f"winogrande_mc_{model_name}_{n_samples}_seed{seed}_{tmpl_tag}.csv"

        if not data_path.exists():
            self._generate_winogrande_mc_csv(
                n_samples=n_samples,
                output_path=data_path,
                seed=seed,
                model=model,
                apply=apply,
            )
        return data_path, apply

    @staticmethod
    def _letter_token(model, prompt: str, letter: str, *, apply: bool) -> int:
        """Resolve the answer-position token ID for ``letter`` ('A' or 'B').

        Tokenizes ``prompt + " " + letter`` and returns the final token, so the
        ID is exactly the letter token as it appears after the "Answer:" tail.
        """
        toks = to_tokens(model, f"{prompt} {letter}", templated=apply)
        return int(toks[0, -1].item())

    @classmethod
    def _generate_winogrande_mc_csv(
        cls,
        n_samples: int,
        output_path: Path,
        seed: int,
        model,
        apply: bool = False,
    ) -> pd.DataFrame:
        """
        Build an option-swap-corrupted WinoGrande-MC CSV from winogrande's train split.

        Each WinoGrande item (sentence with a ``_`` blank, option1, option2,
        answer) becomes a 2-choice MC question. The clean prompt lists the
        options so that the correct option sits at one letter; the corrupted
        prompt is the SAME prompt with the two options swapped, so the correct
        answer letter flips. The swap touches only the two option-word spans,
        keeping the contrastive pair a minimal change.

        Half the rows have the correct option at A, half at B, so the
        logit-diff metric is not biased toward a single letter.

        CSV columns mirror BoolQ / MMLU two-token format:
          * clean         -- prompt with the correct option at its letter
          * corrupted     -- same prompt, options swapped (correct letter flips)
          * correct_idx   -- token id of the clean correct letter (" A" or " B")
          * incorrect_idx -- token id of the clean incorrect letter
        """
        from datasets import load_dataset

        ds = load_dataset("winogrande", "winogrande_xl", split="train")
        examples = list(ds)
        rng = random.Random(seed)
        rng.shuffle(examples)

        # Wrap each prompt at finalization time: clean and corrupted of every
        # pair get the IDENTICAL assistant_prefix + apply, so the chat template
        # adds the same prefix/suffix to both and token alignment is preserved.
        # When apply=False this is byte-identical to ``_format_prompt``.
        def _wrap(sentence: str, option_a: str, option_b: str) -> str:
            return wrap_prompt(
                model, _format_body(sentence, option_a, option_b), _ASSISTANT_PREFIX, apply=apply
            )

        rows: List[Dict[str, Any]] = []
        skipped_no_blank = 0
        for i, ex in enumerate(examples):
            if len(rows) >= n_samples:
                break
            sentence = ex["sentence"]
            option1 = ex["option1"]
            option2 = ex["option2"]
            answer = ex["answer"]

            if "_" not in sentence:
                skipped_no_blank += 1
                continue

            correct_option = option1 if answer == "1" else option2
            incorrect_option = option2 if answer == "1" else option1

            # Alternate the correct option between A and B so neither letter is
            # systematically the answer. The corrupted prompt swaps the two
            # options, which flips the correct letter to the other one.
            correct_at_a = (i % 2) == 0
            if correct_at_a:
                clean_a, clean_b = correct_option, incorrect_option
                clean_letter, corrupt_letter = "A", "B"
            else:
                clean_a, clean_b = incorrect_option, correct_option
                clean_letter, corrupt_letter = "B", "A"

            clean_prompt = _wrap(sentence, clean_a, clean_b)
            corrupted_prompt = _wrap(sentence, clean_b, clean_a)

            try:
                correct_idx = cls._letter_token(model, clean_prompt, clean_letter, apply=apply)
                incorrect_idx = cls._letter_token(model, clean_prompt, corrupt_letter, apply=apply)
            except Exception:
                continue
            if correct_idx == incorrect_idx:
                # Degenerate tokenization -> logit difference collapses; skip.
                continue

            rows.append(
                {
                    "clean": clean_prompt,
                    "corrupted": corrupted_prompt,
                    "correct_idx": correct_idx,
                    "incorrect_idx": incorrect_idx,
                }
            )

        rng.shuffle(rows)
        df = pd.DataFrame(rows)
        df.to_csv(str(output_path), index=False)
        logger.info(
            f"Saved {len(df)} WinoGrande-MC examples to {output_path} "
            f"(skipped {skipped_no_blank} blank-less)"
        )
        return df

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Load WinoGrande-MC finetuning pairs directly from winogrande's train split.

        The query string is the MC prompt ending at "Answer:"; the finetuning
        target appends the correct answer letter.
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("winogrande", "winogrande_xl", split="train")
            indices = list(range(len(ds)))
            rng = random.Random(seed)
            rng.shuffle(indices)

            clean_texts: List[str] = []
            query_strings: List[str] = []
            for idx in indices:
                if len(clean_texts) >= n_examples:
                    break
                ex = ds[int(idx)]
                sentence = ex["sentence"]
                option1 = ex["option1"]
                option2 = ex["option2"]
                answer = ex["answer"]
                if "_" not in sentence:
                    continue

                correct_option = option1 if answer == "1" else option2
                incorrect_option = option2 if answer == "1" else option1

                correct_at_a = (idx % 2) == 0
                if correct_at_a:
                    opt_a, opt_b, letter = correct_option, incorrect_option, "A"
                else:
                    opt_a, opt_b, letter = incorrect_option, correct_option, "B"

                prompt = _format_prompt(sentence, opt_a, opt_b)
                query_strings.append(prompt)
                clean_texts.append(f"{prompt} {letter}")
            return clean_texts, query_strings

        except Exception as e:
            logger.warning(
                f"Falling back to WinoGrande-MC discovery cache for finetuning data "
                f"(HF load failed: {e})"
            )
            cfg = discovery_cfg or {}
            model_name_safe = model_name.replace("/", "_")
            cache_dir = Path(cfg.get("cache_dir", "./cache/winogrande_mc"))
            cache_path = _find_task_cache(cache_dir, self.name, model_name_safe)
            if cache_path is None:
                raise FileNotFoundError(
                    f"No '{self.name}' cache found for model '{model_name}' "
                    f"in {cache_dir}, and HuggingFace fallback failed."
                )
            return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return metric function for circuit discovery.

        ``logit_diff`` (default) -- single-token logit difference between the
        correct and incorrect answer-letter tokens at the last query position.
        ``kl`` -- batch-mean KL between patched and clean output distributions.
        """
        if metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._logit_diff, loss=True, mean=True)

    @staticmethod
    def _logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        Single-token logit difference: logit(correct letter) - logit(incorrect).

        ``labels[:, 0]`` is the correct answer-letter token (" A"/" B"),
        ``labels[:, 1]`` the incorrect one. Scored at the last query position
        (``input_length - 1`` -- the "Answer:" tail), exactly like IOI/BoolQ.
        Fully differentiable -> EAP/EAP-IG compatible.
        """
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        last_logits = logits[idx, input_length - 1]
        good_bad = t.gather(last_logits, -1, labels[:, 0:2].to(logits.device))
        results = good_bad[:, 0] - good_bad[:, 1]

        if loss:
            results = -results  # negate for minimization
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """KL divergence between patched and clean output distributions."""
        import torch.nn.functional as F

        return F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction="batchmean",
        )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for WinoGrande-MC artifacts."""
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        return {
            "task": "winogrande_mc",
            "data_source": "winogrande/winogrande_xl",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "metric": "logit_diff",
            "corruption_mode": "option_swap",
            "n_samples": discovery_cfg.get("data_params", {}).get("n_samples", 128),
            "chat_template_mode": mode,
        }
