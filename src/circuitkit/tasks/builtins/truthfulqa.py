"""
TruthfulQA Task Specification

Thin wrapper for the TruthfulQA multiple-choice task
(truthful_qa, Lin et al. 2021).

Supports EAP and EAP-IG circuit discovery via question-replacement corruption
(analogous to MMLU). IBCircuit is intentionally not supported here.
"""

import random as _random
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
from .._chat import (
    resolve_chat_template,
    resolve_chat_template_from_tokenizer,
    to_tokens,
    wrap_prompt,
    wrap_prompt_with_tokenizer,
)
from ..specs import _find_task_cache, _load_finetuning_data_from_csv

logger = get_logger("task.truthfulqa")

#: Answer-eliciting tail that ends every TruthfulQA prompt. Used to split the
#: prompt into a user turn (question + choices) and an assistant-turn prefix so
#: a chat template can be applied without moving the answer off the next-token
#: slot.
_ANSWER_TAIL = "Answer:"


def _format_prompt(question: str, choices: List[str]) -> str:
    """Standard TruthfulQA prompt format (raw, ends at "Answer:")."""
    letters = ["A", "B", "C", "D"][: len(choices)]
    lines = [f"Q: {question}"]
    for letter, choice in zip(letters, choices):
        lines.append(f"{letter}) {choice}")
    lines.append(_ANSWER_TAIL)
    return "\n".join(lines)


def _wrap_prompt(model, raw_prompt: str, *, apply: bool) -> str:
    """Apply (or not) the chat template to a raw TruthfulQA prompt.

    The raw prompt ends with the answer-eliciting tail ``"Answer:"``; that tail
    becomes the assistant-turn prefix so the answer letter stays the immediate
    next token. When ``apply`` is False this returns ``raw_prompt`` unchanged
    (byte-identical legacy behavior).
    """
    user_text = raw_prompt[: -len(_ANSWER_TAIL)]
    return wrap_prompt(model, user_text, _ANSWER_TAIL, apply=apply)


def _build_length_matched_corrupted_prompt(
    clean_prompt: str,
    choices: List[str],
    model,
    base_question: str = "Which is the most possible answer?",
    *,
    apply: bool = False,
) -> str:
    """
    Build a corrupted prompt whose tokenized length exactly matches clean_prompt.
    Mirrors the MMLU approach.

    Args:
        clean_prompt: the already-wrapped clean prompt to length-match against.
        apply: resolved chat-template flag — corrupted candidates are wrapped
            with the identical template so clean/corrupted stay token-aligned.
    """
    target_len = to_tokens(model, clean_prompt, templated=apply).size(1)

    q = base_question
    prompt = _wrap_prompt(model, _format_prompt(q, choices), apply=apply)
    curr_len = to_tokens(model, prompt, templated=apply).size(1)

    if curr_len == target_len:
        return prompt

    NEUTRAL = " the"
    best_prompt, best_diff = prompt, abs(curr_len - target_len)

    if curr_len < target_len:
        for _ in range((target_len - curr_len) + 5):
            q += NEUTRAL
            p = _wrap_prompt(model, _format_prompt(q, choices), apply=apply)
            lyr = to_tokens(model, p, templated=apply).size(1)
            diff = abs(lyr - target_len)
            if diff < best_diff:
                best_diff, best_prompt = diff, p
            if lyr == target_len:
                return p
            if lyr > target_len:
                break
    else:
        words = q.split()
        while len(words) > 1:
            words = words[:-1]
            p = _wrap_prompt(model, _format_prompt(" ".join(words), choices), apply=apply)
            lyr = to_tokens(model, p, templated=apply).size(1)
            diff = abs(lyr - target_len)
            if diff < best_diff:
                best_diff, best_prompt = diff, p
            if lyr == target_len:
                return p
            if lyr < target_len:
                break

    return best_prompt


def _resolve_letter_token(model, prompt: str, letter: str, *, apply: bool = False) -> int:
    """Resolve the token ID for a letter answer token at the end of a prompt."""
    test_text = f"{prompt} {letter}"
    tokens = to_tokens(model, test_text, templated=apply)
    return tokens[0, -1].item()


class TruthfulQATaskSpec:
    """TruthfulQA task using the simplified TaskSpec wrapper approach."""

    name = "truthfulqa"
    pair_padding_side = "left"
    # Downstream-behavior MC task: wrap prompts in the model's chat template
    # iff the model is instruction-tuned ("auto"). Discovery must match how the
    # model is actually evaluated, or the discovered circuit is misattributed.
    chat_template_mode: str = "auto"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate TruthfulQA-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "TruthfulQA", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"TruthfulQA discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"TruthfulQA has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """Build EAP/EAP-IG DataLoader for TruthfulQA."""
        if model is None:
            raise ValueError("TruthfulQA task requires model for tokenizer.")

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)

        if not is_eap_family(algorithm) and algorithm != "cdt":
            raise ValueError(
                unsupported_algorithm_message(
                    "TruthfulQA", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        # Resolve chat-template handling once; the cached CSV was generated with
        # the same `apply`, so it must drive tokenization here too. When True the
        # CSV text already carries the model's BOS, so the EAP backend must
        # tokenize with prepend_bos=False to avoid a double BOS.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)
        logger.debug(
            f"[DEBUG PADDING] truthfulqa EAP dataloader  pair_padding_side='{side}'  "
            f"batch_size={batch_size}  templated={apply}"
        )
        return EAPDiscoveryDataset(data_path).to_dataloader(
            batch_size, pair_padding_side=side, templated=apply
        )

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """Build fixed-batch DataLoader for IBCircuit. Mirrors BoolQ."""
        import torch as t

        # Resolve chat-template handling once; the cached CSV was generated with
        # the same `apply`, so the same value must drive tokenization here.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        df = pd.read_csv(str(data_path))
        clean_texts = df["clean"].tolist()
        correct_idxs = df["correct_idx"].tolist()

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
            f"[DEBUG PADDING] truthfulqa IBCircuit  within-batch=right-padded  "
            f"max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Path:
        """Return path to cached TruthfulQA CSV, generating it if absent."""
        data_params = discovery_cfg.get("data_params", {})
        n_samples = data_params.get("num_examples", data_params.get("n_samples", 128))
        seed = data_params.get("seed", 42)
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/truthfulqa"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve chat-template handling once; prompts written into the CSV are
        # wrapped with this `apply` value, and it is encoded into the cache file
        # name so a templated and a raw run never share a stale CSV.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        tmpl_tag = "tmpl" if apply else "raw"
        data_path = cache_dir / f"truthfulqa_{model_name}_{n_samples}_seed{seed}_{tmpl_tag}.csv"

        if not data_path.exists():
            self._generate_truthfulqa_csv(
                n_samples=n_samples,
                output_path=data_path,
                seed=seed,
                model=model,
                apply=apply,
            )
        return data_path

    @staticmethod
    def _generate_truthfulqa_csv(
        n_samples: int,
        output_path: Path,
        seed: int,
        model,
        *,
        apply: bool = False,
    ) -> pd.DataFrame:
        """
        Build a question-replacement-corrupted TruthfulQA CSV from the
        truthful_qa validation split.

        Each row has a question with mc1_targets (choices + labels).
        The clean prompt uses the original question.
        The corrupted prompt replaces the question with a generic one.
        CSV columns: clean, corrupted, correct_idx, incorrect_idx.

        Args:
            apply: resolved chat-template flag — clean and corrupted prompts are
                both wrapped through the same template so the contrastive pair
                stays token-aligned. When False this is byte-identical to the
                legacy raw-text behavior.
        """
        from datasets import load_dataset

        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

        examples: List[Dict[str, Any]] = []
        for ex in ds:
            examples.append(ex)

        rng = _random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:n_samples]

        rows: List[Dict[str, Any]] = []
        for ex in examples:
            question = ex["question"]
            mc1 = ex["mc1_targets"]
            choices = list(mc1["choices"])
            labels = list(mc1["labels"])

            if not question or not choices:
                continue

            # Find the correct answer (exactly one label should be 1)
            correct_indices = [i for i, label in enumerate(labels) if label == 1]
            if len(correct_indices) != 1:
                continue
            answer_idx = correct_indices[0]

            # Wrap the clean prompt (the "Answer:" tail stays the assistant-turn
            # prefix). The corrupted prompt is wrapped with the identical
            # template/apply value so the two stay token-aligned.
            clean_prompt = _wrap_prompt(model, _format_prompt(question, choices), apply=apply)

            corrupted_prompt = _build_length_matched_corrupted_prompt(
                clean_prompt, choices, model, apply=apply
            )

            # Resolve A/B/C/D token IDs
            letters = ["A", "B", "C", "D"][: len(choices)]
            try:
                option_tokens = []
                for letter in letters:
                    test_text = f"{clean_prompt} {letter}"
                    tokens = to_tokens(model, test_text, templated=apply)
                    option_tokens.append(tokens[0, -1].item())

                correct_token = option_tokens[answer_idx]
                incorrect_tokens = [
                    option_tokens[i] for i in range(len(option_tokens)) if i != answer_idx
                ]

                while len(incorrect_tokens) < 3:
                    incorrect_tokens.append(correct_token)

                all_option_tokens = [correct_token] + incorrect_tokens[:3]

            except Exception:
                continue

            rows.append(
                {
                    "clean": clean_prompt,
                    "corrupted": corrupted_prompt,
                    "correct_idx": all_option_tokens[0],
                    "incorrect_idx": all_option_tokens[1:4],
                }
            )

        rng.shuffle(rows)
        df = pd.DataFrame(rows)
        df.to_csv(str(output_path), index=False)
        logger.info(f"Saved {len(df)} TruthfulQA examples to {output_path}")
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
        Load TruthfulQA finetuning pairs directly from truthful_qa's validation split.

        When the finetuning model is instruction-tuned (and the task's resolved
        ``chat_template_mode`` is not ``"off"``) prompts are wrapped in the
        model's chat template with the same ``Answer:`` assistant prefix
        discovery uses, so circuit-tuning trains on the discovery prompt
        distribution. For base models / ``"off"`` the prompt text is
        byte-identical to the legacy raw-text behavior.
        """
        cfg = discovery_cfg or {}
        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model); a discovery_cfg override wins.
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)
        try:
            from datasets import load_dataset

            ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
            indices = list(range(len(ds)))
            rng = _random.Random(seed)
            rng.shuffle(indices)
            indices = indices[:n_examples]

            clean_texts: List[str] = []
            query_strings: List[str] = []
            for idx in indices:
                ex = ds[int(idx)]
                question = ex["question"]
                mc1 = ex["mc1_targets"]
                choices = list(mc1["choices"])
                labels = list(mc1["labels"])

                correct_indices = [i for i, label in enumerate(labels) if label == 1]
                if len(correct_indices) != 1:
                    continue
                answer_idx = correct_indices[0]

                # Wrap the prompt with the same "Answer:" assistant prefix
                # discovery uses (no-op when apply is False — byte-identical to
                # _format_prompt).
                raw_prompt = _format_prompt(question, choices)
                user_text = raw_prompt[: -len(_ANSWER_TAIL)]
                prompt = wrap_prompt_with_tokenizer(tokenizer, user_text, _ANSWER_TAIL, apply=apply)
                letter = ["A", "B", "C", "D"][answer_idx]
                clean_texts.append(f"{prompt} {letter}")
                query_strings.append(prompt)
            return clean_texts, query_strings

        except Exception as e:
            logger.warning(
                f"Falling back to TruthfulQA discovery cache for finetuning data "
                f"(HF load failed: {e})"
            )
            model_name_safe = model_name.replace("/", "_")
            cache_dir = Path(cfg.get("cache_dir", "./cache/truthfulqa"))
            # Select the cache variant matching the resolved chat-template mode;
            # its 'clean' column is already templated/raw accordingly, so the
            # CSV loader uses it verbatim (no double-wrapping).
            cache_path = _find_task_cache(cache_dir, self.name, model_name_safe, templated=apply)
            if cache_path is None:
                raise FileNotFoundError(
                    f"No '{self.name}' cache found for model '{model_name}' "
                    f"in {cache_dir}, and HuggingFace fallback failed."
                )
            return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return metric function for circuit discovery."""
        if metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._eap_logit_diff, loss=True, mean=True)

    @staticmethod
    def _eap_logit_diff(
        logits, clean_logits, input_length, labels, mean: bool = True, loss: bool = False
    ):
        """
        Logit difference between the correct letter token and the mean of
        incorrect letter tokens at the last query position.
        Mirrors MMLU's multi-choice logit difference.
        """
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        last_logits = logits[idx, input_length - 1]

        correct_logits = t.gather(last_logits, -1, labels[:, 0:1].to(logits.device))
        incorrect_logits = t.gather(last_logits, -1, labels[:, 1:4].to(logits.device))

        avg_incorrect_logits = incorrect_logits.mean(dim=-1, keepdim=True)

        results = correct_logits.squeeze(-1) - avg_incorrect_logits.squeeze(-1)

        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _kl_divergence(
        logits, clean_logits, input_length, labels, mean: bool = True, loss: bool = False
    ):
        """KL divergence between patched and clean output distributions."""
        import torch.nn.functional as F

        return F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction="batchmean",
        )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for TruthfulQA artifacts."""
        return {
            "task": "truthfulqa",
            "data_source": "truthful_qa/multiple_choice",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "n_samples": discovery_cfg.get("data_params", {}).get("n_samples", 128),
            # Resolved chat-template mode — later stages read this back unchanged.
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }
