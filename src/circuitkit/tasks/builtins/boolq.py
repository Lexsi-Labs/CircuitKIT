"""
BoolQ Task Specification

Thin wrapper for the BoolQ yes/no reading-comprehension task
(google/boolq, Clark et al. 2019).

Supports EAP and EAP-IG circuit discovery via label-swap corruption:
each yes-answered example is paired with a no-answered example sharing
the same prompt template, so the corrupt prompt flips the model's
expected output token.

IBCircuit is intentionally not supported here.
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
from .._chat import (
    resolve_chat_template,
    resolve_chat_template_from_tokenizer,
    to_tokens,
    wrap_prompt,
    wrap_prompt_with_tokenizer,
)
from ..specs import _find_task_cache, _load_finetuning_data_from_csv

logger = get_logger("task.boolq")

#: Text placed at the start of the assistant turn — keeps the yes/no answer the
#: immediate next token so the single-token logit-diff metric stays valid.
_ASSISTANT_PREFIX = "Answer:"

#: Fixed seed for the held-out discovery/eval partition. Identical for every
#: caller so the two halves are always the same disjoint split, independent of
#: any per-cell seed. Do NOT change between runs — it would re-mix the halves.
_PARTITION_SEED = 20240517


def _format_prompt(passage: str, question: str) -> str:
    """Standard BoolQ prompt format used by both EAP and finetuning."""
    return f"{passage}\n\nQuestion: {question}\n{_ASSISTANT_PREFIX}"


def _user_text(passage: str, question: str) -> str:
    """The BoolQ prompt body that belongs in the user turn (no answer tail)."""
    return f"{passage}\n\nQuestion: {question}\n"


def _resolve_yesno_token(model, word: str) -> int:
    """
    Resolve ' yes' / ' no' (single space + word) to a single token ID.
    Raises ValueError if the model tokenises it to more than one token.
    """
    tokens = model.to_tokens(word, prepend_bos=False).squeeze(0)
    if tokens.shape[0] != 1:
        raise ValueError(
            f"BoolQ answer '{word}' tokenises to {tokens.shape[0]} tokens in "
            f"model '{getattr(model.cfg, 'model_name', 'unknown')}'. "
            f"BoolQ requires ' yes' and ' no' to each be a single token."
        )
    return tokens[0].item()


class BoolQTaskSpec:
    """BoolQ task using the simplified TaskSpec wrapper approach."""

    name = "boolq"
    pair_padding_side = "left"
    # Downstream-behavior task: wrap discovery prompts in the model's chat
    # template iff the model is instruction-tuned ("auto"). Frozen into metadata.
    chat_template_mode: str = "auto"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate BoolQ-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "BoolQ", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"BoolQ discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"BoolQ ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"BoolQ has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """Build EAP/EAP-IG/IBCircuit DataLoader for BoolQ."""
        if model is None:
            raise ValueError("BoolQ task requires model for tokenizer.")

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)

        if not is_eap_family(algorithm) and algorithm != "cdt":
            raise ValueError(
                unsupported_algorithm_message(
                    "BoolQ", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        data_path, apply = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        logger.debug(
            f"[DEBUG PADDING] boolq EAP dataloader  pair_padding_side='{side}'  "
            f"batch_size={batch_size}  templated={apply}"
        )
        # `apply` is the resolved chat-template boolean: when True the cached CSV
        # holds chat-templated text (which already carries its own BOS), so the
        # EAP backend must tokenize with prepend_bos=False to avoid a double BOS.
        return EAPDiscoveryDataset(data_path).to_dataloader(
            batch_size, pair_padding_side=side, templated=apply
        )

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """Build fixed-batch DataLoader for IBCircuit. Mirrors SVA."""
        import torch as t

        data_path, apply = self._get_or_generate_csv(discovery_cfg, model)
        df = pd.read_csv(str(data_path))
        clean_texts = df["clean"].tolist()
        correct_idxs = df["correct_idx"].tolist()

        # Tokenize each context individually. The CSV's clean strings are
        # already chat-template-wrapped when apply=True, so route through the
        # BOS-correct helper to avoid a double BOS.
        token_lists = [
            to_tokens(model, text, templated=apply).squeeze(0).cpu() for text in clean_texts
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
            f"[DEBUG PADDING] boolq IBCircuit  within-batch=right-padded  "
            f"max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Tuple[Path, bool]:
        """Return (path, apply) for the cached BoolQ CSV, generating it if absent.

        ``apply`` is the resolved chat-template boolean — prompts in the CSV are
        chat-template-wrapped iff it is True, so callers must tokenize the
        CSV's strings BOS-correctly for that same boolean.
        """
        data_params = discovery_cfg.get("data_params", {})
        # ``.get(key, default)`` returns the value when the key *exists*, so an
        # explicit ``None`` (e.g. ``data_params["num_examples"] = None`` set by
        # a downstream caller that hadn't resolved its own default yet) does
        # NOT fall through to 128 here — it propagates to the ``// 2`` on line
        # 328 and crashes with ``unsupported operand type(s) for //: 'NoneType'
        # and 'int'``. Collapse falsy values with ``or`` ourselves.
        n_samples = (data_params.get("num_examples")
                     or data_params.get("n_samples")
                     or 128)
        seed = data_params.get("seed", 42)
        # Which half of the train pool to draw from. "discovery" and "eval" are
        # disjoint halves of a stratified random partition (see _generate_*_csv),
        # so a selector's calibration set can never overlap the faithfulness
        # eval set. "all" (default) uses the whole train pool unchanged.
        data_partition = data_params.get("data_partition", "all")
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/boolq"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve chat-template handling once; prompts written into the CSV are
        # wrapped with this `apply` value, and it is encoded into the cache file
        # name so a templated and a raw run never share a stale CSV.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        tmpl_tag = "tmpl" if apply else "raw"
        # Encode a non-default partition into the cache name so the discovery
        # and eval halves never collide on disk (a default "all" run keeps its
        # old name).
        part_tag = "" if data_partition == "all" else f"_{data_partition}"
        data_path = (
            cache_dir / f"boolq_{model_name}_{n_samples}_seed{seed}{part_tag}_{tmpl_tag}.csv"
        )

        if not data_path.exists():
            self._generate_boolq_csv(
                n_samples=n_samples,
                output_path=data_path,
                seed=seed,
                model=model,
                apply=apply,
                data_partition=data_partition,
            )
        return data_path, apply

    @staticmethod
    def _generate_boolq_csv(
        n_samples: int,
        output_path: Path,
        seed: int,
        model,
        apply: bool = False,
        data_partition: str = "all",
    ) -> pd.DataFrame:
        """
        Build a label-swap-corrupted BoolQ CSV from google/boolq's train split.

        Pairs each yes-answered example with a no-answered example (and vice
        versa). Half the resulting rows have clean=yes/corrupt=no and half the
        reverse. CSV columns mirror SVA: clean, corrupted, correct_idx, incorrect_idx.

        ``data_partition`` selects a disjoint held-out half of the train pool:
        ``"discovery"`` and ``"eval"`` are the two halves of a stratified
        random partition (each label group shuffled with a fixed seed and cut
        in half), so a circuit's calibration set can never overlap the
        faithfulness eval set. ``"all"`` (default) uses the whole train pool.
        """
        from datasets import load_dataset

        yes_id = _resolve_yesno_token(model, " yes")
        no_id = _resolve_yesno_token(model, " no")

        ds = load_dataset("google/boolq", split="train")

        yes_examples: List[Dict[str, Any]] = []
        no_examples: List[Dict[str, Any]] = []
        for ex in ds:
            (yes_examples if ex["answer"] else no_examples).append(ex)

        # Stratified random held-out partition. Shuffle each label group with a
        # FIXED seed (identical for every caller) and cut it in half: the
        # "discovery" and "eval" halves are then disjoint, IID, and keep the
        # original yes/no balance. The per-cell `seed` below still samples
        # within the chosen half, so per-seed variance is preserved.
        if data_partition in ("discovery", "eval"):
            # Drop exact-duplicate rows first so the two halves are disjoint at
            # the example-text level, not merely the row level.
            def _dedup(lst):
                seen, out = set(), []
                for e in lst:
                    k = (e["passage"], e["question"])
                    if k not in seen:
                        seen.add(k)
                        out.append(e)
                return out

            yes_examples = _dedup(yes_examples)
            no_examples = _dedup(no_examples)
            part_rng = random.Random(_PARTITION_SEED)
            part_rng.shuffle(yes_examples)
            part_rng.shuffle(no_examples)
            hy, hn = len(yes_examples) // 2, len(no_examples) // 2
            if data_partition == "discovery":
                yes_examples, no_examples = yes_examples[:hy], no_examples[:hn]
            else:
                yes_examples, no_examples = yes_examples[hy:], no_examples[hn:]
        elif data_partition != "all":
            raise ValueError(
                f"BoolQ data_partition must be 'all', 'discovery', or 'eval'; "
                f"got {data_partition!r}."
            )

        rng = random.Random(seed)
        rng.shuffle(yes_examples)
        rng.shuffle(no_examples)

        n_yes_clean = n_samples // 2
        n_no_clean = n_samples - n_yes_clean
        n_pairs_needed = max(n_yes_clean, n_no_clean)
        if len(yes_examples) < n_pairs_needed or len(no_examples) < n_pairs_needed:
            raise ValueError(
                f"Not enough BoolQ examples to build {n_samples} pairs "
                f"(yes={len(yes_examples)}, no={len(no_examples)})."
            )

        # Wrap each prompt at finalization time: clean and corrupted of every
        # pair get the IDENTICAL assistant_prefix + apply, so the chat template
        # adds the same prefix/suffix to both and token alignment is preserved.
        # When apply=False this is byte-identical to ``_format_prompt``.
        def _wrap(passage: str, question: str) -> str:
            return wrap_prompt(model, _user_text(passage, question), _ASSISTANT_PREFIX, apply=apply)

        rows: List[Dict[str, Any]] = []
        for i in range(n_yes_clean):
            y, n = yes_examples[i], no_examples[i]
            rows.append(
                {
                    "clean": _wrap(y["passage"], y["question"]),
                    "corrupted": _wrap(n["passage"], n["question"]),
                    "correct_idx": yes_id,
                    "incorrect_idx": no_id,
                }
            )
        for i in range(n_no_clean):
            y, n = yes_examples[n_yes_clean + i], no_examples[n_yes_clean + i]
            rows.append(
                {
                    "clean": _wrap(n["passage"], n["question"]),
                    "corrupted": _wrap(y["passage"], y["question"]),
                    "correct_idx": no_id,
                    "incorrect_idx": yes_id,
                }
            )

        rng.shuffle(rows)
        df = pd.DataFrame(rows)
        df.to_csv(str(output_path), index=False)
        logger.info(f"Saved {len(df)} BoolQ examples to {output_path}")
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
        Load BoolQ finetuning pairs directly from google/boolq's train split.

        When the finetuning model is instruction-tuned (and the task's resolved
        ``chat_template_mode`` is not ``"off"``) the prompts are wrapped in the
        model's chat template with the SAME ``Answer:`` assistant prefix that
        discovery uses, so circuit-tuning trains on the discovery prompt
        distribution. For base models / ``"off"`` the prompt text is
        byte-identical to the legacy raw-text behavior.

        Falls back to the discovery CSV cache only if the HF dataset cannot be
        reached; the primary path goes straight to HuggingFace so finetuning
        does not depend on a prior discovery run.
        """
        cfg = discovery_cfg or {}
        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model). A discovery_cfg override takes
        # precedence over the task default, mirroring discovery-time resolution.
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)
        try:
            from datasets import load_dataset

            ds = load_dataset("google/boolq", split="train")
            indices = list(range(len(ds)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            indices = indices[:n_examples]

            clean_texts: List[str] = []
            query_strings: List[str] = []
            for idx in indices:
                ex = ds[int(idx)]
                # Wrap with the same assistant prefix discovery uses (no-op when
                # apply is False — byte-identical to _format_prompt).
                prompt = wrap_prompt_with_tokenizer(
                    tokenizer,
                    _user_text(ex["passage"], ex["question"]),
                    _ASSISTANT_PREFIX,
                    apply=apply,
                )
                answer = " yes" if ex["answer"] else " no"
                clean_texts.append(prompt + answer)
                query_strings.append(prompt)
            return clean_texts, query_strings

        except Exception as e:
            logger.warning(
                f"Falling back to BoolQ discovery cache for finetuning data "
                f"(HF load failed: {e})"
            )
            model_name_safe = model_name.replace("/", "_")
            cache_dir = Path(cfg.get("cache_dir", "./cache/boolq"))
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

    def metric_fn(self, metric_type: str = "prob") -> Callable:
        """Return metric function for circuit discovery.

        Mirrors SVA's signature: 'prob' returns negative probability of the
        correct answer token at the last query position; 'kl' returns
        batch-mean KL between patched and clean logits.
        """
        if metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._prob_metric, loss=True, mean=True)

    @staticmethod
    def _prob_metric(
        logits, clean_logits, input_length, labels, mean: bool = True, loss: bool = False
    ):
        """Probability of the correct yes/no token at the last query position."""
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)
        probs = t.softmax(logits[idx, input_length - 1], dim=-1)
        results = t.gather(probs, -1, labels[:, 0:1].to(logits.device)).squeeze(-1)
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
        """Generate metadata for BoolQ artifacts."""
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        return {
            "task": "boolq",
            "data_source": "google/boolq",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "n_samples": discovery_cfg.get("data_params", {}).get("n_samples", 128),
            "chat_template_mode": mode,
        }
