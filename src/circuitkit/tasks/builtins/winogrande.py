"""
WinoGrande Task Specification

Thin wrapper for the WinoGrande binary commonsense-reasoning task
(winogrande, Sakaguchi et al. 2019).

Supports EAP and EAP-IG circuit discovery via label-swap corruption:
each example carries its own corrupt counterpart (swap the blank-filling
option), so the corrupt prompt flips the model's expected continuation.

----------------------------------------------------------------------
Scoring design (suffix log-likelihood) -- READ THIS
----------------------------------------------------------------------
WinoGrande examples are a sentence with a blank ``_`` and two options.
The disambiguating cue ALWAYS lies in the text AFTER the blank, e.g.

    "The trophy doesn't fit in the suitcase because _ is too large."
                                                    ^blank   ^cue^^^^^^

The standard (and the only sound) WinoGrande evaluation fills the blank
with each option and compares the model's log-likelihood of the SUFFIX
(the text following the blank) under each completion. The option whose
filling gives the higher suffix log-prob is the prediction.

The previous implementation filled the blank in place and then scored
"which option token is most likely AFTER the final period". That position
carries no signal -- the option word is buried mid-sentence and the cue is
already consumed -- so measured accuracy was ~0.51 (chance) for models
whose true WinoGrande ability is ~0.77.

This task therefore uses a *suffix log-likelihood* metric, NOT the
single-token last-position logit-diff used by BoolQ / SVA / IOI:

  * clean   = prefix + correct_option   + suffix
  * corrupt = prefix + incorrect_option + suffix

  Because both options are required to be single tokens, the clean and
  corrupt prompts are token-length aligned and identical everywhere
  except the one option-token position; the suffix occupies the SAME
  trailing positions in both.

  metric(logits, ...) = sum over the suffix positions of
                        log_softmax(logits)[pos-1, suffix_token]

  i.e. the total log-likelihood the model assigns to the actual suffix
  text. Higher = the filling is more coherent. This is differentiable
  (EAP-compatible), single forward pass, and scored against the SAME
  suffix positions for clean and corrupt.

  base accuracy  = fraction of examples where
                   suffixLL(clean-fill) > suffixLL(corrupt-fill).

NOTE FOR THE PAPER: this metric differs *in kind* from BoolQ's. BoolQ
(and SVA, IOI) score a single answer token at the last query position;
WinoGrande scores a multi-token suffix span. Both remain differentiable
logit/log-prob functions, but the WinoGrande "logit diff" is a
*suffix-span log-likelihood difference*, not a two-token contrast.

IBCircuit is intentionally not the focus here but the legacy path is kept.
"""

import ast
import json
import random
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

import pandas as pd
import torch as t
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Dataset as _Dataset

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

logger = get_logger("task.winogrande")

# Sentinel used to pad the per-example suffix-token label vectors to a
# common length so collate_EAP's torch.tensor(labels) succeeds. The metric
# only ever reads the first n_suffix entries, so the pad value is inert.
_LABEL_PAD = 0

#: Fixed seed for the held-out discovery/eval partition. Identical for every
#: caller so the two halves are always the same disjoint split, independent of
#: any per-cell seed. Do NOT change between runs — it would re-mix the halves.
_PARTITION_SEED = 20240517


def _resolve_option_token(model, word: str) -> int:
    """
    Resolve ' <option>' (single space + word) to a single token ID.
    Raises ValueError if the model tokenises it to more than one token.
    """
    tokens = model.to_tokens(word, prepend_bos=False).squeeze(0)
    if tokens.shape[0] != 1:
        raise ValueError(
            f"WinoGrande option '{word}' tokenises to {tokens.shape[0]} tokens in "
            f"model '{getattr(model.cfg, 'model_name', 'unknown')}'. "
            f"WinoGrande requires each option to be a single token."
        )
    return tokens[0].item()


def _suffix_token_ids(model, prefix_plus_option: str, full_prompt: str) -> List[int]:
    """
    Return the token IDs of the SUFFIX region of ``full_prompt``.

    The suffix is everything after the blank-filling option. We compute it
    exactly the way the EAP harness will tokenise the prompt downstream
    (``model.to_tokens(..., prepend_bos=True)``) and slice off the prefix +
    option tokens, so the precomputed suffix IDs are guaranteed consistent
    with the runtime tokenisation.

    Args:
        model: HookedTransformer (for its tokenizer).
        prefix_plus_option: ``prefix + option`` -- text up to and including
            the filled blank.
        full_prompt: ``prefix + option + suffix`` -- the complete prompt.

    Returns:
        List[int]: token IDs occupying the suffix span (non-empty).
    """
    ctx = model.to_tokens(prefix_plus_option, prepend_bos=True).squeeze(0)
    full = model.to_tokens(full_prompt, prepend_bos=True).squeeze(0)
    n_ctx = ctx.shape[0]
    suffix = full[n_ctx:]
    return [int(x) for x in suffix.tolist()]


class _WinoGrandeEAPDataset(_Dataset):
    """
    CSV-backed dataset for WinoGrande EAP discovery.

    Unlike the generic ``EAPDiscoveryDataset`` (which only carries
    [correct_idx, incorrect_idx]), each WinoGrande sample must carry the
    *suffix token span* so the suffix-log-likelihood metric can score it.

    ``labels`` is a fixed-length integer vector::

        [ n_suffix, suffix_tok_0, suffix_tok_1, ..., suffix_tok_{M-1} ]

    where ``M`` is the dataset-wide max suffix length and entries past
    ``n_suffix`` are padded with ``_LABEL_PAD``. The metric reads
    ``labels[:, 0]`` for the per-example suffix length and ``labels[:, 1:]``
    for the (padded) suffix token IDs.

    Both ``clean`` and ``corrupted`` end with the SAME suffix at the SAME
    trailing token positions (single-token options keep the pair aligned),
    so the metric scores both runs against an identical span.
    """

    def __init__(self, filepath: str):
        self.df = pd.read_csv(filepath)
        suffixes = [self._parse_suffix(s) for s in self.df["suffix_tokens"]]
        self.suffixes = suffixes
        self.max_suffix = max((len(s) for s in suffixes), default=1)

    @staticmethod
    def _parse_suffix(raw: Any) -> List[int]:
        if isinstance(raw, list):
            return [int(x) for x in raw]
        return [int(x) for x in ast.literal_eval(str(raw))]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[str, str, List[int]]:
        row = self.df.iloc[index]
        suffix = self.suffixes[index]
        n_suffix = len(suffix)
        padded = suffix + [_LABEL_PAD] * (self.max_suffix - n_suffix)
        labels = [n_suffix] + padded
        return row["clean"], row["corrupted"], labels

    def to_dataloader(self, batch_size: int, pair_padding_side: str = "left"):
        from ...backends.eap.eap_utils import collate_EAP

        dl = _DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
        dl.pair_padding_side = pair_padding_side
        return dl


class WinoGrandeTaskSpec:
    """WinoGrande task using the simplified TaskSpec wrapper approach."""

    name = "winogrande"
    pair_padding_side = "left"
    # cloze task -- no user/assistant turn to wrap
    chat_template_mode: str = "off"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate WinoGrande-specific discovery configuration."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "WinoGrande", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"WinoGrande discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"WinoGrande ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"WinoGrande has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """Build EAP/EAP-IG/IBCircuit DataLoader for WinoGrande."""
        if model is None:
            raise ValueError("WinoGrande task requires model for tokenizer.")

        # WinoGrande is a cloze task: each prompt is prefix + option + suffix
        # with no user/assistant turn, and the suffix-log-likelihood metric is
        # tied to raw-text suffix-span tokenisation (_suffix_token_ids slices
        # the prefix+option vs full-prompt token streams). Wrapping the prompt
        # in a chat template would shift those spans -- a fragile rewrite.
        # Default "off" keeps everything raw; an explicit override fails loudly
        # rather than being silently ignored.
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

        if not is_eap_family(algorithm) and algorithm != "cdt":
            raise ValueError(
                unsupported_algorithm_message(
                    "WinoGrande", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        logger.debug(
            f"[DEBUG PADDING] winogrande EAP dataloader  pair_padding_side='{side}'  "
            f"batch_size={batch_size}"
        )
        return _WinoGrandeEAPDataset(str(data_path)).to_dataloader(
            batch_size, pair_padding_side=side
        )

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """Build fixed-batch DataLoader for IBCircuit. Mirrors BoolQ."""
        import torch as t

        data_path = self._get_or_generate_csv(discovery_cfg, model)
        df = pd.read_csv(str(data_path))
        clean_texts = df["clean"].tolist()
        correct_idxs = df["correct_idx"].tolist()

        token_lists = [
            model.to_tokens(text, prepend_bos=True).squeeze(0).cpu() for text in clean_texts
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
            f"[DEBUG PADDING] winogrande IBCircuit  within-batch=right-padded  "
            f"max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, "
            f"{answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Path:
        """Return path to cached WinoGrande CSV, generating it if absent."""
        data_params = discovery_cfg.get("data_params", {})
        n_samples = data_params.get("num_examples", data_params.get("n_samples", 128))
        seed = data_params.get("seed", 42)
        # Disjoint held-out half of the train pool ("discovery"/"eval"), or
        # "all" (default) for the whole pool. See _generate_winogrande_csv.
        data_partition = data_params.get("data_partition", "all")
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/winogrande"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        # New schema (suffix-LL) -> bump the cache filename so stale broken
        # CSVs (without the suffix_tokens column) are not silently reused.
        part_tag = "" if data_partition == "all" else f"_{data_partition}"
        data_path = (
            cache_dir / f"winogrande_suffll_{model_name}_{n_samples}_seed{seed}{part_tag}.csv"
        )

        if not data_path.exists():
            self._generate_winogrande_csv(
                n_samples=n_samples,
                output_path=data_path,
                seed=seed,
                model=model,
                data_partition=data_partition,
            )
        return data_path

    @staticmethod
    def _generate_winogrande_csv(
        n_samples: int,
        output_path: Path,
        seed: int,
        model,
        data_partition: str = "all",
    ) -> pd.DataFrame:
        """
        Build a label-swap-corrupted WinoGrande CSV from winogrande's train split.

        Each row is a sentence with a blank ("_") and two options. The clean
        prompt fills the blank with the correct option; the corrupted prompt
        fills it with the incorrect option. Both keep the SAME suffix (text
        after the blank). Examples whose option does not tokenise to a single
        token are skipped, which guarantees the clean/corrupt pair is
        token-length aligned with the suffix at identical trailing positions.

        CSV columns:
          * clean         -- prefix + correct_option   + suffix
          * corrupted     -- prefix + incorrect_option + suffix
          * correct_idx   -- single-token id of ' correct_option'  (legacy/IB)
          * incorrect_idx -- single-token id of ' incorrect_option' (legacy/IB)
          * suffix_tokens -- JSON list of token IDs of the suffix span; the
                             suffix-log-likelihood metric scores these.
        """
        from datasets import load_dataset

        ds = load_dataset("winogrande", "winogrande_xl", split="train")
        examples = list(ds)

        # Stratified random held-out partition. Split by the answer label,
        # shuffle each group with a FIXED seed (identical for every caller),
        # and cut it in half: the "discovery" and "eval" halves are then
        # disjoint, IID, and keep the original answer balance. "all" (default)
        # skips the partition. The per-cell `seed` below still samples within
        # the chosen half, so per-seed variance is preserved.
        if data_partition in ("discovery", "eval"):
            # Collapse rows that share the same sentence + options (even when
            # the answer label differs) to one, so the two halves are disjoint
            # at the example-text level — no sentence can appear in both.
            _seen, _uniq = set(), []
            for e in examples:
                k = (e["sentence"], e["option1"], e["option2"])
                if k not in _seen:
                    _seen.add(k)
                    _uniq.append(e)
            examples = _uniq
            g1 = [e for e in examples if str(e.get("answer")) == "1"]
            g2 = [e for e in examples if str(e.get("answer")) != "1"]
            part_rng = random.Random(_PARTITION_SEED)
            part_rng.shuffle(g1)
            part_rng.shuffle(g2)
            h1, h2 = len(g1) // 2, len(g2) // 2
            if data_partition == "discovery":
                examples = g1[:h1] + g2[:h2]
            else:
                examples = g1[h1:] + g2[h2:]
        elif data_partition != "all":
            raise ValueError(
                f"WinoGrande data_partition must be 'all', 'discovery', or "
                f"'eval'; got {data_partition!r}."
            )

        rng = random.Random(seed)
        rng.shuffle(examples)

        rows: List[Dict[str, Any]] = []
        skipped_multitoken = 0
        skipped_no_suffix = 0
        for ex in examples:
            if len(rows) >= n_samples:
                break
            sentence = ex["sentence"]
            option1 = ex["option1"]
            option2 = ex["option2"]
            answer = ex["answer"]

            if "_" not in sentence:
                continue
            prefix, suffix = sentence.split("_", 1)
            # WinoGrande's cue is in the suffix; an empty suffix carries no
            # signal, so such (rare) examples are unscorable -- skip them.
            if suffix.strip() == "":
                skipped_no_suffix += 1
                continue

            if answer == "1":
                correct_option = option1
                incorrect_option = option2
            else:
                correct_option = option2
                incorrect_option = option1

            try:
                correct_token = _resolve_option_token(model, " " + correct_option)
                incorrect_token = _resolve_option_token(model, " " + incorrect_option)
            except ValueError:
                skipped_multitoken += 1
                continue  # skip multi-token options -> keeps clean/corrupt aligned

            clean_prompt = prefix + correct_option + suffix
            corrupted_prompt = prefix + incorrect_option + suffix

            # Suffix token IDs, computed exactly as the harness will tokenise.
            suffix_ids = _suffix_token_ids(model, prefix + correct_option, clean_prompt)
            corrupt_suffix_ids = _suffix_token_ids(
                model, prefix + incorrect_option, corrupted_prompt
            )
            # Single-token options keep the suffix span identical in length
            # and content across clean/corrupt; if (rarely) tokenisation
            # interacts and they diverge, skip to keep the metric well-posed.
            if suffix_ids != corrupt_suffix_ids or len(suffix_ids) == 0:
                skipped_no_suffix += 1
                continue

            rows.append(
                {
                    "clean": clean_prompt,
                    "corrupted": corrupted_prompt,
                    "correct_idx": correct_token,
                    "incorrect_idx": incorrect_token,
                    "suffix_tokens": json.dumps(suffix_ids),
                }
            )

        rng.shuffle(rows)
        df = pd.DataFrame(rows)
        df.to_csv(str(output_path), index=False)
        logger.info(
            f"Saved {len(df)} WinoGrande examples to {output_path} "
            f"(skipped {skipped_multitoken} multi-token-option, "
            f"{skipped_no_suffix} empty/divergent-suffix)"
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
        Load WinoGrande finetuning pairs directly from winogrande's train split.

        The finetuning target is the full correct-filled sentence; the query
        string is the sentence with the blank shown as ``___``.
        """
        try:
            from datasets import load_dataset

            ds = load_dataset("winogrande", "winogrande_xl", split="train")
            indices = list(range(len(ds)))
            rng = random.Random(seed)
            rng.shuffle(indices)
            indices = indices[:n_examples]

            clean_texts: List[str] = []
            query_strings: List[str] = []
            for idx in indices:
                ex = ds[int(idx)]
                sentence = ex["sentence"]
                option1 = ex["option1"]
                option2 = ex["option2"]
                answer = ex["answer"]

                correct_option = option1 if answer == "1" else option2
                prompt = sentence.replace("_", correct_option)
                clean_texts.append(prompt)
                query_strings.append(sentence.replace("_", "___"))
            return clean_texts, query_strings

        except Exception as e:
            logger.warning(
                f"Falling back to WinoGrande discovery cache for finetuning data "
                f"(HF load failed: {e})"
            )
            cfg = discovery_cfg or {}
            model_name_safe = model_name.replace("/", "_")
            cache_dir = Path(cfg.get("cache_dir", "./cache/winogrande"))
            cache_path = _find_task_cache(cache_dir, self.name, model_name_safe)
            if cache_path is None:
                raise FileNotFoundError(
                    f"No '{self.name}' cache found for model '{model_name}' "
                    f"in {cache_dir}, and HuggingFace fallback failed."
                )
            return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    def metric_fn(self, metric_type: str = "suffix_loglik") -> Callable:
        """Return metric function for circuit discovery.

        ``suffix_loglik`` (default) -- mean suffix log-likelihood of the
        filled sentence (see module docstring). Returned configured with
        ``loss=True`` so EAP minimises ``-suffixLL``. ``kl`` -- batch-mean
        KL between patched and clean output distributions.
        """
        if metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._suffix_loglik, loss=True, mean=True)

    @staticmethod
    def _suffix_loglik(
        logits, clean_logits, input_length, labels, mean: bool = True, loss: bool = False
    ):
        """
        Total log-likelihood the model assigns to the SUFFIX span.

        WinoGrande's disambiguating cue is in the text AFTER the blank, so we
        score how coherent the suffix is given the blank's filling. The suffix
        occupies the last ``n_suffix`` token positions of the (left-padded)
        sequence; the token at position ``p`` is predicted by ``logits[p-1]``.

        ``labels`` layout (see ``_WinoGrandeEAPDataset``):
            labels[:, 0]  -> n_suffix per example
            labels[:, 1:] -> suffix token IDs, right-padded with ``_LABEL_PAD``

        Fully differentiable -> EAP-compatible. Single forward pass: the
        contrast between the correct and incorrect filling is provided by the
        EAP patching mechanism (discovery) or by scoring the clean vs the
        corrupt prompt (``evaluate_baseline`` run_corrupted flag).

        Returns per-example suffix LL (or its negation/mean per the flags).
        """
        batch_size, n_pos, _ = logits.shape
        device = logits.device
        labels = labels.to(device)

        n_suffix = labels[:, 0]  # [batch]
        suffix_tokens = labels[:, 1:]  # [batch, max_suffix]
        max_suffix = suffix_tokens.size(1)

        log_probs = t.log_softmax(logits.float(), dim=-1)  # [batch, n_pos, vocab]

        # Suffix token p (1-indexed offset s within the trailing span) sits at
        # absolute position  (n_pos - n_suffix + s)  and is predicted from the
        # logits one step earlier. Indexing from the sequence END keeps this
        # correct under left padding (n_pos is the shared padded length).
        idx = t.arange(batch_size, device=device)
        per_example = t.zeros(batch_size, device=device)
        for s in range(max_suffix):
            active = s < n_suffix  # [batch] bool
            tok_pos = n_pos - n_suffix + s  # [batch]
            pred_pos = (tok_pos - 1).clamp(min=0, max=n_pos - 1)  # [batch]
            tok_id = suffix_tokens[:, s].clamp(min=0)  # [batch]
            lp = log_probs[idx, pred_pos, tok_id]  # [batch]
            per_example = per_example + t.where(active, lp, t.zeros_like(lp))

        # Mean log-prob per suffix token -> length-normalised, comparable
        # across examples with different suffix lengths.
        results = per_example / n_suffix.clamp(min=1).float()

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
        """Generate metadata for WinoGrande artifacts."""
        return {
            "task": "winogrande",
            "data_source": "winogrande/winogrande_xl",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "metric": "suffix_loglik",
            "n_samples": discovery_cfg.get("data_params", {}).get("n_samples", 128),
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }
