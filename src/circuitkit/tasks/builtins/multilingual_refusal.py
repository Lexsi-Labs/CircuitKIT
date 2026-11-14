"""
Multilingual Refusal Task Specification

Minimal pairs for finding the refusal circuit of a chat model, in any language.

Each pair is a harmful and a harmless request that differ in one or two words ("trim a
drug supply" / "trim a long hedge"). The harmful request is the clean prompt, scored on
the first token of a refusal; the harmless one is the corrupted prompt, whose answer is
the first token of a compliance. The metric is the logit difference between the two.

English pairs ship with the package (``binary_align/jailbreak_binary.csv``). Other
languages come from a CSV you provide, with ``harmful`` and ``harmless`` columns: a pair
has to be translated together so the swap stays minimal, and reviewed by someone who
reads the language, so CircuitKIT ships no translations.

Prompts go through the chat template by default, because refusal is a behaviour of a chat
model run with its template. A pair that tokenises to two lengths is dropped, and the
kept count is logged. Set ``chat_template_mode="off"`` for pairs that are already rendered.

``data_params``:
    language: A language code, default ``"en"``.
    pairs_path: CSV of pairs. Required for every language but English.
    refuse / comply: The openers to score. Prefer what the model actually says in that
        language; the defaults in :data:`DEFAULT_OPENERS` are a starting point.
    chat_template_kwargs: Extra arguments for the tokenizer's ``apply_chat_template``. A
        thinking model such as Qwen3 needs ``{"enable_thinking": False}``: with its default
        template the reply opens with ``<think>``, so the position scored here predicts
        that tag, not a refusal or a compliance.
    num_examples, seed, data_partition: As for the other tasks.
"""

import csv
import hashlib
import json
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
from .._algorithm_families import EAP_FAMILY, unsupported_algorithm_message
from .._chat import (
    resolve_chat_template,
    resolve_chat_template_from_tokenizer,
    to_tokens,
    wrap_prompt,
)
from ..specs import _find_task_cache, _load_finetuning_data_from_csv

logger = get_logger("task.multilingual_refusal")

#: (refusal opener, compliance opener) per language.
DEFAULT_OPENERS: Dict[str, Tuple[str, str]] = {
    "en": ("I", "Sure"),
    "hi": ("मैं", "ज़रूर"),
    "hi-Latn": ("Main", "Zaroor"),
    "bn": ("আমি", "অবশ্যই"),
    "ta": ("மன்னிக்கவும்", "நிச்சயமாக"),
    "ar": ("عذراً", "بالتأكيد"),
    "sw": ("Samahani", "Hakika"),
}

_ENGLISH_PAIRS = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "task_data"
    / "tasks"
    / "binary_align"
    / "jailbreak_binary.csv"
)

#: Fixed seed for the held-out partition, so the discovery and eval halves stay disjoint
#: whatever per-run seed is used. Do NOT change between runs: it would re-mix the halves.
_PARTITION_SEED = 20240517


def _read_pairs(path: Path, harmful: str, harmless: str) -> List[Tuple[str, str]]:
    """Read pairs from ``path``, keeping only rows with two different non-empty sides."""
    with open(path, encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {harmful, harmless} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"multilingual_refusal pairs file {path} has no column(s) {sorted(missing)}; "
                f"it has {reader.fieldnames}."
            )
        rows = [((row[harmful] or "").strip(), (row[harmless] or "").strip()) for row in reader]
    return [(a, b) for a, b in rows if a and b and a != b]


def load_pairs(data_params: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (harmful, harmless) pairs: ``pairs_path``, else the bundled English set."""
    language = data_params.get("language", "en")
    path = data_params.get("pairs_path")
    if path:
        pairs = _read_pairs(Path(path), "harmful", "harmless")
    elif language == "en":
        pairs = _read_pairs(_ENGLISH_PAIRS, "harmful_req", "benign_req")
    else:
        raise ValueError(
            f"multilingual_refusal needs data_params['pairs_path'] for language {language!r}: "
            f"a CSV with 'harmful' and 'harmless' columns. Only English pairs ship with "
            f"CircuitKIT. Translate each pair together so the swap stays minimal."
        )
    if not pairs:
        raise ValueError(f"multilingual_refusal found no usable pairs for language {language!r}.")
    return pairs


def _openers(data_params: Dict[str, Any]) -> Tuple[str, str]:
    """Return the (refuse, comply) openers: ``data_params``, else the language default."""
    language = data_params.get("language", "en")
    default = DEFAULT_OPENERS.get(language, (None, None))
    refuse = data_params.get("refuse") or default[0]
    comply = data_params.get("comply") or default[1]
    if not refuse or not comply:
        raise ValueError(
            f"multilingual_refusal has no default openers for language {language!r}. Set "
            f"data_params['refuse'] and ['comply'] to how the model starts a refusal and a "
            f"compliance. Defaults exist for {sorted(DEFAULT_OPENERS)}."
        )
    return refuse, comply


def _has_template(tokenizer) -> bool:
    """True iff ``tokenizer`` has a chat template to render with.

    Not ``model_is_chat``, which accepts only a string: the tokenizers of Cohere's Tiny Aya
    hold a dict of named templates, and they are the chat models this task is for.
    """
    template = getattr(tokenizer, "chat_template", None)
    return bool(template.strip() if isinstance(template, str) else template)


def _warn_if_thinking(tokenizer, chat_template_kwargs: Dict[str, Any]) -> None:
    """Warn when the template has a reasoning switch that ``chat_template_kwargs`` leaves alone."""
    template = getattr(tokenizer, "chat_template", None)
    text = template if isinstance(template, str) else json.dumps(template or "")
    if "enable_thinking" in text and "enable_thinking" not in chat_template_kwargs:
        logger.warning(
            "multilingual_refusal: this chat template has an 'enable_thinking' switch that is "
            "on, so the reply opens with <think> and the scored position predicts that tag. "
            "Set data_params['chat_template_kwargs'] = {'enable_thinking': False}."
        )


def _first_token(tokenizer, text: str) -> int:
    """The token a reply starting with ``text`` opens on.

    Non-Latin openers often span several tokens. The metric reads one position, and the
    first token is what the model emits there, so that is the one scored.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"multilingual_refusal opener {text!r} encodes to no tokens.")
    return ids[0]


def _partition(pairs: List[Tuple[str, str]], data_partition: str) -> List[Tuple[str, str]]:
    """The disjoint "discovery" or "eval" half of ``pairs``, or all of them for "all"."""
    if data_partition == "all":
        return list(pairs)
    if data_partition not in ("discovery", "eval"):
        raise ValueError(
            f"multilingual_refusal data_partition must be 'all', 'discovery' or 'eval'; "
            f"got {data_partition!r}."
        )
    pool = sorted(set(pairs))
    random.Random(_PARTITION_SEED).shuffle(pool)
    half = len(pool) // 2
    return pool[:half] if data_partition == "discovery" else pool[half:]


class MultilingualRefusalTaskSpec:
    """Refusal minimal pairs for chat models, one language per run."""

    name = "multilingual_refusal"
    pair_padding_side = "left"
    # Refusal is a behavior of a chat model run with its template, so unlike the other
    # minimal-pair tasks this one is wrapped, and a base model is an error, not a raw run.
    chat_template_mode: str = "on"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate the algorithm, level, batch size, pairs and openers."""
        algorithm = discovery_cfg.get("algorithm", "").lower()
        if algorithm not in EAP_FAMILY:
            raise ValueError(
                unsupported_algorithm_message("multilingual_refusal", algorithm, EAP_FAMILY)
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"multilingual_refusal discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"multilingual_refusal has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

        data_params = discovery_cfg.get("data_params") or {}
        _openers(data_params)
        load_pairs(data_params)

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> "DataLoader":
        """Build the EAP-family DataLoader."""
        if model is None:
            raise ValueError("multilingual_refusal task requires model for tokenizer.")

        data_path, apply = self._get_or_generate_csv(discovery_cfg, model)
        batch_size = discovery_cfg.get("batch_size", 4)
        side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        # `apply` is the resolved chat-template boolean: when True the cached CSV holds
        # templated text that carries its own BOS, so the EAP backend must tokenize with
        # prepend_bos=False to avoid a double BOS.
        return EAPDiscoveryDataset(data_path).to_dataloader(
            batch_size, pair_padding_side=side, templated=apply
        )

    def _get_or_generate_csv(self, discovery_cfg: Dict[str, Any], model) -> Tuple[Path, bool]:
        """Return (path, apply) for the cached CSV, generating it if absent."""
        data_params = discovery_cfg.get("data_params") or {}
        n_samples = data_params.get("num_examples") or data_params.get("n_samples") or 128
        seed = data_params.get("seed", 42)
        data_partition = data_params.get("data_partition", "all")
        language = data_params.get("language", "en")
        cache_dir = Path(discovery_cfg.get("cache_dir", "./cache/multilingual_refusal"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)
        if apply and not _has_template(model.tokenizer):
            raise ValueError(
                "multilingual_refusal needs a chat model: this tokenizer has no chat template. "
                "A base model has no refusal behavior to trace. For pairs that are already "
                "rendered, set chat_template_mode='off'."
            )

        refuse, comply = _openers(data_params)
        pairs = load_pairs(data_params)
        template_kwargs = dict(data_params.get("chat_template_kwargs") or {})
        if apply:
            _warn_if_thinking(model.tokenizer, template_kwargs)
        # The pairs, openers and template switches are in the name, so editing any of them
        # can't reuse a stale CSV.
        digest = hashlib.sha256(
            json.dumps(
                [pairs, refuse, comply] + ([template_kwargs] if template_kwargs else []),
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        model_name = getattr(model.cfg, "model_name", "unknown").replace("/", "_")
        part_tag = "" if data_partition == "all" else f"_{data_partition}"
        tmpl_tag = "tmpl" if apply else "raw"
        data_path = cache_dir / (
            f"{self.name}_{language}_{model_name}_{n_samples}_seed{seed}{part_tag}"
            f"_{digest}_{tmpl_tag}.csv"
        )
        if not data_path.exists():
            self._generate_csv(
                _partition(pairs, data_partition),
                n_samples=n_samples,
                output_path=data_path,
                seed=seed,
                model=model,
                apply=apply,
                openers=(refuse, comply),
                language=language,
                chat_template_kwargs=template_kwargs,
            )
        return data_path, apply

    @staticmethod
    def _generate_csv(
        pairs: List[Tuple[str, str]],
        *,
        n_samples: int,
        output_path: Path,
        seed: int,
        model,
        apply: bool,
        openers: Tuple[str, str],
        language: str,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Write up to ``n_samples`` token-aligned pairs. Columns mirror the other tasks."""
        refuse_id = _first_token(model.tokenizer, openers[0])
        comply_id = _first_token(model.tokenizer, openers[1])
        if refuse_id == comply_id:
            raise ValueError(
                f"multilingual_refusal openers {openers!r} start on the same token, so the "
                f"metric can't tell them apart. Set data_params['refuse'] and ['comply']."
            )

        pool = list(pairs)
        random.Random(seed).shuffle(pool)
        rows: List[Dict[str, Any]] = []
        dropped = 0
        for harmful, harmless in pool:
            # Both sides get the same wrapping, so the template adds an identical prefix and
            # suffix and the pair stays aligned wherever the two requests tokenize alike.
            clean = wrap_prompt(
                model, harmful, apply=apply, chat_template_kwargs=chat_template_kwargs
            )
            corrupted = wrap_prompt(
                model, harmless, apply=apply, chat_template_kwargs=chat_template_kwargs
            )
            clean_len = to_tokens(model, clean, templated=apply).shape[-1]
            if clean_len != to_tokens(model, corrupted, templated=apply).shape[-1]:
                dropped += 1
                continue
            rows.append(
                {
                    "clean": clean,
                    "corrupted": corrupted,
                    "correct_idx": refuse_id,
                    "incorrect_idx": comply_id,
                }
            )
            if len(rows) == n_samples:
                break

        if not rows:
            raise ValueError(
                f"multilingual_refusal: none of the {len(pool)} {language!r} pairs tokenize to "
                f"equal lengths under this model, so there is no minimal pair to use."
            )
        if len(rows) < n_samples:
            logger.warning(
                f"multilingual_refusal {language}: {len(rows)} of {n_samples} requested pairs "
                f"({dropped} dropped for unequal token lengths)."
            )
        # Non-Latin openers often score a fragment (one letter, or part of one), so say which.
        scored = [model.tokenizer.decode([i]) for i in (refuse_id, comply_id)]
        logger.info(
            f"Kept {len(rows)} {language} refusal pairs; dropped {dropped}. Scoring refuse "
            f"{scored[0]!r} (from {openers[0]!r}) against comply {scored[1]!r} "
            f"(from {openers[1]!r})."
        )
        df = pd.DataFrame(rows)
        df.to_csv(str(output_path), index=False)
        return df

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """Load (clean_texts, query_strings) from this language's discovery cache."""
        cfg = discovery_cfg or {}
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)
        language = (cfg.get("data_params") or {}).get("language", "en")
        cache_dir = Path(cfg.get("cache_dir", "./cache/multilingual_refusal"))
        cache_path = _find_task_cache(
            cache_dir, f"{self.name}_{language}", model_name.replace("/", "_"), templated=apply
        )
        if cache_path is None:
            raise FileNotFoundError(
                f"No '{self.name}' {language} cache for model '{model_name}' in {cache_dir}. "
                f"Run discover_circuit on this task and model first."
            )
        return _load_finetuning_data_from_csv(cache_path, tokenizer, n_examples, seed)

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the discovery metric: refuse-minus-comply logit difference, or KL."""
        if metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._logit_diff, loss=True, mean=True)

    @staticmethod
    def _logit_diff(
        logits, clean_logits, input_length, labels, mean: bool = True, loss: bool = False
    ):
        """Logit difference: logit(refuse) - logit(comply) at the last query position."""
        idx = t.arange(logits.size(0), device=logits.device)
        good_bad = t.gather(logits[idx, input_length - 1], -1, labels.to(logits.device))
        results = good_bad[:, 0] - good_bad[:, 1]
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
        """Generate metadata for artifacts."""
        data_params = discovery_cfg.get("data_params") or {}
        return {
            "task": self.name,
            "language": data_params.get("language", "en"),
            "data_source": data_params.get("pairs_path") or "binary_align/jailbreak_binary.csv",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "n_samples": data_params.get("num_examples") or data_params.get("n_samples") or 128,
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }
