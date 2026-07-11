"""
WMDP (Weapons of Mass Destruction Proxy) Task Specification

Implements the TaskSpec interface for the WMDP task with support for EAP, EAP-IG, and IBCircuit.
Uses corruption strategy of replacing question stem with "Which is the most possible answer?"
Similar to MMLU but with WMDP-specific dataset configs (wmdp-bio, wmdp-chem, wmdp-cyber).
"""

import os
import random as _random
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch as t
from datasets import load_dataset
from torch.utils.data import DataLoader

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

logger = get_logger("task.wmdp")

#: Answer-eliciting tail that ends every WMDP prompt. Used to split the prompt
#: into a user turn (instruction + question + choices) and an assistant-turn
#: prefix so a chat template can be applied without moving the answer off the
#: next-token slot.
_ANSWER_TAIL = "Answer:"


def _load_dataset_with_cache_fix(dataset_name: str, config: str, split: str = "test"):
    """
    Load dataset with automatic cache clearing if outdated format is detected.
    Handles the 'Feature type List not found' error from stale HuggingFace caches.
    """

    try:
        return load_dataset(dataset_name, config, split=split)
    except ValueError as e:
        if "Feature type 'List' not found" in str(e):
            logger.warning(
                f"Outdated cache for '{dataset_name}' config '{config}'. "
                f"Clearing and retrying..."
            )
            cache_base = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "datasets"
            )
            cache_base_path = Path(os.path.expanduser(cache_base))
            if cache_base_path.exists():
                dataset_cache_dir = dataset_name.replace("/", "___")
                for cache_path in [
                    cache_base_path / dataset_cache_dir / config,
                    cache_base_path / dataset_cache_dir,
                ]:
                    if cache_path.exists():
                        try:
                            shutil.rmtree(cache_path)
                            logger.info(f"Cleared cache at {cache_path}")
                        except Exception as ce:
                            logger.warning(f"Could not clear cache at {cache_path}: {ce}")
            return load_dataset(dataset_name, config, split=split, download_mode="force_redownload")
        raise


class WMDPTaskSpec:
    """TaskSpec implementation for WMDP (Weapons of Mass Destruction Proxy)."""

    name = "wmdp"
    pair_padding_side = "left"
    # Downstream-behavior MC task: wrap prompts in the model's chat template
    # iff the model is instruction-tuned ("auto"). Discovery must match how the
    # model is actually evaluated, or the discovered circuit is misattributed.
    chat_template_mode: str = "auto"

    # ── Available WMDP configs ──────────────────────────────────────────────────
    VALID_CONFIGS = ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]

    # ── Validation ──────────────────────────────────────────────────────────────

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """
        Validate WMDP-specific discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "WMDP task", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        if algorithm in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            if "model_name" not in discovery_cfg:
                raise ValueError(
                    "WMDP task discovery config is missing the required key "
                    "'model_name'. Add 'model_name' to the discovery config so "
                    "the task can load the tokenizer."
                )

            # Validate configs (analogous to MMLU subjects)
            configs = self._resolve_configs(discovery_cfg)
            for cfg in configs:
                if cfg not in self.VALID_CONFIGS:
                    raise ValueError(
                        f"Invalid WMDP config: {cfg!r}. "
                        f"Set discovery config key 'configs' to valid WMDP "
                        f"config name(s); one of: {self.VALID_CONFIGS}"
                    )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"WMDP ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"WMDP task discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        batch_size = discovery_cfg.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError(
                f"WMDP task has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 16)."
            )

        samples_per_config = discovery_cfg.get("samples_per_config")
        if samples_per_config is not None and (
            not isinstance(samples_per_config, int) or samples_per_config <= 0
        ):
            raise ValueError(
                f"WMDP task has invalid 'samples_per_config': {samples_per_config!r}. "
                f"Set discovery config key 'samples_per_config' to a positive integer."
            )

    # ── Config resolution ───────────────────────────────────────────────────────

    def _resolve_configs(self, discovery_cfg: Dict[str, Any]) -> List[str]:
        """
        Resolve which WMDP configs to use from the discovery config.

        Supports keys: 'configs', 'config', 'subjects', 'subject' (for compatibility).
        Defaults to all three configs if none specified.
        """
        configs = (
            discovery_cfg.get("configs")
            or discovery_cfg.get("config")
            or discovery_cfg.get("subjects")
            or discovery_cfg.get("subject")
            or self.VALID_CONFIGS
        )
        if isinstance(configs, str):
            configs = [configs]
        return configs

    # ── Prompt formatting ───────────────────────────────────────────────────────

    def _format_wmdp_prompt(self, question: str, choices: List[str]) -> str:
        """
        Format WMDP question and choices into a prompt.

        Matches the format in wmdp_utils.format_wmdp_prompt but does NOT append
        the answer letter — the prompt ends at "Answer:" so that collate_EAP
        sets input_length correctly.
        """
        letters = ["A", "B", "C", "D"][: len(choices)]
        lines = [
            "Please choose the answer that is most relevant to the question from options A, B, C and D."
        ]
        lines.append("")
        lines.append(f"Q: {question}")
        for letter, choice in zip(letters, choices):
            lines.append(f"{letter}) {choice}")
        lines.append("")
        lines.append(_ANSWER_TAIL)
        return "\n".join(lines)

    def _wrap_wmdp_prompt(self, model, raw_prompt: str, *, apply: bool) -> str:
        """Apply (or not) the chat template to a raw WMDP prompt.

        The raw prompt ends with the answer-eliciting tail ``"Answer:"``; that
        tail becomes the assistant-turn prefix so the answer letter stays the
        immediate next token. When ``apply`` is False this returns ``raw_prompt``
        unchanged (byte-identical legacy behavior).
        """
        # Split off the trailing "Answer:" — it belongs in the assistant turn.
        user_text = raw_prompt[: -len(_ANSWER_TAIL)]
        return wrap_prompt(model, user_text, _ANSWER_TAIL, apply=apply)

    def _build_length_matched_corrupted_prompt(
        self,
        clean_prompt: str,
        choices: List[str],
        model,
        base_question: str = "Which is the most possible answer?",
        *,
        apply: bool = False,
    ) -> str:
        """
        Build a corrupted prompt whose tokenized length exactly matches clean_prompt.

        EAP patches activations at each absolute position i from the corrupted run
        into the clean run. If lengths differ, position i is semantically different
        across the two runs — every attribution score is wrong.

        Strategy: start from base_question, then iteratively append " the" or remove
        trailing words until the tokenized length matches.

        Args:
            clean_prompt: the already-wrapped clean prompt to length-match against.
            apply: resolved chat-template flag — corrupted candidates are wrapped
                with the identical template so clean/corrupted stay token-aligned.
        """
        target_len = to_tokens(model, clean_prompt, templated=apply).size(1)

        q = base_question
        prompt = self._wrap_wmdp_prompt(model, self._format_wmdp_prompt(q, choices), apply=apply)
        curr_len = to_tokens(model, prompt, templated=apply).size(1)

        if curr_len == target_len:
            return prompt

        NEUTRAL = " the"
        best_prompt, best_diff = prompt, abs(curr_len - target_len)

        if curr_len < target_len:
            for _ in range((target_len - curr_len) + 5):
                q += NEUTRAL
                p = self._wrap_wmdp_prompt(model, self._format_wmdp_prompt(q, choices), apply=apply)
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
                p = self._wrap_wmdp_prompt(
                    model, self._format_wmdp_prompt(" ".join(words), choices), apply=apply
                )
                lyr = to_tokens(model, p, templated=apply).size(1)
                diff = abs(lyr - target_len)
                if diff < best_diff:
                    best_diff, best_prompt = diff, p
                if lyr == target_len:
                    return p
                if lyr < target_len:
                    break

        return best_prompt

    # ── Single-example EAP data generation ──────────────────────────────────────

    def _generate_wmdp_eap_data_single(
        self, ex: Dict[str, Any], model, config: str = None, *, apply: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Generate EAP format data from a single WMDP example.

        Returns dict with keys: clean, corrupted, correct_idx, incorrect_idx
        or None if the example is malformed / tokenization fails.

        When ``apply`` is True the prompt is also wrapped in the model's chat
        template (the "Answer:" tail stays the assistant-turn prefix); the
        corrupted prompt is wrapped with the identical template/apply value so
        clean and corrupted stay token-aligned.
        """
        question = ex.get("question", "")
        choices = ex.get("choices", [])
        answer = ex.get("answer")

        if not question or not choices or answer is None:
            return None

        if not isinstance(choices, list):
            choices = list(choices)

        if not (0 <= answer < len(choices)):
            return None

        # Prompt ends at "Answer:" — no letter appended
        clean_prompt = self._wrap_wmdp_prompt(
            model, self._format_wmdp_prompt(question, choices), apply=apply
        )

        # Length-matched corrupted prompt
        corrupted_prompt = self._build_length_matched_corrupted_prompt(
            clean_prompt, choices, model, apply=apply
        )

        # Get token IDs for A/B/C/D as they appear in context
        try:
            option_tokens = []
            for letter in "ABCD":
                test_text = f"{clean_prompt} {letter}"
                tokens = to_tokens(model, test_text, templated=apply)
                option_tokens.append(tokens[0, -1].item())

            correct_token = option_tokens[answer]
            incorrect_tokens = [option_tokens[i] for i in range(4) if i != answer]

            while len(incorrect_tokens) < 3:
                incorrect_tokens.append(correct_token)

            all_option_tokens = [correct_token] + incorrect_tokens[:3]

        except Exception:
            return None

        return {
            "clean": clean_prompt,
            "corrupted": corrupted_prompt,
            "correct_idx": all_option_tokens[0],
            "incorrect_idx": all_option_tokens[1:4],
        }

    # ── Longest-sequence filtering ──────────────────────────────────────────────

    def _filter_longest_sequences_global(
        self,
        examples_with_configs: List[tuple],
        model,
        top_percent: float = 0.1,
        *,
        apply: bool = False,
    ) -> List[tuple]:
        """
        Filter out the top N% longest sequences across all configs.

        Args:
            examples_with_configs: List of (example, config) tuples
            model: HookedTransformer model for tokenization
            top_percent: Fraction of longest sequences to remove (default 10%)
            apply: resolved chat-template flag — length probes must match the
                tokenization used for the actual data.

        Returns:
            Filtered list with longest sequences removed
        """
        sequence_lengths = []

        for idx, (ex, config) in enumerate(examples_with_configs):
            question = ex.get("question", "")
            choices = ex.get("choices", [])
            answer = ex.get("answer")

            if not question or not choices or answer is None:
                continue
            if not isinstance(choices, list):
                choices = list(choices)
            if not (0 <= answer < len(choices)):
                continue

            try:
                prompt = self._wrap_wmdp_prompt(
                    model, self._format_wmdp_prompt(question, choices), apply=apply
                )
                tokens = to_tokens(model, prompt, templated=apply)
                sequence_lengths.append((tokens.size(1), idx))
            except Exception:
                continue

        if len(sequence_lengths) < 10:
            return examples_with_configs

        sequence_lengths.sort(reverse=True, key=lambda x: x[0])
        num_to_filter = max(1, int(len(sequence_lengths) * top_percent))
        longest_indices = {idx for _, idx in sequence_lengths[:num_to_filter]}

        filtered = [
            examples_with_configs[idx]
            for idx in range(len(examples_with_configs))
            if idx not in longest_indices
        ]

        if not filtered:
            logger.warning("Filtering would remove all examples, keeping originals")
            return examples_with_configs

        logger.info(
            f"  Filtered out {num_to_filter} longest sequences "
            f"(kept {len(filtered)}/{len(sequence_lengths)} examples)"
        )
        return filtered

    # ── Validation ──────────────────────────────────────────────────────────────

    def _validate_wmdp_data(  # noqa: C901 - complex function, refactor out of scope for lint pass
        self,
        algorithm: str,
        data,
        model,
        n_samples: int = 5,
        *,
        apply: bool = False,
    ) -> None:
        """
        Validate that WMDP data is correctly formatted for the given algorithm.

        Mirrors MMLUTaskSpec._validate_mmlu_data — same checks, WMDP header.
        """
        import torch

        LETTERS = {"A", "B", "C", "D"}
        header = f"[WMDP validate | {algorithm.upper()}]"

        if algorithm in ("eap", "eap-ig"):
            rows = data[:n_samples]
            if not rows:
                logger.info(f"{header} WARNING: no data rows to validate")
                return

            logger.info(f"{header} Validating {len(rows)} sample rows ...")

            end_ok = []
            len_match_ok = []
            label_ok = []
            distinct_ok = []

            for i, row in enumerate(rows):
                clean = row.get("clean", "")
                corrupted = row.get("corrupted", "")
                correct = row.get("correct_idx")
                incorrect = row.get("incorrect_idx", [])

                # Check 1: prompts end at "Answer:"
                clean_ends = isinstance(clean, str) and clean.rstrip().endswith("Answer:")
                corrupted_ends = isinstance(corrupted, str) and corrupted.rstrip().endswith(
                    "Answer:"
                )
                both_end = clean_ends and corrupted_ends
                end_ok.append(both_end)
                if not both_end:
                    logger.info(
                        f"  FAIL row {i} | prediction position: clean={clean_ends} corrupted={corrupted_ends}"
                    )

                # Check 2: same tokenized length
                try:
                    cl = to_tokens(model, clean, templated=apply).size(1)
                    crl = to_tokens(model, corrupted, templated=apply).size(1)
                    same = cl == crl
                    len_match_ok.append(same)
                    if not same:
                        logger.warning(
                            f"  WARN row {i} | length mismatch: clean={cl} corrupted={crl}"
                        )
                except Exception as e:
                    len_match_ok.append(False)
                    logger.error(f"  WARN row {i} | length check failed: {e}")

                # Check 3 & 4: labels
                if not isinstance(incorrect, list):
                    try:
                        import ast

                        incorrect = (
                            ast.literal_eval(incorrect)
                            if isinstance(incorrect, str)
                            else list(incorrect)
                        )
                    except Exception:
                        incorrect = []

                all_ids = [correct] + list(incorrect[:3])
                ids_valid = len(all_ids) == 4 and all(isinstance(x, int) for x in all_ids)

                if ids_valid:
                    try:
                        decoded = [
                            model.to_string(torch.tensor([tid], dtype=torch.long)).strip()
                            for tid in all_ids
                        ]
                        labels_are_letters = all(d in LETTERS for d in decoded)
                        ids_distinct = len(set(all_ids)) == 4
                        label_ok.append(labels_are_letters)
                        distinct_ok.append(ids_distinct)
                        if not labels_are_letters:
                            logger.info(
                                f"  FAIL row {i} | labels not A/B/C/D: {list(zip(all_ids, decoded))}"
                            )
                        if not ids_distinct:
                            logger.info(f"  FAIL row {i} | duplicate token IDs {all_ids}")
                    except Exception as e:
                        label_ok.append(False)
                        distinct_ok.append(False)
                        logger.error(f"  FAIL row {i} | could not decode labels: {e}")
                else:
                    label_ok.append(False)
                    distinct_ok.append(False)
                    logger.error(
                        f"  FAIL row {i} | malformed labels: correct={correct!r}, incorrect={incorrect!r}"
                    )

            n = len(rows)
            checks = {
                "prediction position (ends at 'Answer:')": end_ok,
                "label tokens are A/B/C/D": label_ok,
                "label token IDs are distinct": distinct_ok,
            }
            warnings_map = {
                "clean/corrupted same token length": len_match_ok,
            }

            critical_fail = False
            for name, results in checks.items():
                passed = sum(results)
                status = "PASS" if passed == n else ("WARN" if passed > 0 else "FAIL")
                logger.info(f"  {status:4s} | {name}: {passed}/{n}")
                if passed == 0:
                    critical_fail = True

            for name, results in warnings_map.items():
                passed = sum(results)
                status = "PASS" if passed == n else "WARN"
                logger.info(f"  {status:4s} | {name}: {passed}/{n}  (warning only)")

            if critical_fail:
                raise ValueError(
                    f"{header} Critical validation failure — every sampled row failed "
                    f"at least one required check."
                )
            logger.info(f"{header} Done.\n")

        elif algorithm == "ibcircuit":
            tokens = data["tokens"]
            labels = data["labels"]
            answer_positions = data["answer_positions"]

            B, max_len = tokens.shape
            n = min(n_samples, B)
            logger.info(f"{header} Validating {n}/{B} samples ...")

            pos_uniform_ok = []
            pos_in_bounds_ok = []
            token_ends_colon = []
            label_letter_ok = []

            for i in range(n):
                pos = answer_positions[i].item()

                pos_uniform_ok.append(pos == max_len - 1)
                if pos != max_len - 1:
                    logger.info(f"  FAIL row {i} | answer_pos={pos} != max_len-1={max_len - 1}")

                in_bounds = 0 <= pos < max_len
                pos_in_bounds_ok.append(in_bounds)
                if not in_bounds:
                    logger.info(f"  FAIL row {i} | answer_pos={pos} out of bounds")
                    continue

                try:
                    tok_str = model.to_string(tokens[i, pos : pos + 1].cpu()).strip()
                    ends_colon = tok_str.endswith(":")
                    token_ends_colon.append(ends_colon)
                    if not ends_colon:
                        logger.info(
                            f"  FAIL row {i} | token at answer_pos={tok_str!r}, expected ':'"
                        )
                except Exception as e:
                    token_ends_colon.append(False)
                    logger.error(f"  FAIL row {i} | decode failed: {e}")

                try:
                    label_str = model.to_string(labels[i : i + 1].cpu()).strip()
                    is_letter = label_str in LETTERS
                    label_letter_ok.append(is_letter)
                    if not is_letter:
                        logger.error(f"  FAIL row {i} | label={label_str!r}, expected A/B/C/D")
                except Exception as e:
                    label_letter_ok.append(False)
                    logger.error(f"  FAIL row {i} | label decode failed: {e}")

            checks = {
                "answer_positions all == max_len-1": pos_uniform_ok,
                "answer_positions within bounds": pos_in_bounds_ok,
                "token at answer_pos ends with ':'": token_ends_colon,
                "label tokens are A/B/C/D": label_letter_ok,
            }

            critical_fail = False
            for name, results in checks.items():
                passed = sum(results)
                status = "PASS" if passed == n else ("WARN" if passed > 0 else "FAIL")
                logger.info(f"  {status:4s} | {name}: {passed}/{n}")
                if passed == 0:
                    critical_fail = True

            if critical_fail:
                raise ValueError(f"{header} Critical validation failure.")
            logger.info(f"{header} Done.\n")
        else:
            logger.info(f"{header} No validation defined for algorithm '{algorithm}'")

    # ── DataLoader builders ─────────────────────────────────────────────────────

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> DataLoader:
        """
        Build DataLoader for WMDP task.

        Args:
            model: HookedTransformer model
            discovery_cfg: Discovery configuration
            device: Target device

        Returns:
            DataLoader configured for WMDP task
        """
        if model is None:
            raise ValueError("WMDP task requires model for tokenization. No default model.")

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if is_eap_family(algorithm) or algorithm == "cdt":
            return self._build_eap_dataloader(discovery_cfg, device, model)
        elif algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)
        else:
            from .._algorithm_families import CDT_FAMILY, EAP_FAMILY, IB_FAMILY

            raise ValueError(
                unsupported_algorithm_message(
                    "WMDP task", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

    def _build_eap_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """Build DataLoader for EAP/EAP-IG with corruption and length matching."""
        import tempfile

        from ...backends.eap.eap_utils import collate_EAP

        configs = self._resolve_configs(discovery_cfg)
        samples_per_config = discovery_cfg.get(
            "samples_per_config",
            discovery_cfg.get(
                "samples_per_subject", discovery_cfg.get("data_params", {}).get("num_examples", 20)
            ),
        )
        seed = discovery_cfg.get("seed", 42)
        filter_longest = discovery_cfg.get("filter_longest_sequences", True)

        # Resolve chat-template handling once: every prompt formatted/tokenized
        # below must use the same `apply` value to stay token-aligned.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        # 1. Load examples from all requested configs
        all_examples_with_configs = []

        logger.info(f"Loading data from {len(configs)} WMDP configs...")
        for config in configs:
            try:
                ds = _load_dataset_with_cache_fix("cais/wmdp", config, split="test")
            except Exception as e:
                logger.error(f"Warning: Failed to load WMDP config '{config}': {e}")
                continue

            if samples_per_config and len(ds) > samples_per_config:
                rng = _random.Random(seed)
                indices = rng.sample(range(len(ds)), samples_per_config)
                indices.sort()  # keep order stable for reproducibility
                ds = ds.select(indices)

            for ex in ds:
                all_examples_with_configs.append((ex, config))

            logger.debug(f"  Loaded {len(ds)} examples from {config}")

        if not all_examples_with_configs:
            raise ValueError("No valid WMDP examples loaded from any config")

        total_before = len(all_examples_with_configs)
        logger.info(f"Total examples collected: {total_before} from {len(configs)} configs")

        # 2. Filter top 10% longest sequences
        if filter_longest and len(all_examples_with_configs) > 10:
            logger.info("Filtering out top 10% longest sequences...")
            all_examples_with_configs = self._filter_longest_sequences_global(
                all_examples_with_configs, model, top_percent=0.1, apply=apply
            )
            logger.info(
                f"After filtering: {len(all_examples_with_configs)} examples "
                f"(removed {total_before - len(all_examples_with_configs)})"
            )

        # 3. Generate EAP data
        logger.info("Generating EAP format data...")
        all_eap_data = []

        for idx, (ex, config) in enumerate(all_examples_with_configs):
            if (idx + 1) % 10 == 0:
                logger.debug(f"  Processing example {idx + 1}/{len(all_examples_with_configs)}...")

            row = self._generate_wmdp_eap_data_single(ex, model, config, apply=apply)
            if row:
                all_eap_data.append(row)

        if not all_eap_data:
            raise ValueError("No valid WMDP EAP examples generated")

        logger.info(f"Total EAP examples generated: {len(all_eap_data)}")

        # 4. Validate
        self._validate_wmdp_data(
            algorithm=discovery_cfg.get("algorithm", "eap"),
            data=all_eap_data,
            model=model,
            apply=apply,
        )

        # 5. Write to CSV and build DataLoader
        df = pd.DataFrame(all_eap_data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        dataset = EAPDiscoveryDataset(temp_path)

        g = t.Generator()
        g.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=discovery_cfg.get("batch_size", 16),
            shuffle=True,
            collate_fn=collate_EAP,
            generator=g,
        )
        loader.pair_padding_side = discovery_cfg.get("pair_padding_side", self.pair_padding_side)
        # `apply` is the resolved chat-template boolean: when True the rows hold
        # chat-templated text (which already carries its own BOS), so the EAP
        # backend must tokenize with prepend_bos=False to avoid a double BOS.
        loader.templated = apply

        logger.debug(
            f"[DEBUG PADDING] wmdp EAP dataloader  pair_padding_side='{loader.pair_padding_side}'  "
            f"n_examples={len(all_eap_data)}  batch_size={discovery_cfg.get('batch_size', 16)}  "
            f"templated={apply}"
        )
        return loader

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """
        Build SingleBatchDataLoader for IBCircuit using WMDP data.

        Left-pads all sequences to uniform length so answer_positions == max_len - 1
        for every example.
        """
        import torch

        configs = self._resolve_configs(discovery_cfg)
        samples_per_config = discovery_cfg.get(
            "samples_per_config",
            discovery_cfg.get("samples_per_subject", 20),
        )
        load_factor = 5

        seed = discovery_cfg.get("seed", 42)

        # Resolve chat-template handling once (see _build_eap_dataloader).
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        pad_token_id = getattr(model.tokenizer, "pad_token_id", None) or 0

        logger.info(
            f"Loading WMDP data for IBCircuit ({len(configs)} configs, "
            f"{samples_per_config} samples each, seed={seed})..."
        )

        all_examples = []
        for config in configs:
            try:
                ds = _load_dataset_with_cache_fix("cais/wmdp", config, split="test")
                pool = ds.select(range(min(len(ds), samples_per_config * load_factor)))

                scored = []
                for ex in pool:
                    q = ex.get("question", "")
                    c = list(ex.get("choices", []))
                    if not q or not c:
                        continue
                    try:
                        wrapped = self._wrap_wmdp_prompt(
                            model, self._format_wmdp_prompt(q, c), apply=apply
                        )
                        toks = to_tokens(model, wrapped, templated=apply)
                        scored.append((toks.size(1), ex))
                    except Exception:
                        continue

                scored.sort(key=lambda x: x[0])
                kept = [ex for _, ex in scored[:samples_per_config]]
                all_examples.extend(kept)
                logger.debug(f"  {config}: kept {len(kept)} shortest examples")
            except Exception as e:
                logger.error(f"  Warning: failed to load '{config}': {e}")

        # Cap the single IBCircuit batch to avoid OOM from the [N, max_len, vocab]
        # logits tensor. Mirrors MMLU. data_params.num_examples is the public knob.
        data_params = discovery_cfg.get("data_params", {}) or {}
        max_examples = data_params.get("num_examples") or discovery_cfg.get("num_examples") or 32
        if isinstance(max_examples, int) and max_examples > 0 and len(all_examples) > max_examples:
            all_examples = all_examples[:max_examples]
            logger.info(
                f"  Capped IBCircuit batch at {len(all_examples)} examples "
                f"(num_examples={max_examples})"
            )

        # Tokenize: prompt + " " + correct_letter → input = all but last, label = last
        token_seqs = []
        label_ids = []

        for ex in all_examples:
            question = ex.get("question", "")
            choices = list(ex.get("choices", []))
            answer = ex.get("answer")
            if not question or not choices or answer is None:
                continue
            if not (0 <= answer < len(choices)):
                continue

            # Wrap the prompt (the "Answer:" tail stays the assistant-turn
            # prefix); the correct letter is appended after it so the label
            # token is exactly the token the model sees in context.
            prompt = self._wrap_wmdp_prompt(
                model, self._format_wmdp_prompt(question, choices), apply=apply
            )
            correct_letter = "ABCD"[answer]
            full_text = f"{prompt} {correct_letter}"

            try:
                full_tokens = to_tokens(model, full_text, templated=apply)[0]
                token_seqs.append(full_tokens[:-1])
                label_ids.append(full_tokens[-1].item())
            except Exception:
                continue

        if not token_seqs:
            raise ValueError("No WMDP examples could be tokenized for IBCircuit")

        # Left-pad to uniform length
        lengths = torch.tensor([s.size(0) for s in token_seqs], dtype=torch.long)
        max_len = lengths.max().item()

        padded = torch.full((len(token_seqs), max_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(token_seqs):
            padded[i, max_len - seq.size(0) :] = seq  # left-pad

        tokens_tensor = padded.to(device)
        labels_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
        answer_positions = torch.full(
            (len(token_seqs),), max_len - 1, dtype=torch.long, device=device
        )

        logger.info(
            f"IBCircuit WMDP batch ready: {len(token_seqs)} examples, "
            f"max_len={max_len}, "
            f"answer_pos range [{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        batch = {
            "tokens": tokens_tensor,
            "labels": labels_tensor,
            "answer_positions": answer_positions,
        }

        logger.debug(
            f"[DEBUG PADDING] wmdp IBCircuit  within-batch=left-padded  max_len={max_len}  "
            f"all answer_pos={max_len - 1}"
        )

        # Validate
        self._validate_wmdp_data(algorithm="ibcircuit", data=batch, model=model, apply=apply)

        class SingleBatchDataLoader:
            """Yields a single fixed batch repeatedly."""

            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        # Debug: verify first 2 examples
        for i in range(min(2, len(token_seqs))):
            pos = answer_positions[i].item()
            logger.debug(
                f"[DEBUG WMDP-IB] Example {i}: seq_len={lengths[i].item()}, answer_pos={pos}"
            )
            logger.debug(f"  Token at answer_pos: {model.to_string(tokens_tensor[i, pos:pos+1])!r}")
            logger.debug(
                f"  Label token: {model.to_string(labels_tensor[i:i+1])!r} (id={labels_tensor[i].item()})"
            )

        if len(token_seqs) < 20:
            logger.warning(
                f"IBCircuit batch has only {len(token_seqs)} examples. "
                f"Minimum ~20-50 recommended for reliable statistics."
            )

        return SingleBatchDataLoader(batch)

    # ── Metric ──────────────────────────────────────────────────────────────────

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the EAP/EAP-IG compatible metric for WMDP."""
        if metric_type == "kl":
            return partial(self._eap_wmdp_kl_divergence, loss=True, mean=True)
        return partial(self._eap_logit_diff_wmdp, loss=True, mean=True)

    @staticmethod
    def _eap_logit_diff_wmdp(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        WMDP metric: logit(correct) - mean(logits(incorrect)).

        Same as MMLU — 4-way multiple choice with identical label layout.
        """
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        last_logits = logits[idx, input_length - 1]

        correct_logits = t.gather(last_logits, -1, labels[:, 0:1].to(logits.device))
        incorrect_logits = t.gather(last_logits, -1, labels[:, 1:4].to(logits.device))

        avg_incorrect = incorrect_logits.mean(dim=-1, keepdim=True)
        results = correct_logits.squeeze(-1) - avg_incorrect.squeeze(-1)

        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _eap_wmdp_kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """KL divergence at the answer position."""
        import torch.nn.functional as F

        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        last_logits = logits[idx, input_length - 1]
        last_clean_logits = clean_logits[idx, input_length - 1]

        results = F.kl_div(
            F.log_softmax(last_logits, dim=-1),
            F.softmax(last_clean_logits, dim=-1),
            reduction="batchmean" if mean else "none",
        )
        return results

    # ── Metadata ────────────────────────────────────────────────────────────────

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for WMDP artifacts."""
        return {
            "task": "wmdp",
            "configs": self._resolve_configs(discovery_cfg),
            "samples_per_config": discovery_cfg.get(
                "samples_per_config",
                discovery_cfg.get("samples_per_subject", 20),
            ),
            "filter_longest_sequences": discovery_cfg.get("filter_longest_sequences", True),
            "corruption_mode": "question_replacement",
            "corruption_text": "Which is the most possible answer?",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "model_name": discovery_cfg.get("model_name", "gpt2"),
            # Resolved chat-template mode — later stages read this back unchanged.
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }

    # ── Finetuning dataset ──────────────────────────────────────────────────────

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate WMDP finetuning data from HuggingFace.

        Draws fresh from 'cais/wmdp'. When the finetuning model is
        instruction-tuned (and the task's resolved ``chat_template_mode`` is not
        ``"off"``) each prompt is wrapped in the model's chat template with the
        same ``Answer:`` assistant prefix discovery uses, so circuit-tuning
        trains on the discovery prompt distribution. For base models / ``"off"``
        the prompt text is byte-identical to the legacy raw-text behavior.
        """
        cfg = discovery_cfg or {}
        configs = self._resolve_configs(cfg)
        samples_per_config = cfg.get(
            "samples_per_config",
            cfg.get("samples_per_subject", 20),
        )

        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model); a discovery_cfg override wins.
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)

        def _wrap_query(raw_query: str) -> str:
            """Wrap a raw WMDP prompt, keeping "Answer:" as the assistant prefix
            (no-op when apply is False — byte-identical to _format_wmdp_prompt)."""
            user_text = raw_query[: -len(_ANSWER_TAIL)]
            return wrap_prompt_with_tokenizer(tokenizer, user_text, _ANSWER_TAIL, apply=apply)

        all_pairs: List[Tuple[str, str]] = []

        for config in configs:
            try:
                ds = _load_dataset_with_cache_fix("cais/wmdp", config, split="test")
                for ex in ds.select(range(min(len(ds), samples_per_config))):
                    question = ex.get("question", "")
                    choices = list(ex.get("choices", []))
                    answer = ex.get("answer")

                    if not question or not choices or answer is None:
                        continue
                    if not (0 <= answer < len(choices)):
                        continue

                    query = _wrap_query(self._format_wmdp_prompt(question, choices))
                    letter = "ABCD"[answer]
                    full_text = f"{query} {letter}"
                    all_pairs.append((query, full_text))
            except Exception:
                continue

        if not all_pairs:
            raise ValueError(
                "No WMDP data could be loaded for finetuning. "
                "Check internet connection and configs in discovery_cfg."
            )

        rng = _random.Random(seed)
        rng.shuffle(all_pairs)
        all_pairs = all_pairs[:n_examples]

        query_strings = [q for q, _ in all_pairs]
        clean_texts = [ft for _, ft in all_pairs]
        return clean_texts, query_strings


def build_wmdp_spec(
    subset: str = "wmdp-bio",
    split: str = "test",
    name: Optional[str] = None,
    max_records: Optional[int] = None,
) -> "WMDPTaskSpec":
    """Factory that returns a WMDPTaskSpec pinned to a specific subset.

    Args:
        subset:      One of ``"wmdp-bio"``, ``"wmdp-chem"``, ``"wmdp-cyber"``.
        split:       Dataset split (default ``"test"``).
        name:        Override the task name (default ``"wmdp"``).
        max_records: Cap on the number of records to use.
    """
    _pinned = subset

    class _PinnedWMDPTaskSpec(WMDPTaskSpec):
        def _resolve_configs(self, discovery_cfg: Dict[str, Any]) -> List[str]:
            cfg_subset = (
                discovery_cfg.get("configs")
                or discovery_cfg.get("config")
                or discovery_cfg.get("subjects")
                or discovery_cfg.get("subject")
            )
            if isinstance(cfg_subset, str):
                return [cfg_subset]
            return cfg_subset if cfg_subset else [_pinned]

    spec = _PinnedWMDPTaskSpec()
    if name is not None:
        spec.name = name  # type: ignore[assignment]
    return spec
