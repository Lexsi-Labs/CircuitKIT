"""
MMLU (Massive Multitask Language Understanding) Task Specification

Implements the TaskSpec interface for the MMLU task with support for EAP and EAP-IG.
Uses corruption strategy of replacing question stem with "Which is the most possible answer?"
"""

import random as _random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch as t
from datasets import load_dataset
from torch.utils.data import DataLoader

from ...data.eap_dataset import EAPDiscoveryDataset
from ...utils.logging import get_logger
logger = get_logger("task.mmlu")

from .._algorithm_families import is_eap_family, unsupported_algorithm_message
from .._chat import (
    resolve_chat_template,
    resolve_chat_template_from_tokenizer,
    to_tokens,
    wrap_prompt,
    wrap_prompt_with_tokenizer,
)

#: Answer-eliciting tail that ends every MMLU prompt. Used to split the prompt
#: into a user turn (question + choices) and an assistant-turn prefix so a chat
#: template can be applied without moving the answer off the next-token slot.
_ANSWER_TAIL = "Answer:"

class MMLUTaskSpec:
    """TaskSpec implementation for MMLU (Massive Multitask Language Understanding)."""

    name = "mmlu"
    pair_padding_side = "left"
    # Downstream-behavior MC task: wrap prompts in the model's chat template
    # iff the model is instruction-tuned ("auto"). Discovery must match how the
    # model is actually evaluated, or the discovered circuit is misattributed.
    chat_template_mode: str = "auto"

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """
        Validate MMLU-specific discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        algorithm = discovery_cfg.get("algorithm", "").lower()

        from .._algorithm_families import CDT_FAMILY, EAP_FAMILY, IB_FAMILY

        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "MMLU task", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        if algorithm in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            # MMLU task requires model for tokenization
            if "model_name" not in discovery_cfg:
                raise ValueError(
                    "MMLU task discovery config is missing the required key "
                    "'model_name'. Add 'model_name' to the discovery config so "
                    "the task can load the tokenizer."
                )

            # Validate subjects
            subjects = (
                discovery_cfg.get("subjects")
                or discovery_cfg.get("subject")
                or self._get_all_mmlu_subjects()
            )
            if isinstance(subjects, str):
                subjects = [subjects]

            valid_subjects = self._get_all_mmlu_subjects()
            for subject in subjects:
                if subject not in valid_subjects:
                    raise ValueError(
                        f"Invalid MMLU subject: {subject!r}. "
                        f"Set discovery config key 'subjects' (or 'subject') to "
                        f"valid MMLU subject name(s); one of: {valid_subjects}"
                    )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"MMLU ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        # Validate level
        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"MMLU task discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        # Validate batch_size if present
        batch_size = discovery_cfg.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError(
                f"MMLU task has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 16)."
            )

        # Validate samples_per_subject if present
        samples_per_subject = discovery_cfg.get("samples_per_subject")
        if samples_per_subject is not None and (
            not isinstance(samples_per_subject, int) or samples_per_subject <= 0
        ):
            raise ValueError(
                f"MMLU task has invalid 'samples_per_subject': {samples_per_subject!r}. "
                f"Set discovery config key 'samples_per_subject' to a positive integer."
            )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> DataLoader:
        """
        Build DataLoader for MMLU task.

        Args:
            model: HookedTransformer model
            discovery_cfg: Discovery configuration
            device: Target device

        Returns:
            DataLoader configured for MMLU task
        """
        if model is None:
            raise ValueError("MMLU task requires model for tokenization. No default model.")

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if is_eap_family(algorithm) or algorithm == "cdt":
            return self._build_mmlu_dataloader(discovery_cfg, device, model)
        elif algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)
        else:
            from .._algorithm_families import CDT_FAMILY, EAP_FAMILY, IB_FAMILY

            raise ValueError(
                unsupported_algorithm_message(
                    "MMLU task", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

    def _get_all_mmlu_subjects(self) -> List[str]:
        """Get list of all MMLU subjects."""
        return [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]

    def _build_mmlu_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """Build DataLoader using MMLU dataset with corruption strategy."""
        import tempfile

        # Get configuration parameters
        subjects = (
            discovery_cfg.get("subjects")
            or discovery_cfg.get("subject")
            or self._get_all_mmlu_subjects()
        )
        samples_per_subject = discovery_cfg.get("samples_per_subject", 20)
        seed = discovery_cfg.get("seed", 42)
        model_name = discovery_cfg["model_name"]
        filter_longest_sequences = discovery_cfg.get(
            "filter_longest_sequences", True
        )  # Default: filter top 10% longest

        # Resolve chat-template handling once: every prompt formatted/tokenized
        # below must use the same `apply` value to stay token-aligned.
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        # Ensure subjects is a list
        if isinstance(subjects, str):
            subjects = [subjects]

        # First, load all datasets and collect examples with subject info
        all_examples_with_subjects = []

        logger.info(f"Loading data from {len(subjects)} subjects...")

        # Fast path: when loading many subjects, pull MMLU's single "all"
        # config once (~3s) and group rows by subject, instead of issuing one
        # load_dataset("cais/mmlu", <subject>) call per subject (~1.4s each,
        # ~80s for the full 56-subject set). The per-subject rows and their
        # order are identical between the "all" config and the per-subject
        # configs, so downstream selection/tokenization is unchanged. This
        # matters because discover + evaluate each rebuild the dataloader, so a
        # 12-algorithm benchmark otherwise reloads the same data ~24 times.
        grouped: dict = {}
        if len(subjects) > 5:
            try:
                ds_all = load_dataset("cais/mmlu", "all", split="test")
                for ex in ds_all.to_list():
                    grouped.setdefault(ex["subject"], []).append(ex)
            except Exception as e:
                logger.error(
                    f"Warning: bulk-load of MMLU 'all' config failed ({e}); "
                    f"falling back to per-subject loads"
                )
                grouped = {}

        for subject in subjects:
            if grouped:
                ds = grouped.get(subject)
                if ds is None:
                    logger.error(
                        f"Warning: subject '{subject}' not present in MMLU 'all' config"
                    )
                    continue
            else:
                try:
                    ds = load_dataset(
                        "cais/mmlu",
                        subject,
                        split="test",
                    )
                except Exception as e:
                    logger.error(
                        f"Warning: Failed to load MMLU dataset for subject '{subject}': {e}"
                    )
                    continue

            # Limit samples per subject (first N rows, matching the historical
            # ds.select(range(N)) behaviour for both Dataset and list forms).
            if samples_per_subject and len(ds) > samples_per_subject:
                ds = (
                    ds.select(range(samples_per_subject))
                    if hasattr(ds, "select")
                    else ds[:samples_per_subject]
                )

            # Collect examples with subject information
            for ex in ds:
                all_examples_with_subjects.append((ex, subject))

            logger.debug(f"  Loaded {len(ds)} examples from {subject}")

        if not all_examples_with_subjects:
            raise ValueError("No valid MMLU examples loaded from any subject")

        total_before_filtering = len(all_examples_with_subjects)
        logger.info(
            f"Total examples collected: {total_before_filtering} from {len(subjects)} subjects"
        )

        # Filter out top 10% longest sequences across ALL subjects if enabled
        if filter_longest_sequences and len(all_examples_with_subjects) > 10:
            logger.info("Filtering out top 10% longest sequences across all subjects...")
            all_examples_with_subjects = self._filter_longest_sequences_global(
                all_examples_with_subjects, model, top_percent=0.1, apply=apply
            )
            logger.info(
                f"After filtering: {len(all_examples_with_subjects)} examples (removed {total_before_filtering - len(all_examples_with_subjects)})"
            )

        # Now generate EAP data from filtered examples
        logger.info("Generating EAP format data...")
        all_eap_data = []
        current_subject = None
        subject_eap_data = []
        total_processed = 0

        for idx, (ex, subject) in enumerate(all_examples_with_subjects):
            if subject != current_subject:
                # Print summary for previous subject
                if current_subject is not None:
                    logger.debug(
                        f"  Generated {len(subject_eap_data)} EAP examples from {current_subject}"
                    )
                    all_eap_data.extend(subject_eap_data)
                # Start new subject
                current_subject = subject
                subject_eap_data = []

            # Progress update every 10 examples
            if (idx + 1) % 10 == 0:
                logger.debug(f"  Processing example {idx + 1}/{len(all_examples_with_subjects)}...")

            # Generate EAP format data for this example
            example_eap_data = self._generate_mmlu_eap_data_single(
                ex, model, model_name, seed, subject, apply=apply
            )
            if example_eap_data:
                subject_eap_data.append(example_eap_data)
                total_processed += 1

        # Handle last subject
        if current_subject is not None and subject_eap_data:
            logger.debug(f"  Generated {len(subject_eap_data)} EAP examples from {current_subject}")
            all_eap_data.extend(subject_eap_data)

        if not all_eap_data:
            raise ValueError("No valid MMLU examples generated from any subject")

        logger.info(
            f"Total EAP examples generated: {len(all_eap_data)} from {len(subjects)} subjects"
        )

        # ── Validate a sample of generated rows before writing to CSV ──
        self._validate_mmlu_data(
            algorithm=discovery_cfg.get("algorithm", "eap"),
            data=all_eap_data,
            model=model,
            apply=apply,
        )

        # Save to temporary CSV
        df = pd.DataFrame(all_eap_data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        # Create EAP dataset
        dataset = EAPDiscoveryDataset(temp_path)

        # Create dataloader
        from ...backends.eap.eap_utils import collate_EAP

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
        # Flag templated prompts so the EAP backend tokenizes with prepend_bos=False
        # (the chat template already renders its own BOS — see tasks/_chat.py).
        loader.templated = apply
        logger.debug(
            f"[DEBUG PADDING] mmlu EAP dataloader  pair_padding_side='{loader.pair_padding_side}'  n_examples={len(all_eap_data)}  batch_size={discovery_cfg.get('batch_size', 16)}"
        )
        return loader

    def _filter_longest_sequences_global(
        self,
        examples_with_subjects: List[tuple],
        model,
        top_percent: float = 0.1,
        *,
        apply: bool = False,
    ) -> List[tuple]:
        """
        Filter out the top N% longest sequences across all subjects.

        Args:
            examples_with_subjects: List of (example, subject) tuples
            model: HookedTransformer model for tokenization
            top_percent: Percentage of longest sequences to filter (default: 0.1 for top 10%)
            apply: resolved chat-template flag — length probes must match the
                tokenization used for the actual data.

        Returns:
            Filtered list of (example, subject) tuples with longest sequences removed
        """

        sequence_lengths = []

        # Measure sequence lengths for all examples across all subjects
        for idx, (ex, subject) in enumerate(examples_with_subjects):
            question = ex.get("question", "")
            choices = ex.get("choices", [])
            answer = ex.get("answer")

            if not question or not choices or answer is None:
                continue

            # Format the prompt to measure its length
            if not isinstance(choices, list):
                choices = list(choices)

            if not (0 <= answer < len(choices)):
                continue  # skip malformed, consistent with _generate_mmlu_eap_data_single

            clean_prompt = self._wrap_mmlu_prompt(
                model, self._format_mmlu_prompt(question, choices), apply=apply
            )
            # Measure prompt-only length — what is actually fed to the model.
            # build_optimized_dataloader and build_smart_dataloader already measure
            # prompt-only length; this makes _filter_longest_sequences_global consistent.
            try:
                tokens = to_tokens(model, clean_prompt, templated=apply)
                seq_len = tokens.size(1)
                sequence_lengths.append((seq_len, idx))
            except Exception:
                # Skip if tokenization fails
                continue

        if len(sequence_lengths) < 10:
            # Too few examples to filter meaningfully
            return examples_with_subjects

        # Sort by length (longest first)
        sequence_lengths.sort(reverse=True, key=lambda x: x[0])

        # Calculate how many to filter (top N%)
        num_to_filter = max(1, int(len(sequence_lengths) * top_percent))
        longest_indices = {idx for _, idx in sequence_lengths[:num_to_filter]}

        # Get examples to keep (exclude longest indices)
        filtered_examples = [
            examples_with_subjects[idx]
            for idx in range(len(examples_with_subjects))
            if idx not in longest_indices
        ]

        if len(filtered_examples) == 0:
            # If filtering would remove everything, keep the examples as-is
            logger.info("  Warning: Filtering would remove all examples, keeping original examples")
            return examples_with_subjects

        logger.info(
            f"  Filtered out {num_to_filter} longest sequences (kept {len(filtered_examples)}/{len(sequence_lengths)} examples)"
        )

        return filtered_examples

    def _generate_mmlu_eap_data_single(
        self, ex, model, model_name: str, seed: int, subject: str = None, *, apply: bool = False
    ) -> Optional[Dict[str, Any]]:
        question = ex.get("question", "")
        choices = ex.get("choices", [])
        answer = ex.get("answer")

        if not question or not choices or answer is None:
            return None

        if not isinstance(choices, list):
            choices = list(choices)

        if not (0 <= answer < len(choices)):
            return None  # malformed example — skip rather than silently mislabel

        # Prompts end at "Answer:" — do NOT append the letter.
        # collate_EAP sets input_length = tokenized length of clean, so the metric
        # evaluates logits[input_length - 1] = logits at "Answer:" predicting the letter.
        # When `apply` is True the prompt is also wrapped in the model's chat
        # template (the "Answer:" tail stays the assistant-turn prefix).
        clean_prompt = self._wrap_mmlu_prompt(
            model, self._format_mmlu_prompt(question, choices), apply=apply
        )

        # Build a corrupted prompt that tokenizes to exactly the same length as
        # clean_prompt. EAP patches activations at each absolute position i from
        # the corrupted run into the clean run. If lengths differ, position i means
        # a different semantic location in each run — every attribution score is wrong.
        # The corrupted prompt is wrapped with the identical template/apply value
        # so the two stay token-aligned.
        corrupted_prompt = self._build_length_matched_corrupted_prompt(
            clean_prompt, choices, model, apply=apply
        )

        # Get token IDs for A/B/C/D as they appear in context (after "Answer:").
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

    def _generate_mmlu_eap_data(
        self,
        dataset,
        model,
        model_name: str,
        seed: int,
        subject: str = None,
        *,
        apply: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate EAP format data from MMLU dataset with corruption strategy.

        Corruption strategy: Replace question stem with "Which is the most possible answer?"
        For MMLU metric, we need all 4 option token IDs: [correct, incorrect1, incorrect2, incorrect3]

        Note: This method is kept for backward compatibility. For new code, use _generate_mmlu_eap_data_single
        for individual examples.
        """

        eap_data = []

        for idx, ex in enumerate(dataset):
            if idx % 10 == 0:
                subject_info = f" ({subject})" if subject else ""
                logger.info(f"Processing MMLU example {idx}/{len(dataset)}{subject_info}...")

            example_eap_data = self._generate_mmlu_eap_data_single(
                ex, model, model_name, seed, subject, apply=apply
            )
            if example_eap_data:
                eap_data.append(example_eap_data)

        return eap_data

    def _format_mmlu_prompt(self, question: str, choices: List[str]) -> str:
        """Format MMLU question and choices into a prompt (raw, ends at "Answer:")."""
        letters = ["A", "B", "C", "D"][: len(choices)]
        lines = [f"Q: {question}"]
        for letter, choice in zip(letters, choices):
            lines.append(f"{letter}) {choice}")
        lines.append(_ANSWER_TAIL)
        return "\n".join(lines)

    def _wrap_mmlu_prompt(self, model, raw_prompt: str, *, apply: bool) -> str:
        """Apply (or not) the chat template to a raw MMLU prompt.

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

        EAP/EAP-IG patches activations at each absolute token position i from the
        corrupted run into the clean run. If the two sequences have different lengths,
        position i in each run is a different semantic location — every attribution
        score is wrong. IOI and Greater-Than avoid this by construction (same template,
        single-token swaps). This method enforces the same invariant for MMLU.

        Strategy: start from base_question, then iteratively append " the" (a reliable
        single token in all standard BPE tokenizers) or remove trailing words until the
        tokenized length matches the clean prompt exactly.

        Args:
            clean_prompt: the already-wrapped clean prompt to length-match against.
            apply: resolved chat-template flag — corrupted candidates are wrapped
                with the identical template so clean/corrupted stay token-aligned.
        """
        target_len = to_tokens(model, clean_prompt, templated=apply).size(1)

        q = base_question
        prompt = self._wrap_mmlu_prompt(model, self._format_mmlu_prompt(q, choices), apply=apply)
        curr_len = to_tokens(model, prompt, templated=apply).size(1)

        if curr_len == target_len:
            return prompt

        NEUTRAL = " the"  # single token in all common BPE tokenizers
        best_prompt, best_diff = prompt, abs(curr_len - target_len)

        if curr_len < target_len:
            # Too short — append neutral tokens one at a time.
            for _ in range((target_len - curr_len) + 5):
                q += NEUTRAL
                p = self._wrap_mmlu_prompt(model, self._format_mmlu_prompt(q, choices), apply=apply)
                lyr = to_tokens(model, p, templated=apply).size(1)
                diff = abs(lyr - target_len)
                if diff < best_diff:
                    best_diff, best_prompt = diff, p
                if lyr == target_len:
                    return p
                if lyr > target_len:
                    break  # overshot due to BPE merge; best_prompt holds closest
        else:
            # Too long — remove words from the end one at a time.
            words = q.split()
            while len(words) > 1:
                words = words[:-1]
                p = self._wrap_mmlu_prompt(
                    model, self._format_mmlu_prompt(" ".join(words), choices), apply=apply
                )
                lyr = to_tokens(model, p, templated=apply).size(1)
                diff = abs(lyr - target_len)
                if diff < best_diff:
                    best_diff, best_prompt = diff, p
                if lyr == target_len:
                    return p
                if lyr < target_len:
                    break  # overshot going down; best_prompt holds closest

        return best_prompt

    def _validate_mmlu_data(  # noqa: C901 - complex function, refactor out of scope for lint pass
        self,
        algorithm: str,
        data,  # List[dict] for EAP, dict-of-tensors for ibcircuit
        model,
        n_samples: int = 5,
        *,
        apply: bool = False,
    ) -> None:
        """
        Validate that MMLU data is correctly formatted for the given algorithm.

        For EAP/EAP-IG validates a sample of raw data rows (List[dict]) checking:
        - clean/corrupted prompts end at "Answer:" (no letter appended)
        - clean and corrupted tokenize to the same length (EAP patches at position i;
            mismatched lengths make position i semantically different across the two runs)
        - correct_idx and all incorrect_idx decode to single letter tokens A/B/C/D
        - all 4 token IDs are distinct (duplicates collapse logit difference to zero)

        For IBCircuit validates the final batch dict (tokens/labels/answer_positions):
        - answer_positions are all max_len - 1 (expected from left-padding design)
        - answer_positions are within tensor bounds
        - token at answer_positions[i] ends with ":" (last token of "Answer:")
        - labels decode to single letter tokens A/B/C/D

        Prints a per-check PASS/FAIL report. Raises ValueError only if every sampled
        example fails a critical check (wrong prediction position or invalid labels).
        Length-mismatch issues are warnings only, as the corruption strategy is a
        pre-existing design decision.
        """
        import torch

        LETTERS = {"A", "B", "C", "D"}
        header = f"[MMLU validate | {algorithm.upper()}]"

        # ── EAP / EAP-IG ──────────────────────────────────────────────────────────
        if algorithm in ("eap", "eap-ig"):
            rows = data[:n_samples]
            if not rows:
                logger.info(f"{header} WARNING: no data rows to validate")
                return

            logger.info(f"{header} Validating {len(rows)} sample rows ...")

            end_ok = []  # clean/corrupted end at "Answer:"
            len_match_ok = []  # clean and corrupted tokenize to same length
            label_ok = []  # correct_idx + incorrect_idx decode to distinct letters
            distinct_ok = []  # all 4 token IDs are distinct

            for i, row in enumerate(rows):
                clean = row.get("clean", "")
                corrupted = row.get("corrupted", "")
                correct = row.get("correct_idx")
                incorrect = row.get("incorrect_idx", [])

                # --- Check 1: prediction position ---
                clean_ends_ok = isinstance(clean, str) and clean.rstrip().endswith("Answer:")
                corrupted_ends_ok = isinstance(corrupted, str) and corrupted.rstrip().endswith(
                    "Answer:"
                )
                both_end_ok = clean_ends_ok and corrupted_ends_ok
                end_ok.append(both_end_ok)
                if not both_end_ok:
                    logger.info(
                        f"  FAIL row {i} | prediction position: "
                        f"clean ends={clean_ends_ok!r} corrupted ends={corrupted_ends_ok!r}"
                    )
                    logger.info(f"    clean    tail: {repr(clean[-30:])}")
                    logger.info(f"    corrupted tail: {repr(corrupted[-30:])}")

                # --- Check 2: clean/corrupted same tokenized length ---
                try:
                    clean_len = to_tokens(model, clean, templated=apply).size(1)
                    corrupted_len = to_tokens(model, corrupted, templated=apply).size(1)
                    same_len = clean_len == corrupted_len
                    len_match_ok.append(same_len)
                    if not same_len:
                        logger.warning(
                            f"  WARN row {i} | length mismatch: "
                            f"clean={clean_len} corrupted={corrupted_len} tokens "
                            f"(EAP patches at position i; mismatched lengths make that "
                            f"position semantically different across the two runs)"
                        )
                except Exception as e:
                    len_match_ok.append(False)
                    logger.error(f"  WARN row {i} | length check failed: {e}")

                # --- Check 3 & 4: label tokens are letters and are distinct ---
                if not isinstance(incorrect, list):
                    # handles string-serialised list from CSV round-trip
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
                                f"  FAIL row {i} | label tokens not all A/B/C/D: "
                                f"{list(zip(all_ids, decoded))}"
                            )
                        if not ids_distinct:
                            logger.info(
                                f"  FAIL row {i} | duplicate token IDs {all_ids} — "
                                f"logit difference collapses to zero for duplicates"
                            )
                    except Exception as e:
                        label_ok.append(False)
                        distinct_ok.append(False)
                        logger.error(f"  FAIL row {i} | could not decode label tokens: {e}")
                else:
                    label_ok.append(False)
                    distinct_ok.append(False)
                    logger.error(
                        f"  FAIL row {i} | malformed labels: "
                        f"correct={correct!r}, incorrect={incorrect!r}"
                    )

            # ── summary ──
            n = len(rows)
            checks = {
                "prediction position (ends at 'Answer:')": end_ok,
                "label tokens are A/B/C/D": label_ok,
                "label token IDs are distinct": distinct_ok,
            }
            warnings = {
                "clean/corrupted same token length": len_match_ok,
            }

            critical_fail = False
            for name, results in checks.items():
                passed = sum(results)
                status = "PASS" if passed == n else ("WARN" if passed > 0 else "FAIL")
                logger.info(f"  {status:4s} | {name}: {passed}/{n}")
                if passed == 0:
                    critical_fail = True

            for name, results in warnings.items():
                passed = sum(results)
                status = "PASS" if passed == n else "WARN"
                logger.info(f"  {status:4s} | {name}: {passed}/{n}  (warning only)")

            if critical_fail:
                raise ValueError(
                    f"{header} Critical validation failure — every sampled row failed "
                    f"at least one required check. Inspect the FAIL lines above."
                )

            logger.info(f"{header} Done.\n")

        # ── IBCircuit ──────────────────────────────────────────────────────────────
        elif algorithm == "ibcircuit":
            tokens = data["tokens"]  # [B, max_len]
            labels = data["labels"]  # [B]
            answer_positions = data["answer_positions"]  # [B]

            B, max_len = tokens.shape
            n = min(n_samples, B)
            logger.info(f"{header} Validating {n}/{B} samples ...")

            max_len_val = max_len  # expected answer_pos for every row

            pos_uniform_ok = []
            pos_in_bounds_ok = []
            token_ends_colon = []
            label_letter_ok = []

            for i in range(n):
                pos = answer_positions[i].item()

                # --- Check 1: answer_positions uniformity ---
                pos_uniform_ok.append(pos == max_len_val - 1)
                if pos != max_len_val - 1:
                    logger.info(
                        f"  FAIL row {i} | answer_pos={pos} != max_len-1={max_len_val-1} "
                        f"(left-padding should align all sequences at max_len-1)"
                    )

                # --- Check 2: answer_positions within bounds ---
                in_bounds = 0 <= pos < max_len
                pos_in_bounds_ok.append(in_bounds)
                if not in_bounds:
                    logger.info(f"  FAIL row {i} | answer_pos={pos} out of bounds [0, {max_len})")
                    continue  # can't decode if out of bounds

                # --- Check 3: token at answer_pos ends with ":" ---
                try:
                    tok_str = model.to_string(tokens[i, pos : pos + 1].cpu()).strip()
                    ends_colon = tok_str.endswith(":")
                    token_ends_colon.append(ends_colon)
                    if not ends_colon:
                        logger.info(
                            f"  FAIL row {i} | token at answer_pos decodes to "
                            f"{tok_str!r}, expected it to end with ':' "
                            f"(should be end of 'Answer:')"
                        )
                        # Also show the surrounding context for diagnosis
                        ctx_start = max(0, pos - 3)
                        ctx_toks = tokens[i, ctx_start : pos + 2].cpu()
                        logger.info(f"    context: {model.to_string(ctx_toks)!r}")
                except Exception as e:
                    token_ends_colon.append(False)
                    logger.error(f"  FAIL row {i} | could not decode token at answer_pos: {e}")

                # --- Check 4: label decodes to a single letter A/B/C/D ---
                try:
                    label_str = model.to_string(labels[i : i + 1].cpu()).strip()
                    is_letter = label_str in LETTERS
                    label_letter_ok.append(is_letter)
                    if not is_letter:
                        logger.error(
                            f"  FAIL row {i} | label token {labels[i].item()} "
                            f"decodes to {label_str!r}, expected A/B/C/D"
                        )
                except Exception as e:
                    label_letter_ok.append(False)
                    logger.error(f"  FAIL row {i} | could not decode label: {e}")

            # ── summary ──
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
                raise ValueError(
                    f"{header} Critical validation failure — every sampled row failed "
                    f"at least one required check. Inspect the FAIL lines above."
                )

            logger.info(f"{header} Done.\n")

        else:
            logger.info(f"{header} No validation defined for algorithm '{algorithm}'")

    def build_optimized_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """
        OOM-Preventing Dataloader:
        1. Loads 5 * samples_per_subject for each subject.
        2. Calculates token length for every candidate.
        3. Selects the shortest 'samples_per_subject' to minimize VRAM usage.
        4. Returns the standard EAP DataLoader.
        """
        import tempfile

        from ...backends.eap.eap_utils import collate_EAP

        # 1. Parse Config
        subjects = discovery_cfg.get("subjects") or self._get_all_mmlu_subjects()
        if isinstance(subjects, str):
            subjects = [subjects]

        target_n = discovery_cfg.get("samples_per_subject", 5)
        load_factor = 5  # Load 5x, keep 1x
        load_n = target_n * load_factor

        model_name = discovery_cfg["model_name"]
        seed = discovery_cfg.get("seed", 42)
        batch_size = discovery_cfg.get("batch_size", 1)

        # Resolve chat-template handling once (see _build_mmlu_dataloader).
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        all_eap_data = []

        logger.info("\n[MMLU-Opt] Starting Memory-Optimized Loading")
        logger.info(
            f"[MMLU-Opt] Strategy: Load {load_n} samples -> Keep shortest {target_n} per subject"
        )

        for subject in subjects:
            try:
                # Load dataset (streaming or slice to save startup time)
                ds = load_dataset("cais/mmlu", subject, split="test")

                # Initial slice: take only what we need to inspect (5 * n)
                candidates_pool = ds.select(range(min(len(ds), load_n)))

                scored_candidates = []

                # Measure lengths
                for ex in candidates_pool:
                    q = ex.get("question", "")
                    c = list(ex.get("choices", []))
                    if not q or not c:
                        continue

                    # Reconstruct prompt to measure length
                    # using existing helper in this class. Wrap it with the same
                    # chat-template `apply` value used for the actual EAP data so
                    # the length probe matches what is fed to the model.
                    prompt_text = self._wrap_mmlu_prompt(
                        model, self._format_mmlu_prompt(q, c), apply=apply
                    )

                    try:
                        # Measure token length (BOS handled by `to_tokens`).
                        tokens = to_tokens(model, prompt_text, templated=apply)
                        length = tokens.size(1)
                        scored_candidates.append((length, ex))
                    except Exception:
                        continue

                # Sort by length (ascending) -> Shortest first
                scored_candidates.sort(key=lambda x: x[0])

                # Keep top N
                final_selection = [x[1] for x in scored_candidates[:target_n]]

                # Stats for verification
                if final_selection:
                    avg_len = sum(x[0] for x in scored_candidates[:target_n]) / len(final_selection)
                    logger.info(
                        f"  -> {subject}: Selected {len(final_selection)} shortest (Avg Len: {avg_len:.1f} tokens)"
                    )

                # Generate EAP Data
                for ex in final_selection:
                    data = self._generate_mmlu_eap_data_single(
                        ex, model, model_name, seed, subject, apply=apply
                    )
                    if data:
                        all_eap_data.append(data)

            except Exception as e:
                logger.error(f"  -> {subject} FAILED: {e}")

        if not all_eap_data:
            raise ValueError("No data generated. Check internet connection or subject names.")

        # 3. Build EAP Dataset (Standard Format)
        df = pd.DataFrame(all_eap_data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        dataset = EAPDiscoveryDataset(temp_path)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_EAP)
        # Templated prompts -> EAP must tokenize with prepend_bos=False (no double-BOS).
        loader.templated = apply
        return loader

    def build_smart_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """
        Smart Dataloader that actively selects efficient subjects.

        Logic:
        1. Randomly shuffles ALL available MMLU subjects.
        2. Iterates through them one by one.
        3. For each subject:
           - Loads (5 * n) samples.
           - Picks the shortest n.
           - Checks if Average Token Length <= 150.
        4. If efficient: Keeps subject & adds data to batch.
           If inefficient: Discards subject & tries the next one.
        5. Stops when 'num_subjects' count is reached.
        """
        import random
        import tempfile

        from ...backends.eap.eap_utils import collate_EAP

        # 1. Parse Config
        target_num_subjects = discovery_cfg.get("num_subjects", 3)
        samples_per_subject = discovery_cfg.get("samples_per_subject", 5)
        model_name = discovery_cfg["model_name"]
        seed = discovery_cfg.get("seed", 42)
        batch_size = discovery_cfg.get("batch_size", 1)

        # Resolve chat-template handling once (see _build_mmlu_dataloader).
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        # Length Threshold
        TOKEN_LIMIT = 150

        # Load factor for "shortest N" logic
        load_factor = 5
        load_n = samples_per_subject * load_factor

        # Get all subjects and shuffle them to ensure random selection
        all_pool = self._get_all_mmlu_subjects()
        random.shuffle(all_pool)

        accepted_subjects = []
        all_eap_data = []

        logger.info(
            f"\n[MMLU Smart-Select] Target: {target_num_subjects} subjects with Avg Len <= {TOKEN_LIMIT}"
        )
        logger.info(f"[MMLU Smart-Select] Pool: {len(all_pool)} subjects available.")

        # 2. Iterative Selection Loop
        for candidate_subject in all_pool:
            # Stop if we have enough subjects
            if len(accepted_subjects) >= target_num_subjects:
                break

            try:
                # Load candidate data
                ds = load_dataset("cais/mmlu", candidate_subject, split="test")
                pool_slice = ds.select(range(min(len(ds), load_n)))

                scored_candidates = []

                # Measure lengths
                for ex in pool_slice:
                    q = ex.get("question", "")
                    c = list(ex.get("choices", []))
                    if not q or not c:
                        continue

                    # Wrap with the same chat-template `apply` value used for
                    # the actual EAP data so the length check matches.
                    prompt_text = self._wrap_mmlu_prompt(
                        model, self._format_mmlu_prompt(q, c), apply=apply
                    )

                    try:
                        # Fast tokenization check (BOS handled by `to_tokens`).
                        tokens = to_tokens(model, prompt_text, templated=apply)
                        length = tokens.size(1)
                        scored_candidates.append((length, ex))
                    except Exception:
                        continue

                # Sort: Shortest first
                scored_candidates.sort(key=lambda x: x[0])

                # Select top N
                final_selection = [x[1] for x in scored_candidates[:samples_per_subject]]

                if not final_selection:
                    continue

                # --- CHECK CRITERIA ---
                avg_len = sum(x[0] for x in scored_candidates[:samples_per_subject]) / len(
                    final_selection
                )

                if avg_len > TOKEN_LIMIT:
                    logger.info(
                        f"  ❌ Discarded '{candidate_subject}' (Avg Len: {avg_len:.1f} > {TOKEN_LIMIT})"
                    )
                    continue

                # --- ACCEPT SUBJECT ---
                logger.info(
                    f"  ✅ Accepted '{candidate_subject}' (Avg Len: {avg_len:.1f} <= {TOKEN_LIMIT})"
                )
                accepted_subjects.append(candidate_subject)

                # Generate Data
                for ex in final_selection:
                    data = self._generate_mmlu_eap_data_single(
                        ex, model, model_name, seed, candidate_subject, apply=apply
                    )
                    if data:
                        all_eap_data.append(data)

            except Exception as e:
                logger.error(f"  ⚠️ Error processing '{candidate_subject}': {e}")
                continue

        # 3. Final Warning if quota not met
        if len(accepted_subjects) < target_num_subjects:
            logger.warning(
                f"\n⚠️ WARNING: Only found {len(accepted_subjects)} valid subjects out of requested {target_num_subjects}!"
            )
            logger.info("Proceeding with what we have...")

        if not all_eap_data:
            raise ValueError(
                "No valid data found! Try increasing the token limit or reducing sample count."
            )

        logger.info(f"\n[MMLU Smart-Select] Final Subject List: {accepted_subjects}")

        # 4. Build Dataset
        df = pd.DataFrame(all_eap_data)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        dataset = EAPDiscoveryDataset(temp_path)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_EAP)
        # Templated prompts -> EAP must tokenize with prepend_bos=False (no double-BOS).
        loader.templated = apply
        return loader

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the EAP/EAP-IG compatible metric for MMLU."""
        if metric_type == "kl":
            return partial(self._eap_mmlu_kl_divergence, loss=True, mean=True)
        # Default to logit difference specialized for multi-choice
        return partial(self._eap_logit_diff_mmlu, loss=True, mean=True)

    @staticmethod
    def _eap_logit_diff_mmlu(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        Calculates logit(correct) - mean(logits(incorrect)).
        Ensures a scalar output for gradient calculation in EAP/EAP-IG.
        """
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        # Target only the last token (the answer choice)
        last_logits = logits[idx, input_length - 1]

        # labels[:, 0] = correct, labels[:, 1:4] = incorrect
        correct_logits = t.gather(last_logits, -1, labels[:, 0:1].to(logits.device))
        incorrect_logits = t.gather(last_logits, -1, labels[:, 1:4].to(logits.device))

        avg_incorrect_logits = incorrect_logits.mean(dim=-1, keepdim=True)

        # Reduce to batch size
        results = correct_logits.squeeze(-1) - avg_incorrect_logits.squeeze(-1)

        if loss:
            results = -results
        if mean:
            results = results.mean()  # Scalar reduction for EAP-IG
        return results

    @staticmethod
    def _eap_mmlu_kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """
        Calculates KL Divergence specifically at the answer position.
        """
        import torch.nn.functional as F

        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)

        # Focus on answer distribution shift
        last_logits = logits[idx, input_length - 1]
        last_clean_logits = clean_logits[idx, input_length - 1]

        results = F.kl_div(
            F.log_softmax(last_logits, dim=-1),
            F.softmax(last_clean_logits, dim=-1),
            reduction="batchmean" if mean else "none",
        )
        return results

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata for MMLU artifacts.

        Args:
            discovery_cfg: Discovery configuration

        Returns:
            Dictionary containing MMLU-specific metadata
        """
        return {
            "task": "mmlu",
            "subjects": discovery_cfg.get("subjects")
            or discovery_cfg.get("subject")
            or self._get_all_mmlu_subjects(),
            "samples_per_subject": discovery_cfg.get("samples_per_subject", 20),
            "filter_longest_sequences": discovery_cfg.get("filter_longest_sequences", True),
            "corruption_mode": "question_replacement",  # MMLU uses question replacement corruption
            "corruption_text": "Which is the most possible answer?",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "seeds": discovery_cfg.get("seeds", [42]),
            "model_name": discovery_cfg["model_name"],
            # Resolved chat-template mode — later stages read this back unchanged.
            "chat_template_mode": discovery_cfg.get("chat_template_mode", self.chat_template_mode),
        }

    def _build_ibcircuit_dataloader(self, discovery_cfg: Dict[str, Any], device: str, model):
        """
        Build SingleBatchDataLoader for IBCircuit using MMLU data.

        Strategy: load (5 * samples_per_subject) candidates per subject, keep the
        shortest N to minimise OOM risk from variable-length MMLU prompts.

        Batch format (IBCircuit contract):
            tokens           [batch, max_seq_len]  left-padded input token IDs
            labels           [batch]               correct answer letter token ID
            answer_positions [batch]               last real token index per sequence
        """
        import torch

        subjects = discovery_cfg.get(
            "subjects", discovery_cfg.get("subject", self._get_all_mmlu_subjects())
        )
        if isinstance(subjects, str):
            subjects = [subjects]
        samples_per_subject = discovery_cfg.get("samples_per_subject", 20)
        load_factor = 5

        # Resolve chat-template handling once (see _build_mmlu_dataloader).
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        pad_token_id = getattr(model.tokenizer, "pad_token_id", None) or 0

        logger.info(
            f"Loading MMLU data for IBCircuit ({len(subjects)} subjects, "
            f"{samples_per_subject} samples each)..."
        )

        all_examples = []
        for subject in subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                pool = ds.select(range(min(len(ds), samples_per_subject * load_factor)))

                scored = []
                for ex in pool:
                    q = ex.get("question", "")
                    c = list(ex.get("choices", []))
                    if not q or not c:
                        continue
                    try:
                        wrapped = self._wrap_mmlu_prompt(
                            model, self._format_mmlu_prompt(q, c), apply=apply
                        )
                        toks = to_tokens(model, wrapped, templated=apply)
                        scored.append((toks.size(1), ex))
                    except Exception:
                        continue

                scored.sort(key=lambda x: x[0])
                kept = [ex for _, ex in scored[:samples_per_subject]]
                all_examples.extend(kept)
                logger.debug(f"  {subject}: kept {len(kept)} shortest examples")
            except Exception as e:
                logger.error(f"  Warning: failed to load '{subject}': {e}")

        if not all_examples:
            raise ValueError("No valid MMLU examples loaded for IBCircuit")

        # Honor the caller's num_examples cap (default 32) so the IBCircuit
        # batch doesn't grow to subjects × samples_per_subject (~1100), which
        # would push a single [N, max_len, vocab] logits tensor into the tens
        # of GB. data_params.num_examples is the public knob for this; we
        # also accept a top-level num_examples for back-compat.
        data_params = discovery_cfg.get("data_params", {}) or {}
        max_examples = data_params.get("num_examples") or discovery_cfg.get("num_examples") or 32
        if isinstance(max_examples, int) and max_examples > 0:
            all_examples = all_examples[:max_examples]
            logger.info(
                f"  Capped IBCircuit batch at {len(all_examples)} examples "
                f"(num_examples={max_examples})"
            )

        # Tokenize: full_text = prompt + " " + correct_letter
        # Split into input tokens (all but last) and label (last token).
        # Using the full-text tokenisation ensures the label token ID is
        # exactly the token the model sees in context — consistent with EAP.
        token_seqs = []
        label_ids = []

        for ex in all_examples:
            question = ex.get("question", "")
            choices = list(ex.get("choices", []))
            answer = ex.get("answer")
            if not question or not choices or answer is None:
                continue

            if not (0 <= answer < len(choices)):
                continue  # skip malformed example
            # Wrap the prompt (the "Answer:" tail stays the assistant-turn
            # prefix); the correct letter is appended after it so the label
            # token is exactly the token the model sees in context.
            prompt = self._wrap_mmlu_prompt(
                model, self._format_mmlu_prompt(question, choices), apply=apply
            )
            correct_letter = "ABCD"[answer]
            full_text = f"{prompt} {correct_letter}"

            try:
                full_tokens = to_tokens(model, full_text, templated=apply)[0]  # [full_len]
                token_seqs.append(full_tokens[:-1])  # prompt tokens (input)
                label_ids.append(full_tokens[-1].item())  # answer letter token ID
            except Exception:
                continue

        if not token_seqs:
            raise ValueError("No MMLU examples could be tokenized for IBCircuit")

        # Left-pad to uniform length
        lengths = torch.tensor([s.size(0) for s in token_seqs], dtype=torch.long)
        max_len = lengths.max().item()

        padded = torch.full((len(token_seqs), max_len), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(token_seqs):
            padded[i, max_len - seq.size(0) :] = seq  # left-pad: align all sequences at the right

        tokens_tensor = padded.to(device)
        labels_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
        answer_positions = torch.full(
            (len(token_seqs),), max_len - 1, dtype=torch.long, device=device
        )

        logger.info(
            f"IBCircuit MMLU batch ready: {len(token_seqs)} examples, "
            f"max_len={max_len}, "
            f"answer_pos range [{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        batch = {
            "tokens": tokens_tensor,
            "labels": labels_tensor,
            "answer_positions": answer_positions,
        }

        logger.debug(
            f"[DEBUG PADDING] mmlu IBCircuit  within-batch=left-padded  max_len={max_len}  all answer_pos={max_len - 1}"
        )
        logger.debug(
            f"[DEBUG PADDING]   tokens[0] head: {model.to_str_tokens(tokens_tensor[0, :5])}  (left-pad visible here)"
        )
        logger.debug(
            f"[DEBUG PADDING]   tokens[0] tail: {model.to_str_tokens(tokens_tensor[0, -5:])}"
        )

        # ── Validate the assembled batch ──
        self._validate_mmlu_data(algorithm="ibcircuit", data=batch, model=model)

        class SingleBatchDataLoader:
            """Yields a single fixed batch repeatedly (IBCircuit trains on one batch)."""

            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        # Verify answer positions and labels for first 2 examples
        for i in range(min(2, len(token_seqs))):
            pos = answer_positions[i].item()
            logger.debug(
                f"[DEBUG MMLU-IB] Example {i}: seq_len={lengths[i].item()}, answer_pos={pos}"
            )
            logger.debug(f"  Token at answer_pos: {model.to_string(tokens_tensor[i, pos:pos+1])!r}")
            logger.debug(
                f"  Token at answer_pos-1: {model.to_string(tokens_tensor[i, pos-1:pos])!r}"
            )
            logger.debug(
                f"  Label token: {model.to_string(labels_tensor[i:i+1])!r} (id={labels_tensor[i].item()})"
            )

        if len(token_seqs) < 20:
            logger.debug(
                f"[WARNING] IBCircuit batch has only {len(token_seqs)} examples. "
                f"Minimum ~20-50 recommended for reliable std_mean statistics."
            )

        return SingleBatchDataLoader(batch)

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate MMLU finetuning data from HuggingFace using tokenizer only.

        Two source modes selected via ``discovery_cfg["split"]``:

          * ``"test"`` (default): Loops over per-subject configs, draws up to
            ``samples_per_subject`` from each subject's test split. Same as
            discovery uses; convenient when finetuning data may overlap
            discovery prompts (e.g., Q4 reusing Q2 circuits).

          * ``"auxiliary_train"``: Loads the ``auxiliary_train`` config (one
            shared 99k-example pool from ARC/RACE/MathQA — the original MMLU
            paper's training corpus). No overlap with the test split, so this
            is the leakage-free option when eval is also MMLU. Schema quirk:
            each row is wrapped in a ``"train"`` sub-dict; we unwrap it.
        """
        cfg = discovery_cfg or {}
        split = cfg.get("split", "test")

        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model); a discovery_cfg override wins.
        # When this resolves True each prompt is wrapped in the model's chat
        # template with the same "Answer:" assistant prefix discovery uses, so
        # circuit-tuning trains on the discovery prompt distribution. When False
        # (base model / "off") the prompt text is byte-identical to the legacy
        # raw-text behavior.
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)

        def _wrap_query(raw_query: str) -> str:
            """Wrap a raw "Q: ... Answer:" prompt, keeping "Answer:" as the
            assistant prefix (no-op when apply is False)."""
            user_text = raw_query[: -len(_ANSWER_TAIL)]
            return wrap_prompt_with_tokenizer(tokenizer, user_text, _ANSWER_TAIL, apply=apply)

        all_pairs: List[Tuple[str, str]] = []  # (query_string, clean_text)

        if split == "auxiliary_train":
            try:
                ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
            except Exception as exc:
                raise ValueError(f"Failed to load cais/mmlu auxiliary_train: {exc}") from exc

            for ex in ds:
                # auxiliary_train wraps each example: {'train': {question, choices, answer, subject}}
                inner = ex.get("train", ex)
                question = inner.get("question", "")
                choices = list(inner.get("choices", []))
                answer = inner.get("answer")
                if not question or not choices or answer is None:
                    continue
                if not (0 <= answer < len(choices)):
                    continue
                query = _wrap_query(self._format_mmlu_prompt(question, choices))
                letter = "ABCD"[answer]
                full_text = f"{query} {letter}"
                all_pairs.append((query, full_text))
        else:
            subjects = cfg.get("subjects", cfg.get("subject")) or self._get_all_mmlu_subjects()
            if isinstance(subjects, str):
                subjects = [subjects]
            samples_per_subject = cfg.get("samples_per_subject", 20)

            for subject in subjects:
                try:
                    ds = load_dataset("cais/mmlu", subject, split=split)
                    for ex in ds.select(range(min(len(ds), samples_per_subject))):
                        question = ex.get("question", "")
                        choices = list(ex.get("choices", []))
                        answer = ex.get("answer")

                        if not question or not choices or answer is None:
                            continue
                        if not (0 <= answer < len(choices)):
                            continue

                        query = _wrap_query(self._format_mmlu_prompt(question, choices))
                        letter = "ABCD"[answer]
                        # Space before the letter matches how BPE tokenizes mid-sentence.
                        full_text = f"{query} {letter}"
                        all_pairs.append((query, full_text))
                except Exception:
                    continue

        if not all_pairs:
            raise ValueError(
                f"No MMLU data could be loaded for finetuning (split={split!r}). "
                "Check internet connection and subjects in discovery_cfg."
            )

        rng = _random.Random(seed)
        rng.shuffle(all_pairs)
        all_pairs = all_pairs[:n_examples]

        query_strings = [q for q, _ in all_pairs]
        clean_texts = [ft for _, ft in all_pairs]
        return clean_texts, query_strings
