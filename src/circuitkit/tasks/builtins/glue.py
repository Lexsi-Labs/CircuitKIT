"""
GLUE (General Language Understanding Evaluation) Task Specification

Implements the TaskSpec interface for various GLUE tasks:
- MRPC (Microsoft Research Paraphrase Corpus)
- QQP (Quora Question Pairs)
- SST-2 (Stanford Sentiment Treebank)
- RTE (Recognizing Textual Entailment)
- CoLA (Corpus of Linguistic Acceptability)

Uses GenericTaskSpec for flexible dataset loading and corruption strategy application.
"""

import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch as t
from datasets import load_dataset
from torch.utils.data import DataLoader

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
    wrap_prompt,
    to_tokens,
    wrap_prompt_with_tokenizer,
)
from ..generic import GenericDataLoader, GenericTaskSpec

logger = get_logger("task.glue")

# GLUE task metadata
GLUE_TASKS = {
    "mrpc": {
        "description": "Microsoft Research Paraphrase Corpus (binary classification: paraphrase or not)",
        "num_classes": 2,
        "metric": "accuracy",
        "fields": ["sentence1", "sentence2", "label"],
    },
    "qqp": {
        "description": "Quora Question Pairs (binary classification: duplicate or not)",
        "num_classes": 2,
        "metric": "accuracy",
        "fields": ["question1", "question2", "label"],
    },
    "sst2": {
        "description": "Stanford Sentiment Treebank (binary classification: positive or negative)",
        "num_classes": 2,
        "metric": "accuracy",
        "fields": ["sentence", "label"],
    },
    "rte": {
        "description": "Recognizing Textual Entailment (binary classification: entailment or not)",
        "num_classes": 2,
        "metric": "accuracy",
        "fields": ["sentence1", "sentence2", "label"],
    },
    "cola": {
        "description": "Corpus of Linguistic Acceptability (binary classification: acceptable or not)",
        "num_classes": 2,
        "metric": "accuracy",
        "fields": ["sentence", "label"],
    },
}


class GLUETaskSpec:
    """GLUE task specification supporting multiple subtasks via GenericTaskSpec.

    This task extends GenericTaskSpec to handle GLUE dataset loading and
    standard classification metrics. Supports EAP, EAP-IG, and IBCircuit algorithms.
    """

    pair_padding_side = "right"
    # Downstream-behavior task: wrap discovery prompts in the model's chat
    # template iff the model is instruction-tuned ("auto"). Frozen into metadata.
    chat_template_mode: str = "auto"

    def __init__(self, task_name: str = "sst2"):
        """Initialize GLUE task.

        Args:
            task_name: GLUE task name (mrpc, qqp, sst2, rte, cola)

        Raises:
            ValueError: If task_name is not a valid GLUE task
        """
        task_name_lower = task_name.lower()
        if task_name_lower not in GLUE_TASKS:
            raise ValueError(
                f"Invalid GLUE task: {task_name!r}. "
                f"Pass task_name as one of: {list(GLUE_TASKS.keys())}."
            )

        self.default_subtask = task_name_lower  # fallback when cfg omits glue_task
        self.task_name = task_name_lower  # active subtask, resolved at runtime
        self.name = "glue"
        self.task_info = GLUE_TASKS[task_name_lower]
        self._generic_spec: Optional[GenericTaskSpec] = None

    def validate_discovery_config(self, discovery_cfg: Dict[str, Any]) -> None:
        """Validate GLUE-specific discovery configuration.

        Args:
            discovery_cfg: Discovery configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        glue_task = discovery_cfg.get("glue_task", self.default_subtask)
        if glue_task not in GLUE_TASKS:
            raise ValueError(
                f"Unknown GLUE subtask '{glue_task}'. "
                f"Set discovery config key 'glue_task' to one of: "
                f"{list(GLUE_TASKS.keys())}."
            )

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if algorithm not in (EAP_FAMILY | IB_FAMILY | CDT_FAMILY):
            raise ValueError(
                unsupported_algorithm_message(
                    "GLUE task", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

        level = discovery_cfg.get("level")
        if level not in ["node", "neuron"]:
            raise ValueError(
                f"GLUE discovery config has invalid 'level': {level!r}. "
                f"Set discovery config key 'level' to 'node' or 'neuron'."
            )

        if algorithm == "ibcircuit":
            scope = discovery_cfg.get("scope", "heads")
            if scope not in ["heads", "mlp", "both"]:
                raise ValueError(
                    f"GLUE ibcircuit has invalid 'scope': {scope!r}. "
                    f"Set discovery config key 'scope' to one of: heads, mlp, both."
                )

        batch_size = discovery_cfg.get("batch_size", 4)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"GLUE has invalid 'batch_size': {batch_size!r}. "
                f"Set discovery config key 'batch_size' to a positive integer (e.g. 4)."
            )

        # Validate samples_per_split if present
        samples_per_split = discovery_cfg.get("samples_per_split")
        if samples_per_split is not None:
            if not isinstance(samples_per_split, int) or samples_per_split <= 0:
                raise ValueError(
                    f"GLUE has invalid 'samples_per_split': {samples_per_split!r}. "
                    f"Set discovery config key 'samples_per_split' to a positive integer."
                )

    def build_dataloader(
        self,
        model,
        discovery_cfg: Dict[str, Any],
        device: str,
    ) -> DataLoader:
        """Build DataLoader for GLUE task.

        Args:
            model: HookedTransformer model
            discovery_cfg: Discovery configuration
            device: Target device

        Returns:
            DataLoader configured for GLUE task

        Raises:
            ValueError: If model is None
        """
        if model is None:
            raise ValueError("GLUE task requires model for tokenization. No default model.")

        # Resolve active subtask from config — mirrors WMDP's configs pattern.
        glue_task = discovery_cfg.get("glue_task", self.default_subtask)
        if glue_task not in GLUE_TASKS:
            raise ValueError(
                f"Unknown GLUE subtask '{glue_task}'. "
                f"Set discovery config key 'glue_task' to one of: "
                f"{list(GLUE_TASKS.keys())}."
            )
        self.task_name = glue_task
        self.task_info = GLUE_TASKS[glue_task]

        algorithm = discovery_cfg.get("algorithm", "").lower()

        if is_eap_family(algorithm) or algorithm == "cdt":
            return self._build_eap_dataloader(discovery_cfg, device, model)
        elif algorithm == "ibcircuit":
            return self._build_ibcircuit_dataloader(discovery_cfg, device, model)
        else:
            raise ValueError(
                unsupported_algorithm_message(
                    "GLUE task", algorithm, EAP_FAMILY | IB_FAMILY | CDT_FAMILY
                )
            )

    def _get_or_load_glue_data(self, discovery_cfg: Dict[str, Any], model) -> List[Dict[str, Any]]:
        """Load GLUE dataset and prepare examples.

        Args:
            discovery_cfg: Discovery configuration
            model: HookedTransformer model

        Returns:
            List of prepared examples with 'prompt' and 'answer' fields
        """
        split = discovery_cfg.get("split", "validation")
        samples_per_split = discovery_cfg.get("samples_per_split")
        seed = discovery_cfg.get("seed", 42)

        # Load dataset
        logger.info(f"Loading GLUE {self.task_name} dataset split={split}")
        dataset = load_dataset("glue", self.task_name, split=split)

        # Sample if requested
        if samples_per_split and len(dataset) > samples_per_split:
            dataset = dataset.shuffle(seed=seed).select(range(samples_per_split))

        # Prepare examples with correct prompt/answer format
        examples = []
        self.task_info

        for item in dataset:
            example = self._format_glue_example(item)
            if example:
                examples.append(example)

        logger.info(f"Loaded {len(examples)} examples from GLUE {self.task_name}")
        return examples

    def _format_glue_example(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format a GLUE example into prompt + answer format.

        Args:
            item: Raw dataset item

        Returns:
            Formatted example or None if invalid
        """
        task_name = self.task_name

        try:
            if task_name in ["sst2"]:
                # Single sentence classification
                prompt = item.get("sentence", "")
                label = item.get("label")
                # Convert label to text (0=negative, 1=positive for sst2)
                label_text = "positive" if label == 1 else "negative"

                return {
                    "prompt": f"Sentiment: {prompt}",
                    "answer": label_text,
                    "label": label,
                }

            elif task_name in ["mrpc", "qqp", "rte"]:
                # Sentence pair classification
                sentence1 = item.get("sentence1", item.get("question1", ""))
                sentence2 = item.get("sentence2", item.get("question2", ""))
                label = item.get("label")

                if task_name == "mrpc":
                    label_text = "paraphrase" if label == 1 else "not_paraphrase"
                    question = "Are these sentences paraphrases?"
                elif task_name == "qqp":
                    label_text = "duplicate" if label == 1 else "not_duplicate"
                    question = "Are these questions duplicates?"
                else:  # rte
                    label_text = "entailment" if label == 1 else "not_entailment"
                    question = "Does the first sentence entail the second?"

                prompt = f"{question}\n1: {sentence1}\n2: {sentence2}"

                return {
                    "prompt": prompt,
                    "answer": label_text,
                    "label": label,
                }

            elif task_name == "cola":
                # Linguistic acceptability
                sentence = item.get("sentence", "")
                label = item.get("label")
                label_text = "acceptable" if label == 1 else "not_acceptable"

                return {
                    "prompt": f"Is this sentence acceptable? {sentence}",
                    "answer": label_text,
                    "label": label,
                }

            else:
                logger.warning(f"Unknown GLUE task: {task_name}")
                return None

        except Exception as e:
            logger.warning(f"Error formatting GLUE example: {e}")
            return None

    def _build_eap_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """Build EAP-compatible dataloader for GLUE.

        Args:
            discovery_cfg: Discovery configuration
            device: Target device
            model: HookedTransformer model

        Returns:
            DataLoader with EAP-compatible format
        """
        from ...corruption.pipeline import CorruptionPipeline

        # Load GLUE data
        examples = self._get_or_load_glue_data(discovery_cfg, model)

        if not examples:
            raise ValueError(f"No examples loaded for GLUE {self.task_name}")

        # Get corruption strategy
        corruption_strategy = discovery_cfg.get("corruption_strategy")

        # If no strategy provided, create a simple one for GLUE
        if corruption_strategy is None:
            from ...corruption.paraphrase import ParaphraseCorruption

            corruption_strategy = ParaphraseCorruption()

        # Resolve chat-template wrapping once for this discovery run. The same
        # ``apply`` boolean is threaded into prompt finalization so clean and
        # corrupted prompts are wrapped identically (token alignment preserved).
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        # Generate tokenizer function
        tokenize_fn = partial(
            self._tokenize_glue_example,
            model=model,
            device=device,
            apply=apply,
        )

        # Apply corruptions and tokenize
        processed_examples = []
        for example in examples:
            # Corrupt
            if isinstance(corruption_strategy, CorruptionPipeline):
                corrupted = corruption_strategy.corrupt([example])[0]
            else:
                rng = __import__("random").Random(42)
                corrupted = corruption_strategy.corrupt(example, rng=rng)

            # Tokenize
            tokens = tokenize_fn(corrupted)
            if tokens:
                processed_examples.append(tokens)

        # Create dataset and dataloader
        dataset = GenericDataLoader(processed_examples)
        batch_size = discovery_cfg.get("batch_size", 4)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_glue_eap,
        )
        # Flag templated prompts so the EAP backend skips the extra BOS.
        dataloader.templated = apply

        return dataloader

    def _build_ibcircuit_dataloader(
        self, discovery_cfg: Dict[str, Any], device: str, model
    ) -> DataLoader:
        """Build IBCircuit-compatible dataloader for GLUE.

        Returns a SingleBatchDataLoader with {tokens, labels, answer_positions}
        dict format. Uses clean prompts only (no corruption needed for IBCircuit).
        """
        import torch as t

        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template(mode, model)

        examples = self._get_or_load_glue_data(discovery_cfg, model)
        if not examples:
            raise ValueError(f"No examples loaded for GLUE {self.task_name}")

        token_lists = []
        correct_idxs = []
        for ex in examples:
            prompt = ex.get("prompt", "")
            answer = ex.get("answer", "")
            if not prompt or not answer:
                continue
            try:
                wrapped = wrap_prompt(model, prompt, apply=apply)
                toks = to_tokens(model, wrapped, templated=apply).squeeze(0).cpu()
                answer_tok = model.to_tokens(answer, prepend_bos=False)[0, -1].item()
                token_lists.append(toks)
                correct_idxs.append(answer_tok)
            except Exception as e:
                logger.warning(f"Skipping GLUE example in IBCircuit build: {e}")
                continue

        if not token_lists:
            raise ValueError(f"No valid examples for GLUE IBCircuit dataloader ({self.task_name})")

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
            f"[DEBUG PADDING] glue IBCircuit  within-batch=right-padded  "
            f"max_len={max_len}  answer_pos range=[{answer_positions.min().item()}, {answer_positions.max().item()}]"
        )

        class SingleBatchDataLoader:
            def __init__(self, batch):
                self.batch = batch

            def __iter__(self):
                yield self.batch

        return SingleBatchDataLoader(batch)

    def _tokenize_glue_example(
        self,
        example: Dict[str, Any],
        model,
        device: str,
        apply: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Tokenize a GLUE example for EAP.

        Args:
            example: Formatted example with 'prompt' and 'answer' fields
            model: HookedTransformer model
            device: Target device
            apply: resolved chat-template boolean. When True the clean and
                corrupted prompts are wrapped in the model's chat template at
                this prompt-finalization point; when False the prompt strings
                are byte-identical to the legacy raw-text behavior.

        Returns:
            Tokenized example in EAP format or None if invalid
        """
        prompt = example.get("prompt", "")
        answer = example.get("answer", "")

        if not prompt or not answer:
            return None

        try:
            # Tokenize prompt and answer
            answer_tokens = model.to_tokens(answer, prepend_bos=False)

            # Get token IDs
            correct_idx = answer_tokens[0, -1].item()

            # Use first token as baseline incorrect
            incorrect_idx = model.to_tokens("not", prepend_bos=False)[0, 0].item()

            # Wrap clean and corrupted prompts at finalization time. GLUE has no
            # answer-eliciting tail (the answer is a separate token), so the
            # assistant prefix is empty; clean and corrupted get the IDENTICAL
            # apply, keeping the EAP token alignment intact.
            corrupted_prompt = example.get("corrupted", prompt)
            clean_wrapped = wrap_prompt(model, prompt, apply=apply)
            corrupted_wrapped = wrap_prompt(model, corrupted_prompt, apply=apply)

            return {
                "clean": clean_wrapped,
                "corrupted": corrupted_wrapped,
                "correct_idx": correct_idx,
                "incorrect_idx": incorrect_idx,
                "metadata": {
                    "task": self.task_name,
                    "answer": answer,
                    "label": example.get("label"),
                },
            }
        except Exception as e:
            logger.warning(f"Error tokenizing example: {e}")
            return None

    def build_finetuning_dataset(
        self,
        tokenizer,
        model_name: str,
        n_examples: int,
        discovery_cfg: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Return (clean_texts, query_strings) GLUE pairs for causal LM finetuning.

        Draws fresh from the HuggingFace ``glue`` dataset and reuses
        :meth:`_format_glue_example` so the prompt body matches discovery. When
        the finetuning model is instruction-tuned (and the task's resolved
        ``chat_template_mode`` is not ``"off"``) each prompt is wrapped in the
        model's chat template; GLUE has no answer-eliciting tail (the answer is
        a separate token), so the assistant prefix is empty — exactly as
        discovery's ``_tokenize_glue_example`` wraps it. For base models /
        ``"off"`` the prompt text is byte-identical to the legacy raw-text
        behavior.
        """
        cfg = discovery_cfg or {}

        # Resolve the active subtask the same way build_dataloader does.
        glue_task = cfg.get("glue_task", self.default_subtask)
        if glue_task not in GLUE_TASKS:
            raise ValueError(
                f"Unknown GLUE subtask '{glue_task}'. "
                f"Set discovery config key 'glue_task' to one of: "
                f"{list(GLUE_TASKS.keys())}."
            )
        self.task_name = glue_task
        self.task_info = GLUE_TASKS[glue_task]

        # Resolve the chat-template decision from the tokenizer (a tokenizer
        # carrying a chat_template ⇒ chat model); a discovery_cfg override wins.
        mode = cfg.get("chat_template_mode", self.chat_template_mode)
        apply = resolve_chat_template_from_tokenizer(mode, tokenizer)

        split = cfg.get("split", "validation")
        dataset = load_dataset("glue", self.task_name, split=split)

        clean_texts: List[str] = []
        query_strings: List[str] = []
        for item in dataset:
            example = self._format_glue_example(item)
            if not example:
                continue
            prompt = example.get("prompt", "")
            answer = example.get("answer", "")
            if not prompt or not answer:
                continue
            # Empty assistant prefix — matches discovery's wrap_prompt call in
            # _tokenize_glue_example (no-op when apply is False).
            query = wrap_prompt_with_tokenizer(tokenizer, prompt, apply=apply)
            # Leading space before the answer word matches mid-sentence BPE.
            clean_texts.append(f"{query} {answer}")
            query_strings.append(query)

        if not clean_texts:
            raise ValueError(
                f"No GLUE data could be loaded for finetuning (task={self.task_name!r}, "
                f"split={split!r})."
            )

        paired = list(zip(clean_texts, query_strings))
        rng = random.Random(seed)
        rng.shuffle(paired)
        paired = paired[:n_examples]
        clean_texts = [c for c, _ in paired]
        query_strings = [q for _, q in paired]
        return clean_texts, query_strings

    def metric_fn(self, metric_type: str = "logit_diff") -> Callable:
        """Return the EAP/EAP-IG-compatible metric for GLUE.

        ``discover_circuit`` calls this for every EAP-family algorithm.
        GLUE's EAP labels are ``[correct_idx, incorrect_idx]`` (see
        :meth:`_tokenize_glue_example`), so the default metric is the logit
        difference between the correct and incorrect answer tokens at the last
        prompt position — the same construction BoolQ / MMLU use.
        """
        if metric_type == "kl":
            return partial(self._kl_divergence, loss=True, mean=True)
        return partial(self._logit_diff, loss=True, mean=True)

    @staticmethod
    def _logit_diff(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """logit(correct) - logit(incorrect) at the answer position."""
        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)
        last_logits = logits[idx, input_length - 1]
        correct = t.gather(last_logits, -1, labels[:, 0:1].to(logits.device))
        incorrect = t.gather(last_logits, -1, labels[:, 1:2].to(logits.device))
        results = correct.squeeze(-1) - incorrect.squeeze(-1)
        if loss:
            results = -results
        if mean:
            results = results.mean()
        return results

    @staticmethod
    def _kl_divergence(logits, clean_logits, input_length, labels, mean=True, loss=False):
        """KL divergence between patched and clean output distributions."""
        import torch.nn.functional as F

        batch_size = logits.size(0)
        idx = t.arange(batch_size, device=logits.device)
        last_logits = logits[idx, input_length - 1]
        last_clean = clean_logits[idx, input_length - 1]
        return F.kl_div(
            F.log_softmax(last_logits, dim=-1),
            F.softmax(last_clean, dim=-1),
            reduction="batchmean" if mean else "none",
        )

    def artifact_metadata(self, discovery_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for GLUE artifacts.

        The resolved ``chat_template_mode`` (honoring a ``discovery_cfg``
        override) is frozen here so downstream stages resolve an identical
        chat-template policy against the same model.
        """
        mode = discovery_cfg.get("chat_template_mode", self.chat_template_mode)
        return {
            "task": "glue",
            "glue_task": discovery_cfg.get("glue_task", self.task_name),
            "data_source": "glue",
            "algorithm": discovery_cfg.get("algorithm"),
            "level": discovery_cfg.get("level"),
            "chat_template_mode": mode,
        }


def _collate_glue_eap(xs):
    """Collate GenericDataLoader dict items into the EAP 3-tuple batch format.

    ``GenericDataLoader.__getitem__`` yields per-example dicts
    (``{"clean", "corrupted", "labels", ...}``), not 3-tuples — so the
    tuple-based ``collate_EAP`` cannot be used directly. This converts each
    dict to the ``(clean_texts, corrupted_texts, labels)`` form the EAP
    backend expects, mirroring ``GenericTaskSpec``'s own dict-aware collate.
    """
    import torch as _torch

    clean = [item["clean"] for item in xs]
    corrupted = [item["corrupted"] for item in xs]
    labels = _torch.tensor([item["labels"] for item in xs])
    return clean, corrupted, labels


def create_glue_task(task_name: str = "sst2") -> GLUETaskSpec:
    """Factory function to create a GLUE task.

    Args:
        task_name: GLUE task name (mrpc, qqp, sst2, rte, cola)

    Returns:
        GLUETaskSpec instance
    """
    return GLUETaskSpec(task_name)
