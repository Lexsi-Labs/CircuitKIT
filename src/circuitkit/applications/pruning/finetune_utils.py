"""
finetune_utils.py — LoRA-based post-pruning fine-tuning.

Two fine-tuning modes
---------------------
1. **Alpaca** (``finetune_on_alpaca``): General-purpose instruction-following
   recovery on a small Alpaca subset.  Good default when the pruned model must
   retain broad capabilities.

2. **Task data** (``finetune_on_task_data``): Fine-tune directly on the same
   circuit-discovery task data (IOI, MMLU, SVA, …).  Each example is a prompt
   with a known correct next-token; the model is trained to predict that token.
   This targets recovery on the specific task used for pruning decisions.

Design choices
--------------
* LoRA (rank-8 by default) is applied to the standard attention projections
  (q, k, v, o) and MLP projections (gate, up, down).  This works on pruned
  models because LoRA wraps existing linear layers regardless of their shape.
* After training the LoRA adapters are merged back into the base weights so
  the returned model requires no extra inference overhead.
* The Prompter / tokenization logic is adapted from LLM-Pruner's
  post_training.py to stay compatible with the Alpaca instruction format.

LLM-Pruner path note
--------------------
This module imports LLMPruner.peft.  The LLM-Pruner root must be on sys.path
before calling ``finetune_on_alpaca``.  The example scripts in this package
add it automatically.
"""

from __future__ import annotations

import tempfile
from typing import List, Optional

import transformers
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_prompter():
    """Return an Alpaca-format Prompter from LLM-Pruner's utilities."""
    try:
        from LLMPruner.utils.prompter import Prompter

        return Prompter("alpaca")
    except ImportError:
        # Minimal inline fallback so the function still works if LLM-Pruner
        # is not available (produces simpler prompts).
        class _FallbackPrompter:
            def generate_prompt(self, instruction, inp=None, output=None):
                prompt = f"### Instruction:\n{instruction}\n"
                if inp:
                    prompt += f"### Input:\n{inp}\n"
                prompt += "### Response:\n"
                if output:
                    prompt += output
                return prompt

        return _FallbackPrompter()


def _get_peft_classes():
    """Import LoraConfig, get_peft_model from LLMPruner.peft or peft."""
    try:
        from LLMPruner.peft import LoraConfig, get_peft_model

        return LoraConfig, get_peft_model
    except ImportError:
        from peft import LoraConfig, get_peft_model

        return LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def finetune_on_alpaca(
    model,
    tokenizer,
    n_examples: int = 1000,
    n_epochs: int = 2,
    learning_rate: float = 3e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    batch_size: int = 16,
    micro_batch_size: int = 4,
    cutoff_len: int = 256,
    device: str = "cuda",
    output_dir: Optional[str] = None,
    data_path: str = "yahma/alpaca-cleaned",
) -> object:
    """
    Fine-tune a (pruned) HuggingFace causal-LM on a small Alpaca subset using LoRA.

    After training the LoRA adapters are merged back into the base weights and
    the plain HuggingFace model is returned.

    Parameters
    ----------
    model               : HuggingFace CausalLM (float16 / bfloat16 OK).
    tokenizer           : Matching HuggingFace tokenizer.
    n_examples          : Number of Alpaca examples to train on (default 1 000).
    n_epochs            : Training epochs.
    learning_rate       : AdamW learning rate.
    lora_r              : LoRA rank.
    lora_alpha          : LoRA scaling factor.
    lora_dropout        : Dropout probability inside LoRA layers.
    lora_target_modules : Linear layers to wrap with LoRA.  Defaults to the
                          standard LLaMA / Qwen projection names.
    batch_size          : Total batch size (gradient-accumulation-aware).
    micro_batch_size    : Per-device micro batch size.
    cutoff_len          : Max sequence length for tokenization.
    device              : Training device.
    output_dir          : Where to write the final merged checkpoint.  If None
                          a temporary directory is used and discarded.
    data_path           : HuggingFace dataset name for training data.

    Returns
    -------
    The fine-tuned model (LoRA weights merged) ready for inference.
    """
    LoraConfig, get_peft_model = _get_peft_classes()
    prompter = _get_prompter()

    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)

    # ------------------------------------------------------------------
    # Tokenizer setup
    # ------------------------------------------------------------------
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # Dataset loading and preprocessing
    # ------------------------------------------------------------------
    logger.info(f"[finetune] Loading {n_examples} examples from '{data_path}' …")
    raw_data = load_dataset(data_path, split="train")
    # Shuffle and select subset
    raw_data = raw_data.shuffle(seed=42).select(range(min(n_examples, len(raw_data))))

    def _tokenize(prompt: str, add_eos: bool = True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            add_eos
            and result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def _process(data_point):
        # Build full prompt (instruction + response)
        full_prompt = prompter.generate_prompt(
            data_point.get("instruction", ""),
            data_point.get("input", None),
            data_point.get("output", ""),
        )
        tokenized = _tokenize(full_prompt)

        # Mask out the instruction part (train only on the response)
        user_prompt = prompter.generate_prompt(
            data_point.get("instruction", ""),
            data_point.get("input", None),
        )
        tokenized_user = _tokenize(user_prompt, add_eos=False)
        user_len = len(tokenized_user["input_ids"])
        tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    logger.info("[finetune] Tokenizing dataset …")
    train_data = raw_data.map(_process, remove_columns=raw_data.column_names)
    train_data.set_format(type="torch")

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    logger.info(f"[finetune] Applying LoRA (r={lora_r}, α={lora_alpha}) …")
    # Filter target modules to only those that exist in the model
    existing_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    active_targets = [m for m in lora_target_modules if m in existing_names]
    if not active_targets:
        # Fallback: use all linear layer names
        import torch.nn as nn

        active_targets = list(
            {
                name.split(".")[-1]
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear)
            }
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=active_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    _output_dir = output_dir or tempfile.mkdtemp(prefix="ck_finetune_")

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    training_args = transformers.TrainingArguments(
        output_dir=_output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        fp16=(device == "cuda"),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=data_collator,
    )

    logger.info(f"[finetune] Starting training: {n_epochs} epoch(s), " f"{len(train_data)} examples …")
    model.config.use_cache = False
    trainer.train()
    logger.info("[finetune] Training complete.")

    # ------------------------------------------------------------------
    # Merge LoRA weights back into base model
    # ------------------------------------------------------------------
    logger.info("[finetune] Merging LoRA weights into base model …")
    model = model.merge_and_unload()
    model.config.use_cache = True
    logger.info("[finetune] Merge complete.")

    return model


# ---------------------------------------------------------------------------
# Task-data fine-tuning
# ---------------------------------------------------------------------------


def finetune_on_task_data(
    model,
    tokenizer,
    eval_data: List[dict],
    n_epochs: int = 2,
    learning_rate: float = 3e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    batch_size: int = 16,
    micro_batch_size: int = 4,
    cutoff_len: int = 256,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> object:
    """
    Fine-tune a (pruned) HuggingFace causal-LM on circuit-discovery task data.

    Each element of *eval_data* is a dict with keys ``clean`` (prompt string),
    ``correct_idx`` (int token ID of the correct completion), and optionally
    ``incorrect_idx``.  The model is trained with standard causal-LM loss,
    but only the **last token position** (predicting ``correct_idx``) receives
    gradient — the prompt tokens are masked with label ``-100``.

    After training the LoRA adapters are merged back into the base weights.

    Parameters
    ----------
    model          : HuggingFace CausalLM (float16 / bfloat16 OK).
    tokenizer      : Matching HuggingFace tokenizer.
    eval_data      : List of dicts from ``collect_eval_data()`` / ``load_eval_data()``.
    n_epochs       : Training epochs.
    learning_rate  : AdamW learning rate.
    lora_r         : LoRA rank.
    lora_alpha     : LoRA scaling factor.
    lora_dropout   : Dropout probability inside LoRA layers.
    lora_target_modules : Linear layers to wrap with LoRA.
    batch_size     : Total batch size (gradient-accumulation-aware).
    micro_batch_size : Per-device micro batch size.
    cutoff_len     : Max sequence length for tokenization.
    device         : Training device.
    output_dir     : Where to write intermediate checkpoints.

    Returns
    -------
    The fine-tuned model (LoRA weights merged) ready for inference.
    """
    LoraConfig, get_peft_model = _get_peft_classes()

    if lora_target_modules is None:
        lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)

    # ------------------------------------------------------------------
    # Tokenizer setup
    # ------------------------------------------------------------------
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # Build dataset from task eval_data
    # ------------------------------------------------------------------
    # Each example: prompt → correct next token.
    # We tokenize the prompt, append the correct token, and set labels so
    # that only the final position (the correct answer) gets gradient.
    # ------------------------------------------------------------------
    logger.info(f"[finetune-task] Building dataset from {len(eval_data)} task examples …")

    processed = []
    for row in eval_data:
        prompt_ids = tokenizer(
            row["clean"],
            truncation=True,
            max_length=cutoff_len - 1,  # leave room for correct token
            padding=False,
            return_tensors=None,
        )["input_ids"]

        correct_token = int(row["correct_idx"])

        # input_ids = [prompt ..., correct_token]
        input_ids = prompt_ids + [correct_token]
        attention_mask = [1] * len(input_ids)

        # labels: -100 for all prompt positions, correct_token at the end
        labels = [-100] * len(prompt_ids) + [correct_token]

        processed.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    # Convert to HuggingFace Dataset
    from datasets import Dataset as HFDataset

    train_data = HFDataset.from_list(processed)
    train_data.set_format(type="torch")
    logger.info(f"[finetune-task] Dataset ready: {len(train_data)} examples")

    # ------------------------------------------------------------------
    # Apply LoRA
    # ------------------------------------------------------------------
    logger.info(f"[finetune-task] Applying LoRA (r={lora_r}, α={lora_alpha}) …")
    existing_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    active_targets = [m for m in lora_target_modules if m in existing_names]
    if not active_targets:
        import torch.nn as nn

        active_targets = list(
            {
                name.split(".")[-1]
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear)
            }
        )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=active_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    _output_dir = output_dir or tempfile.mkdtemp(prefix="ck_finetune_task_")

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    training_args = transformers.TrainingArguments(
        output_dir=_output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        fp16=(device == "cuda"),
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=data_collator,
    )

    logger.info(
        f"[finetune-task] Starting training: {n_epochs} epoch(s), " f"{len(train_data)} examples …"
    )
    model.config.use_cache = False
    trainer.train()
    logger.info("[finetune-task] Training complete.")

    # ------------------------------------------------------------------
    # Merge LoRA weights back into base model
    # ------------------------------------------------------------------
    logger.info("[finetune-task] Merging LoRA weights into base model …")
    model = model.merge_and_unload()
    model.config.use_cache = True
    logger.info("[finetune-task] Merge complete.")

    return model
