"""
finetune_utils.py — Selective finetuning utilities.

Consumes SelectionResult from selector.py and applies gradient-masked training
to a HuggingFace causal LM (LLaMA / Qwen or any model following the same
layer attribute structure).

Public API
----------
setup_selective_training(model, selection, device)
    → (hook_handles, trainable_params, masks_dict)

verify_gradient_masking(trainable_params, masks_dict)
    → None  (prints verification report, call after first backward)

LanguageModelingDataset(tokenizer, clean_texts, query_strings, max_length)
    → torch.utils.data.Dataset

build_finetune_dataloader(task_spec, tokenizer, discovery_cfg,
                          device, n_examples, seed, strict_split)
    → torch.utils.data.DataLoader

run_finetuning(model, selection, finetune_dataloader, **hparams)
    → model  (same object, trained in-place; hooks removed in finally)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from circuitkit.applications.selective_finetuning.selector import SelectionResult
import logging


# ---------------------------------------------------------------------------
# 3a — Weight matrix accessors
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)

def _get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return model.model.layers[layer_idx]."""
    return model.model.layers[layer_idx]


def get_q_proj(model: nn.Module, layer_idx: int) -> nn.Parameter:
    layer = _get_layer(model, layer_idx)
    # LLaMA: self_attn.q_proj  |  Qwen: self_attn.q_proj (same)
    attn = layer.self_attn
    if hasattr(attn, "q_proj"):
        return attn.q_proj.weight
    raise AttributeError(
        f"Layer {layer_idx} self_attn has no q_proj. "
        f"Attributes: {[a for a in dir(attn) if 'proj' in a]}"
    )


def get_k_proj(model: nn.Module, layer_idx: int) -> nn.Parameter:
    attn = _get_layer(model, layer_idx).self_attn
    if hasattr(attn, "k_proj"):
        return attn.k_proj.weight
    raise AttributeError(f"Layer {layer_idx} self_attn has no k_proj.")


def get_v_proj(model: nn.Module, layer_idx: int) -> nn.Parameter:
    attn = _get_layer(model, layer_idx).self_attn
    if hasattr(attn, "v_proj"):
        return attn.v_proj.weight
    raise AttributeError(f"Layer {layer_idx} self_attn has no v_proj.")


def get_o_proj(model: nn.Module, layer_idx: int) -> nn.Parameter:
    attn = _get_layer(model, layer_idx).self_attn
    if hasattr(attn, "o_proj"):
        return attn.o_proj.weight
    raise AttributeError(f"Layer {layer_idx} self_attn has no o_proj.")


def get_down_proj(model: nn.Module, layer_idx: int) -> nn.Parameter:
    mlp = _get_layer(model, layer_idx).mlp
    # LLaMA / Qwen both use .down_proj; fallback for rare naming variants
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj.weight
    if hasattr(mlp, "c_proj"):  # GPT-2 / Mistral variants
        return mlp.c_proj.weight
    raise AttributeError(
        f"Layer {layer_idx} mlp has no down_proj or c_proj. "
        f"Attributes: {[a for a in dir(mlp) if 'proj' in a]}"
    )


# ---------------------------------------------------------------------------
# Hook factory  (closure-factory pattern — avoids late-binding bug)
# ---------------------------------------------------------------------------


def _make_grad_hook(mask: torch.Tensor):
    """
    Return a gradient hook that zeroes out positions where mask == 0.

    mask must already be on the correct device and broadcastable to the
    gradient shape (e.g. [out, 1] for row masks, [1, in] for column masks).

    Using a factory function ensures each hook closes over its own `mask`
    object rather than a shared loop variable — the critical pattern when
    registering hooks inside a loop.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask

    return hook


# ---------------------------------------------------------------------------
# 3b — setup_selective_training
# ---------------------------------------------------------------------------


def setup_selective_training(
    model: nn.Module,
    selection: SelectionResult,
    device: torch.device,
) -> Tuple[List[Any], List[nn.Parameter], Dict[str, torch.Tensor]]:
    """
    Freeze all parameters, then selectively unfreeze and gradient-mask the
    components described by selection.

    Parameters
    ----------
    model     : HuggingFace CausalLM (LLaMA / Qwen / compatible).
    selection : SelectionResult from selector.select_components (or random /
                baseline variants).  None index lists → full matrix, no mask.
    device    : Training device.

    Returns
    -------
    hook_handles    : List of RemovableHook objects.  Call handle.remove() in
                      a finally block after training to restore the model.
    trainable_params: List[nn.Parameter] — pass to the optimiser.
    masks_dict      : Dict[param_id_str, Tensor] — used by verify_gradient_masking.
                      Key is str(id(param)); value is the mask tensor.
    """
    # ── Step 1: freeze everything ──────────────────────────────────────────
    for param in model.parameters():
        param.requires_grad_(False)

    hook_handles: List[Any] = []
    trainable_params: List[nn.Parameter] = []
    masks_dict: Dict[str, torch.Tensor] = {}

    # ── Step 2: attention projections ─────────────────────────────────────
    try:
        for key, proj_dict in selection.attn.items():
            # key = "attn_{layer}"
            layer_idx = int(key.split("_")[1])

            _proj_accessors = {
                "q": get_q_proj,
                "k": get_k_proj,
                "v": get_v_proj,
                "o": get_o_proj,
            }

            for proj_name, index_list in proj_dict.items():
                accessor = _proj_accessors[proj_name]
                param = accessor(model, layer_idx)
                param.requires_grad_(True)

                if index_list is None:
                    # Baseline: no mask, full matrix updates.
                    if not any(param is p for p in trainable_params):
                        trainable_params.append(param)
                    continue

                # Build mask tensor.
                out_features, in_features = param.shape

                if proj_name in ("q", "k", "v"):
                    # Row mask: selected rows gradient flows; rest zeroed.
                    # Shape [out_features, 1] broadcasts over in_features.
                    mask = torch.zeros(out_features, 1, dtype=param.dtype, device=device)
                    mask[index_list] = 1.0
                else:
                    # "o": column mask on o_proj [d_model, n_q_heads * head_dim].
                    # Selected columns → allowed; shape [1, in_features].
                    mask = torch.zeros(1, in_features, dtype=param.dtype, device=device)
                    mask[0, index_list] = 1.0

                handle = param.register_hook(_make_grad_hook(mask))
                hook_handles.append(handle)
                masks_dict[str(id(param))] = mask

                if not any(param is p for p in trainable_params):
                    trainable_params.append(param)

        # ── Step 3: MLP down_proj ─────────────────────────────────────────────
        for key, index_list in selection.mlp.items():
            # key = "mlp_{layer}"
            layer_idx = int(key.split("_")[1])
            param = get_down_proj(model, layer_idx)
            param.requires_grad_(True)

            if index_list is None:
                # Node-level or baseline: full down_proj, no mask.
                if not any(param is p for p in trainable_params):
                    trainable_params.append(param)
                continue

            # Neuron-level: column mask on down_proj [d_model, d_mlp].
            # Selected columns → down_proj input channels for chosen neurons.
            out_features, in_features = param.shape
            mask = torch.zeros(1, in_features, dtype=param.dtype, device=device)
            mask[0, index_list] = 1.0

            handle = param.register_hook(_make_grad_hook(mask))
            hook_handles.append(handle)
            masks_dict[str(id(param))] = mask

            if not any(param is p for p in trainable_params):
                trainable_params.append(param)

    except Exception:
        # ── Absolute Lifecycle Safety ──────────────────────────────────────────
        # If setup crashes halfway, remove any hooks we successfully attached
        # before letting the error bubble up.
        for handle in hook_handles:
            handle.remove()
        logger.warning(
            f"[finetune_utils] Cleaned up {len(hook_handles)} orphaned hooks after setup failure."
        )
        raise

    n_masked = len(masks_dict)
    n_unmasked = len(trainable_params) - n_masked
    logger.warning(
        f"[finetune_utils] Trainable params: {len(trainable_params)}  "
        f"(masked: {n_masked}, unmasked/baseline: {n_unmasked})"
    )
    return hook_handles, trainable_params, masks_dict


# ---------------------------------------------------------------------------
# 3c — verify_gradient_masking
# ---------------------------------------------------------------------------


def verify_gradient_masking(
    trainable_params: List[nn.Parameter],
    masks_dict: Dict[str, torch.Tensor],
) -> None:
    """
    After the first backward pass, verify that:
      - Non-selected positions have near-zero gradients.
      - Selected positions have non-zero gradients.

    Call once after the very first batch of the first epoch.
    Prints a ✓ or ✗ per masked parameter.
    """
    logger.warning("\n[finetune_utils] Gradient masking verification:")
    all_ok = True

    for param in trainable_params:
        pid = str(id(param))
        if pid not in masks_dict:
            continue  # unmasked baseline param — skip
        if param.grad is None:
            logger.info(f"  ✗ param {param.shape} — no gradient (was backward called?)")
            all_ok = False
            continue

        mask = masks_dict[pid]
        grad = param.grad

        # Non-selected positions should be ~0
        inv_mask = (1.0 - mask).expand_as(grad)
        leak = (grad * inv_mask).abs().max().item()
        # Selected positions should have signal
        sel_mask = mask.expand_as(grad)
        signal = (grad * sel_mask).abs().max().item()

        ok = (leak < 1e-6) and (signal > 0.0)
        symbol = "✓" if ok else "✗"
        logger.warning(
            f"  {symbol} param {tuple(param.shape)} | "
            f"max_leak={leak:.2e}  max_signal={signal:.2e}"
        )
        if not ok:
            all_ok = False

    status = "PASSED" if all_ok else "FAILED — check hook registration"
    logger.info(f"  → Verification {status}\n")


# ---------------------------------------------------------------------------
# 3d — LanguageModelingDataset
# ---------------------------------------------------------------------------


class LanguageModelingDataset(Dataset):
    """
    Dataset for causal language modelling finetuning.

    Each sample tokenises the full (clean) text with padding, then separately
    tokenises the query prefix to obtain its exact token length.  The training
    loss can later be restricted to the completion tokens by using query_length
    to mask the prompt prefix.

    Parameters
    ----------
    tokenizer    : HuggingFace tokenizer with a pad token set.
    clean_texts  : List of full input strings (prompt + completion).
    query_strings: List of prompt-only strings, one per clean_text.
    max_length   : Sequence length for padding / truncation.
    templated    : True when the supplied texts are already chat-templated. A
                   chat template renders its own beginning-of-text token into
                   the string, so adding another via add_special_tokens=True
                   would inject a second BOS and shift every position by one.
                   Defaults to False — raw texts keep the tokenizer's default
                   special-token handling, byte-identical for base models /
                   "off" tasks.
    """

    def __init__(
        self,
        tokenizer,
        clean_texts: List[str],
        query_strings: List[str],
        max_length: int,
        templated: bool = False,
    ) -> None:
        assert len(clean_texts) == len(query_strings), (
            f"clean_texts and query_strings must have the same length, "
            f"got {len(clean_texts)} and {len(query_strings)}."
        )
        self.tokenizer = tokenizer
        self.clean_texts = clean_texts
        self.query_strings = query_strings
        self.max_length = max_length
        self.templated = templated

    def __len__(self) -> int:
        return len(self.clean_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Templated text already carries the model's BOS; tokenizing it with
        # add_special_tokens=True would inject a second BOS. add_special_tokens
        # is left at the tokenizer default (True) for raw text — byte-identical
        # to the legacy behavior for base models / "off" tasks.
        add_special = not self.templated
        full_enc = self.tokenizer(
            self.clean_texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=add_special,
        )
        # Tokenise query without padding to get its exact unpadded length.
        query_enc = self.tokenizer(
            self.query_strings[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
            add_special_tokens=add_special,
        )
        query_length = query_enc["input_ids"].shape[1]

        # Sanity: query must not exceed the non-padded full sequence length.
        # (rare edge-case: very short full_text or very long query)
        attention_mask = full_enc["attention_mask"].squeeze(0)
        real_length = attention_mask.sum().item()
        if query_length > real_length:
            query_length = int(real_length)

        return {
            "input_ids": full_enc["input_ids"].squeeze(0),
            "attention_mask": attention_mask,
            "query_length": torch.tensor(query_length, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 3e — build_finetune_dataloader
# ---------------------------------------------------------------------------


def build_finetune_dataloader(
    task_spec,
    tokenizer,
    model_name: str,
    discovery_cfg: Dict[str, Any],
    device: torch.device,
    n_examples: int,
    max_length: int,
    batch_size: int = 8,
    seed: int = 42,
    strict_split: bool = False,
) -> DataLoader:
    """
    Build a DataLoader for causal LM finetuning from a CircuitKit task spec.

    Delegates data generation to task_spec.build_finetuning_dataset(), which
    uses only a HuggingFace tokenizer (no HookedTransformer required). The
    resulting (clean_texts, query_strings) pairs are wrapped in a
    LanguageModelingDataset so the LM loss is restricted to the answer token(s).

    Parameters
    ----------
    task_spec      : CircuitKit task spec with build_finetuning_dataset().
    tokenizer      : HuggingFace tokenizer (already has pad_token set).
    model_name     : Full HuggingFace model identifier — forwarded to
                     build_finetuning_dataset for cache path reconstruction.
    discovery_cfg  : Config dict passed through to build_finetuning_dataset.
                     For MMLU: include 'subjects', 'samples_per_subject'.
                     For all tasks: optionally include 'cache_dir' override.
    device         : Unused directly; kept for API symmetry with the pipeline.
    n_examples     : Number of training examples to collect.
    max_length     : Token sequence length for padding / truncation.
    batch_size     : Batch size for the returned DataLoader.
    seed           : RNG seed for data shuffling and DataLoader generator.
    strict_split   : Documented intent: exclude discovery examples from
                     finetuning. Not yet enforced — finetuning data is drawn
                     from the same cache as discovery. Raises a warning if True.

    Returns
    -------
    DataLoader yielding batches of {input_ids, attention_mask, query_length}.
    """
    if strict_split:
        import warnings

        warnings.warn(
            "strict_split=True is not yet enforced. Finetuning examples may "
            "overlap with discovery examples drawn from the same cache. "
            "Implement index tracking in build_finetuning_dataset to enforce "
            "a true split.",
            UserWarning,
            stacklevel=2,
        )

    clean_texts, query_strings = task_spec.build_finetuning_dataset(
        tokenizer=tokenizer,
        model_name=model_name,
        n_examples=n_examples,
        discovery_cfg=discovery_cfg,
        seed=seed,
    )

    if not clean_texts:
        raise ValueError(
            f"build_finetuning_dataset returned no examples for task "
            f"'{type(task_spec).__name__}'. Check cache or data availability."
        )

    logger.warning(
        f"[finetune_utils] Collected {len(clean_texts)} finetuning examples "
        f"(requested {n_examples})."
    )

    # Resolve the same chat-template decision build_finetuning_dataset used, so
    # the dataset tokenizes templated text without injecting a second BOS. The
    # task's declared chat_template_mode collapses against the tokenizer (a
    # tokenizer carrying a chat_template ⇒ chat model); a discovery_cfg override
    # wins. For base models / "off" tasks this is False — byte-identical.
    from circuitkit.tasks._chat import resolve_chat_template_from_tokenizer

    chat_mode = discovery_cfg.get(
        "chat_template_mode", getattr(task_spec, "chat_template_mode", "auto")
    )
    templated = resolve_chat_template_from_tokenizer(chat_mode, tokenizer)

    dataset = LanguageModelingDataset(
        tokenizer=tokenizer,
        clean_texts=clean_texts,
        query_strings=query_strings,
        max_length=max_length,
        templated=templated,
    )

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# 3f — run_finetuning
# ---------------------------------------------------------------------------


def run_finetuning(
    model: nn.Module,
    selection: SelectionResult,
    finetune_dataloader: DataLoader,
    device: torch.device,
    n_epochs: int = 3,
    lr: float = 2e-5,
    max_grad_norm: float = 1.0,
    eval_dataloader: Optional[DataLoader] = None,
    eval_fn: Optional[Any] = None,
    log_every: int = 10,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Run selective gradient-masked finetuning.

    Parameters
    ----------
    model               : HuggingFace CausalLM on `device`.
    selection           : SelectionResult controlling which params are trained.
    finetune_dataloader : DataLoader from build_finetune_dataloader.
    device              : Training device.
    n_epochs            : Number of full passes over finetune_dataloader.
    lr                  : AdamW learning rate.
    max_grad_norm       : Gradient clipping norm (applied before hooks fire,
                          so effective clipping is on the full gradient; the
                          mask zeroes the irrelevant slice post-clip).
    eval_dataloader     : Optional DataLoader for per-epoch eval.
    eval_fn             : Optional callable(model, eval_dataloader) → Dict[str, float].
                          Called at the end of every epoch if provided.
    log_every           : Print training loss every N steps.

    Returns
    -------
    model       : Same object, trained in-place.
    epoch_logs  : List[Dict] — one dict per epoch with 'loss' and any eval metrics.

    Notes
    -----
    Hooks are always removed in a finally block.  If training is interrupted
    the model is left in its partially-trained state with requires_grad reset.
    """
    hook_handles, trainable_params, masks_dict = setup_selective_training(model, selection, device)

    optimiser = torch.optim.AdamW(trainable_params, lr=lr)
    epoch_logs: List[Dict[str, float]] = []
    verified = False

    try:
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for step, batch in enumerate(finetune_dataloader, 1):
                input_ids = batch["input_ids"].to(device)  # [B, L]
                attention_mask = batch["attention_mask"].to(device)  # [B, L]
                query_lengths = batch["query_length"]  # [B]

                # ── Causal LM loss restricted to completion tokens ─────────
                # Labels: copy input_ids, then set prompt positions to -100 so cross-entropy ignores them.
                labels = input_ids.clone()

                # Dynamically infer padding side from the mask
                is_left_padded = (attention_mask[:, 0] == 0).any().item()
                padding_side = "left" if is_left_padded else "right"

                for i, qlen in enumerate(query_lengths):
                    if padding_side == "left":
                        # Padding is at the beginning; query starts after padding
                        pad_count = (attention_mask[i] == 0).sum().item()
                        labels[i, pad_count : pad_count + qlen] = -100
                    else:
                        # Right padding; query is strictly at the beginning
                        labels[i, :qlen] = -100
                # Also ignore padding
                labels[attention_mask == 0] = -100

                if step == 1 and epoch == 1:
                    logger.warning("\n[DEBUG] Training Batch 1 - Active Labels")
                    for i in range(min(2, input_ids.size(0))):
                        # Extract only the unmasked labels the model is penalized on
                        active_label_ids = labels[i][labels[i] != -100]
                        # We need the tokenizer to decode, so you may need to pass tokenizer to run_finetuning
                        # Or just print the raw IDs to see if they are EOS/Pad IDs (e.g., 128001 or 128009 for Llama 3)
                        logger.info(f"  Example {i} active label IDs: {active_label_ids.tolist()}")

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                optimiser.zero_grad()
                loss.backward()

                # Gradient verification on very first batch of epoch 1.
                if not verified:
                    verify_gradient_masking(trainable_params, masks_dict)
                    verified = True

                # Clip then step — hooks have already been called by autograd
                # during backward, so masks are applied before clipping acts
                # on the masked gradients.
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

                if step % log_every == 0:
                    logger.warning(
                        f"  [Epoch {epoch}  Step {step}/{len(finetune_dataloader)}] "
                        f"loss={loss.item():.4f}"
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            log: Dict[str, float] = {"epoch": epoch, "loss": avg_loss}
            logger.info(f"[finetune_utils] Epoch {epoch} — avg loss: {avg_loss:.4f}")

            # ── Optional eval ──────────────────────────────────────────────
            if eval_fn is not None and eval_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    eval_metrics = eval_fn(model, eval_dataloader)
                log.update(eval_metrics)
                logger.warning(
                    f"[finetune_utils] Epoch {epoch} eval — "
                    + "  ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items())
                )

            epoch_logs.append(log)

    finally:
        # Always remove hooks — even if training crashes — so the model is
        # left in a clean state for the caller.
        for handle in hook_handles:
            handle.remove()
        logger.info(f"[finetune_utils] Removed {len(hook_handles)} gradient hooks.")

    return model, epoch_logs
