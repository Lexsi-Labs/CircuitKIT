# FILE: circuitkit/applications/finetuning/soft_healing.py
"""
Soft Healing via LoRA: Fine-tune pruned models using circuit-guided LoRA.

This module implements CircuitLoRA, which applies Low-Rank Adaptation (LoRA)
to circuit-relevant modules in a pruned transformer model. The goal is to
recover performance of pruned models through targeted fine-tuning.
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer


import logging

logger = logging.getLogger(__name__)

class LoRALayer(nn.Module):
    """
    A LoRA layer that wraps a linear layer with low-rank adapters.

    The adapted output is: output = input @ W + input @ A @ B
    where A has shape (d_in, r) and B has shape (r, d_out), with r being the LoRA rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize a LoRA layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            lora_rank: Rank of the LoRA decomposition
            lora_alpha: Scaling factor for LoRA contributions
            dtype: Parameter dtype. Should match the base model so the LoRA
                contribution can be added to bf16 / fp16 activations without
                a dtype mismatch. Defaults to torch.float32.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0

        # Initialize LoRA weight matrices in the requested dtype so reduced-
        # precision (bf16 / fp16) models do not hit a dtype mismatch.
        self.lora_A = nn.Parameter(torch.zeros(in_features, lora_rank, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, out_features, dtype=dtype))

        # Initialize A with Gaussian, B with zeros (as in original LoRA paper)
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            LoRA contribution of shape (..., out_features)
        """
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class CircuitLoRA(nn.Module):
    """
    LoRA-based soft healing for pruned circuits.
    """

    def __init__(
        self,
        model: HookedTransformer,
        circuit_scores: Optional[Dict[str, float]] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        target_modules: Optional[List[str]] = None,
        score_threshold: float = 0.0,
        *,
        circuit: Optional[Dict[str, float]] = None,
        rank: Optional[int] = None,
    ):

        super().__init__()

        # Accept `circuit` and `rank` as aliases used in application scripts.
        if circuit is not None:
            circuit_scores = circuit
        if rank is not None:
            lora_rank = rank
        # circuit_scores=None means unrestricted LoRA: apply to all nodes.
        if circuit_scores is None:
            circuit_scores = self._all_node_scores(model)

        self.model = model
        self.circuit_scores = circuit_scores
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.score_threshold = score_threshold

        # Capture the base model dtype so LoRA layers are created to match it
        # (bf16 / fp16 models would otherwise hit a dtype mismatch in the
        # hook-based forward).
        try:
            self._model_dtype = next(iter(model.parameters())).dtype
        except StopIteration:
            self._model_dtype = torch.float32

        # FIX: default now includes 'attn_head' so attention heads are not
        # silently skipped when the caller doesn't pass target_modules.
        if target_modules is None:
            target_modules = ["W_out", "W_in", "attn_head"]
        self.target_modules = target_modules

        # BUGFIX: the LoRA hooks read intermediate hook points that are
        # OPT-IN in TransformerLens and do not fire by default:
        #   - the W_in path caches ``blocks.{l}.hook_mlp_in`` (needs
        #     ``use_hook_mlp_in``); without the flag the cache is never
        #     populated and ``finetune`` raises KeyError 'blocks.0.hook_mlp_in'.
        #   - the attn-head path reads ``hook_z`` and writes ``hook_result``
        #     (needs ``use_attn_result``).
        # Enable the flags here so soft-healing works out of the box.
        if "W_in" in self.target_modules:
            model.cfg.use_hook_mlp_in = True
        if "attn_head" in self.target_modules:
            model.cfg.use_attn_result = True

        self.lora_modules = nn.ModuleDict()
        self.lora_metadata = {}

        self.high_score_nodes = {
            name: score for name, score in circuit_scores.items() if score >= score_threshold
        }

        self._apply_lora_to_circuit()

    def get_lora_hooks(self) -> List[Tuple[str, callable]]:
        """
        Generates TransformerLens hooks to physically inject LoRA computations
        into the model's forward pass.

        Attention heads are grouped by layer so that exactly ONE hook is
        registered per layer result point. If multiple heads in the same layer
        are targeted and each had its own hook on hook_result, TransformerLens
        would chain them sequentially — and the backward pass would retain one
        intermediate result tensor per head per layer in the autograd graph
        (O(K) memory). Grouping into a single hook means one clone per layer
        regardless of how many heads are targeted.

        The returned list is structurally static (same hook points every call).
        Build it once per training run rather than once per batch.
        """
        cache = {}
        hooks = []
        added_caches = set()

        # --- Attention: one hook per layer covering all targeted heads ---
        attn_heads_by_layer: Dict[int, List[Tuple[int, str]]] = {}
        for key, meta in self.lora_metadata.items():
            if meta["type"] == "attn_head":
                attn_heads_by_layer.setdefault(meta["layer_idx"], []).append(
                    (meta["head_idx"], key)
                )

        for layer, head_list in attn_heads_by_layer.items():
            z_name = f"blocks.{layer}.attn.hook_z"
            result_name = f"blocks.{layer}.attn.hook_result"

            if z_name not in added_caches:

                def store_z(act, hook):
                    cache[hook.name] = act
                    return act

                hooks.append((z_name, store_z))
                added_caches.add(z_name)

            def add_attn_lora(result, hook, lyr=layer, heads=head_list):
                z_act = cache[f"blocks.{lyr}.attn.hook_z"]
                # Clone once for the whole layer, then apply every targeted
                # head's LoRA contribution in-place on the clone.
                # A freshly-cloned tensor (version counter = 0, no prior
                # references) is safe for in-place ops under autograd.
                result = result.clone()
                for h, k in heads:
                    result[:, :, h, :] = result[:, :, h, :] + self.lora_modules[k](
                        z_act[:, :, h, :]
                    )
                return result

            hooks.append((result_name, add_attn_lora))

        # --- MLP: one hook per module per layer (unchanged) ---
        for key, meta in self.lora_metadata.items():
            if meta["type"] != "mlp":
                continue
            layer = meta["layer_idx"]
            key_in, key_out = meta["keys"]

            # W_in LoRA
            if key_in:
                in_name = f"blocks.{layer}.hook_mlp_in"
                pre_name = f"blocks.{layer}.mlp.hook_pre"

                if in_name not in added_caches:

                    def store_in(act, hook):
                        cache[hook.name] = act
                        return act

                    hooks.append((in_name, store_in))
                    added_caches.add(in_name)

                def add_mlp_in_lora(pre_act, hook, lyr=layer, k_in=key_in):
                    mlp_in = cache[f"blocks.{lyr}.hook_mlp_in"]
                    return pre_act + self.lora_modules[k_in](mlp_in)

                hooks.append((pre_name, add_mlp_in_lora))

            # W_out LoRA
            if key_out:
                post_name = f"blocks.{layer}.mlp.hook_post"
                out_name = f"blocks.{layer}.hook_mlp_out"

                if post_name not in added_caches:

                    def store_post(act, hook):
                        cache[hook.name] = act
                        return act

                    hooks.append((post_name, store_post))
                    added_caches.add(post_name)

                def add_mlp_out_lora(out_act, hook, lyr=layer, k_out=key_out):
                    mlp_post = cache[f"blocks.{lyr}.mlp.hook_post"]
                    return out_act + self.lora_modules[k_out](mlp_post)

                hooks.append((out_name, add_mlp_out_lora))

        return hooks

    def _apply_lora_to_circuit(self):
        """Apply LoRA layers to circuit-relevant modules based on scores."""
        logger.info(f"Applying LoRA to circuit-relevant nodes (threshold: {self.score_threshold})")

        for node_name, score in self.high_score_nodes.items():
            attn_match = re.match(r"A(\d+)\.(\d+)", node_name)
            if attn_match and "attn_head" in self.target_modules:
                self._apply_lora_to_attn_head(
                    int(attn_match.group(1)), int(attn_match.group(2)), score
                )
                continue

            mlp_match = re.match(r"MLP\s+(\d+)", node_name)
            if mlp_match:
                if "W_in" in self.target_modules or "W_out" in self.target_modules:
                    self._apply_lora_to_mlp(int(mlp_match.group(1)), score)
                continue

    def _apply_lora_to_attn_head(self, layer_idx: int, head_idx: int, score: float):
        try:
            d_head = self.model.cfg.d_head
            d_model = self.model.cfg.d_model

            lora_layer = LoRALayer(
                in_features=d_head,
                out_features=d_model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                dtype=self._model_dtype,
            )

            layer_key = f"attn_L{layer_idx}_H{head_idx}"
            self.lora_modules[layer_key] = lora_layer
            self.lora_metadata[layer_key] = {
                "type": "attn_head",
                "layer_idx": layer_idx,
                "head_idx": head_idx,
                "score": score,
            }
            logger.info(f"  Applied LoRA to attention head A{layer_idx}.{head_idx} (score: {score:.4f})")
        except Exception as e:
            warnings.warn(f"Failed to apply LoRA to attention head L{layer_idx}H{head_idx}: {e}")

    def _apply_lora_to_mlp(self, layer_idx: int, score: float):
        try:
            d_model = self.model.cfg.d_model
            d_mlp = self.model.cfg.d_mlp

            key_in = f"mlp_L{layer_idx}_in" if "W_in" in self.target_modules else None
            key_out = f"mlp_L{layer_idx}_out" if "W_out" in self.target_modules else None

            if key_in:
                self.lora_modules[key_in] = LoRALayer(
                    in_features=d_model,
                    out_features=d_mlp,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    dtype=self._model_dtype,
                )
            if key_out:
                self.lora_modules[key_out] = LoRALayer(
                    in_features=d_mlp,
                    out_features=d_model,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    dtype=self._model_dtype,
                )

            self.lora_metadata[f"mlp_L{layer_idx}"] = {
                "type": "mlp",
                "layer_idx": layer_idx,
                "score": score,
                "keys": (key_in, key_out),
            }
            logger.info(f"  Applied LoRA to MLP layer L{layer_idx} (score: {score:.4f})")
        except Exception as e:
            warnings.warn(f"Failed to apply LoRA to MLP L{layer_idx}: {e}")

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""
        return list(self.lora_modules.parameters())

    def _prepare_batch(self, batch: Any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract and move (input_ids, labels) to device from a flexible batch format.
        Applies loss_mask by setting ignored positions to -100.

        Args:
            batch: Dict, list/tuple, or raw tensor from the DataLoader.
            device: Target device string.

        Returns:
            (input_ids, labels) both on the target device.
        """
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids", batch.get("clean"))
            labels = batch.get("labels", input_ids)
            loss_mask = batch.get("loss_mask", None)
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            labels = batch[1] if len(batch) > 1 else batch[0]
            loss_mask = batch[2] if len(batch) > 2 else None
        else:
            input_ids = batch
            labels = batch
            loss_mask = None

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if loss_mask is not None:
            # FIX: clone before masking to avoid mutating the DataLoader's
            # underlying tensor (important with pin_memory=True).
            labels = labels.clone()
            labels[loss_mask.to(device) == 0] = -100

        return input_ids, labels

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Fine-tune LoRA adapters on pruned model.

        Only LoRA parameters are optimized; original model weights are frozen.
        If a val_loader is provided, the best LoRA state (by validation loss)
        is restored at the end of training.

        Args:
            train_loader: DataLoader for training data. Each batch should be
                         a dict with keys 'input_ids', 'attention_mask', 'labels'
            val_loader: Optional validation DataLoader for best-state tracking
            epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
            weight_decay: L2 regularization coefficient

        Returns:
            Dict with training metrics:
                - 'train_loss': List of training losses per epoch
                - 'val_loss': List of validation losses per epoch (if val_loader provided)
                - 'best_epoch': Epoch with best validation loss
                - 'total_params': Total LoRA parameters
        """
        lora_params = self.get_lora_parameters()

        if not lora_params:
            warnings.warn("No LoRA parameters to train. Check circuit_scores and threshold.")
            return {"train_loss": [], "val_loss": [], "total_params": 0}

        device = self.model.cfg.device
        self.model.to(device)
        # FIX: lora_modules must be on the same device as the model; previously
        # only self.model was moved, leaving LoRA layers on CPU.
        self.lora_modules.to(device)

        # Freeze base model, unfreeze LoRA only
        for param in self.model.parameters():
            param.requires_grad = False
        for param in lora_params:
            param.requires_grad = True

        optimizer = torch.optim.Adam(lora_params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        # FIX: ignore_index=-100 so that positions masked to -100 do not
        # contribute to the loss (loss_mask support was silently broken before).
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # FIX: build hooks once per training run, not once per batch.
        # The hook list is structurally static; only the activation values in
        # the shared cache dict change during each forward pass.
        lora_hooks = self.get_lora_hooks()

        metrics = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_val_loss": float("inf"),
            "total_params": sum(p.numel() for p in lora_params),
        }

        best_lora_state: Optional[Dict[str, torch.Tensor]] = None

        logger.info(f"\nStarting LoRA fine-tuning for {epochs} epochs")
        logger.info(f"Total LoRA parameters: {metrics['total_params']:,}")
        logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}\n")

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, loss_fn, optimizer, device, lora_hooks)
            metrics["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, loss_fn, device, lora_hooks)
                metrics["val_loss"].append(val_loss)

                # FIX: actually save the best state so we can restore it.
                if val_loss < metrics["best_val_loss"]:
                    metrics["best_val_loss"] = val_loss
                    metrics["best_epoch"] = epoch
                    best_lora_state = {
                        k: v.clone() for k, v in self.lora_modules.state_dict().items()
                    }

            msg = f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f", val_loss={val_loss:.4f}"
            logger.info(msg)

            scheduler.step()

        # Restore the LoRA weights from the best validation epoch.
        if best_lora_state is not None:
            self.lora_modules.load_state_dict(best_lora_state)
            logger.info(f"\nRestored best LoRA state from epoch {metrics['best_epoch'] + 1}")

        logger.info(f"\nLoRA training complete. Best epoch: {metrics['best_epoch'] + 1}")
        return metrics

    def _train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        lora_hooks: List[Tuple[str, callable]],
    ) -> float:
        """
        Single training epoch.

        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to use
            lora_hooks: Pre-built hook list from get_lora_hooks()

        Returns:
            Average training loss
        """
        self.model.eval()  # Keep base model in eval mode; only LoRA params train
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids, labels = self._prepare_batch(batch, device)

            with self.model.hooks(fwd_hooks=lora_hooks):
                logits = self.model(input_ids)

            if logits.dim() == 3:
                loss = loss_fn(
                    logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1)
                )
            else:
                loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_epoch(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        lora_hooks: List[Tuple[str, callable]],
    ) -> float:
        """
        Single validation epoch.

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            device: Device to use
            lora_hooks: Pre-built hook list from get_lora_hooks()

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = self._prepare_batch(batch, device)

                with self.model.hooks(fwd_hooks=lora_hooks):
                    logits = self.model(input_ids)

                if logits.dim() == 3:
                    loss = loss_fn(
                        logits[..., :-1, :].reshape(-1, logits.size(-1)),
                        labels[..., 1:].reshape(-1),
                    )
                else:
                    loss = loss_fn(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def get_lora_state_dict(self) -> Dict[str, Any]:
        """
        Get state dict of LoRA layers only (for saving).
        Maintains original nested dictionary structure for backwards compatibility.
        """
        state_dict = {}
        for key, meta in self.lora_metadata.items():
            if meta["type"] == "attn_head":
                state_dict[key] = self.lora_modules[key].state_dict()
            elif meta["type"] == "mlp":
                key_in, key_out = meta["keys"]
                state_dict[key] = {
                    "module_in": self.lora_modules[key_in].state_dict() if key_in else None,
                    "module_out": self.lora_modules[key_out].state_dict() if key_out else None,
                }
        return state_dict

    @staticmethod
    def _all_node_scores(model: HookedTransformer) -> Dict[str, float]:
        """Return unit scores for every attention head and MLP layer (unrestricted LoRA)."""
        scores: Dict[str, float] = {}
        for lyr in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                scores[f"A{lyr}.{h}"] = 1.0
            scores[f"MLP {lyr}"] = 1.0
        return scores

    def load_lora_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load LoRA state dict from file, supporting the original nested structure.
        """
        for key, lora_state in state_dict.items():
            if key in self.lora_metadata:
                meta = self.lora_metadata[key]
                if meta["type"] == "attn_head":
                    self.lora_modules[key].load_state_dict(lora_state)
                elif meta["type"] == "mlp":
                    key_in, key_out = meta["keys"]
                    if key_in:
                        self.lora_modules[key_in].load_state_dict(lora_state["module_in"])
                    if key_out:
                        self.lora_modules[key_out].load_state_dict(lora_state["module_out"])


def _build_alpaca_dataloader(
    model: HookedTransformer,
    n_train: int = 1000,
    n_val: int = 128,
    max_length: int = 256,
    batch_size: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Build disjoint train/val dataloaders from the Alpaca dataset.

    Downloads tatsu-lab/alpaca from HuggingFace, tokenizes with the model's
    tokenizer, and returns two DataLoaders with non-overlapping slices.
    """
    import random as _random

    from datasets import load_dataset


    ds = load_dataset("tatsu-lab/alpaca", split="train")
    items = list(ds)

    rng = _random.Random(seed)
    rng.shuffle(items)
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]

    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tokenize(items_list):
        texts = []
        for it in items_list:
            instr = it.get("instruction", "")
            inp = it.get("input", "")
            out = it.get("output", "")
            if inp:
                text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            else:
                text = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
            texts.append(text)
        enc = tok(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return enc["input_ids"]

    class _TextDS(torch.utils.data.Dataset):
        def __init__(self, ids: torch.Tensor):
            self.ids = ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return {"input_ids": self.ids[i], "labels": self.ids[i].clone()}

    g_train = torch.Generator()
    g_train.manual_seed(seed)
    g_val = torch.Generator()
    g_val.manual_seed(seed + 1)

    train_ids = _tokenize(train_items)
    val_ids = _tokenize(val_items)

    train_loader = DataLoader(
        _TextDS(train_ids), batch_size=batch_size, shuffle=True, generator=g_train
    )
    val_loader = DataLoader(_TextDS(val_ids), batch_size=batch_size, shuffle=False, generator=g_val)
    return train_loader, val_loader


def train_healing_lora(
    model: HookedTransformer,
    circuit_lora: CircuitLoRA,
    epochs: int = 2,
    seed: int = 42,
    n_train: int = 1000,
    n_val: int = 128,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
) -> Dict[str, Any]:
    """Train a CircuitLoRA adapter on a small Alpaca slice.

    Builds disjoint train/val dataloaders keyed by seed, calls
    circuit_lora.finetune(), and returns the training metrics dict.
    """
    train_loader, val_loader = _build_alpaca_dataloader(
        model,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        seed=seed,
    )
    return circuit_lora.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
    )
