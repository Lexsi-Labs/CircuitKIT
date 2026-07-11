"""C-ΔΘ contrastive weight steering — paper-faithful implementation.

Mirrors the methodology in /AdityaKasliwal/ckit_theta/examples/
weight_steering_experiment.py (the canonical C-ΔΘ codebase).

Recipe (Aryan / Pratinav et al., NeurIPS 2026):

    Given a target model M and a discovered circuit C (top-k% heads):

    1. Make two deep copies M_pos, M_neg.
    2. Fine-tune M_pos on the POSITIVE behaviour dataset, ONLY updating
       the per-head weight slices W_Q[l, h], W_K[l, h], W_V[l, h],
       W_O[l, h] of every (l, h) in C. All other parameters frozen.
    3. Fine-tune M_neg on the NEGATIVE behaviour dataset, same per-head
       slice mask.
    4. Compute the steering vector

           w_b[(l, h, p)] = θ_pos[(l, h, p)] − θ_neg[(l, h, p)]

       where p ∈ {W_Q, W_K, W_V, W_O}.
    5. Apply to a fresh target model:

           θ_steered[(l, h, p)] = θ_target[(l, h, p)] + k · w_b[(l, h, p)]

       leaving non-circuit weights untouched. Coefficient k controls
       how aggressively the model is steered toward the positive
       behaviour and away from the negative one.

This is fundamentally different from activation steering (which adds
runtime hooks), CircuitLoRA (which adds adapter parameters), and
plain GD on circuit weights (which only sees one dataset). The paper
question is whether the circuit-aware per-head slice steering beats
a generic MLP2-LoRA baseline on the (target effect / side effect)
frontier.

Architecture support: GPT-2 (n_kv_heads == n_heads, simple per-head
slice). Llama / Mistral / Gemma with grouped-query attention map
head_idx -> kv_head_idx via floor-div, mirroring the reference
implementation. Llama-style gated MLPs are NOT yet covered for the
MLP arm (the paper's experiments are head-only).
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from transformer_lens import HookedTransformer

logger = logging.getLogger(__name__)

# Tensor identifying a per-head weight slice into a full attention
# Parameter. ``param_tensor[head_idx]`` is the slice we differentiate.
HeadSlice = Tuple[nn.Parameter, int]  # (full_param, head_idx)


def _parse_head_name(head_name: str) -> Tuple[int, int]:
    m = re.match(r"A(\d+)\.(\d+)", head_name)
    if not m:
        raise ValueError(f"head name must match A<layer>.<head>: got {head_name!r}")
    return int(m.group(1)), int(m.group(2))


def get_head_weight_info(model: HookedTransformer, head_name: str) -> Dict[str, HeadSlice]:
    """Return ``{W_Q | W_K | W_V | W_O: (full_parameter, head_idx)}``.

    Mirrors ckit_theta/examples/weight_steering_experiment.get_head_weight_info.
    Handles grouped-query attention via floor-div mapping for K/V.
    """
    layer_idx, head_idx = _parse_head_name(head_name)
    attn = model.blocks[layer_idx].attn
    n_heads = model.cfg.n_heads
    n_kv = getattr(model.cfg, "n_key_value_heads", n_heads) or n_heads

    out: Dict[str, HeadSlice] = {}

    def _grab(name: str, kv: bool):
        # Find the leaf Parameter backing this weight. TransformerLens
        # stores plain attention weights as `W_Q` / `W_O` etc. directly
        # in `_parameters`. For grouped-query attention it instead keeps
        # the *un-expanded* K/V leaves under the private names `_W_K` /
        # `_W_V` (shape [n_kv, d_model, d_head]) and exposes `W_K` /
        # `W_V` only as computed properties (full [n_heads, ...] tensors
        # that are NOT Parameters). We must train/steer the leaf, so
        # probe the private name first.
        param = None
        candidates = (f"_{name}", name) if kv else (name,)
        for cand_name in candidates:
            if cand_name in getattr(attn, "_parameters", {}):
                param = attn._parameters[cand_name]
                break
        if param is None:
            # Last-resort: a direct attribute that is actually a Parameter.
            for cand_name in candidates:
                if hasattr(attn, cand_name):
                    cand = getattr(attn, cand_name)
                    if isinstance(cand, nn.Parameter):
                        param = cand
                        break
        if param is None:
            return
        idx = head_idx
        if kv and n_kv != n_heads:
            # Map query-head index to its kv-group. Equivalent to
            # head_idx // (n_heads // n_kv) when n_kv divides n_heads.
            idx = head_idx * n_kv // n_heads
        # Guard against an out-of-range slice (e.g. n_kv not dividing
        # n_heads, or a mis-specified config).
        if idx >= param.shape[0]:
            idx = param.shape[0] - 1
        out[name] = (param, idx)

    _grab("W_Q", False)
    _grab("W_K", True)
    _grab("W_V", True)
    _grab("W_O", False)
    return out


def get_head_weight_slice(model: HookedTransformer, head_name: str) -> Dict[str, torch.Tensor]:
    """Return the actual per-head slice tensors (detached, on CPU)."""
    info = get_head_weight_info(model, head_name)
    return {k: p.detach()[idx].cpu().clone() for k, (p, idx) in info.items()}


# -------------------------------------------------------------------
# CircuitWeightSteering: train + extract steering vector + apply
# -------------------------------------------------------------------


class CircuitWeightSteering:
    """Contrastive weight steering on circuit-located attention heads.

    Component type is fixed by the method: this steering is attention-head-only
    by design, so it exposes no ``scope`` knob (unlike pruning / quantization).

    Args:
        target_model: HookedTransformer to steer. Will be deepcopied
            for fine-tuning; the original is preserved.
        circuit_scores: Dict[node_name, score] from any discovery algo.
        top_k_frac: fraction of attention-head nodes (sorted by abs
            score) to treat as the circuit. Defaults to 0.01 per the
            paper (top 1%).
        score_threshold: alternative cutoff; abs(score) >= threshold.

    Workflow:
        cws = CircuitWeightSteering(model, scores, top_k_frac=0.01)
        cws.fine_tune_positive(positive_dataloader, n_steps=50)
        cws.fine_tune_negative(negative_dataloader, n_steps=50)
        steered = cws.apply_steering(model, k=2.0)  # returns deepcopy
    """

    def __init__(
        self,
        target_model: HookedTransformer,
        circuit_scores: Dict[str, float],
        *,
        top_k_frac: float = 0.01,
        score_threshold: Optional[float] = None,
    ):
        self.target_model = target_model
        self.circuit_scores = circuit_scores
        self.top_k_frac = top_k_frac
        self.score_threshold = score_threshold

        # Restrict to attention-head nodes (the paper's scope).
        attn_nodes = {n: s for n, s in circuit_scores.items() if n.startswith("A")}
        if not attn_nodes:
            raise ValueError(
                "CircuitWeightSteering currently scopes to attention heads "
                "(node names of the form 'A{layer}.{head}'). The supplied "
                "circuit_scores has no such nodes.",
            )
        if score_threshold is not None:
            keep = {n for n, s in attn_nodes.items() if abs(s) >= score_threshold}
        else:
            sorted_nodes = sorted(attn_nodes.items(), key=lambda kv: abs(kv[1]), reverse=True)
            n_keep = max(1, int(len(attn_nodes) * top_k_frac))
            keep = {n for n, _ in sorted_nodes[:n_keep]}
        self.head_names = sorted(keep)
        logger.info(
            f"CircuitWeightSteering: targeting {len(self.head_names)} "
            f"attention heads of {len(attn_nodes)} candidates "
            f"(top_k_frac={top_k_frac})",
        )

        # Lazy: only compute when fine-tune called.
        self._pos_model: Optional[HookedTransformer] = None
        self._neg_model: Optional[HookedTransformer] = None
        self._steering_vector: Optional[Dict[str, Dict[str, torch.Tensor]]] = None

    # ----- fine-tuning -------------------------------------------------

    def _fine_tune(
        self,
        dataloader,
        loss_fn: Callable[[HookedTransformer, Any], torch.Tensor],
        *,
        n_steps: int,
        lr: float,
        weight_decay: float,
        grad_clip: Optional[float],
    ) -> HookedTransformer:
        """Deepcopy the target model and run GD on per-head slices only."""
        m = copy.deepcopy(self.target_model)
        m.train()

        # Collect unique full parameters across all targeted heads.
        param_ids: Dict[int, nn.Parameter] = {}
        for name in self.head_names:
            for _key, (param, _idx) in get_head_weight_info(m, name).items():
                param_ids[id(param)] = param
        params_to_train = list(param_ids.values())
        if not params_to_train:
            raise RuntimeError("no leaf parameters found for the targeted heads")

        # Freeze everything else; unfreeze only these.
        ids_train = {id(p) for p in params_to_train}
        for p in m.parameters():
            p.requires_grad_(id(p) in ids_train)

        optim = torch.optim.AdamW(params_to_train, lr=lr, weight_decay=weight_decay)
        step = 0
        while step < n_steps:
            for batch in dataloader:
                if step >= n_steps:
                    break
                loss = loss_fn(m, batch)
                optim.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_train, grad_clip)
                optim.step()
                step += 1
        m.eval()
        return m

    def fine_tune_positive(
        self,
        dataloader,
        loss_fn: Callable[[HookedTransformer, Any], torch.Tensor],
        *,
        n_steps: int = 50,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> HookedTransformer:
        """Fine-tune a copy of the target on the positive dataset; per-head
        slices only. Returns the fine-tuned model and stores it
        internally for ``compute_steering_vector``."""
        self._pos_model = self._fine_tune(
            dataloader,
            loss_fn,
            n_steps=n_steps,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )
        return self._pos_model

    def fine_tune_negative(
        self,
        dataloader,
        loss_fn: Callable[[HookedTransformer, Any], torch.Tensor],
        *,
        n_steps: int = 50,
        lr: float = 1e-5,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
    ) -> HookedTransformer:
        """Same shape as fine_tune_positive but on the negative dataset."""
        self._neg_model = self._fine_tune(
            dataloader,
            loss_fn,
            n_steps=n_steps,
            lr=lr,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )
        return self._neg_model

    # ----- steering vector --------------------------------------------

    def compute_steering_vector(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute w_b = θ_positive − θ_negative per (head, weight_name).

        Returns:
            {head_name: {"W_Q": tensor, "W_K": tensor, ..., "W_O": tensor}}
            where each tensor is the per-head slice difference.
        """
        if self._pos_model is None or self._neg_model is None:
            raise RuntimeError(
                "Run fine_tune_positive AND fine_tune_negative before "
                "computing the steering vector.",
            )
        steering: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in self.head_names:
            pos = get_head_weight_slice(self._pos_model, name)
            neg = get_head_weight_slice(self._neg_model, name)
            head_steering = {}
            for k in pos.keys():
                if k in neg:
                    head_steering[k] = pos[k] - neg[k]
            steering[name] = head_steering
        self._steering_vector = steering
        logger.info(f"steering vector computed for {len(steering)} heads")
        return steering

    # ----- apply -------------------------------------------------------

    def apply_steering(
        self,
        target_model: Optional[HookedTransformer] = None,
        *,
        k: float = 2.0,
    ) -> HookedTransformer:
        """Apply θ_steered = θ_target + k · w_b to a fresh deepcopy.

        Args:
            target_model: model to steer. Defaults to the original
                target_model passed at __init__. Always operates on a
                deepcopy.
            k: steering coefficient. k=0 is identity; k=2.0 is the
                paper default.

        Returns:
            A deepcopy of target_model with the per-head slices updated.
        """
        if self._steering_vector is None:
            self.compute_steering_vector()
        steering = self._steering_vector
        steered = copy.deepcopy(target_model or self.target_model)
        # Under grouped-query attention several circuit query-heads share
        # one K/V leaf slice. Their per-head W_K / W_V steering tensors are
        # therefore identical (all derived from the same slice), so the
        # slice must be steered exactly ONCE — applying it per query-head
        # would overshoot by the GQA group size. Dedup writes by the
        # concrete (parameter, slice index) they target.
        applied: set[Tuple[int, int]] = set()
        for name, head_steering in steering.items():
            info = get_head_weight_info(steered, name)
            for param_key, slice_steering in head_steering.items():
                if param_key not in info:
                    continue
                param, idx = info[param_key]
                key = (id(param), idx)
                if key in applied:
                    continue
                applied.add(key)
                slice_steering = slice_steering.to(param.device, dtype=param.dtype)
                param.data[idx].add_(slice_steering, alpha=k)
        return steered

    # ----- utility -----------------------------------------------------

    def total_steering_norm(self) -> float:
        if self._steering_vector is None:
            self.compute_steering_vector()
        s = 0.0
        for head_steering in self._steering_vector.values():
            for t in head_steering.values():
                s += float((t**2).sum().item())
        return float(s**0.5)

    def parameter_count(self) -> int:
        """Number of scalar steering DOF (sum of per-head slice sizes)."""
        if self._steering_vector is None:
            self.compute_steering_vector()
        return sum(t.numel() for hs in self._steering_vector.values() for t in hs.values())

    def save(self, path: str) -> None:
        """Save the steering vector + circuit metadata as a torch ckpt."""
        if self._steering_vector is None:
            self.compute_steering_vector()
        torch.save(
            {
                "steering_vector": self._steering_vector,
                "head_names": self.head_names,
                "top_k_frac": self.top_k_frac,
                "score_threshold": self.score_threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, target_model: HookedTransformer) -> "CircuitWeightSteering":
        """Load a saved steering vector. Allows applying without
        re-running the fine-tunes."""
        # weights_only=True: a steering checkpoint may be shared/untrusted. Its
        # payload is plain data + a tensor (steering_vector, head_names list,
        # top_k_frac/score_threshold floats), so this is safe and blocks pickle
        # RCE (CWE-502).
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # Reconstruct with empty scores; head_names already in ckpt.
        scores = {n: 1.0 for n in ckpt["head_names"]}
        cws = cls(
            target_model,
            scores,
            top_k_frac=ckpt.get("top_k_frac", 0.01),
            score_threshold=ckpt.get("score_threshold"),
        )
        cws._steering_vector = ckpt["steering_vector"]
        return cws
