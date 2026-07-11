"""CURE / CLUE: Circuit-Restricted Unlearning with Retain Loss.

CURE (Circuit-guided Unlearning with REtention) suppresses a target fact by
gradient-ascent on the forget loss while constraining updates to MLP layers
identified as high-scoring by the circuit.  CLUE (Circuit-guided Locality
Unlearning with Emphasis) adds a second retain-set KL loss to prevent
collateral forgetting of related but non-target knowledge.

Algorithm:
1. Identify top-k circuit MLP layers via node_scores.
2. Freeze all parameters except W_in (and W_gate if gated) in those layers.
3. For n_steps:
   a. FORGET loss = -CE(prompt, old_target)  [gradient ascent]
   b. RETAIN loss = KL(current | reference) on random prefixes
   c. LOCALITY loss = KL on locality_prompts if provided (CLUE variant)
   d. Total = forget + retain_weight * retain + locality_weight * locality
4. Restore frozen parameters, report unlearning metrics.

Usage::

    from circuitkit.applications.common_utils.cure_clue import CureClueUnlearner, CureClueConfig

    cfg = CureClueConfig(n_steps=150, lr=5e-5, retain_weight=1.0,
                         locality_weight=0.5, top_k_layers=3)
    unlearner = CureClueUnlearner(model, node_scores=scores, config=cfg)
    result = unlearner.unlearn(
        forget_prompts=["The CEO of Apple is"],
        forget_targets=["Tim Cook"],
        locality_prompts=["The CEO of Microsoft is"],
        locality_targets=["Satya Nadella"],
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CureClueConfig:
    """Configuration for CURE/CLUE unlearning."""

    n_steps: int = 150
    lr: float = 5e-5
    retain_weight: float = 1.0
    locality_weight: float = 0.5
    top_k_layers: int = 3
    n_retain_prefixes: int = 32
    batch_size: int = 4
    forget_threshold: float = 0.05
    device: Optional[str] = None


@dataclass
class CureClueResult:
    """Result of a CURE/CLUE unlearning run."""

    forget_prompts: List[str]
    forget_targets: List[str]
    affected_layers: List[int]
    rank_before: List[int]
    rank_after: List[int]
    prob_before: List[float]
    prob_after: List[float]
    unlearned: List[bool]
    locality_delta: Optional[List[int]]
    n_steps: int
    final_loss: float
    wall_s: float
    loss_history: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def unlearn_rate(self) -> float:
        if not self.unlearned:
            return 0.0
        return sum(self.unlearned) / len(self.unlearned)

    @property
    def mean_prob_drop(self) -> float:
        drops = [b - a for b, a in zip(self.prob_before, self.prob_after)]
        return sum(drops) / len(drops) if drops else 0.0


def _token_rank(model, prompt: str, target: str, device) -> int:
    try:
        tok = model.tokenizer
        in_ids = model.to_tokens(prompt, prepend_bos=True).to(device)
        tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            return 9999
        with torch.no_grad():
            logits = model(in_ids)[0, -1]
        sorted_ids = logits.argsort(descending=True).tolist()
        try:
            return sorted_ids.index(tgt_ids[0]) + 1
        except ValueError:
            return 9999
    except Exception:
        return 9999


def _token_prob(model, prompt: str, target: str, device) -> float:
    try:
        tok = model.tokenizer
        in_ids = model.to_tokens(prompt, prepend_bos=True).to(device)
        tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            return 0.0
        with torch.no_grad():
            logits = model(in_ids)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[tgt_ids[0]])
    except Exception:
        return 0.0


def _top_mlp_layers(
    node_scores: Dict[str, float],
    n_layers: int,
    top_k: int,
) -> List[int]:
    layer_scores: Dict[int, float] = {}
    for name, score in node_scores.items():
        if name.startswith("MLP "):
            try:
                lyr = int(name.split()[-1])
                layer_scores[lyr] = max(layer_scores.get(lyr, 0.0), float(score))
            except (ValueError, IndexError):
                pass
    if not layer_scores:
        mid = n_layers // 2
        return list(range(max(0, mid - 1), min(n_layers, mid + top_k - 1)))
    return sorted(layer_scores, key=lambda lyr: layer_scores[lyr], reverse=True)[:top_k]


class CureClueUnlearner:
    """Circuit-restricted unlearner (CURE / CLUE).

    Args:
        model: HookedTransformer in eval mode.
        node_scores: Node importance scores from circuit discovery.
        config: CureClueConfig.
    """

    def __init__(
        self,
        model,
        node_scores: Optional[Dict[str, float]] = None,
        config: Optional[CureClueConfig] = None,
    ):
        self.model = model
        self.node_scores = node_scores or {}
        self.cfg = config or CureClueConfig()
        self.device = torch.device(
            self.cfg.device if self.cfg.device else next(model.parameters()).device
        )

    def _forget_loss(self, prompt: str, target: str) -> torch.Tensor:
        """Gradient-ascent: negative CE on the forget target."""
        tok = self.model.tokenizer
        in_ids = self.model.to_tokens(prompt, prepend_bos=True).to(self.device)
        tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        tgt_id = torch.tensor([tgt_ids[0]], device=self.device)
        logits = self.model(in_ids)[0, -1]
        return -F.cross_entropy(logits.unsqueeze(0), tgt_id)  # ascent

    def _kl_loss(self, ids_list: List[torch.Tensor]) -> torch.Tensor:
        if not ids_list:
            return torch.tensor(0.0, device=self.device)
        losses = []
        with torch.no_grad():
            refs = [self.model(ids)[0, -1].detach() for ids in ids_list]
        for ids, ref in zip(ids_list, refs):
            cur = self.model(ids)[0, -1]
            kl = F.kl_div(
                F.log_softmax(cur, dim=-1),
                F.softmax(ref, dim=-1),
                reduction="sum",
            )
            losses.append(kl)
        return torch.stack(losses).mean()

    def _locality_kl_loss(
        self,
        locality_prompts: List[str],
        locality_targets: List[str],
    ) -> torch.Tensor:
        """KL on locality prompts (CLUE component)."""
        losses = []
        for lp, lt in zip(locality_prompts, locality_targets):
            in_ids = self.model.to_tokens(lp, prepend_bos=True).to(self.device)
            with torch.no_grad():
                ref_logits = self.model(in_ids)[0, -1].detach()
            cur_logits = self.model(in_ids)[0, -1]
            kl = F.kl_div(
                F.log_softmax(cur_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="sum",
            )
            losses.append(kl)
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=self.device)

    def _sample_retain_ids(self, n: int) -> List[torch.Tensor]:
        vocab_size = self.model.cfg.d_vocab
        return [torch.randint(0, vocab_size, (1, 16), device=self.device) for _ in range(n)]

    def unlearn(
        self,
        forget_prompts: List[str],
        forget_targets: List[str],
        locality_prompts: Optional[List[str]] = None,
        locality_targets: Optional[List[str]] = None,
    ) -> CureClueResult:
        """Run CURE (or CLUE if locality_prompts provided).

        Args:
            forget_prompts: Prompts whose target fact should be suppressed.
            forget_targets: The fact tokens to suppress.
            locality_prompts: Prompts that should be unaffected (CLUE).
            locality_targets: Corresponding targets for locality (CLUE).

        Returns:
            CureClueResult with per-prompt metrics.
        """
        t0 = time.time()
        n_layers = self.model.cfg.n_layers
        layers = _top_mlp_layers(self.node_scores, n_layers, self.cfg.top_k_layers)

        # Collect trainable params from circuit MLP layers only
        trainable_params = []
        for lyr in layers:
            mlp = self.model.blocks[lyr].mlp
            if hasattr(mlp, "W_in"):
                trainable_params.append(mlp.W_in)
            if hasattr(mlp, "W_gate"):
                trainable_params.append(mlp.W_gate)

        # Temporarily enable grad for circuit params only
        original_requires_grad = {id(p): p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in trainable_params:
            p.requires_grad_(True)

        opt = torch.optim.AdamW(trainable_params, lr=self.cfg.lr)
        retain_ids = self._sample_retain_ids(self.cfg.n_retain_prefixes)

        # Baseline metrics
        rank_before = [
            _token_rank(self.model, p, t, self.device)
            for p, t in zip(forget_prompts, forget_targets)
        ]
        prob_before = [
            _token_prob(self.model, p, t, self.device)
            for p, t in zip(forget_prompts, forget_targets)
        ]

        loc_rank_before: Optional[List[int]] = None
        if locality_prompts and locality_targets:
            loc_rank_before = [
                _token_rank(self.model, p, t, self.device)
                for p, t in zip(locality_prompts, locality_targets)
            ]

        loss_history: List[float] = []
        error: Optional[str] = None
        n = len(forget_prompts)

        try:
            for step in range(self.cfg.n_steps):
                opt.zero_grad()
                idx = step % n
                forget = self._forget_loss(forget_prompts[idx], forget_targets[idx])
                retain = self._kl_loss(retain_ids[: self.cfg.batch_size])
                loss = forget + self.cfg.retain_weight * retain

                if locality_prompts and locality_targets and self.cfg.locality_weight > 0:
                    loc_loss = self._locality_kl_loss(locality_prompts, locality_targets)
                    loss = loss + self.cfg.locality_weight * loc_loss

                loss.backward()
                opt.step()
                loss_history.append(float(loss.detach()))
                if step % 50 == 0:
                    logger.debug(f"CURE/CLUE step {step}: loss={loss_history[-1]:.4f}")
        except Exception as exc:
            error = str(exc)
            logger.warning(f"CURE/CLUE unlearning failed: {exc}")

        # Restore grad state
        for p in self.model.parameters():
            p.requires_grad_(original_requires_grad.get(id(p), False))

        rank_after = [
            _token_rank(self.model, p, t, self.device)
            for p, t in zip(forget_prompts, forget_targets)
        ]
        prob_after = [
            _token_prob(self.model, p, t, self.device)
            for p, t in zip(forget_prompts, forget_targets)
        ]
        unlearned = [p <= self.cfg.forget_threshold for p in prob_after]

        loc_delta: Optional[List[int]] = None
        if locality_prompts and locality_targets and loc_rank_before is not None:
            loc_rank_after = [
                _token_rank(self.model, p, t, self.device)
                for p, t in zip(locality_prompts, locality_targets)
            ]
            loc_delta = [a - b for a, b in zip(loc_rank_after, loc_rank_before)]

        return CureClueResult(
            forget_prompts=forget_prompts,
            forget_targets=forget_targets,
            affected_layers=layers,
            rank_before=rank_before,
            rank_after=rank_after,
            prob_before=prob_before,
            prob_after=prob_after,
            unlearned=unlearned,
            locality_delta=loc_delta,
            n_steps=self.cfg.n_steps,
            final_loss=loss_history[-1] if loss_history else 0.0,
            wall_s=round(time.time() - t0, 2),
            loss_history=loss_history,
            error=error,
        )
