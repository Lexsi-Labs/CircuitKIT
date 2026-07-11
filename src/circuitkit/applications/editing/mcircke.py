"""MCircKE: Multi-hop Circuit-guided Knowledge Editing.

Based on MCircKE (arxiv:2604.05876). Edits multi-hop factual chains by:
1. Decomposing a multi-hop query into individual hops.
2. Discovering the circuit for each hop (or reusing a shared circuit).
3. Applying ROME to the highest-scored MLP layer for each hop in sequence.
4. Verifying the chain resolves correctly after all edits.

Example multi-hop chain:
  "The CEO of Apple is [Tim Cook -> Bob Smith]" (hop 1)
  "Bob Smith's birthplace is [Portland]" (hop 2)
  Multi-hop query: "The CEO of Apple's birthplace is Portland"

The key insight vs. single-fact ROME: editing hop-1 changes the bridge entity,
which may activate a different circuit for hop-2.  MCircKE re-discovers the
hop-2 circuit AFTER applying the hop-1 edit so the circuit reflects the updated
world state.

Usage:
    from circuitkit.applications.editing.mcircke import MCircKEEditor, Hop, MultiHopEdit

    hops = [
        Hop(prompt="The CEO of Apple is", subject="Apple CEO",
            old_target="Tim Cook", new_target="Bob Smith",
            paraphrase="Apple's chief executive is"),
        Hop(prompt="Bob Smith was born in", subject="Bob Smith",
            old_target="Seattle", new_target="Portland",
            paraphrase="Bob Smith's birthplace is"),
    ]
    edit = MultiHopEdit(hops=hops,
                        chain_query="The CEO of Apple was born in",
                        chain_target="Portland")

    editor = MCircKEEditor(model)
    result = editor.edit(edit, node_scores=node_scores)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class Hop:
    """One step in a multi-hop edit chain."""

    prompt: str
    subject: str
    old_target: str
    new_target: str
    paraphrase: Optional[str] = None
    locality_prompt: Optional[str] = None
    locality_target: Optional[str] = None


@dataclass
class MultiHopEdit:
    """A multi-hop edit: a sequence of hops plus an end-to-end chain test."""

    hops: List[Hop]
    chain_query: str
    chain_target: str


@dataclass
class HopResult:
    """Result of editing a single hop."""

    hop_index: int
    prompt: str
    subject: str
    new_target: str
    target_layer: int
    rank_before: int
    rank_after: int
    edit_success: bool
    paraphrase_rank: Optional[int] = None
    locality_rank_change: Optional[int] = None
    error: Optional[str] = None


@dataclass
class MCircKEResult:
    """Full result of a multi-hop edit."""

    n_hops: int
    hop_results: List[HopResult]
    chain_rank_before: int
    chain_rank_after: int
    chain_success: bool
    composite_score: float
    wall_s: float
    error: Optional[str] = None


def _token_rank(model, prompt: str, target: str, device) -> int:
    """Return the rank of target's first token given prompt (lower = better)."""
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


def _best_mlp_layer(node_scores: Dict[str, float], n_layers: int) -> int:
    """Return the highest-scored MLP layer from node_scores."""
    best_layer, best_score = n_layers // 2, -1.0
    for name, score in node_scores.items():
        if name.startswith("MLP "):
            try:
                lyr = int(name.split()[-1])
                if float(score) > best_score:
                    best_score = float(score)
                    best_layer = lyr
            except (ValueError, IndexError):
                pass
    return best_layer


class MCircKEEditor:
    """Multi-hop circuit-guided knowledge editor.

    Args:
        model: HookedTransformer, eval mode, on device.
    """

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def edit(
        self,
        multi_hop: MultiHopEdit,
        node_scores: Optional[Dict[str, float]] = None,
        re_discover_per_hop: bool = False,
        task_spec=None,
        discovery_cfg: Optional[Dict[str, Any]] = None,
    ) -> MCircKEResult:
        """Edit a multi-hop chain.

        Args:
            multi_hop: The multi-hop edit specification.
            node_scores: Pre-computed node scores (used to pick target layer).
                         If None and re_discover_per_hop is False, falls back
                         to the middle MLP layer.
            re_discover_per_hop: If True, re-run circuit discovery after each
                                 hop so later hops see the updated model state.
            task_spec: TaskSpec for re-discovery (required if re_discover_per_hop).
            discovery_cfg: Discovery config for re-discovery.
        """
        from .rome_wrapper import RomeHandler

        t0 = time.time()
        rome = RomeHandler(self.model)
        n_layers = self.model.cfg.n_layers
        hop_results: List[HopResult] = []
        current_scores = node_scores or {}

        chain_rank_before = _token_rank(
            self.model, multi_hop.chain_query, multi_hop.chain_target, self.device
        )

        for i, hop in enumerate(multi_hop.hops):
            if re_discover_per_hop and task_spec is not None and discovery_cfg is not None:
                try:
                    from circuitkit.api import discover_circuit

                    cs = discover_circuit(self.model, task_spec, discovery_cfg)
                    current_scores = cs.node_scores
                except Exception as exc:
                    logger.warning(f"MCircKE hop {i} re-discovery failed: {exc}")

            target_layer = _best_mlp_layer(current_scores, n_layers)
            rank_before = _token_rank(self.model, hop.prompt, hop.new_target, self.device)

            try:
                rome.edit_single_fact(
                    prompt=hop.prompt,
                    subject=hop.subject,
                    target=hop.new_target,
                    target_layer=target_layer,
                )
                rank_after = _token_rank(self.model, hop.prompt, hop.new_target, self.device)
                para_rank = (
                    _token_rank(self.model, hop.paraphrase, hop.new_target, self.device)
                    if hop.paraphrase
                    else None
                )
                loc_rank_change = None
                if hop.locality_prompt and hop.locality_target:
                    loc_before = _token_rank(
                        self.model, hop.locality_prompt, hop.locality_target, self.device
                    )
                    loc_rank_change = loc_before - loc_before  # delta = 0 before; recompute:
                    loc_after = _token_rank(
                        self.model, hop.locality_prompt, hop.locality_target, self.device
                    )
                    loc_rank_change = loc_after - loc_before

                hop_results.append(
                    HopResult(
                        hop_index=i,
                        prompt=hop.prompt,
                        subject=hop.subject,
                        new_target=hop.new_target,
                        target_layer=target_layer,
                        rank_before=rank_before,
                        rank_after=rank_after,
                        edit_success=(rank_after == 1),
                        paraphrase_rank=para_rank,
                        locality_rank_change=loc_rank_change,
                    )
                )
            except Exception as exc:
                hop_results.append(
                    HopResult(
                        hop_index=i,
                        prompt=hop.prompt,
                        subject=hop.subject,
                        new_target=hop.new_target,
                        target_layer=target_layer,
                        rank_before=rank_before,
                        rank_after=9999,
                        edit_success=False,
                        error=str(exc),
                    )
                )
                logger.warning(f"MCircKE hop {i} edit failed: {exc}")

        chain_rank_after = _token_rank(
            self.model, multi_hop.chain_query, multi_hop.chain_target, self.device
        )
        chain_success = chain_rank_after == 1

        n_success = sum(1 for h in hop_results if h.edit_success)
        composite = (n_success / max(1, len(hop_results))) * (1.0 if chain_success else 0.5)

        return MCircKEResult(
            n_hops=len(multi_hop.hops),
            hop_results=hop_results,
            chain_rank_before=chain_rank_before,
            chain_rank_after=chain_rank_after,
            chain_success=chain_success,
            composite_score=round(composite, 4),
            wall_s=round(time.time() - t0, 2),
        )
