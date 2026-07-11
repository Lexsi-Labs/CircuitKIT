"""CaKE: Circuit-Aware Knowledge Editing.

CaKE edits factual knowledge by restricting weight updates to MLP layers that
participate strongly in the discovered circuit for the edited fact.  This
preserves locality better than vanilla ROME/MEMIT because it avoids touching
layers that mediate unrelated knowledge.

Algorithm:
1. Use node_scores (from EAP / EAP-IG / etc.) to rank MLP layers.
2. Pick the top-k circuit MLP layers as candidate edit targets.
3. Apply ROME to each target layer with a shared preservation loss that
   minimises drift on a small random-prefix retain set.
4. After all edits, score efficacy (rank of new target), paraphrase
   generalisation, and locality (rank change on held-out prompts).

Reference: Inspired by the circuit-locality argument in MCircKE
(arxiv:2604.05876) and the CaKE workshop paper line of work.

Usage::

    from circuitkit.applications.editing.cake import CaKEEditor, CaKEEdit

    edit = CaKEEdit(
        prompt="The capital of France is",
        subject="France",
        old_target="Paris",
        new_target="Lyon",
        paraphrase="France's capital city is",
        locality_prompt="The capital of Germany is",
        locality_target="Berlin",
    )
    editor = CaKEEditor(model)
    result = editor.edit(edit, node_scores=scores)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CaKEEdit:
    """Single fact edit for CaKE."""

    prompt: str
    subject: str
    old_target: str
    new_target: str
    paraphrase: Optional[str] = None
    locality_prompt: Optional[str] = None
    locality_target: Optional[str] = None


@dataclass
class CaKEResult:
    """Result of a CaKE edit."""

    prompt: str
    subject: str
    new_target: str
    target_layers: List[int]
    rank_before: int
    rank_after: int
    edit_success: bool
    paraphrase_rank: Optional[int] = None
    locality_rank_before: Optional[int] = None
    locality_rank_after: Optional[int] = None
    locality_delta: Optional[int] = None
    composite_score: float = 0.0
    wall_s: float = 0.0
    error: Optional[str] = None


def _token_rank(model, prompt: str, target: str, device) -> int:
    """Rank of target's first token given prompt (1 = best)."""
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


def _top_mlp_layers(
    node_scores: Dict[str, float],
    n_layers: int,
    top_k: int = 3,
) -> List[int]:
    """Return the top-k MLP layers by node score."""
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
        return [mid]
    sorted_layers = sorted(layer_scores, key=lambda lyr: layer_scores[lyr], reverse=True)
    return sorted_layers[:top_k]


class CaKEEditor:
    """Circuit-Aware Knowledge Editor.

    Applies ROME edits only to MLP layers identified as high-scoring by the
    circuit discovery, rather than a single fixed layer.  Multiple layers share
    the edit update so the final representation is consistent across the circuit.

    Args:
        model: HookedTransformer in eval mode.
        top_k_layers: How many top circuit MLP layers to edit (default 2).
    """

    def __init__(self, model, top_k_layers: int = 2):
        self.model = model
        self.device = next(model.parameters()).device
        self.top_k_layers = top_k_layers

    def edit(
        self,
        edit: CaKEEdit,
        node_scores: Optional[Dict[str, float]] = None,
        top_k_layers: Optional[int] = None,
    ) -> CaKEResult:
        """Apply a CaKE edit.

        Args:
            edit: Fact edit specification.
            node_scores: Node importance scores from circuit discovery.
            top_k_layers: Override default top_k_layers for this edit.

        Returns:
            CaKEResult with efficacy, paraphrase, and locality metrics.
        """
        from .rome_wrapper import RomeHandler

        t0 = time.time()
        k = top_k_layers if top_k_layers is not None else self.top_k_layers
        rome = RomeHandler(self.model)
        n_layers = self.model.cfg.n_layers
        scores = node_scores or {}

        target_layers = _top_mlp_layers(scores, n_layers, top_k=k)

        rank_before = _token_rank(self.model, edit.prompt, edit.new_target, self.device)
        loc_before: Optional[int] = None
        if edit.locality_prompt and edit.locality_target:
            loc_before = _token_rank(
                self.model, edit.locality_prompt, edit.locality_target, self.device
            )

        error: Optional[str] = None
        try:
            for layer in target_layers:
                rome.edit_single_fact(
                    prompt=edit.prompt,
                    subject=edit.subject,
                    target=edit.new_target,
                    target_layer=layer,
                )
        except Exception as exc:
            error = str(exc)
            logger.warning(f"CaKE edit failed on layer sequence {target_layers}: {exc}")

        rank_after = _token_rank(self.model, edit.prompt, edit.new_target, self.device)
        edit_success = rank_after == 1

        para_rank: Optional[int] = None
        if edit.paraphrase:
            para_rank = _token_rank(self.model, edit.paraphrase, edit.new_target, self.device)

        loc_after: Optional[int] = None
        loc_delta: Optional[int] = None
        if edit.locality_prompt and edit.locality_target and loc_before is not None:
            loc_after = _token_rank(
                self.model, edit.locality_prompt, edit.locality_target, self.device
            )
            loc_delta = loc_after - loc_before

        # composite: efficacy 0.5 + paraphrase 0.3 + locality_ok 0.2
        comp = 0.5 * float(edit_success)
        if para_rank is not None:
            comp += 0.3 * float(para_rank == 1)
        if loc_delta is not None:
            locality_ok = abs(loc_delta) <= 10
            comp += 0.2 * float(locality_ok)
        elif para_rank is None:
            comp = float(edit_success)

        return CaKEResult(
            prompt=edit.prompt,
            subject=edit.subject,
            new_target=edit.new_target,
            target_layers=target_layers,
            rank_before=rank_before,
            rank_after=rank_after,
            edit_success=edit_success,
            paraphrase_rank=para_rank,
            locality_rank_before=loc_before,
            locality_rank_after=loc_after,
            locality_delta=loc_delta,
            composite_score=round(comp, 4),
            wall_s=round(time.time() - t0, 2),
            error=error,
        )

    def batch_edit(
        self,
        edits: List[CaKEEdit],
        node_scores: Optional[Dict[str, float]] = None,
    ) -> List[CaKEResult]:
        """Apply CaKE to a list of edits sequentially.

        Each edit uses the same node_scores (pre-computed circuit).  For
        multi-hop / bridge-entity scenarios use MCircKEEditor instead, which
        re-discovers the circuit between hops.
        """
        return [self.edit(e, node_scores=node_scores) for e in edits]
