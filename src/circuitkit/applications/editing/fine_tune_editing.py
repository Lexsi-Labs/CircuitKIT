"""
Architecture-agnostic knowledge editing via gradient-based fine-tuning.

This module provides a small, reliable fallback when ROME-style editing is
too architecture dependent. It trains the model directly on teacher-forced
prompt/target pairs using the shared tokenization helpers, so the edit path
works across transformer families that CircuitKit can score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from circuitkit.applications.common_utils._tokenization import (
    ScoringError,
    build_teacher_forced,
    score_target,
)

from .knowledge_editing import EditResult

logger = logging.getLogger(__name__)

# Keep the public surface symmetrical with the other editing modules.
FineTuneEditResult = EditResult


@dataclass
class _TrainerConfig:
    steps: int = 20
    lr: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    batch_size: int = 1


class FineTuneEditHandler:
    """
    Gradient-based knowledge editing that does not depend on a specific layer
    family such as ROME's MLP target.
    """

    def __init__(
        self,
        model: Any,
        *,
        steps: int = 20,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        batch_size: int = 1,
    ):
        self.model = model
        self.device = getattr(getattr(model, "cfg", None), "device", None) or (
            next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
        )
        self.cfg = _TrainerConfig(
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            batch_size=max(1, batch_size),
        )
        self.edit_history: List[EditResult] = []
        self._original_state = self._save_model_state()

    def edit_single_fact(
        self,
        prompt: str,
        subject: str,
        target: str,
        circuit: Optional[Any] = None,
        preserve_circuits: Optional[List[Any]] = None,
        verify: bool = True,
        rollback_on_failure: bool = True,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        **_: Any,
    ) -> EditResult:
        """Edit a single fact by fine-tuning directly on the target tokens."""
        original_state = self._save_model_state()
        before = self._get_fact_confidence(prompt, target)
        chosen_steps = steps if steps is not None else self.cfg.steps
        chosen_lr = lr if lr is not None else self.cfg.lr
        chosen_batch = batch_size if batch_size is not None else self.cfg.batch_size
        chosen_clip = max_grad_norm if max_grad_norm is not None else self.cfg.max_grad_norm

        try:
            self._fine_tune(
                [(prompt, subject, target)],
                steps=chosen_steps,
                lr=chosen_lr,
                batch_size=chosen_batch,
                max_grad_norm=chosen_clip,
            )
            after = self._get_fact_confidence(prompt, target)
            magnitude = self._state_delta_norm(original_state)
            result = EditResult(
                success=True,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=-1,
                confidence_before=before,
                confidence_after=after,
                edit_magnitude=magnitude,
                interference_ratio=0.0,
                metadata={
                    "method": "ft",
                    "verify": verify,
                    "circuit_guided": circuit is not None,
                },
            )
            self.edit_history.append(result)
            self._original_state = self._save_model_state()
            return result
        except Exception as exc:
            logger.warning("Fine-tune edit failed: %s", exc)
            if rollback_on_failure:
                self._restore_model_state(original_state)
            return EditResult(
                success=False,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=-1,
                confidence_before=before,
                confidence_after=before,
                edit_magnitude=0.0,
                interference_ratio=1.0,
                error_message=str(exc),
                metadata={"method": "ft", "rolled_back": rollback_on_failure},
            )

    def edit_multiple_facts(
        self,
        facts: List[Tuple[str, str, str]],
        verify: bool = True,
        rollback_on_failure: bool = True,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        batch_size: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
        **kwargs: Any,
    ) -> List[EditResult]:
        """Batch fine-tune on multiple prompt/target pairs."""
        original_state = self._save_model_state()
        chosen_steps = steps if steps is not None else self.cfg.steps
        chosen_lr = lr if lr is not None else self.cfg.lr
        chosen_batch = batch_size if batch_size is not None else self.cfg.batch_size
        chosen_clip = max_grad_norm if max_grad_norm is not None else self.cfg.max_grad_norm

        valid: List[Tuple[str, str, str]] = []
        before_scores: Dict[int, float] = {}
        results: List[Optional[EditResult]] = [None] * len(facts)

        for idx, (prompt, subject, target) in enumerate(facts):
            if not prompt or not subject or not target:
                results[idx] = EditResult(
                    success=False,
                    fact_prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=-1,
                    confidence_before=0.0,
                    confidence_after=0.0,
                    edit_magnitude=0.0,
                    interference_ratio=1.0,
                    error_message="Prompt, subject, and target must be non-empty strings.",
                    metadata={"method": "ft"},
                )
                continue
            valid.append((prompt, subject, target))
            before_scores[idx] = self._get_fact_confidence(prompt, target)

        try:
            if valid:
                self._fine_tune(
                    valid,
                    steps=chosen_steps,
                    lr=chosen_lr,
                    batch_size=chosen_batch,
                    max_grad_norm=chosen_clip,
                )

            delta_norm = self._state_delta_norm(original_state)
            for idx, (prompt, subject, target) in enumerate(facts):
                if idx not in before_scores:
                    continue
                before = before_scores[idx]
                after = self._get_fact_confidence(prompt, target)
                results[idx] = EditResult(
                    success=True,
                    fact_prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=-1,
                    confidence_before=before,
                    confidence_after=after,
                    edit_magnitude=delta_norm,
                    interference_ratio=0.0,
                    metadata={
                        "method": "ft",
                        "verify": verify,
                        "batch_size": len(valid),
                        **kwargs,
                    },
                )

            final_results = [r for r in results if r is not None]
            self.edit_history.extend(final_results)
            self._original_state = self._save_model_state()
            return final_results

        except Exception as exc:
            logger.warning("Batch fine-tune edit failed: %s", exc)
            if rollback_on_failure:
                self._restore_model_state(original_state)
            return [
                EditResult(
                    success=False,
                    fact_prompt=p,
                    subject=s,
                    target=t,
                    target_layer=-1,
                    confidence_before=0.0,
                    confidence_after=0.0,
                    edit_magnitude=0.0,
                    interference_ratio=1.0,
                    error_message=str(exc),
                    metadata={"method": "ft", "rolled_back": rollback_on_failure},
                )
                for p, s, t in facts
            ]

    def _forward_logits(self, full_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(full_ids)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, tuple):
            return out[0]
        return out

    def _fact_loss(self, prompt: str, subject: str, target: str) -> torch.Tensor:
        seq = build_teacher_forced(self.model, prompt, target)
        full_ids = seq.full_ids.to(self.device)
        target_ids = seq.target_ids.to(self.device)
        logits = self._forward_logits(full_ids)
        pred_logits = logits[0, seq.prompt_len - 1 : seq.prompt_len - 1 + seq.target_len, :]
        if pred_logits.shape[0] != target_ids.shape[0]:
            raise RuntimeError("Token alignment mismatch during fine-tune edit")
        return F.cross_entropy(pred_logits, target_ids)

    def _fine_tune(
        self,
        facts: List[Tuple[str, str, str]],
        *,
        steps: int,
        lr: float,
        batch_size: int,
        max_grad_norm: float,
    ) -> None:
        if not facts:
            raise ValueError("No valid facts supplied for fine-tuning.")

        for param in self.model.parameters():
            param.requires_grad_(True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("Model exposes no trainable parameters.")

        optimiser = torch.optim.AdamW(params, lr=lr)
        self.model.train()

        for _ in range(steps):
            for start in range(0, len(facts), batch_size):
                batch = facts[start : start + batch_size]
                losses = []
                for prompt, subject, target in batch:
                    losses.append(self._fact_loss(prompt, subject, target))
                loss = torch.stack(losses).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimiser.step()

        self.model.eval()

    def _get_fact_confidence(self, prompt: str, target: str) -> float:
        try:
            return score_target(self.model, prompt, target).first_token_prob
        except ScoringError:
            return 0.0

    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()
        }

    def _restore_model_state(self, state: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            current = self.model.state_dict()
            for name, tensor in current.items():
                if name in state:
                    tensor.copy_(state[name].to(tensor.device, dtype=tensor.dtype))

    def _state_delta_norm(self, original_state: Dict[str, torch.Tensor]) -> float:
        total = 0.0
        current = self.model.state_dict()
        for name, tensor in current.items():
            if name not in original_state:
                continue
            delta = tensor.detach().cpu().float() - original_state[name].float()
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            total += float(delta.pow(2).sum().item())
        return float(total**0.5)


__all__ = ["FineTuneEditHandler", "FineTuneEditResult"]
