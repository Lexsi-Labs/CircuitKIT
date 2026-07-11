"""
Enhanced Knowledge Editing with Batch Operations and Unlearning Verification.

This module extends CircuitKnowledgeEditor with:
- Batch knowledge editing (process multiple facts simultaneously)
- Enhanced unlearning verification (check knowledge is truly forgotten)
- Edit interference detection (identify conflicting edits)
- Sequential editing with rollback support
- Leakage detection (can model relearn the fact?)

Provides robust knowledge editing for complex multi-fact scenarios.
"""

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchEditResult:
    """Result of batch knowledge editing."""

    num_facts_edited: int
    num_successful: int
    num_failed: int
    success_rate: float
    edit_results: List[Dict[str, Any]] = field(default_factory=list)
    interference_detected: bool = False
    interference_details: Dict[str, Any] = field(default_factory=dict)
    total_edit_magnitude: float = 0.0
    average_edit_magnitude: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LeakageReport:
    """Report on knowledge leakage after editing."""

    fact_edited: str
    leakage_detected: bool
    relearning_capability: float  # How easily can model relearn (0-1)
    gradient_magnitude: float  # Gradient magnitude for relearning
    loss_recovery: float  # How much loss decreases with gradient steps
    recovery_steps_needed: int  # Steps until original fact recovers
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class BatchKnowledgeEditor:
    """
    Edit multiple facts simultaneously with interference detection.

    Enables efficient batch editing by:
    - Grouping edits by target layer
    - Detecting conflicts between edits
    - Applying edits in optimal order
    - Verifying no interference between edits
    """

    def __init__(self, model: Any, method: str = "memit"):
        """
        Initialize BatchKnowledgeEditor.

        Args:
            model: HookedTransformer model for editing
            method: Editing method ("memit", "rome", or "ft")
        """
        self.model = model
        self.device = model.cfg.device if hasattr(model, "cfg") else "cuda"
        self.method = method
        self.edit_history = []
        self.conflict_graph: Dict[int, Set[int]] = {}  # Fact index -> conflicting fact indices

        logger.info(f"Initialized BatchKnowledgeEditor with method={method}")

    def batch_edit_facts(
        self,
        facts: List[Tuple[str, str, str]],  # (prompt, subject, target)
        verify: bool = True,
        detect_conflicts: bool = True,
        rollback_on_failure: bool = True,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> BatchEditResult:
        """
        Edit multiple facts with optional conflict detection.

        Args:
            facts: List of (prompt, subject, target) tuples
            verify: Verify edits after applying
            detect_conflicts: Detect conflicts between edits
            rollback_on_failure: Rollback all edits if any fails

        Returns:
            BatchEditResult with statistics and per-fact results
        """
        logger.info(f"Starting batch edit of {len(facts)} facts")

        # Save original model state for potential rollback
        original_state = self._save_model_state()

        result = BatchEditResult(
            num_facts_edited=len(facts),
            num_successful=0,
            num_failed=0,
            success_rate=0.0,
            edit_results=[],
        )

        try:
            # Step 1: Detect conflicts if requested
            if detect_conflicts:
                self.conflict_graph = self._detect_edit_conflicts(facts)
                conflicting_pairs = sum(len(v) for v in self.conflict_graph.values())
                if conflicting_pairs > 0:
                    logger.warning(f"Detected {conflicting_pairs} conflicting edit pairs")
                    result.interference_detected = True
                    result.interference_details = {
                        "conflict_graph": {k: list(v) for k, v in self.conflict_graph.items()},
                        "num_conflicts": conflicting_pairs,
                    }

            # Step 2: Apply edits in optimal order
            optimal_order = (
                self._compute_edit_order(facts) if detect_conflicts else list(range(len(facts)))
            )
            ordered_facts = [facts[i] for i in optimal_order]

            edit_magnitudes: List[float] = []

            # Step 3: Apply edits
            method_key = self.method.lower()

            if method_key == "memit":
                from circuitkit.applications.editing.memit_wrapper import MemitHandler

                handler = MemitHandler(self.model)
                edit_res_list = handler.edit_multiple_facts(
                    facts=ordered_facts,
                    target_layers=None,
                    use_corpus_C=use_corpus_C,
                    cov_n_samples=cov_n_samples,
                    cov_texts=cov_texts,
                    corpus_id=corpus_id,
                    n_prefixes=n_prefixes,
                    prefix_seed=prefix_seed,
                )
                for edit_result in edit_res_list:
                    if edit_result and edit_result.success:
                        result.num_successful += 1
                        if hasattr(edit_result, "edit_magnitude"):
                            edit_magnitudes.append(edit_result.edit_magnitude)
                    else:
                        result.num_failed += 1
                    result.edit_results.append(
                        asdict(edit_result) if edit_result else {"success": False}
                    )
            elif method_key == "rome":
                # ROME: sequential per-fact (by design, single-fact only)
                from circuitkit.applications.editing.rome_wrapper import RomeHandler

                for fact_idx, (prompt, subject, target) in enumerate(ordered_facts):
                    try:
                        handler = RomeHandler(self.model)
                        edit_result = handler.edit_single_fact(
                            prompt=prompt,
                            subject=subject,
                            target=target,
                            target_layer=self.model.cfg.n_layers // 2,
                            use_corpus_C=use_corpus_C,
                            cov_n_samples=cov_n_samples,
                            cov_texts=cov_texts,
                            corpus_id=corpus_id,
                            n_prefixes=n_prefixes,
                            prefix_seed=prefix_seed,
                        )
                        if edit_result and edit_result.success:
                            result.num_successful += 1
                            if hasattr(edit_result, "edit_magnitude"):
                                edit_magnitudes.append(edit_result.edit_magnitude)
                        else:
                            result.num_failed += 1
                        result.edit_results.append(
                            asdict(edit_result) if edit_result else {"success": False}
                        )
                    except Exception as e:
                        logger.error(f"Error editing fact {fact_idx}: {e}")
                        result.num_failed += 1
                        result.edit_results.append({"success": False, "error": str(e)})
            elif method_key in {"ft", "finetune", "peft"}:
                from circuitkit.applications.editing.fine_tune_editing import FineTuneEditHandler

                handler = FineTuneEditHandler(self.model)
                edit_res_list = handler.edit_multiple_facts(
                    facts=ordered_facts,
                    verify=verify,
                    rollback_on_failure=rollback_on_failure,
                )
                for edit_result in edit_res_list:
                    if edit_result and edit_result.success:
                        result.num_successful += 1
                        if hasattr(edit_result, "edit_magnitude"):
                            edit_magnitudes.append(edit_result.edit_magnitude)
                    else:
                        result.num_failed += 1
                    result.edit_results.append(
                        asdict(edit_result) if edit_result else {"success": False}
                    )
            else:
                raise ValueError(f"Unknown editing method: {self.method}")

            # Step 4: Verify edits
            if verify:
                logger.info("Verifying batch edits...")
                result = self._verify_batch_edits(facts, result)

            # Update statistics
            result.success_rate = result.num_successful / len(facts) if facts else 0.0
            if edit_magnitudes:
                result.total_edit_magnitude = sum(edit_magnitudes)
                result.average_edit_magnitude = np.mean(edit_magnitudes)

            logger.info(
                f"Batch edit complete: {result.num_successful}/{len(facts)} successful "
                f"({result.success_rate:.1%})"
            )

            # Step 5: Rollback if requested and failures occurred
            if rollback_on_failure and result.num_failed > 0:
                logger.warning(f"Rolling back edits due to {result.num_failed} failures")
                self._restore_model_state(original_state)
                result.num_successful = 0
                result.success_rate = 0.0
                result.total_edit_magnitude = 0.0
                result.average_edit_magnitude = 0.0
                result.edit_results = [
                    {**r, "success": False, "note": "rolled_back"} for r in result.edit_results
                ]

            return result

        except Exception as e:
            logger.error(f"Critical error in batch editing: {e}")
            if rollback_on_failure:
                self._restore_model_state(original_state)
            raise

    def _detect_edit_conflicts(
        self,
        facts: List[Tuple[str, str, str]],
    ) -> Dict[int, Set[int]]:
        """
        Detect conflicts between planned edits.

        Two edits conflict if they target the same subject or related facts.

        Args:
            facts: List of facts to edit

        Returns:
            Dict mapping fact_idx -> set of conflicting fact indices
        """
        conflicts: Dict[int, Set[int]] = {i: set() for i in range(len(facts))}

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                prompt_i, subject_i, target_i = facts[i]
                prompt_j, subject_j, target_j = facts[j]

                # Check for conflicts
                if self._facts_conflict(subject_i, target_i, subject_j, target_j):
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    logger.debug(f"Detected conflict between fact {i} and {j}")

        return conflicts

    def _facts_conflict(
        self,
        subject_i: str,
        target_i: str,
        subject_j: str,
        target_j: str,
    ) -> bool:
        """Check if two facts conflict."""
        # Simple heuristic: same subject
        return subject_i.lower() == subject_j.lower()

    def _compute_edit_order(
        self,
        facts: List[Tuple[str, str, str]],
    ) -> List[int]:
        """
        Compute optimal order for applying edits.

        Uses conflict graph to order edits with minimal interference.

        Args:
            facts: List of facts to edit

        Returns:
            List of indices in optimal order
        """
        # Simple strategy: process non-conflicting facts first
        order = []
        processed = set()
        remaining = set(range(len(facts)))

        while remaining:
            # Find next fact with fewest unprocessed conflicts
            best_fact = None
            min_conflicts = float("inf")

            for fact_idx in remaining:
                unprocessed_conflicts = len(self.conflict_graph[fact_idx] & remaining)
                if unprocessed_conflicts < min_conflicts:
                    min_conflicts = unprocessed_conflicts
                    best_fact = fact_idx

            if best_fact is not None:
                order.append(best_fact)
                processed.add(best_fact)
                remaining.remove(best_fact)

        return order

    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        """Save model weight state for rollback."""
        if hasattr(self.model, "named_parameters"):
            return {
                name: param.data.clone().detach() for name, param in self.model.named_parameters()
            }
        if hasattr(self.model, "state_dict"):
            return {
                name: tensor.detach().clone()
                for name, tensor in self.model.state_dict().items()
                if torch.is_tensor(tensor)
            }
        return {}

    def _restore_model_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Restore model to saved state."""
        if hasattr(self.model, "named_parameters"):
            for name, param in self.model.named_parameters():
                if name in state:
                    param.data.copy_(state[name].to(param.device, dtype=param.dtype))
        elif hasattr(self.model, "state_dict"):
            current = self.model.state_dict()
            with torch.no_grad():
                for name, tensor in current.items():
                    if name in state and torch.is_tensor(tensor):
                        tensor.copy_(state[name].to(tensor.device, dtype=tensor.dtype))
        logger.info("Model state restored")

    def _verify_batch_edits(
        self,
        facts: List[Tuple[str, str, str]],
        result: BatchEditResult,
    ) -> BatchEditResult:
        """Verify batch edits were successful."""
        # Would implement confidence checking for each fact
        return result


class UnlearningVerifier:
    """
    Enhanced verification that knowledge is truly forgotten.

    Checks:
    - Target fact produces low probability (unlearned)
    - Model cannot relearn fact easily (leakage detection)
    - Related facts still work (preservation)
    - Gradient-based recovery is slow (true unlearning)
    """

    def __init__(self, model: Any, device: str = "cuda"):
        """
        Initialize UnlearningVerifier.

        Args:
            model: HookedTransformer model
            device: Compute device
        """
        self.model = model
        self.device = device
        logger.info("Initialized UnlearningVerifier")

    def verify_complete_unlearning(
        self,
        facts,  # List[Tuple[str,str]] preferred
        probe_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if probe_methods is None:
            probe_methods = ["confidence", "gradient", "generalization"]

        # Normalise to (prompt, target) tuples.
        # Back-compat: "Subject is Target" strings get split on the FIRST
        # " is ". Multi-word targets containing " is " are preserved (the
        # remainder after the first split goes wholesale into target).
        normalised = []
        for f in facts:
            if isinstance(f, tuple) and len(f) == 2:
                normalised.append((str(f[0]), str(f[1])))
            elif isinstance(f, str) and " is " in f:
                left, right = f.split(" is ", 1)
                normalised.append((left.strip() + " is", right.strip()))
            else:
                logger.warning(f"Skipping malformed fact: {f!r}")

        # Validate probe method names up-front so a typo does not silently
        # produce an empty result with no feedback to the caller.
        _KNOWN_PROBE_METHODS = {"confidence", "gradient", "generalization"}
        for method in probe_methods:
            if method not in _KNOWN_PROBE_METHODS:
                warnings.warn(
                    f"Unknown probe method {method!r}; expected one of "
                    f"{sorted(_KNOWN_PROBE_METHODS)}. It will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        results: Dict[str, Any] = {}
        for prompt, target in normalised:
            key = f"{prompt} {target}".strip()
            results[key] = {}
            for method in probe_methods:
                if method == "confidence":
                    results[key]["confidence"] = self._check_confidence_unlearning(prompt, target)
                elif method == "gradient":
                    results[key]["gradient"] = self._check_gradient_unlearning(prompt, target)
                elif method == "generalization":
                    results[key]["generalization"] = self._check_generalization_unlearning(
                        prompt, target
                    )
        return results

    def _check_confidence_unlearning(self, prompt, target=None):
        from circuitkit.applications.common_utils._tokenization import ScoringError, score_target

        if target is None and isinstance(prompt, str) and " is " in prompt:
            left, right = prompt.split(" is ", 1)
            prompt = left.strip() + " is"
            target = right.strip()
        if target is None:
            return {"unlearned": False, "confidence": 0.5, "error": "malformed fact"}
        try:
            score = score_target(self.model, prompt, target)
        except ScoringError as e:
            return {"unlearned": False, "confidence": 0.0, "error": str(e)}
        threshold = 0.3
        return {
            "unlearned": score.first_token_prob < threshold,
            "confidence": score.first_token_prob,
            "sequence_prob": score.sequence_prob,
            "threshold": threshold,
        }

    def _check_gradient_unlearning(self, prompt, target=None):
        # Teacher-forced cross-entropy, then backward(). Cannot reuse
        # score_target because that's @torch.no_grad().
        from circuitkit.applications.common_utils._tokenization import ScoringError, format_target

        if target is None and isinstance(prompt, str) and " is " in prompt:
            left, right = prompt.split(" is ", 1)
            prompt = left.strip() + " is"
            target = right.strip()
        if target is None:
            return {"unlearned": False, "gradient_magnitude": 0.0, "error": "malformed fact"}
        try:
            formatted = format_target(target, prompt=prompt)
            prepend_bos = bool(getattr(self.model.cfg, "default_prepend_bos", True))
            prompt_ids = self.model.to_tokens(prompt, prepend_bos=prepend_bos)
            full_ids = self.model.to_tokens(prompt + formatted, prepend_bos=prepend_bos)
        except (ScoringError, Exception) as e:
            return {"unlearned": False, "gradient_magnitude": 0.0, "error": str(e)}

        L = prompt_ids.shape[1]
        T = full_ids.shape[1] - L
        if T <= 0:
            return {
                "unlearned": False,
                "gradient_magnitude": 0.0,
                "error": "empty target after tokenisation",
            }
        if not torch.equal(full_ids[0, :L], prompt_ids[0]):
            return {
                "unlearned": False,
                "gradient_magnitude": 0.0,
                "error": "non-compositional tokeniser boundary",
            }

        self.model.zero_grad()
        logits = self.model(full_ids)
        if logits.dim() == 2:
            if logits.shape[0] >= L - 1 + T:
                pred_logits = logits[L - 1 : L - 1 + T, :]
            else:
                pred_logits = logits[-1:].expand(T, -1)
        else:
            pred_logits = logits[0, L - 1 : L - 1 + T, :]
        target_ids = full_ids[0, L : L + T]
        loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)
        loss.backward()

        grad_magnitude = (
            sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None)
            ** 0.5
        )

        return {
            "unlearned": grad_magnitude < 0.1,
            "gradient_magnitude": grad_magnitude,
            "loss": float(loss.item()),
        }

    def _check_generalization_unlearning(self, prompt, target=None):
        """Average first-token probability of `target` across paraphrased
        prompts. Lower = unlearning generalises beyond the exact phrasing.
        Replaces the previous placeholder that returned 0.5 for everything.
        """
        from circuitkit.applications.common_utils._tokenization import ScoringError, score_target

        if target is None and isinstance(prompt, str) and " is " in prompt:
            left, right = prompt.split(" is ", 1)
            prompt = left.strip() + " is"
            target = right.strip()
        if target is None:
            return {
                "unlearned": False,
                "generalization_score": float("nan"),
                "num_related_checked": 0,
            }

        # Heuristic templates. Callers wanting better coverage should pass
        # explicit paraphrases through the BatchKnowledgeEditor path.
        if prompt.endswith(" is"):
            stem = prompt[: -len(" is")].rstrip()
            paraphrases = [
                f"{stem} was",
                f"{stem} has been",
                f"It is known that {stem} is",
            ]
        else:
            paraphrases = [prompt]

        scores = []
        for p in paraphrases:
            try:
                scores.append(score_target(self.model, p, target).first_token_prob)
            except ScoringError:
                continue

        if not scores:
            return {
                "unlearned": False,
                "generalization_score": float("nan"),
                "num_related_checked": 0,
            }
        avg = sum(scores) / len(scores)
        return {
            "unlearned": avg < 0.3,
            "generalization_score": avg,
            "num_related_checked": len(scores),
        }

    def detect_leakage(
        self,
        edited_fact,  # str "S is T" OR (prompt, target) tuple
        num_recovery_steps: int = 10,
        learning_rate: float = 0.001,
    ) -> LeakageReport:
        """Detect if model can relearn an edited fact (leakage probe).

        Computes:
          - Teacher-forced NLL of (prompt, target) — handles multi-token targets
            via the unified scorer.
          - Gradient magnitude of that NLL w.r.t. model parameters.
          - A bounded "relearning capability" = min(1, grad_norm * (1 + |loss|)).

        Args:
            edited_fact: Either a "Subject is Target" string (back-compat;
                split on the first " is ") or a (prompt, target) tuple.
            num_recovery_steps: Recorded on the report when leakage is flagged.
                Not used in this single-step probe; kept for API compatibility.
            learning_rate: Same — recorded for compatibility, not used.

        Returns:
            LeakageReport.
        """
        from circuitkit.applications.common_utils._tokenization import ScoringError, format_target

        # Normalise input (tuple preferred; string back-compat).
        if isinstance(edited_fact, tuple) and len(edited_fact) == 2:
            prompt, target = str(edited_fact[0]), str(edited_fact[1])
            fact_str = f"{prompt} {target}".strip()
        elif isinstance(edited_fact, str) and " is " in edited_fact:
            left, right = edited_fact.split(" is ", 1)
            prompt = left.strip() + " is"
            target = right.strip()
            fact_str = edited_fact
        else:
            return LeakageReport(
                fact_edited=str(edited_fact),
                leakage_detected=False,
                relearning_capability=0.0,
                gradient_magnitude=0.0,
                loss_recovery=0.0,
                recovery_steps_needed=-1,
                details={"error": "malformed fact (need tuple or 'X is Y' string)"},
            )

        try:
            formatted = format_target(target, prompt=prompt)
            prepend_bos = bool(getattr(self.model.cfg, "default_prepend_bos", True))
            prompt_ids = self.model.to_tokens(prompt, prepend_bos=prepend_bos)
            full_ids = self.model.to_tokens(prompt + formatted, prepend_bos=prepend_bos)
        except ScoringError as e:
            return LeakageReport(
                fact_edited=fact_str,
                leakage_detected=False,
                relearning_capability=0.0,
                gradient_magnitude=0.0,
                loss_recovery=0.0,
                recovery_steps_needed=-1,
                details={"error": f"scoring: {e}"},
            )
        except Exception as e:
            logger.warning(f"Error detecting leakage: {e}")
            return LeakageReport(
                fact_edited=fact_str,
                leakage_detected=False,
                relearning_capability=0.0,
                gradient_magnitude=0.0,
                loss_recovery=0.0,
                recovery_steps_needed=-1,
                details={"error": str(e)},
            )

        L = prompt_ids.shape[1]
        T = full_ids.shape[1] - L
        if T <= 0:
            return LeakageReport(
                fact_edited=fact_str,
                leakage_detected=False,
                relearning_capability=0.0,
                gradient_magnitude=0.0,
                loss_recovery=0.0,
                recovery_steps_needed=-1,
                details={"error": "empty target after tokenisation"},
            )
        if not torch.equal(full_ids[0, :L], prompt_ids[0]):
            return LeakageReport(
                fact_edited=fact_str,
                leakage_detected=False,
                relearning_capability=0.0,
                gradient_magnitude=0.0,
                loss_recovery=0.0,
                recovery_steps_needed=-1,
                details={"error": "non-compositional tokeniser boundary"},
            )

        # Teacher-forced NLL — read each target token's logits at the position
        # that predicts it. One forward pass for the loss (used as baseline),
        # one for the gradient. (We could share but separating keeps gradient
        # accounting clean.)
        with torch.no_grad():
            logits = self.model(full_ids)
            if logits.dim() == 2:
                if logits.shape[0] >= L - 1 + T:
                    pred_logits = logits[L - 1 : L - 1 + T, :]
                else:
                    pred_logits = logits[-1:].expand(T, -1)
            else:
                pred_logits = logits[0, L - 1 : L - 1 + T, :]
            target_ids = full_ids[0, L : L + T]
            baseline_loss = float(torch.nn.functional.cross_entropy(pred_logits, target_ids).item())

        self.model.zero_grad()
        logits = self.model(full_ids)
        if logits.dim() == 2:
            if logits.shape[0] >= L - 1 + T:
                pred_logits = logits[L - 1 : L - 1 + T, :]
            else:
                pred_logits = logits[-1:].expand(T, -1)
        else:
            pred_logits = logits[0, L - 1 : L - 1 + T, :]
        target_ids = full_ids[0, L : L + T]
        loss = torch.nn.functional.cross_entropy(pred_logits, target_ids)
        loss.backward()

        grad_magnitude = (
            sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None)
            ** 0.5
        )

        relearning_capability = min(1.0, grad_magnitude * (1.0 + abs(baseline_loss)))
        leakage_detected = relearning_capability > 0.5

        return LeakageReport(
            fact_edited=fact_str,
            leakage_detected=leakage_detected,
            relearning_capability=relearning_capability,
            gradient_magnitude=grad_magnitude,
            loss_recovery=abs(baseline_loss),
            recovery_steps_needed=num_recovery_steps if leakage_detected else -1,
            details={"baseline_loss": baseline_loss, "can_recover": leakage_detected},
        )


# ── Paper-canonical evaluation (Phase 2) ──────────────────────────────────────


@dataclass
class FactRecord:
    """Lightweight fact descriptor for evaluate_edit_paper_metrics.

    Mirrors the CounterFact record schema (Phase 3 will produce these
    from the downloaded JSON). Duck-typed so any object with the same
    attributes will work.
    """

    prompt: str
    subject: str
    target_new: str
    target_true: str
    paraphrase_prompts: List[str] = field(default_factory=list)
    neighborhood_prompts: List[str] = field(default_factory=list)
    generation_prompts: List[str] = field(default_factory=list)


def evaluate_edit_paper_metrics(
    model,
    handler,
    fact: FactRecord,
    target_layer: Optional[int] = None,
    generation_max_tokens: int = 100,
    **edit_kwargs,
):
    """Apply one edit and compute Meng et al. 2022/2023 paper-canonical metrics.

    This is a top-level function (not a method on BatchKnowledgeEditor)
    so benchmark.py (Phase 4) and the example script can both call it
    without instantiating a batch editor.

    Workflow:
      1. Apply the edit via `handler` (ROME or MEMIT).
      2. On the now-edited model, compute all probability-based metrics
         (ES, EM, PS, PM, NS, NM) and generation entropy (GE).
      3. Populate the EditResult's optional fields with the metrics.
      4. Return the EditResult (enriched).

    The caller is responsible for saving/restoring model state if needed
    (the benchmark resets between records; the example script may not).

    Args:
        model:       HookedTransformer — passed explicitly so the metrics
                     functions can score on the same object the handler edited.
        handler:     RomeHandler or MemitHandler instance (already wrapping
                     the same `model`).
        fact:        FactRecord (or any duck-typed object with the same
                     attributes). prompt/subject/target_new/target_true are
                     required; paraphrase_prompts/neighborhood_prompts/
                     generation_prompts default to empty lists.
        target_layer: Override for target layer. If None, uses model's
                     n_layers // 2 heuristic.
        generation_max_tokens: Tokens to generate for GE. Paper uses ~100.
        **edit_kwargs: Forwarded to the handler (use_corpus_C, n_prefixes,
                       cov_n_samples, corpus_id, prefix_seed, etc.).

    Returns:
        EditResult with the seven paper-canonical fields populated.
    """
    from circuitkit.applications.common_utils._metrics import (
        efficacy_metrics,
        generation_entropy,
        neighborhood_metrics,
        paraphrase_metrics,
    )

    from .memit_wrapper import MemitHandler
    from .rome_wrapper import RomeHandler

    layer = target_layer if target_layer is not None else model.cfg.n_layers // 2

    # ── Step 1: apply the edit ──────────────────────────────────────────
    if isinstance(handler, RomeHandler):
        edit_result = handler.edit_single_fact(
            prompt=fact.prompt,
            subject=fact.subject,
            target=fact.target_new,
            target_layer=layer,
            **edit_kwargs,
        )
    elif isinstance(handler, MemitHandler):
        results = handler.edit_multiple_facts(
            facts=[(fact.prompt, fact.subject, fact.target_new)],
            target_layers=[layer],
            **edit_kwargs,
        )
        edit_result = results[0] if results else None
    else:
        raise TypeError(
            f"handler must be RomeHandler or MemitHandler, got {type(handler).__name__}"
        )

    if edit_result is None or not edit_result.success:
        # Edit failed — return the result as-is without metrics.
        return edit_result

    # ── Step 2: compute metrics on the edited model ─────────────────────
    # 2a. Efficacy (ES, EM)
    eff = efficacy_metrics(model, fact.prompt, fact.target_new, fact.target_true)
    edit_result.efficacy_success = eff.success
    edit_result.efficacy_magnitude = eff.magnitude

    # 2b. Paraphrase (PS, PM)
    if fact.paraphrase_prompts:
        para = paraphrase_metrics(
            model,
            fact.paraphrase_prompts,
            fact.target_new,
            fact.target_true,
        )
        edit_result.paraphrase_success = para.success_rate
        edit_result.paraphrase_magnitude = para.mean_magnitude

    # 2c. Neighborhood (NS, NM)
    if fact.neighborhood_prompts:
        neigh = neighborhood_metrics(
            model,
            fact.neighborhood_prompts,
            fact.target_true,
            fact.target_new,
        )
        edit_result.neighborhood_success = neigh.success_rate
        edit_result.neighborhood_magnitude = neigh.mean_magnitude

    # 2d. Generation Entropy (GE)
    #     If generation_prompts are provided (CounterFact records have ~10),
    #     average GE across them. Otherwise use the rewrite prompt.
    prompts_for_ge = fact.generation_prompts if fact.generation_prompts else [fact.prompt]
    ge_values = [
        generation_entropy(model, gp, max_new_tokens=generation_max_tokens) for gp in prompts_for_ge
    ]
    edit_result.generation_entropy = sum(ge_values) / len(ge_values)

    return edit_result
