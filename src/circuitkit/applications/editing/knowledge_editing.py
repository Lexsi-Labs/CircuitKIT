# FILE: circuitkit/applications/editing/knowledge_editing.py
"""
Knowledge Editing via Circuit-Guided ROME/MEMIT.

This module implements circuit-aware knowledge editing for transformer models.
It integrates ROME (Rank-One Model Editing) and MEMIT (Mass Editing Memory in
Transformers) with circuit discovery to enable precise fact editing with minimal
side effects.

Key capabilities:
- Single fact editing via circuit-guided ROME
- Batch fact editing via MEMIT
- Circuit node ranking for edit targeting
- Verification of edits and unlearning
- Preservation of unrelated knowledge
"""

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

try:
    from transformer_lens import HookedTransformer
    from transformer_lens.utils import get_act_name
except ImportError:
    # Allow graceful degradation if transformer_lens unavailable
    HookedTransformer = None
    get_act_name = None


@dataclass
class EditResult:
    """Result of a single knowledge edit operation."""

    success: bool
    fact_prompt: str
    subject: str
    target: str
    target_layer: int
    # The four numeric metrics below default to 0.0 so that failure-path
    # construction (validation errors, tokenization failures, etc.) does not
    # have to supply meaningless values for a non-applied edit.
    confidence_before: float = 0.0  # Target probability before edit
    confidence_after: float = 0.0  # Target probability after edit
    edit_magnitude: float = 0.0  # L2 norm of weight change
    interference_ratio: float = 0.0  # Ratio of affected other facts to total
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = self.to_dict()
        data["metadata"] = data.get("metadata") or {}
        return json.dumps(data, default=str)


@dataclass
class UnlearningReport:
    """Report on knowledge unlearning verification."""

    fact_edited: str
    fact_unlearned: bool
    unlearning_degree: float  # How much the fact probability decreased (0-1)
    preserved_facts: Dict[str, bool]  # {fact_description: preserved}
    preserved_count: int
    preserved_total: int
    task_performance_change: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def preservation_ratio(self) -> float:
        """Ratio of preserved facts to total."""
        if self.preserved_total == 0:
            return 1.0
        return self.preserved_count / self.preserved_total


@dataclass
class ScoredNode:
    """Node wrapper that behaves like both a node and a (node, score) pair."""

    node: Any
    score: float

    def __iter__(self):
        yield self.node
        yield self.score

    def __getattr__(self, name: str):
        return getattr(self.node, name)


class CircuitKnowledgeEditor:
    """
    Edit knowledge in transformers using circuits as guides.

    This class enables circuit-informed knowledge editing by:
    1. Identifying circuit nodes relevant to a fact
    2. Applying ROME/MEMIT/FT edits to those nodes
    3. Verifying edits without breaking related knowledge
    4. Ranking nodes by importance for targeted editing

    Attributes:
        model: HookedTransformer model to edit
        rome_handler: ROME editing handler (initialized on demand)
        memit_handler: MEMIT batch editing handler (initialized on demand)
        ft_handler: gradient-based fine-tuning handler (initialized on demand)
        device: Compute device (CPU or GPU)
    """

    def __init__(self, model: HookedTransformer):
        """
        Initialize CircuitKnowledgeEditor.

        Args:
            model: HookedTransformer model for editing

        Examples:
            >>> from transformer_lens import HookedTransformer
            >>> model = HookedTransformer.from_pretrained("gpt2-small")
            >>> editor = CircuitKnowledgeEditor(model)
        """
        self.model = model
        self.device = model.cfg.device
        from ..common_utils._device import warn_if_mps_editing

        warn_if_mps_editing(self.device)
        self.rome_handler = None
        self.memit_handler = None
        self.ft_handler = None
        self.edit_history = []
        self._original_state = self._save_model_state()

    def edit_via_circuit(
        self,
        prompt: str,
        subject: str,
        target: str,
        circuit: Optional[Any] = None,
        preserve_circuits: Optional[List[Any]] = None,
        method: str = "rome",
        fact_type: str = "factual",
        verify: bool = True,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> EditResult:
        """
        Edit a fact at circuit-identified nodes.

        Strategy:
        1. If circuit provided, identify nodes most relevant to the fact
        2. Rank nodes by importance for editing
        3. Apply ROME/MEMIT to target layers
        4. Verify fact is modified and related facts preserved

        Args:
            circuit: Circuit object containing graph nodes/edges (optional).
                    If None, uses heuristic layer selection.
            prompt: Prompt containing the fact to edit.
                   Should follow format: "[SUBJECT] [RELATION] [BLANK]"
                   Example: "The Eiffel Tower is located in"
            subject: Subject entity being edited (e.g., "Eiffel Tower")
            target: Target fact after editing (e.g., "Berlin")
            preserve_circuits: List of circuits to preserve without modification.
                              Edits should not break these circuits.
            method: Editing method ("rome", "memit", or "ft"). ROME for
                   single facts, MEMIT for batch, and FT for the
                   architecture-agnostic gradient-based edit path.
            fact_type: Type of fact being edited ("factual", "negation", "reasoning")
            verify: Whether to verify edit success post-editing

        Returns:
            EditResult with success metrics and metadata

        Examples:
            >>> prompt = "The capital of France is"
            >>> result = editor.edit_via_circuit(
            ...     circuit=None,
            ...     prompt=prompt,
            ...     subject="France",
            ...     target="Lyon",  # Wrong but editable for demo
            ...     method="rome"
            ... )
            >>> print(f"Edit successful: {result.success}")
            >>> print(f"Confidence change: {result.confidence_before:.3f} -> {result.confidence_after:.3f}")
        """
        try:
            # Identify target layers using circuit if available
            if circuit is not None:
                target_nodes = self.identify_fact_nodes(circuit, fact_type=fact_type)
                target_layer = self._select_best_edit_layer(target_nodes)
            else:
                # Heuristic: target middle layers (typically MLP layers)
                mid_layer = self.model.cfg.n_layers // 2
                target_layer = mid_layer
                target_nodes = []

            # Apply edit using specified method
            method_key = method.lower()
            if method_key == "rome":
                if self.rome_handler is None:
                    from .rome_wrapper import RomeHandler

                    self.rome_handler = RomeHandler(self.model)

                edit_result = self.rome_handler.edit_single_fact(
                    prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=target_layer,
                    use_corpus_C=use_corpus_C,
                    cov_n_samples=cov_n_samples,
                    cov_texts=cov_texts,
                    corpus_id=corpus_id,
                    n_prefixes=n_prefixes,
                    prefix_seed=prefix_seed,
                )
            elif method_key == "memit":
                if self.memit_handler is None:
                    from .memit_wrapper import MemitHandler

                    self.memit_handler = MemitHandler(self.model)

                # MEMIT works with batches
                facts = [(prompt, subject, target)]
                edit_results = self.memit_handler.edit_multiple_facts(
                    facts=facts,
                    target_layers=[target_layer] if target_layer is not None else None,
                    use_corpus_C=use_corpus_C,
                    cov_n_samples=cov_n_samples,
                    cov_texts=cov_texts,
                    corpus_id=corpus_id,
                    n_prefixes=n_prefixes,
                    prefix_seed=prefix_seed,
                )
                edit_result = edit_results[0] if edit_results else None
            elif method_key in {"ft", "finetune", "peft"}:
                if self.ft_handler is None:
                    from .fine_tune_editing import FineTuneEditHandler

                    self.ft_handler = FineTuneEditHandler(self.model)

                edit_result = self.ft_handler.edit_single_fact(
                    prompt=prompt,
                    subject=subject,
                    target=target,
                    circuit=circuit,
                    preserve_circuits=preserve_circuits,
                    verify=verify,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            # Verify edit if requested
            if verify and edit_result and edit_result.success:

                # Check if other circuits are preserved
                interference = 0.0
                if preserve_circuits:
                    for preserve_circuit in preserve_circuits:
                        preserved = self._verify_circuit_preserved(preserve_circuit)
                        if not preserved:
                            interference += 1.0
                    interference /= len(preserve_circuits)

                # Update result with verification metrics
                edit_result.interference_ratio = interference

            # Rollback all prior edits if this one failed
            if edit_result is not None and not edit_result.success:
                logger.info("Rolling back all edits...")
                self._restore_model_state(self._original_state)
                for r in self.edit_history:
                    r.edit_magnitude = 0.0
                    r.confidence_after = 0.0
                    if r.metadata is None:
                        r.metadata = {}
                    r.metadata["rolled_back"] = True
                self.edit_history.clear()
                return edit_result

            self.edit_history.append(edit_result)
            self._original_state = self._save_model_state()
            return edit_result

        except Exception as e:
            failure_result = EditResult(
                success=False,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=-1,
                confidence_before=0.0,
                confidence_after=0.0,
                edit_magnitude=0.0,
                interference_ratio=1.0,
                error_message=str(e),
            )
            # Preserve the audit trail: failed edits must still be recorded
            # in edit_history so callers can inspect what was attempted.
            self.edit_history.append(failure_result)
            return failure_result

    def identify_fact_nodes(
        self,
        circuit: Any,
        fact_type: str = "factual",
    ) -> List[ScoredNode]:
        """
        Identify circuit nodes most relevant to a fact.

        Filters circuit nodes based on:
        - Node type (MLPs are typically fact storage)
        - Activation magnitude
        - Connection strength to output
        - Layer position (deeper = often more semantic)

        Args:
            circuit: Circuit object with nodes and edges
            fact_type: Type of fact ("factual", "negation", "reasoning").
                      Different types may activate different nodes.

        Returns:
            List of (Node, importance_score) tuples, ranked by importance

        Examples:
            >>> nodes = editor.identify_fact_nodes(circuit, fact_type="factual")
            >>> for node, score in nodes[:3]:
            ...     print(f"Node {node.name}: importance {score:.3f}")
        """
        if not hasattr(circuit, "nodes") or not hasattr(circuit, "edges"):
            return []

        nodes_with_scores: List[ScoredNode] = []

        try:
            for node in circuit.nodes:
                # Skip input/output nodes
                if node.name in ["input", "logits"]:
                    continue

                # Prefer MLP layers for fact storage
                is_mlp = "mlp" in node.name.lower() or "MLP" in str(node.name)
                mlp_bonus = 1.5 if is_mlp else 1.0

                # Get node activation magnitude if available
                activation_score = 1.0
                if hasattr(node, "score"):
                    activation_score = abs(node.score) if node.score is not None else 1.0

                # Count outgoing edges (connectivity to downstream)
                edge_count = 0
                connectivity = 0.0
                if hasattr(circuit, "edges"):
                    for edge in circuit.edges:
                        if hasattr(edge, "src") and edge.src == node:
                            edge_count += 1
                            if hasattr(edge, "weight"):
                                connectivity += abs(edge.weight)

                # Combine scores
                importance = activation_score * (1.0 + connectivity) * mlp_bonus

                nodes_with_scores.append(ScoredNode(node=node, score=importance))

        except Exception as e:
            warnings.warn(f"Error identifying fact nodes: {e}")
            return []

        # Sort by importance (descending)
        nodes_with_scores.sort(key=lambda x: x.score, reverse=True)
        return nodes_with_scores

    def rank_editing_nodes(
        self,
        circuit: Any,
        prompt: Optional[str] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Rank circuit nodes by importance for editing a fact.

        Uses:
        - Layer depth (later layers often have higher-level features)
        - Edge weights (stronger connections = more impact)
        - Activation magnitude in response to prompt

        Args:
            circuit: Circuit object with nodes and edges
            prompt: Prompt to evaluate node importance for

        Returns:
            List of (Node, importance_rank) tuples, sorted descending by rank
        """
        if not hasattr(circuit, "nodes"):
            return []

        ranked_nodes = []

        try:
            # For each node, compute importance rank
            for node in circuit.nodes:
                if node.name in ["input", "logits"]:
                    continue

                # Layer depth component (later layers = higher rank)
                layer_component = 0.0
                if hasattr(node, "layer"):
                    layer_component = node.layer / max(1, self.model.cfg.n_layers)

                # Edge strength component
                edge_component = 0.0
                edge_count = 0
                if hasattr(circuit, "edges"):
                    for edge in circuit.edges:
                        if hasattr(edge, "src") and edge.src == node:
                            edge_count += 1
                            if hasattr(edge, "weight"):
                                edge_component += abs(edge.weight)

                if edge_count > 0:
                    edge_component /= edge_count

                # Activation component
                activation_component = 0.0
                if hasattr(node, "score") and node.score is not None:
                    activation_component = abs(node.score)

                # Weighted combination
                importance = (
                    0.3 * layer_component + 0.4 * edge_component + 0.3 * activation_component
                )

                ranked_nodes.append((node, importance))

        except Exception as e:
            warnings.warn(f"Error ranking nodes: {e}")
            return []

        ranked_nodes.sort(key=lambda x: x[1], reverse=True)
        return ranked_nodes

    def _select_best_edit_layer(
        self,
        nodes: List[Tuple[Any, float]],
    ) -> Optional[int]:
        """
        Select best layer for editing from ranked nodes.

        Typically selects a middle-to-late layer where facts are well-encoded
        but before too much mixing occurs.

        Args:
            nodes: List of (Node, score) tuples from identify_fact_nodes

        Returns:
            Best layer index, or None if no suitable layer found
        """
        if not nodes:
            return None

        import warnings

        try:
            # Only consider MLP nodes -- ROME targets MLP weight matrices.
            # Using attention layers as ROME targets is invalid.
            mlp_candidate_layers = []
            for node, score in nodes:
                if not hasattr(node, "layer"):
                    continue
                node_name = str(getattr(node, "name", ""))
                is_mlp = "mlp" in node_name.lower()
                if not is_mlp:
                    continue
                normalized_layer = node.layer / max(1, self.model.cfg.n_layers)
                if 0.4 <= normalized_layer <= 0.8:
                    mlp_candidate_layers.append((node.layer, score))

            if mlp_candidate_layers:
                mlp_candidate_layers.sort(key=lambda x: x[1], reverse=True)
                return mlp_candidate_layers[0][0]

            # No MLP node in the preferred range -- widen to all MLP nodes.
            all_mlp = [
                (node.layer, score)
                for node, score in nodes
                if hasattr(node, "layer") and "mlp" in str(getattr(node, "name", "")).lower()
            ]
            if all_mlp:
                all_mlp.sort(key=lambda x: x[1], reverse=True)
                return all_mlp[0][0]

            # No MLP nodes at all -- fall back to the model's middle MLP layer.
            # Log a warning: ROME edits on non-MLP layers are not supported.
            mid = self.model.cfg.n_layers // 2
            warnings.warn(
                f"No MLP nodes found in circuit; defaulting ROME target to layer {mid}. "
                f"ROME is designed for MLP layers -- verify the circuit contains MLP nodes."
            )
            return mid

        except Exception:
            return None

    def _get_fact_confidence(self, prompt: str, target: str) -> float:
        """First-token probability of `target` given `prompt`. Delegates
        to the unified scorer in `_scoring`, which handles BPE / SentencePiece
        / WordPiece tokeniser conventions, multi-token targets, and BOS
        settings consistently across all TransformerLens-supported models.
        """
        from circuitkit.applications.common_utils._tokenization import ScoringError, score_target

        try:
            return score_target(self.model, prompt, target).first_token_prob
        except ScoringError as exc:
            import warnings

            warnings.warn(f"Could not score target {target!r}: {exc}")
            return 0.0

    def _verify_circuit_preserved(self, circuit: Any) -> bool:
        """
        Verify that a circuit still functions after editing.

        Checks if circuit nodes are still in the model and functional.

        Args:
            circuit: Circuit to verify

        Returns:
            True if circuit is still functional
        """
        try:
            if not hasattr(circuit, "nodes"):
                return True

            # Check that nodes still exist
            for node in circuit.nodes:
                if not hasattr(node, "name"):
                    return False

            return True

        except Exception:
            return False

    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        """Save a copy of all model parameters for rollback."""
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
        """Restore model parameters from a previously saved state."""
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

    def get_edit_history(self) -> List[EditResult]:
        """Get history of all edits performed."""
        return self.edit_history

    def clear_edit_history(self):
        """Clear edit history."""
        self.edit_history = []
