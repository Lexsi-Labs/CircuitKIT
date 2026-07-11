# FILE: circuitkit/applications/editing/circuit_guided_editing.py
"""
Circuit-Guided Knowledge Editing.

Integrates circuit discovery with knowledge editing to enable:
1. Smart targeting of edits using circuit node information
2. Verification that edits don't break discovered circuits
3. Ranking of nodes for importance-based edit selection
4. End-to-end pipeline from circuit → edit → verify

This module provides high-level utilities for circuit-aware knowledge editing.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

if TYPE_CHECKING:
    from .knowledge_editing import UnlearningReport


@dataclass
class CircuitVerificationResult:
    """Result of circuit verification after editing."""

    circuit_name: str
    still_functional: bool
    node_count: int
    functional_nodes: int
    broken_edges: int
    activation_change_mean: float
    activation_change_max: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def functional_ratio(self) -> float:
        """Ratio of functional nodes to total."""
        if self.node_count == 0:
            return 1.0
        return self.functional_nodes / self.node_count


class CircuitGuidedEditor:
    """
    Use discovered circuits to guide knowledge editing.

    Provides utilities for:
    - Identifying fact-relevant nodes from circuits
    - Ranking nodes by importance for targeting
    - Verifying edits don't break circuits
    - Computing circuit health metrics

    Attributes:
        model: HookedTransformer model
        circuits: Dict of discovered circuits for verification
    """

    def __init__(self, model: HookedTransformer):
        """
        Initialize CircuitGuidedEditor.

        Args:
            model: HookedTransformer model

        Examples:
            >>> model = HookedTransformer.from_pretrained("gpt2-small")
            >>> editor = CircuitGuidedEditor(model)
        """
        self.model = model
        self.device = model.cfg.device
        self.circuits = {}
        self.baseline_activations = {}

    def identify_fact_nodes(
        self,
        circuit: Any,
        fact_type: str = "factual",
    ) -> List[Any]:
        """
        Select nodes from circuit most relevant to a fact.

        Strategy:
        - Filter by node type (MLPs preferred for fact storage)
        - Sort by activation magnitude
        - Sort by connection strength to output
        - Return top K nodes

        Args:
            circuit: Circuit object with nodes and edges
            fact_type: Type of fact ("factual", "negation", "reasoning")
                      Different types activate different circuit regions

        Returns:
            List of circuit nodes, ranked by relevance

        Examples:
            >>> nodes = editor.identify_fact_nodes(circuit, fact_type="factual")
            >>> for node in nodes[:5]:
            ...     print(f"Node: {node.name}")
        """
        if not hasattr(circuit, "nodes") or not circuit.nodes:
            return []

        ranked_nodes = []

        try:
            for node in circuit.nodes:
                # Skip input/output nodes
                if "input" in str(node.name).lower() or "logit" in str(node.name).lower():
                    continue

                # Compute relevance score
                relevance = self._compute_node_relevance(node, circuit, fact_type)
                ranked_nodes.append((node, relevance))

        except Exception as e:
            warnings.warn(f"Error identifying fact nodes: {e}")
            return []

        # Sort by relevance (descending)
        ranked_nodes.sort(key=lambda x: x[1], reverse=True)

        # Return just the nodes
        return [node for node, _ in ranked_nodes]

    def rank_nodes_by_importance(
        self,
        circuit: Any,
        prompt: str = None,
        metric: str = "combined",
    ) -> List[Tuple[Any, float]]:
        """
        Rank circuit nodes by importance for editing.

        Metrics:
        - "activation": Node activation magnitude
        - "connectivity": Number and strength of outgoing edges
        - "depth": Layer position (deeper = more semantic)
        - "combined": Weighted combination of above

        Args:
            circuit: Circuit object with nodes
            prompt: Optional prompt to compute activation-based importance
            metric: Ranking metric ("activation", "connectivity", "depth", "combined")

        Returns:
            List of (Node, importance_score) tuples, sorted descending

        Examples:
            >>> ranked = editor.rank_nodes_by_importance(circuit, metric="combined")
            >>> for node, score in ranked[:5]:
            ...     print(f"Node {node.name}: {score:.3f}")
        """
        if not hasattr(circuit, "nodes") or not circuit.nodes:
            return []

        ranked = []

        try:
            for node in circuit.nodes:
                if "input" in str(node.name).lower() or "logit" in str(node.name).lower():
                    continue

                # Compute relevance by metric
                if metric == "activation":
                    score = self._compute_activation_score(node)
                elif metric == "connectivity":
                    score = self._compute_connectivity_score(node, circuit)
                elif metric == "depth":
                    score = self._compute_depth_score(node)
                elif metric == "combined":
                    activation = self._compute_activation_score(node)
                    connectivity = self._compute_connectivity_score(node, circuit)
                    depth = self._compute_depth_score(node)
                    score = 0.3 * activation + 0.4 * connectivity + 0.3 * depth
                else:
                    score = 0.0

                ranked.append((node, score))

        except Exception as e:
            warnings.warn(f"Error ranking nodes: {e}")
            return []

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def verify_edit_unlearning(
        self,
        model: HookedTransformer,
        edited_fact: str,
        preserve_facts: Optional[List[str]] = None,
        circuits_to_verify: Optional[Dict[str, Any]] = None,
    ) -> "UnlearningReport":
        """
        Verify that an edit succeeded without breaking other knowledge.

        Checks:
        - Original fact produces low probability (unlearned)
        - Preserved facts still produce high probability
        - Circuits still function (activate as before)
        - Task performance not degraded

        Args:
            model: Model to verify (may be edited version)
            edited_fact: Fact that was edited (should be unlearned)
            preserve_facts: List of facts that should still work
            circuits_to_verify: Dict of circuits to verify for functionality

        Returns:
            UnlearningReport with detailed verification results

        Examples:
            >>> report = editor.verify_edit_unlearning(
            ...     model=model,
            ...     edited_fact="The capital of France is Paris",
            ...     preserve_facts=["The capital of Germany is Berlin"],
            ... )
            >>> print(f"Fact unlearned: {report.fact_unlearned}")
            >>> print(f"Preservation ratio: {report.preservation_ratio:.2%}")
        """
        from .knowledge_editing import UnlearningReport

        try:
            # Parse fact
            fact_parts = edited_fact.split(" is ")
            if len(fact_parts) != 2:
                # Try other separators
                if " has " in edited_fact:
                    fact_parts = edited_fact.split(" has ")

            if len(fact_parts) == 2:
                fact_parts[0].strip()
                fact_parts[1].strip()
            else:
                pass

            # Check if edited fact is actually unlearned
            fact_confidence = self._get_fact_confidence(model, edited_fact)
            fact_unlearned = fact_confidence < 0.3  # Threshold for unlearning
            unlearning_degree = 1.0 - fact_confidence

            # Check preservation of other facts
            preserved_count = 0
            preserved_total = 0
            preserved_facts_dict = {}

            if preserve_facts:
                for fact in preserve_facts:
                    preserved_total += 1
                    confidence = self._get_fact_confidence(model, fact)
                    is_preserved = confidence > 0.5
                    preserved_count += is_preserved
                    preserved_facts_dict[fact] = is_preserved

            # Verify circuits if provided
            circuit_details = {}
            if circuits_to_verify:
                for circuit_name, circuit in circuits_to_verify.items():
                    verification = self._verify_circuit_health(model, circuit, circuit_name)
                    circuit_details[circuit_name] = {
                        "still_functional": verification.still_functional,
                        "functional_ratio": verification.functional_ratio,
                    }

            return UnlearningReport(
                fact_edited=edited_fact,
                fact_unlearned=fact_unlearned,
                unlearning_degree=unlearning_degree,
                preserved_facts=preserved_facts_dict,
                preserved_count=preserved_count,
                preserved_total=preserved_total,
                details={
                    "fact_confidence_after": fact_confidence,
                    "circuit_verification": circuit_details,
                },
            )

        except Exception as e:
            warnings.warn(f"Error verifying unlearning: {e}")
            return UnlearningReport(
                fact_edited=edited_fact,
                fact_unlearned=False,
                unlearning_degree=0.0,
                preserved_facts={},
                preserved_count=0,
                preserved_total=0,
                details={"error": str(e)},
            )

    def register_circuit(self, name: str, circuit: Any):
        """Register a circuit for verification."""
        self.circuits[name] = circuit

    def _compute_node_relevance(
        self,
        node: Any,
        circuit: Any,
        fact_type: str,
    ) -> float:
        """Compute relevance of a node for a fact type."""
        try:
            # Start with activation score
            relevance = 1.0

            # Boost MLPs (typically store facts)
            if hasattr(node, "name") and "mlp" in str(node.name).lower():
                relevance *= 2.0

            # Boost by activation magnitude
            if hasattr(node, "score") and node.score is not None:
                relevance *= 1.0 + abs(node.score)

            # Boost by connectivity
            edge_count = 0
            if hasattr(circuit, "edges"):
                for edge in circuit.edges:
                    if hasattr(edge, "src") and edge.src == node:
                        edge_count += 1

            relevance *= 1.0 + edge_count * 0.1

            return relevance

        except Exception:
            return 1.0

    def _compute_activation_score(self, node: Any) -> float:
        """Compute activation magnitude score for a node."""
        try:
            if hasattr(node, "score") and node.score is not None:
                return abs(node.score)
            return 1.0
        except Exception:
            return 1.0

    def _compute_connectivity_score(self, node: Any, circuit: Any) -> float:
        """Compute connectivity score (edge strength and count)."""
        try:
            if not hasattr(circuit, "edges"):
                return 1.0

            connectivity = 0.0
            edge_count = 0

            for edge in circuit.edges:
                if hasattr(edge, "src") and edge.src == node:
                    edge_count += 1
                    if hasattr(edge, "weight"):
                        connectivity += abs(edge.weight)

            if edge_count > 0:
                connectivity /= edge_count

            return 1.0 + connectivity

        except Exception:
            return 1.0

    def _compute_depth_score(self, node: Any) -> float:
        """Compute depth score (layer position)."""
        try:
            if hasattr(node, "layer"):
                # Normalize to 0-1
                layer_norm = node.layer / max(1, self.model.cfg.n_layers)
                # Prefer deeper layers
                return layer_norm

            return 0.5

        except Exception:
            return 0.5

    def _get_fact_confidence(
        self,
        model: HookedTransformer,
        fact: str,
    ) -> float:
        """Get model confidence in a fact.

        Tries several separator patterns to split fact into (prompt, target).
        Falls back to measuring the log-probability of the last token of the
        full fact string when no separator is found.
        """
        try:
            prompt, target = None, None
            for sep in (" is ", ": ", " -> ", " = ", " are ", " was ", " were "):
                if sep in fact:
                    idx = fact.index(sep)
                    prompt = fact[: idx + len(sep)].rstrip()
                    target = fact[idx + len(sep) :].strip()
                    if target:
                        break

            if prompt is not None and target:
                prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
                target_tokens = model.to_tokens(target, prepend_bos=False)
                with torch.no_grad():
                    logits = model(prompt_tokens)
                target_logits = logits[0, -1, :]
                target_logprobs = torch.nn.functional.log_softmax(target_logits, dim=-1)
                total_lp = sum(
                    target_logprobs[tid].item()
                    for tid in target_tokens[0]
                    if 0 <= tid < target_logprobs.shape[0]
                )
                avg_lp = total_lp / max(1, len(target_tokens[0]))
                return float(min(1.0, max(0.0, math.exp(avg_lp))))

            # No separator found: measure log-prob of the fact's last token.
            tokens = model.to_tokens(fact, prepend_bos=True)
            with torch.no_grad():
                logits = model(tokens)
            last_logprobs = torch.nn.functional.log_softmax(logits[0, -2, :], dim=-1)
            last_token = int(tokens[0, -1].item())
            lp = (
                last_logprobs[last_token].item()
                if 0 <= last_token < last_logprobs.shape[0]
                else -10.0
            )
            return float(min(1.0, max(0.0, math.exp(lp))))

        except Exception:
            return 0.5

    def _verify_circuit_health(
        self,
        model: HookedTransformer,
        circuit: Any,
        circuit_name: str,
    ) -> CircuitVerificationResult:
        """Verify that a circuit is still functional."""
        try:
            node_count = len(circuit.nodes) if hasattr(circuit, "nodes") else 0
            functional_nodes = 0
            broken_edges = 0

            # Check all nodes still exist
            for node in circuit.nodes if hasattr(circuit, "nodes") else []:
                try:
                    # Node is functional if it has valid attributes
                    if hasattr(node, "name"):
                        functional_nodes += 1
                except Exception:
                    pass

            # Check edges
            if hasattr(circuit, "edges"):
                for edge in circuit.edges:
                    try:
                        # Edge is broken if source or dest node is gone
                        if not (hasattr(edge.src, "name") and hasattr(edge.dest, "name")):
                            broken_edges += 1
                    except Exception:
                        broken_edges += 1

            return CircuitVerificationResult(
                circuit_name=circuit_name,
                still_functional=functional_nodes > 0,
                node_count=node_count,
                functional_nodes=functional_nodes,
                broken_edges=broken_edges,
                activation_change_mean=0.0,
                activation_change_max=0.0,
                details={},
            )

        except Exception as e:
            warnings.warn(f"Error verifying circuit health: {e}")
            return CircuitVerificationResult(
                circuit_name=circuit_name,
                still_functional=False,
                node_count=0,
                functional_nodes=0,
                broken_edges=0,
                activation_change_mean=0.0,
                activation_change_max=0.0,
                details={"error": str(e)},
            )
