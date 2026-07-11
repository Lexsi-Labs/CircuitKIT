"""
Tests for knowledge editing via circuit-guided ROME/MEMIT.

Tests verify:
1. Core CircuitKnowledgeEditor functionality
2. Single fact editing via ROME
3. Batch fact editing via MEMIT
4. Circuit node ranking and identification
5. Edit result dataclass and serialization
6. Fact confidence verification
7. Preservation of other facts
8. Circuit-guided targeting
9. Error handling and edge cases
10. Edit history tracking
"""

import json
import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import only what we need to avoid transformer_lens dependency issues
import warnings  # noqa: E402 - import after intentional pre-import setup

warnings.filterwarnings("ignore")

# Direct imports to avoid full circuitkit initialization
try:
    from circuitkit.applications.editing.circuit_guided_editing import (
        CircuitGuidedEditor,
        CircuitVerificationResult,
    )
    from circuitkit.applications.editing.knowledge_editing import (
        CircuitKnowledgeEditor,
        EditResult,
        UnlearningReport,
    )
    from circuitkit.applications.editing.memit_wrapper import MemitBatchEdit, MemitHandler
    from circuitkit.applications.editing.rome_wrapper import RomeEditVectors, RomeHandler
except ImportError as e:
    print(f"Warning: Import error ({e}), will continue with mock tests")

try:
    pass

    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False


# Mock model for testing (avoid loading full model in tests)
class MockModel:
    """Mock HookedTransformer for testing without loading weights."""

    def __init__(self):
        self.cfg = type(
            "Config",
            (),
            {
                "device": "cpu",
                "n_layers": 12,
                "d_model": 768,
            },
        )()
        self.blocks = [
            type(
                "Block",
                (),
                {
                    "mlp": type(
                        "MLP",
                        (),
                        {
                            "W_in": torch.nn.Parameter(torch.randn(768, 3072)),
                            "W_out": torch.nn.Parameter(torch.randn(3072, 768)),
                        },
                    )()
                },
            )()
            for _ in range(12)
        ]

    def to_tokens(self, text, prepend_bos=True):
        """Mock tokenization."""
        # Simple mock: return tensor of token IDs
        tokens = torch.tensor([[50256] + [1] * len(text.split())])
        if not prepend_bos:
            tokens = tokens[:, 1:]
        return tokens

    def __call__(self, tokens):
        """Mock forward pass."""
        batch_size, seq_len = tokens.shape
        return torch.randn(batch_size, seq_len, 50257)  # Vocab size

    def add_hook(self, point, fn, level=0):
        """Mock hook registration."""

    def remove_hook(self, point, level=0):
        """Mock hook removal."""


class MockNode:
    """Mock circuit node."""

    def __init__(self, name, layer=0, score=0.5):
        self.name = name
        self.layer = layer
        self.score = score


class MockEdge:
    """Mock circuit edge."""

    def __init__(self, src, dest, weight=0.5):
        self.src = src
        self.dest = dest
        self.weight = weight


class MockCircuit:
    """Mock circuit for testing."""

    def __init__(self):
        self.nodes = [
            MockNode("input", layer=-1),
            MockNode("mlp_0", layer=0, score=0.9),
            MockNode("attn_1", layer=1, score=0.7),
            MockNode("mlp_2", layer=2, score=0.8),
            MockNode("logits", layer=12),
        ]
        self.edges = [
            MockEdge(self.nodes[0], self.nodes[1], weight=0.9),
            MockEdge(self.nodes[1], self.nodes[2], weight=0.7),
            MockEdge(self.nodes[2], self.nodes[3], weight=0.8),
            MockEdge(self.nodes[3], self.nodes[4], weight=0.6),
        ]


class TestEditResult:
    """Test EditResult dataclass."""

    def test_edit_result_creation(self):
        """Test creating an EditResult."""
        result = EditResult(
            success=True,
            fact_prompt="The capital of France is",
            subject="France",
            target="Paris",
            target_layer=6,
            confidence_before=0.1,
            confidence_after=0.95,
            edit_magnitude=0.5,
            interference_ratio=0.1,
        )

        assert result.success is True
        assert result.subject == "France"
        assert result.target == "Paris"
        assert result.confidence_after > result.confidence_before

    def test_edit_result_to_dict(self):
        """Test EditResult serialization to dict."""
        result = EditResult(
            success=True,
            fact_prompt="test",
            subject="test",
            target="test",
            target_layer=5,
            confidence_before=0.1,
            confidence_after=0.9,
            edit_magnitude=0.5,
            interference_ratio=0.1,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["subject"] == "test"

    def test_edit_result_to_json(self):
        """Test EditResult serialization to JSON."""
        result = EditResult(
            success=True,
            fact_prompt="test",
            subject="test",
            target="test",
            target_layer=5,
            confidence_before=0.1,
            confidence_after=0.9,
            edit_magnitude=0.5,
            interference_ratio=0.1,
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["subject"] == "test"


class TestUnlearningReport:
    """Test UnlearningReport dataclass."""

    def test_unlearning_report_creation(self):
        """Test creating an UnlearningReport."""
        report = UnlearningReport(
            fact_edited="The capital of France is Paris",
            fact_unlearned=True,
            unlearning_degree=0.9,
            preserved_facts={
                "The capital of Germany is Berlin": True,
                "The capital of Italy is Rome": True,
            },
            preserved_count=2,
            preserved_total=2,
        )

        assert report.fact_unlearned is True
        assert report.unlearning_degree == 0.9
        assert report.preservation_ratio == 1.0

    def test_preservation_ratio(self):
        """Test preservation ratio calculation."""
        report = UnlearningReport(
            fact_edited="test",
            fact_unlearned=True,
            unlearning_degree=0.8,
            preserved_facts={"fact1": True, "fact2": False},
            preserved_count=1,
            preserved_total=2,
        )

        assert report.preservation_ratio == 0.5

    def test_preservation_ratio_empty(self):
        """Test preservation ratio with no preserved facts."""
        report = UnlearningReport(
            fact_edited="test",
            fact_unlearned=True,
            unlearning_degree=0.8,
            preserved_facts={},
            preserved_count=0,
            preserved_total=0,
        )

        assert report.preservation_ratio == 1.0


class TestCircuitKnowledgeEditor:
    """Test CircuitKnowledgeEditor class."""

    def setup_method(self):
        """Setup for each test."""
        self.model = MockModel()
        self.editor = CircuitKnowledgeEditor(self.model)

    def test_initialization(self):
        """Test CircuitKnowledgeEditor initialization."""
        assert self.editor.model is self.model
        assert self.editor.rome_handler is None
        assert self.editor.memit_handler is None
        assert self.editor.device == "cpu"

    def test_identify_fact_nodes_with_circuit(self):
        """Test identifying fact-relevant nodes from circuit."""
        circuit = MockCircuit()

        nodes = self.editor.identify_fact_nodes(circuit, fact_type="factual")

        assert len(nodes) > 0
        # Should prefer MLPs
        assert any("mlp" in str(node.name).lower() for node in nodes)

    def test_identify_fact_nodes_empty_circuit(self):
        """Test with circuit that has no nodes."""
        circuit = type("Circuit", (), {"nodes": []})()

        nodes = self.editor.identify_fact_nodes(circuit)

        assert nodes == []

    def test_identify_fact_nodes_no_circuit(self):
        """Test with None circuit."""
        nodes = self.editor.identify_fact_nodes(None)

        assert nodes == []

    def test_rank_editing_nodes(self):
        """Test ranking nodes by importance."""
        circuit = MockCircuit()

        ranked = self.editor.rank_editing_nodes(circuit)

        assert len(ranked) > 0
        assert all(
            isinstance(node, MockNode) and isinstance(score, float) for node, score in ranked
        )
        # Should be sorted descending
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_get_fact_confidence(self):
        """Test getting fact confidence."""
        prompt = "The capital of France is"
        target = "Paris"

        confidence = self.editor._get_fact_confidence(prompt, target)

        assert 0.0 <= confidence <= 1.0

    def test_verify_circuit_preserved(self):
        """Test circuit preservation verification."""
        circuit = MockCircuit()

        is_preserved = self.editor._verify_circuit_preserved(circuit)

        assert is_preserved is True

    def test_edit_history_tracking(self):
        """Test that edit history is tracked."""
        assert len(self.editor.get_edit_history()) == 0

        # Create a fake result
        result = EditResult(
            success=True,
            fact_prompt="test",
            subject="test",
            target="test",
            target_layer=0,
            confidence_before=0.1,
            confidence_after=0.9,
            edit_magnitude=0.5,
            interference_ratio=0.1,
        )
        self.editor.edit_history.append(result)

        assert len(self.editor.get_edit_history()) == 1

    def test_clear_edit_history(self):
        """Test clearing edit history."""
        result = EditResult(
            success=True,
            fact_prompt="test",
            subject="test",
            target="test",
            target_layer=0,
            confidence_before=0.1,
            confidence_after=0.9,
            edit_magnitude=0.5,
            interference_ratio=0.1,
        )
        self.editor.edit_history.append(result)

        assert len(self.editor.get_edit_history()) > 0

        self.editor.clear_edit_history()

        assert len(self.editor.get_edit_history()) == 0


class TestRomeHandler:
    """Test ROME (Rank-One Model Editing) handler."""

    def setup_method(self):
        """Setup for each test."""
        self.model = MockModel()
        self.rome = RomeHandler(self.model)

    def test_rome_initialization(self):
        """Test RomeHandler initialization."""
        assert self.rome.model is self.model
        assert self.rome.device == "cpu"
        assert len(self.rome.hessian_cache) == 0

    def test_rome_get_target_confidence(self):
        """Test getting target confidence."""
        confidence = self.rome._get_target_confidence(
            prompt="test prompt",
            target="target",
        )

        assert 0.0 <= confidence <= 1.0

    def test_rome_cache_operations(self):
        """Test caching edit vectors."""
        vectors = RomeEditVectors(
            rank_one_matrix=torch.randn(768, 768),
            update_vector=torch.randn(768),
            original_weight=torch.randn(768, 3072),
            edited_weight=torch.randn(768, 3072),
        )

        self.rome._cache_edit_vectors(5, vectors)

        cached = self.rome.get_cached_edit_vectors(5)
        assert cached is vectors

    def test_rome_clear_cache(self):
        """Test clearing ROME cache."""
        self.rome.hessian_cache[0] = torch.randn(768, 768)
        self.rome.edit_vectors_cache[0] = torch.randn(768)

        assert len(self.rome.hessian_cache) > 0

        self.rome.clear_cache()

        assert len(self.rome.hessian_cache) == 0
        assert len(self.rome.edit_vectors_cache) == 0


class TestMemitHandler:
    """Test MEMIT (Mass Editing Memory in Transformers) handler."""

    def setup_method(self):
        """Setup for each test."""
        self.model = MockModel()
        self.memit = MemitHandler(self.model)

    def test_memit_initialization(self):
        """Test MemitHandler initialization."""
        assert self.memit.model is self.model
        assert self.memit.device == "cpu"

    def test_memit_select_target_layers_single_fact(self):
        """Test selecting target layers for single fact."""
        layers = self.memit._select_target_layers(1)

        assert len(layers) > 0
        assert all(0 <= layer < 12 for layer in layers)

    def test_memit_select_target_layers_multiple_facts(self):
        """Test selecting target layers for multiple facts."""
        layers = self.memit._select_target_layers(20)

        assert len(layers) <= 3

    def test_memit_get_fact_confidence(self):
        """Test getting fact confidence."""
        confidence = self.memit._get_fact_confidence(
            prompt="test prompt",
            target="target",
        )

        assert 0.0 <= confidence <= 1.0

    def test_memit_clear_cache(self):
        """Test clearing MEMIT cache."""
        self.memit.batch_edit_cache["test"] = {}

        assert len(self.memit.batch_edit_cache) > 0

        self.memit.clear_cache()

        assert len(self.memit.batch_edit_cache) == 0


class TestCircuitGuidedEditor:
    """Test CircuitGuidedEditor class."""

    def setup_method(self):
        """Setup for each test."""
        self.model = MockModel()
        self.editor = CircuitGuidedEditor(self.model)

    def test_initialization(self):
        """Test CircuitGuidedEditor initialization."""
        assert self.editor.model is self.model
        assert len(self.editor.circuits) == 0

    def test_identify_fact_nodes(self):
        """Test identifying fact nodes."""
        circuit = MockCircuit()

        nodes = self.editor.identify_fact_nodes(circuit, fact_type="factual")

        assert len(nodes) > 0
        # Should filter out input/logits
        assert not any("input" in str(n.name).lower() for n in nodes)
        assert not any("logit" in str(n.name).lower() for n in nodes)

    def test_rank_nodes_by_importance_activation(self):
        """Test ranking by activation metric."""
        circuit = MockCircuit()

        ranked = self.editor.rank_nodes_by_importance(circuit, metric="activation")

        assert len(ranked) > 0
        # First node should have highest activation
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_nodes_by_importance_connectivity(self):
        """Test ranking by connectivity metric."""
        circuit = MockCircuit()

        ranked = self.editor.rank_nodes_by_importance(circuit, metric="connectivity")

        assert len(ranked) > 0

    def test_rank_nodes_by_importance_depth(self):
        """Test ranking by depth metric."""
        circuit = MockCircuit()

        ranked = self.editor.rank_nodes_by_importance(circuit, metric="depth")

        assert len(ranked) > 0

    def test_rank_nodes_by_importance_combined(self):
        """Test ranking by combined metric."""
        circuit = MockCircuit()

        ranked = self.editor.rank_nodes_by_importance(circuit, metric="combined")

        assert len(ranked) > 0

    def test_register_circuit(self):
        """Test registering a circuit."""
        circuit = MockCircuit()

        self.editor.register_circuit("test_circuit", circuit)

        assert "test_circuit" in self.editor.circuits
        assert self.editor.circuits["test_circuit"] is circuit

    def test_get_fact_confidence(self):
        """Test getting fact confidence."""
        confidence = self.editor._get_fact_confidence(
            self.model,
            "The capital of France is Paris",
        )

        assert 0.0 <= confidence <= 1.0

    def test_verify_circuit_health(self):
        """Test verifying circuit health."""
        circuit = MockCircuit()

        result = self.editor._verify_circuit_health(
            self.model,
            circuit,
            "test_circuit",
        )

        assert isinstance(result, CircuitVerificationResult)
        assert result.circuit_name == "test_circuit"
        assert result.node_count > 0


class TestCircuitVerificationResult:
    """Test CircuitVerificationResult dataclass."""

    def test_verification_result_creation(self):
        """Test creating a verification result."""
        result = CircuitVerificationResult(
            circuit_name="test",
            still_functional=True,
            node_count=10,
            functional_nodes=10,
            broken_edges=0,
            activation_change_mean=0.05,
            activation_change_max=0.15,
        )

        assert result.still_functional is True
        assert result.functional_ratio == 1.0

    def test_functional_ratio(self):
        """Test functional ratio calculation."""
        result = CircuitVerificationResult(
            circuit_name="test",
            still_functional=True,
            node_count=10,
            functional_nodes=7,
            broken_edges=2,
            activation_change_mean=0.1,
            activation_change_max=0.2,
        )

        assert result.functional_ratio == 0.7

    def test_functional_ratio_empty(self):
        """Test functional ratio with no nodes."""
        result = CircuitVerificationResult(
            circuit_name="test",
            still_functional=False,
            node_count=0,
            functional_nodes=0,
            broken_edges=0,
            activation_change_mean=0.0,
            activation_change_max=0.0,
        )

        assert result.functional_ratio == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_editor_with_invalid_layer(self):
        """Test editing with invalid layer index."""
        model = MockModel()
        editor = CircuitKnowledgeEditor(model)

        result = editor.edit_via_circuit(
            circuit=None,
            prompt="test",
            subject="test",
            target="test",
            method="rome",
        )

        # Should not crash, even if method is not fully implemented
        assert isinstance(result, EditResult)

    def test_rome_with_invalid_layer(self):
        """Test ROME with layer out of bounds."""
        model = MockModel()
        rome = RomeHandler(model)

        result = rome.edit_single_fact(
            prompt="test",
            subject="test",
            target="test",
            target_layer=999,  # Invalid layer
        )

        assert result.success is False
        assert "Invalid layer" in result.error_message

    def test_circuit_guided_editor_with_none_circuit(self):
        """Test circuit editor with None circuit."""
        model = MockModel()
        editor = CircuitGuidedEditor(model)

        # Should handle gracefully
        result = editor.verify_edit_unlearning(
            model,
            "fact",
            preserve_facts=None,
        )

        assert isinstance(result, UnlearningReport)

    def test_unlearning_report_with_bad_fact(self):
        """Test unlearning verification with malformed fact."""
        model = MockModel()
        editor = CircuitGuidedEditor(model)

        report = editor.verify_edit_unlearning(
            model,
            "some random text with no structure",
        )

        assert isinstance(report, UnlearningReport)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_knowledge_editing_workflow(self):
        """Test complete knowledge editing workflow."""
        model = MockModel()
        editor = CircuitKnowledgeEditor(model)
        circuit = MockCircuit()

        # Step 1: Identify nodes
        nodes = editor.identify_fact_nodes(circuit)
        assert len(nodes) > 0

        # Step 2: Rank nodes
        ranked = editor.rank_editing_nodes(circuit)
        assert len(ranked) > 0

        # Step 3: Edit (will fail due to mock, but should handle gracefully)
        result = editor.edit_via_circuit(
            circuit=circuit,
            prompt="The capital of France is",
            subject="France",
            target="Lyon",
            method="rome",
            verify=False,
        )

        assert isinstance(result, EditResult)

    def test_circuit_guided_editing_workflow(self):
        """Test complete circuit-guided editing workflow."""
        model = MockModel()
        editor = CircuitGuidedEditor(model)
        circuit = MockCircuit()

        # Register circuit
        editor.register_circuit("test", circuit)

        # Identify nodes
        nodes = editor.identify_fact_nodes(circuit)
        assert len(nodes) > 0

        # Rank nodes
        ranked = editor.rank_nodes_by_importance(circuit)
        assert len(ranked) > 0

        # Verify unlearning (will test with mock data)
        report = editor.verify_edit_unlearning(
            model,
            "The capital of France is Paris",
            preserve_facts=["The capital of Germany is Berlin"],
            circuits_to_verify={"test": circuit},
        )

        assert isinstance(report, UnlearningReport)


class TestMemitBatchEdit:
    """Test MemitBatchEdit result dataclass."""

    def test_memit_batch_edit_creation(self):
        """Test creating a batch edit result."""
        result = MemitBatchEdit(
            facts_edited=[("prompt1", "subject1", "target1")],
            target_layers=[5, 6],
            success_count=1,
            total_count=1,
            avg_confidence_before=0.1,
            avg_confidence_after=0.9,
            avg_edit_magnitude=0.5,
        )

        assert result.success_rate == 1.0
        assert len(result.facts_edited) == 1

    def test_memit_batch_edit_success_rate(self):
        """Test success rate calculation."""
        result = MemitBatchEdit(
            facts_edited=[("p1", "s1", "t1"), ("p2", "s2", "t2")],
            target_layers=[5],
            success_count=1,
            total_count=2,
            avg_confidence_before=0.1,
            avg_confidence_after=0.5,
            avg_edit_magnitude=0.3,
        )

        assert result.success_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
