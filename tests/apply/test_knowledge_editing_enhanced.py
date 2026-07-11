"""
Tests for enhanced knowledge editing with batch operations and unlearning verification.

Tests BatchKnowledgeEditor, UnlearningVerifier, and related functionality.
"""

import pytest
import torch
import torch.nn as nn

from circuitkit.applications.editing.knowledge_editing_enhanced import (
    BatchEditResult,
    BatchKnowledgeEditor,
    LeakageReport,
    UnlearningVerifier,
)

# Fixtures


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""

    class MockConfig:
        device = "cpu"
        n_layers = 12

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 256)
            self.cfg = MockConfig()
            self.vocab_size = 50257

        def forward(self, x):
            # Handle dtype conversion
            x = x.float()
            # Pad to 256 features if needed
            if x.shape[-1] < 256:
                x = torch.cat([x, torch.zeros(x.shape[0], 256 - x.shape[-1])], dim=-1)
            elif x.shape[-1] > 256:
                x = x[:, :256]
            return self.linear(x)

        def to_tokens(self, text, prepend_bos=True):
            """Mock tokenization."""
            tokens = torch.tensor([ord(c) % self.vocab_size for c in text[:50]], dtype=torch.long)
            if prepend_bos:
                tokens = torch.cat([torch.tensor([1], dtype=torch.long), tokens])
            return tokens.unsqueeze(0)

    return MockModel()


@pytest.fixture
def sample_facts():
    """Create sample facts for editing."""
    return [
        ("The capital of France is", "France", "Paris"),
        ("The capital of Germany is", "Germany", "Berlin"),
        ("The capital of Italy is", "Italy", "Rome"),
    ]


# Tests for BatchKnowledgeEditor


class TestBatchKnowledgeEditor:
    """Test batch knowledge editing."""

    def test_editor_initialization(self, mock_model):
        """Test editor initializes correctly."""
        editor = BatchKnowledgeEditor(mock_model, method="memit")

        assert editor.model is mock_model
        assert editor.method == "memit"
        assert len(editor.edit_history) == 0

    def test_editor_with_rome_method(self, mock_model):
        """Test editor with ROME method."""
        editor = BatchKnowledgeEditor(mock_model, method="rome")

        assert editor.method == "rome"

    def test_detect_edit_conflicts(self, mock_model, sample_facts):
        """Test conflict detection between edits."""
        editor = BatchKnowledgeEditor(mock_model)

        # Create facts with and without conflicts
        facts_with_conflict = [
            ("The capital of France is", "France", "Paris"),
            ("The capital of France is", "France", "Berlin"),  # Conflict!
            ("The capital of Germany is", "Germany", "Berlin"),
        ]

        conflicts = editor._detect_edit_conflicts(facts_with_conflict)

        # First and second should conflict
        assert 1 in conflicts[0]
        assert 0 in conflicts[1]

    def test_compute_edit_order(self, mock_model):
        """Test optimal edit order computation."""
        editor = BatchKnowledgeEditor(mock_model)

        facts = [
            ("The capital of France is", "France", "Paris"),
            ("The capital of Germany is", "Germany", "Berlin"),
            ("The capital of France is", "France", "Lyon"),  # Conflict with 0
        ]

        editor.conflict_graph = editor._detect_edit_conflicts(facts)
        order = editor._compute_edit_order(facts)

        assert len(order) == 3
        assert isinstance(order, list)

    def test_save_and_restore_model_state(self, mock_model):
        """Test model state save/restore."""
        editor = BatchKnowledgeEditor(mock_model)

        # Snapshot the real parameter values before anything touches them.
        initial = {name: p.detach().clone() for name, p in mock_model.named_parameters()}

        # Save initial state
        state = editor._save_model_state()

        # Modify model
        with torch.no_grad():
            for param in mock_model.parameters():
                param += 1.0

        # Restore state
        editor._restore_model_state(state)

        # Every parameter must be back to its pre-mutation value; a broken
        # _restore_model_state would leave the +1.0 offset and fail here.
        for name, p in mock_model.named_parameters():
            assert torch.allclose(p, initial[name]), f"parameter {name} not restored"


# Tests for UnlearningVerifier


class TestUnlearningVerifier:
    """Test unlearning verification."""

    def test_verifier_initialization(self, mock_model):
        """Test verifier initializes."""
        verifier = UnlearningVerifier(mock_model, device="cpu")

        assert verifier.model is mock_model
        assert verifier.device == "cpu"

    def test_verify_complete_unlearning(self, mock_model):
        """Test complete unlearning verification."""
        verifier = UnlearningVerifier(mock_model)

        facts = [
            "The capital of France is Paris",
            "The capital of Germany is Berlin",
        ]

        results = verifier.verify_complete_unlearning(
            facts,
            probe_methods=["confidence"],
        )

        assert isinstance(results, dict)
        for fact in facts:
            assert fact in results
            assert "confidence" in results[fact]

    def test_check_confidence_unlearning_valid_fact(self, mock_model):
        """Test confidence unlearning check."""
        verifier = UnlearningVerifier(mock_model)

        result = verifier._check_confidence_unlearning("The capital of France is Paris")

        # Should return dict with required keys (even if error occurred)
        assert isinstance(result, dict)
        assert "unlearned" in result
        assert "confidence" in result
        # Confidence should be between 0 and 1 (even on error, returns 0.5)
        assert 0 <= result["confidence"] <= 1

    def test_check_confidence_unlearning_invalid_fact(self, mock_model):
        """Test confidence check with invalid fact format."""
        verifier = UnlearningVerifier(mock_model)

        result = verifier._check_confidence_unlearning("Invalid fact without is")

        assert result["unlearned"] is False
        assert result["confidence"] == 0.5

    def test_check_gradient_unlearning(self, mock_model):
        """Test gradient unlearning check."""
        verifier = UnlearningVerifier(mock_model)

        result = verifier._check_gradient_unlearning("The capital of France is Paris")

        assert "unlearned" in result
        assert "gradient_magnitude" in result
        assert result["gradient_magnitude"] >= 0

    def test_check_generalization_unlearning(self, mock_model):
        """Test generalization unlearning check."""
        verifier = UnlearningVerifier(mock_model)

        result = verifier._check_generalization_unlearning("The capital of France is Paris")

        assert "unlearned" in result
        assert "generalization_score" in result
        assert 0 <= result["generalization_score"] <= 1

    def test_detect_leakage(self, mock_model):
        """Test leakage detection."""
        verifier = UnlearningVerifier(mock_model)

        report = verifier.detect_leakage("The capital of France is Paris")

        assert isinstance(report, LeakageReport)
        assert report.fact_edited == "The capital of France is Paris"
        assert "relearning_capability" in report.__dict__
        assert 0 <= report.relearning_capability <= 1

    def test_detect_leakage_invalid_fact(self, mock_model):
        """Test leakage detection with invalid fact."""
        verifier = UnlearningVerifier(mock_model)

        report = verifier.detect_leakage("Invalid fact")

        assert not report.leakage_detected
        assert report.relearning_capability == 0.0


# Tests for BatchEditResult


class TestBatchEditResult:
    """Test batch edit result dataclass."""

    def test_batch_result_creation(self):
        """Test creating batch result."""
        result = BatchEditResult(
            num_facts_edited=5,
            num_successful=4,
            num_failed=1,
            success_rate=0.8,
        )

        assert result.num_facts_edited == 5
        assert result.success_rate == 0.8

    def test_batch_result_to_dict(self):
        """Test converting result to dict."""
        result = BatchEditResult(
            num_facts_edited=5,
            num_successful=4,
            num_failed=1,
            success_rate=0.8,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["num_facts_edited"] == 5
        assert result_dict["success_rate"] == 0.8


# Tests for LeakageReport


class TestLeakageReport:
    """Test leakage report dataclass."""

    def test_leakage_report_creation(self):
        """Test creating leakage report."""
        report = LeakageReport(
            fact_edited="Test fact",
            leakage_detected=True,
            relearning_capability=0.7,
            gradient_magnitude=0.5,
            loss_recovery=0.3,
            recovery_steps_needed=10,
        )

        assert report.fact_edited == "Test fact"
        assert report.leakage_detected
        assert report.relearning_capability == 0.7

    def test_leakage_report_to_dict(self):
        """Test converting report to dict."""
        report = LeakageReport(
            fact_edited="Test fact",
            leakage_detected=False,
            relearning_capability=0.0,
            gradient_magnitude=0.0,
            loss_recovery=0.0,
            recovery_steps_needed=-1,
        )

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict["fact_edited"] == "Test fact"


# Integration Tests


class TestKnowledgeEditingIntegration:
    """Integration tests for knowledge editing."""

    def test_batch_editing_workflow(self, mock_model, sample_facts):
        """batch_edit_facts is the documented batch entry point and is wired to
        accept the (prompt, subject, target) fact tuples.

        Full execution (MemitHandler) requires real transformer weights, which a
        Linear-only mock cannot provide — that path is exercised by the CUDA
        integration suite. Here we assert the callable interface and that it
        rejects a malformed fact list, so an API rename/removal fails loudly.
        """
        editor = BatchKnowledgeEditor(mock_model, method="memit")

        assert callable(getattr(editor, "batch_edit_facts", None))
        # sample_facts are 3-tuples (prompt, subject, target) — sanity-check the
        # fixture shape the real method depends on.
        assert all(len(fact) == 3 for fact in sample_facts)

        # A non-tuple fact list must not silently succeed against the mock.
        with pytest.raises(Exception):
            editor.batch_edit_facts(
                ["not-a-fact-tuple"], verify=False, detect_conflicts=False
            )

    def test_unlearning_verification_workflow(self, mock_model):
        """Test complete unlearning verification workflow."""
        verifier = UnlearningVerifier(mock_model)

        facts = [
            "The capital of France is Paris",
            "The capital of Germany is Berlin",
        ]

        # Check all facts
        results = verifier.verify_complete_unlearning(facts)
        assert len(results) == 2

        # Detect leakage for each
        for fact in facts:
            report = verifier.detect_leakage(fact)
            assert isinstance(report, LeakageReport)

    def test_conflict_detection_and_ordering(self, mock_model):
        """Test conflict detection followed by optimal ordering."""
        editor = BatchKnowledgeEditor(mock_model)

        # Facts with conflicts
        facts = [
            ("Q1", "S1", "T1"),
            ("Q2", "S1", "T2"),  # Conflict with fact 0
            ("Q3", "S2", "T3"),
            ("Q4", "S1", "T4"),  # Conflict with 0 and 1
        ]

        # Detect conflicts
        conflicts = editor._detect_edit_conflicts(facts)
        editor.conflict_graph = conflicts

        # Compute order
        order = editor._compute_edit_order(facts)

        # Order should have all facts
        assert len(order) == len(facts)
        assert set(order) == set(range(len(facts)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
