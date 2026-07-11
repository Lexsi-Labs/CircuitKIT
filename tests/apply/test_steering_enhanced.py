"""
Tests for enhanced steering with composition and safety features.

Tests SteeringComposer, SafetyDatasetSynthesis, and SteeringEvaluationGates.
"""

import pytest
import torch
import torch.nn as nn

from circuitkit.applications.steering.steering_enhanced import (
    SafetyDatasetSynthesis,
    SteeringComposer,
    SteeringEvaluationGates,
)

# Fixtures


@pytest.fixture
def mock_model():
    """Create a mock transformer model."""

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 256)
            self.cfg = type("Config", (), {"device": "cpu", "n_layers": 12})()

        def forward(self, x):
            return self.linear(x)

    return MockModel()


@pytest.fixture
def mock_steering_vectors():
    """Create mock steering vectors."""
    return {
        "A0.0": torch.randn(256),
        "A0.1": torch.randn(256),
        "MLP1": torch.randn(256),
    }


@pytest.fixture
def second_steering_vectors():
    """Create second set of steering vectors."""
    return {
        "A0.0": torch.randn(256) * 0.5,
        "A0.1": torch.randn(256) * 0.5,
        "MLP1": torch.randn(256) * 0.5,
    }


# Tests for SteeringComposer


class TestSteeringComposer:
    """Test steering composition functionality."""

    def test_composer_initialization(self):
        """Test composer initializes correctly."""
        composer = SteeringComposer()
        assert len(composer.steering_dict) == 0
        assert len(composer.coefficients) == 0

    def test_add_steering(self, mock_steering_vectors):
        """Test adding steering to composer."""
        composer = SteeringComposer()
        composer.add_steering("correction1", mock_steering_vectors, coefficient=1.0)

        assert "correction1" in composer.steering_dict
        assert composer.coefficients["correction1"] == 1.0
        assert len(composer.steering_dict["correction1"]) == 3

    def test_add_multiple_steerings(self, mock_steering_vectors, second_steering_vectors):
        """Test adding multiple steerings."""
        composer = SteeringComposer()
        composer.add_steering("correction1", mock_steering_vectors, coefficient=1.0)
        composer.add_steering("correction2", second_steering_vectors, coefficient=0.5)

        assert len(composer.steering_dict) == 2
        assert composer.coefficients["correction1"] == 1.0
        assert composer.coefficients["correction2"] == 0.5

    def test_remove_steering(self, mock_steering_vectors):
        """Test removing steering."""
        composer = SteeringComposer()
        composer.add_steering("correction1", mock_steering_vectors)
        composer.remove_steering("correction1")

        assert "correction1" not in composer.steering_dict
        assert len(composer.steering_dict) == 0

    def test_get_composed_vectors_sum(self, mock_steering_vectors, second_steering_vectors):
        """Test composed vectors with sum aggregation."""
        composer = SteeringComposer()
        composer.add_steering("c1", mock_steering_vectors, coefficient=1.0)
        composer.add_steering("c2", second_steering_vectors, coefficient=1.0)

        composed = composer.get_composed_vectors(aggregate="sum")

        assert len(composed) == 3
        for node_name in mock_steering_vectors.keys():
            assert node_name in composed
            # Should be sum of two vectors
            expected = mock_steering_vectors[node_name] + second_steering_vectors[node_name]
            assert torch.allclose(composed[node_name], expected)

    def test_get_composed_vectors_mean(self, mock_steering_vectors, second_steering_vectors):
        """Test composed vectors with mean aggregation."""
        composer = SteeringComposer()
        composer.add_steering("c1", mock_steering_vectors)
        composer.add_steering("c2", second_steering_vectors)

        composed = composer.get_composed_vectors(aggregate="mean")

        assert len(composed) == 3
        for node_name in mock_steering_vectors.keys():
            expected = (mock_steering_vectors[node_name] + second_steering_vectors[node_name]) / 2
            assert torch.allclose(composed[node_name], expected, atol=1e-5)

    def test_get_composed_vectors_weighted(self, mock_steering_vectors, second_steering_vectors):
        """Test composed vectors with weighted mean aggregation."""
        composer = SteeringComposer()
        composer.add_steering("c1", mock_steering_vectors, coefficient=2.0)
        composer.add_steering("c2", second_steering_vectors, coefficient=1.0)

        composed = composer.get_composed_vectors(aggregate="weighted_mean")

        assert len(composed) == 3
        # Weights should be 2/3 and 1/3
        for node_name in mock_steering_vectors.keys():
            expected = mock_steering_vectors[node_name] * (2.0 / 3.0) + second_steering_vectors[
                node_name
            ] * (1.0 / 3.0)
            assert torch.allclose(composed[node_name], expected, atol=1e-5)

    def test_compute_interference_matrix(self, mock_steering_vectors, second_steering_vectors):
        """Test interference matrix computation."""
        composer = SteeringComposer()
        composer.add_steering("c1", mock_steering_vectors)
        composer.add_steering("c2", second_steering_vectors)

        interference = composer.compute_interference_matrix()

        assert len(interference) == 1
        key = ("c1", "c2")
        assert key in interference
        # Interference should be in [0, 2]
        assert 0 <= interference[key] <= 2

    def test_detect_high_interference(self, mock_steering_vectors):
        """Test detection of high interference."""
        composer = SteeringComposer()

        # Create very similar vectors (low interference)
        similar_vectors = {k: v.clone() for k, v in mock_steering_vectors.items()}
        composer.add_steering("c1", mock_steering_vectors)
        composer.add_steering("c2", similar_vectors)

        high_interference = composer.detect_high_interference(threshold=0.9)
        # Similar vectors should have low interference
        assert len(high_interference) == 0

    def test_get_parameter_counts(self, mock_steering_vectors, second_steering_vectors):
        """Test parameter counting."""
        composer = SteeringComposer()
        composer.add_steering("c1", mock_steering_vectors)
        composer.add_steering("c2", second_steering_vectors)

        counts = composer.get_parameter_counts()

        assert "total_steering_params" in counts
        assert "per_steering" in counts
        assert "num_steerings" in counts
        assert counts["num_steerings"] == 2
        assert counts["total_steering_params"] > 0

    def test_summary(self, mock_steering_vectors, second_steering_vectors):
        """Test summary generation."""
        composer = SteeringComposer()
        composer.add_steering("c1", mock_steering_vectors, coefficient=1.0)
        composer.add_steering("c2", second_steering_vectors, coefficient=0.5)

        summary = composer.summary()

        assert isinstance(summary, str)
        assert "Steering Composition" in summary
        assert "c1" in summary
        assert "c2" in summary


# Tests for SafetyDatasetSynthesis


class TestSafetyDatasetSynthesis:
    """Test safety dataset synthesis."""

    def test_synthesis_initialization(self, mock_model):
        """Test synthesizer initializes."""
        synthesizer = SafetyDatasetSynthesis(mock_model, device="cpu")
        assert synthesizer.model is mock_model
        assert synthesizer.device == "cpu"

    def test_generate_paraphrase_variations(self, mock_model):
        """Test paraphrase variation generation."""
        synthesizer = SafetyDatasetSynthesis(mock_model)
        prompt = "What is the capital of France"

        variations = synthesizer.generate_adversarial_prompts(
            prompt,
            num_variations=5,
            perturbation_type="paraphrase",
        )

        assert len(variations) == 5
        assert variations[0] == prompt  # First is original

    def test_generate_noise_variations(self, mock_model):
        """Test noise variation generation."""
        synthesizer = SafetyDatasetSynthesis(mock_model)
        prompt = "What is the capital of France"

        variations = synthesizer.generate_adversarial_prompts(
            prompt,
            num_variations=5,
            perturbation_type="noise",
        )

        assert len(variations) == 5
        assert variations[0] == prompt
        # Some variations should differ
        assert any(v != prompt for v in variations[1:])

    def test_generate_reorder_variations(self, mock_model):
        """Test word reorder variations."""
        synthesizer = SafetyDatasetSynthesis(mock_model)
        prompt = "What is the capital of France and Germany"

        variations = synthesizer.generate_adversarial_prompts(
            prompt,
            num_variations=5,
            perturbation_type="reorder",
        )

        assert len(variations) == 5

    def test_create_safety_benchmark(self, mock_model):
        """Test safety benchmark creation."""
        synthesizer = SafetyDatasetSynthesis(mock_model)

        core_prompts = ["Prompt A", "Prompt B"]
        related_prompts = ["Related 1", "Related 2"]

        benchmark = synthesizer.create_safety_benchmark(
            core_prompts=core_prompts,
            related_prompts=related_prompts,
            num_adversarial_per_core=3,
        )

        assert "core" in benchmark
        assert "related" in benchmark
        assert "adversarial" in benchmark
        assert len(benchmark["core"]) == 2
        assert len(benchmark["related"]) == 2
        assert len(benchmark["adversarial"]) >= 2  # At least 2 adversarial (excluding originals)


# Tests for SteeringEvaluationGates


class TestSteeringEvaluationGates:
    """Test steering evaluation gates."""

    def test_gates_initialization(self, mock_model):
        """Test gates initialize."""
        gates = SteeringEvaluationGates(mock_model, device="cpu")
        assert gates.model is mock_model
        assert gates.device == "cpu"

    def test_set_baseline_activations(self, mock_model):
        """Test setting baseline activations."""
        gates = SteeringEvaluationGates(mock_model)
        baseline_prompts = ["Prompt 1", "Prompt 2"]

        gates.set_baseline_activations(baseline_prompts)
        # Should not raise

    def test_check_activation_bounds_valid(self, mock_steering_vectors):
        """Test activation bounds check with valid vectors."""
        gates = SteeringEvaluationGates(None, device="cpu")

        # Scale vectors to be small
        small_vectors = {k: v * 0.5 for k, v in mock_steering_vectors.items()}

        is_valid, violations = gates.check_activation_bounds(
            small_vectors,
            max_magnitude=10.0,
        )

        assert is_valid
        assert len(violations) == 0

    def test_check_activation_bounds_invalid(self, mock_steering_vectors):
        """Test activation bounds check with invalid vectors."""
        gates = SteeringEvaluationGates(None, device="cpu")

        # Scale vectors to be large
        large_vectors = {k: v * 100 for k, v in mock_steering_vectors.items()}

        is_valid, violations = gates.check_activation_bounds(
            large_vectors,
            max_magnitude=1.0,
        )

        assert not is_valid
        assert len(violations) > 0

    def test_check_steering_consistency_high(self, mock_steering_vectors):
        """Test consistency check with consistent vectors."""
        gates = SteeringEvaluationGates(None, device="cpu")

        # Create consistent vectors (similar norms)
        consistent_vectors = {
            k: v.norm() * torch.ones(256) for k, v in mock_steering_vectors.items()
        }

        is_valid, consistency = gates.check_steering_consistency(
            consistent_vectors,
            min_consistency=0.7,
        )

        assert is_valid or consistency >= 0.7

    def test_check_steering_consistency_low(self):
        """Test consistency check with inconsistent vectors."""
        gates = SteeringEvaluationGates(None, device="cpu")

        # Create very inconsistent vectors
        inconsistent = {
            "A": torch.ones(256),
            "B": torch.ones(256) * 10,
            "C": torch.ones(256) * 0.1,
        }

        is_valid, consistency = gates.check_steering_consistency(
            inconsistent,
            min_consistency=0.9,
        )

        assert not is_valid
        assert consistency < 0.9

    def test_check_semantic_preservation_high(self):
        """Test semantic preservation with similar outputs."""
        gates = SteeringEvaluationGates(None, device="cpu")

        output = torch.randn(1, 100)
        similar_output = output + torch.randn(1, 100) * 0.01  # Small noise

        is_valid, similarity = gates.check_semantic_preservation(
            output,
            similar_output,
            min_similarity=0.9,
        )

        assert similarity > 0.9

    def test_check_semantic_preservation_low(self):
        """Test semantic preservation with different outputs."""
        gates = SteeringEvaluationGates(None, device="cpu")

        output1 = torch.randn(1, 100)
        output2 = torch.randn(1, 100)  # Completely different

        is_valid, similarity = gates.check_semantic_preservation(
            output1,
            output2,
            min_similarity=0.9,
        )

        assert similarity < 0.5

    def test_run_all_checks(self, mock_steering_vectors):
        """Test running all checks together."""
        gates = SteeringEvaluationGates(None, device="cpu")

        # Valid vectors
        valid_vectors = {k: v * 0.5 for k, v in mock_steering_vectors.items()}

        output1 = torch.randn(1, 100)
        output2 = output1 + torch.randn(1, 100) * 0.01

        results = gates.run_all_checks(
            valid_vectors,
            base_output=output1,
            steered_output=output2,
        )

        assert "activation_bounds" in results
        assert "consistency" in results
        assert "semantic_preservation" in results
        assert "overall_valid" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
