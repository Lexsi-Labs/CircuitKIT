# FILE: tests/unit/test_steering.py
"""
Unit tests for ActivationSteering module.

Tests cover:
1. Hook point conversion from circuit nodes to transformer_lens points
2. Steering vector computation from source/target examples
3. Steering application during inference
4. Coefficient-based steering strength control
5. Statistics and metadata about steering vectors
"""

import math
from unittest.mock import MagicMock

import pytest
import torch

try:
    from transformer_lens import HookedTransformer

    from circuitkit.applications.steering.steering import ActivationSteering
except ImportError:
    pytest.skip("circuitkit not available", allow_module_level=True)


class TestActivationSteeringInitialization:
    """Test ActivationSteering initialization."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock HookedTransformer."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.device = "cpu"
        return model

    def test_initialization_basic(self, mock_model):
        """Test basic initialization."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.92, "MLP 1": 0.88}

        steering = ActivationSteering(mock_model, circuit_scores)

        assert steering.model == mock_model
        assert steering.circuit_scores == circuit_scores
        assert len(steering.high_score_nodes) == 3

    def test_initialization_with_threshold(self, mock_model):
        """Test initialization with score threshold."""
        circuit_scores = {"A0.0": 0.95, "A0.1": 0.50, "MLP 1": 0.30}

        steering = ActivationSteering(mock_model, circuit_scores, score_threshold=0.6)

        assert len(steering.high_score_nodes) == 1
        assert "A0.0" in steering.high_score_nodes

    def test_initialization_empty_scores(self, mock_model):
        """Test initialization with empty circuit scores."""
        circuit_scores = {}
        steering = ActivationSteering(mock_model, circuit_scores)

        assert len(steering.high_score_nodes) == 0
        assert len(steering.steering_vectors) == 0


class TestHookPointConversion:
    """Test conversion of circuit nodes to hook points."""

    @pytest.fixture
    def steering(self):
        """Create ActivationSteering instance."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.device = "cpu"

        circuit_scores = {"A0.0": 0.95, "MLP 1": 0.88}
        return ActivationSteering(model, circuit_scores)

    def test_attention_hook_point_conversion(self, steering):
        """Test converting attention node to hook point."""
        hook_point = steering._get_hook_point_from_node("A0.0")
        assert hook_point == "blocks.0.attn.hook_result"

        hook_point = steering._get_hook_point_from_node("A5.3")
        assert hook_point == "blocks.5.attn.hook_result"

    def test_mlp_hook_point_conversion(self, steering):
        """Test converting MLP node to hook point."""
        hook_point = steering._get_hook_point_from_node("MLP 1")
        assert hook_point == "blocks.1.hook_mlp_out"

        hook_point = steering._get_hook_point_from_node("MLP 10")
        assert hook_point == "blocks.10.hook_mlp_out"

    def test_invalid_node_name(self, steering):
        """Test handling of invalid node names."""
        hook_point = steering._get_hook_point_from_node("INVALID")
        assert hook_point is None

        hook_point = steering._get_hook_point_from_node("")
        assert hook_point is None


class TestSteeringVectorComputation:
    """Test steering vector computation against a real tiny model.

    These previously wrapped the call in try/except: pass over a MagicMock, so
    the asserts never ran. They now use the real ``tiny_model`` fixture (a
    2-layer HookedTransformer with a gpt2 tokenizer, built once per session) so
    the assertions actually execute.
    """

    def test_steering_vector_computation_structure(self, tiny_model):
        """compute_steering_vector returns a dict keyed by the circuit nodes."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})

        source_examples = [{"text": "the cat sat"}, {"text": "a dog ran"}]
        target_examples = [{"text": "the sky is"}, {"text": "a bird flew"}]

        vectors = steering.compute_steering_vector(
            source_examples, target_examples, batch_size=1
        )
        assert isinstance(vectors, dict)
        assert set(vectors) == {"A0.0"}
        assert all(isinstance(v, torch.Tensor) and v.numel() > 0 for v in vectors.values())

    def test_steering_vector_with_input_ids(self, tiny_model):
        """compute_steering_vector works with pre-tokenized input_ids too."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})

        n = tiny_model.cfg.d_vocab
        source_examples = [{"input_ids": torch.randint(0, n, (16,))} for _ in range(2)]
        target_examples = [{"input_ids": torch.randint(0, n, (16,))} for _ in range(2)]

        vectors = steering.compute_steering_vector(
            source_examples, target_examples, batch_size=2
        )
        assert isinstance(vectors, dict)
        assert set(vectors) == {"A0.0"}


class TestSteeringApplication:
    """Test steering application during inference against a real tiny model.

    Previously each steer() call was wrapped in ``try: ... except Exception:
    pass`` over a MagicMock (which can't forward-pass), so the asserts never
    ran. They now use the real ``tiny_model`` fixture with steering vectors
    computed from it, and the asserts actually execute.
    """

    @staticmethod
    def _steering(tiny_model, scores):
        steering = ActivationSteering(tiny_model, scores)
        steering.steering_vectors = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )
        return steering

    def test_steer_without_vectors(self, tiny_model):
        """steer() with no computed vectors raises ValueError."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})
        with pytest.raises(ValueError):
            steering.steer("test input")

    def test_steer_with_string_input(self, tiny_model):
        """steer() accepts a raw string and echoes the applied coefficient."""
        steering = self._steering(tiny_model, {"A0.0": 0.95})
        result = steering.steer("test input", coefficient=0.5)
        assert "output" in result
        assert "steered_nodes" in result
        assert result["coefficient"] == 0.5

    def test_steer_with_tensor_input(self, tiny_model):
        """steer() accepts pre-tokenized input_ids."""
        steering = self._steering(tiny_model, {"A0.0": 0.95})
        input_ids = torch.randint(0, tiny_model.cfg.d_vocab, (1, 16), device=tiny_model.cfg.device)
        result = steering.steer(input_ids, coefficient=0.75)
        assert "output" in result
        assert result["coefficient"] == 0.75

    def test_steer_with_dict_input(self, tiny_model):
        """steer() accepts a {'text': ...} dict input."""
        steering = self._steering(tiny_model, {"A0.0": 0.95})
        result = steering.steer({"text": "test"}, coefficient=0.5)
        assert "output" in result

    def test_steer_with_dict_coefficients(self, tiny_model):
        """steer() accepts per-node coefficients and echoes them back."""
        steering = self._steering(tiny_model, {"A0.0": 0.95, "A0.1": 0.92})
        coeffs = {"A0.0": 0.8, "A0.1": 0.3}
        result = steering.steer("test", coefficient=coeffs)
        assert result["coefficient"] == coeffs


class TestSteeringCoefficients:
    """Test different coefficient settings."""

    # These previously asserted `steering_vec * coeff == steering_vec * coeff`
    # (pure arithmetic — no ActivationSteering method ran, so they could never
    # fail). They now drive the real steer() path on the tiny_model fixture and
    # verify the coefficient actually gates the applied steering.

    @staticmethod
    def _steering(tiny_model):
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})
        vectors = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )
        return steering, vectors

    def test_coefficient_zero_is_noop(self, tiny_model):
        """coefficient=0.0 must leave the model output identical to no steering."""
        steering, vectors = self._steering(tiny_model)
        ids = torch.randint(0, tiny_model.cfg.d_vocab, (1, 16), device=tiny_model.cfg.device)
        base = tiny_model(ids)
        out0 = steering.steer(ids, steering_vectors=vectors, coefficient=0.0)["output"]
        assert torch.allclose(base, out0, atol=1e-4), "coefficient=0 must be a no-op"

    def test_nonzero_coefficient_changes_output(self, tiny_model):
        """A non-zero coefficient must actually change the logits vs the baseline."""
        steering, vectors = self._steering(tiny_model)
        ids = torch.randint(0, tiny_model.cfg.d_vocab, (1, 16), device=tiny_model.cfg.device)
        base = tiny_model(ids)
        out2 = steering.steer(ids, steering_vectors=vectors, coefficient=2.0)["output"]
        assert not torch.allclose(base, out2, atol=1e-4), "non-zero coefficient must change output"

    def test_coefficient_scales_effect(self, tiny_model):
        """Larger |coefficient| must deviate more from the unsteered baseline."""
        steering, vectors = self._steering(tiny_model)
        ids = torch.randint(0, tiny_model.cfg.d_vocab, (1, 16), device=tiny_model.cfg.device)
        base = tiny_model(ids)
        d1 = (steering.steer(ids, steering_vectors=vectors, coefficient=1.0)["output"] - base).norm()
        d3 = (steering.steer(ids, steering_vectors=vectors, coefficient=3.0)["output"] - base).norm()
        assert d3 > d1, "larger coefficient must deviate more from baseline"


class TestSteeringStatistics:
    """Test steering statistics and metadata."""

    @pytest.fixture
    def steering_with_vectors(self):
        """Create ActivationSteering with steering vectors."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.device = "cpu"

        circuit_scores = {"A0.0": 0.95, "MLP 1": 0.88}
        steering = ActivationSteering(model, circuit_scores)

        # Manually add steering vectors and metadata
        steering.steering_vectors = {"A0.0": torch.randn(768), "MLP 1": torch.randn(768)}

        steering.steering_metadata = {
            "A0.0": {
                "norm": 1.5,
                "shape": (768,),
                "source_mean": torch.randn(768),
                "target_mean": torch.randn(768),
            },
            "MLP 1": {
                "norm": 2.3,
                "shape": (768,),
                "source_mean": torch.randn(768),
                "target_mean": torch.randn(768),
            },
        }

        return steering

    def test_get_steering_statistics(self, steering_with_vectors):
        """Test retrieving steering statistics."""
        stats = steering_with_vectors.get_steering_statistics()

        assert "A0.0" in stats
        assert "MLP 1" in stats
        assert "steering_norm" in stats["A0.0"]
        assert "shape" in stats["A0.0"]
        assert "source_norm" in stats["A0.0"]
        assert "target_norm" in stats["A0.0"]

    def test_steering_statistics_values(self, steering_with_vectors):
        """Test that statistics have reasonable values."""
        stats = steering_with_vectors.get_steering_statistics()

        for node_name, node_stats in stats.items():
            assert node_stats["steering_norm"] >= 0
            assert node_stats["source_norm"] >= 0
            assert node_stats["target_norm"] >= 0
            assert isinstance(node_stats["shape"], tuple)


class TestActivationSteeringEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.device = "cpu"
        model.register_forward_hook = MagicMock(return_value=MagicMock(remove=MagicMock()))
        return model

    def test_invalid_input_type(self, mock_model):
        """Test handling of invalid input types."""
        circuit_scores = {"A0.0": 0.95}
        steering = ActivationSteering(mock_model, circuit_scores)
        steering.steering_vectors = {"A0.0": torch.randn(768)}

        with pytest.raises(TypeError):
            steering.steer(12345)  # Invalid type

    def test_steering_with_batch_dimension(self, tiny_model):
        """steer() handles a batched (B>1) input and preserves the batch dim."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})
        steering.steering_vectors = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )
        input_ids = torch.randint(
            0, tiny_model.cfg.d_vocab, (4, 16), device=tiny_model.cfg.device
        )
        result = steering.steer(input_ids)
        assert "output" in result
        assert result["output"].shape[0] == 4

    def test_steering_without_matching_vectors(self, tiny_model):
        """steer() still runs when only a subset of scored nodes have vectors."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95, "A0.1": 0.92})
        vecs = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )
        # Keep a vector for only one of the scored nodes.
        first = next(iter(vecs))
        steering.steering_vectors = {first: vecs[first]}

        result = steering.steer("test")
        # Should still work, just steering fewer nodes.
        assert "steered_nodes" in result


class TestMultiVectorSteering:
    """Test multi-vector steering functionality."""

    @pytest.fixture
    def steering_with_vectors(self):
        """Create ActivationSteering instance with vectors."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.device = "cpu"

        model._hooks = []

        def register_forward_hook(hook_fn, hook_point):
            handle = MagicMock()
            model._hooks.append((hook_fn, hook_point, handle))
            handle.remove = MagicMock()
            return handle

        model.register_forward_hook = register_forward_hook
        model.__call__ = MagicMock(return_value=torch.randn(1, 64, 50257))

        circuit_scores = {"A0.0": 0.95, "A0.1": 0.92, "MLP 1": 0.88}
        steering = ActivationSteering(model, circuit_scores)

        steering.steering_vectors = {
            "A0.0": torch.randn(768),
            "A0.1": torch.randn(768),
            "MLP 1": torch.randn(768),
        }

        return steering

    def test_multi_vector_steering_sum(self, tiny_model):
        """steer_with_multi_vectors combines two vector sets with sum aggregation."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})
        v1 = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )
        v2 = steering.compute_steering_vector(
            [{"text": "a dog ran"}], [{"text": "the sea is"}], batch_size=1
        )
        result = steering.steer_with_multi_vectors(
            "test", [v1, v2], coefficients=[0.5, 0.5], aggregate="sum"
        )
        assert "output" in result

    def test_multi_vector_steering_mean(self, tiny_model):
        """steer_with_multi_vectors combines vector sets with mean aggregation."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})
        v1 = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )
        v2 = steering.compute_steering_vector(
            [{"text": "a dog ran"}], [{"text": "the sea is"}], batch_size=1
        )
        result = steering.steer_with_multi_vectors("test", [v1, v2], aggregate="mean")
        assert "output" in result

    def test_multi_vector_steering_empty(self, steering_with_vectors):
        """Test multi-vector steering with empty list."""
        with pytest.raises(ValueError):
            steering_with_vectors.steer_with_multi_vectors("test", [])

    def test_multi_vector_coefficient_mismatch(self, steering_with_vectors):
        """Test multi-vector steering with mismatched coefficients."""
        vectors_list = [{"A0.0": torch.randn(768)}, {"A0.0": torch.randn(768)}]
        coefficients = [0.5]  # Wrong length

        with pytest.raises(ValueError):
            steering_with_vectors.steer_with_multi_vectors(
                "test", vectors_list, coefficients=coefficients
            )


class TestSteeringBaselines:
    """Test baseline steering functionality."""

    @pytest.fixture
    def steering_with_vectors(self):
        """Create ActivationSteering instance with vectors."""
        model = MagicMock(spec=HookedTransformer)
        model.cfg = MagicMock()
        model.cfg.d_model = 768
        model.cfg.device = "cpu"

        circuit_scores = {"A0.0": 0.95, "A0.1": 0.92}
        steering = ActivationSteering(model, circuit_scores)

        steering.steering_vectors = {"A0.0": torch.randn(768), "A0.1": torch.randn(768)}

        return steering

    def test_random_baseline(self, steering_with_vectors):
        """Test random baseline vector generation."""
        random_vecs = steering_with_vectors.get_random_baseline_vectors()

        assert len(random_vecs) == 2
        assert "A0.0" in random_vecs
        assert "A0.1" in random_vecs

        # Check shapes match
        for node_name, vec in random_vecs.items():
            original_vec = steering_with_vectors.steering_vectors[node_name]
            assert vec.shape == original_vec.shape

    def test_semantic_baseline_opposite(self, steering_with_vectors):
        """Test opposite semantic baseline."""
        baseline_vecs = steering_with_vectors.get_semantic_baseline_vectors(
            baseline_type="opposite"
        )

        for node_name, baseline_vec in baseline_vecs.items():
            original = steering_with_vectors.steering_vectors[node_name]
            # Should be negated
            assert torch.allclose(baseline_vec, -original)

    def test_semantic_baseline_zero(self, steering_with_vectors):
        """Test zero semantic baseline."""
        baseline_vecs = steering_with_vectors.get_semantic_baseline_vectors(baseline_type="zero")

        for node_name, baseline_vec in baseline_vecs.items():
            # Should be all zeros
            assert torch.allclose(baseline_vec, torch.zeros_like(baseline_vec))

    def test_semantic_baseline_half(self, steering_with_vectors):
        """Test half-strength semantic baseline."""
        baseline_vecs = steering_with_vectors.get_semantic_baseline_vectors(baseline_type="half")

        for node_name, baseline_vec in baseline_vecs.items():
            original = steering_with_vectors.steering_vectors[node_name]
            # Should be 0.5x
            assert torch.allclose(baseline_vec, original * 0.5)

    def test_invalid_baseline_type(self, steering_with_vectors):
        """Test invalid baseline type."""
        with pytest.raises(ValueError):
            steering_with_vectors.get_semantic_baseline_vectors(baseline_type="invalid")


class TestSteeringCoefficientOptimization:
    """Test coefficient optimization."""

    def test_find_steering_coefficient(self, tiny_model):
        """find_steering_coefficient grid-searches and returns a coeff in range."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95})
        steering.steering_vectors = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )

        def metric_fn(logits):
            return logits.mean().item()

        optimal_coeff = steering.find_steering_coefficient(
            "test input", metric_fn, coef_range=(0.0, 1.0), num_steps=5
        )

        assert 0.0 <= optimal_coeff <= 1.0


class TestSteeringNodeImportance:
    """Test node importance analysis."""

    def test_analyze_steering_importance(self, tiny_model):
        """analyze_steering_importance returns a finite importance per node."""
        steering = ActivationSteering(tiny_model, {"A0.0": 0.95, "A0.1": 0.92})
        steering.steering_vectors = steering.compute_steering_vector(
            [{"text": "the cat sat"}], [{"text": "the sky is"}], batch_size=1
        )

        def metric_fn(logits):
            return logits.mean().item()

        importances = steering.analyze_steering_importance(
            "test input", metric_fn, steering.steering_vectors
        )

        # One importance score per steered node.
        assert len(importances) == len(steering.steering_vectors)
        for node_name, importance in importances.items():
            assert math.isfinite(importance)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
