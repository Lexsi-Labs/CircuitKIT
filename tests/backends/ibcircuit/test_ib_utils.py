"""
Comprehensive test suite for ib_utils.py.

This module thoroughly tests all utility functions used in IBCircuit training,
including edge cases, error conditions, and numerical correctness.
"""

import sys

import pytest
import torch
import torch.nn.functional as F


# Mock HookedTransformer for testing
class MockHookedTransformer:
    """Minimal mock for testing baseline computation."""

    class Config:
        def __init__(self):
            self.model_name = "mock-gpt2"

    def __init__(self):
        self.cfg = self.Config()

    def eval(self):
        """Set to eval mode."""

    def __call__(self, input_ids):
        """Return mock output with predictable logits."""
        batch_size, seq_len = input_ids.shape

        class MockOutput:
            def __init__(self, batch_size, seq_len, vocab_size=50257):
                # Use deterministic logits for testing
                torch.manual_seed(42)
                self.logits = torch.randn(batch_size, seq_len, vocab_size)

        return MockOutput(batch_size, seq_len)


class TestExtractLogitsAtPositions:
    """Test suite for extract_logits_at_positions()."""

    def test_basic_extraction(self):
        """Test basic position extraction with normal inputs."""
        from circuitkit.backends.ibcircuit.ib_utils import extract_logits_at_positions

        # Create test data
        batch_size, seq_len, vocab_size = 4, 20, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        positions = torch.tensor([5, 10, 15, 18])

        # Extract
        result = extract_logits_at_positions(logits, positions)

        # Verify shape
        assert result.shape == (
            batch_size,
            vocab_size,
        ), f"Expected shape ({batch_size}, {vocab_size}), got {result.shape}"

        # Verify correctness: result[i] should equal logits[i, positions[i], :]
        for i in range(batch_size):
            expected = logits[i, positions[i], :]
            actual = result[i]
            assert torch.allclose(
                actual, expected
            ), f"Mismatch at batch {i}: position {positions[i]}"

    def test_edge_positions(self):
        """Test extraction at edge positions (first and last)."""
        from circuitkit.backends.ibcircuit.ib_utils import extract_logits_at_positions

        logits = torch.randn(3, 10, 50)

        # Test first position
        positions_first = torch.tensor([0, 0, 0])
        result = extract_logits_at_positions(logits, positions_first)
        assert result.shape == (3, 50)
        for i in range(3):
            assert torch.allclose(result[i], logits[i, 0, :])

        # Test last position
        positions_last = torch.tensor([9, 9, 9])
        result = extract_logits_at_positions(logits, positions_last)
        assert result.shape == (3, 50)
        for i in range(3):
            assert torch.allclose(result[i], logits[i, 9, :])

    def test_different_positions_per_batch(self):
        """Test that different positions work correctly per batch element."""
        from circuitkit.backends.ibcircuit.ib_utils import extract_logits_at_positions

        logits = torch.randn(5, 15, 30)
        positions = torch.tensor([0, 3, 7, 10, 14])  # All different

        result = extract_logits_at_positions(logits, positions)

        # Verify each extraction is correct
        for i in range(5):
            assert torch.allclose(result[i], logits[i, positions[i], :])

    def test_single_batch_element(self):
        """Test with batch size of 1."""
        from circuitkit.backends.ibcircuit.ib_utils import extract_logits_at_positions

        logits = torch.randn(1, 20, 100)
        positions = torch.tensor([10])

        result = extract_logits_at_positions(logits, positions)

        assert result.shape == (1, 100)
        assert torch.allclose(result[0], logits[0, 10, :])

    def test_large_batch(self):
        """Test with large batch size."""
        from circuitkit.backends.ibcircuit.ib_utils import extract_logits_at_positions

        batch_size = 128
        logits = torch.randn(batch_size, 30, 50257)
        positions = torch.randint(0, 30, (batch_size,))

        result = extract_logits_at_positions(logits, positions)

        assert result.shape == (batch_size, 50257)
        # Spot check a few random elements
        for i in [0, 50, 100, 127]:
            assert torch.allclose(result[i], logits[i, positions[i], :])

    def test_gradient_flow(self):
        """Test that gradients flow through correctly."""
        from circuitkit.backends.ibcircuit.ib_utils import extract_logits_at_positions

        logits = torch.randn(4, 10, 20, requires_grad=True)
        positions = torch.tensor([2, 5, 7, 9])

        result = extract_logits_at_positions(logits, positions)
        loss = result.sum()
        loss.backward()

        # Verify gradients exist
        assert logits.grad is not None

        # Verify gradients only at extracted positions
        for i in range(4):
            pos = positions[i]
            # Gradient should be 1 at extracted position
            assert torch.allclose(logits.grad[i, pos, :], torch.ones(20))
            # Gradient should be 0 at other positions
            for j in range(10):
                if j != pos:
                    assert torch.allclose(logits.grad[i, j, :], torch.zeros(20))


class TestComputeBaselineReference:
    """Test suite for compute_baseline_reference()."""

    def test_basic_baseline_computation(self):
        """Test basic baseline computation."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_baseline_reference

        model = MockHookedTransformer()
        input_ids = torch.randint(0, 1000, (10, 25))
        answer_positions = torch.randint(10, 20, (10,))

        baseline = compute_baseline_reference(model, input_ids, answer_positions, "cpu")

        # Verify shape
        assert baseline.shape == (10, 50257), f"Wrong shape: {baseline.shape}"

        # Verify it's log probabilities (should sum to ~1 when exp'd)
        probs = torch.exp(baseline)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(
            prob_sums, torch.ones(10), atol=1e-5
        ), "Log probabilities don't sum to 1 when exponentiated"

    def test_deterministic_output(self):
        """Test that baseline computation is deterministic."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_baseline_reference

        model = MockHookedTransformer()
        input_ids = torch.randint(0, 1000, (5, 15))
        answer_positions = torch.tensor([10, 11, 12, 13, 14])

        # Compute twice
        baseline1 = compute_baseline_reference(model, input_ids, answer_positions, "cpu")
        baseline2 = compute_baseline_reference(model, input_ids, answer_positions, "cpu")

        # Should be identical (model is deterministic with seed)
        assert torch.allclose(baseline1, baseline2), "Baseline computation should be deterministic"

    def test_no_gradients(self):
        """Test that baseline computation doesn't track gradients."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_baseline_reference

        model = MockHookedTransformer()
        input_ids = torch.randint(0, 1000, (5, 15))
        answer_positions = torch.tensor([5, 6, 7, 8, 9])

        baseline = compute_baseline_reference(model, input_ids, answer_positions, "cpu")

        # Should not require gradients
        assert (
            not baseline.requires_grad
        ), "Baseline should not require gradients (computed with no_grad)"

    def test_single_example(self):
        """Test with single example."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_baseline_reference

        model = MockHookedTransformer()
        input_ids = torch.randint(0, 1000, (1, 20))
        answer_positions = torch.tensor([15])

        baseline = compute_baseline_reference(model, input_ids, answer_positions, "cpu")

        assert baseline.shape == (1, 50257)
        assert torch.allclose(torch.exp(baseline).sum(), torch.tensor(1.0), atol=1e-5)


class TestComputeTaskLoss:
    """Test suite for compute_task_loss()."""

    def test_kl_mode_basic(self):
        """Test KL divergence mode with basic inputs."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        # Create test data
        ib_logits = torch.randn(10, 100)
        answer_tokens = torch.randint(0, 100, (10,))

        # Create baseline (must be log probabilities)
        baseline_logits = torch.randn(10, 100)
        baseline_logprobs = F.log_softmax(baseline_logits, dim=-1)

        # Compute loss
        loss = compute_task_loss(ib_logits, answer_tokens, baseline_logprobs, "kl")

        # Verify it's a scalar
        assert loss.numel() == 1, "Loss should be scalar"

        # KL divergence should be non-negative
        assert loss.item() >= 0, f"KL loss should be >= 0, got {loss.item()}"

    def test_kl_mode_identical_distributions(self):
        """Test KL mode when distributions are identical (should be ~0)."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        # Same logits → same distribution → KL = 0
        logits = torch.randn(5, 50)
        baseline_logprobs = F.log_softmax(logits, dim=-1)
        answer_tokens = torch.randint(0, 50, (5,))

        loss = compute_task_loss(
            logits.clone(), answer_tokens, baseline_logprobs, "kl"  # Same distribution
        )

        # Should be very close to 0
        assert (
            loss.item() < 1e-5
        ), f"KL between identical distributions should be ~0, got {loss.item()}"

    def test_kl_mode_gradient_flow(self):
        """Test that gradients flow through KL loss."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        ib_logits = torch.randn(5, 30, requires_grad=True)
        baseline_logprobs = F.log_softmax(torch.randn(5, 30), dim=-1)
        answer_tokens = torch.randint(0, 30, (5,))

        loss = compute_task_loss(ib_logits, answer_tokens, baseline_logprobs, "kl")
        loss.backward()

        assert ib_logits.grad is not None, "Gradients should flow through KL loss"
        assert not torch.isnan(ib_logits.grad).any(), "Gradients should not be NaN"

    def test_ce_mode_basic(self):
        """Test cross-entropy mode with basic inputs."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        ib_logits = torch.randn(10, 100)
        answer_tokens = torch.randint(0, 100, (10,))
        baseline_ce_loss = 2.5

        loss = compute_task_loss(
            ib_logits,
            answer_tokens,
            None,  # Not needed for CE mode
            "ce",
            baseline_ce_loss=baseline_ce_loss,
        )

        # Verify it's a scalar
        assert loss.numel() == 1, "Loss should be scalar"

        # Should be absolute difference from baseline
        assert loss.item() >= 0, "CE loss should be >= 0"

    def test_ce_mode_identical_to_baseline(self):
        """Test CE mode when current loss equals baseline (should be ~0)."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        # Create logits that produce a specific CE loss
        ib_logits = torch.randn(5, 50)
        answer_tokens = torch.randint(0, 50, (5,))

        # Compute what the baseline loss would be
        baseline_loss = F.cross_entropy(ib_logits, answer_tokens).item()

        # Use same logits (should give same CE loss)
        loss = compute_task_loss(
            ib_logits, answer_tokens, None, "ce", baseline_ce_loss=baseline_loss
        )

        # Should be very close to 0
        assert loss.item() < 1e-5, f"CE loss difference should be ~0, got {loss.item()}"

    def test_ce_mode_gradient_flow(self):
        """Test that gradients flow through CE loss."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        ib_logits = torch.randn(5, 30, requires_grad=True)
        answer_tokens = torch.randint(0, 30, (5,))

        loss = compute_task_loss(ib_logits, answer_tokens, None, "ce", baseline_ce_loss=2.0)
        loss.backward()

        assert ib_logits.grad is not None, "Gradients should flow through CE loss"
        assert not torch.isnan(ib_logits.grad).any(), "Gradients should not be NaN"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        ib_logits = torch.randn(5, 30)
        answer_tokens = torch.randint(0, 30, (5,))
        baseline_logprobs = F.log_softmax(torch.randn(5, 30), dim=-1)

        with pytest.raises(ValueError, match="Invalid loss_mode"):
            compute_task_loss(ib_logits, answer_tokens, baseline_logprobs, "invalid_mode")

    def test_kl_missing_baseline_raises_error(self):
        """Test that KL mode without baseline raises ValueError."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        ib_logits = torch.randn(5, 30)
        answer_tokens = torch.randint(0, 30, (5,))

        with pytest.raises(ValueError, match="baseline_logprobs required"):
            compute_task_loss(ib_logits, answer_tokens, None, "kl")

    def test_ce_missing_baseline_raises_error(self):
        """Test that CE mode without baseline loss raises ValueError."""
        from circuitkit.backends.ibcircuit.ib_utils import compute_task_loss

        ib_logits = torch.randn(5, 30)
        answer_tokens = torch.randint(0, 30, (5,))

        with pytest.raises(ValueError, match="baseline_ce_loss required"):
            compute_task_loss(ib_logits, answer_tokens, None, "ce")


class TestValidateBatchData:
    """Test suite for validate_batch_data()."""

    def test_valid_batch(self):
        """Test that valid batch passes validation."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        batch = {
            "tokens": torch.randint(0, 100, (10, 20)),
            "labels": torch.randint(0, 100, (10,)),
            "answer_positions": torch.randint(0, 15, (10,)),
        }

        # Should not raise any exception
        validate_batch_data(batch)

    def test_missing_keys_raises_error(self):
        """Test that missing required keys raise KeyError."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        # Missing 'labels'
        batch = {
            "tokens": torch.randint(0, 100, (10, 20)),
            "answer_positions": torch.randint(0, 15, (10,)),
        }

        with pytest.raises(KeyError, match="missing required keys"):
            validate_batch_data(batch)

    def test_inconsistent_batch_sizes_raises_error(self):
        """Test that inconsistent batch sizes raise ValueError."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        # tokens has batch size 10, labels has batch size 5
        batch = {
            "tokens": torch.randint(0, 100, (10, 20)),
            "labels": torch.randint(0, 100, (5,)),  # Wrong size!
            "answer_positions": torch.randint(0, 15, (10,)),
        }

        with pytest.raises(ValueError, match="Inconsistent batch sizes"):
            validate_batch_data(batch)

    def test_invalid_answer_positions_raises_error(self):
        """Test that answer positions beyond sequence length raise ValueError."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        # Sequence length is 20, but max position is 25
        batch = {
            "tokens": torch.randint(0, 100, (10, 20)),
            "labels": torch.randint(0, 100, (10,)),
            "answer_positions": torch.tensor([5, 10, 25, 15, 8, 12, 18, 3, 7, 19]),
        }

        with pytest.raises(ValueError, match="Invalid answer_positions"):
            validate_batch_data(batch)

    def test_custom_required_keys(self):
        """Test validation with custom required keys."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        batch = {
            "input_ids": torch.randint(0, 100, (5, 10)),
            "targets": torch.randint(0, 100, (5,)),
        }

        # Should pass with custom keys
        validate_batch_data(batch, required_keys=["input_ids", "targets"])

        # Should fail if custom key is missing
        with pytest.raises(KeyError):
            validate_batch_data(batch, required_keys=["input_ids", "missing_key"])

    def test_edge_case_single_example(self):
        """Test validation with single example."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        batch = {
            "tokens": torch.randint(0, 100, (1, 20)),
            "labels": torch.randint(0, 100, (1,)),
            "answer_positions": torch.tensor([10]),
        }

        # Should pass
        validate_batch_data(batch)

    def test_edge_case_last_position(self):
        """Test that last valid position is accepted."""
        from circuitkit.backends.ibcircuit.ib_utils import validate_batch_data

        seq_len = 20
        batch = {
            "tokens": torch.randint(0, 100, (5, seq_len)),
            "labels": torch.randint(0, 100, (5,)),
            "answer_positions": torch.tensor([seq_len - 1] * 5),  # Last valid position
        }

        # Should pass
        validate_batch_data(batch)


def run_all_tests():
    """Run all ib_utils tests and report results."""
    print("\n" + "=" * 70)
    print("IB_UTILS COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    test_classes = [
        TestExtractLogitsAtPositions,
        TestComputeBaselineReference,
        TestComputeTaskLoss,
        TestValidateBatchData,
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)

        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith("test_")]

        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"  ✅ {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  ❌ {method_name}: {e}")
                total_failed += 1
                import traceback

                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    total_tests = total_passed + total_failed
    print(f"Passed: {total_passed}/{total_tests}")
    print(f"Failed: {total_failed}/{total_tests}")

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n❌ {total_failed} test(s) failed")
        return False


if __name__ == "__main__":
    # Install pytest if needed for better error messages
    try:
        import pytest  # noqa: F811 - intentional re-import for script-mode fallback
    except ImportError:
        print("Note: pytest not available, using basic error handling")

        # Define a simple pytest.raises context manager
        class SimpleRaises:
            def __init__(self, exception_type, match=None):
                self.exception_type = exception_type
                self.match = match

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(
                        f"Expected {self.exception_type.__name__} but no exception was raised"
                    )
                if not issubclass(exc_type, self.exception_type):
                    return False  # Re-raise the exception
                if self.match and self.match not in str(exc_val):
                    raise AssertionError(
                        f"Expected exception message to contain '{self.match}', " f"got '{exc_val}'"
                    )
                return True  # Suppress the exception

        # Replace pytest.raises with our simple version
        class MockPytest:
            raises = SimpleRaises

        pytest = MockPytest()

    success = run_all_tests()
    sys.exit(0 if success else 1)
