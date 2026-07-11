"""
tests/test_model_wrapper.py

Robust unit tests for IBHookedTransformer in model_wrapper.py.
Uses a minimal mock HookedTransformer so no real model weights are needed.
Run with: pytest tests/test_model_wrapper.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# ── Minimal mock of HookedTransformer ─────────────────────────────────────────


def _make_mock_model(n_layers=2, n_heads=4, d_head=8, d_model=32, d_mlp=64):
    """
    Build a minimal stand-in for HookedTransformer.
    Only the attributes and methods that IBHookedTransformer actually touches
    are implemented — everything else raises AttributeError immediately,
    making broken assumptions visible.
    """
    cfg = SimpleNamespace(
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        d_model=d_model,
        d_mlp=d_mlp,
        model_name="mock-model",
    )

    # A single real parameter so next(model.parameters()).device works
    _param = nn.Parameter(torch.zeros(1))

    model = MagicMock()
    model.cfg = cfg
    model.parameters = MagicMock(side_effect=lambda: iter([_param]))

    # hooks() context manager: just runs the callable without real hooking
    from contextlib import contextmanager

    @contextmanager
    def _hooks(hook_list):
        yield

    model.hooks = _hooks

    # forward returns a fake output with .logits
    fake_logits = torch.randn(8, 10, d_model)  # [batch, seq, vocab]
    fake_output = SimpleNamespace(logits=fake_logits)
    model.__call__ = MagicMock(return_value=fake_output)

    return model


# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH, N_LAYERS, N_HEADS, D_HEAD, D_MODEL = 8, 2, 4, 8, 32


@pytest.fixture
def mock_model():
    return _make_mock_model(N_LAYERS, N_HEADS, D_HEAD, D_MODEL)


@pytest.fixture
def node_wrapper(mock_model):
    from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

    return IBHookedTransformer(
        model=mock_model,
        batch_size=BATCH,
        device="cpu",
        scope="both",
        level="node",
    )


@pytest.fixture
def neuron_wrapper(mock_model):
    from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

    return IBHookedTransformer(
        model=mock_model,
        batch_size=BATCH,
        device="cpu",
        scope="both",
        level="neuron",
    )


# ── 1. Initialisation ─────────────────────────────────────────────────────────


class TestInit:
    def test_node_level_stored(self, node_wrapper):
        assert node_wrapper.level == "node"

    def test_neuron_level_stored(self, neuron_wrapper):
        assert neuron_wrapper.level == "neuron"

    def test_node_attn_weight_shape(self, node_wrapper):
        for layer_w in node_wrapper.attn_ib_weights:
            assert layer_w.shape == (
                BATCH,
                N_HEADS,
                1,
                1,
            ), f"node attn weight: expected ({BATCH},{N_HEADS},1,1) got {layer_w.shape}"

    def test_neuron_attn_weight_shape(self, neuron_wrapper):
        for layer_w in neuron_wrapper.attn_ib_weights:
            assert layer_w.shape == (
                BATCH,
                N_HEADS,
                1,
                D_HEAD,
            ), f"neuron attn weight: expected ({BATCH},{N_HEADS},1,{D_HEAD}) got {layer_w.shape}"

    def test_node_mlp_weight_shape(self, node_wrapper):
        for layer_w in node_wrapper.mlp_ib_weights:
            assert layer_w.shape == (
                BATCH,
                1,
                1,
            ), f"node MLP weight: expected ({BATCH},1,1) got {layer_w.shape}"

    def test_neuron_mlp_weight_shape(self, neuron_wrapper):
        for layer_w in neuron_wrapper.mlp_ib_weights:
            assert layer_w.shape == (
                BATCH,
                1,
                D_MODEL,
            ), f"neuron MLP weight: expected ({BATCH},1,{D_MODEL}) got {layer_w.shape}"

    def test_correct_number_of_weight_layers(self, node_wrapper, neuron_wrapper):
        assert len(node_wrapper.attn_ib_weights) == N_LAYERS
        assert len(node_wrapper.mlp_ib_weights) == N_LAYERS
        assert len(neuron_wrapper.attn_ib_weights) == N_LAYERS
        assert len(neuron_wrapper.mlp_ib_weights) == N_LAYERS

    def test_weights_are_trainable(self, neuron_wrapper):
        for w in neuron_wrapper.attn_ib_weights:
            assert w.requires_grad
        for w in neuron_wrapper.mlp_ib_weights:
            assert w.requires_grad

    def test_base_model_frozen(self, node_wrapper):
        for p in node_wrapper.model.parameters():
            assert not p.requires_grad

    def test_invalid_level_raises(self, mock_model):
        from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

        with pytest.raises(ValueError):
            IBHookedTransformer(model=mock_model, batch_size=BATCH, device="cpu", level="edge")

    def test_invalid_batch_size_raises(self, mock_model):
        from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

        with pytest.raises(ValueError):
            IBHookedTransformer(model=mock_model, batch_size=0, device="cpu")


# ── 2. Scope wiring ───────────────────────────────────────────────────────────


class TestScope:
    def test_heads_only_scope(self, mock_model):
        from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

        w = IBHookedTransformer(mock_model, BATCH, "cpu", scope="heads", level="node")
        assert len(w.attn_ib_weights) == N_LAYERS
        assert len(w.mlp_ib_weights) == 0

    def test_mlp_only_scope(self, mock_model):
        from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

        w = IBHookedTransformer(mock_model, BATCH, "cpu", scope="mlp", level="node")
        assert len(w.attn_ib_weights) == 0
        assert len(w.mlp_ib_weights) == N_LAYERS

    def test_both_scope(self, node_wrapper):
        assert len(node_wrapper.attn_ib_weights) == N_LAYERS
        assert len(node_wrapper.mlp_ib_weights) == N_LAYERS


# ── 3. extract_node_scores ────────────────────────────────────────────────────


class TestExtractNodeScores:
    def test_returns_correct_keys_heads_and_mlp(self, node_wrapper):
        scores = node_wrapper.extract_node_scores()
        expected_attn = {f"A{lyr}.{h}" for lyr in range(N_LAYERS) for h in range(N_HEADS)}
        expected_mlp = {f"MLP {lyr}" for lyr in range(N_LAYERS)}
        assert set(scores.keys()) == expected_attn | expected_mlp

    def test_scores_in_zero_one_range(self, node_wrapper):
        scores = node_wrapper.extract_node_scores()
        for name, s in scores.items():
            assert 0.0 <= s <= 1.0, f"{name}: score {s} out of [0,1]"

    def test_scores_are_python_floats(self, node_wrapper):
        scores = node_wrapper.extract_node_scores()
        for name, s in scores.items():
            assert isinstance(s, float), f"{name}: expected float, got {type(s)}"

    def test_threshold_binarises_scores(self, node_wrapper):
        scores = node_wrapper.extract_node_scores(threshold=0.5)
        for name, s in scores.items():
            assert s in (0.0, 1.0), f"{name}: threshold didn't binarise, got {s}"

    def test_raises_on_neuron_model(self, neuron_wrapper):
        with pytest.raises(RuntimeError, match="neuron"):
            neuron_wrapper.extract_node_scores()


# ── 4. extract_neuron_scores ──────────────────────────────────────────────────


class TestExtractNeuronScores:
    def test_returns_tensors_not_floats(self, neuron_wrapper):
        scores = neuron_wrapper.extract_neuron_scores()
        for name, t in scores.items():
            assert isinstance(t, torch.Tensor), f"{name}: expected Tensor, got {type(t)}"

    def test_attn_tensor_shape_is_d_head(self, neuron_wrapper):
        scores = neuron_wrapper.extract_neuron_scores()
        for lyr in range(N_LAYERS):
            for h in range(N_HEADS):
                key = f"A{lyr}.{h}"
                assert scores[key].shape == (
                    D_HEAD,
                ), f"{key}: expected ({D_HEAD},) got {scores[key].shape}"

    def test_mlp_tensor_shape_is_d_model(self, neuron_wrapper):
        scores = neuron_wrapper.extract_neuron_scores()
        for lyr in range(N_LAYERS):
            key = f"MLP {lyr}"
            assert scores[key].shape == (
                D_MODEL,
            ), f"{key}: expected ({D_MODEL},) got {scores[key].shape}"

    def test_values_in_zero_one_range(self, neuron_wrapper):
        scores = neuron_wrapper.extract_neuron_scores()
        for name, t in scores.items():
            assert (
                t.min() >= 0.0 and t.max() <= 1.0
            ), f"{name}: values outside [0,1]: min={t.min()}, max={t.max()}"

    def test_returns_correct_number_of_keys(self, neuron_wrapper):
        scores = neuron_wrapper.extract_neuron_scores()
        expected = N_LAYERS * N_HEADS + N_LAYERS  # attn heads + MLPs
        assert len(scores) == expected

    def test_raises_on_node_model(self, node_wrapper):
        with pytest.raises(RuntimeError, match="node"):
            node_wrapper.extract_neuron_scores()

    def test_scores_heads_only_scope(self, mock_model):
        from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

        w = IBHookedTransformer(mock_model, BATCH, "cpu", scope="heads", level="neuron")
        scores = w.extract_neuron_scores()
        # Only attention keys, no MLP keys
        assert all(k.startswith("A") for k in scores)
        assert not any(k.startswith("MLP") for k in scores)

    def test_scores_mlp_only_scope(self, mock_model):
        from circuitkit.backends.ibcircuit.model_wrapper import IBHookedTransformer

        w = IBHookedTransformer(mock_model, BATCH, "cpu", scope="mlp", level="neuron")
        scores = w.extract_neuron_scores()
        assert all(k.startswith("MLP") for k in scores)
        assert not any(k.startswith("A") for k in scores)

    def test_no_grad_on_output(self, neuron_wrapper):
        """Scores are used for ranking — they must not carry a grad_fn."""
        scores = neuron_wrapper.extract_neuron_scores()
        for name, t in scores.items():
            assert not t.requires_grad, f"{name}: tensor should be detached"


# ── 5. get_statistics ─────────────────────────────────────────────────────────


class TestGetStatistics:
    def test_node_stats_keys_present(self, node_wrapper):
        stats = node_wrapper.get_statistics()
        assert "n_attn_heads_important" in stats
        assert "total_attn_heads" in stats
        assert "n_mlps_important" in stats

    def test_node_total_heads_is_correct(self, node_wrapper):
        stats = node_wrapper.get_statistics()
        assert stats["total_attn_heads"] == N_LAYERS * N_HEADS

    def test_node_important_heads_within_total(self, node_wrapper):
        stats = node_wrapper.get_statistics()
        assert 0 <= stats["n_attn_heads_important"] <= stats["total_attn_heads"]

    def test_neuron_important_count_bounded_by_total_neurons(self, neuron_wrapper):
        """
        For neuron-level, n_attn_heads_important counts individual neurons,
        so its ceiling is n_layers * n_heads * d_head (not just n_layers * n_heads).
        """
        stats = neuron_wrapper.get_statistics()
        max_neurons = N_LAYERS * N_HEADS * D_HEAD
        assert 0 <= stats["n_attn_heads_important"] <= max_neurons

    def test_neuron_mlp_count_bounded_by_total_mlp_neurons(self, neuron_wrapper):
        stats = neuron_wrapper.get_statistics()
        max_mlp_neurons = N_LAYERS * D_MODEL
        assert 0 <= stats["n_mlps_important"] <= max_mlp_neurons

    def test_node_stats_never_exceed_n_heads(self, node_wrapper):
        """
        Core regression: node-level must count heads not neurons.
        Forcing all weights high (→ all important) should give exactly n_heads*n_layers.
        """
        for w in node_wrapper.attn_ib_weights:
            w.data.fill_(10.0)  # sigmoid(10) ≈ 1.0 → all important
        stats = node_wrapper.get_statistics()
        assert stats["n_attn_heads_important"] == N_LAYERS * N_HEADS

    def test_neuron_stats_count_neurons_not_heads(self, neuron_wrapper):
        """
        Core regression: neuron-level must count neurons not whole heads.
        Forcing all weights high should give n_layers * n_heads * d_head.
        """
        for w in neuron_wrapper.attn_ib_weights:
            w.data.fill_(10.0)
        stats = neuron_wrapper.get_statistics()
        assert stats["n_attn_heads_important"] == N_LAYERS * N_HEADS * D_HEAD


# ── 6. __repr__ ───────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_level_node(self, node_wrapper):
        assert "level=node" in repr(node_wrapper)

    def test_repr_contains_level_neuron(self, neuron_wrapper):
        assert "level=neuron" in repr(neuron_wrapper)

    def test_repr_contains_scope(self, node_wrapper):
        assert "scope=both" in repr(node_wrapper)
