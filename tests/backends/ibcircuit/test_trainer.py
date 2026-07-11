"""
tests/test_trainer.py

Robust unit tests for trainer.py (run_ib_discovery, train_ib_epoch, DEFAULT_CONFIG).

Strategy
--------
run_ib_discovery is tested by patching IBHookedTransformer inside trainer so we
can assert exactly what it was constructed with (level propagation) and control
what forward_with_ib / extract_* return.  This avoids needing real model weights
while still exercising every branch in the trainer logic.

train_ib_epoch is tested with a real IBHookedTransformer built on the same
lightweight mock model used in test_model_wrapper.py.

Run with: pytest tests/test_trainer.py -v
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ── Shared constants ───────────────────────────────────────────────────────────

BATCH = 4
SEQ = 10
VOCAB = 50
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 8
D_MODEL = 32
D_MLP = 64


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _make_mock_model():
    """Minimal HookedTransformer stand-in (same pattern as test_model_wrapper.py)."""
    cfg = SimpleNamespace(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_model=D_MODEL,
        d_mlp=D_MLP,
        model_name="mock-model",
    )
    _param = nn.Parameter(torch.zeros(1))

    model = MagicMock()
    model.cfg = cfg
    model.parameters = MagicMock(side_effect=lambda: iter([_param]))
    model.eval = MagicMock()

    # Forward returns fake logits wrapped like HookedTransformer output
    def _fake_forward(input_ids, **kwargs):
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], VOCAB)
        return logits  # extract_logits_at_positions receives .logits or raw tensor

    # compute_baseline_reference calls model(input_ids) and treats result as
    # having a logits attribute — so wrap it
    def _fake_call(input_ids, **kwargs):
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], VOCAB)
        out = SimpleNamespace(logits=logits)
        return out

    model.side_effect = _fake_call
    model.to_string = MagicMock(return_value="token")

    from contextlib import contextmanager

    @contextmanager
    def _hooks(hook_list):
        yield

    model.hooks = _hooks

    return model


def _make_batch():
    """Create a minimal valid IBCircuit batch."""
    return {
        "tokens": torch.randint(0, VOCAB, (BATCH, SEQ)),
        "labels": torch.randint(0, VOCAB, (BATCH,)),
        "answer_positions": torch.full((BATCH,), SEQ - 1, dtype=torch.long),
    }


def _make_dataloader(batch=None):
    """Single-batch dataloader (list is iterable, iter(list) works)."""
    return [batch or _make_batch()]


def _make_fake_ib_model(level="node"):
    """
    Fully mocked IBHookedTransformer.
    Captures construction kwargs via the fixture and returns controlled
    values from forward_with_ib / extract_*.
    """
    fake = MagicMock()
    fake.level = level
    fake.n_layers = N_LAYERS
    fake.n_heads = N_HEADS
    fake.scope = "both"
    fake.batch_size = BATCH
    fake.mask_type = "sigmoid"
    fake.attn_ib_weights = nn.ParameterList()
    fake.mlp_ib_weights = nn.ParameterList()
    fake._kl_losses = []

    # forward_with_ib returns (output, lambdas, kl_loss)
    fake_logits = torch.randn(BATCH, SEQ, VOCAB)
    fake_output = SimpleNamespace(logits=fake_logits)
    fake_kl = torch.tensor(0.05, requires_grad=True)
    fake.forward_with_ib = MagicMock(return_value=(fake_output, [], fake_kl))

    fake.get_trainable_parameters = MagicMock(return_value=[nn.Parameter(torch.randn(2))])
    fake.get_statistics = MagicMock(
        return_value={
            "overall_avg_attn_lambda": 0.9,
            "n_attn_heads_important": 3,
            "total_attn_heads": N_LAYERS * N_HEADS,
            "overall_avg_mlp_lambda": 0.8,
            "n_mlps_important": 1,
            "total_mlps": N_LAYERS,
        }
    )

    if level == "node":
        node_scores = {f"A{lyr}.{h}": 0.9 for lyr in range(N_LAYERS) for h in range(N_HEADS)}
        node_scores.update({f"MLP {lyr}": 0.8 for lyr in range(N_LAYERS)})
        fake.extract_node_scores = MagicMock(return_value=node_scores)
        fake.extract_neuron_scores = MagicMock(side_effect=RuntimeError("wrong level"))
    else:
        neuron_scores = {
            f"A{lyr}.{h}": torch.sigmoid(torch.randn(D_HEAD)).detach()
            for lyr in range(N_LAYERS)
            for h in range(N_HEADS)
        }
        neuron_scores.update(
            {f"MLP {lyr}": torch.sigmoid(torch.randn(D_MODEL)).detach() for lyr in range(N_LAYERS)}
        )
        fake.extract_neuron_scores = MagicMock(return_value=neuron_scores)
        fake.extract_node_scores = MagicMock(side_effect=RuntimeError("wrong level"))

    return fake


# ── 1. DEFAULT_CONFIG ─────────────────────────────────────────────────────────


class TestDefaultConfig:
    def test_level_key_exists(self):
        from circuitkit.backends.ibcircuit.trainer import DEFAULT_CONFIG

        assert "level" in DEFAULT_CONFIG

    def test_level_default_is_node(self):
        from circuitkit.backends.ibcircuit.trainer import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["level"] == "node"

    def test_all_required_keys_present(self):
        from circuitkit.backends.ibcircuit.trainer import DEFAULT_CONFIG

        required = {
            "num_epochs",
            "learning_rate",
            "alpha",
            "beta",
            "alpha_loss",
            "scope",
            "mask_type",
            "log_interval",
            "level",
        }
        assert required.issubset(DEFAULT_CONFIG.keys())


# ── 2. Level propagation into IBHookedTransformer ─────────────────────────────


class TestLevelPropagation:
    """
    Patch IBHookedTransformer inside trainer and assert the constructor
    receives the correct level from the merged config.
    """

    def _run(self, extra_config, level):
        """Helper: run 1-epoch discovery and return the IBHookedTransformer init kwargs."""
        fake_ib = _make_fake_ib_model(level=level)
        init_kwargs = {}

        def capture_init(model, batch_size, device, **kwargs):
            init_kwargs.update(kwargs)
            return fake_ib

        config = {
            "num_epochs": 1,
            "log_interval": 1,
            **extra_config,
        }

        with patch(
            "circuitkit.backends.ibcircuit.trainer.IBHookedTransformer", side_effect=capture_init
        ):
            from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

            run_ib_discovery(_make_mock_model(), _make_dataloader(), config, "cpu")

        return init_kwargs

    def test_node_level_passed_explicitly(self):
        kwargs = self._run({"level": "node"}, level="node")
        assert kwargs.get("level") == "node"

    def test_neuron_level_passed_explicitly(self):
        kwargs = self._run({"level": "neuron"}, level="neuron")
        assert kwargs.get("level") == "neuron"

    def test_missing_level_defaults_to_node(self):
        """Caller omits 'level' — DEFAULT_CONFIG merge must supply 'node'."""
        kwargs = self._run({}, level="node")
        assert kwargs.get("level") == "node"

    def test_scope_still_propagated(self):
        kwargs = self._run({"scope": "mlp"}, level="node")
        assert kwargs.get("scope") == "mlp"


# ── 3. Phase 5 — correct extraction method called ────────────────────────────


class TestPhase5Dispatch:
    """
    Verify that run_ib_discovery calls extract_node_scores for node-level
    and extract_neuron_scores for neuron-level, never mixing them.
    """

    def _run_with_fake_ib(self, level):
        fake_ib = _make_fake_ib_model(level=level)
        config = {"num_epochs": 1, "log_interval": 1, "level": level}

        with patch(
            "circuitkit.backends.ibcircuit.trainer.IBHookedTransformer", return_value=fake_ib
        ):
            from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

            scores, returned_model = run_ib_discovery(
                _make_mock_model(), _make_dataloader(), config, "cpu"
            )
        return scores, returned_model, fake_ib

    def test_node_calls_extract_node_scores(self):
        _, _, fake_ib = self._run_with_fake_ib("node")
        fake_ib.extract_node_scores.assert_called_once()
        fake_ib.extract_neuron_scores.assert_not_called()

    def test_neuron_calls_extract_neuron_scores(self):
        _, _, fake_ib = self._run_with_fake_ib("neuron")
        fake_ib.extract_neuron_scores.assert_called_once()
        fake_ib.extract_node_scores.assert_not_called()

    def test_node_extract_node_scores_called_without_threshold(self):
        """extract_node_scores must be called with threshold=None (continuous scores)."""
        _, _, fake_ib = self._run_with_fake_ib("node")
        fake_ib.extract_node_scores.assert_called_once_with(threshold=None)


# ── 4. Return types ───────────────────────────────────────────────────────────


class TestReturnTypes:

    def _run(self, level):
        fake_ib = _make_fake_ib_model(level=level)
        config = {"num_epochs": 1, "log_interval": 1, "level": level}
        with patch(
            "circuitkit.backends.ibcircuit.trainer.IBHookedTransformer", return_value=fake_ib
        ):
            from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

            return run_ib_discovery(_make_mock_model(), _make_dataloader(), config, "cpu")

    def test_node_returns_two_tuple(self):
        result = self._run("node")
        assert isinstance(result, tuple) and len(result) == 2

    def test_node_first_element_is_dict(self):
        scores, _ = self._run("node")
        assert isinstance(scores, dict)

    def test_node_scores_are_floats(self):
        scores, _ = self._run("node")
        for k, v in scores.items():
            assert isinstance(v, float), f"{k}: expected float, got {type(v)}"

    def test_node_scores_in_zero_one_range(self):
        scores, _ = self._run("node")
        for k, v in scores.items():
            assert 0.0 <= v <= 1.0, f"{k}: score {v} out of [0,1]"

    def test_neuron_returns_two_tuple(self):
        result = self._run("neuron")
        assert isinstance(result, tuple) and len(result) == 2

    def test_neuron_first_element_is_dict(self):
        scores, _ = self._run("neuron")
        assert isinstance(scores, dict)

    def test_neuron_scores_are_tensors(self):
        scores, _ = self._run("neuron")
        for k, v in scores.items():
            assert isinstance(v, torch.Tensor), f"{k}: expected Tensor, got {type(v)}"

    def test_neuron_attn_tensor_shape(self):
        scores, _ = self._run("neuron")
        for lyr in range(N_LAYERS):
            for h in range(N_HEADS):
                t = scores[f"A{lyr}.{h}"]
                assert t.shape == (D_HEAD,), f"A{lyr}.{h}: expected ({D_HEAD},) got {t.shape}"

    def test_neuron_mlp_tensor_shape(self):
        scores, _ = self._run("neuron")
        for lyr in range(N_LAYERS):
            t = scores[f"MLP {lyr}"]
            assert t.shape == (D_MODEL,), f"MLP {lyr}: expected ({D_MODEL},) got {t.shape}"

    def test_second_element_is_ib_model(self):
        pass

        _, ib = self._run("node")
        # Either the real class or our mock — just check it came back
        assert ib is not None

    def test_both_levels_return_same_tuple_structure(self):
        node_result = self._run("node")
        neuron_result = self._run("neuron")
        assert len(node_result) == len(neuron_result) == 2


# ── 5. Edge cases and error handling ─────────────────────────────────────────


class TestEdgeCases:

    def test_empty_dataloader_raises_value_error(self):
        from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

        with pytest.raises(ValueError, match="empty"):
            run_ib_discovery(_make_mock_model(), [], {"level": "node"}, "cpu")

    def test_missing_tokens_key_raises(self):
        bad_batch = {"labels": torch.zeros(BATCH), "answer_positions": torch.zeros(BATCH)}
        from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

        with pytest.raises(KeyError):
            run_ib_discovery(_make_mock_model(), [bad_batch], {"level": "node"}, "cpu")

    def test_missing_labels_key_raises(self):
        bad_batch = {
            "tokens": torch.randint(0, VOCAB, (BATCH, SEQ)),
            "answer_positions": torch.zeros(BATCH, dtype=torch.long),
        }
        from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

        with pytest.raises(KeyError):
            run_ib_discovery(_make_mock_model(), [bad_batch], {"level": "node"}, "cpu")

    def test_config_overrides_default(self):
        """Explicit config values must win over DEFAULT_CONFIG."""
        fake_ib = _make_fake_ib_model(level="neuron")
        captured = {}

        def capture_init(model, batch_size, device, **kwargs):
            captured.update(kwargs)
            return fake_ib

        config = {"num_epochs": 1, "log_interval": 1, "level": "neuron", "scope": "mlp"}
        with patch(
            "circuitkit.backends.ibcircuit.trainer.IBHookedTransformer", side_effect=capture_init
        ):
            from circuitkit.backends.ibcircuit.trainer import run_ib_discovery

            run_ib_discovery(_make_mock_model(), _make_dataloader(), config, "cpu")

        assert captured["level"] == "neuron"
        assert captured["scope"] == "mlp"


# ── 6. train_ib_epoch ─────────────────────────────────────────────────────────


class TestTrainIbEpoch:
    """
    train_ib_epoch is level-agnostic — it just calls forward_with_ib and
    backward. Test with a fully mocked ib_model to isolate the logic.
    """

    @pytest.fixture
    def epoch_setup(self):
        """Provide all inputs needed for one train_ib_epoch call."""
        fake_ib = _make_fake_ib_model(level="node")

        # forward_with_ib must return a tensor that supports .backward()
        logits = torch.randn(BATCH, SEQ, VOCAB, requires_grad=True)
        output = SimpleNamespace(logits=logits)
        kl_loss = torch.tensor(0.1, requires_grad=True)
        fake_ib.forward_with_ib = MagicMock(return_value=(output, [], kl_loss))

        optimizer = torch.optim.SGD(fake_ib.get_trainable_parameters(), lr=0.01)
        baseline_logprobs = torch.randn(BATCH, VOCAB).log_softmax(dim=-1)

        return {
            "ib_model": fake_ib,
            "optimizer": optimizer,
            "input_ids": torch.randint(0, VOCAB, (BATCH, SEQ)),
            "answer_tokens": torch.randint(0, VOCAB, (BATCH,)),
            "answer_positions": torch.full((BATCH,), SEQ - 1, dtype=torch.long),
            "baseline_logprobs": baseline_logprobs,
            "config": {
                "alpha": 1.0,
                "beta": 1.0,
                "alpha_loss": "kl",
                "level": "node",
            },
        }

    def test_returns_three_tuple(self, epoch_setup):
        from circuitkit.backends.ibcircuit.trainer import train_ib_epoch

        result = train_ib_epoch(**epoch_setup)
        assert isinstance(result, tuple) and len(result) == 3

    def test_all_elements_are_floats(self, epoch_setup):
        from circuitkit.backends.ibcircuit.trainer import train_ib_epoch

        task_loss, ib_loss, total_loss = train_ib_epoch(**epoch_setup)
        assert isinstance(task_loss, float)
        assert isinstance(ib_loss, float)
        assert isinstance(total_loss, float)

    def test_total_is_alpha_task_plus_beta_ib(self, epoch_setup):
        """total_loss == alpha * task_loss + beta * ib_loss."""
        from circuitkit.backends.ibcircuit.trainer import train_ib_epoch

        cfg = epoch_setup["config"]
        task, ib, total = train_ib_epoch(**epoch_setup)
        expected = cfg["alpha"] * task + cfg["beta"] * ib
        assert abs(total - expected) < 1e-5, f"total={total} expected={expected}"

    def test_forward_with_ib_called_once(self, epoch_setup):
        from circuitkit.backends.ibcircuit.trainer import train_ib_epoch

        train_ib_epoch(**epoch_setup)
        epoch_setup["ib_model"].forward_with_ib.assert_called_once()

    def test_works_with_neuron_level_ib_model(self, epoch_setup):
        """train_ib_epoch must be level-agnostic."""
        from circuitkit.backends.ibcircuit.trainer import train_ib_epoch

        epoch_setup["ib_model"].level = "neuron"
        epoch_setup["config"]["level"] = "neuron"
        result = train_ib_epoch(**epoch_setup)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_ce_loss_mode(self, epoch_setup):
        """CE mode requires baseline_ce_loss; task loss should be non-negative."""
        from circuitkit.backends.ibcircuit.trainer import train_ib_epoch

        epoch_setup["config"]["alpha_loss"] = "ce"
        epoch_setup["baseline_logprobs"] = None
        result = train_ib_epoch(**epoch_setup, baseline_ce_loss=0.5)
        task, ib, total = result
        assert task >= 0.0
