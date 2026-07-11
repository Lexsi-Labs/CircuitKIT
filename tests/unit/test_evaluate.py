"""
Unit tests for evaluate.py — evaluate_graph, evaluate_baseline, evaluate_ibcircuit_neuron_circuit.

Strategy:
  These functions depend heavily on HookedTransformer, Graph, and custom dataloaders.
  We test them via lightweight mocks that replicate the interface contracts rather
  than loading real models.  This validates argument validation, control-flow
  branches (intervention types, metric dispatch, answer_spans, skip_clean),
  and output shapes/types.

Covers:
- evaluate_graph
    - assertion on use_attn_result
    - assertion on ungroup_grouped_query_attention
    - all intervention types (patching, zero, mean, mean-positional)
    - mean intervention without dataloader raises
    - single vs list metrics
    - answer_spans forwarding
    - skip_clean flag
    - quiet flag
    - output shape (single metric → 1-D tensor, multi metric → list of tensors)
- evaluate_baseline
    - single vs list metrics
    - run_corrupted flag
    - answer_spans forwarding
    - output shape
- evaluate_ibcircuit_neuron_circuit
    - mean / zero / patching interventions
    - single vs list metrics
    - answer_spans forwarding
    - empty pruning_dict
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor

try:
    pass
except ImportError:
    sys.modules["transformer_lens"] = MagicMock()
    # Mock any other heavy dependencies here if they break CI collection

# Centralized imports
from circuitkit.evaluation.evaluate import (
    evaluate_baseline,
    evaluate_graph,
    evaluate_ibcircuit_neuron_circuit,
)

# ---------------------------------------------------------------------------
# We patch out heavy imports so the test file can be imported without
# transformer_lens or the full circuitkit backend installed in CI.
# If you *do* have the full env, the real imports work too.
# ---------------------------------------------------------------------------

# ═══════════════════════════════════════════════════════════════════════════
# Mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════


class MockModelCfg:
    """Mimics model.cfg attributes used in evaluate.py."""

    def __init__(self, **overrides):
        self.use_attn_result = True
        self.n_key_value_heads = None
        self.ungroup_grouped_query_attention = False
        self.device = "cpu"
        self.dtype = torch.float32
        self.n_layers = 2
        self.n_heads = 2
        self.d_model = 16
        self.use_normalization_before_and_after = False
        self.n_ctx = 1024
        for k, v in overrides.items():
            setattr(self, k, v)


class MockModel:
    """Lightweight stand-in for HookedTransformer."""

    def __init__(self, cfg=None):
        self.cfg = cfg or MockModelCfg()
        self.tokenizer = MagicMock()
        self.tokenizer.padding_side = "right"

        # --- ADD THESE 3 LINES ---
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2
        # -------------------------

    def to_tokens(self, text, prepend_bos=True, **kwargs):
        """Mock tokenization returning a dummy tensor."""
        batch_size = 1 if isinstance(text, str) else len(text)
        seq_len = 5  # Arbitrary sequence length for the mock
        return torch.randint(0, 100, (batch_size, seq_len))

    def __call__(self, tokens, attention_mask=None):
        batch = tokens.shape[0]
        # Return dummy logits: [batch, seq, vocab]
        return torch.randn(batch, tokens.shape[1], 50)

    def hooks(self, hook_list):
        """Context manager that does nothing — hooks are ignored in mock."""
        return _NullCtx()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _make_simple_metric(logits, clean_logits, input_lengths, labels):
    """A trivial metric returning a per-sample score."""
    return torch.ones(logits.shape[0])


def _make_metric_with_spans(logits, clean_logits, input_lengths, labels, answer_spans=None):
    """Metric that accepts answer_spans."""
    return torch.ones(logits.shape[0]) * (2.0 if answer_spans is not None else 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Mock Graph (minimal interface expected by evaluate_graph)
# ═══════════════════════════════════════════════════════════════════════════


class _MockEdge:
    def __init__(self, in_graph=True):
        self.in_graph = in_graph


class _MockNode:
    def __init__(self, name, layer=0, in_graph=True, n_heads=2):
        self.name = name
        self.layer = layer
        self.in_graph = in_graph
        self.score = 0.5
        self.parent_edges = [_MockEdge(True)]
        self.in_hook = f"blocks.{layer}.hook_mlp_out"
        self.out_hook = f"blocks.{layer}.hook_mlp_out"
        self.qkv_inputs = [
            f"blocks.{layer}.hook_q_input",
            f"blocks.{layer}.hook_k_input",
            f"blocks.{layer}.hook_v_input",
        ]


class MockGraph:
    """Minimal graph that satisfies evaluate_graph's interface."""

    def __init__(self, n_layers=2, n_heads=2, d_model=16):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.nodes = {}
        # Build nodes for each layer
        for layer in range(n_layers):
            for head in range(n_heads):
                name = f"a{layer}.h{head}"
                self.nodes[name] = _MockNode(name, layer=layer)
            mlp_name = f"m{layer}"
            self.nodes[mlp_name] = _MockNode(mlp_name, layer=layer)
        self.nodes["logits"] = _MockNode("logits", layer=n_layers)

        self.n_forward = len(self.nodes)
        self.n_backward = len(self.nodes)

        n = len(self.nodes)
        self.in_graph = torch.ones(n, n)
        self.neurons_in_graph = None

    def prune(self):
        pass

    def forward_index(self, node):
        keys = list(self.nodes.keys())
        return keys.index(node.name) if node.name in keys else 0

    def backward_index(self, node, qkv=None, attn_slice=False):
        return 0

    def prev_index(self, node):
        return self.n_forward


# ═══════════════════════════════════════════════════════════════════════════
# Fake tokenize / hooks factories — we patch these at call-site
# ═══════════════════════════════════════════════════════════════════════════


def _fake_tokenize_batch_pair(model, clean, corrupted, pair_padding_side=None, templated=False):
    batch = len(clean)
    seq_len = 5
    tokens = torch.randint(0, 100, (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    input_lengths = torch.full((batch,), seq_len, dtype=torch.long)
    return tokens, tokens.clone(), mask, mask.clone(), input_lengths, seq_len


def _fake_tokenize_batch_pair_first_misaligned(
    model, clean, corrupted, pair_padding_side=None, templated=False
):
    """Like _fake_tokenize_batch_pair, but the first pair in every batch has a
    shorter real-token count on the corrupted side (clean_mask.sum != corr_mask.sum),
    simulating a mid-sequence length-changing corruption."""
    batch = len(clean)
    seq_len = 5
    tokens = torch.randint(0, 100, (batch, seq_len))
    clean_mask = torch.ones(batch, seq_len)
    corr_mask = torch.ones(batch, seq_len)
    corr_mask[0, -1] = 0  # first pair: corrupted has one fewer real token
    input_lengths = torch.full((batch,), seq_len, dtype=torch.long)
    return tokens, tokens.clone(), clean_mask, corr_mask, input_lengths, seq_len


def _fake_make_hooks_and_matrices(model, graph, batch_size, n_pos, _):
    n_fwd = graph.n_forward
    d = model.cfg.d_model
    act_diff = torch.zeros(batch_size, n_pos, n_fwd, d)
    empty_hooks = []
    return (empty_hooks, empty_hooks, empty_hooks), act_diff


def _fake_compute_mean_activations(
    model, graph, dataloader, per_position=False, padding_side=None, templated=False
):
    n = graph.n_forward
    d = model.cfg.d_model
    if per_position:
        return torch.zeros(5, n, d)
    return torch.zeros(n, d)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def model():
    return MockModel()


@pytest.fixture
def graph():
    return MockGraph()


@pytest.fixture
def dataloader_3():
    """Dataloader with 3 batches of (clean, corrupted, label)."""
    batches = [
        (["hello", "world"], ["hi", "earth"], torch.tensor([0, 1])),
        (["foo", "bar"], ["baz", "qux"], torch.tensor([1, 0])),
    ]
    return batches


@pytest.fixture
def dataloader_4():
    """Dataloader with 4-tuple batches (includes answer_spans)."""
    spans = [(0, 3), (1, 4)]
    batches = [
        (["hello", "world"], ["hi", "earth"], torch.tensor([0, 1]), spans),
    ]
    return batches


# ═══════════════════════════════════════════════════════════════════════════
# Tests: evaluate_graph
# ═══════════════════════════════════════════════════════════════════════════


@patch("circuitkit.evaluation.evaluate.compute_mean_activations", _fake_compute_mean_activations)
@patch("circuitkit.evaluation.evaluate.make_hooks_and_matrices", _fake_make_hooks_and_matrices)
@patch("circuitkit.evaluation.evaluate.tokenize_batch_pair", _fake_tokenize_batch_pair)
class TestEvaluateGraph:
    """Tests for the evaluate_graph function."""

    def test_patching_intervention(self, model, graph, dataloader_3):

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="patching",
        )
        assert isinstance(result, Tensor)
        assert result.shape[0] == 4  # 2 batches × 2 samples

    def test_zero_intervention(self, model, graph, dataloader_3):

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="zero",
        )
        assert isinstance(result, Tensor)
        assert result.shape[0] == 4

    def test_mean_intervention_requires_dataloader(self, model, graph, dataloader_3):

        with pytest.raises(ValueError, match="requires an intervention_dataloader"):
            evaluate_graph(
                model,
                graph,
                dataloader_3,
                metrics=_make_simple_metric,
                quiet=True,
                intervention="mean",
                intervention_dataloader=None,
            )

    def test_mean_intervention_with_dataloader(self, model, graph, dataloader_3):

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="mean",
            intervention_dataloader=dataloader_3,
        )
        assert isinstance(result, Tensor)

    def test_mean_positional_intervention(self, model, graph, dataloader_3):

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="mean-positional",
            intervention_dataloader=dataloader_3,
        )
        assert isinstance(result, Tensor)

    def test_invalid_intervention_raises(self, model, graph, dataloader_3):

        with pytest.raises(ValueError, match="Invalid intervention"):
            evaluate_graph(
                model,
                graph,
                dataloader_3,
                metrics=_make_simple_metric,
                quiet=True,
                intervention="bogus",
            )

    def test_use_attn_result_false_raises(self, graph, dataloader_3):
        model = MockModel(cfg=MockModelCfg(use_attn_result=False))

        with pytest.raises(ValueError, match="use_attn_result"):
            evaluate_graph(model, graph, dataloader_3, metrics=_make_simple_metric, quiet=True)

    def test_grouped_attention_without_ungroup_raises(self, graph, dataloader_3):
        cfg = MockModelCfg(n_key_value_heads=4, ungroup_grouped_query_attention=False)
        model = MockModel(cfg=cfg)

        with pytest.raises(ValueError, match="ungroup_grouped_query_attention"):
            evaluate_graph(model, graph, dataloader_3, metrics=_make_simple_metric, quiet=True)

    def test_multiple_metrics_returns_list(self, model, graph, dataloader_3):

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=[_make_simple_metric, _make_simple_metric],
            quiet=True,
            intervention="patching",
        )
        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, Tensor)
            assert r.shape[0] == 4

    def test_answer_spans_forwarded_to_metric(self, model, graph, dataloader_4):

        result = evaluate_graph(
            model,
            graph,
            dataloader_4,
            metrics=_make_metric_with_spans,
            quiet=True,
            intervention="patching",
        )
        # metric returns 2.0 when answer_spans is provided
        assert result.mean().item() == pytest.approx(2.0)

    def test_skip_clean_true(self, model, graph, dataloader_3):
        """When skip_clean=True, clean_logits passed to metric should be None."""

        received_clean = []

        def spy_metric(logits, clean_logits, input_lengths, labels):
            received_clean.append(clean_logits)
            return torch.ones(logits.shape[0])

        evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=spy_metric,
            quiet=True,
            intervention="patching",
            skip_clean=True,
        )
        assert all(c is None for c in received_clean)

    def test_skip_clean_false(self, model, graph, dataloader_3):
        """When skip_clean=False, clean_logits should be a tensor."""

        received_clean = []

        def spy_metric(logits, clean_logits, input_lengths, labels):
            received_clean.append(clean_logits)
            return torch.ones(logits.shape[0])

        evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=spy_metric,
            quiet=True,
            intervention="patching",
            skip_clean=False,
        )
        assert all(c is not None for c in received_clean)

    def test_scalar_metric_result_unsqueezed(self, model, graph, dataloader_3):
        """Scalar metric results should be unsqueezed to 1-D."""

        def scalar_metric(logits, clean_logits, input_lengths, labels):
            return torch.tensor(0.5)  # scalar

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=scalar_metric,
            quiet=True,
            intervention="zero",
        )
        assert result.ndim == 1

    def test_exclude_misaligned_false_keeps_all(self, model, graph, dataloader_3):
        """Default: exclude_misaligned=False scores every pair (legacy behaviour)."""

        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="patching",
        )
        assert result.shape[0] == 4  # 2 batches x 2 samples, nothing dropped


@patch("circuitkit.evaluation.evaluate.compute_mean_activations", _fake_compute_mean_activations)
@patch("circuitkit.evaluation.evaluate.make_hooks_and_matrices", _fake_make_hooks_and_matrices)
@patch(
    "circuitkit.evaluation.evaluate.tokenize_batch_pair",
    _fake_tokenize_batch_pair_first_misaligned,
)
class TestEvaluateGraphExcludeMisaligned:
    """exclude_misaligned drops pairs whose clean/corrupt token counts differ."""

    def test_exclude_misaligned_true_drops_mismatched_pair(self, model, graph, dataloader_3):
        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="patching",
            exclude_misaligned=True,
        )
        # Each of the 2 batches (size 2) has its first pair misaligned and dropped,
        # leaving 1 kept pair per batch -> 2 total (down from 4).
        assert result.shape[0] == 2

    def test_exclude_misaligned_false_ignores_mismatch(self, model, graph, dataloader_3):
        """Without the flag, a token-count mismatch does not affect scoring."""
        result = evaluate_graph(
            model,
            graph,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="patching",
            exclude_misaligned=False,
        )
        assert result.shape[0] == 4

    def test_exclude_misaligned_rejects_aggregated_metric(self, model, graph, dataloader_3):
        """An aggregating metric (e.g. task_spec.metric_fn()'s mean=True default,
        used unwrapped e.g. via ck.faithfulness()) returns one score per BATCH,
        not per sample. exclude_misaligned's boolean mask is per-sample-sized, so
        this must raise a clear, actionable error rather than crash on a shape
        mismatch or silently misindex.
        """

        def aggregated_metric(logits, clean_logits, input_lengths, labels):
            return torch.tensor(0.5)  # scalar: the mean=True / loss=True shape

        with pytest.raises(ValueError, match="per-sample metric"):
            evaluate_graph(
                model,
                graph,
                dataloader_3,
                metrics=aggregated_metric,
                quiet=True,
                intervention="patching",
                exclude_misaligned=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: evaluate_baseline
# ═══════════════════════════════════════════════════════════════════════════


@patch("circuitkit.evaluation.evaluate.tokenize_batch_pair", _fake_tokenize_batch_pair)
class TestEvaluateBaseline:
    def test_clean_evaluation(self, model, dataloader_3):

        result = evaluate_baseline(
            model,
            dataloader_3,
            metrics=_make_simple_metric,
            run_corrupted=False,
            quiet=True,
        )
        assert isinstance(result, Tensor)
        assert result.shape[0] == 4

    def test_corrupted_evaluation(self, model, dataloader_3):

        result = evaluate_baseline(
            model,
            dataloader_3,
            metrics=_make_simple_metric,
            run_corrupted=True,
            quiet=True,
        )
        assert isinstance(result, Tensor)
        assert result.shape[0] == 4

    def test_multiple_metrics(self, model, dataloader_3):

        result = evaluate_baseline(
            model,
            dataloader_3,
            metrics=[_make_simple_metric, _make_simple_metric],
            quiet=True,
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_answer_spans_forwarded(self, model, dataloader_4):

        result = evaluate_baseline(
            model,
            dataloader_4,
            metrics=_make_metric_with_spans,
            quiet=True,
        )
        assert result.mean().item() == pytest.approx(2.0)

    def test_scalar_metric_unsqueezed(self, model, dataloader_3):

        def scalar_metric(logits, clean_logits, input_lengths, labels):
            return torch.tensor(0.5)

        result = evaluate_baseline(model, dataloader_3, metrics=scalar_metric, quiet=True)
        assert result.ndim == 1

    def test_exclude_misaligned_false_keeps_all(self, model, dataloader_3):
        result = evaluate_baseline(
            model,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            exclude_misaligned=False,
        )
        assert result.shape[0] == 4


@patch(
    "circuitkit.evaluation.evaluate.tokenize_batch_pair",
    _fake_tokenize_batch_pair_first_misaligned,
)
class TestEvaluateBaselineExcludeMisaligned:
    """exclude_misaligned drops pairs whose clean/corrupt token counts differ,
    matching evaluate_graph so a pillar can compute clean/corrupt/circuit scores
    over the identical example set."""

    def test_exclude_misaligned_true_drops_mismatched_pair(self, model, dataloader_3):
        result = evaluate_baseline(
            model,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            exclude_misaligned=True,
        )
        assert result.shape[0] == 2

    def test_exclude_misaligned_false_ignores_mismatch(self, model, dataloader_3):
        result = evaluate_baseline(
            model,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            exclude_misaligned=False,
        )
        assert result.shape[0] == 4

    def test_exclude_misaligned_rejects_aggregated_metric(self, model, dataloader_3):
        """Same aggregated-metric guard as evaluate_graph (see there for why)."""

        def aggregated_metric(logits, clean_logits, input_lengths, labels):
            return torch.tensor(0.5)

        with pytest.raises(ValueError, match="per-sample metric"):
            evaluate_baseline(
                model,
                dataloader_3,
                metrics=aggregated_metric,
                quiet=True,
                exclude_misaligned=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: evaluate_ibcircuit_neuron_circuit
# ═══════════════════════════════════════════════════════════════════════════


@patch("circuitkit.evaluation.evaluate.tokenize_batch_pair", _fake_tokenize_batch_pair)
class TestEvaluateIBCircuit:
    def test_zero_intervention_empty_pruning(self, model, dataloader_3):

        pruning_dict = {"mlp": {}, "heads": {}}
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="zero",
        )
        assert isinstance(result, Tensor)

    def test_mean_intervention_with_mlp(self, model, dataloader_3):

        pruning_dict = {
            "mlp": {0: [0, 1, 2]},  # layer 0, neurons 0-2
            "heads": {},
        }
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="mean",
        )
        assert isinstance(result, Tensor)

    def test_patching_intervention(self, model, dataloader_3):

        pruning_dict = {
            "mlp": {0: [0, 1]},
            "heads": {},
        }
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="patching",
        )
        assert isinstance(result, Tensor)

    def test_multiple_metrics(self, model, dataloader_3):

        pruning_dict = {"mlp": {}, "heads": {}}
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_3,
            metrics=[_make_simple_metric, _make_simple_metric],
            quiet=True,
            intervention="zero",
        )
        assert isinstance(result, list)
        assert len(result) == 2

    def test_answer_spans_forwarded(self, model, dataloader_4):

        pruning_dict = {"mlp": {}, "heads": {}}
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_4,
            metrics=_make_metric_with_spans,
            quiet=True,
            intervention="zero",
        )
        assert result.mean().item() == pytest.approx(2.0)

    def test_meta_mlp_hook_post_act(self, model, dataloader_3):
        """_meta.mlp_hook = 'post_act' should use mlp.hook_post."""

        pruning_dict = {
            "mlp": {0: [0]},
            "heads": {},
            "_meta": {"mlp_hook": "post_act"},
        }
        # Should not crash — it changes the hook name internally
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="zero",
        )
        assert isinstance(result, Tensor)

    def test_heads_pruning(self, model, dataloader_3):

        pruning_dict = {
            "mlp": {},
            "heads": {(0, 0): [0, 1]},  # layer 0, head 0, neurons 0-1
        }
        result = evaluate_ibcircuit_neuron_circuit(
            model,
            pruning_dict,
            dataloader_3,
            metrics=_make_simple_metric,
            quiet=True,
            intervention="zero",
        )
        assert isinstance(result, Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
