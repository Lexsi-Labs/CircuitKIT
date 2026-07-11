"""
Unit tests for circuitkit/src/circuitkit/api.py

Location: circuitkit/tests/unit/test_api.py

Tests cover all public functions and private helpers, including:
- EAPDiscoveryDataset: CSV loading, item access, dataloader creation
- _eap_accuracy: scalar/batched, 1D/2D labels, device handling
- _convert_eap_scores_to_ck_format: attention/MLP nodes, naming convention
- _ib_name_to_graph_name: regex round-trip for all node name patterns
- _populate_graph_from_ib_scores: score injection, out-of-scope pinning to inf
- _validate_ibcircuit_dataloader: missing keys, wrong types, shape mismatches
- _avg_scores: tensor, list-of-tensors, single-element
- _make_eval_metric: partial override, non-partial passthrough
- _compute_n_topn: head/mlp/both scopes, zero-sparsity, full-sparsity
- _build_artifact_stem: algo-specific extras, org-prefix stripping
- _save_artifact: file creation, suffix injection, directory input
- _build_circuit_scores: field mapping, timestamp presence
- _save_evaluation_results_to_txt: file content, custom path, failure graceful
- load_circuit: missing file raises, valid pt roundtrip
- discover_circuit: config validation, missing task key, unsupported algorithm
- benchmark_circuit: invalid evaluation_mode, artifact type detection
"""

import os
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_logits(batch=2, seq=5, vocab=10, device="cpu"):
    """Return a simple logits tensor of given shape."""
    return torch.randn(batch, seq, vocab, device=device)


def _make_input_length(batch=2, seq=5, device="cpu"):
    """Return input_length tensor pointing to the last real token."""
    return torch.full((batch,), seq, dtype=torch.long, device=device)


def _make_labels_1d(batch=2, vocab=10, device="cpu"):
    return torch.randint(0, vocab, (batch,), device=device)


def _make_labels_2d(batch=2, n_options=4, vocab=10, device="cpu"):
    return torch.randint(0, vocab, (batch, n_options), device=device)


# ---------------------------------------------------------------------------
# EAPDiscoveryDataset
# ---------------------------------------------------------------------------


class TestEAPDiscoveryDataset:
    """Tests for the EAPDiscoveryDataset class."""

    # ── fixtures ──────────────────────────────────────────────────────────

    @pytest.fixture
    def ioi_csv(self, tmp_path):
        """Two-column (IOI-style) correct/incorrect CSV."""
        path = tmp_path / "ioi.csv"
        df = pd.DataFrame(
            {
                "clean": ["John gave Mary", "Alice saw Bob"],
                "corrupted": ["Mary gave John", "Bob saw Alice"],
                "correct_idx": [1234, 5678],
                "incorrect_idx": [4321, 8765],
            }
        )
        df.to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def mmlu_csv(self, tmp_path):
        """Four-token (MMLU-style) CSV where incorrect_idx is a list string."""
        path = tmp_path / "mmlu.csv"
        df = pd.DataFrame(
            {
                "clean": ["Q: What is 2+2? A:"],
                "corrupted": ["Q: What is 3+3? A:"],
                "correct_idx": [10],
                "incorrect_idx": ["[20, 30, 40]"],
            }
        )
        df.to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def empty_csv(self, tmp_path):
        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["clean", "corrupted", "correct_idx", "incorrect_idx"]).to_csv(
            path, index=False
        )
        return str(path)

    # ── construction & length ─────────────────────────────────────────────

    def test_len_ioi(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        assert len(ds) == 2

    def test_len_empty(self, empty_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(empty_csv)
        assert len(ds) == 0

    # ── __getitem__ ───────────────────────────────────────────────────────

    def test_ioi_item_returns_tuple_of_three(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        clean, corrupted, labels = ds[0]
        assert isinstance(clean, str)
        assert isinstance(corrupted, str)
        assert isinstance(labels, list)

    def test_ioi_labels_are_two_elements(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        _, _, labels = ds[0]
        assert len(labels) == 2

    def test_ioi_correct_idx_is_first_label(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        _, _, labels = ds[0]
        assert labels[0] == 1234

    def test_mmlu_labels_are_four_elements(self, mmlu_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(mmlu_csv)
        _, _, labels = ds[0]
        assert len(labels) == 4

    def test_mmlu_parses_string_list_incorrect_idx(self, mmlu_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(mmlu_csv)
        _, _, labels = ds[0]
        assert labels == [10, 20, 30, 40]

    def test_clean_corrupted_values_match_csv(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        clean, corrupted, _ = ds[1]
        assert clean == "Alice saw Bob"
        assert corrupted == "Bob saw Alice"

    def test_index_out_of_bounds_raises(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        with pytest.raises(IndexError):
            _ = ds[99]

    # ── to_dataloader ─────────────────────────────────────────────────────

    def test_to_dataloader_returns_dataloader(self, ioi_csv):
        from torch.utils.data import DataLoader

        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        dl = ds.to_dataloader(batch_size=2)
        assert isinstance(dl, DataLoader)

    def test_to_dataloader_sets_pair_padding_side_default(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        dl = ds.to_dataloader(batch_size=1)
        assert dl.pair_padding_side == "right"

    def test_to_dataloader_sets_pair_padding_side_custom(self, ioi_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(ioi_csv)
        dl = ds.to_dataloader(batch_size=1, pair_padding_side="left")
        assert dl.pair_padding_side == "left"


# ---------------------------------------------------------------------------
# _eap_accuracy
# ---------------------------------------------------------------------------


class TestEapAccuracy:
    """Tests for the _eap_accuracy metric helper."""

    def _call(self, logits, input_length, labels, mean=True):
        from circuitkit.api import _eap_accuracy

        clean_logits = torch.zeros_like(logits)  # unused
        return _eap_accuracy(logits, clean_logits, input_length, labels, mean=mean)

    def test_returns_scalar_when_mean_true(self):
        logits = _make_logits(batch=4, seq=5, vocab=10)
        il = _make_input_length(batch=4, seq=5)
        labels = _make_labels_1d(batch=4, vocab=10)
        result = self._call(logits, il, labels, mean=True)
        assert result.ndim == 0  # scalar

    def test_returns_per_sample_when_mean_false(self):
        logits = _make_logits(batch=4, seq=5, vocab=10)
        il = _make_input_length(batch=4, seq=5)
        labels = _make_labels_1d(batch=4, vocab=10)
        result = self._call(logits, il, labels, mean=False)
        assert result.shape == (4,)

    def test_perfect_accuracy_with_1d_labels(self):
        """When predicted token equals correct_idx, accuracy = 1.0."""
        vocab = 10
        batch = 3
        seq = 4
        # Build logits where argmax at last position is exactly the label
        logits = torch.zeros(batch, seq, vocab)
        labels = torch.tensor([2, 5, 7])
        for b, lbl in enumerate(labels):
            logits[b, seq - 1, lbl] = 100.0  # dominate softmax
        il = torch.full((batch,), seq, dtype=torch.long)
        result = self._call(logits, il, labels, mean=True)
        assert result.item() == pytest.approx(1.0)

    def test_zero_accuracy_with_1d_labels(self):
        vocab = 10
        batch = 3
        seq = 4
        logits = torch.zeros(batch, seq, vocab)
        labels = torch.tensor([0, 1, 2])
        # Force argmax to a different token for each
        for b in range(batch):
            logits[b, seq - 1, 9] = 100.0
        il = torch.full((batch,), seq, dtype=torch.long)
        result = self._call(logits, il, labels, mean=True)
        assert result.item() == pytest.approx(0.0)

    def test_uses_correct_sequence_position(self):
        """Input_length - 1 must select the last real token."""
        vocab = 5
        logits = torch.zeros(1, 6, vocab)
        # Place winning logit at position 3 (input_length=4 → index 3)
        logits[0, 3, 2] = 100.0
        il = torch.tensor([4])
        labels = torch.tensor([2])
        result = self._call(logits, il, labels, mean=True)
        assert result.item() == pytest.approx(1.0)

    def test_2d_labels_uses_column_0_as_correct(self):
        vocab = 10
        batch = 2
        seq = 3
        logits = torch.zeros(batch, seq, vocab)
        # Label matrix: column 0 is correct token
        labels = torch.tensor([[3, 5, 6, 7], [1, 2, 8, 9]])
        for b in range(batch):
            logits[b, seq - 1, labels[b, 0]] = 100.0
        il = torch.full((batch,), seq, dtype=torch.long)
        result = self._call(logits, il, labels, mean=True)
        assert result.item() == pytest.approx(1.0)

    def test_values_are_float(self):
        logits = _make_logits(batch=2, seq=3, vocab=5)
        il = _make_input_length(batch=2, seq=3)
        labels = _make_labels_1d(batch=2, vocab=5)
        result = self._call(logits, il, labels, mean=False)
        assert result.dtype == torch.float32

    def test_result_in_zero_one_range(self):
        logits = _make_logits(batch=8, seq=4, vocab=10)
        il = _make_input_length(batch=8, seq=4)
        labels = _make_labels_1d(batch=8, vocab=10)
        result = self._call(logits, il, labels, mean=False)
        assert (result >= 0).all() and (result <= 1).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_works_on_gpu(self):
        device = "cuda"
        logits = _make_logits(batch=2, seq=3, vocab=5, device=device)
        il = _make_input_length(batch=2, seq=3, device=device)
        labels = _make_labels_1d(batch=2, vocab=5, device=device)
        result = self._call(logits, il, labels)
        assert result.device.type == "cuda"


# ---------------------------------------------------------------------------
# _ib_name_to_graph_name
# ---------------------------------------------------------------------------


class TestIbNameToGraphName:
    """Tests for the private regex-based name converter."""

    def _call(self, name):
        from circuitkit.api import _ib_name_to_graph_name

        return _ib_name_to_graph_name(name)

    def test_attention_head_basic(self):
        assert self._call("A0.0") == "a0.h0"

    def test_attention_head_multi_digit(self):
        assert self._call("A11.23") == "a11.h23"

    def test_mlp_basic(self):
        assert self._call("MLP 0") == "m0"

    def test_mlp_multi_digit(self):
        assert self._call("MLP 12") == "m12"

    def test_invalid_name_returns_none(self):
        assert self._call("garbage") is None

    def test_partial_match_returns_none(self):
        # 'A0' without .head is not a valid attention node
        assert self._call("A0") is None

    def test_mlp_without_space_returns_none(self):
        assert self._call("MLP0") is None

    def test_empty_string_returns_none(self):
        assert self._call("") is None

    def test_roundtrip_consistency(self):
        """Converted names should follow the graph naming convention."""
        result = self._call("A3.7")
        assert result == "a3.h7"


# ---------------------------------------------------------------------------
# _convert_eap_scores_to_ck_format
# ---------------------------------------------------------------------------


class TestConvertEapScoresToCkFormat:
    """Tests for EAP graph → CircuitKit score dict conversion."""

    def _make_mock_graph(self, n_layers=2, n_heads=2):
        """
        Build a minimal mock Graph with AttentionNode and MLPNode instances
        whose .score attributes can be set.
        """
        from circuitkit.api import AttentionNode, MLPNode

        nodes = {}
        for layer in range(n_layers):
            for head in range(n_heads):
                node = MagicMock(spec=AttentionNode)
                node.layer = layer
                node.head = head
                node.score = torch.tensor(float((layer + 1) * (head + 1)))
                nodes[f"a{layer}.h{head}"] = node
            mlp = MagicMock(spec=MLPNode)
            mlp.layer = layer
            mlp.score = torch.tensor(float(layer + 0.5))
            nodes[f"m{layer}"] = mlp

        graph = MagicMock()
        graph.nodes = nodes
        return graph

    def test_output_contains_attention_keys(self):
        from circuitkit.api import _convert_eap_scores_to_ck_format

        graph = self._make_mock_graph(n_layers=2, n_heads=2)
        result = _convert_eap_scores_to_ck_format(graph)
        assert "A0.0" in result
        assert "A1.1" in result

    def test_output_contains_mlp_keys(self):
        from circuitkit.api import _convert_eap_scores_to_ck_format

        graph = self._make_mock_graph(n_layers=2, n_heads=2)
        result = _convert_eap_scores_to_ck_format(graph)
        assert "MLP 0" in result
        assert "MLP 1" in result

    def test_scores_are_absolute_values(self):
        from circuitkit.api import _convert_eap_scores_to_ck_format

        graph = self._make_mock_graph(n_layers=1, n_heads=1)
        # Set a negative score
        graph.nodes["a0.h0"].score = torch.tensor(-3.14)
        result = _convert_eap_scores_to_ck_format(graph)
        assert result["A0.0"] == pytest.approx(3.14)

    def test_total_keys_matches_nodes(self):
        from circuitkit.api import _convert_eap_scores_to_ck_format

        graph = self._make_mock_graph(n_layers=3, n_heads=4)
        result = _convert_eap_scores_to_ck_format(graph)
        # 3*4 attention + 3 MLP = 15 keys
        assert len(result) == 15

    def test_scores_are_floats(self):
        from circuitkit.api import _convert_eap_scores_to_ck_format

        graph = self._make_mock_graph(n_layers=1, n_heads=1)
        result = _convert_eap_scores_to_ck_format(graph)
        for v in result.values():
            assert isinstance(v, float)

    def test_empty_graph_returns_empty_dict(self):
        from circuitkit.api import _convert_eap_scores_to_ck_format

        graph = MagicMock()
        graph.nodes = {}
        result = _convert_eap_scores_to_ck_format(graph)
        assert result == {}


# ---------------------------------------------------------------------------
# _populate_graph_from_ib_scores
# ---------------------------------------------------------------------------


class TestPopulateGraphFromIbScores:
    """Tests for score injection into the graph's nodes_scores tensor."""

    def _make_graph(self, n_layers=2, n_heads=2):
        """Build a mock graph with a real nodes_scores tensor."""
        from circuitkit.api import AttentionNode, MLPNode

        n_layers * (n_heads + 1)
        nodes = {}
        fwd_map = {}
        idx = 0
        for layer in range(n_layers):
            for head in range(n_heads):
                node = MagicMock(spec=AttentionNode)
                node.layer = layer
                node.head = head
                node.score = torch.tensor(0.0)
                name = f"a{layer}.h{head}"
                nodes[name] = node
                fwd_map[name] = idx
                idx += 1
            node = MagicMock(spec=MLPNode)
            node.layer = layer
            node.score = torch.tensor(0.0)
            name = f"m{layer}"
            nodes[name] = node
            fwd_map[name] = idx
            idx += 1

        graph = MagicMock()
        graph.nodes = nodes
        graph.n_forward = idx
        graph.nodes_scores = torch.zeros(idx)
        graph.forward_index = lambda node, attn_slice=False: fwd_map[
            f"a{node.layer}.h{node.head}" if isinstance(node, AttentionNode) else f"m{node.layer}"
        ]
        return graph

    def test_known_node_score_is_written(self):
        from circuitkit.api import _populate_graph_from_ib_scores

        graph = self._make_graph(n_layers=1, n_heads=1)
        _populate_graph_from_ib_scores(graph, {"A0.0": 0.75})
        assert graph.nodes["a0.h0"].score.item() == pytest.approx(0.75)

    def test_missing_node_pinned_to_inf(self):
        from circuitkit.api import _populate_graph_from_ib_scores

        graph = self._make_graph(n_layers=1, n_heads=1)
        # Only score the MLP; attention head should be pinned to inf
        _populate_graph_from_ib_scores(graph, {"MLP 0": 0.5})
        assert graph.nodes["a0.h0"].score.item() == float("inf")

    def test_mlp_score_is_written(self):
        from circuitkit.api import _populate_graph_from_ib_scores

        graph = self._make_graph(n_layers=2, n_heads=1)
        _populate_graph_from_ib_scores(graph, {"MLP 1": 0.33})
        assert graph.nodes["m1"].score.item() == pytest.approx(0.33)

    def test_scores_are_absolute(self):
        from circuitkit.api import _populate_graph_from_ib_scores

        graph = self._make_graph(n_layers=1, n_heads=1)
        _populate_graph_from_ib_scores(graph, {"A0.0": -2.0})
        assert graph.nodes["a0.h0"].score.item() == pytest.approx(2.0)

    def test_returns_graph_object(self):
        from circuitkit.api import _populate_graph_from_ib_scores

        graph = self._make_graph(n_layers=1, n_heads=1)
        ret = _populate_graph_from_ib_scores(graph, {})
        assert ret is graph


# ---------------------------------------------------------------------------
# _validate_ibcircuit_dataloader
# ---------------------------------------------------------------------------


class TestValidateIbcircuitDataloader:
    """Tests for IBCircuit dataloader format validation."""

    def _valid_batch(self, batch_size=2, seq_len=5, vocab=10):
        return {
            "tokens": torch.randint(0, vocab, (batch_size, seq_len)),
            "labels": torch.randint(0, vocab, (batch_size,)),
            "answer_positions": torch.zeros(batch_size, dtype=torch.long),
        }

    def _make_dl(self, batch):
        dl = MagicMock()
        dl.__iter__ = MagicMock(return_value=iter([batch]))
        return dl

    def test_valid_dataloader_passes(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        dl = self._make_dl(self._valid_batch())
        _validate_ibcircuit_dataloader(dl)  # should not raise

    def test_empty_dataloader_raises_value_error(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        dl = MagicMock()
        dl.__iter__ = MagicMock(return_value=iter([]))
        with pytest.raises(ValueError, match="empty"):
            _validate_ibcircuit_dataloader(dl)

    def test_missing_tokens_key_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch()
        del batch["tokens"]
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="tokens"):
            _validate_ibcircuit_dataloader(dl)

    def test_missing_labels_key_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch()
        del batch["labels"]
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="labels"):
            _validate_ibcircuit_dataloader(dl)

    def test_missing_answer_positions_key_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch()
        del batch["answer_positions"]
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="answer_positions"):
            _validate_ibcircuit_dataloader(dl)

    def test_tokens_not_tensor_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch()
        batch["tokens"] = [[1, 2, 3], [4, 5, 6]]
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="torch.Tensor"):
            _validate_ibcircuit_dataloader(dl)

    def test_labels_not_tensor_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch()
        batch["labels"] = [0, 1]
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="torch.Tensor"):
            _validate_ibcircuit_dataloader(dl)

    def test_answer_positions_not_tensor_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch()
        batch["answer_positions"] = [0, 0]
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="torch.Tensor"):
            _validate_ibcircuit_dataloader(dl)

    def test_batch_size_mismatch_labels_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch(batch_size=4)
        batch["labels"] = torch.zeros(2, dtype=torch.long)  # wrong batch dim
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="[Bb]atch size"):
            _validate_ibcircuit_dataloader(dl)

    def test_batch_size_mismatch_answer_positions_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch(batch_size=4)
        batch["answer_positions"] = torch.zeros(2, dtype=torch.long)
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="[Bb]atch size"):
            _validate_ibcircuit_dataloader(dl)

    def test_out_of_bounds_answer_positions_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch(batch_size=2, seq_len=5)
        batch["answer_positions"] = torch.tensor([10, 10])  # >= seq_len
        dl = self._make_dl(batch)
        with pytest.raises(ValueError, match="answer_positions"):
            _validate_ibcircuit_dataloader(dl)

    def test_answer_position_exactly_at_boundary_raises(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch(batch_size=1, seq_len=5)
        batch["answer_positions"] = torch.tensor([5])  # == seq_len, invalid
        dl = self._make_dl(batch)
        with pytest.raises(ValueError):
            _validate_ibcircuit_dataloader(dl)

    def test_answer_position_at_last_valid_index_passes(self):
        from circuitkit.api import _validate_ibcircuit_dataloader

        batch = self._valid_batch(batch_size=1, seq_len=5)
        batch["answer_positions"] = torch.tensor([4])  # valid last index
        dl = self._make_dl(batch)
        _validate_ibcircuit_dataloader(dl)  # should not raise


# ---------------------------------------------------------------------------
# _avg_scores
# ---------------------------------------------------------------------------


class TestAvgScores:
    """Tests for the per-sample score aggregator."""

    def _call(self, scores):
        from circuitkit.api import _avg_scores

        return _avg_scores(scores)

    def test_1d_tensor_mean(self):
        scores = torch.tensor([0.2, 0.4, 0.6])
        result = self._call(scores)
        assert result == pytest.approx(0.4, abs=1e-6)

    def test_single_element_tensor(self):
        scores = torch.tensor([0.75])
        result = self._call(scores)
        assert result == pytest.approx(0.75)

    def test_scalar_tensor(self):
        scores = torch.tensor(0.9)
        result = self._call(scores)
        assert result == pytest.approx(0.9)

    def test_list_of_tensors(self):
        scores = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])]
        result = self._call(scores)
        # mean([0.5, 1.0]) = 0.75
        assert result == pytest.approx(0.75)

    def test_list_of_single_tensors(self):
        scores = [torch.tensor([0.8]), torch.tensor([0.2])]
        result = self._call(scores)
        assert result == pytest.approx(0.5)

    def test_returns_python_float(self):
        result = self._call(torch.tensor([0.5, 0.5]))
        assert isinstance(result, float)

    def test_all_zeros(self):
        assert self._call(torch.zeros(5)) == pytest.approx(0.0)

    def test_all_ones(self):
        assert self._call(torch.ones(5)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _make_eval_metric
# ---------------------------------------------------------------------------


class TestMakeEvalMetric:
    """Tests for metric factory that strips loss/mean from partials."""

    def _make_task_spec_with_partial(self, base_fn, **kwargs):
        spec = MagicMock()
        spec.metric_fn.return_value = partial(base_fn, **kwargs)
        return spec

    def _make_task_spec_with_plain(self, fn):
        spec = MagicMock()
        spec.metric_fn.return_value = fn
        return spec

    def test_partial_has_loss_set_to_false(self):
        from circuitkit.api import _make_eval_metric

        def dummy_fn(logits, loss=True, mean=True):
            return torch.tensor(0.0)

        spec = self._make_task_spec_with_partial(dummy_fn, loss=True, mean=True)
        metric = _make_eval_metric(spec)
        assert isinstance(metric, partial)
        assert metric.keywords["loss"] is False

    def test_partial_has_mean_set_to_false(self):
        from circuitkit.api import _make_eval_metric

        def dummy_fn(logits, loss=True, mean=True):
            return torch.tensor(0.0)

        spec = self._make_task_spec_with_partial(dummy_fn, loss=True, mean=True)
        metric = _make_eval_metric(spec)
        assert metric.keywords["mean"] is False

    def test_non_partial_returned_unchanged(self):
        from circuitkit.api import _make_eval_metric

        def plain_fn(logits, clean, il, labels):
            return torch.tensor(1.0)

        spec = self._make_task_spec_with_plain(plain_fn)
        metric = _make_eval_metric(spec)
        assert metric is plain_fn

    def test_resulting_partial_is_callable(self):
        from circuitkit.api import _make_eval_metric

        def dummy_fn(logits, loss=True, mean=True):
            return torch.tensor(0.5)

        spec = self._make_task_spec_with_partial(dummy_fn, loss=True, mean=True)
        metric = _make_eval_metric(spec)
        assert callable(metric)


# ---------------------------------------------------------------------------
# _compute_n_topn
# ---------------------------------------------------------------------------


class TestComputeNTopn:
    """Tests for the pruning budget calculator."""

    def _make_graph(self, n_layers, n_heads):
        graph = MagicMock()
        graph.cfg = {"n_layers": n_layers, "n_heads": n_heads}
        return graph

    def _call(self, n_layers, n_heads, scope, sparsity):
        from circuitkit.api import _compute_n_topn

        graph = self._make_graph(n_layers, n_heads)
        return _compute_n_topn(graph, scope, sparsity)

    def test_heads_scope_zero_sparsity_keeps_all_heads(self):
        n_topn, n_to_keep = self._call(n_layers=2, n_heads=4, scope="heads", sparsity=0.0)
        # n_to_keep = 2*4 = 8, n_always = 2 (MLPs)
        assert n_to_keep == 8
        assert n_topn == 10  # 8 + 2 MLPs always kept

    def test_mlp_scope_zero_sparsity_keeps_all_mlps(self):
        n_topn, n_to_keep = self._call(n_layers=3, n_heads=2, scope="mlp", sparsity=0.0)
        assert n_to_keep == 3  # all 3 MLPs
        # n_always = n_layers * n_heads = 6 heads
        assert n_topn == 9

    def test_both_scope_zero_sparsity(self):
        n_topn, n_to_keep = self._call(n_layers=2, n_heads=2, scope="both", sparsity=0.0)
        # heads = 4, mlps = 2, total = 6
        assert n_to_keep == 6
        assert n_topn == 6

    def test_both_scope_full_sparsity_keeps_zero(self):
        n_topn, n_to_keep = self._call(n_layers=2, n_heads=2, scope="both", sparsity=1.0)
        assert n_to_keep == 0
        assert n_topn == 0

    def test_heads_scope_half_sparsity(self):
        n_topn, n_to_keep = self._call(n_layers=2, n_heads=4, scope="heads", sparsity=0.5)
        # n_heads_total = 8, keep = 4
        assert n_to_keep == 4
        assert n_topn == 4 + 2  # + n_mlps always

    def test_n_topn_gte_n_to_keep(self):
        for scope in ("heads", "mlp", "both"):
            n_topn, n_to_keep = self._call(n_layers=3, n_heads=3, scope=scope, sparsity=0.3)
            assert n_topn >= n_to_keep


# ---------------------------------------------------------------------------
# _build_artifact_stem
# ---------------------------------------------------------------------------


class TestBuildArtifactStem:
    """Tests for discovery artifact filename stem builder."""

    def _base_config(self, algo="eap", task="ioi", model="gpt2", sparsity=0.3, **extra):
        pass

        cfg = {
            "model": {"name": model},
            "discovery": {"algorithm": algo, "task": task, **extra},
            "pruning": {"target_sparsity": sparsity, "scope": "both"},
        }
        return cfg

    def test_strips_org_prefix_from_model(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config(model="EleutherAI/pythia-70m")
        stem = _build_artifact_stem(cfg)
        assert "pythia-70m" in stem
        assert "EleutherAI" not in stem

    def test_algo_appears_first(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config(algo="acdc")
        stem = _build_artifact_stem(cfg)
        assert stem.startswith("acdc")

    def test_task_appears_in_stem(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config(task="greater-than")
        stem = _build_artifact_stem(cfg)
        assert "greater-than" in stem

    def test_sparsity_appears_with_sp_prefix(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config(sparsity=0.5)
        stem = _build_artifact_stem(cfg)
        assert "sp0.5" in stem

    def test_ibcircuit_adds_epoch_suffix(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config(algo="ibcircuit")
        cfg["discovery"]["num_epochs"] = 1000
        stem = _build_artifact_stem(cfg)
        assert "e1000" in stem

    def test_eap_ig_adds_method_if_set(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config(algo="eap-ig", method="IG")
        stem = _build_artifact_stem(cfg)
        assert "ig" in stem.lower()

    def test_stem_has_no_spaces(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config()
        stem = _build_artifact_stem(cfg)
        assert " " not in stem

    def test_parts_joined_by_underscore(self):
        from circuitkit.api import _build_artifact_stem

        cfg = self._base_config()
        stem = _build_artifact_stem(cfg)
        assert "_" in stem


# ---------------------------------------------------------------------------
# _save_artifact
# ---------------------------------------------------------------------------


class TestSaveArtifact:
    """Tests for the generic artifact save helper."""

    def _make_logger(self):
        logger = MagicMock()
        return logger

    def test_creates_file_at_given_path(self, tmp_path):
        from circuitkit.api import _save_artifact

        out = str(tmp_path / "my_circuit.pt")
        _save_artifact({"key": "val"}, out, "_scores", self._make_logger())
        expected = str(tmp_path / "my_circuit_scores.pt")
        assert os.path.exists(expected)

    def test_suffix_injected_before_extension(self, tmp_path):
        from circuitkit.api import _save_artifact

        out = str(tmp_path / "artifact.pt")
        _save_artifact([1, 2, 3], out, "_test", self._make_logger())
        expected = str(tmp_path / "artifact_test.pt")
        assert os.path.exists(expected)

    def test_creates_parent_directories(self, tmp_path):
        from circuitkit.api import _save_artifact

        out = str(tmp_path / "deep" / "nested" / "artifact.pt")
        _save_artifact({"x": 1}, out, "_scores", self._make_logger())
        assert os.path.exists(str(tmp_path / "deep" / "nested" / "artifact_scores.pt"))

    def test_saved_data_can_be_loaded(self, tmp_path):
        from circuitkit.api import _save_artifact

        data = {"a": torch.tensor([1.0, 2.0])}
        out = str(tmp_path / "test.pt")
        _save_artifact(data, out, "_x", self._make_logger())
        loaded = torch.load(str(tmp_path / "test_x.pt"), map_location="cpu")
        assert torch.allclose(loaded["a"], data["a"])

    def test_no_output_path_and_no_config_returns_none(self):
        from circuitkit.api import _save_artifact

        result = _save_artifact({"x": 1}, None, "_scores", self._make_logger(), config=None)
        assert result is None

    def test_returns_saved_path_string(self, tmp_path):
        from circuitkit.api import _save_artifact

        out = str(tmp_path / "res.pt")
        path = _save_artifact(42, out, "_tag", self._make_logger())
        assert isinstance(path, str)
        assert path.endswith("_tag.pt")

    def test_directory_path_generates_filename_from_config(self, tmp_path):
        from circuitkit.api import _save_artifact

        # When output_path is a directory, a stem is built from config
        config = {
            "model": {"name": "gpt2"},
            "discovery": {"algorithm": "eap", "task": "ioi"},
            "pruning": {"target_sparsity": 0.3, "scope": "both"},
        }
        path = _save_artifact({"d": 1}, str(tmp_path), "_s", self._make_logger(), config=config)
        assert path is not None
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# _build_circuit_scores
# ---------------------------------------------------------------------------


class TestBuildCircuitScores:
    """Tests for the CircuitScores artifact factory."""

    def test_task_field_set(self):
        from circuitkit.api import _build_circuit_scores

        cs = _build_circuit_scores("ioi", "gpt2", "eap", {"A0.0": 0.5})
        assert cs.task == "ioi"

    def test_model_field_set(self):
        from circuitkit.api import _build_circuit_scores

        cs = _build_circuit_scores("ioi", "gpt2", "eap", {"A0.0": 0.5})
        assert cs.model == "gpt2"

    def test_algorithm_field_set(self):
        from circuitkit.api import _build_circuit_scores

        cs = _build_circuit_scores("ioi", "gpt2", "eap", {"A0.0": 0.5})
        assert cs.algorithm == "eap"

    def test_node_scores_field_set(self):
        from circuitkit.api import _build_circuit_scores

        scores = {"A0.0": 0.5, "MLP 1": 0.3}
        cs = _build_circuit_scores("ioi", "gpt2", "eap", scores)
        assert cs.node_scores == scores

    def test_level_is_node(self):
        from circuitkit.api import _build_circuit_scores

        cs = _build_circuit_scores("ioi", "gpt2", "eap", {})
        assert cs.level == "node"

    def test_timestamp_is_string(self):
        from circuitkit.api import _build_circuit_scores

        cs = _build_circuit_scores("ioi", "gpt2", "eap", {})
        assert isinstance(cs.timestamp, str)

    def test_discovery_cfg_defaults_to_empty_dict(self):
        from circuitkit.api import _build_circuit_scores

        cs = _build_circuit_scores("ioi", "gpt2", "eap", {}, discovery_cfg=None)
        assert cs.discovery_cfg == {}

    def test_discovery_cfg_is_stored(self):
        from circuitkit.api import _build_circuit_scores

        cfg = {"method": "IG"}
        cs = _build_circuit_scores("ioi", "gpt2", "eap-ig", {}, discovery_cfg=cfg)
        assert cs.discovery_cfg == cfg


# ---------------------------------------------------------------------------
# _save_evaluation_results_to_txt
# ---------------------------------------------------------------------------


class TestSaveEvaluationResultsToTxt:
    """Tests for the plain-text result writer."""

    def _make_results(self):
        return [
            {"model_type": "original", "results": {"gsm8k": {"acc": 0.42}}},
            {"model_type": "pruned", "results": {"gsm8k": {"acc": 0.38}}},
        ]

    def _call(self, results, custom_path=None):
        from circuitkit.api import _save_evaluation_results_to_txt

        logger = MagicMock()
        return _save_evaluation_results_to_txt(
            results,
            model_name="gpt2",
            pruned_artifact_path="/fake/path.pt",
            evaluation_mode="both",
            logger=logger,
            custom_path=custom_path,
        )

    def test_writes_file_to_custom_path(self, tmp_path):
        path = str(tmp_path / "results.txt")
        ret = self._call(self._make_results(), custom_path=path)
        assert ret == path
        assert os.path.exists(path)

    def test_file_contains_model_name(self, tmp_path):
        path = str(tmp_path / "r.txt")
        self._call(self._make_results(), custom_path=path)
        content = Path(path).read_text()
        assert "gpt2" in content

    def test_file_contains_evaluation_mode(self, tmp_path):
        path = str(tmp_path / "r.txt")
        self._call(self._make_results(), custom_path=path)
        content = Path(path).read_text()
        assert "both" in content

    def test_file_contains_task_results(self, tmp_path):
        path = str(tmp_path / "r.txt")
        self._call(self._make_results(), custom_path=path)
        content = Path(path).read_text()
        assert "gsm8k" in content

    def test_file_marks_original_and_pruned_sections(self, tmp_path):
        path = str(tmp_path / "r.txt")
        self._call(self._make_results(), custom_path=path)
        content = Path(path).read_text()
        assert "ORIGINAL" in content.upper()
        assert "PRUNED" in content.upper()

    def test_auto_generates_path_when_none_given(self, tmp_path):
        # Don't provide custom_path — it should auto-generate in cwd
        from circuitkit.api import _save_evaluation_results_to_txt

        logger = MagicMock()
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            ret = _save_evaluation_results_to_txt(
                self._make_results(), "gpt2", "/p.pt", "both", logger, custom_path=None
            )
        finally:
            os.chdir(original_cwd)
        assert ret != ""
        assert os.path.exists(ret)

    def test_returns_empty_string_on_failure(self):
        from circuitkit.api import _save_evaluation_results_to_txt

        logger = MagicMock()
        # Pass an invalid path (directory that can't be created)
        ret = _save_evaluation_results_to_txt(
            self._make_results(),
            "gpt2",
            "/p.pt",
            "both",
            logger,
            custom_path="/dev/null/impossible/path/r.txt",
        )
        assert ret == ""


# ---------------------------------------------------------------------------
# _build_random_ibcircuit_neuron_pruning_dict
# ---------------------------------------------------------------------------


class TestBuildRandomIbcircuitNeuronPruningDict:
    """Tests for the random neuron baseline builder."""

    def _make_model(self, n_layers=2, n_heads=2, d_head=4, d_mlp=8, d_model=16):
        model = MagicMock()
        model.cfg.n_layers = n_layers
        model.cfg.n_heads = n_heads
        model.cfg.d_head = d_head
        model.cfg.d_mlp = d_mlp
        model.cfg.d_model = d_model
        return model

    def test_returns_dict_with_required_keys(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        ref = {"mlp": {0: [0, 1]}, "heads": {(0, 0): [2]}, "_meta": {"mlp_hook": "mlp_out"}}
        model = self._make_model()
        result = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="both")
        assert "mlp" in result and "heads" in result and "_meta" in result

    def test_total_pruned_neurons_matches_reference(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        # Reference pruned 5 neurons total
        ref = {
            "mlp": {0: [0, 1, 2]},
            "heads": {(0, 1): [3, 4]},
            "_meta": {"mlp_hook": "mlp_out"},
        }
        model = self._make_model()
        result = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="both", seed=42)
        total = sum(len(v) for v in result["mlp"].values()) + sum(
            len(v) for v in result["heads"].values()
        )
        assert total == 5

    def test_scope_heads_only_samples_attention(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        ref = {"mlp": {}, "heads": {(0, 0): [0]}, "_meta": {"mlp_hook": "mlp_out"}}
        model = self._make_model()
        result = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="heads", seed=0)
        assert result["mlp"] == {}

    def test_scope_mlp_only_samples_mlp(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        ref = {"mlp": {0: [0]}, "heads": {}, "_meta": {"mlp_hook": "mlp_out"}}
        model = self._make_model()
        result = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="mlp", seed=0)
        assert result["heads"] == {}

    def test_mlp_hook_preserved_in_meta(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        ref = {"mlp": {0: [0]}, "heads": {}, "_meta": {"mlp_hook": "post_act"}}
        model = self._make_model()
        result = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="mlp", seed=0)
        assert result["_meta"]["mlp_hook"] == "post_act"

    def test_seeded_result_is_reproducible(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        ref = {"mlp": {0: [0, 1, 2]}, "heads": {}, "_meta": {"mlp_hook": "mlp_out"}}
        model = self._make_model()
        r1 = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="mlp", seed=123)
        r2 = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="mlp", seed=123)
        assert r1["mlp"] == r2["mlp"]

    def test_empty_reference_produces_empty_pruning(self):
        from circuitkit.api import _build_random_ibcircuit_neuron_pruning_dict

        ref = {"mlp": {}, "heads": {}, "_meta": {"mlp_hook": "mlp_out"}}
        model = self._make_model()
        result = _build_random_ibcircuit_neuron_pruning_dict(model, ref, scope="both", seed=0)
        assert result["mlp"] == {} and result["heads"] == {}


# ---------------------------------------------------------------------------
# load_circuit
# ---------------------------------------------------------------------------


class TestLoadCircuit:
    """Tests for the circuit artifact loader."""

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        from circuitkit.api import load_circuit

        with pytest.raises(Exception):  # FileNotFoundError or FileError
            load_circuit(str(tmp_path / "nonexistent.pt"))

    def test_loads_node_level_list(self, tmp_path):
        from circuitkit.api import load_circuit

        data = ["A0.0", "MLP 1", "A2.3"]
        path = str(tmp_path / "circuit.pt")
        torch.save(data, path)
        loaded = load_circuit(path)
        assert loaded == data

    def test_loads_neuron_level_dict(self, tmp_path):
        from circuitkit.api import load_circuit

        data = {
            "mlp": {0: [1, 2, 3]},
            "heads": {(1, 0): [0]},
            "_meta": {"mlp_hook": "mlp_out"},
        }
        path = str(tmp_path / "neuron_circuit.pt")
        torch.save(data, path)
        loaded = load_circuit(path)
        assert loaded["_meta"]["mlp_hook"] == "mlp_out"
        assert loaded["mlp"][0] == [1, 2, 3]

    def test_loaded_type_matches_saved_type_list(self, tmp_path):
        from circuitkit.api import load_circuit

        path = str(tmp_path / "c.pt")
        torch.save(["A0.0"], path)
        result = load_circuit(path)
        assert isinstance(result, list)

    def test_loaded_type_matches_saved_type_dict(self, tmp_path):
        from circuitkit.api import load_circuit

        path = str(tmp_path / "c.pt")
        torch.save({"mlp": {}, "heads": {}}, path)
        result = load_circuit(path)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# discover_circuit — config validation (no real model loading)
# ---------------------------------------------------------------------------


class TestDiscoverCircuitConfigValidation:
    """
    Light-weight config-level tests for discover_circuit that do NOT
    load a real model.  All model/backend calls are mocked out.
    """

    def _minimal_config(self, algo="eap", task="ioi"):
        return {
            "model": {"name": "gpt2", "precision": "float32"},
            "discovery": {"algorithm": algo, "task": task, "level": "node"},
            "pruning": {"target_sparsity": 0.3, "scope": "both"},
        }

    def test_missing_task_key_raises(self):
        """Omitting discovery.task is an error unless an inline 'data' section
        supplies the dataset. 'task' is intentionally NOT defaulted (the old
        'ioi' default was removed so a missing task fails loud instead of
        silently discovering the wrong circuit)."""
        from circuitkit.utils.config import load_and_validate_config

        cfg = self._minimal_config()
        del cfg["discovery"]["task"]
        with pytest.raises(ValueError, match="requires 'discovery.task'"):
            load_and_validate_config(cfg)

    def test_missing_task_key_ok_with_inline_data(self):
        """A missing task is fine when an inline 'data' section with a 'type'
        provides the dataset instead."""
        from circuitkit.utils.config import load_and_validate_config

        cfg = self._minimal_config()
        del cfg["discovery"]["task"]
        cfg["data"] = {"type": "clean_only", "path": "dummy.csv"}
        merged = load_and_validate_config(cfg)  # must not raise
        assert "task" not in merged["discovery"]

    def test_unknown_algorithm_raises(self):
        from circuitkit.api import discover_circuit

        cfg = self._minimal_config(algo="unknown_algo")
        # AlgorithmError or ValueError should be raised
        with pytest.raises(Exception):
            discover_circuit(cfg)

    def test_invalid_model_name_raises(self):
        from circuitkit.api import discover_circuit
        from circuitkit.utils.exceptions import ValidationError

        cfg = self._minimal_config()
        cfg["model"]["name"] = ""  # empty name
        # An empty name is rejected up front by validate_model_name; assert the
        # specific ValidationError so this can't be satisfied by an unrelated
        # later failure (network, model load) masking a regression in the guard.
        with pytest.raises(ValidationError):
            discover_circuit(cfg)


# ---------------------------------------------------------------------------
# benchmark_circuit — input validation (no real model loading)
# ---------------------------------------------------------------------------


class TestBenchmarkCircuitInputValidation:
    """Validation tests for benchmark_circuit that avoid loading models."""

    def test_invalid_evaluation_mode_raises(self, tmp_path):
        from circuitkit.api import benchmark_circuit
        from circuitkit.utils.exceptions import ValidationError

        path = str(tmp_path / "fake.pt")
        torch.save(["A0.0"], path)

        with (
            patch("circuitkit.api.validate_model_name"),
            patch("circuitkit.api.validate_file_exists"),
            patch("circuitkit.api.HookedTransformer.from_pretrained"),
            patch("circuitkit.api.t.load", return_value=["A0.0"]),
        ):
            with pytest.raises((ValueError, ValidationError), match="evaluation_mode"):
                benchmark_circuit(
                    model_name="gpt2",
                    pruned_artifact_path=path,
                    eval_params={},
                    config_for_report={},
                    evaluation_mode="invalid_mode",
                )

    def test_nonexistent_artifact_path_raises(self, tmp_path):
        from circuitkit.api import benchmark_circuit

        bad_path = str(tmp_path / "does_not_exist.pt")
        with pytest.raises(Exception):
            benchmark_circuit(
                model_name="gpt2",
                pruned_artifact_path=bad_path,
                eval_params={},
                config_for_report={},
            )

    def test_valid_modes_are_both_original_pruned(self, tmp_path):
        """Verify the three valid modes don't raise on mode-string check."""
        from circuitkit.api import benchmark_circuit

        path = str(tmp_path / "art.pt")
        torch.save(["A0.0"], path)

        for mode in ("both", "original", "pruned"):
            with (
                patch("circuitkit.api.validate_model_name"),
                patch("circuitkit.api.validate_file_exists"),
                patch("circuitkit.api.HookedTransformer.from_pretrained") as mock_model,
                patch("circuitkit.api.t.load", return_value=["A0.0"]),
            ):
                mock_model.return_value = MagicMock()
                # Should NOT raise ValueError about evaluation_mode
                # (may raise later due to mocked model, that's acceptable)
                try:
                    benchmark_circuit(
                        model_name="gpt2",
                        pruned_artifact_path=path,
                        eval_params={"lm_eval": {"enabled": False}},
                        config_for_report={},
                        evaluation_mode=mode,
                    )
                except ValueError as e:
                    assert "evaluation_mode" not in str(
                        e
                    ), f"Mode '{mode}' should be valid but got ValueError: {e}"
                except Exception:
                    pass  # Non-mode errors are fine in mocked context


# ---------------------------------------------------------------------------
# Integration-style: EAPDiscoveryDataset → DataLoader batch structure
# ---------------------------------------------------------------------------


class TestEapDatasetToDataLoaderIntegration:
    """End-to-end test: CSV → Dataset → DataLoader → batch inspection."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame(
            {
                "clean": [f"sentence {i}" for i in range(8)],
                "corrupted": [f"corrupt {i}" for i in range(8)],
                "correct_idx": list(range(8)),
                "incorrect_idx": list(range(8, 16)),
            }
        ).to_csv(path, index=False)
        return str(path)

    def test_dataloader_iterates_without_error(self, sample_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(sample_csv)
        dl = ds.to_dataloader(batch_size=4)
        batches = list(dl)
        assert len(batches) == 2  # 8 samples / batch_size 4

    def test_each_batch_has_three_elements(self, sample_csv):
        from circuitkit.api import EAPDiscoveryDataset

        ds = EAPDiscoveryDataset(sample_csv)
        dl = ds.to_dataloader(batch_size=4)
        for batch in dl:
            assert len(batch) == 3  # clean, corrupted, labels


# ---------------------------------------------------------------------------
# _eap_accuracy — realistic multi-class scenario
# ---------------------------------------------------------------------------


class TestEapAccuracyRealisticScenario:
    """
    Simulate a realistic token prediction scenario: a small "model" that
    correctly predicts half the batch to verify the 0.5 accuracy output.
    """

    def test_half_correct_gives_05_accuracy(self):
        from circuitkit.api import _eap_accuracy

        vocab = 20
        batch = 4
        seq = 6
        logits = torch.zeros(batch, seq, vocab)
        labels = torch.tensor([3, 7, 11, 15])

        # Correct predictions for first 2 examples
        logits[0, seq - 1, 3] = 100.0
        logits[1, seq - 1, 7] = 100.0
        # Wrong predictions for last 2
        logits[2, seq - 1, 0] = 100.0  # label is 11
        logits[3, seq - 1, 0] = 100.0  # label is 15

        il = torch.full((batch,), seq, dtype=torch.long)
        clean = torch.zeros_like(logits)

        result = _eap_accuracy(logits, clean, il, labels, mean=True)
        assert result.item() == pytest.approx(0.5)

    def test_batch_size_one_accuracy(self):
        from circuitkit.api import _eap_accuracy

        vocab = 5
        logits = torch.zeros(1, 3, vocab)
        logits[0, 2, 4] = 100.0  # correct
        il = torch.tensor([3])
        labels = torch.tensor([4])
        clean = torch.zeros_like(logits)
        result = _eap_accuracy(logits, clean, il, labels, mean=True)
        assert result.item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GPU smoke tests (skipped if CUDA unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
class TestGpuSmoke:
    """Smoke tests that run on GPU when available."""

    def test_eap_accuracy_gpu_output_device(self):
        from circuitkit.api import _eap_accuracy

        device = "cuda"
        batch, seq, vocab = 4, 5, 10
        logits = torch.randn(batch, seq, vocab, device=device)
        clean = torch.zeros_like(logits)
        il = torch.full((batch,), seq, dtype=torch.long, device=device)
        labels = torch.randint(0, vocab, (batch,), device=device)
        result = _eap_accuracy(logits, clean, il, labels)
        assert result.device.type == "cuda"

    def test_avg_scores_on_gpu_tensor(self):
        from circuitkit.api import _avg_scores

        scores = torch.tensor([0.1, 0.5, 0.9], device="cuda")
        result = _avg_scores(scores)
        assert abs(result - 0.5) < 1e-5

    def test_save_and_load_circuit_with_gpu_tensor(self, tmp_path):
        from circuitkit.api import load_circuit

        data = {"mlp": {0: [0]}, "heads": {}, "_meta": {}}
        path = str(tmp_path / "gpu_circuit.pt")
        torch.save(data, path)
        loaded = load_circuit(path)
        assert "mlp" in loaded
