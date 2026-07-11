"""
Unit tests for the flat front-door API.

Location: circuitkit/tests/unit/test_quick_api.py

Covers:
- circuitkit.quick.build_discovery_config: config shape, validation errors.
- circuitkit.quick.discover / faithfulness / prune / quantize / export_checkpoint
  / benchmark: argument validation and that they call the engine correctly
  (engine calls are mocked — no GPU, no real model required).
- circuitkit.circuit.Circuit: construction, dunders, save/load round-trip,
  top_nodes, graceful plot degradation.
- circuitkit package: the flat names are exported and import stays light.

These tests do not require a GPU. The single model-touching test is gated
behind importorskip + gpt2.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import circuitkit as ck
from circuitkit import quick
from circuitkit.circuit import Circuit


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _fake_model(model_name="gpt2", n_layers=12):
    """A minimal stand-in for a HookedTransformer (just the cfg attrs we read)."""
    model = MagicMock()
    model.cfg.model_name = model_name
    # TransformerLens stores the full HF repo name on cfg.tokenizer_name;
    # build_discovery_config prefers it over the stripped model_name alias.
    model.cfg.tokenizer_name = model_name
    model.cfg.dtype = "torch.float32"
    model.cfg.n_layers = n_layers
    model.cfg.n_heads = 12
    return model


# --------------------------------------------------------------------------- #
# Package surface                                                             #
# --------------------------------------------------------------------------- #
def test_flat_api_exported():
    surface = [x for x in dir(ck) if not x.startswith("_")]
    for name in (
        "load_model",
        "discover",
        "faithfulness",
        "prune",
        "quantize",
        "export_checkpoint",
        "benchmark",
        "Circuit",
    ):
        assert name in surface, f"{name} missing from circuitkit surface"


def test_circuit_export_is_the_class():
    assert ck.Circuit is Circuit


# --------------------------------------------------------------------------- #
# build_discovery_config                                                      #
# --------------------------------------------------------------------------- #
def test_build_discovery_config_shape():
    cfg = quick.build_discovery_config(
        _fake_model(), "ioi", algorithm="eap-ig", n_examples=32, batch_size=8
    )
    assert set(cfg) == {"model", "discovery", "pruning", "output_path"}
    assert cfg["model"]["name"] == "gpt2"
    assert cfg["model"]["precision"] == "float32"
    assert cfg["discovery"]["algorithm"] == "eap-ig"
    assert cfg["discovery"]["task"] == "ioi"
    assert cfg["discovery"]["level"] == "node"
    assert cfg["discovery"]["batch_size"] == 8
    assert cfg["discovery"]["data_params"]["num_examples"] == 32
    assert cfg["discovery"]["data_params"]["batch_size"] == 8
    assert cfg["pruning"]["target_sparsity"] == 0.3
    assert cfg["pruning"]["scope"] == "both"


def test_build_discovery_config_extra_kwargs_merged():
    cfg = quick.build_discovery_config(_fake_model(), "ioi", ig_steps=7, intervention="patching")
    assert cfg["discovery"]["ig_steps"] == 7
    assert cfg["discovery"]["intervention"] == "patching"


def test_build_discovery_config_seed_reaches_data_params():
    """seed must land in discovery.data_params.seed, not just discovery.seed.

    Regression: data generation (IOI) reads discovery.data_params.seed and
    otherwise defaults to 42, so ck.discover(seed=...) left the data seed fixed
    -> identical circuits across seeds -> zero-variance error bars.
    """
    cfg = quick.build_discovery_config(_fake_model(), "ioi", seed=5)
    assert cfg["discovery"]["data_params"]["seed"] == 5
    assert cfg["discovery"]["seed"] == 5


def test_build_discovery_config_no_seed_leaves_data_params_unset():
    cfg = quick.build_discovery_config(_fake_model(), "ioi")
    assert "seed" not in cfg["discovery"]["data_params"]


def test_build_discovery_config_rejects_bad_algorithm():
    with pytest.raises(ValueError, match="Unknown discovery algorithm"):
        quick.build_discovery_config(_fake_model(), "ioi", algorithm="not-an-algo")


def test_build_discovery_config_bad_algorithm_lists_valid():
    with pytest.raises(ValueError) as exc:
        quick.build_discovery_config(_fake_model(), "ioi", algorithm="xyz")
    # Error message should help the user by listing real algorithms.
    assert "eap-ig" in str(exc.value)


def test_build_discovery_config_rejects_bad_level():
    with pytest.raises(ValueError, match="level must be"):
        quick.build_discovery_config(_fake_model(), "ioi", level="atom")


def test_build_discovery_config_rejects_bad_scope():
    with pytest.raises(ValueError, match="scope must be"):
        quick.build_discovery_config(_fake_model(), "ioi", scope="everything")


def test_build_discovery_config_rejects_bad_sparsity():
    with pytest.raises(ValueError, match="sparsity"):
        quick.build_discovery_config(_fake_model(), "ioi", sparsity=1.5)


def test_build_discovery_config_rejects_bad_n_examples():
    with pytest.raises(ValueError, match="n_examples"):
        quick.build_discovery_config(_fake_model(), "ioi", n_examples=0)


# --------------------------------------------------------------------------- #
# discover                                                                    #
# --------------------------------------------------------------------------- #
def test_discover_calls_engine_and_returns_circuit(tmp_path):
    out = str(tmp_path / "circ.pt")
    nodes = ["A0.1", "MLP 3"]

    with patch("circuitkit.api.discover_circuit", return_value=nodes) as mock_dc:
        circuit = quick.discover(
            _fake_model(), "ioi", algorithm="eap", n_examples=16, output_path=out
        )

    # Engine was called once with a fully-formed dict config.
    assert mock_dc.call_count == 1
    sent_cfg = mock_dc.call_args[0][0]
    assert sent_cfg["discovery"]["algorithm"] == "eap"
    assert sent_cfg["discovery"]["data_params"]["num_examples"] == 16
    assert sent_cfg["output_path"] == out

    assert isinstance(circuit, Circuit)
    assert circuit.nodes == nodes
    assert circuit.task == "ioi"
    assert circuit.algorithm == "eap"
    assert len(circuit) == 2


def test_discover_rejects_bad_algorithm():
    with pytest.raises(ValueError, match="Unknown discovery algorithm"):
        quick.discover(_fake_model(), "ioi", algorithm="bogus")


# --------------------------------------------------------------------------- #
# Circuit class                                                               #
# --------------------------------------------------------------------------- #
def test_circuit_node_level_dunders():
    c = Circuit(["A0.1", "A1.2", "MLP 0"], {"A0.1": 0.9, "A1.2": 0.5})
    assert len(c) == 3
    assert "A0.1" in c
    assert "A9.9" not in c
    assert list(c) == ["A0.1", "A1.2", "MLP 0"]
    assert "Circuit(" in repr(c)
    assert "n_nodes=3" in repr(c)


def test_circuit_neuron_level_len():
    nodes = {"mlp": {0: [1, 2]}, "heads": {(0, 1): [3]}, "_meta": {}}
    c = Circuit(nodes, level="neuron")
    assert len(c) == 2  # one mlp entry + one head entry
    assert c.level == "neuron"


def test_circuit_top_nodes():
    c = Circuit(["a", "b", "c"], {"a": 0.1, "b": 0.9, "c": 0.5})
    top2 = c.top_nodes(2)
    assert list(top2) == ["b", "c"]
    assert top2["b"] == 0.9


def test_circuit_top_nodes_empty_when_no_scores():
    c = Circuit(["a", "b"])
    assert c.top_nodes(5) == {}


def test_circuit_save_and_from_artifact_roundtrip(tmp_path):
    out = tmp_path / "circ.pt"
    c = Circuit(
        ["A0.1", "MLP 2"],
        {"A0.1": 0.7, "MLP 2": 0.3},
        task="ioi",
        algorithm="eap-ig",
        model_name="gpt2",
    )
    saved = c.save(out)
    assert saved.exists()

    # A scores side-car should have been written next to the artifact.
    scores_json = tmp_path / "circ_scores.json"
    assert scores_json.exists()
    with open(scores_json) as f:
        blob = json.load(f)
    assert blob["node_scores"]["A0.1"] == 0.7

    reloaded = Circuit.from_artifact(out)
    assert reloaded.nodes == ["A0.1", "MLP 2"]
    assert reloaded.scores["A0.1"] == 0.7
    assert reloaded.level == "node"


def test_circuit_from_artifact_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Circuit.from_artifact(tmp_path / "does_not_exist.pt")


def test_circuit_plot_degrades_without_scores(caplog):
    import logging

    c = Circuit(["A0.1"])  # no scores
    with caplog.at_level(logging.INFO):
        result = c.plot()
    assert result is None
    assert "no node scores" in caplog.text.lower()


def test_circuit_backfills_metadata_from_circuit_scores():
    from circuitkit.artifacts.scores import CircuitScores

    cs = CircuitScores(
        task="ioi",
        model="gpt2",
        algorithm="eap-ig",
        level="node",
        node_scores={"A0.1": 0.5},
        timestamp=CircuitScores.create_timestamp(),
    )
    c = Circuit(["A0.1"], circuit_scores=cs)
    assert c.task == "ioi"
    assert c.algorithm == "eap-ig"
    assert c.model_name == "gpt2"
    assert c.scores == {"A0.1": 0.5}


# --------------------------------------------------------------------------- #
# faithfulness                                                                #
# --------------------------------------------------------------------------- #
def test_faithfulness_rejects_circuit_without_scores():
    c = Circuit(["A0.1"])  # no scores
    with pytest.raises(ValueError, match="no node scores"):
        quick.faithfulness(_fake_model(), c, "ioi")


def test_faithfulness_calls_run_full_faithfulness():
    c = Circuit(["A0.1"], {"A0.1": 0.5}, algorithm="eap-ig", task="ioi")
    fake_report = MagicMock()

    with (
        patch("circuitkit.tasks.bootstrap._bootstrap_builtin_tasks"),
        patch("circuitkit.tasks.registry.get_task") as mock_get_task,
        patch("circuitkit.api._reconstruct_circuit_graph", return_value=MagicMock()),
        patch("circuitkit.evaluation.run_full_faithfulness", return_value=fake_report) as mock_rff,
    ):
        task_spec = MagicMock()
        mock_get_task.return_value = task_spec
        report = quick.faithfulness(_fake_model(), c, "ioi", pillars=["patching"], device="cpu")

    assert report is fake_report
    assert mock_rff.call_count == 1
    kwargs = mock_rff.call_args.kwargs
    assert kwargs["pillars"] == ["patching"]
    assert kwargs["discovery_cfg"]["task"] == "ioi"
    assert kwargs["device"] == "cpu"


def test_faithfulness_reconstructs_at_circuit_sparsity():
    """faithfulness must reconstruct at the circuit's ACTUAL sparsity, not 0.0.

    Regression: the reconstruction hardcoded target_sparsity=0.0 and only set
    node.in_graph (never the 2-D edge matrix), so every circuit evaluated as the
    corrupt baseline (faithfulness ~0). It must now derive sparsity from the
    pruned-node fraction and route through api._reconstruct_circuit_graph.
    """
    # 12 layers x (12 heads + 1 MLP) = 156 scoreable; prune 46 -> sparsity ~0.295
    pruned = [f"A{lyr}.{h}" for lyr in range(4) for h in range(12)][:46]
    c = Circuit(pruned, {n: 0.1 for n in pruned}, algorithm="eap-ig", task="ioi")
    fake_report = MagicMock()
    with (
        patch("circuitkit.tasks.bootstrap._bootstrap_builtin_tasks"),
        patch("circuitkit.tasks.registry.get_task", return_value=MagicMock()),
        patch("circuitkit.api._reconstruct_circuit_graph", return_value=MagicMock()) as mock_rc,
        patch("circuitkit.evaluation.run_full_faithfulness", return_value=fake_report),
    ):
        quick.faithfulness(_fake_model(), c, "ioi", pillars=["patching"], device="cpu")
    # api._reconstruct_circuit_graph(model, scores_data, discovery_cfg, pruning_cfg, device)
    pruning_cfg = mock_rc.call_args[0][3]
    assert pruning_cfg["target_sparsity"] > 0.0
    assert abs(pruning_cfg["target_sparsity"] - 46 / 156) < 1e-6


def test_faithfulness_seed_reaches_stability_seed():
    """faithfulness(seed=s) must reach discovery.data_params.seed.

    run_full_faithfulness reads discovery.data_params.seed as Pillar 3's
    seed_start (else 42). Without threading it, per-seed stability was measured
    over the same fixed neighbourhood for every circuit -> constant jaccard.
    """
    c = Circuit(["A0.1"], {"A0.1": 0.5}, algorithm="eap-ig", task="ioi")
    fake_report = MagicMock()
    with (
        patch("circuitkit.tasks.bootstrap._bootstrap_builtin_tasks"),
        patch("circuitkit.tasks.registry.get_task", return_value=MagicMock()),
        patch("circuitkit.api._reconstruct_circuit_graph", return_value=MagicMock()),
        patch("circuitkit.evaluation.run_full_faithfulness", return_value=fake_report) as mock_rff,
    ):
        quick.faithfulness(_fake_model(), c, "ioi", pillars=["stability"], device="cpu", seed=11)
    assert mock_rff.call_args.kwargs["discovery_cfg"]["data_params"]["seed"] == 11


# --------------------------------------------------------------------------- #
# prune                                                                       #
# --------------------------------------------------------------------------- #
def test_prune_rejects_neuron_level_circuit():
    c = Circuit({"mlp": {}, "heads": {}, "_meta": {}}, level="neuron")
    with pytest.raises(ValueError, match="node-level"):
        quick.prune(_fake_model(), c)


def test_prune_rejects_circuit_without_scores():
    c = Circuit(["A0.1"])
    with pytest.raises(ValueError, match="no node scores"):
        quick.prune(_fake_model(), c)


def test_prune_calls_structural_pruner():
    c = Circuit(["A0.1"], {"A0.1": 0.5}, algorithm="eap-ig", task="ioi", model_name="gpt2")
    pruned = MagicMock()

    with patch("circuitkit.applications.pruning.StructuralPruner") as MockPruner:
        instance = MockPruner.return_value
        instance.prune.return_value = pruned
        result = quick.prune(_fake_model(), c, sparsity=0.4)

    assert result is pruned
    kwargs = instance.prune.call_args.kwargs
    assert kwargs["sparsity"] == 0.4
    # Default is copy mode: inplace=False, and no deprecated dry_run passed.
    assert kwargs["inplace"] is False
    assert kwargs["dry_run"] is None


def test_prune_inplace_passthrough():
    c = Circuit(["A0.1"], {"A0.1": 0.5}, algorithm="eap-ig", task="ioi", model_name="gpt2")

    with patch("circuitkit.applications.pruning.StructuralPruner") as MockPruner:
        instance = MockPruner.return_value
        instance.prune.return_value = MagicMock()
        quick.prune(_fake_model(), c, inplace=True)

    assert instance.prune.call_args.kwargs["inplace"] is True


def test_prune_scope_and_protect_layers_passthrough():
    """scope / protect_layers are forwarded to StructuralPruner.prune."""
    c = Circuit(["A0.1"], {"A0.1": 0.5}, algorithm="eap-ig", task="ioi", model_name="gpt2")

    with patch("circuitkit.applications.pruning.StructuralPruner") as MockPruner:
        instance = MockPruner.return_value
        instance.prune.return_value = MagicMock()
        quick.prune(_fake_model(), c, scope="mlp", protect_layers=[0, 1])

    kwargs = instance.prune.call_args.kwargs
    assert kwargs["scope"] == "mlp"
    assert kwargs["protect_layers"] == [0, 1]


def test_prune_defaults_scope_both_no_protect():
    """Default prune() passes scope='both' and protect_layers=None."""
    c = Circuit(["A0.1"], {"A0.1": 0.5}, algorithm="eap-ig", task="ioi", model_name="gpt2")

    with patch("circuitkit.applications.pruning.StructuralPruner") as MockPruner:
        instance = MockPruner.return_value
        instance.prune.return_value = MagicMock()
        quick.prune(_fake_model(), c)

    kwargs = instance.prune.call_args.kwargs
    assert kwargs["scope"] == "both"
    assert kwargs["protect_layers"] is None


# --------------------------------------------------------------------------- #
# quantize                                                                    #
# --------------------------------------------------------------------------- #
def test_quantize_rejects_circuit_without_scores():
    c = Circuit(["A0.1"])
    with pytest.raises(ValueError, match="no node scores"):
        quick.quantize(MagicMock(), c, n_layers=12)


def test_quantize_derives_scores_and_calls_circuit_quantize():
    c = Circuit(
        ["A0.1", "MLP 2"],
        {"A0.1": 0.8, "A1.3": 0.2, "MLP 2": 0.5},
    )
    plan = {"low": {}, "high": {}}

    with patch(
        "circuitkit.applications.quantization.circuit_quantize", return_value=plan
    ) as mock_cq:
        result = quick.quantize(MagicMock(), c, n_layers=4, high_fraction=0.25)

    assert result is plan
    kwargs = mock_cq.call_args.kwargs
    # Heads parsed to (layer, head) tuples.
    assert kwargs["q_head_scores"][(0, 1)] == 0.8
    assert kwargs["q_head_scores"][(1, 3)] == 0.2
    # MLP parsed to {layer: score}.
    assert kwargs["mlp_scores"][2] == 0.5
    assert kwargs["n_layers"] == 4
    assert kwargs["high_fraction"] == 0.25
    # protect_layers defaults to None.
    assert kwargs["protect_layers"] is None


def test_quantize_protect_layers_passthrough():
    """protect_layers is forwarded to circuit_quantize."""
    c = Circuit(["A0.1", "MLP 2"], {"A0.1": 0.8, "MLP 2": 0.5})
    plan = {"low": {}, "high": {}}

    with patch(
        "circuitkit.applications.quantization.circuit_quantize", return_value=plan
    ) as mock_cq:
        quick.quantize(MagicMock(), c, n_layers=4, protect_layers=[0, 3])

    assert mock_cq.call_args.kwargs["protect_layers"] == [0, 3]


# --------------------------------------------------------------------------- #
# export_checkpoint                                                           #
# --------------------------------------------------------------------------- #
def test_export_checkpoint_pruning_unwraps_circuit():
    c = Circuit(["A0.1", "MLP 1"], {"A0.1": 0.5})
    with (
        patch("circuitkit.evaluation.save_pruned_checkpoint") as mock_save,
        patch("circuitkit.evaluation.save_quantized_checkpoint"),
    ):
        path = quick.export_checkpoint(MagicMock(), c, "ckpt/x")
    assert path == "ckpt/x"
    # The node list (not the Circuit object) is forwarded to the engine.
    assert mock_save.call_args[0][1] == ["A0.1", "MLP 1"]


def test_export_checkpoint_pruning_requires_artifact():
    with pytest.raises(ValueError, match="needs an artifact"):
        quick.export_checkpoint(MagicMock(), None, "ckpt/x")


def test_export_checkpoint_rejects_unknown_intervention():
    with pytest.raises(ValueError, match="intervention must be"):
        quick.export_checkpoint(MagicMock(), ["A0.1"], "ckpt/x", intervention="magic")


def test_export_checkpoint_quantization_path():
    with (
        patch("circuitkit.evaluation.save_quantized_checkpoint") as mock_save,
        patch("circuitkit.evaluation.save_pruned_checkpoint"),
    ):
        quick.export_checkpoint(MagicMock(), None, "ckpt/q", intervention="quantization")
    assert mock_save.call_count == 1


# --------------------------------------------------------------------------- #
# benchmark                                                                   #
# --------------------------------------------------------------------------- #
def test_benchmark_accepts_string_task():
    with patch("circuitkit.evaluation.run_lm_eval", return_value={"boolq": {}}) as mock_le:
        result = quick.benchmark("ckpt/x", "boolq", limit=10, device="cpu")
    assert result == {"boolq": {}}
    # String task is normalised to a list before reaching the engine.
    assert mock_le.call_args[0][1] == ["boolq"]
    assert mock_le.call_args.kwargs["limit"] == 10


def test_benchmark_rejects_empty_tasks():
    with pytest.raises(ValueError, match="at least one"):
        quick.benchmark("ckpt/x", [])


# --------------------------------------------------------------------------- #
# load_model                                                                  #
# --------------------------------------------------------------------------- #
def test_load_model_rejects_bad_dtype():
    with pytest.raises(ValueError, match="Unknown dtype"):
        quick.load_model("gpt2", dtype="not-a-dtype")


def test_load_model_rejects_bad_algorithm():
    with pytest.raises(ValueError, match="Unknown discovery algorithm"):
        quick.load_model("gpt2", algorithm="bogus")


def test_load_model_sets_discovery_flags():
    """load_model must set the four hook flags so discovery doesn't crash."""
    pytest.importorskip("transformer_lens")

    fake_model = MagicMock()
    fake_model.cfg = MagicMock()

    with patch(
        "transformer_lens.HookedTransformer.HookedTransformer.from_pretrained",
        return_value=fake_model,
    ):
        model = quick.load_model("gpt2", dtype="float32", device="cpu")

    assert model.cfg.use_attn_result is True
    assert model.cfg.use_split_qkv_input is True
    assert model.cfg.use_hook_mlp_in is True
    assert model.cfg.ungroup_grouped_query_attention is True
