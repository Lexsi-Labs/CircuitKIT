"""EAP's own node names (``a{layer}.h{head}``, ``m{layer}``) work wherever circuit scores are read (#51).

Node-level discovery writes ``A{layer}.{head}`` / ``MLP {layer}``, but a neuron-level EAP run
keeps the EAP graph's names in ``Circuit.scores``. Every consumer must read both styles, map a
name to the right layer and head, and ignore names it does not know.
"""

import pytest
import torch

from circuitkit.circuit import _canonical_node_name, _parse_head_name, _parse_mlp_name

EAP_SCORES = {"a1.h0": 0.9, "a0.h1": 0.4, "m1": 0.8, "m0": 0.1}


@pytest.mark.parametrize(
    "name, expected",
    [("A14.3", (14, 3)), ("a14.h3", (14, 3)), ("A0.0", (0, 0)), ("a10.h12", (10, 12))],
)
def test_head_names_in_both_styles_parse_to_layer_and_head(name, expected):
    assert _parse_head_name(name) == expected


@pytest.mark.parametrize(
    "name",
    ["A0.1.Q", "a1.h2<q>", "a1.2", "A1.h2", "A1", "m5", "MLP 3", "Resid Start", "input", ""],
)
def test_other_names_are_not_heads(name):
    assert _parse_head_name(name) is None


@pytest.mark.parametrize("name, expected", [("m5", 5), ("MLP 5", 5), ("MLP 12", 12)])
def test_mlp_names_in_both_styles_parse_to_layer(name, expected):
    assert _parse_mlp_name(name) == expected


@pytest.mark.parametrize("name", ["a1.h2", "A1.2", "MLP", "m", "m5x", "MLP 3.5", "logits"])
def test_other_names_are_not_mlps(name):
    assert _parse_mlp_name(name) is None


def test_canonical_name():
    assert _canonical_node_name("a3.h1") == "A3.1"
    assert _canonical_node_name("m2") == "MLP 2"
    assert _canonical_node_name("A3.1") == "A3.1"
    assert _canonical_node_name("Resid Start") == "Resid Start"


# --------------------------------------------------------------------------- #
# CircuitWeightSteering                                                       #
# --------------------------------------------------------------------------- #


def test_weight_steering_takes_eap_head_names():
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering

    cws = CircuitWeightSteering(object(), {"a14.h3": 0.9, "a2.h7": 0.4, "m5": 0.1}, top_k_frac=1.0)
    assert cws.head_names == ["A14.3", "A2.7"]


def test_weight_steering_counts_a_head_named_both_ways_once():
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering

    scores = {"A0.1": 0.2, "a0.h1": 0.9, "A0.2": 0.5, "a1.h0": 0.1}
    cws = CircuitWeightSteering(object(), scores, top_k_frac=0.5)
    # Three distinct heads; the top half by score is A0.1 (at its larger score, 0.9).
    assert cws.head_names == ["A0.1"]


def test_weight_steering_still_rejects_scores_without_heads():
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering

    with pytest.raises(ValueError, match="attention heads"):
        CircuitWeightSteering(object(), {"m5": 0.1, "MLP 2": 0.3, "A0.1.Q": 0.4}, top_k_frac=1.0)


# --------------------------------------------------------------------------- #
# The other consumers of node scores                                          #
# --------------------------------------------------------------------------- #


def test_activation_steering_maps_eap_names_to_hooks():
    from circuitkit.applications.steering.steering import ActivationSteering

    s = ActivationSteering.__new__(ActivationSteering)
    assert s._get_hook_point_from_node("a1.h0") == "blocks.1.attn.hook_result"
    assert s._get_hook_point_from_node("m1") == "blocks.1.hook_mlp_out"
    assert ActivationSteering._node_head_index("a1.h3") == 3
    assert s._get_hook_point_from_node("Resid Start") is None


@pytest.mark.parametrize("module, fn", [
    ("circuitkit.applications.pruning.node_pruner", "get_nodes_to_prune"),
    ("circuitkit.applications.pruning.weight_pruner", "get_nodes_to_prune_from_weights"),
])
def test_node_pruning_selects_eap_names(module, fn):
    import importlib

    select = getattr(importlib.import_module(module), fn)
    scores = {"a0.h0": 0.1, "a0.h1": 0.9, "a1.h0": 0.8, "a1.h1": 0.7, "m0": 0.1, "m1": 0.8}
    # Half the heads (one per layer at most) and half the MLPs, lowest scores first.
    assert sorted(select(scores, 0.5, pruning_scope="both")) == ["a0.h0", "a1.h1", "m0"]


def test_structural_pruner_selects_eap_names():
    from circuitkit.applications.pruning.pruner import StructuralPruner
    from circuitkit.artifacts.scores import CircuitScores

    cs = CircuitScores(
        task="ioi", model="gpt2", algorithm="eap", level="node",
        node_scores=EAP_SCORES, timestamp=CircuitScores.create_timestamp(),
    )
    pruner = StructuralPruner.__new__(StructuralPruner)
    assert sorted(pruner._select_nodes_to_prune(cs, 0.5)) == ["a0.h1", "m0"]
    assert sorted(pruner._select_nodes_to_prune(cs, 1.0, protect_layers=[1])) == ["a0.h1", "m0"]


def test_hook_pruning_builds_hooks_for_eap_names():
    from circuitkit.evaluation.lm_harness import _build_node_hooks

    hooks = [name for name, _ in _build_node_hooks(["a1.h0", "m0"])]
    assert hooks == ["blocks.1.attn.hook_result", "blocks.0.hook_mlp_out"]


def test_hf_checkpoint_expects_zeros_for_eap_names():
    from circuitkit.evaluation.hf_checkpoint import ARCH_CONFIGS, expected_zero_keys

    upper = expected_zero_keys(["A1.0", "MLP 0"], ARCH_CONFIGS["GPT2LMHeadModel"])
    assert upper and expected_zero_keys(["a1.h0", "m0"], ARCH_CONFIGS["GPT2LMHeadModel"]) == upper


def test_quantization_score_conversion_reads_eap_names():
    from circuitkit.applications.quantization.score_extractor import (
        head_scores_str_to_tuple,
        mlp_scores_str_to_int,
    )

    assert head_scores_str_to_tuple(EAP_SCORES) == {(1, 0): 0.9, (0, 1): 0.4}
    assert mlp_scores_str_to_int(EAP_SCORES) == {1: 0.8, 0: 0.1}


@pytest.mark.parametrize("module", [
    "circuitkit.applications.common_utils.cure_clue",
    "circuitkit.applications.finetuning.circuit_tuning",
    "circuitkit.applications.editing.cake",
])
def test_top_mlp_layers_read_eap_names(module):
    import importlib

    top = importlib.import_module(module)._top_mlp_layers
    # Without the fix the m-names are skipped and the middle layers come back instead.
    assert top({"m5": 0.9, "m2": 0.3, "a1.h0": 1.0}, n_layers=8, top_k=1) == [5]


def test_mcircke_best_mlp_layer_reads_eap_names():
    from circuitkit.applications.editing.mcircke import _best_mlp_layer

    assert _best_mlp_layer({"m6": 0.9, "m2": 0.3}, n_layers=8) == 6


def test_interop_fields_read_eap_names():
    from circuitkit.artifacts.scores import interop_fields

    units = interop_fields(EAP_SCORES, "gpt2", top_fraction=1.0)["safety_units"]
    assert units["layer_indices"] == [0, 1]
    assert sorted(units["unit_ids"]) == sorted(EAP_SCORES)


def test_eap_to_artifact_reads_eap_names():
    from circuitkit.artifacts import eap_to_artifact

    art = eap_to_artifact(EAP_SCORES, model_id="gpt2", task="ioi", dataset="ioi")
    assert sorted(art.nodes) == ["L0H1", "L0M0", "L1H0", "L1M0"]


def test_eap_to_artifact_reads_eap_graph_edges():
    from circuitkit.artifacts import eap_to_artifact
    from circuitkit.backends.eap.graph import Graph

    graph = Graph.from_model({"n_layers": 2, "n_heads": 2, "d_model": 8, "d_mlp": 16})
    art = eap_to_artifact({"A0.0": 0.5}, model_id="gpt2", task="ioi", dataset="ioi", graph=graph)
    pairs = {(e.src_id, e.dst_id) for e in art.edges.values()}
    assert ("L0H0", "L1H1") in pairs and ("L0M0", "L1M0") in pairs


def test_graph_node_lookup_reads_both_styles():
    from circuitkit.api import _ib_name_to_graph_name

    assert _ib_name_to_graph_name("A0.1") == _ib_name_to_graph_name("a0.h1") == "a0.h1"
    assert _ib_name_to_graph_name("MLP 2") == _ib_name_to_graph_name("m2") == "m2"
    assert _ib_name_to_graph_name("Resid Start") is None


# --------------------------------------------------------------------------- #
# End to end: neuron-level EAP discovery -> CircuitWeightSteering              #
# --------------------------------------------------------------------------- #


def test_neuron_level_eap_scores_drive_weight_steering(tmp_path, monkeypatch):
    """The report in #51: a real neuron-level EAP run, its scores straight into the steering."""
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformers import AutoTokenizer

    from circuitkit.api import discover_circuit
    from circuitkit.applications.steering.weight_steering import CircuitWeightSteering
    from circuitkit.circuit import Circuit

    monkeypatch.chdir(tmp_path)  # discovery caches its task data under ./cache
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    cfg = HookedTransformerConfig(
        n_layers=2, n_heads=2, d_model=32, d_head=16, d_mlp=64, n_ctx=64,
        d_vocab=tok.vocab_size, act_fn="gelu", normalization_type="LN", device="cpu",
    )
    model = HookedTransformer(cfg)
    model.set_tokenizer(tok)

    out = tmp_path / "neuron.pt"
    discover_circuit(
        {
            "model": {"name": "gpt2", "device": "cpu"},
            "discovery": {
                "algorithm": "eap", "task": "ioi", "level": "neuron", "batch_size": 4,
                "data_params": {"num_examples": 4, "batch_size": 4},
            },
            "pruning": {"target_sparsity": 0.3, "scope": "both"},
            "output_path": str(out),
        },
        _model=model,
    )
    scores = Circuit.from_artifact(out).scores
    assert "a0.h0" in scores and "m0" in scores  # EAP's own names

    cws = CircuitWeightSteering(model, scores, top_k_frac=0.5)
    assert len(cws.head_names) == 2 and all(n.startswith("A") for n in cws.head_names)

    tokens = model.to_tokens(["The cat sat on the mat"] * 2)

    def loss(sign):
        return lambda m, batch: sign * m(batch, return_type="loss")

    cws.fine_tune_positive([tokens], loss(1.0), n_steps=1, lr=1e-2)
    cws.fine_tune_negative([tokens], loss(-1.0), n_steps=1, lr=1e-2)
    steering = cws.compute_steering_vector()
    assert set(steering) == set(cws.head_names)

    steered = cws.apply_steering(k=1.0)
    layer, head = int(cws.head_names[0][1]), int(cws.head_names[0][3])
    assert not torch.equal(steered.blocks[layer].attn.W_Q[head], model.blocks[layer].attn.W_Q[head])

    path = tmp_path / "steer.pt"
    cws.save(str(path))
    loaded = CircuitWeightSteering.load(str(path), model)
    assert torch.equal(
        loaded.apply_steering(k=1.0).blocks[layer].attn.W_Q[head], steered.blocks[layer].attn.W_Q[head]
    )
