"""``lexsi.provenance/1``: ``base_model`` is the immediate parent, found among the inputs."""

import pytest

from circuitkit.provenance import make_provenance, read_provenance, write_provenance


def _model_input(ref):
    return {"kind": "model", "ref": str(ref), "provenance": read_provenance(ref)}


def _follows_definition(record):
    """The shared rule: base_model is the ref of a kind='model' entry in inputs (or both absent)."""
    refs = [e["ref"] for e in record["inputs"] if e["kind"] == "model" and e["ref"]]
    return record["base_model"] in refs if refs else record["base_model"] is None


def test_two_step_chain_names_the_immediate_parent(tmp_path):
    step1, step2 = tmp_path / "pruned", tmp_path / "quantized"
    step1.mkdir()
    step2.mkdir()
    first = write_provenance(
        step1,
        "export_checkpoint.pruning",
        inputs=[_model_input("gpt2")],
        base_model_relation="finetune",
    )
    second = write_provenance(
        step2,
        "export_checkpoint.quantization",
        inputs=[_model_input(step1)],
        base_model_relation="quantized",
    )
    assert first["base_model"] == "gpt2" and first["base_model_relation"] == "finetune"
    # step 2 loaded step 1's folder: that is its parent, not "gpt2"
    assert second["base_model"] == str(step1) and second["base_model_relation"] == "quantized"
    assert _follows_definition(first) and _follows_definition(second)
    # the original model is reached by following the embedded record of the parent
    parent = next(e for e in second["inputs"] if e["ref"] == second["base_model"])["provenance"]
    assert parent == first == read_provenance(step1) and parent["base_model"] == "gpt2"
    assert read_provenance(step2) == second


def test_relation_is_left_out_when_not_given():
    record = make_provenance("discover.eap", inputs=[_model_input("gpt2")])
    assert record["base_model"] == "gpt2" and "base_model_relation" not in record


def test_base_model_is_the_first_model_input_and_none_without_one():
    data = {"kind": "dataset", "ref": "some/data", "provenance": None}
    record = make_provenance("merge", inputs=[data, _model_input("a"), _model_input("b")])
    assert record["base_model"] == "a"
    assert (
        make_provenance(
            "merge", inputs=[data, _model_input("a"), _model_input("b")], base_model="b"
        )["base_model"]
        == "b"
    )
    assert make_provenance("curate", inputs=[data])["base_model"] is None
    assert make_provenance("x")["base_model"] is None


def test_wrong_base_model_or_relation_is_rejected():
    with pytest.raises(ValueError, match="kind='model' entry"):
        make_provenance("x", inputs=[_model_input("a")], base_model="root-model")
    with pytest.raises(ValueError, match="base_model_relation"):
        make_provenance("x", inputs=[_model_input("a")], base_model_relation="pruned")
