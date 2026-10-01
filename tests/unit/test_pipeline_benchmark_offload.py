"""Pipeline.benchmark parks resident TransformerLens models on CPU and restores them."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformer_lens import HookedTransformer

from circuitkit.pipeline import Pipeline


def _fake_tl_model(device: str):
    model = MagicMock(spec=HookedTransformer)
    model.parameters.side_effect = lambda: iter([MagicMock(device=torch.device(device))])
    return model


def _pipe(tmp_path, model):
    art = tmp_path / "c.pt"
    art.touch()
    p = Pipeline("gpt2", task="ioi", output_dir=str(tmp_path))
    p._artifact_path = str(art)
    p._model = model
    return p


def test_model_is_on_cpu_during_benchmark_and_restored_after(tmp_path):
    model = _fake_tl_model("cuda:0")
    seen = []
    with patch("circuitkit.api.benchmark_circuit", side_effect=lambda *a, **k: seen.extend(model.to.call_args_list)):
        _pipe(tmp_path, model).benchmark()
    assert [c.args[0] for c in seen] == ["cpu"]
    assert [str(c.args[0]) for c in model.to.call_args_list] == ["cpu", "cuda:0"]


def test_cpu_model_is_left_alone(tmp_path):
    model = _fake_tl_model("cpu")
    with patch("circuitkit.api.benchmark_circuit"):
        _pipe(tmp_path, model).benchmark()
    model.to.assert_not_called()


def test_restored_even_if_benchmark_raises(tmp_path):
    model = _fake_tl_model("cuda:0")
    p = _pipe(tmp_path, model)
    with patch("circuitkit.api.benchmark_circuit", side_effect=RuntimeError("boom")):
        try:
            p.benchmark()
        except RuntimeError:
            pass
    assert str(model.to.call_args_list[-1].args[0]) == "cuda:0"


def test_no_model_is_fine(tmp_path):
    with patch("circuitkit.api.benchmark_circuit") as bench:
        _pipe(tmp_path, None).benchmark()
    bench.assert_called_once()
