"""PR #115 follow-ups: quantize/export composition and the quanto ImportError guard.

Covers the review findings on the quantization fixes:
1. Pipeline.export() with no explicit intervention must follow whichever
   intervention ran last — pipe.quantize().export(path) previously fed the
   quantized HF AutoModelForCausalLM into save_pruned_checkpoint (which
   requires a TransformerLens model) because export defaulted to "pruning".
2. The lazy `from optimum.quanto import qint4` default in
   circuit_quantize/random_quantize must raise an ACTIONABLE ImportError
   (naming circuitkit[quantization]) when the optional extra is missing,
   not a raw ModuleNotFoundError — per pyproject's documented contract.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from circuitkit.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Fixtures (mirroring test_pipeline_unit.py's no-model pattern)
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipe():
    p = Pipeline("gpt2", task="ioi", precision="float32")
    p._circuit = MagicMock()
    return p


# ---------------------------------------------------------------------------
# 1. export() intervention default follows the last intervention
# ---------------------------------------------------------------------------


class TestExportFollowsLastIntervention:
    def test_quantize_records_intervention_and_export_uses_it(self, pipe):
        fake_hf_model = MagicMock()
        with (
            patch.object(Pipeline, "_ensure_hf_model", return_value=fake_hf_model),
            patch("circuitkit.quick.quantize", return_value={"plan": True}),
        ):
            pipe.quantize()
        assert pipe._last_intervention == "quantization"

        with patch("circuitkit.quick.export_checkpoint", return_value="/out") as exp:
            pipe.export("/out")
        # export() must route through the quantization branch by default now.
        assert exp.call_args.kwargs["intervention"] == "quantization"
        # And the quantization branch passes no pruning artifact.
        assert exp.call_args.args[1] is None

    def test_prune_records_intervention(self, pipe):
        with (
            patch.object(Pipeline, "_ensure_model", return_value=MagicMock()),
            patch("circuitkit.quick.prune", return_value=MagicMock()),
        ):
            pipe.prune()
        assert pipe._last_intervention == "pruning"

        with patch("circuitkit.quick.export_checkpoint", return_value="/out") as exp:
            pipe.export("/out")
        assert exp.call_args.kwargs["intervention"] == "pruning"

    def test_explicit_intervention_still_wins(self, pipe):
        fake_hf_model = MagicMock()
        with (
            patch.object(Pipeline, "_ensure_hf_model", return_value=fake_hf_model),
            patch("circuitkit.quick.quantize", return_value={}),
        ):
            pipe.quantize()
        with patch("circuitkit.quick.export_checkpoint", return_value="/out") as exp:
            pipe.export("/out", intervention="pruning")
        assert exp.call_args.kwargs["intervention"] == "pruning"

    def test_no_intervention_recorded_falls_back_to_pruning(self, pipe):
        pipe._pruned_model = MagicMock()  # simulate legacy state without a tag
        with patch("circuitkit.quick.export_checkpoint", return_value="/out") as exp:
            pipe.export("/out")
        assert exp.call_args.kwargs["intervention"] == "pruning"


# ---------------------------------------------------------------------------
# 2. quantize(release_original=True) frees the TL model
# ---------------------------------------------------------------------------


class TestQuantizeReleaseOriginal:
    def test_release_original_drops_tl_model(self, pipe):
        pipe._model = MagicMock()  # the discovery-time HookedTransformer
        with (
            patch.object(Pipeline, "_ensure_hf_model", return_value=MagicMock()),
            patch("circuitkit.quick.quantize", return_value={}),
        ):
            pipe.quantize(release_original=True)
        assert pipe._model is None

    def test_default_keeps_tl_model(self, pipe):
        tl = MagicMock()
        pipe._model = tl
        with (
            patch.object(Pipeline, "_ensure_hf_model", return_value=MagicMock()),
            patch("circuitkit.quick.quantize", return_value={}),
        ):
            pipe.quantize()
        assert pipe._model is tl


# ---------------------------------------------------------------------------
# 3. Actionable ImportError when optimum-quanto is missing
# ---------------------------------------------------------------------------


def _hide_optimum():
    """Force `import optimum.quanto` to fail even if the extra is installed."""
    return patch.dict(sys.modules, {"optimum": None, "optimum.quanto": None})


class TestQuantoImportGuard:
    def test_circuit_quantize_default_raises_actionable_importerror(self):
        from circuitkit.applications.quantization.quant_utils import circuit_quantize

        with _hide_optimum():
            with pytest.raises(ImportError, match=r"circuitkit\[quantization\]"):
                circuit_quantize(
                    model=MagicMock(),
                    q_head_scores={},
                    mlp_scores={},
                    n_layers=2,
                )

    def test_random_quantize_default_raises_actionable_importerror(self):
        from circuitkit.applications.quantization.quant_utils import random_quantize

        with _hide_optimum():
            with pytest.raises(ImportError, match=r"circuitkit\[quantization\]"):
                random_quantize(model=MagicMock(), n_layers=2)

    def test_default_resolves_to_qint4_when_available(self):
        """With the extra installed, low_weights=None resolves to qint4 and
        control proceeds past the import to architecture detection (proving
        the default no longer short-circuits into the old `return {}`)."""
        pytest.importorskip("optimum.quanto")
        from circuitkit.applications.quantization.quant_utils import circuit_quantize

        captured = {}

        def fake_detect(model):
            captured["reached"] = True
            raise RuntimeError("stop after default resolution")

        # circuit_quantize does a function-local `from circuitkit.applications
        # import detect_model_architecture`, so patch it on that module:
        with patch(
            "circuitkit.applications.detect_model_architecture",
            side_effect=fake_detect,
        ):
            with pytest.raises(RuntimeError, match="stop after default resolution"):
                circuit_quantize(
                    model=MagicMock(),
                    q_head_scores={},
                    mlp_scores={},
                    n_layers=2,
                )
        assert captured.get("reached"), "default resolution should proceed past the import"
