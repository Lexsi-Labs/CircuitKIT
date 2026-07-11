"""
Unit tests for report.py — FaithfulnessReport dataclass.

Covers:
- Construction with defaults and explicit values
- JSON round-trip (to_json / from_json)
- _ReportEncoder handling of numpy types
- __repr__ formatting for all pillar combinations
- summary() method behavior (including None handling)
- _format_dict helper
- Error paths: missing file, bad JSON, missing required fields, invalid structure
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import under test — adjust the import path to match your project layout.
# If running with pytest from the repo root you may need:
#   PYTHONPATH=. pytest test_report.py
# ---------------------------------------------------------------------------
from circuitkit.evaluation.report import FaithfulnessReport, _ReportEncoder

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _tmp_path(suffix=".json"):
    """Return a temporary file path (caller is responsible for cleanup)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return Path(path)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_defaults(self):
        r = FaithfulnessReport()
        assert r.patching_score is None
        assert r.ablation_score is None
        assert r.stability is None
        assert r.robustness is None
        assert r.baseline_comparison is None
        assert r.generalization is None
        assert r.metadata == {}

    def test_explicit_values(self):
        r = FaithfulnessReport(
            patching_score=0.85,
            ablation_score=0.72,
            stability={"mean_jaccard": 0.6},
            robustness={"delta": 0.1},
            baseline_comparison={"random": 0.3},
            generalization={"transfer_ratio": 0.9},
            metadata={"model": "gpt2"},
        )
        assert r.patching_score == 0.85
        assert r.ablation_score == 0.72
        assert r.stability["mean_jaccard"] == 0.6
        assert r.metadata["model"] == "gpt2"

    def test_partial_scores(self):
        r = FaithfulnessReport(patching_score=0.5)
        assert r.patching_score == 0.5
        assert r.ablation_score is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. JSON Round-Trip
# ═══════════════════════════════════════════════════════════════════════════


class TestJsonRoundTrip:
    def test_minimal_round_trip(self):
        path = _tmp_path()
        try:
            r = FaithfulnessReport(patching_score=0.9, ablation_score=0.8)
            r.to_json(path)
            loaded = FaithfulnessReport.from_json(path)
            assert loaded.patching_score == pytest.approx(0.9)
            assert loaded.ablation_score == pytest.approx(0.8)
        finally:
            path.unlink(missing_ok=True)

    def test_full_round_trip(self):
        path = _tmp_path()
        try:
            r = FaithfulnessReport(
                patching_score=0.85,
                ablation_score=0.72,
                stability={"mean_jaccard": 0.65, "std_jaccard": 0.04},
                robustness={"paraphrase": {"delta": 0.05}},
                baseline_comparison={"random": {"score": 0.3}},
                generalization={"transfer_ratio": 0.78},
                metadata={"task": "ioi", "model": "gpt2", "sparsity": 0.3},
            )
            r.to_json(path)
            loaded = FaithfulnessReport.from_json(path)

            assert loaded.patching_score == pytest.approx(0.85)
            assert loaded.stability["mean_jaccard"] == pytest.approx(0.65)
            assert loaded.robustness["paraphrase"]["delta"] == pytest.approx(0.05)
            assert loaded.metadata["task"] == "ioi"
        finally:
            path.unlink(missing_ok=True)

    def test_none_scores_round_trip(self):
        """Pillars set to None should survive serialization."""
        path = _tmp_path()
        try:
            r = FaithfulnessReport(patching_score=None, ablation_score=None)
            r.to_json(path)
            loaded = FaithfulnessReport.from_json(path)
            assert loaded.patching_score is None
            assert loaded.ablation_score is None
        finally:
            path.unlink(missing_ok=True)

    def test_numpy_values_in_json(self):
        """numpy scalars and arrays should serialize without error."""
        path = _tmp_path()
        try:
            r = FaithfulnessReport(
                patching_score=0.5,
                ablation_score=0.6,
                stability={
                    "jaccard_matrix": np.array([[1.0, 0.5], [0.5, 1.0]]),
                    "mean_jaccard": np.float64(0.75),
                    "n_stable": np.int64(42),
                },
            )
            r.to_json(path)

            # Verify file is valid JSON
            with open(path) as f:
                data = json.load(f)

            # numpy array → list
            assert isinstance(data["stability"]["jaccard_matrix"], list)
            assert len(data["stability"]["jaccard_matrix"]) == 2

            # numpy scalar → Python native
            assert isinstance(data["stability"]["mean_jaccard"], float)
            assert isinstance(data["stability"]["n_stable"], int)
        finally:
            path.unlink(missing_ok=True)

    def test_creates_parent_directories(self):
        """to_json should create intermediate directories."""
        import tempfile

        # FIX: Use mkdtemp to create a base directory, not a file
        base = Path(tempfile.mkdtemp())
        nested = base / "sub" / "dir" / "report.json"
        try:
            r = FaithfulnessReport(patching_score=0.1, ablation_score=0.2)
            r.to_json(nested)
            assert nested.exists()
            loaded = FaithfulnessReport.from_json(nested)
            assert loaded.patching_score == pytest.approx(0.1)
        finally:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. JSON Error Paths
# ═══════════════════════════════════════════════════════════════════════════


class TestJsonErrors:
    def test_from_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            FaithfulnessReport.from_json(Path("/tmp/nonexistent_report_xyz.json"))

    def test_from_json_invalid_json(self):
        path = _tmp_path()
        try:
            path.write_text("NOT VALID JSON {{{")
            with pytest.raises(json.JSONDecodeError):
                FaithfulnessReport.from_json(path)
        finally:
            path.unlink(missing_ok=True)

    def test_from_json_missing_required_field(self):
        path = _tmp_path()
        try:
            # Missing ablation_score
            path.write_text(json.dumps({"patching_score": 0.5}))
            with pytest.raises(ValueError, match="ablation_score"):
                FaithfulnessReport.from_json(path)
        finally:
            path.unlink(missing_ok=True)

    def test_from_json_invalid_structure(self):
        path = _tmp_path()
        try:
            # Extra unknown field should raise TypeError → caught as ValueError
            path.write_text(
                json.dumps(
                    {
                        "patching_score": 0.5,
                        "ablation_score": 0.6,
                        "totally_unknown_field": 123,
                    }
                )
            )
            with pytest.raises(ValueError, match="Invalid report JSON"):
                FaithfulnessReport.from_json(path)
        finally:
            path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. _ReportEncoder
# ═══════════════════════════════════════════════════════════════════════════


class TestReportEncoder:
    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = json.loads(json.dumps(arr, cls=_ReportEncoder))
        assert result == [1, 2, 3]

    def test_numpy_int(self):
        val = np.int64(42)
        result = json.loads(json.dumps({"v": val}, cls=_ReportEncoder))
        assert result["v"] == 42
        assert isinstance(result["v"], int)

    def test_numpy_float(self):
        val = np.float32(3.14)
        result = json.loads(json.dumps({"v": val}, cls=_ReportEncoder))
        assert result["v"] == pytest.approx(3.14, abs=1e-5)

    def test_non_numpy_fallback(self):
        """Non-serializable objects should raise TypeError via super()."""
        with pytest.raises(TypeError):
            json.dumps({"v": object()}, cls=_ReportEncoder)


# ═══════════════════════════════════════════════════════════════════════════
# 5. __repr__
# ═══════════════════════════════════════════════════════════════════════════


class TestRepr:
    def test_repr_minimal(self):
        r = FaithfulnessReport(patching_score=0.9, ablation_score=0.8)
        text = repr(r)
        assert "PILLAR 1" in text
        assert "PILLAR 2" in text
        assert "0.9000" in text
        assert "0.8000" in text

    def test_repr_none_scores(self):
        r = FaithfulnessReport()
        text = repr(r)
        assert "N/A" in text

    def test_repr_all_pillars(self):
        r = FaithfulnessReport(
            patching_score=0.9,
            ablation_score=0.8,
            stability={"mean_jaccard": 0.7},
            robustness={"delta": 0.05},
            baseline_comparison={"random": 0.3},
            generalization={"transfer_ratio": 0.6},
            metadata={"task": "ioi"},
        )
        text = repr(r)
        for pillar_num in range(1, 7):
            assert f"PILLAR {pillar_num}" in text

    def test_repr_with_numpy_in_dicts(self):
        """Repr should handle numpy arrays/scalars without crashing."""
        r = FaithfulnessReport(
            patching_score=0.5,
            ablation_score=0.6,
            stability={
                "jaccard_matrix": np.array([[1.0, 0.5], [0.5, 1.0]]),
                "mean_jaccard": np.float64(0.75),
                "n_stable": np.int64(42),
                "nested": {"inner_list": [1, 2, 3]},
            },
        )
        text = repr(r)
        assert "ndarray" in text
        assert "3 items" in text


# ═══════════════════════════════════════════════════════════════════════════
# 6. summary()
# ═══════════════════════════════════════════════════════════════════════════


class TestSummary:
    def test_summary_with_both_scores(self):
        r = FaithfulnessReport(patching_score=0.8, ablation_score=0.6)
        s = r.summary()
        assert s["pillar_1_patching"] == 0.8
        assert s["pillar_2_ablation"] == 0.6
        assert s["overall_average"] == pytest.approx(0.7)

    def test_summary_with_none_scores(self):
        """summary() should gracefully handle None scores returning None for overall_average."""
        r = FaithfulnessReport()
        s = r.summary()
        assert s["pillar_1_patching"] is None
        assert s["pillar_2_ablation"] is None
        assert s["overall_average"] is None

    # FIX: Rename the test and assert that it gracefully handles a single None score
    def test_summary_one_none(self):
        """One None score should result in overall_average being None without errors."""
        r = FaithfulnessReport(patching_score=0.5)
        s = r.summary()
        assert s["pillar_1_patching"] == 0.5
        assert s["pillar_2_ablation"] is None
        assert s["overall_average"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. _format_dict edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatDict:
    def test_empty_dict(self):
        lines = FaithfulnessReport._format_dict({})
        assert lines == []

    def test_nested_dict(self):
        d = {"outer": {"inner_key": 0.5}}
        lines = FaithfulnessReport._format_dict(d)
        assert any("outer" in lyr for lyr in lines)
        assert any("inner_key" in lyr for lyr in lines)

    def test_various_types(self):
        d = {
            "float_val": 3.14,
            "int_val": 42,
            "str_val": "hello",
            "list_val": [1, 2, 3],
            "np_array": np.zeros(5),
            "np_float": np.float64(2.71),
            "np_int": np.int32(7),
        }
        lines = FaithfulnessReport._format_dict(d)
        assert len(lines) == 7
        # Check that numpy array shows shape info
        assert any("ndarray" in lyr for lyr in lines)
        # Check list shows item count
        assert any("3 items" in lyr for lyr in lines)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
