"""End-to-end integration tests for the template data pipeline.

Exercises the full flow: CSV → template_normalize → NormalizedDataset
→ save/load roundtrip → NormalizedTaskSpec → worthiness checks.
Also tests template_normalize with list-of-dicts and DataFrame inputs,
max_records capping, auto_peer mode, and various error conditions.
"""
import random

import pytest

TEMPLATE_SPEC = {
    "clean_prompt": "The capital of {country} is",
    "corrupt_prompt": "The capital of {other_country} is",
    "clean_answer": "{capital}",
    "corrupt_answer": "{other_capital}",
}

CSV_CONTENT = """\
country,other_country,capital,other_capital
France,Germany,Paris,Berlin
Japan,China,Tokyo,Beijing
Italy,Spain,Rome,Madrid
Brazil,Argentina,Brasilia,Buenos Aires
India,Pakistan,New Delhi,Islamabad
"""

# Auto-peer CSV: only direct columns — no other_* columns
AUTO_PEER_CSV = """\
country,capital
France,Paris
Japan,Tokyo
Italy,Rome
Brazil,Brasilia
India,New Delhi
"""

# Template spec for auto_peer (uses other_* in templates but not in CSV columns)
AUTO_PEER_SPEC = {
    "clean_prompt": "The capital of {country} is",
    "corrupt_prompt": "The capital of {other_country} is",
    "clean_answer": "{capital}",
    "corrupt_answer": "{other_capital}",
}


@pytest.fixture
def csv_file(tmp_path):
    p = tmp_path / "capitals.csv"
    p.write_text(CSV_CONTENT)
    return str(p)


@pytest.fixture
def auto_peer_csv(tmp_path):
    p = tmp_path / "capitals_auto.csv"
    p.write_text(AUTO_PEER_CSV)
    return str(p)


# ────────────────────────────────────────────────────────────────────────────
# CSV → NormalizedDataset → JSON roundtrip
# ────────────────────────────────────────────────────────────────────────────
class TestCsvToNormalizedJson:
    def test_roundtrip_preserves_all_fields(self, csv_file, tmp_path):
        from circuitkit.data.template import template_normalize
        from circuitkit.data.normalized import NormalizedDataset

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC, name="capitals")
        out_path = str(tmp_path / "out.json")
        ds.save_json(out_path)

        loaded = NormalizedDataset.load_json(out_path)
        assert len(loaded) == len(ds)
        assert loaded.shape == ds.shape
        assert loaded.name == ds.name
        for orig, loaded_r in zip(ds.records, loaded.records):
            assert loaded_r.clean_prompt == orig.clean_prompt
            assert loaded_r.corrupt_prompt == orig.corrupt_prompt
            assert loaded_r.clean_answer == orig.clean_answer
            assert loaded_r.corrupt_answer == orig.corrupt_answer
            assert loaded_r.record_id == orig.record_id

    def test_shape_is_template(self, csv_file):
        from circuitkit.data.template import template_normalize
        from circuitkit.data.normalized import DatasetShape

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        assert ds.shape == DatasetShape.TEMPLATE

    def test_record_count_matches_csv_rows(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        assert len(ds) == 5

    def test_all_records_fully_paired(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        for r in ds.records:
            assert r.clean_prompt is not None
            assert r.corrupt_prompt is not None
            assert r.clean_answer is not None
            assert r.corrupt_answer is not None

    def test_contrast_source_is_generated(self, csv_file):
        from circuitkit.data.template import template_normalize
        from circuitkit.data.normalized import ContrastSource

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        for r in ds.records:
            assert r.contrast_source == ContrastSource.GENERATED

    def test_meta_contains_template_values(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        for r in ds.records:
            assert "_template_values" in r.meta
            assert "_corruption" in r.meta
            assert r.meta["_corruption"] == "template"

    def test_dataset_meta_contains_spec(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        assert ds.meta.get("template_spec") == TEMPLATE_SPEC
        assert ds.meta.get("pairing_mode") == "explicit"


# ────────────────────────────────────────────────────────────────────────────
# Input formats: list-of-dicts and DataFrame
# ────────────────────────────────────────────────────────────────────────────
class TestInputFormats:
    def test_list_of_dicts(self):
        from circuitkit.data.template import template_normalize

        rows = [
            {"country": "France", "other_country": "Germany", "capital": "Paris", "other_capital": "Berlin"},
            {"country": "Japan", "other_country": "China", "capital": "Tokyo", "other_capital": "Beijing"},
        ]
        ds = template_normalize(rows, template_spec=TEMPLATE_SPEC)
        assert len(ds) == 2
        assert ds.records[0].clean_prompt == "The capital of France is"
        assert ds.records[0].corrupt_prompt == "The capital of Germany is"

    def test_pandas_dataframe(self):
        import pandas as pd
        from circuitkit.data.template import template_normalize

        df = pd.DataFrame([
            {"country": "France", "other_country": "Germany", "capital": "Paris", "other_capital": "Berlin"},
        ])
        ds = template_normalize(df, template_spec=TEMPLATE_SPEC)
        assert len(ds) == 1
        assert ds.records[0].corrupt_answer == "Berlin"

    def test_unsupported_type_raises(self):
        from circuitkit.data.template import template_normalize

        with pytest.raises(TypeError, match="Unsupported raw type"):
            template_normalize(12345, template_spec=TEMPLATE_SPEC)

    def test_empty_list_raises(self):
        from circuitkit.data.template import template_normalize

        with pytest.raises(ValueError, match="No rows"):
            template_normalize([], template_spec=TEMPLATE_SPEC)


# ────────────────────────────────────────────────────────────────────────────
# max_records
# ────────────────────────────────────────────────────────────────────────────
class TestMaxRecords:
    def test_caps_output(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC, max_records=2)
        assert len(ds) == 2

    def test_max_records_larger_than_csv_uses_all(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC, max_records=999)
        assert len(ds) == 5

    def test_max_records_none_uses_all(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC, max_records=None)
        assert len(ds) == 5


# ────────────────────────────────────────────────────────────────────────────
# auto_peer mode via template_normalize
# ────────────────────────────────────────────────────────────────────────────
class TestAutoPeerNormalize:
    def test_auto_peer_produces_paired_records(self, auto_peer_csv):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(
            auto_peer_csv, template_spec=AUTO_PEER_SPEC, pairing_mode="auto_peer"
        )
        assert len(ds) == 5
        for r in ds.records:
            assert r.clean_prompt is not None
            assert r.corrupt_prompt is not None
            assert r.clean_answer is not None
            assert r.corrupt_answer is not None

    def test_auto_peer_corrupt_differs_from_clean(self, auto_peer_csv):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(
            auto_peer_csv, template_spec=AUTO_PEER_SPEC, pairing_mode="auto_peer"
        )
        # With 5 distinct countries, at least some pairs should differ
        any_differ = any(r.clean_prompt != r.corrupt_prompt for r in ds.records)
        assert any_differ, "Expected at least one pair where clean != corrupt"

    def test_explicit_mode_with_auto_peer_csv_raises(self, auto_peer_csv):
        """Explicit mode requires other_* columns in CSV — should fail validation."""
        from circuitkit.data.template import template_normalize

        with pytest.raises(ValueError, match="not found in CSV columns"):
            template_normalize(
                auto_peer_csv, template_spec=AUTO_PEER_SPEC, pairing_mode="explicit"
            )


# ────────────────────────────────────────────────────────────────────────────
# Practical: verify actual prompt content
# ────────────────────────────────────────────────────────────────────────────
class TestPromptContent:
    def test_first_record_france_germany(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        r = ds.records[0]
        assert r.clean_prompt == "The capital of France is"
        assert r.corrupt_prompt == "The capital of Germany is"
        assert r.clean_answer == "Paris"
        assert r.corrupt_answer == "Berlin"

    def test_last_record_india_pakistan(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        r = ds.records[4]
        assert r.clean_prompt == "The capital of India is"
        assert r.corrupt_prompt == "The capital of Pakistan is"
        assert r.clean_answer == "New Delhi"
        assert r.corrupt_answer == "Islamabad"

    def test_record_ids_are_zero_padded(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        assert ds.records[0].record_id == "000000"
        assert ds.records[4].record_id == "000004"


# ────────────────────────────────────────────────────────────────────────────
# NormalizedTaskSpec integration
# ────────────────────────────────────────────────────────────────────────────
class TestTemplateToTaskSpec:
    def test_fully_paired_dataset_accepted(self, csv_file):
        from circuitkit.data.template import template_normalize
        from circuitkit.data.normalized_task import NormalizedTaskSpec

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        assert ds.fully_paired
        spec = NormalizedTaskSpec(ds, name="_ephemeral:test:5")
        assert spec is not None

    def test_unpaired_dataset_rejected_for_eap(self):
        """Unpaired data is accepted at construction (it's valid for CD-T /
        IBCircuit, which use only the clean prompt). It's rejected lazily at
        build_dataloader time, and only when an algorithm that needs pairing
        (EAP family) is requested."""
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        from circuitkit.data.normalized_task import NormalizedTaskSpec

        record = ContrastiveRecord(
            record_id="r0",
            clean_prompt="Hello",
            corrupt_prompt=None,
            clean_answer="world",
            corrupt_answer=None,
            contrast_source=ContrastSource.GENERATED,
            meta={},
        )
        ds = NormalizedDataset(
            name="bad", shape=DatasetShape.TEMPLATE,
            records=[record], source="test", meta={},
        )

        # Construction succeeds — unpaired data is not rejected here.
        spec = NormalizedTaskSpec(ds, name="_ephemeral:bad:1")

        # Requesting an EAP-family algorithm rejects the unpaired data. The
        # fully-paired guard fires before the model is used, so a sentinel
        # non-None model is enough to reach it.
        with pytest.raises(ValueError, match="fully-paired"):
            spec.build_dataloader(object(), {"algorithm": "eap", "batch_size": 1}, "cpu")


# ────────────────────────────────────────────────────────────────────────────
# Worthiness checks
# ────────────────────────────────────────────────────────────────────────────
class TestWorthiness:
    def test_green_verdict_on_well_formed_data(self, csv_file):
        from circuitkit.data.template import template_normalize
        from circuitkit.data.worthiness import evaluate_worthiness

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        report = evaluate_worthiness(ds)
        shape_check = next((c for c in report.checks if c.name == "shape_specific"), None)
        assert shape_check is not None
        assert shape_check.passed

    def test_red_unresolved_placeholders(self):
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        from circuitkit.data.worthiness import evaluate_worthiness

        record = ContrastiveRecord(
            record_id="r0",
            clean_prompt="The capital of {country} is",  # unresolved placeholder
            corrupt_prompt="The capital of Germany is",
            clean_answer="Paris",
            corrupt_answer="Berlin",
            contrast_source=ContrastSource.GENERATED,
            meta={"_corruption": "template"},
        )
        ds = NormalizedDataset(
            name="bad", shape=DatasetShape.TEMPLATE,
            records=[record], source="test", meta={},
        )
        report = evaluate_worthiness(ds)
        shape_check = next((c for c in report.checks if c.name == "shape_specific"), None)
        assert shape_check is not None
        assert not shape_check.passed
        assert "unresolved" in shape_check.message

    def test_red_identical_clean_corrupt_prompts(self):
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        from circuitkit.data.worthiness import evaluate_worthiness

        record = ContrastiveRecord(
            record_id="r0",
            clean_prompt="The capital of France is",
            corrupt_prompt="The capital of France is",  # identical to clean
            clean_answer="Paris",
            corrupt_answer="Berlin",
            contrast_source=ContrastSource.GENERATED,
            meta={"_corruption": "template"},
        )
        ds = NormalizedDataset(
            name="ident", shape=DatasetShape.TEMPLATE,
            records=[record], source="test", meta={},
        )
        report = evaluate_worthiness(ds)
        shape_check = next((c for c in report.checks if c.name == "shape_specific"), None)
        assert shape_check is not None
        assert not shape_check.passed
        assert "identical" in shape_check.message

    def test_red_empty_answers(self):
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        from circuitkit.data.worthiness import evaluate_worthiness

        record = ContrastiveRecord(
            record_id="r0",
            clean_prompt="Prompt A",
            corrupt_prompt="Prompt B",
            clean_answer="",
            corrupt_answer="",
            contrast_source=ContrastSource.GENERATED,
            meta={"_corruption": "template"},
        )
        ds = NormalizedDataset(
            name="empty", shape=DatasetShape.TEMPLATE,
            records=[record], source="test", meta={},
        )
        report = evaluate_worthiness(ds)
        shape_check = next((c for c in report.checks if c.name == "shape_specific"), None)
        assert shape_check is not None
        assert not shape_check.passed
        assert "empty" in shape_check.message

    def test_multiple_issues_all_reported(self):
        """A record with both unresolved placeholders AND identical prompts."""
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        from circuitkit.data.worthiness import evaluate_worthiness

        record = ContrastiveRecord(
            record_id="r0",
            clean_prompt="{unresolved}",
            corrupt_prompt="{unresolved}",
            clean_answer="",
            corrupt_answer="",
            contrast_source=ContrastSource.GENERATED,
            meta={"_corruption": "template"},
        )
        ds = NormalizedDataset(
            name="multi", shape=DatasetShape.TEMPLATE,
            records=[record], source="test", meta={},
        )
        report = evaluate_worthiness(ds)
        shape_check = next((c for c in report.checks if c.name == "shape_specific"), None)
        assert shape_check is not None
        assert not shape_check.passed
        # Should report multiple issues in the message (separated by ';')
        assert shape_check.message.count(";") >= 1


# ────────────────────────────────────────────────────────────────────────────
# Edge cases: special characters, whitespace, large datasets
# ────────────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_values_with_commas_in_csv(self, tmp_path):
        """CSV values containing commas (quoted) should work."""
        from circuitkit.data.template import template_normalize

        csv = 'country,other_country,capital,other_capital\n"South Korea","North Korea","Seoul","Pyongyang"\n'
        p = tmp_path / "comma.csv"
        p.write_text(csv)
        ds = template_normalize(str(p), template_spec=TEMPLATE_SPEC)
        assert ds.records[0].clean_prompt == "The capital of South Korea is"
        assert ds.records[0].corrupt_prompt == "The capital of North Korea is"

    def test_unicode_csv(self, tmp_path):
        from circuitkit.data.template import template_normalize

        csv = "country,other_country,capital,other_capital\n日本,中国,東京,北京\n"
        p = tmp_path / "unicode.csv"
        p.write_text(csv, encoding="utf-8")
        ds = template_normalize(str(p), template_spec=TEMPLATE_SPEC)
        assert ds.records[0].clean_answer == "東京"
        assert ds.records[0].corrupt_answer == "北京"

    def test_numeric_column_values_cast_to_str(self, tmp_path):
        """Numeric CSV values should be stringified in template resolution."""
        from circuitkit.data.template import template_normalize

        spec = {
            "clean_prompt": "Number {n} is",
            "corrupt_prompt": "Number {other_n} is",
            "clean_answer": "{label}",
            "corrupt_answer": "{other_label}",
        }
        csv = "n,other_n,label,other_label\n42,99,even,odd\n"
        p = tmp_path / "nums.csv"
        p.write_text(csv)
        ds = template_normalize(str(p), template_spec=spec)
        # pandas reads 42 as int; template_normalize should stringify
        assert "42" in ds.records[0].clean_prompt
        assert "99" in ds.records[0].corrupt_prompt

    def test_large_dataset(self, tmp_path):
        """Smoke test with 500 rows — no errors, correct count."""
        from circuitkit.data.template import template_normalize

        header = "country,other_country,capital,other_capital\n"
        rows = "".join(
            f"Country{i},Country{i+1},Cap{i},Cap{i+1}\n" for i in range(500)
        )
        p = tmp_path / "large.csv"
        p.write_text(header + rows)
        ds = template_normalize(str(p), template_spec=TEMPLATE_SPEC)
        assert len(ds) == 500

    def test_max_records_on_large_dataset(self, tmp_path):
        from circuitkit.data.template import template_normalize

        header = "country,other_country,capital,other_capital\n"
        rows = "".join(f"C{i},C{i+1},Cap{i},Cap{i+1}\n" for i in range(100))
        p = tmp_path / "large2.csv"
        p.write_text(header + rows)
        ds = template_normalize(str(p), template_spec=TEMPLATE_SPEC, max_records=10)
        assert len(ds) == 10

    def test_name_and_source_defaults(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(csv_file, template_spec=TEMPLATE_SPEC)
        assert ds.name == "template"  # default when name=None
        assert ds.source == csv_file  # str path used as source

    def test_custom_name_and_source(self, csv_file):
        from circuitkit.data.template import template_normalize

        ds = template_normalize(
            csv_file, template_spec=TEMPLATE_SPEC, name="my_ds", source="custom_src"
        )
        assert ds.name == "my_ds"
        assert ds.source == "custom_src"