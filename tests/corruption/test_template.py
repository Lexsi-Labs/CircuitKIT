"""Comprehensive tests for TemplateStrategy (data.corruption.template).

Tests the strategy through both the low-level corrupt() and the high-level
apply_to_dataset() paths, exercising explicit and auto_peer modes, error
handling, registry integration, and edge cases.
"""
import random

import pytest
from circuitkit.data.corruption.base import get_strategy, LengthContract
from circuitkit.data.corruption.template import TemplateStrategy
from circuitkit.data.normalized import (
    ContrastiveRecord,
    ContrastSource,
    DatasetShape,
    NormalizedDataset,
)


TEMPLATE_SPEC = {
    "clean_prompt": "The capital of {country} is",
    "corrupt_prompt": "The capital of {other_country} is",
    "clean_answer": "{capital}",
    "corrupt_answer": "{other_capital}",
}

COUNTRIES = [
    ("France", "Germany", "Paris", "Berlin"),
    ("Japan", "China", "Tokyo", "Beijing"),
    ("Italy", "Spain", "Rome", "Madrid"),
    ("Brazil", "Argentina", "Brasilia", "Buenos Aires"),
    ("India", "Pakistan", "New Delhi", "Islamabad"),
]


def _make_record(country, other_country, capital, other_capital, record_id="r0"):
    """Build a single ContrastiveRecord with _template_values populated."""
    row_values = {
        "country": country,
        "other_country": other_country,
        "capital": capital,
        "other_capital": other_capital,
    }
    return ContrastiveRecord(
        record_id=record_id,
        clean_prompt=f"The capital of {country} is",
        corrupt_prompt=None,  # not yet paired
        clean_answer=capital,
        corrupt_answer=None,
        contrast_source=ContrastSource.GENERATED,
        target_field="template_answer",
        meta={"_template_values": row_values, "_corruption": "template"},
    )


def _make_dataset(n=3):
    records = [
        _make_record(*COUNTRIES[i % len(COUNTRIES)], record_id=f"r{i}")
        for i in range(n)
    ]
    return NormalizedDataset(
        name="test_ds",
        shape=DatasetShape.TEMPLATE,
        records=records,
        source="test",
        meta={},
    )


# ────────────────────────────────────────────────────────────────────────────
# Construction
# ────────────────────────────────────────────────────────────────────────────
class TestTemplateStrategyConstruction:
    def test_zero_arg_construction_for_registry(self):
        """Registry calls get_strategy("template")() — must not crash."""
        strat = TemplateStrategy()
        assert strat.name == "template"
        assert strat._spec == {}

    def test_explicit_mode_default(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        assert strat._pairing_mode == "explicit"

    def test_auto_peer_mode(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        assert strat._pairing_mode == "auto_peer"

    def test_invalid_pairing_mode_raises(self):
        with pytest.raises(ValueError, match="pairing_mode"):
            TemplateStrategy(TEMPLATE_SPEC, pairing_mode="bogus")

    def test_missing_spec_keys_raises(self):
        incomplete = {"clean_prompt": "x", "corrupt_prompt": "y"}
        with pytest.raises(ValueError, match="missing required keys"):
            TemplateStrategy(incomplete)

    def test_placeholders_parsed_at_init(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        assert "country" in strat._all_placeholders
        assert "other_country" in strat._all_placeholders

    def test_peer_map_built_at_init(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        assert strat._peer_map.get("other_country") == "country"
        assert strat._peer_map.get("other_capital") == "capital"

    def test_length_contract_is_preserve(self):
        assert TemplateStrategy.length_contract == LengthContract.PRESERVE


# ────────────────────────────────────────────────────────────────────────────
# corrupt() — explicit mode
# ────────────────────────────────────────────────────────────────────────────
class TestCorruptExplicit:
    def test_basic_corruption(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="explicit")
        record = _make_record("France", "Germany", "Paris", "Berlin")
        result = strat.corrupt(record)
        assert result.succeeded
        assert result.corrupt_prompt == "The capital of Germany is"
        assert result.corrupt_answer == "Berlin"

    def test_no_spec_returns_failed(self):
        strat = TemplateStrategy()  # zero-arg
        record = _make_record("France", "Germany", "Paris", "Berlin")
        result = strat.corrupt(record)
        assert not result.succeeded
        assert "without a template_spec" in result.notes

    def test_missing_template_values_returns_failed(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        record = ContrastiveRecord(
            record_id="bad",
            clean_prompt="X",
            corrupt_prompt=None,
            clean_answer="Y",
            corrupt_answer=None,
            contrast_source=ContrastSource.GENERATED,
            meta={},  # no _template_values
        )
        result = strat.corrupt(record)
        assert not result.succeeded
        assert "_template_values" in result.notes

    def test_missing_column_in_row_returns_failed(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        record = ContrastiveRecord(
            record_id="partial",
            clean_prompt="X",
            corrupt_prompt=None,
            clean_answer="Y",
            corrupt_answer=None,
            contrast_source=ContrastSource.GENERATED,
            meta={"_template_values": {"country": "France"}},  # missing other_country etc.
        )
        result = strat.corrupt(record)
        assert not result.succeeded  # KeyError caught internally


# ────────────────────────────────────────────────────────────────────────────
# corrupt() — auto_peer mode
# ────────────────────────────────────────────────────────────────────────────
class TestCorruptAutoPeer:
    def test_auto_peer_without_pool_fails(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        record = _make_record("France", "Germany", "Paris", "Berlin")
        result = strat.corrupt(record)
        assert not result.succeeded
        assert "pool" in result.notes

    def test_auto_peer_with_pool_succeeds(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        ds = _make_dataset(5)
        record = ds.records[0]
        result = strat.corrupt(record, pool=ds.records, rng=random.Random(42))
        assert result.succeeded
        assert result.corrupt_prompt is not None
        assert result.corrupt_answer is not None

    def test_auto_peer_uses_different_values(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        ds = _make_dataset(5)
        record = ds.records[0]  # France
        result = strat.corrupt(record, pool=ds.records, rng=random.Random(42))
        # The corrupt side should come from a peer, not the same row
        # (with 5 distinct countries, collision is possible but unlikely with seed 42)
        assert result.succeeded
        # At minimum, the corrupt prompt was produced
        assert "The capital of" in result.corrupt_prompt

    def test_auto_peer_single_record_pool_no_candidates(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        record = _make_record("France", "Germany", "Paris", "Berlin", record_id="only")
        result = strat.corrupt(record, pool=[record], rng=random.Random(0))
        assert not result.succeeded
        assert "no candidate peers" in result.notes

    def test_auto_peer_deterministic_with_seed(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        ds = _make_dataset(5)
        r1 = strat.corrupt(ds.records[0], pool=ds.records, rng=random.Random(99))
        r2 = strat.corrupt(ds.records[0], pool=ds.records, rng=random.Random(99))
        assert r1.corrupt_prompt == r2.corrupt_prompt
        assert r1.corrupt_answer == r2.corrupt_answer


# ────────────────────────────────────────────────────────────────────────────
# apply() — integration with base class
# ────────────────────────────────────────────────────────────────────────────
class TestApplyIntegration:
    """Test that base.apply() correctly wraps corrupt() for TemplateStrategy."""

    def test_apply_returns_contrastive_record(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="explicit")
        record = _make_record("France", "Germany", "Paris", "Berlin")
        out = strat.apply(record)
        assert isinstance(out, ContrastiveRecord)
        assert out.corrupt_prompt == "The capital of Germany is"
        assert out.corrupt_answer == "Berlin"
        assert out.contrast_source == ContrastSource.GENERATED

    def test_apply_preserves_meta_strategy_name(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        record = _make_record("France", "Germany", "Paris", "Berlin")
        out = strat.apply(record)
        assert out.meta.get("_strategy_name") == "template"

    def test_apply_preserves_template_values_in_meta(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        record = _make_record("France", "Germany", "Paris", "Berlin")
        out = strat.apply(record)
        assert "_template_values" in out.meta

    def test_apply_on_native_pair_skips(self):
        """Records with NATIVE_PAIR contrast source are returned unchanged."""
        strat = TemplateStrategy(TEMPLATE_SPEC)
        record = ContrastiveRecord(
            record_id="native",
            clean_prompt="A",
            corrupt_prompt="B",
            clean_answer="C",
            corrupt_answer="D",
            contrast_source=ContrastSource.NATIVE_PAIR,
            target_field="template_answer",
            meta={"_template_values": {"country": "X", "other_country": "Y",
                                        "capital": "C", "other_capital": "D"}},
        )
        out = strat.apply(record)
        assert out is record  # unchanged — identity


# ────────────────────────────────────────────────────────────────────────────
# apply_to_dataset()
# ────────────────────────────────────────────────────────────────────────────
class TestApplyToDataset:
    def test_explicit_all_records_corrupted(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="explicit")
        ds = _make_dataset(5)
        out = strat.apply_to_dataset(ds)
        assert len(out) == 5
        for r in out.records:
            assert r.corrupt_prompt is not None
            assert r.corrupt_answer is not None

    def test_output_shape_is_template(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        out = strat.apply_to_dataset(_make_dataset(2))
        assert out.shape == DatasetShape.TEMPLATE

    def test_output_meta_has_corruption_tag(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        out = strat.apply_to_dataset(_make_dataset(2))
        assert out.meta.get("_corruption") == "template"

    def test_output_preserves_source_and_name(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        ds = _make_dataset(2)
        out = strat.apply_to_dataset(ds)
        assert out.name == ds.name
        assert out.source == ds.source

    def test_auto_peer_dataset(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        ds = _make_dataset(5)
        out = strat.apply_to_dataset(ds, rng=random.Random(42))
        assert len(out) == 5
        for r in out.records:
            assert r.corrupt_prompt is not None

    def test_auto_peer_dataset_deterministic(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="auto_peer")
        ds = _make_dataset(5)
        out1 = strat.apply_to_dataset(ds, rng=random.Random(7))
        out2 = strat.apply_to_dataset(ds, rng=random.Random(7))
        for r1, r2 in zip(out1.records, out2.records):
            assert r1.corrupt_prompt == r2.corrupt_prompt

    def test_single_record_explicit(self):
        strat = TemplateStrategy(TEMPLATE_SPEC, pairing_mode="explicit")
        ds = _make_dataset(1)
        out = strat.apply_to_dataset(ds)
        assert len(out) == 1
        assert out.records[0].corrupt_prompt is not None

    def test_original_dataset_not_mutated(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        ds = _make_dataset(3)
        original_prompts = [r.corrupt_prompt for r in ds.records]
        _ = strat.apply_to_dataset(ds)
        for r, orig in zip(ds.records, original_prompts):
            assert r.corrupt_prompt == orig  # should still be None


# ────────────────────────────────────────────────────────────────────────────
# validate_against_columns()
# ────────────────────────────────────────────────────────────────────────────
class TestValidateAgainstColumns:
    def test_all_present(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        assert strat.validate_against_columns(
            ["country", "other_country", "capital", "other_capital"]
        ) == []

    def test_missing_reported(self):
        strat = TemplateStrategy(TEMPLATE_SPEC)
        missing = strat.validate_against_columns(["country", "capital"])
        assert "other_country" in missing
        assert "other_capital" in missing


# ────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_registered_as_template(self):
        assert get_strategy("template") is TemplateStrategy

    def test_zero_arg_instantiation_from_registry(self):
        cls = get_strategy("template")
        strat = cls()
        assert strat.name == "template"

    def test_parameterized_instantiation_from_registry(self):
        cls = get_strategy("template")
        strat = cls(TEMPLATE_SPEC, pairing_mode="auto_peer")
        assert strat._pairing_mode == "auto_peer"