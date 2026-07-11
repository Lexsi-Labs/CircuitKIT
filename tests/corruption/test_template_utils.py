"""Comprehensive tests for data.corruption.template_utils.

Covers every public function: parse_placeholders, validate_placeholders_against_columns,
detect_peer_columns, resolve_template, build_pair_from_templates.
"""
import pytest
from circuitkit.data.corruption.template_utils import (
    parse_placeholders,
    validate_placeholders_against_columns,
    detect_peer_columns,
    resolve_template,
    build_pair_from_templates,
)


# ────────────────────────────────────────────────────────────────────────────
# parse_placeholders
# ────────────────────────────────────────────────────────────────────────────
class TestParsePlaceholders:
    def test_single_placeholder(self):
        assert parse_placeholders("{name}") == ["name"]

    def test_multiple_distinct(self):
        assert parse_placeholders("The {country} capital is {city}") == ["city", "country"]

    def test_empty_string(self):
        assert parse_placeholders("") == []

    def test_no_placeholders(self):
        assert parse_placeholders("No placeholders here") == []

    def test_escaped_double_braces_ignored(self):
        result = parse_placeholders("Use {{literal}} and {real}")
        assert result == ["real"]

    def test_duplicates_deduplicated(self):
        assert parse_placeholders("{x} and {x} again") == ["x"]

    def test_returns_sorted(self):
        result = parse_placeholders("{zebra} {apple} {mango}")
        assert result == ["apple", "mango", "zebra"]

    def test_underscored_names(self):
        result = parse_placeholders("{other_country} vs {base_name}")
        assert result == ["base_name", "other_country"]

    def test_adjacent_placeholders(self):
        result = parse_placeholders("{a}{b}{c}")
        assert result == ["a", "b", "c"]

    def test_numeric_in_name(self):
        # Template regex matches [^{}]+, so digits are valid
        result = parse_placeholders("{field1} {field2}")
        assert result == ["field1", "field2"]

    def test_nested_braces_not_matched(self):
        # At outer "{": [^{}]+ matches "outer" but hits "{" before "}" — no match.
        # At inner "{": [^{}]+ matches "inner", "}" found but lookahead sees next "}" — rejected.
        result = parse_placeholders("{outer{inner}}")
        assert result == []


# ────────────────────────────────────────────────────────────────────────────
# validate_placeholders_against_columns
# ────────────────────────────────────────────────────────────────────────────
class TestValidatePlaceholdersAgainstColumns:
    SPEC = {
        "clean_prompt": "The {country} capital",
        "corrupt_prompt": "The {other_country} capital",
        "clean_answer": "{capital}",
        "corrupt_answer": "{other_capital}",
    }

    def test_all_present_returns_empty(self):
        cols = ["country", "other_country", "capital", "other_capital"]
        assert validate_placeholders_against_columns(self.SPEC, cols) == []

    def test_missing_returns_sorted_list(self):
        cols = ["country", "capital"]
        missing = validate_placeholders_against_columns(self.SPEC, cols)
        assert missing == ["other_capital", "other_country"]

    def test_empty_spec_values(self):
        spec = {"clean_prompt": "", "corrupt_prompt": "", "clean_answer": "", "corrupt_answer": ""}
        assert validate_placeholders_against_columns(spec, []) == []

    def test_extra_columns_no_harm(self):
        cols = ["country", "other_country", "capital", "other_capital", "extra1", "extra2"]
        assert validate_placeholders_against_columns(self.SPEC, cols) == []

    def test_partial_spec_keys(self):
        # Only some spec keys have placeholders
        spec = {"clean_prompt": "{name}", "corrupt_prompt": "static", "clean_answer": "", "corrupt_answer": ""}
        assert validate_placeholders_against_columns(spec, ["name"]) == []
        assert validate_placeholders_against_columns(spec, []) == ["name"]

    def test_shared_placeholder_across_templates(self):
        # Same placeholder in multiple templates counted once
        spec = {
            "clean_prompt": "{x}",
            "corrupt_prompt": "{x}",
            "clean_answer": "{x}",
            "corrupt_answer": "{y}",
        }
        assert validate_placeholders_against_columns(spec, ["x", "y"]) == []
        assert validate_placeholders_against_columns(spec, ["x"]) == ["y"]


# ────────────────────────────────────────────────────────────────────────────
# detect_peer_columns
# ────────────────────────────────────────────────────────────────────────────
class TestDetectPeerColumns:
    def test_basic_peer_detection(self):
        peer_map, direct = detect_peer_columns(["country", "other_country", "capital", "other_capital"])
        assert peer_map == {"other_country": "country", "other_capital": "capital"}
        assert set(direct) == {"country", "capital"}

    def test_no_peers(self):
        peer_map, direct = detect_peer_columns(["country", "capital", "city"])
        assert peer_map == {}
        assert set(direct) == {"country", "capital", "city"}

    def test_orphaned_peer_raises(self):
        with pytest.raises(ValueError, match="other_missing"):
            detect_peer_columns(["other_missing"])

    def test_orphaned_peer_with_partial_direct(self):
        # other_x exists but x doesn't — raises even though other direct columns exist
        with pytest.raises(ValueError, match="other_ghost"):
            detect_peer_columns(["country", "other_ghost"])

    def test_empty_input(self):
        peer_map, direct = detect_peer_columns([])
        assert peer_map == {}
        assert direct == []

    def test_peer_map_values_point_to_base_names(self):
        peer_map, _ = detect_peer_columns(["name", "other_name"])
        assert peer_map["other_name"] == "name"

    def test_multiple_peers(self):
        placeholders = ["a", "other_a", "b", "other_b", "c"]
        peer_map, direct = detect_peer_columns(placeholders)
        assert len(peer_map) == 2
        assert peer_map["other_a"] == "a"
        assert peer_map["other_b"] == "b"
        assert set(direct) == {"a", "b", "c"}


# ────────────────────────────────────────────────────────────────────────────
# resolve_template
# ────────────────────────────────────────────────────────────────────────────
class TestResolveTemplate:
    def test_simple_substitution(self):
        assert resolve_template("{greeting} {name}", {"greeting": "Hello", "name": "World"}) == "Hello World"

    def test_no_placeholders(self):
        assert resolve_template("static text", {}) == "static text"

    def test_missing_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="name"):
            resolve_template("Hello {name}", {})

    def test_keyerror_message_lists_available_keys(self):
        with pytest.raises(KeyError, match="Available keys"):
            resolve_template("{missing}", {"present": "val"})

    def test_escaped_braces_preserved(self):
        # {{literal}} should not be substituted
        result = resolve_template("{{not_a_var}} and {real}", {"real": "yes"})
        assert "yes" in result
        # The {{...}} sequences remain (regex doesn't match them)
        assert "{{not_a_var}}" in result

    def test_value_with_special_characters(self):
        result = resolve_template("{x}", {"x": "value with {braces} and $pecial"})
        assert result == "value with {braces} and $pecial"

    def test_empty_value_substitution(self):
        assert resolve_template("{x}", {"x": ""}) == ""

    def test_multiple_same_placeholder(self):
        result = resolve_template("{x} and {x}", {"x": "val"})
        assert result == "val and val"


# ────────────────────────────────────────────────────────────────────────────
# build_pair_from_templates
# ────────────────────────────────────────────────────────────────────────────
class TestBuildPairFromTemplates:
    SPEC = {
        "clean_prompt": "The capital of {country} is",
        "corrupt_prompt": "The capital of {other_country} is",
        "clean_answer": "{capital}",
        "corrupt_answer": "{other_capital}",
    }

    def test_explicit_mode_same_row(self):
        row = {"country": "France", "other_country": "Germany", "capital": "Paris", "other_capital": "Berlin"}
        cp, crp, ca, cra = build_pair_from_templates(self.SPEC, row, row)
        assert cp == "The capital of France is"
        assert crp == "The capital of Germany is"
        assert ca == "Paris"
        assert cra == "Berlin"

    def test_auto_peer_different_dicts(self):
        clean = {"country": "France", "capital": "Paris"}
        corrupt = {"other_country": "Germany", "other_capital": "Berlin"}
        cp, crp, ca, cra = build_pair_from_templates(self.SPEC, clean, corrupt)
        assert cp == "The capital of France is"
        assert crp == "The capital of Germany is"
        assert ca == "Paris"
        assert cra == "Berlin"

    def test_missing_key_in_clean_values_raises(self):
        with pytest.raises(KeyError):
            build_pair_from_templates(self.SPEC, {}, {"other_country": "X", "other_capital": "Y"})

    def test_missing_key_in_corrupt_values_raises(self):
        with pytest.raises(KeyError):
            build_pair_from_templates(self.SPEC, {"country": "France", "capital": "Paris"}, {})

    def test_return_order_is_clean_corrupt_clean_corrupt(self):
        row = {"country": "A", "other_country": "B", "capital": "C", "other_capital": "D"}
        result = build_pair_from_templates(self.SPEC, row, row)
        assert result == ("The capital of A is", "The capital of B is", "C", "D")

    def test_templates_with_no_placeholders(self):
        static_spec = {
            "clean_prompt": "Hello",
            "corrupt_prompt": "Goodbye",
            "clean_answer": "yes",
            "corrupt_answer": "no",
        }
        cp, crp, ca, cra = build_pair_from_templates(static_spec, {}, {})
        assert (cp, crp, ca, cra) == ("Hello", "Goodbye", "yes", "no")

    def test_unicode_values(self):
        spec = {"clean_prompt": "{city}", "corrupt_prompt": "{city}", "clean_answer": "{a}", "corrupt_answer": "{a}"}
        vals = {"city": "東京", "a": "こんにちは"}
        cp, crp, ca, cra = build_pair_from_templates(spec, vals, vals)
        assert cp == "東京"
        assert ca == "こんにちは"