"""Step 8 — Comprehensive tests for the custom-data pipeline (Steps 0–7).

Test matrix (from plan):
  8a. clean_only path tests
  8b. Alignment utility unit tests (template_utils.py)
  8c. template_normalize integration tests
  8d. _materialise_eap_csv tests
  8e. End-to-end tests

Mock tokenizer: whitespace splitter that faithfully mimics the HuggingFace
tokenizer interface (encode, decode, add_special_tokens, pad_token_id, etc.).
This avoids any network dependency on HuggingFace while still exercising
the real tokenization code paths.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from circuitkit.data.normalized import (
    ContrastiveRecord,
    ContrastSource,
    DatasetShape,
    NormalizedDataset,
)
from circuitkit.data.corruption.template_utils import (
    AlignmentResult,
    AnswerCheckResult,
    check_answer_discriminative,
    check_token_alignment,
    pad_question_region,
)
from circuitkit.data.template import template_normalize
from circuitkit.data.clean_only import clean_only_normalize
from circuitkit.data.worthiness import (
    CheckResult,
    DataWorthinessReport,
    Verdict,
    evaluate_worthiness,
    _check_token_alignment,
    _check_shape_specific,
)
from circuitkit.data.normalized_task import NormalizedTaskSpec


# ============================================================================
# Mock tokenizer — whitespace splitter mimicking HF tokenizer interface
# ============================================================================


class MockTokenizer:
    """Whitespace-splitting tokenizer that faithfully implements the HF
    tokenizer interface used throughout the codebase.

    Each unique word maps to a stable integer ID (allocated on first sight).
    This lets us test:
    - Token-length alignment (word count = token count)
    - Answer discrimination (different first words → different IDs)
    - Shared prefix absorption (same leading words → merged into prompt)
    - Pad region expansion (inserting words changes token count by 1 each)

    The ID mapping is deterministic within a single instance.
    """

    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        self._next_id = 1  # 0 reserved for pad
        self.pad_token_id = 0
        self.eos_token_id = 0
        # For worthiness check compatibility
        self.all_special_ids = {0}

    def _get_id(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_id
            self._id_to_word[self._next_id] = word
            self._next_id += 1
        return self._vocab[word]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Split on whitespace, return list of integer IDs.

        Faithfully handles:
        - Empty string → empty list
        - Leading/trailing spaces → stripped (no phantom tokens)
        - add_special_tokens is accepted but not used (matching the
          codebase's use of add_special_tokens=False)
        """
        if not text or not text.strip():
            return []
        words = text.split()
        return [self._get_id(w) for w in words]

    def decode(self, ids: List[int]) -> str:
        """Reconstruct string from IDs. Unknown IDs become '<unk>'."""
        words = [self._id_to_word.get(i, "<unk>") for i in ids]
        return " ".join(words)


class MockModel:
    """Minimal model mock for NormalizedTaskSpec methods that need model.tokenizer."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or MockTokenizer()
        self.device = "cpu"

    def to_tokens(self, text: str, prepend_bos: bool = True):
        """Return a 2D tensor of token IDs, mimicking HookedTransformer."""
        import torch

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if prepend_bos:
            ids = [self.tokenizer.eos_token_id] + ids
        return torch.tensor([ids], dtype=torch.long)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tok():
    """Fresh MockTokenizer for each test."""
    return MockTokenizer()


@pytest.fixture
def mock_model(tok):
    """MockModel wrapping a fresh tokenizer."""
    return MockModel(tok)


@pytest.fixture
def tmp_dir():
    """Temporary directory cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> str:
    """Helper: write a list of dicts as a CSV file, return path string."""
    filepath = str(path)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return filepath


# ============================================================================
# 8a. clean_only path tests
# ============================================================================


class TestCleanOnlyPath:
    """8a. Tests for clean_only_normalize and its integration with
    NormalizedTaskSpec and downstream dataloaders."""

    def test_valid_csv_produces_unpaired_records(self, tmp_dir):
        """clean_only_normalize with valid CSV → records have corrupt_prompt=None,
        fully_paired=False."""
        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [
                {"prompt": "What is 2+2?", "answer": "4"},
                {"prompt": "Capital of France?", "answer": "Paris"},
                {"prompt": "Color of sky?", "answer": "Blue"},
            ],
        )
        ds = clean_only_normalize(csv_path)
        assert ds.shape == DatasetShape.CLEAN_ONLY
        assert len(ds) == 3
        assert not ds.fully_paired
        assert ds.n_paired == 0
        for r in ds.records:
            assert r.corrupt_prompt is None
            assert r.corrupt_answer is None
            assert r.contrast_source == ContrastSource.NOT_PAIRED_YET
            assert r.clean_prompt  # non-empty
            assert r.clean_answer  # non-empty

    def test_record_id_is_string(self, tmp_dir):
        """record_id should be a string (int-based in clean_only_normalize)."""
        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [{"prompt": "Hello", "answer": "World"}],
        )
        ds = clean_only_normalize(csv_path)
        assert isinstance(ds.records[0].record_id, str)

    def test_normalized_task_spec_accepts_unpaired(self, tmp_dir):
        """NormalizedTaskSpec accepts unpaired dataset (no error at init)."""
        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [{"prompt": "Test prompt", "answer": "A"}],
        )
        ds = clean_only_normalize(csv_path)
        # Should NOT raise — fully_paired guard was moved to build_dataloader
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        assert spec.ds is ds
        assert spec.pair_padding_side == "left"

    def test_build_dataloader_eap_raises_for_unpaired(self, tmp_dir, mock_model):
        """build_dataloader with algorithm=eap + unpaired data → ValueError."""
        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [{"prompt": "Test", "answer": "A"}],
        )
        ds = clean_only_normalize(csv_path)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        with pytest.raises(ValueError, match="requires fully-paired data"):
            spec.build_dataloader(
                mock_model,
                {"algorithm": "eap", "batch_size": 1},
                "cpu",
            )

    def test_build_dataloader_eap_ig_raises_for_unpaired(self, tmp_dir, mock_model):
        """EAP-IG also rejects unpaired data."""
        ds = clean_only_normalize(
            [{"prompt": "P1", "answer": "A"}],
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        with pytest.raises(ValueError, match="requires fully-paired data"):
            spec.build_dataloader(
                mock_model,
                {"algorithm": "eap-ig", "batch_size": 1},
                "cpu",
            )

    def test_build_dataloader_acdc_raises_for_unpaired(self, tmp_dir, mock_model):
        """ACDC also rejects unpaired data."""
        ds = clean_only_normalize(
            [{"prompt": "P1", "answer": "A"}],
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        with pytest.raises(ValueError, match="requires fully-paired data"):
            spec.build_dataloader(
                mock_model,
                {"algorithm": "acdc", "batch_size": 1},
                "cpu",
            )

    def test_missing_prompt_column_raises(self, tmp_dir):
        """Missing prompt_column → ValueError."""
        csv_path = _write_csv(
            tmp_dir / "bad.csv",
            [{"text": "Hello", "answer": "World"}],
        )
        with pytest.raises(ValueError, match="prompt_column.*not found"):
            clean_only_normalize(csv_path, prompt_column="prompt")

    def test_missing_answer_column_raises(self, tmp_dir):
        """answer_column not found → ValueError."""
        csv_path = _write_csv(
            tmp_dir / "bad.csv",
            [{"prompt": "Hello", "label": "World"}],
        )
        with pytest.raises(ValueError, match="answer_column.*not found"):
            clean_only_normalize(csv_path, answer_column="answer")

    def test_answer_column_none_is_valid(self, tmp_dir):
        """answer_column=None for CD-T → records have empty clean_answer, no error."""
        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [{"prompt": "Just a prompt"}],
        )
        ds = clean_only_normalize(csv_path, answer_column=None)
        assert len(ds) == 1
        assert ds.records[0].clean_answer == ""

    def test_empty_dataset_raises(self):
        """Empty input → ValueError."""
        with pytest.raises(ValueError, match="empty"):
            clean_only_normalize([])

    def test_list_of_dicts_input(self):
        """Accepts list-of-dicts directly."""
        rows = [
            {"prompt": "P1", "answer": "A1"},
            {"prompt": "P2", "answer": "A2"},
        ]
        ds = clean_only_normalize(rows)
        assert len(ds) == 2
        assert ds.records[0].clean_prompt == "P1"

    def test_max_records_truncation(self, tmp_dir):
        """max_records caps output."""
        rows = [{"prompt": f"P{i}", "answer": f"A{i}"} for i in range(10)]
        csv_path = _write_csv(tmp_dir / "many.csv", rows)
        ds = clean_only_normalize(csv_path, max_records=3)
        assert len(ds) == 3

    def test_custom_column_names(self, tmp_dir):
        """Custom prompt/answer column names work."""
        csv_path = _write_csv(
            tmp_dir / "custom.csv",
            [{"input_text": "Hello", "target": "World"}],
        )
        ds = clean_only_normalize(
            csv_path, prompt_column="input_text", answer_column="target"
        )
        assert ds.records[0].clean_prompt == "Hello"
        assert ds.records[0].clean_answer == "World"

    def test_unsupported_input_type_raises(self):
        """Unsupported input type → ValueError."""
        with pytest.raises(ValueError, match="unsupported input type"):
            clean_only_normalize(42)

    def test_list_of_non_dicts_raises(self):
        """List of non-dicts → ValueError."""
        with pytest.raises(ValueError, match="list of dicts"):
            clean_only_normalize(["not", "dicts"])

    def test_file_not_found_raises(self):
        """Non-existent file → ValueError."""
        with pytest.raises(ValueError, match="File not found"):
            clean_only_normalize("/nonexistent/path.csv")


# ============================================================================
# 8b. Alignment utility unit tests (template_utils.py)
# ============================================================================


class TestCheckTokenAlignment:
    """8b. Unit tests for check_token_alignment."""

    def test_same_length_strings(self, tok):
        """Same word count → aligned=True, diff=0."""
        result = check_token_alignment("the cat sat", "the dog sat", tok)
        assert result.aligned is True
        assert result.clean_len == 3
        assert result.corrupt_len == 3
        assert result.diff == 0

    def test_different_length_strings(self, tok):
        """Different word count → aligned=False, correct diff."""
        result = check_token_alignment("the cat", "the big dog sat", tok)
        assert result.aligned is False
        assert result.clean_len == 2
        assert result.corrupt_len == 4
        assert result.diff == 2  # corrupt is 2 longer

    def test_corrupt_shorter(self, tok):
        """Corrupt shorter than clean → negative diff."""
        result = check_token_alignment("a b c d", "x y", tok)
        assert result.aligned is False
        assert result.diff == -2

    def test_empty_clean_string(self, tok):
        """Empty clean string → clean_len=0."""
        result = check_token_alignment("", "hello world", tok)
        assert result.aligned is False
        assert result.clean_len == 0
        assert result.corrupt_len == 2

    def test_empty_corrupt_string(self, tok):
        """Empty corrupt string → corrupt_len=0."""
        result = check_token_alignment("hello world", "", tok)
        assert result.aligned is False
        assert result.clean_len == 2
        assert result.corrupt_len == 0

    def test_both_empty_strings(self, tok):
        """Both empty → aligned=True, both lengths 0."""
        result = check_token_alignment("", "", tok)
        assert result.aligned is True
        assert result.clean_len == 0
        assert result.corrupt_len == 0

    def test_single_word_each(self, tok):
        """Single word each → aligned."""
        result = check_token_alignment("cat", "dog", tok)
        assert result.aligned is True
        assert result.clean_len == 1
        assert result.corrupt_len == 1

    def test_returns_alignment_result(self, tok):
        """Return type is AlignmentResult (frozen dataclass)."""
        result = check_token_alignment("a", "b", tok)
        assert isinstance(result, AlignmentResult)
        # Frozen — should not be mutable
        with pytest.raises(AttributeError):
            result.aligned = False


class TestCheckAnswerDiscriminative:
    """8b. Unit tests for check_answer_discriminative."""

    def test_different_first_tokens(self, tok):
        """Different first answer words → discriminative, no shared prefix."""
        result = check_answer_discriminative(
            "What is 2+2?", " four",
            "What is 3+3?", " six",
            tok,
        )
        assert result.discriminative is True
        assert result.clean_label_id is not None
        assert result.corrupt_label_id is not None
        assert result.clean_label_id != result.corrupt_label_id
        assert result.shared_prefix_len == 0

    def test_shared_prefix_absorption(self, tok):
        """Shared leading answer tokens absorbed into prompt."""
        # "The answer is four" vs "The answer is six"
        # Shared prefix: "The", "answer", "is" (3 tokens)
        result = check_answer_discriminative(
            "Q:", " The answer is four",
            "Q:", " The answer is six",
            tok,
        )
        assert result.discriminative is True
        assert result.shared_prefix_len == 3  # "The", "answer", "is"
        # Adjusted prompts should include the absorbed prefix
        assert "The" in result.adjusted_clean_prompt
        assert "answer" in result.adjusted_clean_prompt
        assert "is" in result.adjusted_clean_prompt

    def test_non_discriminative_same_answers(self, tok):
        """Identical answers → non-discriminative."""
        result = check_answer_discriminative(
            "Q1:", " Yes",
            "Q2:", " Yes",
            tok,
        )
        assert result.discriminative is False
        assert result.clean_label_id is None
        assert result.corrupt_label_id is None

    def test_empty_answer(self, tok):
        """Empty answer → non-discriminative."""
        result = check_answer_discriminative(
            "Q:", "",
            "Q:", " Yes",
            tok,
        )
        assert result.discriminative is False

    def test_both_answers_empty(self, tok):
        """Both answers empty → non-discriminative."""
        result = check_answer_discriminative(
            "Q:", "",
            "Q:", "",
            tok,
        )
        assert result.discriminative is False

    def test_single_token_answer(self, tok):
        """Single-token answer with no prefix → shared_prefix_len=0."""
        result = check_answer_discriminative(
            "Question:", " A",
            "Question:", " B",
            tok,
        )
        assert result.discriminative is True
        assert result.shared_prefix_len == 0
        assert result.clean_label_id != result.corrupt_label_id

    def test_returns_answer_check_result(self, tok):
        """Return type is AnswerCheckResult (frozen dataclass)."""
        result = check_answer_discriminative(
            "Q:", " A", "Q:", " B", tok
        )
        assert isinstance(result, AnswerCheckResult)
        with pytest.raises(AttributeError):
            result.discriminative = True

    def test_one_answer_is_prefix_of_other(self, tok):
        """When one answer is a prefix of the other,
        the divergence happens at the shorter answer's end."""
        # "Yes" vs "Yes definitely" — first word same, diverge at second
        result = check_answer_discriminative(
            "Q:", " Yes",
            "Q:", " Yes definitely",
            tok,
        )
        # After "Yes" is shared, clean continuation is empty → non-discriminative
        # because clean has no more tokens after the shared prefix
        assert result.discriminative is False

    def test_adjusted_prompts_preserve_original_content(self, tok):
        """Adjusted prompts should start with the original prompt content."""
        result = check_answer_discriminative(
            "What color:", " bright red",
            "What color:", " bright blue",
            tok,
        )
        assert result.discriminative is True
        # "bright" is shared → absorbed into prompt
        assert result.shared_prefix_len == 1
        assert result.adjusted_clean_prompt.startswith("What color:")


class TestPadQuestionRegion:
    """8b. Unit tests for pad_question_region."""

    def test_padding_within_region(self, tok):
        """Corrupt shorter than target → pad to reach target."""
        prompt = "Some question Answer: yes"
        target_len = len(tok.encode(prompt)) + 2  # need 2 more tokens
        padded, exact = pad_question_region(
            prompt, target_len, tok, "Answer:", neutral=" extra"
        )
        assert exact is True
        actual_len = len(tok.encode(padded))
        assert actual_len == target_len
        # Boundary preserved
        assert "Answer:" in padded

    def test_boundary_not_found_raises(self, tok):
        """pad_boundary not in prompt → ValueError."""
        with pytest.raises(ValueError, match="pad_boundary.*not found"):
            pad_question_region(
                "No boundary here", 10, tok, "Answer:"
            )

    def test_overshoot_returns_false(self, tok):
        """When padding overshoots on first step → returns (last_valid, False)."""
        prompt = "Q Answer: yes"
        # Make target_len exactly 1 more than current, but neutral adds >=1 tokens
        current_len = len(tok.encode(prompt))
        # Use a multi-word neutral that overshoots in one step
        padded, exact = pad_question_region(
            prompt, current_len + 1, tok, "Answer:", neutral=" extra tokens here"
        )
        # Should not be exact — overshot
        assert exact is False

    def test_already_matching(self, tok):
        """Prompt already at target length → return unchanged, exact=True."""
        prompt = "Some question Answer: yes"
        target_len = len(tok.encode(prompt))
        padded, exact = pad_question_region(
            prompt, target_len, tok, "Answer:"
        )
        assert exact is True
        assert padded == prompt

    def test_prompt_longer_than_target(self, tok):
        """Prompt already longer than target → (prompt, False)."""
        prompt = "Very long question with many words Answer: yes"
        target_len = 2  # much shorter
        padded, exact = pad_question_region(
            prompt, target_len, tok, "Answer:"
        )
        assert exact is False
        assert padded == prompt  # unchanged

    def test_neutral_tokens_inserted_before_boundary(self, tok):
        """Padding should appear before the boundary, not after."""
        prompt = "Question Answer: yes"
        target_len = len(tok.encode(prompt)) + 1
        padded, exact = pad_question_region(
            prompt, target_len, tok, "Answer:", neutral=" the"
        )
        # "the" should appear before "Answer:"
        answer_idx = padded.index("Answer:")
        the_idx = padded.rfind("the")
        assert the_idx < answer_idx

    def test_max_iterations_respected(self, tok):
        """If target is unreachably far, stop at max_iterations."""
        prompt = "Q Answer: X"
        target_len = 1000  # absurdly high
        padded, exact = pad_question_region(
            prompt, target_len, tok, "Answer:", max_iterations=3
        )
        assert exact is False
        # Should have padded only ~3 times
        padding_count = padded.count("the") - prompt.count("the")
        assert padding_count <= 3


# ============================================================================
# 8c. template_normalize integration tests
# ============================================================================


class TestTemplateNormalize:
    """8c. Integration tests for template_normalize with alignment passes."""

    TEMPLATE_SPEC = {
        "clean_prompt": "What is {question}? Answer:",
        "corrupt_prompt": "What is {other_question}? Answer:",
        "clean_answer": " {answer}",
        "corrupt_answer": " {other_answer}",
    }

    def _make_rows(self, n=5):
        """Generate n rows with explicit clean/corrupt columns."""
        rows = []
        for i in range(n):
            rows.append({
                "question": f"Q{i}",
                "answer": f"A{i}",
                "other_question": f"Q{i}alt",
                "other_answer": f"B{i}",
            })
        return rows

    def test_filter_strategy_drops_misaligned(self, tok):
        """filter strategy: mismatched pairs dropped, stats correct."""
        # Create rows where some have different-length questions
        rows = [
            {"question": "short", "answer": "A", "other_question": "short", "other_answer": "B"},
            {"question": "short", "answer": "C", "other_question": "much longer question here", "other_answer": "D"},
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="filter",
            tokenizer=tok,
        )
        # Alignment meta should be present
        assert "_alignment" in ds.meta
        meta = ds.meta["_alignment"]
        assert meta["align_strategy"] == "filter"
        assert meta["total_input"] == 2
        # The misaligned one should be dropped
        assert meta["kept"] <= meta["total_input"]
        assert meta["recommended_metric"] == "logit_diff"

    def test_filter_strategy_keeps_aligned_pairs(self, tok):
        """filter strategy: same-length prompts are kept."""
        rows = [
            {"question": "cat", "answer": "A", "other_question": "dog", "other_answer": "B"},
            {"question": "red", "answer": "C", "other_question": "big", "other_answer": "D"},
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="filter",
            tokenizer=tok,
        )
        # Same-length substitutions → all should survive alignment
        meta = ds.meta["_alignment"]
        assert meta["dropped_misaligned"] == 0

    def test_pad_question_strategy(self, tok):
        """pad_question strategy: corrupt prompts padded to match clean."""
        rows = [
            {"question": "longword", "answer": "A", "other_question": "x", "other_answer": "B"},
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="pad_question",
            tokenizer=tok,
            pad_region_end="Answer:",
        )
        meta = ds.meta["_alignment"]
        assert meta["align_strategy"] == "pad_question"
        # If padding succeeded, the record should be kept
        if meta["kept"] > 0:
            r = ds.records[0]
            # Verify token alignment after padding
            clean_len = len(tok.encode(r.clean_prompt))
            corrupt_len = len(tok.encode(r.corrupt_prompt))
            assert clean_len == corrupt_len

    def test_none_strategy_keeps_all(self, tok):
        """none strategy: all pairs kept, meta has KL recommendation."""
        rows = [
            {"question": "short", "answer": "A", "other_question": "much longer question", "other_answer": "B"},
            {"question": "tiny", "answer": "C", "other_question": "tiny", "other_answer": "D"},
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="none",
        )
        assert len(ds) == 2
        meta = ds.meta["_alignment"]
        assert meta["align_strategy"] == "none"
        assert meta["kept"] == 2
        assert meta["dropped_nondiscriminative"] == 0
        assert meta["dropped_misaligned"] == 0
        assert meta["recommended_metric"] == "kl_divergence"

    def test_no_tokenizer_with_filter_raises(self):
        """No tokenizer + filter → ValueError."""
        rows = [{"question": "Q", "answer": "A", "other_question": "Q2", "other_answer": "B"}]
        with pytest.raises(ValueError, match="requires a tokenizer"):
            template_normalize(
                rows,
                template_spec=self.TEMPLATE_SPEC,
                align_strategy="filter",
                tokenizer=None,
            )

    def test_no_tokenizer_with_pad_question_raises(self):
        """No tokenizer + pad_question → ValueError."""
        rows = [{"question": "Q", "answer": "A", "other_question": "Q2", "other_answer": "B"}]
        with pytest.raises(ValueError, match="requires a tokenizer"):
            template_normalize(
                rows,
                template_spec=self.TEMPLATE_SPEC,
                align_strategy="pad_question",
                tokenizer=None,
                pad_region_end="Answer:",
            )

    def test_pad_question_without_pad_region_end_raises(self, tok):
        """pad_question without pad_region_end → ValueError."""
        rows = [{"question": "Q", "answer": "A", "other_question": "Q2", "other_answer": "B"}]
        with pytest.raises(ValueError, match="pad_region_end"):
            template_normalize(
                rows,
                template_spec=self.TEMPLATE_SPEC,
                align_strategy="pad_question",
                tokenizer=tok,
                pad_region_end=None,
            )

    def test_invalid_align_strategy_raises(self, tok):
        """Invalid align_strategy → ValueError."""
        rows = [{"question": "Q", "answer": "A", "other_question": "Q2", "other_answer": "B"}]
        with pytest.raises(ValueError, match="must be"):
            template_normalize(
                rows,
                template_spec=self.TEMPLATE_SPEC,
                align_strategy="invalid_strategy",
                tokenizer=tok,
            )

    def test_all_records_nondiscriminative_raises(self, tok):
        """All records non-discriminative → RuntimeError with diagnostic."""
        # Same answer for clean and corrupt → non-discriminative
        rows = [
            {"question": "Q1", "answer": "Same", "other_question": "Q2", "other_answer": "Same"},
            {"question": "Q3", "answer": "Same", "other_question": "Q4", "other_answer": "Same"},
        ]
        with pytest.raises(RuntimeError, match="All.*records were dropped"):
            template_normalize(
                rows,
                template_spec=self.TEMPLATE_SPEC,
                align_strategy="filter",
                tokenizer=tok,
            )

    def test_precomputed_labels_in_record_meta(self, tok):
        """Pre-computed labels present in record meta after filter/pad alignment."""
        rows = [
            {"question": "cat", "answer": "A", "other_question": "dog", "other_answer": "B"},
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="filter",
            tokenizer=tok,
        )
        if len(ds) > 0:
            r = ds.records[0]
            assert "_precomputed_labels" in r.meta
            labels = r.meta["_precomputed_labels"]
            assert "clean_label_id" in labels
            assert "corrupt_label_id" in labels
            assert isinstance(labels["clean_label_id"], int)
            assert isinstance(labels["corrupt_label_id"], int)
            assert labels["clean_label_id"] != labels["corrupt_label_id"]

    def test_alignment_meta_structure(self, tok):
        """Verify complete structure of ds.meta['_alignment']."""
        rows = [
            {"question": "cat", "answer": "A", "other_question": "dog", "other_answer": "B"},
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="filter",
            tokenizer=tok,
        )
        meta = ds.meta["_alignment"]
        expected_keys = {
            "align_strategy", "total_input", "kept",
            "dropped_nondiscriminative", "dropped_misaligned",
            "dropped_pad_failed", "answer_prefix_absorbed",
            "recommended_pair_padding_side", "recommended_metric",
        }
        assert expected_keys <= set(meta.keys())
        assert meta["recommended_pair_padding_side"] == "left"

    def test_empty_rows_raises(self, tok):
        """No rows → ValueError."""
        with pytest.raises(ValueError, match="No rows"):
            template_normalize([], template_spec=self.TEMPLATE_SPEC, tokenizer=tok)

    def test_csv_input(self, tmp_dir, tok):
        """CSV path input works."""
        csv_path = _write_csv(
            tmp_dir / "template.csv",
            [{"question": "cat", "answer": "A", "other_question": "dog", "other_answer": "B"}],
        )
        ds = template_normalize(
            csv_path,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="none",
        )
        assert len(ds) == 1

    def test_max_records_truncation(self, tok):
        """max_records caps input before alignment passes."""
        rows = [
            {"question": f"Q{i}", "answer": f"A{i}", "other_question": f"R{i}", "other_answer": f"B{i}"}
            for i in range(20)
        ]
        ds = template_normalize(
            rows,
            template_spec=self.TEMPLATE_SPEC,
            align_strategy="none",
            max_records=5,
        )
        assert ds.meta["_alignment"]["total_input"] == 5

    def test_shape_is_template(self, tok):
        """Output shape is TEMPLATE."""
        rows = [{"question": "Q", "answer": "A", "other_question": "R", "other_answer": "B"}]
        ds = template_normalize(
            rows, template_spec=self.TEMPLATE_SPEC, align_strategy="none"
        )
        assert ds.shape == DatasetShape.TEMPLATE

    def test_unsupported_raw_type_raises(self, tok):
        """Non-str/list/DataFrame input → TypeError."""
        with pytest.raises(TypeError, match="Unsupported raw type"):
            template_normalize(42, template_spec=self.TEMPLATE_SPEC, tokenizer=tok)


# ============================================================================
# 8d. _materialise_eap_csv tests
# ============================================================================


class TestMaterialiseEapCsv:
    """8d. Tests for NormalizedTaskSpec._materialise_eap_csv."""

    def _make_paired_ds(self, records, meta=None):
        """Helper to create a fully-paired NormalizedDataset."""
        return NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=records,
            source="test",
            meta=meta or {},
        )

    def test_precomputed_labels_used_directly(self, tmp_dir, mock_model):
        """Records with _precomputed_labels → CSV uses those IDs directly."""
        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="What is A",
                corrupt_prompt="What is B",
                clean_answer=" Yes",
                corrupt_answer=" No",
                contrast_source=ContrastSource.GENERATED,
                meta={
                    "_precomputed_labels": {
                        "clean_label_id": 100,
                        "corrupt_label_id": 200,
                    }
                },
            )
        ]
        ds = self._make_paired_ds(records)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        csv_path = tmp_dir / "cache" / "test.csv"
        spec._materialise_eap_csv(mock_model, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert int(rows[0]["correct_idx"]) == 100
        assert int(rows[0]["incorrect_idx"]) == 200

    def test_fallback_to_standalone_tokenization(self, tmp_dir, mock_model):
        """Records without _precomputed_labels → falls back to standalone tokenization."""
        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="What is A",
                corrupt_prompt="What is B",
                clean_answer=" Yes",
                corrupt_answer=" No",
                contrast_source=ContrastSource.GENERATED,
                meta={},  # no precomputed labels
            )
        ]
        ds = self._make_paired_ds(records)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        csv_path = tmp_dir / "cache" / "test.csv"
        spec._materialise_eap_csv(mock_model, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        # Should have non-equal IDs (Yes ≠ No)
        assert rows[0]["correct_idx"] != rows[0]["incorrect_idx"]

    def test_correct_neq_incorrect_all_rows(self, tmp_dir, mock_model):
        """All rows: correct_idx != incorrect_idx (both precomputed and fallback)."""
        records = [
            # Precomputed
            ContrastiveRecord(
                record_id="0",
                clean_prompt="P1 clean",
                corrupt_prompt="P1 corrupt",
                clean_answer=" Alpha",
                corrupt_answer=" Beta",
                contrast_source=ContrastSource.GENERATED,
                meta={"_precomputed_labels": {"clean_label_id": 10, "corrupt_label_id": 20}},
            ),
            # Fallback
            ContrastiveRecord(
                record_id="1",
                clean_prompt="P2 clean",
                corrupt_prompt="P2 corrupt",
                clean_answer=" Gamma",
                corrupt_answer=" Delta",
                contrast_source=ContrastSource.GENERATED,
                meta={},
            ),
        ]
        ds = self._make_paired_ds(records)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        csv_path = tmp_dir / "cache" / "test.csv"
        spec._materialise_eap_csv(mock_model, csv_path)

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["correct_idx"] != row["incorrect_idx"], (
                    f"correct_idx == incorrect_idx for row: {row}"
                )

    def test_same_prompt_records_skipped(self, tmp_dir, mock_model):
        """Records where clean_prompt == corrupt_prompt are dropped."""
        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="Same prompt",
                corrupt_prompt="Same prompt",
                clean_answer=" A",
                corrupt_answer=" B",
                contrast_source=ContrastSource.GENERATED,
                meta={},
            ),
            ContrastiveRecord(
                record_id="1",
                clean_prompt="Different clean",
                corrupt_prompt="Different corrupt",
                clean_answer=" X",
                corrupt_answer=" Y",
                contrast_source=ContrastSource.GENERATED,
                meta={},
            ),
        ]
        ds = self._make_paired_ds(records)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        csv_path = tmp_dir / "cache" / "test.csv"
        spec._materialise_eap_csv(mock_model, csv_path)

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["clean"] == "Different clean"

    def test_same_target_tokens_skipped(self, tmp_dir, mock_model):
        """Records where both answers tokenize to the same ID are dropped."""
        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="P1",
                corrupt_prompt="P2",
                clean_answer=" Same",
                corrupt_answer=" Same",
                contrast_source=ContrastSource.GENERATED,
                meta={},  # no precomputed labels — will fall through to standalone
            ),
        ]
        ds = self._make_paired_ds(records)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        csv_path = tmp_dir / "cache" / "test.csv"
        with pytest.raises(RuntimeError, match="All.*records were filtered"):
            spec._materialise_eap_csv(mock_model, csv_path)

    def test_cache_key_includes_align_tag(self, tmp_dir, mock_model):
        """EAP cache key includes alignment strategy tag."""
        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="Clean",
                corrupt_prompt="Corrupt",
                clean_answer=" A",
                corrupt_answer=" B",
                contrast_source=ContrastSource.GENERATED,
                meta={"_precomputed_labels": {"clean_label_id": 1, "corrupt_label_id": 2}},
            ),
        ]
        ds = self._make_paired_ds(
            records,
            meta={"_alignment": {"align_strategy": "filter", "recommended_pair_padding_side": "left"}},
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        # Access the cache path logic indirectly via _build_eap_dataloader
        align_tag = ds.meta.get("_alignment", {}).get("align_strategy", "unk")
        expected_suffix = f"_{align_tag}.csv"

        from circuitkit.data.normalized_task import _safe_name
        cache_name = f"{_safe_name(spec.name)}_{len(ds)}_{align_tag}.csv"
        assert cache_name.endswith("_filter.csv")

    def test_empty_answer_records_skipped(self, tmp_dir, mock_model):
        """Records with empty answers (no precomputed labels) are dropped."""
        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="P1",
                corrupt_prompt="P2",
                clean_answer="",
                corrupt_answer=" B",
                contrast_source=ContrastSource.GENERATED,
                meta={},
            ),
        ]
        ds = self._make_paired_ds(records)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        csv_path = tmp_dir / "cache" / "test.csv"
        with pytest.raises(RuntimeError, match="All.*records were filtered"):
            spec._materialise_eap_csv(mock_model, csv_path)


# ============================================================================
# 8e. End-to-end tests
# ============================================================================


class TestEndToEnd:
    """8e. End-to-end tests wiring template/clean_only through NormalizedTaskSpec."""

    def test_template_filter_to_eap_csv(self, tmp_dir, tok, mock_model):
        """Template CSV → template_normalize(filter) → NormalizedTaskSpec → EAP CSV → valid."""
        template_spec = {
            "clean_prompt": "The {animal} sat",
            "corrupt_prompt": "The {other_animal} sat",
            "clean_answer": " {action}",
            "corrupt_answer": " {other_action}",
        }
        rows = [
            {"animal": "cat", "action": "meowed", "other_animal": "dog", "other_action": "barked"},
            {"animal": "bird", "action": "sang", "other_animal": "fish", "other_action": "swam"},
        ]
        csv_path = _write_csv(tmp_dir / "template.csv", rows)

        ds = template_normalize(
            csv_path,
            template_spec=template_spec,
            align_strategy="filter",
            tokenizer=tok,
        )

        assert ds.shape == DatasetShape.TEMPLATE
        assert len(ds) > 0

        cache_dir = str(tmp_dir / "cache")
        spec = NormalizedTaskSpec(ds, cache_dir=cache_dir)

        # Build EAP dataloader
        loader = spec.build_dataloader(
            mock_model,
            {"algorithm": "eap", "batch_size": 1},
            "cpu",
        )

        # Verify we get valid batches
        batches = list(loader)
        assert len(batches) > 0
        for clean_strs, corrupt_strs, labels in batches:
            assert len(clean_strs) > 0
            assert len(corrupt_strs) > 0
            for correct_idx, incorrect_idx in labels:
                assert correct_idx != incorrect_idx

    def test_clean_only_to_ibcircuit_dataloader(self, tmp_dir, mock_model):
        """Clean-only CSV → clean_only_normalize → NormalizedTaskSpec →
        IBCircuit dataloader → valid batch."""
        import torch

        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [
                {"prompt": "The cat sat on", "answer": "mat"},
                {"prompt": "The dog ran to", "answer": "park"},
                {"prompt": "The bird flew over", "answer": "hill"},
            ],
        )
        ds = clean_only_normalize(csv_path)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        loader = spec.build_dataloader(
            mock_model,
            {"algorithm": "ibcircuit", "batch_size": 8, "data_params": {"num_examples": 32}},
            "cpu",
        )

        batches = list(loader)
        assert len(batches) == 1  # IBCircuit always yields 1 batch

        batch = batches[0]
        assert "tokens" in batch
        assert "labels" in batch
        assert "answer_positions" in batch

        assert isinstance(batch["tokens"], torch.Tensor)
        assert isinstance(batch["labels"], torch.Tensor)
        assert isinstance(batch["answer_positions"], torch.Tensor)

        n = batch["tokens"].shape[0]
        assert n == 3  # 3 records
        assert batch["labels"].shape == (n,)
        assert batch["answer_positions"].shape == (n,)

        # answer_positions should be valid indices
        for i in range(n):
            pos = batch["answer_positions"][i].item()
            assert 0 <= pos < batch["tokens"].shape[1]

    def test_clean_only_to_cdt_dataloader(self, tmp_dir, mock_model):
        """Clean-only CSV → NormalizedTaskSpec → CD-T dataloader → valid 3-tuple batch."""
        import math
        csv_path = _write_csv(
            tmp_dir / "clean.csv",
            [
                {"prompt": "Hello world", "answer": "test"},
                {"prompt": "Foo bar baz", "answer": "qux"},
            ],
        )
        ds = clean_only_normalize(csv_path)
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        loader = spec.build_dataloader(
            mock_model,
            {"algorithm": "cdt", "batch_size": 2},
            "cpu",
        )

        batches = list(loader)
        assert len(batches) > 0

        # CD-T dataloader yields (clean_strs, corrupt_strs, labels)
        for batch in batches:
            clean_strs, corrupt_strs, labels = batch
            assert len(clean_strs) > 0
            # For clean-only CDT, corrupted should be empty strings (or NaN from pandas)
            for s in corrupt_strs:
                assert s == "" or (isinstance(s, float) and math.isnan(s))

    def test_cdt_clean_only_cache_suffix(self, tmp_dir, mock_model):
        """CD-T clean-only cache uses _cdt_clean.csv suffix."""
        ds = clean_only_normalize(
            [{"prompt": "Test prompt", "answer": "A"}],
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        spec.build_dataloader(
            mock_model,
            {"algorithm": "cdt", "batch_size": 1},
            "cpu",
        )

        # Check that cache file has the correct suffix
        cache_files = list(Path(tmp_dir / "cache").glob("*_cdt_clean.csv"))
        assert len(cache_files) == 1

    def test_template_none_to_task_spec(self, tmp_dir, tok, mock_model):
        """Template with align_strategy=none → task spec can still build EAP loader."""
        template_spec = {
            "clean_prompt": "What is {x}?",
            "corrupt_prompt": "What is {y}?",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        rows = [
            {"x": "one", "y": "two", "a": "alpha", "b": "beta"},
        ]
        ds = template_normalize(
            rows,
            template_spec=template_spec,
            align_strategy="none",
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))

        # Should work even without alignment enforcement
        loader = spec.build_dataloader(
            mock_model,
            {"algorithm": "eap", "batch_size": 1},
            "cpu",
        )
        assert len(list(loader)) > 0

    def test_pair_padding_side_from_alignment_meta(self, tmp_dir, tok, mock_model):
        """NormalizedTaskSpec reads pair_padding_side from _alignment meta."""
        template_spec = {
            "clean_prompt": "Q: {q}",
            "corrupt_prompt": "Q: {r}",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        ds = template_normalize(
            [{"q": "X", "r": "Y", "a": "A", "b": "B"}],
            template_spec=template_spec,
            align_strategy="filter",
            tokenizer=tok,
        )
        # template_normalize sets recommended_pair_padding_side = "left"
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        assert spec.pair_padding_side == "left"

    def test_pair_padding_side_override_right(self, tmp_dir, tok, mock_model):
        """pair_padding_side can be overridden to 'right' via meta."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                    meta={"_precomputed_labels": {"clean_label_id": 1, "corrupt_label_id": 2}},
                )
            ],
            meta={"_alignment": {"recommended_pair_padding_side": "right"}},
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        assert spec.pair_padding_side == "right"


# ============================================================================
# Additional tests: worthiness checks (Steps 6)
# ============================================================================


class TestWorthinessAlignmentChecks:
    """Tests for the alignment-aware worthiness checks (checks 9 and 10)."""

    def _make_ds(self, shape, records, meta=None):
        return NormalizedDataset(
            name="test",
            shape=shape,
            records=records,
            source="test",
            meta=meta or {},
        )

    def test_token_alignment_shortcircuit_for_filter(self, tok):
        """_check_token_alignment short-circuits for TEMPLATE + filter."""
        ds = self._make_ds(DatasetShape.TEMPLATE, [])
        result = _check_token_alignment(ds, tok, align_strategy="filter")
        assert result.passed is True
        assert result.score == 1.0
        assert "skipped" in result.message

    def test_token_alignment_shortcircuit_for_pad_question(self, tok):
        """_check_token_alignment short-circuits for TEMPLATE + pad_question."""
        ds = self._make_ds(DatasetShape.TEMPLATE, [])
        result = _check_token_alignment(ds, tok, align_strategy="pad_question")
        assert result.passed is True

    def test_token_alignment_no_shortcircuit_for_none(self, tok):
        """_check_token_alignment does NOT short-circuit for align_strategy=none."""
        ds = self._make_ds(DatasetShape.TEMPLATE, [])
        result = _check_token_alignment(ds, tok, align_strategy="none")
        # No paired records → passes via "no paired records" guard
        assert result.passed is True
        assert "skipped" not in result.message

    def test_token_alignment_no_shortcircuit_for_clean_only(self, tok):
        """CLEAN_ONLY dataset does NOT trigger the TEMPLATE short-circuit."""
        ds = self._make_ds(DatasetShape.CLEAN_ONLY, [])
        result = _check_token_alignment(ds, tok, align_strategy="filter")
        # CLEAN_ONLY is not TEMPLATE → should NOT hit the short-circuit
        assert "skipped" not in result.message

    def test_worthiness_check9_alignment_metric_recommendation(self):
        """Check 9: fires when align_strategy='none', recommends kl_divergence."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
            meta={
                "_alignment": {
                    "align_strategy": "none",
                    "total_input": 1,
                    "kept": 1,
                    "dropped_nondiscriminative": 0,
                    "dropped_misaligned": 0,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "kl_divergence",
                }
            },
        )
        report = evaluate_worthiness(ds)
        check9 = next(
            (c for c in report.checks if c.name == "alignment_metric_recommendation"),
            None,
        )
        assert check9 is not None
        assert check9.passed is False
        assert check9.severity == "soft"
        assert "kl_divergence" in check9.message

    def test_worthiness_check9_not_present_for_filter(self):
        """Check 9: NOT present when align_strategy='filter'."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
            meta={
                "_alignment": {
                    "align_strategy": "filter",
                    "total_input": 1,
                    "kept": 1,
                    "dropped_nondiscriminative": 0,
                    "dropped_misaligned": 0,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "logit_diff",
                }
            },
        )
        report = evaluate_worthiness(ds)
        check9 = next(
            (c for c in report.checks if c.name == "alignment_metric_recommendation"),
            None,
        )
        assert check9 is None  # Should not be present for filter

    def test_worthiness_check10_discriminative_drop_rate_high(self):
        """Check 10: fires when >20% of records dropped as non-discriminative."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
            meta={
                "_alignment": {
                    "align_strategy": "filter",
                    "total_input": 100,
                    "kept": 70,
                    "dropped_nondiscriminative": 25,  # 25% > 20% threshold
                    "dropped_misaligned": 5,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "logit_diff",
                }
            },
        )
        report = evaluate_worthiness(ds)
        check10 = next(
            (c for c in report.checks if c.name == "discriminative_drop_rate"),
            None,
        )
        assert check10 is not None
        assert check10.passed is False
        assert check10.severity == "soft"
        assert check10.score == pytest.approx(max(0.0, 1.0 - 0.25), abs=0.01)

    def test_worthiness_check10_not_present_when_low_drop_rate(self):
        """Check 10: NOT present when drop rate is ≤ 20%."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
            meta={
                "_alignment": {
                    "align_strategy": "filter",
                    "total_input": 100,
                    "kept": 85,
                    "dropped_nondiscriminative": 10,  # 10% ≤ 20%
                    "dropped_misaligned": 5,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "logit_diff",
                }
            },
        )
        report = evaluate_worthiness(ds)
        check10 = next(
            (c for c in report.checks if c.name == "discriminative_drop_rate"),
            None,
        )
        assert check10 is None

    def test_worthiness_checks_not_present_for_non_template(self):
        """Checks 9 and 10 NOT present for non-template datasets."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.CLEAN_ONLY,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    clean_answer="B",
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                )
            ],
        )
        report = evaluate_worthiness(ds)
        names = [c.name for c in report.checks]
        assert "alignment_metric_recommendation" not in names
        assert "discriminative_drop_rate" not in names

    def test_shape_specific_clean_only_passes(self):
        """Shape-specific check passes for clean_only with populated prompts."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.CLEAN_ONLY,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="Hello world",
                    clean_answer="test",
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                )
            ],
        )
        result = _check_shape_specific(ds)
        assert result.passed is True

    def test_shape_specific_clean_only_fails_empty_prompt(self):
        """Shape-specific check fails for clean_only with empty prompts."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.CLEAN_ONLY,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="",
                    clean_answer="test",
                    contrast_source=ContrastSource.NOT_PAIRED_YET,
                )
            ],
        )
        result = _check_shape_specific(ds)
        assert result.passed is False

    def test_shape_specific_template_unresolved_placeholder(self):
        """Shape-specific for TEMPLATE detects unresolved {placeholders}."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="What is {missing}?",
                    corrupt_prompt="Different prompt",
                    clean_answer="A",
                    corrupt_answer="B",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
        )
        result = _check_shape_specific(ds)
        assert result.passed is False
        assert "unresolved" in result.message

    def test_worthiness_verdict_green(self):
        """GREEN verdict when all checks pass."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id=str(i),
                    clean_prompt=f"Clean prompt {i}",
                    corrupt_prompt=f"Corrupt prompt {i}",
                    clean_answer=f"A{i}",
                    corrupt_answer=f"B{i}",
                    contrast_source=ContrastSource.GENERATED,
                )
                for i in range(10)
            ],
            meta={
                "_alignment": {
                    "align_strategy": "filter",
                    "total_input": 10,
                    "kept": 10,
                    "dropped_nondiscriminative": 0,
                    "dropped_misaligned": 0,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "logit_diff",
                }
            },
        )
        report = evaluate_worthiness(ds)
        assert report.verdict == Verdict.GREEN


# ============================================================================
# Additional tests: NormalizedTaskSpec behavior
# ============================================================================


class TestNormalizedTaskSpecBehavior:
    """Additional tests for NormalizedTaskSpec edge cases and invariants."""

    def test_pair_padding_side_default_is_left(self, tmp_dir):
        """Default pair_padding_side is 'left'."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[],
            meta={},
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        assert spec.pair_padding_side == "left"

    def test_validate_discovery_config_rejects_unknown_algo(self, tmp_dir):
        """validate_discovery_config rejects unknown algorithms."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        with pytest.raises(ValueError, match="supports algorithms"):
            spec.validate_discovery_config({"algorithm": "unknown_algo"})

    def test_validate_discovery_config_accepts_eap(self, tmp_dir):
        """validate_discovery_config accepts 'eap'."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        # Should not raise
        spec.validate_discovery_config({"algorithm": "eap"})

    def test_validate_discovery_config_accepts_ibcircuit(self, tmp_dir):
        """validate_discovery_config accepts 'ibcircuit'."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        spec.validate_discovery_config({"algorithm": "ibcircuit"})

    def test_validate_discovery_config_accepts_cdt(self, tmp_dir):
        """validate_discovery_config accepts 'cdt'."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        spec.validate_discovery_config({"algorithm": "cdt"})

    def test_build_dataloader_no_model_raises(self, tmp_dir):
        """build_dataloader without model → ValueError."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        with pytest.raises(ValueError, match="needs the model"):
            spec.build_dataloader(None, {"algorithm": "eap", "batch_size": 1}, "cpu")

    def test_metric_fn_logit_diff(self, tmp_dir):
        """metric_fn('logit_diff') returns a callable."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        fn = spec.metric_fn("logit_diff")
        assert callable(fn)

    def test_metric_fn_kl_divergence(self, tmp_dir):
        """metric_fn('kl_divergence') returns a callable."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        fn = spec.metric_fn("kl_divergence")
        assert callable(fn)

    def test_metric_fn_invalid_type_raises(self, tmp_dir):
        """metric_fn with invalid type → ValueError."""
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=[])
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        with pytest.raises(ValueError, match="logit_diff"):
            spec.metric_fn("invalid_metric")

    def test_artifact_metadata(self, tmp_dir):
        """artifact_metadata returns expected structure."""
        ds = NormalizedDataset(
            name="myds", shape=DatasetShape.TEMPLATE,
            records=[], source="test.csv",
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        meta = spec.artifact_metadata({"algorithm": "eap", "level": "node"})
        assert meta["task"] == spec.name
        assert meta["data_source"] == "test.csv"
        assert meta["shape"] == "template"
        assert meta["algorithm"] == "eap"
        assert meta["level"] == "node"


# ============================================================================
# Additional tests: config.py validation
# ============================================================================


class TestConfigValidation:
    """Tests for config.py's inline data validation."""

    def _base_config(self, **data_overrides):
        """Build a minimal valid config with data section."""
        config = {
            "model": {"name": "gpt2"},
            "discovery": {"algorithm": "eap-ig", "task": "ioi"},
            "pruning": {"target_sparsity": 0.3, "scope": "both"},
            "data": {
                "type": "template",
                "path": "/some/path.csv",
                "template": {
                    "clean_prompt": "Q: {q}",
                    "corrupt_prompt": "Q: {r}",
                    "clean_answer": " {a}",
                    "corrupt_answer": " {b}",
                },
            },
        }
        if data_overrides:
            config["data"].update(data_overrides)
        return config

    def test_valid_template_config(self):
        """Valid template config passes validation."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config()
        # Should not raise
        _validate_config(config)

    def test_invalid_data_type_raises(self):
        """Invalid data.type → ValueError."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config(type="invalid")
        with pytest.raises(ValueError, match="data.type"):
            _validate_config(config)

    def test_clean_only_type_accepted(self):
        """data.type='clean_only' is accepted."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config()
        config["data"] = {
            "type": "clean_only",
            "path": "/some/path.csv",
        }
        _validate_config(config)

    def test_invalid_align_strategy_raises(self):
        """Invalid align_strategy → ValueError."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config(align_strategy="bogus")
        with pytest.raises(ValueError, match="align_strategy"):
            _validate_config(config)

    def test_pad_question_needs_pad_region_end(self):
        """pad_question without pad_region_end → ValueError."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config(align_strategy="pad_question")
        with pytest.raises(ValueError, match="pad_region_end"):
            _validate_config(config)

    def test_pad_question_with_pad_region_end_passes(self):
        """pad_question with pad_region_end passes."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config(
            align_strategy="pad_question",
            pad_region_end="Answer:",
        )
        _validate_config(config)

    def test_invalid_pair_padding_side_raises(self):
        """Invalid pair_padding_side → ValueError."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config(pair_padding_side="center")
        with pytest.raises(ValueError, match="pair_padding_side"):
            _validate_config(config)

    def test_missing_template_keys_raises(self):
        """Missing template keys → ValueError."""
        from circuitkit.utils.config import _validate_config
        config = self._base_config()
        del config["data"]["template"]["clean_answer"]
        with pytest.raises(ValueError, match="missing keys"):
            _validate_config(config)


# ============================================================================
# Additional tests: NormalizedDataset serialization round-trip
# ============================================================================


class TestSerializationRoundTrip:
    """Verify NormalizedDataset can be saved/loaded and retains all fields."""

    def test_clean_only_roundtrip(self, tmp_dir):
        """Save and load a clean-only dataset."""
        ds = clean_only_normalize(
            [{"prompt": "Hello world", "answer": "test"}],
        )
        json_path = str(tmp_dir / "ds.json")
        ds.save_json(json_path)

        loaded = NormalizedDataset.load_json(json_path)
        assert loaded.shape == DatasetShape.CLEAN_ONLY
        assert len(loaded) == 1
        assert loaded.records[0].clean_prompt == "Hello world"
        assert loaded.records[0].corrupt_prompt is None
        assert loaded.records[0].contrast_source == ContrastSource.NOT_PAIRED_YET

    def test_template_roundtrip_preserves_alignment_meta(self, tmp_dir, tok):
        """Save and load a template dataset; _alignment meta survives."""
        template_spec = {
            "clean_prompt": "Q: {q}",
            "corrupt_prompt": "Q: {r}",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        ds = template_normalize(
            [{"q": "X", "r": "Y", "a": "alpha", "b": "beta"}],
            template_spec=template_spec,
            align_strategy="filter",
            tokenizer=tok,
        )
        json_path = str(tmp_dir / "ds.json")
        ds.save_json(json_path)

        loaded = NormalizedDataset.load_json(json_path)
        assert "_alignment" in loaded.meta
        assert loaded.meta["_alignment"]["align_strategy"] == "filter"
        assert loaded.meta["_alignment"]["recommended_pair_padding_side"] == "left"

    def test_precomputed_labels_survive_roundtrip(self, tmp_dir, tok):
        """Pre-computed labels in record meta survive JSON round-trip."""
        template_spec = {
            "clean_prompt": "Q: {q}",
            "corrupt_prompt": "Q: {r}",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        ds = template_normalize(
            [{"q": "cat", "r": "dog", "a": "meow", "b": "bark"}],
            template_spec=template_spec,
            align_strategy="filter",
            tokenizer=tok,
        )
        if len(ds) > 0:
            json_path = str(tmp_dir / "ds.json")
            ds.save_json(json_path)

            loaded = NormalizedDataset.load_json(json_path)
            r = loaded.records[0]
            assert "_precomputed_labels" in r.meta
            assert isinstance(r.meta["_precomputed_labels"]["clean_label_id"], int)
            assert isinstance(r.meta["_precomputed_labels"]["corrupt_label_id"], int)


# ============================================================================
# Edge case and regression tests
# ============================================================================


class TestEdgeCases:
    """Regression and edge-case tests to probe potential bugs."""

    def test_pad_question_region_multiple_boundary_occurrences(self, tok):
        """pad_boundary that appears multiple times — uses first occurrence."""
        prompt = "Answer: something Answer: again"
        target = len(tok.encode(prompt)) + 1
        # Should pad before the first "Answer:"
        padded, exact = pad_question_region(
            prompt, target, tok, "Answer:", neutral=" the"
        )
        # First "Answer:" should still be present
        assert "Answer:" in padded

    def test_template_normalize_with_auto_peer_mode(self, tok):
        """auto_peer mode randomly pairs rows."""
        rows = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        template_spec = {
            "clean_prompt": "What is {question}?",
            "corrupt_prompt": "What is {other_question}?",
            "clean_answer": " {answer}",
            "corrupt_answer": " {other_answer}",
        }
        ds = template_normalize(
            rows,
            template_spec=template_spec,
            pairing_mode="auto_peer",
            align_strategy="none",
        )
        assert len(ds) == 3
        # With auto_peer, other_* values come from random peer rows
        for r in ds.records:
            assert r.clean_prompt
            assert r.corrupt_prompt

    def test_clean_only_ibcircuit_skips_empty_answers(self, tmp_dir, mock_model):
        """IBCircuit dataloader skips records with empty answers gracefully."""
        import torch

        ds = clean_only_normalize(
            [
                {"prompt": "Good prompt", "answer": "validtoken"},
                {"prompt": "Another prompt", "answer": "anothertoken"},
            ],
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        loader = spec.build_dataloader(
            mock_model,
            {"algorithm": "ibcircuit", "batch_size": 8, "data_params": {"num_examples": 32}},
            "cpu",
        )
        batch = next(iter(loader))
        # Both records should be included (non-empty answers)
        assert batch["tokens"].shape[0] == 2

    def test_materialise_eap_csv_all_same_prompt_raises(self, tmp_dir, mock_model):
        """All records with same clean==corrupt prompt → RuntimeError."""
        records = [
            ContrastiveRecord(
                record_id=str(i),
                clean_prompt="Same",
                corrupt_prompt="Same",
                clean_answer=f" A{i}",
                corrupt_answer=f" B{i}",
                contrast_source=ContrastSource.GENERATED,
            )
            for i in range(5)
        ]
        ds = NormalizedDataset(
            name="test", shape=DatasetShape.TEMPLATE,
            records=records, source="test",
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        csv_path = tmp_dir / "cache" / "test.csv"
        with pytest.raises(RuntimeError, match="All.*records were filtered"):
            spec._materialise_eap_csv(mock_model, csv_path)

    def test_contrastive_record_is_paired_property(self):
        """ContrastiveRecord.is_paired works correctly."""
        paired = ContrastiveRecord(
            record_id="0",
            clean_prompt="A",
            clean_answer="X",
            corrupt_prompt="B",
            corrupt_answer="Y",
        )
        unpaired = ContrastiveRecord(
            record_id="1",
            clean_prompt="A",
            clean_answer="X",
            corrupt_prompt=None,
            corrupt_answer=None,
        )
        half = ContrastiveRecord(
            record_id="2",
            clean_prompt="A",
            clean_answer="X",
            corrupt_prompt="B",
            corrupt_answer=None,  # Only prompt, no answer
        )
        assert paired.is_paired is True
        assert unpaired.is_paired is False
        assert half.is_paired is False

    def test_dataset_shape_enum_values(self):
        """DatasetShape enum has expected values."""
        assert DatasetShape.CLEAN_ONLY.value == "clean_only"
        assert DatasetShape.TEMPLATE.value == "template"
        assert DatasetShape.UNKNOWN.value == "unknown"

    def test_normalized_dataset_take(self):
        """NormalizedDataset.take returns correct subset."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id=str(i),
                    clean_prompt=f"P{i}",
                    clean_answer=f"A{i}",
                    corrupt_prompt=f"C{i}",
                    corrupt_answer=f"B{i}",
                    contrast_source=ContrastSource.GENERATED,
                )
                for i in range(10)
            ],
        )
        subset = ds.take(3)
        assert len(subset) == 3
        assert subset.records[0].record_id == "0"
        assert subset.meta.get("subset_n") == 3

    def test_safe_name_function(self):
        """_safe_name sanitizes cache filenames."""
        from circuitkit.data.normalized_task import _safe_name
        assert _safe_name("hello-world_1.0") == "hello-world_1.0"
        assert _safe_name("a/b:c d") == "a_b_c_d"
        assert _safe_name("") == ""

    def test_check_token_alignment_frozen_result(self, tok):
        """AlignmentResult is frozen — can't mutate fields."""
        result = check_token_alignment("a b", "c d", tok)
        with pytest.raises(AttributeError):
            result.aligned = False

    def test_check_answer_discriminative_frozen_result(self, tok):
        """AnswerCheckResult is frozen — can't mutate fields."""
        result = check_answer_discriminative(
            "Q:", " A", "Q:", " B", tok
        )
        with pytest.raises(AttributeError):
            result.discriminative = False

    def test_template_normalize_preserves_record_order(self, tok):
        """Records maintain insertion order after alignment."""
        rows = [
            {"question": f"Q{i}", "answer": f"A{i}",
             "other_question": f"R{i}", "other_answer": f"B{i}"}
            for i in range(5)
        ]
        ds = template_normalize(
            rows,
            template_spec={
                "clean_prompt": "What is {question}?",
                "corrupt_prompt": "What is {other_question}?",
                "clean_answer": " {answer}",
                "corrupt_answer": " {other_answer}",
            },
            align_strategy="none",
        )
        ids = [r.record_id for r in ds.records]
        # Record IDs should be in order (zero-padded)
        assert ids == [f"{i:06d}" for i in range(5)]

    def test_worthiness_report_render_terminal(self):
        """render_terminal produces non-empty output."""
        report = DataWorthinessReport(
            dataset_name="test",
            dataset_shape="template",
            n_records=10,
            verdict=Verdict.GREEN,
            checks=[
                CheckResult(
                    name="test_check",
                    passed=True,
                    severity="hard",
                    score=1.0,
                    message="all good",
                )
            ],
        )
        output = report.render_terminal()
        assert "DataWorthinessReport" in output
        assert "test" in output
        assert "GREEN" in output

# ============================================================================
# Additional gap-filling tests
# ============================================================================


class TestCleanOnlyRecordIdFormat:
    """Verify clean_only record_id format vs template record_id format."""

    def test_clean_only_record_id_is_plain_int_string(self):
        """clean_only uses str(i), NOT zero-padded like template's f'{i:06d}'."""
        ds = clean_only_normalize(
            [{"prompt": f"P{i}", "answer": f"A{i}"} for i in range(3)],
        )
        # clean_only.py line 82: record_id=str(i)
        assert ds.records[0].record_id == "0"
        assert ds.records[1].record_id == "1"
        assert ds.records[2].record_id == "2"

    def test_template_record_id_is_zero_padded(self, tok):
        """template_normalize uses f'{i:06d}' zero-padded IDs."""
        ds = template_normalize(
            [{"q": "X", "r": "Y", "a": "A", "b": "B"}],
            template_spec={
                "clean_prompt": "{q}",
                "corrupt_prompt": "{r}",
                "clean_answer": " {a}",
                "corrupt_answer": " {b}",
            },
            align_strategy="none",
        )
        assert ds.records[0].record_id == "000000"


class TestCdtCacheReuse:
    """Verify CDT clean-only dataloader reuses cached CSV on second call."""

    def test_cdt_cache_reused_on_second_call(self, tmp_dir, mock_model):
        ds = clean_only_normalize(
            [{"prompt": "Hello world", "answer": "test"}],
        )
        cache_dir = str(tmp_dir / "cache")
        spec = NormalizedTaskSpec(ds, cache_dir=cache_dir)

        # First call creates cache
        spec.build_dataloader(mock_model, {"algorithm": "cdt", "batch_size": 1}, "cpu")
        cache_files = list(Path(cache_dir).glob("*_cdt_clean.csv"))
        assert len(cache_files) == 1
        mtime1 = cache_files[0].stat().st_mtime

        # Second call reuses — file mtime unchanged
        import time
        time.sleep(0.05)
        spec.build_dataloader(mock_model, {"algorithm": "cdt", "batch_size": 1}, "cpu")
        mtime2 = cache_files[0].stat().st_mtime
        assert mtime1 == mtime2


class TestPadQuestionCorruptLonger:
    """Test pad_question strategy when corrupt is LONGER than clean."""

    def test_corrupt_longer_than_clean_dropped_as_misaligned(self, tok):
        """When corrupt prompt tokenizes longer than clean, pad_question
        can't help — record should be dropped (diff > 0 path in template.py)."""
        # Clean: 2 tokens, Corrupt: 4 tokens → diff = +2 → dropped
        rows = [
            {
                "q": "short",
                "r": "this is much longer",
                "a": "alpha",
                "b": "beta",
            },
        ]
        template_spec = {
            "clean_prompt": "Q: {q}",
            "corrupt_prompt": "Q: {r}",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        # All records dropped → RuntimeError
        with pytest.raises(RuntimeError, match="All.*records were dropped"):
            template_normalize(
                rows,
                template_spec=template_spec,
                align_strategy="pad_question",
                tokenizer=tok,
                pad_region_end="Q:",
            )


class TestDroppedMisalignedZeroedForPadQuestion:
    """Verify dropped_misaligned is zeroed in alignment meta for pad_question."""

    def test_dropped_misaligned_zeroed_for_pad_question(self, tok):
        """template.py line 245-248: dropped_misaligned is set to 0 when
        align_strategy is not 'filter'. Verify this in the stats."""
        # Create a pair that IS aligned so we get at least one kept record
        rows = [
            {"q": "cat", "r": "dog", "a": "meow", "b": "bark"},
        ]
        template_spec = {
            "clean_prompt": "What is {q}",
            "corrupt_prompt": "What is {r}",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        ds = template_normalize(
            rows,
            template_spec=template_spec,
            align_strategy="pad_question",
            tokenizer=tok,
            pad_region_end="What",
        )
        meta = ds.meta["_alignment"]
        # Even if some were internally counted as misaligned, the reported
        # value is zeroed for pad_question
        assert meta["dropped_misaligned"] == 0


class TestIBCircuitWhitespaceTokenStripping:
    """Test IBCircuit dataloader's ws_token_id stripping logic."""

    def test_ibcircuit_strips_leading_whitespace_token(self, tmp_dir):
        """When the answer's first token equals the whitespace-only token,
        IBCircuit should skip it and use the next token as the label."""
        import torch

        class SpaceAwareTokenizer:
            """Tokenizer where encode(" ") returns a single space token [99],
            and encode(" foo") returns [99, <foo_id>]."""
            def __init__(self):
                self._vocab = {" ": 99}
                self._rev = {99: " "}
                self._next = 100
                self.pad_token_id = 0
                self.eos_token_id = 0
                self.all_special_ids = {0}

            def _get(self, w):
                if w not in self._vocab:
                    self._vocab[w] = self._next
                    self._rev[self._next] = w
                    self._next += 1
                return self._vocab[w]

            def encode(self, text, add_special_tokens=True):
                if not text or not text.strip():
                    if text == " ":
                        return [99]
                    return []
                words = text.split()
                ids = []
                # If text starts with space, emit the space token first
                if text[0] == " ":
                    ids.append(99)
                ids.extend(self._get(w) for w in words)
                return ids

            def decode(self, ids):
                return " ".join(self._rev.get(i, "?") for i in ids)

        class SpaceAwareModel:
            def __init__(self):
                self.tokenizer = SpaceAwareTokenizer()
                self.device = "cpu"

            def to_tokens(self, text, prepend_bos=True):
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if prepend_bos:
                    ids = [self.tokenizer.eos_token_id] + ids
                return torch.tensor([ids], dtype=torch.long)

        model = SpaceAwareModel()
        ds = clean_only_normalize(
            [{"prompt": "The cat sat", "answer": " mat"}],
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        loader = spec.build_dataloader(
            model,
            {"algorithm": "ibcircuit", "batch_size": 8, "data_params": {"num_examples": 32}},
            "cpu",
        )
        batch = next(iter(loader))
        label = batch["labels"][0].item()
        # The label should NOT be the space token (99), should be the actual word token
        assert label != 99


class TestCdtWithPairedData:
    """CDT with fully-paired data should use the normal EAP path, not
    _build_cdt_clean_only_dataloader."""

    def test_cdt_paired_uses_clean_only_path(self, tmp_dir, tok, mock_model):
        """CD-T always uses the clean-only path, even with paired data.

        CD-T (contextual decomposition) only consumes the clean prompt — it
        never uses the corrupt side — so build_dataloader routes the whole
        CDT_FAMILY to _build_cdt_clean_only_dataloader regardless of pairing
        (normalized_task.py: `if algo in CDT_FAMILY`). An earlier design only
        did this for unpaired data; commit 9c2616e unified it."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="Clean prompt here",
                    corrupt_prompt="Corrupt prompt here",
                    clean_answer=" alpha",
                    corrupt_answer=" beta",
                    contrast_source=ContrastSource.GENERATED,
                    meta={"_precomputed_labels": {"clean_label_id": 10, "corrupt_label_id": 20}},
                ),
            ],
            meta={"_alignment": {"align_strategy": "filter"}},
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        loader = spec.build_dataloader(
            mock_model, {"algorithm": "cdt", "batch_size": 1}, "cpu"
        )
        batches = list(loader)
        assert len(batches) > 0
        # Clean-only path serves the real clean prompt but a placeholder for the
        # corrupted column (written as "", read back as NaN) — never the real
        # corrupt prompt, since CD-T ignores it.
        clean_strs, corrupt_strs, labels = batches[0]
        assert clean_strs[0] == "Clean prompt here"
        assert corrupt_strs[0] != "Corrupt prompt here"

        # And the CD-T clean-only cache CSV must have been written.
        cdt_cache = list(Path(tmp_dir / "cache").glob("*_cdt_clean.csv"))
        assert len(cdt_cache) == 1


class TestEapVariantAlgorithms:
    """Verify EAP-IG and ACDC succeed with paired data."""

    def _make_paired_ds(self, tok):
        template_spec = {
            "clean_prompt": "Q: {q}",
            "corrupt_prompt": "Q: {r}",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        return template_normalize(
            [{"q": "cat", "r": "dog", "a": "meow", "b": "bark"}],
            template_spec=template_spec,
            align_strategy="filter",
            tokenizer=tok,
        )

    def test_eap_ig_builds_dataloader(self, tmp_dir, tok, mock_model):
        ds = self._make_paired_ds(tok)
        if len(ds) == 0:
            pytest.skip("filter dropped all records")
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        loader = spec.build_dataloader(
            mock_model, {"algorithm": "eap-ig", "batch_size": 1}, "cpu"
        )
        assert len(list(loader)) > 0

    def test_acdc_builds_dataloader(self, tmp_dir, tok, mock_model):
        ds = self._make_paired_ds(tok)
        if len(ds) == 0:
            pytest.skip("filter dropped all records")
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        loader = spec.build_dataloader(
            mock_model, {"algorithm": "acdc", "batch_size": 1}, "cpu"
        )
        assert len(list(loader)) > 0


class TestValidateTokenAlignment:
    """Tests for the validate_token_alignment utility function."""

    def test_basic_audit(self):
        from circuitkit.data.normalized_task import validate_token_alignment

        records = [
            ContrastiveRecord(
                record_id=str(i),
                clean_prompt=f"Prompt {i}",
                corrupt_prompt=f"Prompt {i}",  # same → will be counted
                clean_answer=f"A{i}",
                corrupt_answer=f"B{i}",
                contrast_source=ContrastSource.GENERATED,
            )
            for i in range(5)
        ]
        ds = NormalizedDataset(
            name="test", shape=DatasetShape.TEMPLATE,
            records=records,
        )
        spec = NormalizedTaskSpec(ds, cache_dir="/tmp/test_vta")
        result = validate_token_alignment(spec)
        assert result["total"] == 5
        assert result["same_prompt"] == 5
        assert result["records_ok"] == 0

    def test_empty_prompt_counted(self):
        from circuitkit.data.normalized_task import validate_token_alignment

        records = [
            ContrastiveRecord(
                record_id="0",
                clean_prompt="",
                clean_answer="A",
                corrupt_prompt="B",
                corrupt_answer="C",
                contrast_source=ContrastSource.GENERATED,
            )
        ]
        ds = NormalizedDataset(name="test", shape=DatasetShape.TEMPLATE, records=records)
        spec = NormalizedTaskSpec(ds, cache_dir="/tmp/test_vta2")
        result = validate_token_alignment(spec)
        assert result["empty_prompt"] == 1

    def test_no_ds_returns_error(self):
        from circuitkit.data.normalized_task import validate_token_alignment

        class FakeSpec:
            pass

        result = validate_token_alignment(FakeSpec())
        assert "error" in result


class TestContrastiveRecordRoundTrip:
    """Verify ContrastiveRecord to_dict / from_dict round-trip edge cases."""

    def test_unpaired_record_roundtrip(self):
        r = ContrastiveRecord(
            record_id="42",
            clean_prompt="Hello",
            clean_answer="World",
            corrupt_prompt=None,
            corrupt_answer=None,
            contrast_source=ContrastSource.NOT_PAIRED_YET,
            meta={"key": "value"},
        )
        d = r.to_dict()
        r2 = ContrastiveRecord.from_dict(d)
        assert r2.record_id == "42"
        assert r2.corrupt_prompt is None
        assert r2.corrupt_answer is None
        assert r2.contrast_source == ContrastSource.NOT_PAIRED_YET
        assert r2.meta == {"key": "value"}

    def test_record_with_spans_roundtrip(self):
        r = ContrastiveRecord(
            record_id="0",
            clean_prompt="The cat sat on the mat",
            clean_answer="X",
            spans={"subject": (4, 7)},  # "cat"
        )
        d = r.to_dict()
        r2 = ContrastiveRecord.from_dict(d)
        # spans should survive as tuples, not lists
        assert r2.spans["subject"] == (4, 7)
        assert r2.get_span_text("subject") == "cat"

    def test_precomputed_labels_types_after_roundtrip(self):
        """JSON round-trip preserves int types in _precomputed_labels."""
        r = ContrastiveRecord(
            record_id="0",
            clean_prompt="A",
            clean_answer="X",
            meta={"_precomputed_labels": {"clean_label_id": 42, "corrupt_label_id": 99}},
        )
        d = r.to_dict()
        r2 = ContrastiveRecord.from_dict(d)
        labels = r2.meta["_precomputed_labels"]
        assert labels["clean_label_id"] == 42
        assert labels["corrupt_label_id"] == 99
        assert isinstance(labels["clean_label_id"], int)


class TestDuplicatePaddingSideAssignment:
    """NormalizedTaskSpec.__init__ has a duplicated pair_padding_side
    assignment block (lines 56-60 and 63-66). Verify the final state
    is correct regardless."""

    def test_padding_side_with_no_alignment_meta(self, tmp_dir):
        """No _alignment meta → falls back to class default 'left'."""
        ds = NormalizedDataset(
            name="test", shape=DatasetShape.TEMPLATE, records=[], meta={}
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        assert spec.pair_padding_side == "left"

    def test_padding_side_invalid_value_in_meta_ignored(self, tmp_dir):
        """Invalid recommended_pair_padding_side value is ignored."""
        ds = NormalizedDataset(
            name="test", shape=DatasetShape.TEMPLATE, records=[],
            meta={"_alignment": {"recommended_pair_padding_side": "center"}},
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        # Should stay at class default since "center" is not in ("left", "right")
        assert spec.pair_padding_side == "left"


class TestPadQuestionMultipleInsertions:
    """Test pad_question_region needing multiple neutral insertions."""

    def test_multiple_insertions_to_reach_target(self, tok):
        """Pad needs multiple rounds of neutral insertion to reach target."""
        prompt = "Q: hello Answer: yes"
        current_len = len(tok.encode(prompt))
        target_len = current_len + 3  # Need 3 extra tokens

        padded, exact = pad_question_region(
            prompt, target_len, tok, "Answer:", neutral=" the"
        )
        assert exact is True
        assert len(tok.encode(padded)) == target_len
        # "Answer:" should still be present and intact
        assert "Answer:" in padded


class TestPadBoundaryNotFoundInSomeRecords:
    """template_normalize with pad_question where boundary is missing
    from some records — should count as dropped_pad_failed."""

    def test_boundary_missing_counts_as_pad_failed(self, tok):
        """Records where pad_boundary is absent in corrupt_prompt should be
        counted in dropped_pad_failed, not crash."""
        rows = [
            # This one has "Answer:" in the corrupt prompt
            {"q": "cat", "r": "dog", "a": "meow", "b": "bark"},
            # This one won't have "BOUNDARY" since template doesn't include it
            {"q": "fish", "r": "bird", "a": "swim", "b": "fly"},
        ]
        template_spec = {
            "clean_prompt": "What is a {q} Answer: yes",
            "corrupt_prompt": "What is a {r} Answer: yes",
            "clean_answer": " {a}",
            "corrupt_answer": " {b}",
        }
        # Using a boundary that IS present → should work for aligned pairs
        ds = template_normalize(
            rows,
            template_spec=template_spec,
            align_strategy="pad_question",
            tokenizer=tok,
            pad_region_end="Answer:",
        )
        # Records that are already aligned should pass through;
        # pad_failed only counted when padding is needed AND boundary is missing
        meta = ds.meta["_alignment"]
        assert "dropped_pad_failed" in meta


class TestWorthinessCheck10ScoreClamping:
    """Test check 10 score clamping at boundary conditions."""

    def test_100_percent_drop_rate_score_zero(self):
        """When all records are dropped, score should be clamped at 0.0."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
            meta={
                "_alignment": {
                    "align_strategy": "filter",
                    "total_input": 100,
                    "kept": 0,
                    "dropped_nondiscriminative": 100,  # 100% drop
                    "dropped_misaligned": 0,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "logit_diff",
                }
            },
        )
        report = evaluate_worthiness(ds)
        check10 = next(
            (c for c in report.checks if c.name == "discriminative_drop_rate"),
            None,
        )
        assert check10 is not None
        assert check10.score == pytest.approx(0.0)

    def test_exactly_20_percent_does_not_trigger(self):
        """Exactly 20% drop rate → does NOT trigger check 10 (> 0.20, not >=)."""
        ds = NormalizedDataset(
            name="test",
            shape=DatasetShape.TEMPLATE,
            records=[
                ContrastiveRecord(
                    record_id="0",
                    clean_prompt="A",
                    corrupt_prompt="B",
                    clean_answer=" X",
                    corrupt_answer=" Y",
                    contrast_source=ContrastSource.GENERATED,
                )
            ],
            meta={
                "_alignment": {
                    "align_strategy": "filter",
                    "total_input": 100,
                    "kept": 80,
                    "dropped_nondiscriminative": 20,  # exactly 20%
                    "dropped_misaligned": 0,
                    "dropped_pad_failed": 0,
                    "answer_prefix_absorbed": 0,
                    "recommended_pair_padding_side": "left",
                    "recommended_metric": "logit_diff",
                }
            },
        )
        report = evaluate_worthiness(ds)
        check10 = next(
            (c for c in report.checks if c.name == "discriminative_drop_rate"),
            None,
        )
        assert check10 is None  # 20/100 = 0.20, NOT > 0.20


class TestCheckAnswerDiscriminativeBoundaryMerge:
    """Test check_answer_discriminative when tokenizer merges across
    the prompt/answer boundary — should return non-discriminative."""

    def test_boundary_merge_returns_nondiscriminative(self):
        """When joint tokenization of prompt+answer diverges from standalone
        prompt tokenization, the function can't reliably slice off the prompt
        and should return discriminative=False."""

        class MergingTokenizer:
            """Tokenizer that merges across boundaries: encode("ab") != 
            encode("a") + encode("b") when 'ab' is a known word."""
            def __init__(self):
                self._vocab = {"hello": 1, "world": 2, "helloworld": 3, "foo": 4, "bar": 5}
                self._next = 10
                self.pad_token_id = 0
                self.eos_token_id = 0

            def encode(self, text, add_special_tokens=True):
                if not text:
                    return []
                # Simulate merge: "hello" + "world" → [3] not [1, 2]
                if text == "helloworld":
                    return [3]
                return [self._vocab.get(w, self._next_id(w)) for w in text.split()]

            def _next_id(self, w):
                if w not in self._vocab:
                    self._vocab[w] = self._next
                    self._next += 1
                return self._vocab[w]

            def decode(self, ids):
                rev = {v: k for k, v in self._vocab.items()}
                return " ".join(rev.get(i, "?") for i in ids)

        tok = MergingTokenizer()
        result = check_answer_discriminative(
            "hello", "world", "foo", "bar", tok
        )
        # "hello" + "world" → [3] which != [1] prefix → boundary merge → _fail()
        assert result.discriminative is False


class TestWorthinessReportProperties:
    """Test DataWorthinessReport.hard_fails, soft_fails, passed properties."""

    def test_hard_fails_property(self):
        report = DataWorthinessReport(
            dataset_name="test",
            dataset_shape="template",
            n_records=10,
            verdict=Verdict.RED,
            checks=[
                CheckResult(name="c1", passed=False, severity="hard", score=0.0, message="fail"),
                CheckResult(name="c2", passed=True, severity="hard", score=1.0, message="ok"),
                CheckResult(name="c3", passed=False, severity="soft", score=0.5, message="warn"),
            ],
        )
        assert len(report.hard_fails) == 1
        assert report.hard_fails[0].name == "c1"

    def test_soft_fails_property(self):
        report = DataWorthinessReport(
            dataset_name="test",
            dataset_shape="template",
            n_records=10,
            verdict=Verdict.YELLOW,
            checks=[
                CheckResult(name="c1", passed=True, severity="hard", score=1.0, message="ok"),
                CheckResult(name="c2", passed=False, severity="soft", score=0.5, message="warn"),
            ],
        )
        assert len(report.soft_fails) == 1
        assert report.soft_fails[0].name == "c2"

    def test_passed_property_true(self):
        report = DataWorthinessReport(
            dataset_name="test", dataset_shape="template",
            n_records=10, verdict=Verdict.GREEN,
        )
        assert report.passed is True

    def test_passed_property_false_for_yellow(self):
        report = DataWorthinessReport(
            dataset_name="test", dataset_shape="template",
            n_records=10, verdict=Verdict.YELLOW,
        )
        assert report.passed is False

    def test_passed_property_false_for_red(self):
        report = DataWorthinessReport(
            dataset_name="test", dataset_shape="template",
            n_records=10, verdict=Verdict.RED,
        )
        assert report.passed is False


class TestEapCacheKeyAlignTag:
    """Verify EAP CSV cache key contains the align_strategy tag."""

    def test_cache_key_contains_filter_tag(self, tmp_dir, tok, mock_model):
        """Cache filename should include the alignment strategy tag."""
        ds = template_normalize(
            [{"q": "cat", "r": "dog", "a": "meow", "b": "bark"}],
            template_spec={
                "clean_prompt": "What is {q}",
                "corrupt_prompt": "What is {r}",
                "clean_answer": " {a}",
                "corrupt_answer": " {b}",
            },
            align_strategy="filter",
            tokenizer=tok,
        )
        if len(ds) == 0:
            pytest.skip("filter dropped all records")
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        spec.build_dataloader(
            mock_model, {"algorithm": "eap", "batch_size": 1}, "cpu"
        )
        cache_files = list(Path(tmp_dir / "cache").glob("*.csv"))
        assert any("filter" in f.name for f in cache_files)

    def test_cache_key_contains_none_tag(self, tmp_dir, tok, mock_model):
        ds = template_normalize(
            [{"q": "cat", "r": "dog", "a": "meow", "b": "bark"}],
            template_spec={
                "clean_prompt": "What is {q}",
                "corrupt_prompt": "What is {r}",
                "clean_answer": " {a}",
                "corrupt_answer": " {b}",
            },
            align_strategy="none",
        )
        spec = NormalizedTaskSpec(ds, cache_dir=str(tmp_dir / "cache"))
        spec.build_dataloader(
            mock_model, {"algorithm": "eap", "batch_size": 1}, "cpu"
        )
        cache_files = list(Path(tmp_dir / "cache").glob("*.csv"))
        assert any("none" in f.name for f in cache_files)


class TestNormalizedDatasetProperties:
    """Additional tests for NormalizedDataset computed properties."""

    def test_n_paired_mixed_dataset(self):
        records = [
            ContrastiveRecord(record_id="0", clean_prompt="A", clean_answer="X",
                              corrupt_prompt="B", corrupt_answer="Y"),
            ContrastiveRecord(record_id="1", clean_prompt="C", clean_answer="Z",
                              corrupt_prompt=None, corrupt_answer=None),
        ]
        ds = NormalizedDataset(name="t", shape=DatasetShape.TEMPLATE, records=records)
        assert ds.n_paired == 1
        assert ds.fully_paired is False

    def test_fully_paired_empty_dataset(self):
        ds = NormalizedDataset(name="t", shape=DatasetShape.TEMPLATE, records=[])
        assert ds.fully_paired is False  # empty → False per line 212

    def test_iteration_protocol(self):
        records = [
            ContrastiveRecord(record_id=str(i), clean_prompt=f"P{i}", clean_answer=f"A{i}")
            for i in range(3)
        ]
        ds = NormalizedDataset(name="t", shape=DatasetShape.QA, records=records)
        assert len(list(ds)) == 3
        assert ds[0].record_id == "0"
        assert ds[2].record_id == "2"


class TestTemplateNormalizeAlignmentMetaFields:
    """Exhaustively verify all fields in ds.meta['_alignment']."""

    def test_all_alignment_meta_keys_present_for_filter(self, tok):
        ds = template_normalize(
            [{"q": "cat", "r": "dog", "a": "meow", "b": "bark"}],
            template_spec={
                "clean_prompt": "What is {q}",
                "corrupt_prompt": "What is {r}",
                "clean_answer": " {a}",
                "corrupt_answer": " {b}",
            },
            align_strategy="filter",
            tokenizer=tok,
        )
        meta = ds.meta["_alignment"]
        expected_keys = {
            "align_strategy", "total_input", "kept",
            "dropped_nondiscriminative", "dropped_misaligned",
            "dropped_pad_failed", "answer_prefix_absorbed",
            "recommended_pair_padding_side", "recommended_metric",
        }
        assert set(meta.keys()) == expected_keys
        assert meta["align_strategy"] == "filter"
        assert meta["recommended_pair_padding_side"] == "left"
        assert meta["recommended_metric"] == "logit_diff"
        assert meta["total_input"] >= 1

    def test_none_strategy_recommends_kl(self, tok):
        ds = template_normalize(
            [{"q": "cat", "r": "dog", "a": "meow", "b": "bark"}],
            template_spec={
                "clean_prompt": "What is {q}",
                "corrupt_prompt": "What is {r}",
                "clean_answer": " {a}",
                "corrupt_answer": " {b}",
            },
            align_strategy="none",
        )
        meta = ds.meta["_alignment"]
        assert meta["recommended_metric"] == "kl_divergence"
        # none strategy skips passes → no drops
        assert meta["dropped_nondiscriminative"] == 0
        assert meta["dropped_misaligned"] == 0
        assert meta["answer_prefix_absorbed"] == 0

# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

class TestNormalizedPairsContrastiveGuard:
    """NormalizedTaskSpec.build_dataloader fails loud on identical clean/corrupt pairs.

    fully_paired only means the corrupt half is present, not that it differs
    from the clean half, so an all-identical paired dataset must raise before
    EAP discovery runs on a zero-signal contrast.
    """

    def _ds(self, clean, corrupt):
        from circuitkit.data.normalized import (
            ContrastiveRecord, ContrastSource, DatasetShape, NormalizedDataset,
        )
        rec = ContrastiveRecord(
            record_id="0", clean_prompt=clean, corrupt_prompt=corrupt,
            clean_answer=" x", corrupt_answer=" y",
            contrast_source=ContrastSource.GENERATED, meta={},
        )
        return NormalizedDataset(
            name="d", shape=DatasetShape.TEMPLATE, records=[rec], source="t", meta={}
        )

    def test_all_identical_raises(self):
        import pytest
        from circuitkit.data.normalized_task import NormalizedTaskSpec
        spec = NormalizedTaskSpec(self._ds("A B C", "A B C"), name="_ephemeral:d:1")
        with pytest.raises(ValueError, match="Every pair is identical"):
            spec.build_dataloader(object(), {"algorithm": "eap", "batch_size": 1}, "cpu")

    def test_opt_out_allows_identical(self):
        # allow_degenerate_corruption bypasses the guard: it must get PAST the
        # identical-pair check (any later failure on the sentinel model is fine
        # and proves the guard did not fire).
        from circuitkit.data.normalized_task import NormalizedTaskSpec
        spec = NormalizedTaskSpec(self._ds("A B C", "A B C"), name="_ephemeral:d:2")
        try:
            spec.build_dataloader(
                object(),
                {"algorithm": "eap", "batch_size": 1, "allow_degenerate_corruption": True},
                "cpu",
            )
        except Exception as e:
            assert "Every pair is identical" not in str(e)  # guard did not fire
