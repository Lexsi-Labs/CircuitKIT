"""
Tests for DoubleIO Task — focused on Pillar 6 (Generalization) target-task usage.

Validates that DoubleIO produces correct data for evaluate_graph() in Pillar 6,
which expects an EAP-format dataloader yielding (clean, corrupted, labels) batches.

Run from: circuitkit/tests/unit/
    pytest test_double_io.py -v
    pytest test_double_io.py -v -k "test_name_counts"  # run one test

These tests do NOT require a GPU or a loaded model — they test the data
generation and TaskSpec layers using mocks where needed.
"""

import re
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset_module():
    """Import the dataset module."""
    from circuitkit.data.task_data.tasks.double_io.double_io_dataset import (
        DOUBLE_IO_ABBA_TEMPLATES,
        DOUBLE_IO_BABA_TEMPLATES,
        IO_FILLER_CLAUSES,
        NAMES,
        OBJECTS,
        PLACES,
        gen_double_io_corrupted_prompts,
        gen_double_io_prompts,
        get_double_io_data_only,
    )

    return {
        "NAMES": NAMES,
        "PLACES": PLACES,
        "OBJECTS": OBJECTS,
        "IO_FILLER_CLAUSES": IO_FILLER_CLAUSES,
        "BABA": DOUBLE_IO_BABA_TEMPLATES,
        "ABBA": DOUBLE_IO_ABBA_TEMPLATES,
        "gen_prompts": gen_double_io_prompts,
        "gen_corrupted": gen_double_io_corrupted_prompts,
        "get_data": get_double_io_data_only,
    }


@pytest.fixture
def taskspec():
    """Import and instantiate the DoubleIOTaskSpec."""
    from circuitkit.tasks.builtins.double_io import DoubleIOTaskSpec

    return DoubleIOTaskSpec()


@pytest.fixture
def sample_prompts(dataset_module):
    """Generate a small batch of DoubleIO prompts for testing."""
    return dataset_module["gen_prompts"](
        templates=dataset_module["ABBA"],
        names=dataset_module["NAMES"],
        nouns_dict={"[PLACE]": dataset_module["PLACES"], "[OBJECT]": dataset_module["OBJECTS"]},
        filler_clauses=dataset_module["IO_FILLER_CLAUSES"],
        N=20,
        seed=42,
    )


@pytest.fixture
def mock_model():
    """
    Create a mock HookedTransformer that behaves enough for data generation.

    to_tokens: returns a tensor of sequential ints (simulating token IDs)
    to_string: returns a placeholder string
    tokenizer.pad_token_id: 0
    cfg.model_name: "test_model"
    """
    model = MagicMock()
    model.cfg.model_name = "test_model"
    model.tokenizer.pad_token_id = 0

    def fake_to_tokens(text, prepend_bos=False):
        """Simulate tokenization: 1 token per word, IDs are hashes mod 50000."""
        words = text.strip().split()
        ids = [abs(hash(w)) % 50000 + 1 for w in words]  # +1 to avoid pad=0
        return torch.tensor([ids], dtype=torch.long)

    def fake_to_string(token_tensor):
        """Simulate detokenization."""
        return " ".join([f"tok_{t.item()}" for t in token_tensor if t.item() != 0])

    model.to_tokens = fake_to_tokens
    model.to_string = fake_to_string

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Template Structure Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTemplateStructure:
    """Verify the DoubleIO templates have correct placeholder structure."""

    def test_baba_templates_have_io_filler(self, dataset_module):
        """Every BABA template must contain the [IO_FILLER] placeholder."""
        for i, tmpl in enumerate(dataset_module["BABA"]):
            assert "[IO_FILLER]" in tmpl, f"BABA template {i} missing [IO_FILLER]"

    def test_abba_templates_have_io_filler(self, dataset_module):
        """Every ABBA template must contain the [IO_FILLER] placeholder."""
        for i, tmpl in enumerate(dataset_module["ABBA"]):
            assert "[IO_FILLER]" in tmpl, f"ABBA template {i} missing [IO_FILLER]"

    def test_baba_abba_same_count(self, dataset_module):
        """BABA and ABBA template lists must have the same length."""
        assert len(dataset_module["BABA"]) == len(dataset_module["ABBA"])

    def test_templates_have_required_placeholders(self, dataset_module):
        """Every template must have [A], [B], [PLACE] or [OBJECT], and [IO_FILLER]."""
        for label, templates in [
            ("BABA", dataset_module["BABA"]),
            ("ABBA", dataset_module["ABBA"]),
        ]:
            for i, tmpl in enumerate(templates):
                assert "[A]" in tmpl, f"{label} template {i} missing [A]"
                assert "[B]" in tmpl, f"{label} template {i} missing [B]"
                assert "[IO_FILLER]" in tmpl, f"{label} template {i} missing [IO_FILLER]"

    def test_baba_b_before_a_in_first_clause(self, dataset_module):
        """In BABA templates, [B] should appear before [A] in the first clause."""
        for i, tmpl in enumerate(dataset_module["BABA"]):
            filler_pos = tmpl.index("[IO_FILLER]")
            first_clause = tmpl[:filler_pos]
            first_b = first_clause.find("[B]")
            first_a = first_clause.find("[A]")
            if first_b >= 0 and first_a >= 0:
                assert (
                    first_b < first_a
                ), f"BABA template {i}: [B] should come before [A] in first clause"

    def test_abba_a_before_b_in_first_clause(self, dataset_module):
        """In ABBA templates, [A] should appear before [B] in the first clause."""
        for i, tmpl in enumerate(dataset_module["ABBA"]):
            filler_pos = tmpl.index("[IO_FILLER]")
            first_clause = tmpl[:filler_pos]
            first_a = first_clause.find("[A]")
            first_b = first_clause.find("[B]")
            if first_a >= 0 and first_b >= 0:
                assert (
                    first_a < first_b
                ), f"ABBA template {i}: [A] should come before [B] in first clause"

    def test_filler_clauses_contain_io_placeholder(self, dataset_module):
        """Every filler clause must contain [IO]."""
        for i, clause in enumerate(dataset_module["IO_FILLER_CLAUSES"]):
            assert "[IO]" in clause, f"Filler clause {i} missing [IO]: {clause}"

    def test_no_residual_placeholders_after_generation(self, sample_prompts):
        """Generated prompts should have no unresolved placeholders."""
        placeholders = ["[A]", "[B]", "[IO]", "[PLACE]", "[OBJECT]", "[IO_FILLER]"]
        for i, prompt in enumerate(sample_prompts):
            for ph in placeholders:
                assert (
                    ph not in prompt["text"]
                ), f"Prompt {i} still contains placeholder {ph}: {prompt['text']}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Name Count Tests (Core DoubleIO invariant)
# ─────────────────────────────────────────────────────────────────────────────


class TestNameCounts:
    """
    The defining property of DoubleIO: both IO and S appear exactly 2x
    in the INPUT (i.e., excluding the final answer token).

    In the full prompt text, the final "[A]" is the answer token.
    When we strip the last word (simulating label removal), both names
    should appear exactly twice.
    """

    def test_io_appears_twice_in_input(self, sample_prompts):
        """IO name must appear exactly 2x in the input (prompt minus last word)."""
        for i, prompt in enumerate(sample_prompts):
            text = prompt["text"]
            io_name = prompt["IO"]
            # Remove the last word (the answer token = IO name at end)
            words = text.split()
            input_text = " ".join(words[:-1])
            count = self._count_name(input_text, io_name)
            assert count == 2, (
                f"Prompt {i}: IO '{io_name}' appears {count}x in input "
                f"(expected 2). Input: {input_text}"
            )

    def test_s_appears_twice_in_input(self, sample_prompts):
        """S name must appear exactly 2x in the input (prompt minus last word)."""
        for i, prompt in enumerate(sample_prompts):
            text = prompt["text"]
            s_name = prompt["S"]
            words = text.split()
            input_text = " ".join(words[:-1])
            count = self._count_name(input_text, s_name)
            assert count == 2, (
                f"Prompt {i}: S '{s_name}' appears {count}x in input "
                f"(expected 2). Input: {input_text}"
            )

    def test_io_is_last_word(self, sample_prompts):
        """The final word of every prompt must be the IO name (the answer)."""
        for i, prompt in enumerate(sample_prompts):
            last_word = prompt["text"].split()[-1]
            assert (
                last_word == prompt["IO"]
            ), f"Prompt {i}: last word is '{last_word}', expected IO='{prompt['IO']}'"

    def test_io_and_s_are_distinct(self, sample_prompts):
        """IO and S names must be different for every prompt."""
        for i, prompt in enumerate(sample_prompts):
            assert (
                prompt["IO"] != prompt["S"]
            ), f"Prompt {i}: IO and S are the same name '{prompt['IO']}'"

    @staticmethod
    def _count_name(text, name):
        """Count standalone occurrences of `name` in `text`, ignoring punctuation."""
        # Match name as a whole word, allowing trailing punctuation
        pattern = r"\b" + re.escape(name) + r"\b"
        return len(re.findall(pattern, text))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Corruption Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCorruption:
    """Verify the S2→RAND corruption produces valid corrupted prompts."""

    def test_corruption_changes_s2(self, dataset_module, sample_prompts):
        """The corrupted version should have a different name at the S2 position."""
        corrupted = dataset_module["gen_corrupted"](
            sample_prompts, dataset_module["NAMES"], seed=99
        )
        changed_count = 0
        for clean, corrupt in zip(sample_prompts, corrupted):
            if clean["text"] != corrupt["text"]:
                changed_count += 1
        # At least 90% should be changed (some edge cases might not)
        assert (
            changed_count >= len(sample_prompts) * 0.9
        ), f"Only {changed_count}/{len(sample_prompts)} prompts were corrupted"

    def test_corruption_preserves_io_name(self, dataset_module, sample_prompts):
        """Corruption should NOT change the IO name in the prompt."""
        corrupted = dataset_module["gen_corrupted"](
            sample_prompts, dataset_module["NAMES"], seed=99
        )
        for i, (clean, corrupt) in enumerate(zip(sample_prompts, corrupted)):
            io_name = clean["IO"]
            clean_io_count = clean["text"].count(io_name)
            corrupt_io_count = corrupt["text"].count(io_name)
            assert clean_io_count == corrupt_io_count, (
                f"Prompt {i}: IO '{io_name}' count changed from "
                f"{clean_io_count} to {corrupt_io_count} after corruption"
            )

    def test_corruption_replaces_with_different_name(self, dataset_module, sample_prompts):
        """The replacement name must differ from both IO and S."""
        corrupted = dataset_module["gen_corrupted"](
            sample_prompts, dataset_module["NAMES"], seed=99
        )
        for i, (clean, corrupt) in enumerate(zip(sample_prompts, corrupted)):
            if clean["text"] == corrupt["text"]:
                continue  # skip unchanged (edge case)
            # Find what replaced S2
            clean_words = clean["text"].split()
            corrupt_words = corrupt["text"].split()
            for cw, crw in zip(clean_words, corrupt_words):
                if cw != crw:
                    new_name = crw.rstrip(".,!?;:")
                    assert (
                        new_name != clean["IO"]
                    ), f"Prompt {i}: replacement name is same as IO '{clean['IO']}'"
                    assert (
                        new_name != clean["S"]
                    ), f"Prompt {i}: replacement name is same as S '{clean['S']}'"
                    break

    def test_corruption_output_same_length(self, dataset_module, sample_prompts):
        """Corrupted prompts list must have the same length as input."""
        corrupted = dataset_module["gen_corrupted"](
            sample_prompts, dataset_module["NAMES"], seed=99
        )
        assert len(corrupted) == len(sample_prompts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Seed Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


class TestReproducibility:
    """Same seed must produce identical data."""

    def test_prompt_generation_deterministic(self, dataset_module):
        """Two calls with the same seed must produce identical prompts."""
        kwargs = dict(
            templates=dataset_module["ABBA"],
            names=dataset_module["NAMES"],
            nouns_dict={"[PLACE]": dataset_module["PLACES"], "[OBJECT]": dataset_module["OBJECTS"]},
            filler_clauses=dataset_module["IO_FILLER_CLAUSES"],
            N=50,
            seed=777,
        )
        prompts_a = dataset_module["gen_prompts"](**kwargs)
        prompts_b = dataset_module["gen_prompts"](**kwargs)
        for i, (a, b) in enumerate(zip(prompts_a, prompts_b)):
            assert a["text"] == b["text"], f"Prompt {i} differs across runs"
            assert a["IO"] == b["IO"]
            assert a["S"] == b["S"]

    def test_different_seeds_produce_different_data(self, dataset_module):
        """Different seeds must produce different prompts."""
        kwargs = dict(
            templates=dataset_module["ABBA"],
            names=dataset_module["NAMES"],
            nouns_dict={"[PLACE]": dataset_module["PLACES"], "[OBJECT]": dataset_module["OBJECTS"]},
            filler_clauses=dataset_module["IO_FILLER_CLAUSES"],
            N=20,
        )
        prompts_a = dataset_module["gen_prompts"](**kwargs, seed=1)
        prompts_b = dataset_module["gen_prompts"](**kwargs, seed=2)
        texts_a = [p["text"] for p in prompts_a]
        texts_b = [p["text"] for p in prompts_b]
        assert texts_a != texts_b, "Different seeds produced identical prompts"

    def test_corruption_deterministic(self, dataset_module, sample_prompts):
        """Same seed for corruption must produce identical results."""
        c1 = dataset_module["gen_corrupted"](sample_prompts, dataset_module["NAMES"], seed=42)
        c2 = dataset_module["gen_corrupted"](sample_prompts, dataset_module["NAMES"], seed=42)
        for i, (a, b) in enumerate(zip(c1, c2)):
            assert a["text"] == b["text"], f"Corruption {i} differs across runs"


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_double_io_data_only Tests (with mock model)
# ─────────────────────────────────────────────────────────────────────────────


class TestGetDoubleIODataOnly:
    """Test the main data generation function that produces tensors."""

    def test_output_keys(self, dataset_module, mock_model):
        """Output dict must have all required keys."""
        result = dataset_module["get_data"](
            num_examples=10, device="cpu", model=mock_model, seed=42
        )
        required_keys = {
            "validation_data",
            "validation_patch_data",
            "validation_labels",
            "validation_wrong_labels",
            "end_idxs",
        }
        assert required_keys.issubset(
            result.keys()
        ), f"Missing keys: {required_keys - set(result.keys())}"

    def test_output_shapes(self, dataset_module, mock_model):
        """All output tensors must have consistent batch dimension."""
        N = 15
        result = dataset_module["get_data"](num_examples=N, device="cpu", model=mock_model, seed=42)
        assert result["validation_data"].shape[0] == N
        assert result["validation_patch_data"].shape[0] == N
        assert result["validation_labels"].shape[0] == N
        assert result["validation_wrong_labels"].shape[0] == N
        assert result["end_idxs"].shape[0] == N

    def test_validation_data_and_patch_same_seq_len(self, dataset_module, mock_model):
        """Clean and corrupted tensors must have the same sequence length."""
        result = dataset_module["get_data"](
            num_examples=10, device="cpu", model=mock_model, seed=42
        )
        assert result["validation_data"].shape[1] == result["validation_patch_data"].shape[1]

    def test_labels_are_1d(self, dataset_module, mock_model):
        """Labels and wrong_labels must be 1-D tensors (not 2-D)."""
        result = dataset_module["get_data"](
            num_examples=10, device="cpu", model=mock_model, seed=42
        )
        assert result["validation_labels"].ndim == 1
        assert result["validation_wrong_labels"].ndim == 1

    def test_end_idxs_within_bounds(self, dataset_module, mock_model):
        """end_idxs must be valid indices into validation_data."""
        result = dataset_module["get_data"](
            num_examples=10, device="cpu", model=mock_model, seed=42
        )
        seq_len = result["validation_data"].shape[1]
        assert (result["end_idxs"] >= 0).all()
        assert (result["end_idxs"] < seq_len).all()

    def test_labels_differ_from_wrong_labels(self, dataset_module, mock_model):
        """Correct and incorrect labels must differ for each example."""
        result = dataset_module["get_data"](
            num_examples=20, device="cpu", model=mock_model, seed=42
        )
        # They should differ in most cases (IO != S)
        differ_count = (
            (result["validation_labels"] != result["validation_wrong_labels"]).sum().item()
        )
        assert differ_count >= 18, f"Only {differ_count}/20 labels differ from wrong_labels"

    def test_prompt_type_baba(self, dataset_module, mock_model):
        """BABA prompt type should work without error."""
        result = dataset_module["get_data"](
            num_examples=5, device="cpu", model=mock_model, seed=42, prompt_type="BABA"
        )
        assert result["validation_data"].shape[0] == 5

    def test_prompt_type_mixed(self, dataset_module, mock_model):
        """Mixed prompt type should work without error."""
        result = dataset_module["get_data"](
            num_examples=5, device="cpu", model=mock_model, seed=42, prompt_type="mixed"
        )
        assert result["validation_data"].shape[0] == 5

    def test_invalid_prompt_type_raises(self, dataset_module, mock_model):
        """Invalid prompt type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown prompt_type"):
            dataset_module["get_data"](
                num_examples=5, device="cpu", model=mock_model, seed=42, prompt_type="XYZ"
            )

    def test_model_none_raises(self, dataset_module):
        """model=None should raise ValueError."""
        with pytest.raises(ValueError, match="Model required"):
            dataset_module["get_data"](num_examples=5, device="cpu", model=None, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TaskSpec Interface Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTaskSpecInterface:
    """Verify DoubleIOTaskSpec implements the TaskSpec protocol correctly."""

    def test_name_attribute(self, taskspec):
        assert taskspec.name == "double_io"

    def test_pair_padding_side(self, taskspec):
        assert taskspec.pair_padding_side == "right"

    def test_validate_config_accepts_eap(self, taskspec):
        """EAP should be accepted."""
        taskspec.validate_discovery_config({"algorithm": "eap", "level": "node"})

    def test_validate_config_accepts_eap_ig(self, taskspec):
        """EAP-IG should be accepted."""
        taskspec.validate_discovery_config({"algorithm": "eap-ig", "level": "neuron"})

    def test_validate_config_accepts_ibcircuit(self, taskspec):
        """IBCircuit should be accepted."""
        taskspec.validate_discovery_config({"algorithm": "ibcircuit", "level": "node"})

    def test_validate_config_rejects_unknown_algo(self, taskspec):
        """Unknown algorithms should raise ValueError."""
        with pytest.raises(ValueError, match="does not support"):
            taskspec.validate_discovery_config({"algorithm": "unknown_algo", "level": "node"})

    def test_validate_config_rejects_bad_level(self, taskspec):
        """Invalid level should raise ValueError."""
        with pytest.raises(ValueError, match="level"):
            taskspec.validate_discovery_config({"algorithm": "eap", "level": "edge"})

    def test_validate_config_rejects_bad_batch_size(self, taskspec):
        """Non-positive batch_size should raise ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            taskspec.validate_discovery_config(
                {"algorithm": "eap", "level": "node", "batch_size": -1}
            )

    def test_validate_config_accepts_no_batch_size(self, taskspec):
        """Missing batch_size should be fine (uses default)."""
        taskspec.validate_discovery_config({"algorithm": "eap", "level": "node"})

    def test_metric_fn_returns_callable(self, taskspec):
        """metric_fn() should return a callable."""
        metric = taskspec.metric_fn()
        assert callable(metric)

    def test_metric_fn_logit_diff_default(self, taskspec):
        """Default metric should be logit_diff (a partial)."""
        metric = taskspec.metric_fn()
        assert isinstance(metric, partial)
        assert metric.func.__name__ == "_logit_diff"

    def test_metric_fn_kl(self, taskspec):
        """KL metric type should work."""
        metric = taskspec.metric_fn("kl")
        assert callable(metric)

    def test_metric_fn_unknown_raises(self, taskspec):
        """Unknown metric type should raise ValueError."""
        with pytest.raises(ValueError, match="does not support metric_type"):
            taskspec.metric_fn("mse")

    def test_metric_fn_no_args_works(self, taskspec):
        """metric_fn() with no args must work — this is how api.py calls it."""
        metric = taskspec.metric_fn()
        assert callable(metric)

    def test_artifact_metadata_keys(self, taskspec):
        """artifact_metadata should return expected keys."""
        meta = taskspec.artifact_metadata({"algorithm": "eap", "level": "node"})
        assert meta["task"] == "double_io"
        assert "algorithm" in meta
        assert "level" in meta

    def test_build_dataloader_rejects_none_model(self, taskspec):
        """build_dataloader with model=None should raise ValueError."""
        with pytest.raises(ValueError):
            taskspec.build_dataloader(
                model=None,
                discovery_cfg={"algorithm": "eap", "data_params": {}},
                device="cpu",
            )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Metric Compatibility Tests (IOI ↔ DoubleIO)
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricCompatibility:
    """
    Pillar 6's transfer_ratio = target_score / source_score.
    This only makes sense if both tasks use the same metric signature.
    Verify that DoubleIO's metric is call-compatible with IOI's.
    """

    def _make_fake_inputs(self, batch_size=4, seq_len=16, vocab_size=100):
        """Create fake metric inputs matching the EAP metric signature."""
        logits = torch.randn(batch_size, seq_len, vocab_size)
        clean_logits = torch.randn(batch_size, seq_len, vocab_size)
        input_length = torch.full((batch_size,), seq_len, dtype=torch.long)
        labels = torch.randint(0, vocab_size, (batch_size, 2))
        return logits, clean_logits, input_length, labels

    def test_logit_diff_runs_without_error(self, taskspec):
        """DoubleIO logit_diff metric should execute on fake inputs."""
        metric = taskspec.metric_fn("logit_diff")
        logits, clean_logits, input_length, labels = self._make_fake_inputs()
        result = metric(logits, clean_logits, input_length, labels)
        assert isinstance(result, torch.Tensor)

    def test_logit_diff_scalar_output(self, taskspec):
        """With mean=True (default), metric should return a scalar."""
        metric = taskspec.metric_fn("logit_diff")
        logits, clean_logits, input_length, labels = self._make_fake_inputs()
        result = metric(logits, clean_logits, input_length, labels)
        assert result.ndim == 0, f"Expected scalar, got shape {result.shape}"

    def test_logit_diff_per_sample_output(self):
        """With mean=False, metric should return per-sample scores."""
        from circuitkit.tasks.builtins.double_io import DoubleIOTaskSpec

        # Call the raw static method with mean=False
        logits, clean_logits, input_length, labels = self._make_fake_inputs(batch_size=8)
        result = DoubleIOTaskSpec._logit_diff(
            logits, clean_logits, input_length, labels, mean=False, loss=False
        )
        assert result.shape == (8,), f"Expected (8,), got {result.shape}"

    def test_make_eval_metric_compatibility(self, taskspec):
        """
        Simulate what api.py's _make_eval_metric does:
        1. Call task_spec.metric_fn() → get partial with loss=True, mean=True
        2. Override to loss=False, mean=False for per-sample eval
        3. Call the result on fake inputs

        This must work identically for both IOI and DoubleIO.
        """
        base = taskspec.metric_fn()
        assert isinstance(base, partial)
        # Simulate _make_eval_metric
        kw = base.keywords.copy()
        kw["loss"] = False
        kw["mean"] = False
        eval_metric = partial(base.func, **kw)

        logits, clean_logits, input_length, labels = self._make_fake_inputs(batch_size=6)
        result = eval_metric(logits, clean_logits, input_length, labels)
        assert result.shape == (6,), f"Expected (6,), got {result.shape}"

    def test_ioi_and_double_io_metrics_have_same_signature(self):
        """
        Both IOI and DoubleIO metrics must accept the same args.
        If IOI is available, verify signature compatibility.
        """
        try:
            from circuitkit.tasks.builtins.ioi import IOITaskSpec

            ioi_spec = IOITaskSpec()
            ioi_metric = ioi_spec.metric_fn()

            from circuitkit.tasks.builtins.double_io import DoubleIOTaskSpec

            dio_spec = DoubleIOTaskSpec()
            dio_metric = dio_spec.metric_fn()

            # Both should be partials with the same keyword keys
            assert isinstance(ioi_metric, partial)
            assert isinstance(dio_metric, partial)
            assert set(ioi_metric.keywords.keys()) == set(dio_metric.keywords.keys()), (
                f"IOI keywords: {ioi_metric.keywords.keys()}, "
                f"DoubleIO keywords: {dio_metric.keywords.keys()}"
            )

            # Both should accept the same positional args
            logits, clean_logits, input_length, labels = self._make_fake_inputs()
            ioi_result = ioi_metric(logits, clean_logits, input_length, labels)
            dio_result = dio_metric(logits, clean_logits, input_length, labels)
            assert ioi_result.shape == dio_result.shape
        except ImportError:
            pytest.skip("IOI TaskSpec not available for cross-comparison")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_example(self, dataset_module, mock_model):
        """N=1 should work without error."""
        result = dataset_module["get_data"](num_examples=1, device="cpu", model=mock_model, seed=42)
        assert result["validation_data"].shape[0] == 1

    def test_large_batch(self, dataset_module, mock_model):
        """Large N should work and produce correct count."""
        result = dataset_module["get_data"](
            num_examples=200, device="cpu", model=mock_model, seed=42
        )
        assert result["validation_data"].shape[0] == 200

    def test_all_names_in_pool_are_single_word(self, dataset_module):
        """All names must be single whitespace-delimited tokens."""
        for name in dataset_module["NAMES"]:
            assert " " not in name, f"Name '{name}' contains spaces"

    def test_filler_clauses_end_with_period(self, dataset_module):
        """Filler clauses should end with punctuation for clean sentence breaks."""
        for i, clause in enumerate(dataset_module["IO_FILLER_CLAUSES"]):
            assert clause.endswith("."), f"Filler clause {i} doesn't end with period: '{clause}'"

    def test_prompt_ends_with_io_name_not_punctuation(self, sample_prompts):
        """The final token must be the raw IO name (no trailing punctuation)."""
        for i, prompt in enumerate(sample_prompts):
            last_word = prompt["text"].split()[-1]
            assert last_word[
                -1
            ].isalpha(), f"Prompt {i}: last word '{last_word}' ends with non-alpha character"

    def test_symmetric_generation(self, dataset_module):
        """Symmetric mode should produce paired prompts with swapped IO/S."""
        prompts = dataset_module["gen_prompts"](
            templates=dataset_module["BABA"][:2],
            names=dataset_module["NAMES"],
            nouns_dict={"[PLACE]": dataset_module["PLACES"], "[OBJECT]": dataset_module["OBJECTS"]},
            filler_clauses=dataset_module["IO_FILLER_CLAUSES"],
            N=10,
            symmetric=True,
            seed=42,
        )
        assert len(prompts) == 10
        # Check that pairs have swapped IO/S
        for idx in range(0, len(prompts) - 1, 2):
            p1 = prompts[idx]
            p2 = prompts[idx + 1]
            assert (
                p1["IO"] == p2["S"]
            ), f"Pair {idx}: p1.IO='{p1['IO']}' should equal p2.S='{p2['S']}'"
            assert (
                p1["S"] == p2["IO"]
            ), f"Pair {idx}: p1.S='{p1['S']}' should equal p2.IO='{p2['IO']}'"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Pillar 6 Integration Smoke Test
# ─────────────────────────────────────────────────────────────────────────────


class TestPillar6Integration:
    """
    Simulate the Pillar 6 data flow without loading a real model.

    Pillar 6 calls:
      1. target_task_spec.build_dataloader(model, dl_cfg, device)
      2. evaluate_graph(model, graph, dataloader=target_dl, ...)

    evaluate_graph expects batches of (clean, corrupted, label).
    We test that DoubleIO's build_dataloader would produce this format.
    """

    def test_dl_cfg_has_algorithm_eap(self, taskspec):
        """
        In evaluate_circuit, dl_cfg always sets algorithm='eap'.
        Verify validate_discovery_config accepts this.
        """
        dl_cfg = {
            "algorithm": "eap",
            "level": "node",
            "data_params": {"num_examples": 10, "seed": 42},
            "batch_size": 4,
        }
        # Should not raise
        taskspec.validate_discovery_config(dl_cfg)

    def test_build_dataloader_produces_iterable(self, taskspec, mock_model, tmp_path):
        """
        build_dataloader should return something iterable that yields batches.
        We patch the cache dir to use tmp_path for isolation.
        """
        dl_cfg = {
            "algorithm": "eap",
            "level": "node",
            "data_params": {
                "num_examples": 8,
                "seed": 42,
                "cache_dir": str(tmp_path / "cache"),
            },
            "batch_size": 4,
        }
        dl = taskspec.build_dataloader(mock_model, dl_cfg, "cpu")
        assert hasattr(dl, "__iter__"), "Dataloader must be iterable"
        assert hasattr(dl, "pair_padding_side"), "Dataloader must have pair_padding_side"
        assert dl.pair_padding_side == "right"

    def test_build_dataloader_caching(self, taskspec, mock_model, tmp_path):
        """Second call with same params should load from cache."""
        cache_dir = str(tmp_path / "cache")
        dl_cfg = {
            "algorithm": "eap",
            "level": "node",
            "data_params": {
                "num_examples": 5,
                "seed": 42,
                "cache_dir": cache_dir,
            },
            "batch_size": 2,
        }
        # First call: generates and caches
        taskspec.build_dataloader(mock_model, dl_cfg, "cpu")

        # Verify CSV was written
        cache_path = Path(cache_dir)
        csv_files = list(cache_path.glob("double_io_*.csv"))
        assert len(csv_files) == 1, f"Expected 1 cache file, found {len(csv_files)}"

        # Second call: should load from cache (no generation)
        dl2 = taskspec.build_dataloader(mock_model, dl_cfg, "cpu")
        assert hasattr(dl2, "__iter__")

    def test_task_registered_in_registry(self):
        """After bootstrap, 'double_io' should be findable via get_task."""
        try:
            from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks

            _bootstrap_builtin_tasks()
            from circuitkit.tasks.registry import get_task

            spec = get_task("double_io")
            assert spec.name == "double_io"
        except ImportError:
            pytest.skip("Full circuitkit not available for registry test")
        except ValueError:
            pytest.fail("double_io not registered. Did you add DoubleIOTaskSpec to bootstrap.py?")
