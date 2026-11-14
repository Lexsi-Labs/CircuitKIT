"""
Test suite for the multilingual_refusal task.

Uses a whitespace tokenizer and a fake model, so nothing here loads a real model:
- bundled English pairs and user-supplied pairs in other languages
- chat-template rendering, and the BOS handling that goes with it
- token-length filtering of pairs, and the answer-token contract
- the discovery / eval partition, the cache key, the metric and config validation
"""

import csv
import types

import pandas as pd
import pytest
import torch as t

import circuitkit
from circuitkit.tasks.builtins.multilingual_refusal import (
    DEFAULT_OPENERS,
    MultilingualRefusalTaskSpec,
    load_pairs,
)

TEMPLATE = "<bos> <user> {} </user> <assistant>"


class FakeTokenizer:
    """Whitespace tokenizer with stable ids and an optional chat template."""

    def __init__(self, chat_template="{{ messages }}"):
        self.chat_template = chat_template
        self.vocab = {}

    def encode(self, text, add_special_tokens=True):
        return [self.vocab.setdefault(word, len(self.vocab)) for word in text.split()]

    def decode(self, ids, **kwargs):
        reverse = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse[i] for i in ids)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    ):
        # Like Qwen3's template: with the switch off the assistant turn opens already closed.
        text = TEMPLATE.format(messages[0]["content"])
        return text if enable_thinking else text + " <think> </think>"


class FakeModel:
    """Exposes .tokenizer, .cfg and .to_tokens; records every prepend_bos it is given."""

    def __init__(self, chat_template="{{ messages }}", name="org/fake-chat"):
        self.tokenizer = FakeTokenizer(chat_template)
        self.cfg = types.SimpleNamespace(model_name=name)
        self.prepend_bos = []

    def to_tokens(self, text, prepend_bos=True):
        self.prepend_bos.append(prepend_bos)
        ids = self.tokenizer.encode(text)
        return t.tensor([([0] if prepend_bos else []) + ids])


def _pairs_file(path, rows, header=("harmful", "harmless")):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def _cfg(tmp_path, **data_params):
    return {
        "algorithm": "eap-ig",
        "level": "node",
        "batch_size": 2,
        "cache_dir": str(tmp_path),
        "data_params": {"num_examples": 4, **data_params},
    }


def _rows(spec, cfg, model):
    path, apply = spec._get_or_generate_csv(cfg, model)
    return pd.read_csv(path), apply


@pytest.fixture
def spec():
    return MultilingualRefusalTaskSpec()


@pytest.fixture
def hindi_pairs(tmp_path):
    return _pairs_file(
        tmp_path / "pairs_hi.csv",
        [(f"हानिकारक अनुरोध {i}", f"सुरक्षित अनुरोध {i}") for i in range(10)],
    )


def test_registered_as_a_builtin():
    assert "multilingual_refusal" in circuitkit.list_tasks()
    assert circuitkit.get_task("multilingual_refusal").chat_template_mode == "on"


class TestPairs:
    def test_english_comes_from_the_bundled_file(self):
        pairs = load_pairs({})
        assert len(pairs) == 334
        assert all(harmful != harmless for harmful, harmless in pairs)

    def test_another_language_comes_from_the_callers_file(self, hindi_pairs):
        pairs = load_pairs({"language": "hi", "pairs_path": hindi_pairs})
        assert pairs[0] == ("हानिकारक अनुरोध 0", "सुरक्षित अनुरोध 0")

    def test_another_language_without_a_file_says_what_to_provide(self):
        with pytest.raises(ValueError, match="pairs_path.*'harmful' and 'harmless'"):
            load_pairs({"language": "hi"})

    def test_a_file_without_the_columns_is_an_error(self, tmp_path):
        path = _pairs_file(tmp_path / "bad.csv", [("a", "b")], header=("prompt", "other"))
        with pytest.raises(ValueError, match=r"no column\(s\) \['harmful', 'harmless'\]"):
            load_pairs({"pairs_path": path})

    def test_blank_and_identical_rows_are_dropped(self, tmp_path):
        path = _pairs_file(tmp_path / "p.csv", [("same", "same"), ("", "x"), ("a b", "a c")])
        assert load_pairs({"pairs_path": path}) == [("a b", "a c")]

    def test_a_file_with_no_usable_pairs_is_an_error(self, tmp_path):
        path = _pairs_file(tmp_path / "p.csv", [("same", "same")])
        with pytest.raises(ValueError, match="no usable pairs"):
            load_pairs({"pairs_path": path})


class TestChatTemplate:
    def test_both_halves_are_rendered(self, spec, tmp_path, hindi_pairs):
        df, apply = _rows(spec, _cfg(tmp_path, language="hi", pairs_path=hindi_pairs), FakeModel())
        assert apply is True
        for text in list(df["clean"]) + list(df["corrupted"]):
            assert text.startswith("<bos> <user> ") and text.endswith("</user> <assistant>")
        assert all("हानिकारक" in text for text in df["clean"])
        assert all("सुरक्षित" in text for text in df["corrupted"])

    def test_a_templated_run_does_not_add_a_second_bos(self, spec, tmp_path, hindi_pairs):
        model = FakeModel()
        cfg = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs)
        loader = spec.build_dataloader(model, cfg, "cpu")
        assert loader.templated is True
        assert model.prepend_bos and not any(model.prepend_bos)

    def test_chat_template_kwargs_reach_the_template_and_the_cache_name(
        self, spec, tmp_path, hindi_pairs
    ):
        cfg = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs)
        default, _ = _rows(spec, cfg, FakeModel())
        off_cfg = _cfg(
            tmp_path,
            language="hi",
            pairs_path=hindi_pairs,
            chat_template_kwargs={"enable_thinking": False},
        )
        path, _ = spec._get_or_generate_csv(off_cfg, FakeModel())
        off = pd.read_csv(path)
        assert not any("</think>" in text for text in default["clean"])
        assert all(text.endswith("<assistant> <think> </think>") for text in off["clean"])
        assert all(text.endswith("<assistant> <think> </think>") for text in off["corrupted"])
        assert path != spec._get_or_generate_csv(cfg, FakeModel())[0]

    def test_a_thinking_template_left_on_is_warned_about(
        self, spec, tmp_path, hindi_pairs, monkeypatch
    ):
        # The task's logger may not propagate to caplog (CircuitKIT's own handler), so record the calls.
        from circuitkit.tasks.builtins import multilingual_refusal as module

        warnings = []
        monkeypatch.setattr(module.logger, "warning", warnings.append)
        model = FakeModel(chat_template="{% if enable_thinking %}...{% endif %}")
        spec._get_or_generate_csv(_cfg(tmp_path, language="hi", pairs_path=hindi_pairs), model)
        assert len(warnings) == 1 and "enable_thinking" in warnings[0] and "<think>" in warnings[0]
        warnings.clear()
        off = _cfg(
            tmp_path,
            language="hi",
            pairs_path=hindi_pairs,
            chat_template_kwargs={"enable_thinking": False},
        )
        spec._get_or_generate_csv(off, model)
        assert warnings == []

    @pytest.mark.parametrize("template", [None, "", "   ", {}, []])
    def test_a_base_model_is_an_error_not_a_raw_run(self, spec, tmp_path, template):
        with pytest.raises(ValueError, match="needs a chat model"):
            spec.build_dataloader(FakeModel(chat_template=template), _cfg(tmp_path), "cpu")

    @pytest.mark.parametrize(
        "template",
        [
            "{{ messages }}",
            {"default": "{{ messages }}"},
            [{"name": "default", "template": "{{ messages }}"}],
        ],
    )
    def test_a_named_template_counts_as_a_chat_model(self, spec, tmp_path, template):
        """Tiny Aya's tokenizer holds a dict of named templates, not a string."""
        model = FakeModel(chat_template=template)
        assert spec.build_dataloader(model, _cfg(tmp_path), "cpu").templated is True

    def test_off_takes_pairs_that_are_already_rendered(self, spec, tmp_path, hindi_pairs):
        model = FakeModel(chat_template=None)
        cfg = {**_cfg(tmp_path, language="hi", pairs_path=hindi_pairs), "chat_template_mode": "off"}
        loader = spec.build_dataloader(model, cfg, "cpu")
        assert loader.templated is False
        assert all(model.prepend_bos)
        assert next(iter(loader))[0][0].startswith(("हानिकारक"))


class TestAnswers:
    def test_clean_scores_the_refusal_and_corrupted_the_compliance(
        self, spec, tmp_path, hindi_pairs
    ):
        model = FakeModel()
        df, _ = _rows(spec, _cfg(tmp_path, language="hi", pairs_path=hindi_pairs), model)
        refuse, comply = DEFAULT_OPENERS["hi"]
        assert set(df["correct_idx"]) == {model.tokenizer.encode(refuse)[0]}
        assert set(df["incorrect_idx"]) == {model.tokenizer.encode(comply)[0]}

    def test_the_callers_openers_win(self, spec, tmp_path, hindi_pairs):
        model = FakeModel()
        cfg = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs, refuse="नहीं", comply="हाँ")
        df, _ = _rows(spec, cfg, model)
        assert set(df["correct_idx"]) == {model.tokenizer.encode("नहीं")[0]}

    def test_a_multi_token_opener_is_scored_on_its_first_token(self, spec, tmp_path, hindi_pairs):
        model = FakeModel()
        cfg = _cfg(
            tmp_path,
            language="hi",
            pairs_path=hindi_pairs,
            refuse="मैं नहीं कर सकता",
            comply="ज़रूर यह",
        )
        df, _ = _rows(spec, cfg, model)
        assert set(df["correct_idx"]) == {model.tokenizer.encode("मैं")[0]}

    def test_openers_sharing_a_first_token_are_an_error(self, spec, tmp_path):
        cfg = _cfg(tmp_path, refuse="I cannot", comply="I can")
        with pytest.raises(ValueError, match="same token"):
            spec.build_dataloader(FakeModel(), cfg, "cpu")

    def test_a_language_without_default_openers_needs_them_given(self, spec, tmp_path, hindi_pairs):
        cfg = _cfg(tmp_path, language="fr", pairs_path=hindi_pairs)
        with pytest.raises(ValueError, match="no default openers for language 'fr'"):
            spec.build_dataloader(FakeModel(), cfg, "cpu")
        cfg["data_params"].update(refuse="Désolé", comply="Bien")
        assert spec.build_dataloader(FakeModel(), cfg, "cpu") is not None

    def test_every_default_language_has_two_different_openers(self):
        assert {"en", "hi", "hi-Latn", "bn", "ta", "ar", "sw"} == set(DEFAULT_OPENERS)
        assert all(refuse != comply for refuse, comply in DEFAULT_OPENERS.values())


class TestTokenAlignment:
    def test_a_pair_of_unequal_token_length_is_dropped(self, spec, tmp_path):
        path = _pairs_file(tmp_path / "p.csv", [("one two three", "one two"), ("a b", "a c")])
        df, _ = _rows(spec, _cfg(tmp_path, pairs_path=path), FakeModel())
        assert len(df) == 1
        assert "a b" in df["clean"][0]

    def test_no_aligned_pair_at_all_is_an_error(self, spec, tmp_path):
        path = _pairs_file(tmp_path / "p.csv", [("one two three", "one two")])
        with pytest.raises(ValueError, match="tokenize to equal lengths"):
            spec.build_dataloader(FakeModel(), _cfg(tmp_path, pairs_path=path), "cpu")

    def test_num_examples_counts_kept_pairs(self, spec, tmp_path, hindi_pairs):
        df, _ = _rows(spec, _cfg(tmp_path, language="hi", pairs_path=hindi_pairs), FakeModel())
        assert len(df) == 4


class TestPartition:
    def _prompts(self, spec, tmp_path, pairs, partition):
        cfg = _cfg(
            tmp_path, language="hi", pairs_path=pairs, data_partition=partition, num_examples=100
        )
        return set(_rows(spec, cfg, FakeModel())[0]["clean"])

    def test_discovery_and_eval_are_disjoint_and_cover_the_pairs(self, spec, tmp_path, hindi_pairs):
        discovery = self._prompts(spec, tmp_path, hindi_pairs, "discovery")
        evaluation = self._prompts(spec, tmp_path, hindi_pairs, "eval")
        assert discovery and evaluation
        assert not discovery & evaluation
        assert len(discovery | evaluation) == 10

    def test_an_unknown_partition_is_an_error(self, spec, tmp_path):
        with pytest.raises(ValueError, match="data_partition"):
            spec.build_dataloader(FakeModel(), _cfg(tmp_path, data_partition="holdout"), "cpu")


class TestCache:
    def test_a_second_run_reuses_the_file(self, spec, tmp_path, hindi_pairs):
        cfg = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs)
        first, _ = spec._get_or_generate_csv(cfg, FakeModel())
        stamp = first.stat().st_mtime_ns
        second, _ = spec._get_or_generate_csv(cfg, FakeModel())
        assert first == second and first.stat().st_mtime_ns == stamp

    def test_editing_the_pairs_or_openers_does_not_reuse_a_stale_file(
        self, spec, tmp_path, hindi_pairs
    ):
        cfg = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs)
        path, _ = spec._get_or_generate_csv(cfg, FakeModel())
        edited = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs, refuse="नहीं")
        assert spec._get_or_generate_csv(edited, FakeModel())[0] != path
        _pairs_file(hindi_pairs, [("नया एक", "नया दो")])
        assert spec._get_or_generate_csv(cfg, FakeModel())[0] != path

    def test_templated_and_raw_runs_do_not_share_a_file(self, spec, tmp_path, hindi_pairs):
        cfg = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs)
        templated, _ = spec._get_or_generate_csv(cfg, FakeModel())
        raw, _ = spec._get_or_generate_csv({**cfg, "chat_template_mode": "off"}, FakeModel())
        assert templated.stem.endswith("_tmpl") and raw.stem.endswith("_raw")

    def test_finetuning_reads_its_own_languages_cache(self, spec, tmp_path, hindi_pairs):
        model = FakeModel()
        hi = _cfg(tmp_path, language="hi", pairs_path=hindi_pairs)
        spec._get_or_generate_csv(_cfg(tmp_path), model)  # a larger English cache beside it
        spec._get_or_generate_csv(hi, model)
        clean, queries = spec.build_finetuning_dataset(model.tokenizer, "org/fake-chat", 3, hi)
        assert len(clean) == len(queries) == 3
        assert all("हानिकारक" in query for query in queries)

    def test_finetuning_without_a_discovery_run_says_so(self, spec, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run discover_circuit"):
            spec.build_finetuning_dataset(FakeTokenizer(), "org/other", 3, _cfg(tmp_path))


class TestMetric:
    def test_logit_diff_is_refuse_minus_comply_at_the_last_position(self, spec):
        logits = t.zeros(1, 3, 5)
        logits[0, 1, 2], logits[0, 1, 4] = 3.0, 1.0
        labels = t.tensor([[2, 4]])
        value = spec._logit_diff(logits, None, t.tensor([2]), labels, mean=True, loss=False)
        assert value.item() == pytest.approx(2.0)

    def test_the_default_metric_is_a_loss(self, spec):
        logits = t.zeros(1, 3, 5)
        logits[0, 2, 2] = 1.0
        value = spec.metric_fn()(logits, None, t.tensor([3]), t.tensor([[2, 4]]))
        assert value.item() == pytest.approx(-1.0)

    def test_kl_is_available(self, spec):
        logits = t.zeros(1, 3, 5)
        assert spec.metric_fn("kl")(logits, logits, t.tensor([3]), None).item() == pytest.approx(
            0.0
        )


class TestValidation:
    def test_a_good_config_passes(self, spec, tmp_path):
        spec.validate_discovery_config(_cfg(tmp_path))

    def test_an_algorithm_outside_the_eap_family_is_rejected(self, spec, tmp_path):
        with pytest.raises(ValueError):
            spec.validate_discovery_config({**_cfg(tmp_path), "algorithm": "ibcircuit"})

    @pytest.mark.parametrize(
        "key,value,match", [("level", "edge", "level"), ("batch_size", 0, "batch_size")]
    )
    def test_bad_settings_are_rejected(self, spec, tmp_path, key, value, match):
        with pytest.raises(ValueError, match=match):
            spec.validate_discovery_config({**_cfg(tmp_path), key: value})

    def test_missing_pairs_fail_before_any_model_work(self, spec, tmp_path):
        with pytest.raises(ValueError, match="pairs_path"):
            spec.validate_discovery_config(_cfg(tmp_path, language="hi"))

    def test_metadata_names_the_language_and_the_source(self, spec, tmp_path, hindi_pairs):
        meta = spec.artifact_metadata(_cfg(tmp_path, language="hi", pairs_path=hindi_pairs))
        assert meta["task"] == "multilingual_refusal"
        assert meta["language"] == "hi"
        assert meta["data_source"] == hindi_pairs
        assert meta["n_samples"] == 4
        assert meta["chat_template_mode"] == "on"
