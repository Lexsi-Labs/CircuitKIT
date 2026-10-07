"""Hyperparameter registry: valid vs sensible ranges (pure Python, no torch)."""

import warnings
from pathlib import Path

import pytest

from circuitkit.utils import hparams as hp
from circuitkit.utils.config import load_and_validate_config
from circuitkit.utils.hparams import (
    HPARAMS,
    PILLARS_BASIC,
    PILLARS_VALID,
    STRICT_ENV_VAR,
    HyperparameterError,
    HyperparameterWarning,
    check_rules,
    check_value,
    collect_config_values,
    render_markdown,
    validate,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _severities(key, value):
    return [issue.severity for issue in check_value(key, value)]


class TestRegistry:
    def test_keys_are_section_dot_name(self):
        for key, spec in HPARAMS.items():
            assert key == spec.key
            assert key.count(".") == 1, key

    def test_ranges_are_ordered(self):
        for key, spec in HPARAMS.items():
            if spec.sensible_min is not None and spec.sensible_max is not None:
                assert spec.sensible_min <= spec.sensible_max, key
            if spec.valid_min is not None and spec.sensible_min is not None:
                assert spec.valid_min <= spec.sensible_min, key
            if spec.valid_max is not None and spec.sensible_max is not None:
                assert spec.sensible_max <= spec.valid_max, key

    def test_defaults_are_valid_and_never_warn(self):
        """A default must never trigger a warning (only explicit values are warned about)."""
        for key, spec in HPARAMS.items():
            if spec.default is None or spec.kind not in ("int", "float"):
                continue
            assert check_value(key, spec.default) == [], key

    def test_every_entry_explains_itself(self):
        for key, spec in HPARAMS.items():
            assert spec.summary and spec.why, key
            assert spec.confidence in ("high", "medium"), key
            assert spec.checked_by in ("hparams", "existing", "docs"), key

    def test_only_enforced_entries_warn(self):
        """A sensible range is warned on only when this module enforces it."""
        for key, spec in HPARAMS.items():
            if spec.warns:
                assert spec.checked_by == "hparams", key

    def test_pillar_sets(self):
        assert set(PILLARS_BASIC) <= set(PILLARS_VALID)
        assert PILLARS_BASIC == ("patching", "ablation")
        assert len(set(PILLARS_VALID)) == len(PILLARS_VALID)


@pytest.mark.parametrize(
    "key, value, expected",
    [
        # ig_steps: valid >= 1, no sensible range (it only affects the EAP-IG family)
        ("discovery.ig_steps", 0, ["error"]),
        ("discovery.ig_steps", 1, []),
        ("discovery.ig_steps", 3, []),
        ("discovery.ig_steps", 50, []),
        ("discovery.ig_steps", 2.5, ["error"]),
        ("discovery.ig_steps", 3.0, ["error"]),  # an integral float still fails in range(1, steps + 1)
        ("discovery.ig_steps", True, ["error"]),
        ("discovery.ig_steps", "3", ["error"]),
        # discovery examples: valid >= 1, sensible 32-1024
        ("discovery.num_examples", 0, ["error"]),
        ("discovery.num_examples", 8, ["warning"]),
        ("discovery.num_examples", 32, []),
        ("discovery.num_examples", 128, []),
        ("discovery.num_examples", 1024, []),
        ("discovery.num_examples", 5000, ["warning"]),
        ("discovery.num_examples", None, []),
        ("discovery.batch_size", 0, ["error"]),
        ("discovery.batch_size", 1, []),
        # sparsity: valid [0, 1], sensible <= 0.4 (low values are fine)
        ("pruning.target_sparsity", -0.1, ["error"]),
        ("pruning.target_sparsity", 0.0, []),
        ("pruning.target_sparsity", 0.3, []),
        ("pruning.target_sparsity", 0.4, []),
        ("pruning.target_sparsity", 0.5, ["warning"]),
        ("pruning.target_sparsity", 1.0, ["warning"]),
        ("pruning.target_sparsity", 1.2, ["error"]),
        ("pruning.target_sparsity", float("nan"), ["error"]),
        # evaluation
        ("eval.num_examples", 50, ["warning"]),
        ("eval.num_examples", 256, []),
        ("eval.n_stability_runs", 0, ["error"]),
        ("eval.n_stability_runs", 1, ["warning"]),  # runs, but reports a trivial overlap of 1.0
        ("eval.n_stability_runs", 2, ["warning"]),
        ("eval.n_stability_runs", 3, []),
        ("eval.n_stability_runs", 11, ["warning"]),
        ("eval.pillars", ["patching", "ablation"], []),
        ("eval.pillars", list(PILLARS_VALID), []),
        ("eval.pillars", ["patchng"], ["error"]),
        ("eval.pillars", "all", []),
        ("eval.pillars", "patching", ["error"]),
        # ACDC grid
        ("discovery.tao_bases", [1, 5], []),
        ("discovery.tao_bases", [], ["error"]),
        ("discovery.tao_bases", [0], ["error"]),
        ("discovery.tao_exps", [-5, -2], []),
        ("discovery.tao_exps", [-3.5], ["error"]),
        # quantization
        ("quantization.bits", 4, []),
        ("quantization.bits", 5, ["error"]),
        ("quantization.high_fraction", 0.05, []),
        ("quantization.high_fraction", 1.5, ["error"]),
    ],
)
def test_check_value(key, value, expected):
    assert _severities(key, value) == expected


def test_error_message_names_parameter_and_range():
    (issue,) = check_value("discovery.ig_steps", 0)
    assert "discovery.ig_steps" in issue.message
    assert ">= 1" in issue.message
    assert "0" in issue.message


def test_warning_message_names_range_and_evidence():
    (issue,) = check_value("pruning.target_sparsity", 0.5)
    assert "<= 0.4" in issue.message
    assert "collapsed" in issue.message  # the evidence travels with the warning


class TestRules:
    def test_acdc_tau_of_one_or_more(self):
        issues = check_rules({"discovery.tao_bases": [1, 5], "discovery.tao_exps": [-3, 0]})
        assert [i.severity for i in issues] == ["warning"]
        assert "tau >= 1" in issues[0].message

    def test_acdc_default_grid_is_fine(self):
        assert check_rules({"discovery.tao_bases": [1, 5], "discovery.tao_exps": [-5, -4, -3, -2]}) == []

    def test_low_bit_with_few_protected_layers(self):
        values = {
            "quantization.bits": 3,
            "quantization.high_fraction": 0.05,
            "quantization.backend": "llmcompressor",
        }
        assert [i.severity for i in check_rules(values)] == ["warning"]

    @pytest.mark.parametrize(
        "bits, high, backend",
        [
            (4, 0.05, "llmcompressor"),  # 4-bit with 5% protected kept 0.98 retention
            (3, 0.15, "llmcompressor"),  # 3-bit needs >= 15%
            (3, 0.05, "quanto"),  # quanto ignores `bits`
        ],
    )
    def test_low_bit_rule_does_not_fire(self, bits, high, backend):
        values = {
            "quantization.bits": bits,
            "quantization.high_fraction": high,
            "quantization.backend": backend,
        }
        assert check_rules(values) == []


class TestValidate:
    def test_invalid_values_raise_with_all_problems(self):
        with pytest.raises(HyperparameterError) as exc:
            validate({"discovery.ig_steps": 0, "eval.n_stability_runs": 0})
        assert isinstance(exc.value, ValueError)
        assert "discovery.ig_steps" in str(exc.value)
        assert "eval.n_stability_runs" in str(exc.value)

    def test_unknown_keys_are_ignored(self):
        assert validate({"not.a.param": -1}) == []

    def test_sensible_violations_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate({"discovery.num_examples": 8, "pruning.target_sparsity": 0.6})
        assert [w.category for w in caught] == [HyperparameterWarning] * 2

    def test_the_warning_is_not_a_user_warning(self):
        """circuitkit.api ignores every UserWarning at import, so this class must not be one."""
        assert issubclass(HyperparameterWarning, Warning)
        assert not issubclass(HyperparameterWarning, UserWarning)

    def test_delivered_under_the_blanket_user_warning_filter(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")  # independent of how pytest was started
            warnings.filterwarnings("ignore", category=UserWarning)  # what circuitkit.api installs
            validate({"pruning.target_sparsity": 0.9})
        assert [w.category for w in caught] == [HyperparameterWarning]

    def test_only_user_set_values_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate(
                {"discovery.num_examples": 8, "pruning.target_sparsity": 0.6},
                user_set={"pruning.target_sparsity"},
            )
        assert len(caught) == 1
        assert "target_sparsity" in str(caught[0].message)

    def test_invalid_values_raise_even_when_not_user_set(self):
        with pytest.raises(HyperparameterError):
            validate({"discovery.ig_steps": 0}, user_set=set())

    def test_strict_turns_warnings_into_errors(self):
        with pytest.raises(HyperparameterError):
            validate({"discovery.num_examples": 8}, strict=True)

    def test_strict_from_environment(self, monkeypatch):
        monkeypatch.setenv(STRICT_ENV_VAR, "1")
        with pytest.raises(HyperparameterError):
            validate({"discovery.num_examples": 8})
        # an explicit argument beats the environment
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            validate({"discovery.num_examples": 8}, strict=False)

    def test_clean_values_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert validate({"discovery.num_examples": 128, "pruning.target_sparsity": 0.3}) == []


class TestCollectConfigValues:
    CONFIG = {
        "discovery": {
            "algorithm": "eap",
            "ig_steps": 0,
            "batch_size": 4,
            "data_params": {"num_examples": 8},
        },
        "pruning": {"target_sparsity": 0.3},
        "eval": {"num_examples": 256},
    }

    def test_ig_steps_is_only_read_for_the_eap_ig_family(self):
        values, _ = collect_config_values(self.CONFIG)
        assert "discovery.ig_steps" not in values  # plain eap ignores it
        for algo in ("eap-ig", "eap-ig-activations", "eap-gp"):
            cfg = {**self.CONFIG, "discovery": {**self.CONFIG["discovery"], "algorithm": algo}}
            values, _ = collect_config_values(cfg)
            assert values["discovery.ig_steps"] == 0, algo

    def test_user_set_comes_from_the_unmerged_config(self):
        user = {"discovery": {"data_params": {"num_examples": 8}}}
        values, user_set = collect_config_values(self.CONFIG, user)
        assert values["discovery.num_examples"] == 8
        assert user_set == {"discovery.num_examples"}

    def test_acdc_keys_only_for_acdc(self):
        cfg = {"discovery": {"algorithm": "eap-ig", "tao_bases": [1]}}
        assert "discovery.tao_bases" not in collect_config_values(cfg)[0]
        cfg["discovery"]["algorithm"] = "acdc"
        assert collect_config_values(cfg)[0]["discovery.tao_bases"] == [1]


class TestLoadAndValidateConfig:
    """The dict/YAML config path (shared by discover_circuit, evaluate_circuit and the flat API)."""

    @staticmethod
    def _cfg(**discovery):
        return {"model": {"name": "gpt2"}, "discovery": {"algorithm": "eap-ig", "task": "ioi", **discovery}}

    def test_defaults_never_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", HyperparameterWarning)
            load_and_validate_config(self._cfg())

    def test_invalid_value_raises_a_value_error(self):
        with pytest.raises(ValueError, match="discovery.ig_steps"):
            load_and_validate_config(self._cfg(ig_steps=0))

    def test_irrelevant_parameter_is_not_checked(self):
        cfg = self._cfg(ig_steps=0)
        cfg["discovery"]["algorithm"] = "eap"  # plain EAP never reads ig_steps
        load_and_validate_config(cfg)

    def test_explicit_value_outside_the_sensible_range_warns(self):
        cfg = self._cfg(data_params={"num_examples": 8})
        cfg["pruning"] = {"target_sparsity": 0.6}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_and_validate_config(cfg)
        text = " ".join(str(w.message) for w in caught if issubclass(w.category, HyperparameterWarning))
        assert "discovery.num_examples=8" in text
        assert "pruning.target_sparsity=0.6" in text

    def test_n_examples_alias_is_checked_too(self):
        cfg = self._cfg(data_params={"n_examples": 8})
        with pytest.warns(HyperparameterWarning, match="discovery.num_examples=8"):
            load_and_validate_config(cfg)

    def test_unknown_pillar_fails_before_any_work(self):
        cfg = self._cfg()
        cfg["eval"] = {"pillars": ["patchng"]}
        with pytest.raises(HyperparameterError, match="patchng"):
            load_and_validate_config(cfg)

    def test_one_stability_run_is_valid_but_warns(self):
        cfg = self._cfg()
        cfg["eval"] = {"n_stability_runs": 1}
        with pytest.warns(HyperparameterWarning, match="n_stability_runs"):
            load_and_validate_config(cfg)

    def test_strict_in_the_config_turns_warnings_into_errors(self):
        cfg = self._cfg(data_params={"num_examples": 8})
        cfg["validation"] = {"strict": True}
        with pytest.raises(HyperparameterError, match="discovery.num_examples"):
            load_and_validate_config(cfg)

    def test_strict_from_the_environment(self, monkeypatch):
        monkeypatch.setenv(STRICT_ENV_VAR, "1")
        with pytest.raises(HyperparameterError):
            load_and_validate_config(self._cfg(data_params={"num_examples": 8}))

    def test_existing_validation_still_runs(self):
        with pytest.raises(ValueError, match="Invalid algorithm"):
            load_and_validate_config(self._cfg(algorithm="not-an-algorithm"))

    def test_registry_defaults_match_default_config(self):
        """Defaults recorded in the registry must be the ones DEFAULT_CONFIG actually uses."""
        from circuitkit.utils.config import DEFAULT_CONFIG

        for key, spec in HPARAMS.items():
            if not spec.config_path or spec.default is None:
                continue
            node = DEFAULT_CONFIG
            for part in spec.config_path:
                node = node[part]
            assert node == spec.default, key


class TestDocs:
    PAGE = REPO_ROOT / "docs" / "reference" / "hyperparameters.md"

    def test_generated_block_is_in_sync_with_the_registry(self):
        text = self.PAGE.read_text(encoding="utf-8")
        start, end = "<!-- hparams:start -->", "<!-- hparams:end -->"
        block = text.split(start, 1)[1].split(end, 1)[0]
        assert block.strip() == render_markdown().strip(), (
            "docs/reference/hyperparameters.md is stale; run "
            "`python -m circuitkit.utils.hparams --write docs/reference/hyperparameters.md`"
        )

    def test_page_is_in_the_nav(self):
        nav = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        assert "reference/hyperparameters.md" in nav

    def test_every_entry_is_rendered(self):
        rendered = render_markdown()
        for spec in HPARAMS.values():
            assert f"`{spec.name}`" in rendered, spec.key


def test_module_only_imports_the_standard_library():
    """Config validation and the docs build must not pull in torch / TransformerLens."""
    import ast
    import sys

    tree = ast.parse(Path(hp.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    not_stdlib = {name for name in imported if name not in sys.stdlib_module_names}
    assert not not_stdlib, f"circuitkit.utils.hparams must stay pure Python, found {sorted(not_stdlib)}"


class TestReviewFixes:
    """Behaviour pinned after review: integer typing, odd inputs, strict parsing, ACDC defaults."""

    @pytest.mark.parametrize(
        "key, value",
        [
            ("discovery.ig_steps", 3.0),
            ("discovery.num_examples", 128.0),
            ("eval.n_stability_runs", 3.0),
            ("discovery.tao_exps", [-3.0, -2]),
            ("quantization.bits", 4.0),
        ],
    )
    def test_integer_parameters_reject_floats(self, key, value):
        with pytest.raises(HyperparameterError, match="integer"):
            validate({key: value})

    def test_numpy_integers_are_integers(self):
        np = pytest.importorskip("numpy")
        assert validate({"discovery.ig_steps": np.int64(3), "discovery.tao_exps": np.array([-3, -2])}) == []

    def test_a_huge_int_is_a_value_not_a_crash(self):
        with pytest.raises(HyperparameterError, match="target_sparsity"):
            validate({"pruning.target_sparsity": 10**400})
        with pytest.warns(HyperparameterWarning, match="num_examples"):
            validate({"discovery.num_examples": 10**400})
        with pytest.raises(HyperparameterError, match="integer|valid range"):
            validate({"eval.n_stability_runs": float("inf")})

    def test_arrays_never_raise_a_raw_error(self):
        np = pytest.importorskip("numpy")
        validate({"discovery.tao_bases": np.array([1, 5]), "discovery.tao_exps": np.array([-3, -2])})
        with pytest.warns(HyperparameterWarning, match="tau >= 1"):
            validate({"discovery.tao_bases": np.array([1, 5]), "discovery.tao_exps": np.array([0, 1])})

    def test_unexpected_types_become_hyperparameter_errors(self):
        with pytest.raises(HyperparameterError):
            validate({"discovery.tao_exps": [object()]})
        with pytest.raises(HyperparameterError):
            validate({"eval.pillars": 3})

    @pytest.mark.parametrize("word", ["false", "False", "no", "off", "0", ""])
    def test_a_quoted_false_does_not_turn_strict_mode_on(self, word):
        with pytest.warns(HyperparameterWarning):
            validate({"pruning.target_sparsity": 0.9}, strict=word)

    @pytest.mark.parametrize("word", ["true", "True", "yes", "on", "1"])
    def test_a_quoted_true_does_turn_it_on(self, word):
        with pytest.raises(HyperparameterError):
            validate({"pruning.target_sparsity": 0.9}, strict=word)

    def test_an_unrecognised_strict_value_is_an_error(self):
        with pytest.raises(HyperparameterError, match="strict"):
            validate({"pruning.target_sparsity": 0.3}, strict="maybe")

    def test_strict_false_string_in_a_config(self):
        cfg = TestLoadAndValidateConfig._cfg(data_params={"num_examples": 8})
        cfg["validation"] = {"strict": "false"}
        with pytest.warns(HyperparameterWarning, match="num_examples"):
            load_and_validate_config(cfg)
        cfg["validation"] = {"strict": "maybe"}
        with pytest.raises(HyperparameterError, match="validation.strict"):
            load_and_validate_config(cfg)

    def test_tau_rule_uses_the_backend_default_for_the_list_not_given(self):
        # tao_bases defaults to [1, 5], so exps [0, 1] gives taus 1, 5, 10, 50
        with pytest.warns(HyperparameterWarning, match="tau >= 1"):
            validate({"discovery.tao_exps": [0, 1]}, user_set={"discovery.tao_exps"})
        # tao_exps defaults to [-5..-2], so a base of 100 gives tau = 1 at exp -2
        with pytest.warns(HyperparameterWarning, match="tau >= 1"):
            validate({"discovery.tao_bases": [100]}, user_set={"discovery.tao_bases"})

    def test_tau_rule_in_a_config_with_one_list(self):
        cfg = TestLoadAndValidateConfig._cfg(algorithm="acdc", tao_exps=[0, 1])
        with pytest.warns(HyperparameterWarning, match="tau >= 1"):
            load_and_validate_config(cfg)

    def test_pillars_all_is_accepted_in_a_config_and_means_every_pillar(self):
        cfg = TestLoadAndValidateConfig._cfg()
        cfg["eval"] = {"pillars": "all"}
        assert load_and_validate_config(cfg)["eval"]["pillars"] is None

    def test_out_of_range_sparsity_in_a_config_raises_the_hyperparameter_error(self):
        cfg = TestLoadAndValidateConfig._cfg()
        cfg["pruning"] = {"target_sparsity": 1.5}
        with pytest.raises(HyperparameterError, match="target_sparsity"):
            load_and_validate_config(cfg)

    def test_non_positive_num_epochs_raises_the_hyperparameter_error(self):
        cfg = TestLoadAndValidateConfig._cfg(algorithm="ibcircuit", num_epochs=0)
        with pytest.raises(HyperparameterError, match="num_epochs"):
            load_and_validate_config(cfg)
