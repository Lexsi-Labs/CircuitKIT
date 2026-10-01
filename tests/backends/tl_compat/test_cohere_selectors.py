"""
Group D validation: selector pass-through for tiny-aya (cohere2).

Scope: the plan classifies selectors as pure pass-through — every selector
under ``src/circuitkit/selection/`` takes ``(model, task_name, config)`` and
reads ``model.cfg`` only via generic attributes (``model_name``, dtype,
``next(model.parameters()).device``). None of them inspect architecture-
specific fields. **No code changes expected**; this file confirms it.

The 5 selectors covered are the ones the plan explicitly names:
``eap`` / ``eap-ig`` (both live in ``eap_selector.py``), ``eap-gp``,
``relp``, ``cdt``, and ``ibcircuit``. Pruning selectors (magnitude,
wanda, random, gptq) and application selectors (multi-granular, taylor,
awq, tacq) are out of scope: they operate on weights, not attribution
graphs, and were not raised as tiny-aya-facing surfaces by the plan.

Tests confirm:
  1. Every selector is registered under its expected key.
  2. Every selector's signature is exactly ``(model, task_name, config)``.
  3. No selector reads model-family-specific attributes (parallel_attn_mlp,
     n_key_value_heads, positional_embedding_type, rotary_adjacent_pairs)
     — a source-level regression sentry so an accidental architecture
     assumption in a future refactor fails a test instead of silently
     breaking cohere2.
  4. The selectors' post-processing (score normalisation) works uniformly
     for cohere2-scale score dictionaries produced by mocked backends.

Every test runs offline — no gated weights, no network, no GPU.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from circuitkit.backends import _tl_compat  # noqa: F401 — apply_patches() on import
from circuitkit.selection import get_selector, list_selectors


# --------------------------------------------------------------------------
# 1. Registration coverage — every expected selector name resolves.
# --------------------------------------------------------------------------


# The 5 discovery-family selectors the plan names for cohere2 pass-through
# validation. Their names must match the keys registered in
# circuitkit/selection/__init__.py via the @register decorator.
_EXPECTED_DISCOVERY_SELECTORS = (
    "eap",
    "eap-ig",
    "eap-gp",
    "relp",
    "cdt",
    "ibcircuit",
)


class TestSelectorRegistration:
    def test_every_expected_selector_is_registered(self):
        registered = set(list_selectors())
        for name in _EXPECTED_DISCOVERY_SELECTORS:
            assert name in registered, (
                f"Selector {name!r} is missing from the registry. Available: "
                f"{sorted(registered)}"
            )

    def test_every_registered_selector_is_callable(self):
        for name in _EXPECTED_DISCOVERY_SELECTORS:
            sel = get_selector(name)
            assert callable(sel), f"Selector {name!r} is registered but not callable"

    def test_get_selector_is_case_insensitive(self):
        """``get_selector`` normalises with ``.lower()``; catches the case
        where a caller passes ``"EAP-IG"`` from a UI or config file."""
        assert get_selector("EAP-IG") is get_selector("eap-ig")
        assert get_selector("CDT") is get_selector("cdt")


# --------------------------------------------------------------------------
# 2. Signature stability — every discovery selector matches the contract
#    ``(model, task_name, config) -> dict``.
# --------------------------------------------------------------------------


class TestSelectorSignatures:
    @pytest.mark.parametrize("name", _EXPECTED_DISCOVERY_SELECTORS)
    def test_signature_is_model_taskname_config(self, name):
        sel = get_selector(name)
        params = list(inspect.signature(sel).parameters)
        assert params == ["model", "task_name", "config"], (
            f"Selector {name!r} has non-standard signature {params!r}; the "
            f"discovery pipeline calls every selector as sel(model, task, cfg)."
        )


# --------------------------------------------------------------------------
# 3. Source-level regression sentry — no selector may pin cohere2's
#    architecture assumptions elsewhere. If a future refactor starts
#    reading n_key_value_heads / parallel_attn_mlp in a selector, that
#    selector has crossed the pass-through boundary and this test fails.
# --------------------------------------------------------------------------


class TestSelectorArchitectureAgnostic:
    _ARCH_TRIP_WORDS = (
        "parallel_attn_mlp",
        "n_key_value_heads",
        "rotary_adjacent_pairs",
        "positional_embedding_type",
        "sliding_window",
        "layer_types",
    )

    @pytest.mark.parametrize(
        "module_path",
        [
            "circuitkit.selection.eap_selector",
            "circuitkit.selection.eap_gp_selector",
            "circuitkit.selection.relp_selector",
            "circuitkit.selection.cdt_selector",
            "circuitkit.selection.ibcircuit_selector",
        ],
    )
    def test_selector_module_has_no_architecture_specific_reads(self, module_path):
        """None of the selectors should read cohere2-specific cfg fields;
        that's the backends' job. A future selector that started peering at
        ``model.cfg.parallel_attn_mlp`` would be crossing the pass-through
        boundary and needs a deliberate review — surface it here first."""
        module = __import__(module_path, fromlist=["__file__"])
        source = inspect.getsource(module)
        for word in self._ARCH_TRIP_WORDS:
            assert word not in source, (
                f"{module_path} contains {word!r}, an architecture-specific "
                f"cfg field. Selectors must stay architecture-agnostic — move "
                f"any GQA / parallel-block / rotary handling into the backend."
            )


# --------------------------------------------------------------------------
# 4. Selector produces the expected score-dict shape for cohere2 sizes,
#    given a mocked backend that returns per-head scores at cohere2 scale
#    (36 layers × 16 heads = 576 heads + 36 MLPs = 612 total keys).
# --------------------------------------------------------------------------


def _make_fake_cohere2_model():
    """Minimal duck-typed HookedTransformer for selector pass-through: only
    the fields the selector reads before dispatching to the backend, no more.
    """
    cfg = SimpleNamespace(
        model_name="tiny-aya-base",
        n_layers=36,
        n_heads=16,
        n_key_value_heads=4,
        d_head=128,
        d_model=2048,
        d_mlp=11008,
        parallel_attn_mlp=True,
        use_attn_result=False,
        use_split_qkv_input=False,
        use_hook_mlp_in=False,
    )
    model = SimpleNamespace(cfg=cfg)
    # Selectors read next(model.parameters()).device — supply a lightweight
    # parameter iterator that returns one CPU tensor.
    model.parameters = lambda: iter([torch.zeros(1)])
    return model


def _make_cohere2_scale_scores() -> dict:
    """Score dict at real cohere2 scale (36 layers × 16 query heads + 36
    MLPs = 612 keys). All 16 query heads are populated because api.py sets
    ``ungroup_grouped_query_attention=True`` and backends emit per-query-head
    scores after GQA expansion (see Group C's CDT fix)."""
    scores = {}
    for lyr in range(36):
        for h in range(16):
            # Deterministic non-uniform values so normalisation is meaningful.
            scores[f"A{lyr}.{h}"] = float(lyr * 16 + h + 1)
        scores[f"MLP {lyr}"] = float(lyr + 1) * 10.0
    return scores


class TestSelectorScoreDictShape:
    """Sanity-check that each selector's post-processing (normalisation +
    key formatting) handles cohere2-scale score dicts without dropping keys
    or corrupting the [0, 1] normalised range. Backends are mocked so
    these run offline."""

    def test_cdt_selector_normalises_cohere2_scores(self):
        """cdt_selector delegates to run_cdt_discovery, then min-max
        normalises. Mock the backend, feed cohere2-scale scores, verify
        the return shape and normalised range."""
        fake_scores = _make_cohere2_scale_scores()

        with patch(
            "circuitkit.backends.cdt.adapter.run_cdt_discovery",
            return_value=dict(fake_scores),
        ), patch(
            "circuitkit.tasks.bootstrap._bootstrap_builtin_tasks",
            lambda: None,
        ), patch(
            "circuitkit.tasks.registry.get_task"
        ) as _get_task:
            task_spec = SimpleNamespace(
                validate_discovery_config=lambda cfg: None,
                build_dataloader=lambda model, cfg, device: [],
            )
            _get_task.return_value = task_spec
            sel = get_selector("cdt")
            out = sel(_make_fake_cohere2_model(), "ioi", {"num_examples": 4})

        assert isinstance(out, dict)
        assert set(out) == set(fake_scores)
        vals = list(out.values())
        # Min-max normalisation: min should be 0, max should be 1.
        assert min(vals) == pytest.approx(0.0)
        assert max(vals) == pytest.approx(1.0)

    def test_ibcircuit_selector_handles_cohere2_scale_and_tensor_scores(self):
        """ibcircuit_selector accepts both plain floats and 0-d tensor scores
        (backend may return either). Test both mixed together at cohere2
        scale and confirm the wrapper coerces cleanly."""
        fake_scores = _make_cohere2_scale_scores()
        # Half of the values as 0-d tensors — the wrapper must handle both.
        mixed = {}
        for i, (k, v) in enumerate(fake_scores.items()):
            mixed[k] = torch.tensor(v) if i % 2 == 0 else v

        with patch(
            "circuitkit.backends.ibcircuit.trainer.run_ib_discovery",
            return_value=(mixed, None),
        ), patch(
            "circuitkit.tasks.bootstrap._bootstrap_builtin_tasks",
            lambda: None,
        ), patch(
            "circuitkit.tasks.registry.get_task"
        ) as _get_task:
            task_spec = SimpleNamespace(
                validate_discovery_config=lambda cfg: None,
                build_dataloader=lambda model, cfg, device: [],
            )
            _get_task.return_value = task_spec
            sel = get_selector("ibcircuit")
            out = sel(_make_fake_cohere2_model(), "ioi", {"num_examples": 4})

        assert isinstance(out, dict)
        assert set(out) == set(fake_scores)
        assert all(isinstance(v, float) for v in out.values())
        vals = list(out.values())
        assert min(vals) == pytest.approx(0.0)
        assert max(vals) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# 5. Preserve the min-max normalisation contract across the discovery
#    selector family — every selector maps scores to [0, 1]. Regression
#    guard so we don't accidentally swap in a z-score or softmax later.
# --------------------------------------------------------------------------


class TestSelectorNormalisationContract:
    _NORMALISATION_SIGNATURE = 'scores[k] = (scores[k] - mn) / (mx - mn)'
    _NORMALISED_MODULES = (
        "circuitkit.selection.eap_selector",
        "circuitkit.selection.eap_gp_selector",
        "circuitkit.selection.relp_selector",
        "circuitkit.selection.cdt_selector",
        "circuitkit.selection.ibcircuit_selector",
    )

    @pytest.mark.parametrize("module_path", _NORMALISED_MODULES)
    def test_selector_module_min_max_normalises(self, module_path):
        module = __import__(module_path, fromlist=["__file__"])
        source = inspect.getsource(module)
        assert self._NORMALISATION_SIGNATURE in source, (
            f"{module_path} no longer performs the standard min-max "
            f"normalisation. Selector consumers depend on scores being in "
            f"[0, 1] for cross-selector comparison — pick a different "
            f"normalisation deliberately, not by accident."
        )
