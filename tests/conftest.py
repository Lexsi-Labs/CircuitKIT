import logging
import os
import tempfile
from pathlib import Path

import pytest
import torch

import circuitkit  # noqa: F401  - installs the TransformerLens patch hook first

# ---------------------------------------------------------------------------
# Test tiering
# ---------------------------------------------------------------------------
# The default `python -m pytest` run (see pytest.ini `addopts`) excludes the
# `slow`, `network`, and `gated` markers. Rather than sprinkle module-level
# markers across ~40 files, the heavy modules are listed here explicitly — one
# reviewable place — and marked at collection time. A trailing "/" entry marks
# an entire directory.
#
# Bias is intentionally toward OVER-marking: a fast test wrongly moved to the
# nightly tier only loses a little gate coverage, whereas a heavy test left in
# the gate can hang the release. Anything not listed here runs in the fast tier.
#
# `network`  -> also implies deselection by default AND is what the offline
#               guard below protects against (a stalled download can't hang CI).
# `slow`     -> heavy compute (real training / discovery / transfer), offline-safe.
_NETWORK_MODULES = (
    # real HF model / dataset downloads
    "tests/apply/test_covariance.py",
    "tests/apply/test_knowledge_editing_pipeline.py",
    "tests/apply/test_pruner.py",
    "tests/apply/test_steering.py",
    "tests/apply/test_tokenization.py",
    "tests/apply/test_weight_steering.py",
    "tests/unit/test_api.py",
    "tests/unit/test_hf_checkpoint.py",
    "tests/unit/test_pillars.py",
    "tests/unit/test_quick_api.py",
    "tests/unit/test_quick_extensions.py",
    "tests/unit/test_score_loader.py",
    "tests/unit/test_selector_bugs.py",
    "tests/unit/test_end_to_end.py",
    "tests/unit/test_pipeline_smoke.py",
    "tests/test_custom_data.py",
    # whole integration-level directories exercise real discover/evaluate
    # pipelines (they load a real model and/or a real dataset)
    "tests/integration/",
    "tests/regression/",
    "tests/tasks/",
    "tests/benchmarks/",
)
_SLOW_MODULES = (
    # heavy compute without a guaranteed download
    "tests/unit/test_soft_healing.py",
    "tests/unit/test_finetune_utils.py",
)
_GATED_MODULES = (
    # need a gated 3.35B checkpoint + HF_TOKEN + explicit opt-in env; already
    # self-skip, but tag them so `-m "slow and not gated"` excludes them cleanly
    "tests/backends/tl_compat/test_cohere_parity.py",
    "tests/regression/test_cohere_discovery.py",
)


def _norm(path: str) -> str:
    return str(path).replace("\\", "/")


def _matches(nodepath: str, patterns) -> bool:
    for p in patterns:
        if p.endswith("/"):
            if p in nodepath:  # directory prefix, e.g. "tests/integration/"
                return True
        elif nodepath.endswith(p):  # exact file, e.g. ".../test_covariance.py"
            return True
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):  # noqa: D401
    """Auto-tag heavy/network modules so the default `-m` filter deselects them.

    Runs ``tryfirst`` so the markers are attached before pytest's own ``-m``
    keyword/marker deselection pass sees the items.
    """
    for item in items:
        nodepath = _norm(item.fspath)
        if _matches(nodepath, _GATED_MODULES):
            item.add_marker(pytest.mark.gated)
            item.add_marker(pytest.mark.slow)
        if _matches(nodepath, _NETWORK_MODULES):
            item.add_marker(pytest.mark.network)
            item.add_marker(pytest.mark.slow)
        elif _matches(nodepath, _SLOW_MODULES):
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Force HuggingFace offline mode whenever the fast (default) tier is selected.

    If a `network` module is ever missed above, this turns a silent multi-hour
    download hang into an immediate, obvious error — so a mis-tag surfaces as a
    fast red X in CI instead of burning the 6-hour job ceiling. When the heavy
    tiers are explicitly requested (`-m slow`, `-m network`, `-m ""`), real
    network access is left enabled.
    """
    markexpr = config.getoption("markexpr", default="") or ""
    fast_tier = "not slow" in markexpr and "not network" in markexpr
    if fast_tier:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _circuitkit_logger_propagate():
    """Let ``caplog`` capture CircuitKIT log records.

    CircuitKIT's custom loggers set ``propagate = False`` (they own a pretty
    console handler). pytest's ``caplog`` fixture captures via the *root*
    logger, so without propagation caplog-based assertions see nothing.
    Re-enable propagation for the duration of each test (test-only — library
    runtime behavior is unchanged).
    """
    changed = []
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("circuitkit") and isinstance(lg, logging.Logger) and not lg.propagate:
            lg.propagate = True
            changed.append(lg)
    yield
    for lg in changed:
        lg.propagate = False


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {"name": "gpt2", "precision": "float32"},
        "discovery": {
            "algorithm": "eap-ig",
            "task": "ioi",  # Built-in task - data is auto-generated
            "level": "node",
            "batch_size": 1,
            "ig_steps": 2,
            "data_params": {"num_examples": 32},
        },
        "pruning": {"target_sparsity": 0.1, "scope": "heads"},
        "output_path": "tests/temp_results.pt",
    }


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"clean": "Hello world", "corrupted": "Hi world", "correct_idx": 1, "incorrect_idx": 2},
        {
            "clean": "Test sentence",
            "corrupted": "Test phrase",
            "correct_idx": 3,
            "incorrect_idx": 4,
        },
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def device():
    """Test device (CPU for testing)."""
    return torch.device("cpu")
