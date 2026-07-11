import logging
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def _circuitkit_logger_propagate():
    """Let ``caplog`` capture CircuitKit log records.

    CircuitKit's custom loggers set ``propagate = False`` (they own a pretty
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
