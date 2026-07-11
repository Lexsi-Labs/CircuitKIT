"""
conftest.py
===========
pytest configuration for the knowledge editing test suite.

Provides:
- Marker registrations (slow, xfail tracking)
- autouse caplog fixture at DEBUG level for all circuitkit loggers
- assert_no_user_warnings fixture factory (returned as a context-manager factory)
"""

import logging
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: real TransformerLens model tests; opt-in with -m slow",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """
    Turn every UserWarning into a hard error so swallowed exceptions surface,
    but ONLY for tests under tests/apply/.

    Previously this used ``config.addinivalue_line("filterwarnings",
    "error::UserWarning")`` inside ``pytest_configure``. That mutates the
    *session-global* filterwarnings ini option, so once this conftest loaded,
    every test in the whole run (e.g. tests/visualize) treated a benign
    UserWarning as an error — classic warning-filter pollution.

    Applying the filter as a per-item marker keeps it scoped to this package.
    ``add_marker(..., append=False)`` prepends the marker so it is processed
    *before* a test's own ``@pytest.mark.filterwarnings("default::UserWarning")``
    decorator. pytest applies each marker filter at the front of the warnings
    filter list, so the last-applied marker has highest priority — prepending
    here lets a test's own opt-out marker still win.
    """
    apply_dir = str(Path(__file__).parent)
    for item in items:
        if str(item.fspath).startswith(apply_dir):
            item.add_marker(pytest.mark.filterwarnings("error::UserWarning"), append=False)


# ---------------------------------------------------------------------------
# Auto-capture circuitkit logs for richer failure output
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _capture_circuitkit_logs(caplog):
    """
    Capture WARNING+ logs globally; tests that assert INFO messages must
    call caplog.set_level(logging.INFO, logger="circuitkit...") themselves.
    """
    with caplog.at_level(logging.WARNING, logger="circuitkit"):
        yield


# ---------------------------------------------------------------------------
# Shared fixture: assert_no_user_warnings
# ---------------------------------------------------------------------------


@pytest.fixture
def assert_no_user_warnings():
    """
    Returns a context-manager factory.  Inside the block any UserWarning
    causes the test to fail immediately with a helpful message that names
    the warning — directly surfacing errors that source code swallows with
    ``warnings.warn(...)`` inside try/except blocks.

    Usage in a test::

        def test_happy_path(assert_no_user_warnings):
            with assert_no_user_warnings():
                result = my_function(valid_input)
                assert result.success is True
    """

    @contextmanager
    def _check():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        user_warnings = [x for x in caught if issubclass(x.category, UserWarning)]
        if user_warnings:
            msgs = "\n  ".join(str(x.message) for x in user_warnings)
            pytest.fail(
                "Unexpected UserWarning(s) emitted on a happy-path call.\n"
                "This means a try/except block swallowed a real error.\n"
                f"Warnings caught:\n  {msgs}"
            )

    return _check
