"""Regression tests — CLI / API argument-handling bugs (hardening pass).

Covers:
  * Bug 4 — ``circuitkit discover --algorithm taylor`` must be rejected;
            only the 13 discovery algorithms are valid for ``discover``.
  * Bug 5 — ``discover_circuit`` with algorithm 'acdc' and a ``data_params``
            dict containing ``num_examples`` must not crash on the kwarg.
"""

from __future__ import annotations

import pytest

pytest.importorskip("click")


# ---------------------------------------------------------------------------
# Bug 4 — CLI rejects non-discovery algorithms
# ---------------------------------------------------------------------------

# The 13 discovery algorithms (everything else — taylor/wanda/magnitude/random/
# multi_granular/gptq/awq/tacq — is a pruning or quantization algo).
EXPECTED_DISCOVERY_ALGOS = {
    "acdc",
    "atp-gd",
    "cdt",
    "eap",
    "eap-clean-corrupted",
    "eap-exact",
    "eap-gp",
    "eap-ifr",
    "eap-ig",
    "eap-ig-activations",
    "ibcircuit",
    "peap",
    "relp",
}


def test_discovery_algorithm_set_has_13_members():
    """The discovery-algorithm registry must contain exactly 13 algorithms."""
    from circuitkit.backends import DISCOVERY_ALGORITHMS

    assert set(DISCOVERY_ALGORITHMS) == EXPECTED_DISCOVERY_ALGOS
    assert len(DISCOVERY_ALGORITHMS) == 13


def test_cli_discover_rejects_pruning_algorithm():
    """``discover --algorithm taylor`` must fail — taylor is a pruning algo."""
    from click.testing import CliRunner

    from circuitkit.cli.main import cli

    result = CliRunner().invoke(cli, ["discover", "-m", "gpt2", "-a", "taylor", "-t", "ioi"])
    assert result.exit_code != 0, (
        "CLI accepted 'taylor' for discover — non-discovery algorithms must "
        "be rejected by the --algorithm Choice."
    )
    assert "taylor" in result.output and "is not one of" in result.output


@pytest.mark.parametrize("algo", ["wanda", "magnitude", "random", "gptq", "awq"])
def test_cli_discover_rejects_other_non_discovery_algorithms(algo):
    """No pruning/quantization algorithm may be passed to ``discover``."""
    from click.testing import CliRunner

    from circuitkit.cli.main import cli

    result = CliRunner().invoke(cli, ["discover", "-m", "gpt2", "-a", algo, "-t", "ioi"])
    assert result.exit_code != 0, f"CLI wrongly accepted '{algo}' for discover"


def test_cli_discover_accepts_a_discovery_algorithm():
    """A genuine discovery algorithm must pass the --algorithm validation.

    We stub ``discover_circuit`` so no model is loaded — we only assert the
    Choice did not reject 'eap-ig' (exit code 2 == bad option value).
    """
    from click.testing import CliRunner

    import circuitkit.api as api
    from circuitkit.cli.main import cli

    orig = api.discover_circuit
    api.discover_circuit = lambda cfg: ["A0.0"]  # noqa: E731
    try:
        result = CliRunner().invoke(cli, ["discover", "-m", "gpt2", "-a", "eap-ig", "-t", "ioi"])
    finally:
        api.discover_circuit = orig
    # Exit code 2 is click's "bad parameter"; anything else means the Choice
    # accepted 'eap-ig'.
    assert (
        result.exit_code != 2
    ), f"CLI rejected the valid discovery algorithm 'eap-ig': {result.output}"


# ---------------------------------------------------------------------------
# Bug 5 — ACDC + data_params(num_examples) must not crash
# ---------------------------------------------------------------------------


def test_acdc_load_task_data_accepts_num_examples_kwarg():
    """ACDC's data loader must accept a ``data_params`` dict with ``num_examples``.

    ``api.py`` runs ACDC discovery via::

        load_task_data(task_name=..., model=..., device=...,
                       **discovery_cfg.get('data_params', {}))

    so the *entire* ``data_params`` dict is splatted as kwargs. The historical
    bug was that ``load_task_data`` did not accept ``num_examples`` (the
    discovery-config-style scalar), so an 'acdc' run with
    ``data_params={'num_examples': N}`` crashed with a TypeError before any
    discovery work began.

    This test exercises ``load_task_data`` directly (the exact crash site) —
    far cheaper than the multi-hour ACDC edge sweep — and asserts that
    ``num_examples`` plus arbitrary extra ``data_params`` keys are accepted.
    """
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    from circuitkit.backends.acdc.data import load_task_data

    model = HookedTransformer.from_pretrained("gpt2", device="cpu")

    # Splat a discovery-style data_params dict — must NOT raise TypeError on
    # 'num_examples' (the fix) nor on extra keys ('seed' is forwarded verbatim).
    data_params = {"num_examples": 4, "seed": 42}
    train_loader, test_loader = load_task_data(
        task_name="ioi", model=model, device="cpu", **data_params
    )

    assert train_loader is not None
    # num_examples => all examples in the train split, none in test.
    assert len(train_loader) > 0


def test_acdc_is_a_registered_discovery_algorithm():
    """'acdc' must remain classified as a discovery algorithm."""
    from circuitkit.backends import DISCOVERY_ALGORITHMS

    assert "acdc" in DISCOVERY_ALGORITHMS
