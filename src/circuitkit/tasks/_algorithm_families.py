"""Shared algorithm-family constants for TaskSpec implementations.

Keeps the per-task algorithm whitelists in one place so when we add a
new EAP-internal method as a top-level key (e.g. ``eap-ig-activations``)
it gets recognised by every task at once instead of requiring 13 edits.

EAP_FAMILY: algorithms that consume the same EAP-format CSV dataloader
            and route through ``backends.eap.attribute_node`` with a
            different ``method=`` argument under the hood.

ACDC_FAMILY / IB_FAMILY: kept explicit for readability.
"""

from __future__ import annotations

from typing import FrozenSet

# Top-level algorithm keys in the EAP family (all share the EAP CSV
# dataloader and route through backends/eap/attribute_node).
EAP_FAMILY: FrozenSet[str] = frozenset(
    {
        "eap",
        "eap-ig",
        "eap-ig-activations",
        "eap-clean-corrupted",
        "eap-exact",
        # AtP+GradDrop reuses the EAP CSV dataloader and attribute_node dispatch.
        "atp-gd",
        # EAP-GP / GradPath (Zhang et al. 2025) — same EAP machinery,
        # different integration path.
        "eap-gp",
        # RelP / Relevance Patching (Mohebbi et al. 2025) — same EAP machinery,
        # gradient is rerouted via forward detach hooks (LRP-ε style).
        "relp",
        # PEAP (Haklay et al. 2025) — same EAP machinery, retains per-position
        # scores instead of summing them.
        "peap",
        # Information Flow Routes (Ferrando et al. 2024) — uses the EAP-CSV
        # dataloader; algorithmically computes proximity scores from a
        # single clean forward pass.
        "eap-ifr",
    }
)


ACDC_FAMILY: FrozenSet[str] = frozenset({"acdc"})
IB_FAMILY: FrozenSet[str] = frozenset({"ibcircuit"})
CDT_FAMILY: FrozenSet[str] = frozenset({"cdt"})


def is_eap_family(algorithm: str) -> bool:
    """Return True for any algorithm using the EAP-CSV / attribute_node path."""
    return (algorithm or "").lower() in EAP_FAMILY


def is_acdc(algorithm: str) -> bool:
    return (algorithm or "").lower() in ACDC_FAMILY


def is_ibcircuit(algorithm: str) -> bool:
    return (algorithm or "").lower() in IB_FAMILY


def is_cdt(algorithm: str) -> bool:
    return (algorithm or "").lower() in CDT_FAMILY


def unsupported_algorithm_message(
    task_name: str, algorithm: str, supported: "FrozenSet[str]"
) -> str:
    """Build an actionable error message for an unsupported / missing algorithm.

    Distinguishes the two failure modes a user hits:

    * ``algorithm`` empty/missing -> the required key was omitted from the
      discovery config; tell them to add it and list the valid values.
    * ``algorithm`` present but unknown -> name the bad value and list the
      valid values.

    The phrase "does not support algorithm" is preserved so callers/tests that
    match on it keep working.
    """
    valid = ", ".join(sorted(supported))
    if not (algorithm or "").strip():
        return (
            f"{task_name} discovery config is missing the required key 'algorithm'. "
            f"Add 'algorithm' to the discovery config. "
            f"{task_name} supports these discovery algorithms: {valid}."
        )
    return (
        f"{task_name} does not support algorithm '{algorithm}'. "
        f"Set discovery config key 'algorithm' to one of: {valid}."
    )


__all__ = [
    "EAP_FAMILY",
    "ACDC_FAMILY",
    "IB_FAMILY",
    "CDT_FAMILY",
    "is_eap_family",
    "is_acdc",
    "is_ibcircuit",
    "is_cdt",
    "unsupported_algorithm_message",
]
