"""
CircuitKit Discovery Backends — algorithm registry & stability tiers.

This module is the SINGLE SOURCE OF TRUTH for every algorithm name CircuitKit
knows about, its category (discovery / pruning / quantization), and its
stability tier. ``circuitkit.utils.exceptions`` derives its validation
registries from here — do not maintain a second copy.

Stability tiers:
  stable        — Production-ready. Tested on GPT-2, Llama 1B/3B, Gemma 1B/4B.
  experimental  — Works on IOI. May fail on larger models. Use at own risk.
  research      — Implemented but unvalidated outside GPT-2 IOI. For exploration only.

Usage:
    from circuitkit.backends import STABILITY, is_stable, default_algorithm
    algo = default_algorithm()  # → "eap-ig"
    if is_stable(algo):
        logger.info("Safe to use on any model")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (category, stability) for every algorithm CircuitKit knows about.
#   category   ∈ {"discovery", "pruning", "quantization"}
#   stability  ∈ {"stable", "experimental", "research"}
ALGORITHMS: dict[str, tuple[str, str]] = {
    # ── Discovery: EAP family — full production pipeline ──
    "eap": ("discovery", "stable"),
    "eap-ig": ("discovery", "stable"),
    "eap-ig-activations": ("discovery", "research"),
    "eap-clean-corrupted": ("discovery", "research"),
    # ── Discovery: EAP research variants — GPT-2 IOI only ──
    "eap-exact": ("discovery", "research"),
    "atp-gd": ("discovery", "research"),
    "eap-gp": ("discovery", "stable"),
    "relp": ("discovery", "research"),
    "peap": ("discovery", "research"),
    "eap-ifr": ("discovery", "research"),
    # ── Discovery: other backends ──
    "acdc": ("discovery", "stable"),  # node-only by construction (edge search)
    "ibcircuit": ("discovery", "stable"),  # memory ceiling on multi-B at aggressive settings (documented)
    "cdt": ("discovery", "stable"),  # clean-only; RoPE approximation documented
    # ── Pruning selectors / baselines ──
    "random": ("pruning", "stable"),
    "magnitude": ("pruning", "stable"),
    "taylor": ("pruning", "stable"),
    "wanda": ("pruning", "stable"),
    "multi_granular": ("pruning", "stable"),
    # ── Quantization selectors ──
    "gptq": ("quantization", "stable"),
    "awq": ("quantization", "stable"),
    "tacq": ("quantization", "stable"),
}

DEFAULT_ALGORITHM = "eap-ig"

# ── Derived views — DO NOT EDIT; change ALGORITHMS above instead ──────────
STABILITY: dict[str, str] = {name: tier for name, (_cat, tier) in ALGORITHMS.items()}

STABLE_ALGORITHMS = frozenset(n for n, t in STABILITY.items() if t == "stable")
EXPERIMENTAL_ALGORITHMS = frozenset(n for n, t in STABILITY.items() if t == "experimental")
RESEARCH_ALGORITHMS = frozenset(n for n, t in STABILITY.items() if t == "research")

DISCOVERY_ALGORITHMS = frozenset(n for n, (c, _t) in ALGORITHMS.items() if c == "discovery")
PRUNING_ALGORITHMS = frozenset(n for n, (c, _t) in ALGORITHMS.items() if c == "pruning")
QUANTIZATION_ALGORITHMS = frozenset(n for n, (c, _t) in ALGORITHMS.items() if c == "quantization")
SUPPORTED_ALGORITHMS = sorted(ALGORITHMS)


def is_stable(algo: str) -> bool:
    """Return True if the algorithm is production-ready."""
    return STABILITY.get(algo, "unknown") == "stable"


def is_experimental(algo: str) -> bool:
    """Return True if the algorithm is experimental (may fail on larger models)."""
    return STABILITY.get(algo, "unknown") == "experimental"


def is_research(algo: str) -> bool:
    """Return True if the algorithm is research-quality (GPT-2 IOI only)."""
    return STABILITY.get(algo, "unknown") == "research"


def category_of(algo: str) -> str:
    """Return the category ('discovery' / 'pruning' / 'quantization') of an algorithm."""
    return ALGORITHMS.get(algo, ("unknown", "unknown"))[0]


def default_algorithm() -> str:
    """Return the recommended default discovery algorithm."""
    return DEFAULT_ALGORITHM
