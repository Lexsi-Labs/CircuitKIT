"""CircuitKit Selector Registry — unified interface for component scoring.

Consumed by both circuitkit applications (pruning/quantization selectors)
and the experiments experiment framework.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

SelectorFn = Callable[[Any, str, Dict], Dict[str, float]]
_registry: Dict[str, SelectorFn] = {}


def register(name: str):
    """Decorator to register a selector function."""

    def wrapper(fn: SelectorFn) -> SelectorFn:
        _registry[name.lower()] = fn
        return fn

    return wrapper


def get_selector(name: str) -> SelectorFn:
    key = name.lower()
    if key not in _registry:
        raise KeyError(f"Unknown selector: {name!r}. Available: {sorted(_registry.keys())}")
    return _registry[key]


def list_selectors() -> list[str]:
    return sorted(_registry.keys())


# Application selectors (pruning + quantization)
from circuitkit.applications.pruning.selectors import multi_granular_selector  # noqa: F401,E402
from circuitkit.applications.pruning.selectors import taylor_selector  # noqa: F401,E402
from circuitkit.applications.quantization.selectors import awq_selector  # noqa: F401,E402
from circuitkit.applications.quantization.selectors import tacq_selector  # noqa: F401,E402

# Circuit discovery selectors (migrated from experiments/selector_lib/)
from . import eap_selector        # noqa: F401,E402  — registers "eap", "eap-ig"
from . import eap_gp_selector     # noqa: F401,E402  — registers "eap-gp"
from . import relp_selector       # noqa: F401,E402  — registers "relp"
from . import ibcircuit_selector  # noqa: F401,E402  — registers "ibcircuit"
from . import cdt_selector        # noqa: F401,E402  — registers "cdt"
from . import random_selector     # noqa: F401,E402  — registers "random"
from . import magnitude_selector  # noqa: F401,E402  — registers "magnitude"
from . import wanda_selector      # noqa: F401,E402  — registers "wanda"
from . import gptq_selector       # noqa: F401,E402  — registers "gptq"