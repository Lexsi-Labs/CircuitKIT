"""
Built-in Task Specifications

This module contains the standard task specifications that come with CircuitKit.

Note: Task registration is now centralized in tasks/bootstrap.py to avoid
duplication and circular import issues. These specs are imported here for
convenience and re-exported for direct access.
"""

from .boolq import BoolQTaskSpec
from .capital_country import CapitalCountryTaskSpec
from .gender_bias import GenderBiasTaskSpec
from .glue import GLUETaskSpec
from .greater_than import GreaterThanTaskSpec
from .gsm8k import GSM8KTaskSpec
from .hypernymy import HypernymyTaskSpec
from .ifeval import IFEvalTaskSpec

# Import all built-in task specs for re-export
from .ioi import IOITaskSpec, IOITaskSpecLegacy
from .mmlu import MMLUTaskSpec
from .sva import SVATaskSpec
from .truthfulqa import TruthfulQATaskSpec
from .winogrande import WinoGrandeTaskSpec
from .winogrande_mc import WinoGrandeMCTaskSpec

__all__ = [
    "IOITaskSpec",
    "IOITaskSpecLegacy",  # Deprecated, for backwards compatibility
    "SVATaskSpec",
    "GenderBiasTaskSpec",
    "CapitalCountryTaskSpec",
    "HypernymyTaskSpec",
    "GreaterThanTaskSpec",
    "MMLUTaskSpec",
    "GLUETaskSpec",
    "BoolQTaskSpec",
    "WinoGrandeTaskSpec",
    "WinoGrandeMCTaskSpec",
    "TruthfulQATaskSpec",
    "IFEvalTaskSpec",
    "GSM8KTaskSpec",
]
