"""
Bootstrap built-in task registration.

This module provides the single source of truth for registering built-in tasks.
All task registration should go through _bootstrap_builtin_tasks() to avoid
duplication across the codebase.
"""

from typing import Dict

from .registry import is_task_registered, register_task
from .specs import TaskSpec


def _bootstrap_builtin_tasks() -> Dict[str, TaskSpec]:
    """
    Bootstrap and register all built-in tasks.

    Imports all task specs from src/circuitkit/tasks/builtins/ and registers
    them via the task registry. This function is idempotent - it checks if
    tasks are already registered before re-registering.

    Returns:
        Dict mapping task names to TaskSpec instances for all built-in tasks.

    Note:
        This function is called automatically at module import time in
        src/circuitkit/__init__.py to ensure tasks are available throughout
        the application.
    """
    if is_task_registered("ioi"):
        from .registry import _TASKS

        return dict(_TASKS)

    try:
        # Local imports to avoid circular dependencies
        from .builtins.boolq import BoolQTaskSpec
        from .builtins.capital_country import CapitalCountryTaskSpec
        from .builtins.double_io import DoubleIOTaskSpec
        from .builtins.gender_bias import GenderBiasTaskSpec
        from .builtins.glue import GLUETaskSpec
        from .builtins.greater_than import GreaterThanTaskSpec
        from .builtins.gsm8k import GSM8KTaskSpec
        from .builtins.hypernymy import HypernymyTaskSpec
        from .builtins.ifeval import IFEvalTaskSpec
        from .builtins.ioi import IOITaskSpec
        from .builtins.mmlu import MMLUTaskSpec
        from .builtins.sva import SVATaskSpec
        from .builtins.truthfulqa import TruthfulQATaskSpec
        from .builtins.winogrande import WinoGrandeTaskSpec
        from .builtins.winogrande_mc import WinoGrandeMCTaskSpec
        from .builtins.wmdp import WMDPTaskSpec

        # Create instances and register
        specs = [
            IOITaskSpec(),
            GreaterThanTaskSpec(),
            SVATaskSpec(),
            HypernymyTaskSpec(),
            GenderBiasTaskSpec(),
            CapitalCountryTaskSpec(),
            MMLUTaskSpec(),
            GLUETaskSpec(),
            DoubleIOTaskSpec(),
            WMDPTaskSpec(),
            BoolQTaskSpec(),
            WinoGrandeTaskSpec(),
            WinoGrandeMCTaskSpec(),
            TruthfulQATaskSpec(),
            IFEvalTaskSpec(),
            GSM8KTaskSpec(),
        ]

        registered = {}
        for spec in specs:
            if not is_task_registered(spec.name):
                register_task(spec)
            registered[spec.name] = spec

        return registered

    except Exception as e:
        try:
            from ..utils.logging import get_logger

            get_logger("circuitkit").warning(f"Failed to bootstrap built-in tasks: {e}")
        except Exception:
            # Fallback if logging utils not available
            import logging

            logging.getLogger("circuitkit").warning(f"Failed to bootstrap built-in tasks: {e}")
        return {}
