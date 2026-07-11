"""
Report classes for stability and robustness evaluation.

Provides structured, JSON-serializable reports for circuit evaluation results:
- E2: StabilityReport - Pillar 3 (Stability) evaluation results
- E3: RobustnessReport - Pillar 4 (Robustness) evaluation results
- E4: StabilityRobustnessReport - Combined Pillar 3 + 4 report
- E4: ComprehensiveEvaluationReport - Full evaluation (all pillars)
"""

from .aggregator import ComprehensiveEvaluationReport, StabilityRobustnessReport
from .robustness_report import RobustnessReport
from .stability_report import StabilityReport

__all__ = [
    "StabilityReport",
    "RobustnessReport",
    "StabilityRobustnessReport",
    "ComprehensiveEvaluationReport",
]
