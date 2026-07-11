"""
CorruptionEffectiveness: Metrics for measuring corruption impact on model outputs.

Tracks how corruption affects:
- avg_impact: How much model output logits change (embedding distance)
- label_consistency: Percentage of answers remaining valid/comparable
- semantic_shift: Embedding distance between clean and corrupted texts
- difficulty_impact: How corruption changes task difficulty (accuracy change)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CorruptionEffectiveness:
    """Metrics measuring corruption effectiveness and impact.

    Attributes:
        strategy_name: Name of corruption strategy applied
        num_examples: Number of examples tested
        avg_impact: Average output change (0.0 to 1.0, higher = larger change)
        label_consistency: Percentage of answers staying valid [0.0, 1.0]
        semantic_shift: Average embedding distance before/after corruption
        difficulty_impact: Change in task difficulty (negative = easier, positive = harder)
        severity: Average severity scores from validate() (0.0 to 1.0)
        validation_pass_rate: Percentage of corruptions that passed validate()
        error_rate: Percentage of corruptions that failed/raised errors
        per_example_impacts: List of per-example impact scores
        notes: Optional notes about corruption effectiveness
    """

    strategy_name: str
    num_examples: int
    avg_impact: float = 0.0
    label_consistency: float = 0.0
    semantic_shift: float = 0.0
    difficulty_impact: float = 0.0
    severity: float = 0.0
    validation_pass_rate: float = 0.0
    error_rate: float = 0.0
    per_example_impacts: List[float] = field(default_factory=list)
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate effectiveness metrics."""
        # Clamp all percentages to [0, 1]
        self.avg_impact = max(0.0, min(1.0, self.avg_impact))
        self.label_consistency = max(0.0, min(1.0, self.label_consistency))
        self.severity = max(0.0, min(1.0, self.severity))
        self.validation_pass_rate = max(0.0, min(1.0, self.validation_pass_rate))
        self.error_rate = max(0.0, min(1.0, self.error_rate))

    def is_effective(self, impact_threshold: float = 0.3) -> bool:
        """Check if corruption is effective.

        Effective = sufficient impact AND high validation pass rate

        Args:
            impact_threshold: Minimum avg_impact for effectiveness (default 0.3)

        Returns:
            True if corruption meets effectiveness criteria
        """
        return (
            self.avg_impact >= impact_threshold
            and self.validation_pass_rate >= 0.8
            and self.error_rate <= 0.2
        )

    def summary(self) -> str:
        """Generate human-readable summary of effectiveness.

        Returns:
            Formatted summary string
        """
        effectiveness = "EFFECTIVE" if self.is_effective() else "INEFFECTIVE"

        return f"""
CorruptionEffectiveness Report
==============================
Strategy: {self.strategy_name}
Status: {effectiveness}
Examples: {self.num_examples}

Impact Metrics:
  Average Output Impact: {self.avg_impact:.3f} (0=no change, 1=large change)
  Label Consistency: {self.label_consistency:.1%} (answers stay valid)
  Semantic Shift: {self.semantic_shift:.3f} (embedding distance)
  Difficulty Impact: {self.difficulty_impact:+.3f} (negative=easier, positive=harder)

Quality Metrics:
  Average Severity: {self.severity:.3f} (0=mild, 1=severe)
  Validation Pass Rate: {self.validation_pass_rate:.1%}
  Error Rate: {self.error_rate:.1%}

{f"Notes: {self.notes}" if self.notes else ""}
""".strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dict representation of effectiveness metrics
        """
        return {
            "strategy_name": self.strategy_name,
            "num_examples": self.num_examples,
            "avg_impact": self.avg_impact,
            "label_consistency": self.label_consistency,
            "semantic_shift": self.semantic_shift,
            "difficulty_impact": self.difficulty_impact,
            "severity": self.severity,
            "validation_pass_rate": self.validation_pass_rate,
            "error_rate": self.error_rate,
            "per_example_impacts": self.per_example_impacts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorruptionEffectiveness":
        """Create from dictionary (for deserialization).

        Args:
            data: Dictionary with effectiveness metrics

        Returns:
            CorruptionEffectiveness instance
        """
        return cls(**data)


class EffectivenessCalculator:
    """Calculates corruption effectiveness metrics.

    Computes impact scores from model outputs, embeddings, and validation results.
    """

    @staticmethod
    def calculate_output_impact(
        clean_logits: List[float],
        corrupted_logits: List[float],
    ) -> float:
        """Calculate impact on model logits.

        Args:
            clean_logits: Model logits on clean examples
            corrupted_logits: Model logits on corrupted examples

        Returns:
            Impact score in [0.0, 1.0] (Euclidean distance normalized)
        """
        if not clean_logits or len(clean_logits) != len(corrupted_logits):
            return 0.0

        # Calculate L2 distance
        sum_sq = sum((c - d) ** 2 for c, d in zip(clean_logits, corrupted_logits))
        euclidean = math.sqrt(sum_sq)

        # Normalize by max logit magnitude to get [0, 1]
        max_magnitude = max(
            abs(max(clean_logits + corrupted_logits)),
            abs(min(clean_logits + corrupted_logits)),
            1.0,
        )

        impact = min(1.0, euclidean / (max_magnitude * math.sqrt(len(clean_logits))))
        return impact

    @staticmethod
    def calculate_semantic_shift(
        clean_embedding: List[float],
        corrupted_embedding: List[float],
    ) -> float:
        """Calculate semantic shift using embedding distance.

        Args:
            clean_embedding: Embedding vector for clean text
            corrupted_embedding: Embedding vector for corrupted text

        Returns:
            Cosine distance in [0.0, 1.0]
        """
        if not clean_embedding or len(clean_embedding) != len(corrupted_embedding):
            return 0.0

        # Calculate cosine similarity
        dot_product = sum(c * d for c, d in zip(clean_embedding, corrupted_embedding))
        clean_norm = math.sqrt(sum(c**2 for c in clean_embedding))
        corrupted_norm = math.sqrt(sum(d**2 for d in corrupted_embedding))

        if clean_norm == 0 or corrupted_norm == 0:
            return 0.0

        cosine_sim = dot_product / (clean_norm * corrupted_norm)

        # Convert to distance [0, 1]
        distance = (1.0 - cosine_sim) / 2.0
        return max(0.0, min(1.0, distance))

    @staticmethod
    def calculate_label_consistency(
        clean_labels: List[str],
        corrupted_labels: List[str],
    ) -> float:
        """Calculate how consistently answers are preserved.

        Args:
            clean_labels: Original answer labels
            corrupted_labels: Answer labels after corruption

        Returns:
            Consistency score in [0.0, 1.0]
        """
        if not clean_labels or len(clean_labels) != len(corrupted_labels):
            return 0.0

        # Check if labels are compatible (e.g., same answer type)
        compatible = sum(
            1
            for c, d in zip(clean_labels, corrupted_labels)
            if c.strip().lower() == d.strip().lower() and c != ""
        )

        return compatible / len(clean_labels)

    @staticmethod
    def calculate_difficulty_impact(
        clean_accuracy: float,
        corrupted_accuracy: float,
    ) -> float:
        """Calculate impact on task difficulty.

        Args:
            clean_accuracy: Model accuracy on clean examples
            corrupted_accuracy: Model accuracy on corrupted examples

        Returns:
            Difficulty impact (negative = easier, positive = harder)
        """
        return corrupted_accuracy - clean_accuracy

    @staticmethod
    def aggregate_effectiveness(
        strategy_name: str,
        impacts: List[float],
        consistency_scores: List[float],
        semantic_shifts: List[float],
        validation_severities: List[float],
        validation_passes: List[bool],
        errors_occurred: List[bool],
        difficulty_impact: Optional[float] = None,
    ) -> CorruptionEffectiveness:
        """Aggregate per-example metrics into overall effectiveness.

        Args:
            strategy_name: Name of corruption strategy
            impacts: Per-example output impact scores
            consistency_scores: Per-example label consistency scores
            semantic_shifts: Per-example semantic shift distances
            validation_severities: Per-example validation severity scores
            validation_passes: Per-example validation results
            errors_occurred: Per-example error flags
            difficulty_impact: Optional overall difficulty impact

        Returns:
            CorruptionEffectiveness with aggregated metrics
        """
        num_examples = len(impacts)
        if num_examples == 0:
            return CorruptionEffectiveness(
                strategy_name=strategy_name, num_examples=0, notes="No examples evaluated"
            )

        # Calculate averages
        avg_impact = sum(impacts) / num_examples if impacts else 0.0
        avg_consistency = sum(consistency_scores) / num_examples if consistency_scores else 0.0
        avg_semantic = sum(semantic_shifts) / num_examples if semantic_shifts else 0.0
        avg_severity = sum(validation_severities) / num_examples if validation_severities else 0.0

        # Calculate pass rates
        validation_pass_rate = sum(validation_passes) / num_examples if validation_passes else 0.0
        error_rate = sum(errors_occurred) / num_examples if errors_occurred else 0.0

        return CorruptionEffectiveness(
            strategy_name=strategy_name,
            num_examples=num_examples,
            avg_impact=avg_impact,
            label_consistency=avg_consistency,
            semantic_shift=avg_semantic,
            difficulty_impact=difficulty_impact or 0.0,
            severity=avg_severity,
            validation_pass_rate=validation_pass_rate,
            error_rate=error_rate,
            per_example_impacts=impacts,
        )
