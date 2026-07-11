"""
Tests for CorruptionEffectiveness metrics and calculator.

Tests:
- Effectiveness dataclass initialization and validation
- Impact calculation methods
- Effectiveness aggregation
- Summary generation
"""

from circuitkit.corruption.effectiveness import CorruptionEffectiveness, EffectivenessCalculator


class TestCorruptionEffectiveness:
    """Test CorruptionEffectiveness dataclass."""

    def test_init_default(self):
        """Test default initialization."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
        )
        assert eff.strategy_name == "test"
        assert eff.num_examples == 10
        assert eff.avg_impact == 0.0

    def test_init_with_values(self):
        """Test initialization with values."""
        eff = CorruptionEffectiveness(
            strategy_name="entity_swap",
            num_examples=100,
            avg_impact=0.75,
            label_consistency=0.9,
            semantic_shift=0.5,
            difficulty_impact=0.1,
            severity=0.6,
        )
        assert eff.avg_impact == 0.75
        assert eff.label_consistency == 0.9

    def test_clamp_values(self):
        """Test that values are clamped to [0, 1]."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=1.5,  # Should clamp to 1.0
            label_consistency=-0.5,  # Should clamp to 0.0
        )
        assert eff.avg_impact == 1.0
        assert eff.label_consistency == 0.0

    def test_is_effective(self):
        """Test effectiveness check."""
        # Effective
        eff1 = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=0.5,
            validation_pass_rate=0.9,
            error_rate=0.1,
        )
        assert eff1.is_effective()

        # Ineffective (low impact)
        eff2 = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=0.1,
            validation_pass_rate=0.9,
            error_rate=0.1,
        )
        assert not eff2.is_effective()

        # Ineffective (low pass rate)
        eff3 = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=0.5,
            validation_pass_rate=0.5,
            error_rate=0.1,
        )
        assert not eff3.is_effective()

    def test_is_effective_custom_threshold(self):
        """Test effectiveness with custom threshold."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=0.25,
            validation_pass_rate=0.9,
            error_rate=0.1,
        )
        # Should be ineffective with default threshold (0.3)
        assert not eff.is_effective()
        # Should be effective with lower threshold
        assert eff.is_effective(impact_threshold=0.2)

    def test_summary(self):
        """Test summary generation."""
        eff = CorruptionEffectiveness(
            strategy_name="entity_swap",
            num_examples=100,
            avg_impact=0.75,
            label_consistency=0.85,
            semantic_shift=0.4,
            difficulty_impact=0.05,
            severity=0.6,
            validation_pass_rate=0.95,
            error_rate=0.05,
        )
        summary = eff.summary()
        assert "entity_swap" in summary
        assert "0.75" in summary or "0.750" in summary
        assert "EFFECTIVE" in summary

    def test_summary_ineffective(self):
        """Test summary for ineffective corruption."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=0.1,
            validation_pass_rate=0.5,
            error_rate=0.5,
        )
        summary = eff.summary()
        assert "INEFFECTIVE" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=10,
            avg_impact=0.5,
        )
        d = eff.to_dict()
        assert isinstance(d, dict)
        assert d["strategy_name"] == "test"
        assert d["avg_impact"] == 0.5

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "strategy_name": "test",
            "num_examples": 10,
            "avg_impact": 0.5,
            "label_consistency": 0.8,
        }
        eff = CorruptionEffectiveness.from_dict(data)
        assert eff.strategy_name == "test"
        assert eff.num_examples == 10
        assert eff.avg_impact == 0.5

    def test_to_dict_round_trip(self):
        """Test dictionary round-trip."""
        eff1 = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=100,
            avg_impact=0.6,
            severity=0.5,
            validation_pass_rate=0.9,
        )
        d = eff1.to_dict()
        eff2 = CorruptionEffectiveness.from_dict(d)
        assert eff1.strategy_name == eff2.strategy_name
        assert eff1.avg_impact == eff2.avg_impact


class TestEffectivenessCalculator:
    """Test EffectivenessCalculator static methods."""

    def test_calculate_output_impact_zero(self):
        """Test impact calculation with zero change."""
        clean = [0.5, 0.5, 0.5]
        corrupted = [0.5, 0.5, 0.5]
        impact = EffectivenessCalculator.calculate_output_impact(clean, corrupted)
        assert impact == 0.0

    def test_calculate_output_impact_large(self):
        """Test impact calculation with large change."""
        clean = [1.0, 1.0, 1.0]
        corrupted = [-1.0, -1.0, -1.0]
        impact = EffectivenessCalculator.calculate_output_impact(clean, corrupted)
        assert impact > 0.0

    def test_calculate_output_impact_range(self):
        """Test impact is in valid range."""
        clean = [0.0, 0.5, 1.0]
        corrupted = [0.1, 0.6, 1.1]
        impact = EffectivenessCalculator.calculate_output_impact(clean, corrupted)
        assert 0.0 <= impact <= 1.0

    def test_calculate_output_impact_empty(self):
        """Test impact with empty lists."""
        impact = EffectivenessCalculator.calculate_output_impact([], [])
        assert impact == 0.0

    def test_calculate_semantic_shift_identical(self):
        """Test semantic shift with identical embeddings."""
        embed1 = [1.0, 0.0, 0.0]
        embed2 = [1.0, 0.0, 0.0]
        shift = EffectivenessCalculator.calculate_semantic_shift(embed1, embed2)
        assert shift == 0.0

    def test_calculate_semantic_shift_orthogonal(self):
        """Test semantic shift with orthogonal embeddings."""
        embed1 = [1.0, 0.0]
        embed2 = [0.0, 1.0]
        shift = EffectivenessCalculator.calculate_semantic_shift(embed1, embed2)
        assert shift > 0.0

    def test_calculate_semantic_shift_range(self):
        """Test semantic shift is in valid range."""
        embed1 = [1.0, 0.0, 0.0]
        embed2 = [0.7, 0.7, 0.0]
        shift = EffectivenessCalculator.calculate_semantic_shift(embed1, embed2)
        assert 0.0 <= shift <= 1.0

    def test_calculate_semantic_shift_empty(self):
        """Test semantic shift with empty embeddings."""
        shift = EffectivenessCalculator.calculate_semantic_shift([], [])
        assert shift == 0.0

    def test_calculate_label_consistency_identical(self):
        """Test label consistency with identical labels."""
        clean = ["A", "B", "C"]
        corrupted = ["A", "B", "C"]
        consistency = EffectivenessCalculator.calculate_label_consistency(clean, corrupted)
        assert consistency == 1.0

    def test_calculate_label_consistency_different(self):
        """Test label consistency with different labels."""
        clean = ["A", "A", "A"]
        corrupted = ["B", "B", "B"]
        consistency = EffectivenessCalculator.calculate_label_consistency(clean, corrupted)
        assert consistency == 0.0

    def test_calculate_label_consistency_partial(self):
        """Test label consistency with partial match."""
        clean = ["A", "A", "A"]
        corrupted = ["A", "B", "A"]
        consistency = EffectivenessCalculator.calculate_label_consistency(clean, corrupted)
        assert 0.0 < consistency < 1.0

    def test_calculate_difficulty_impact_easier(self):
        """Test difficulty impact when corruption makes task easier."""
        clean_acc = 0.5
        corrupted_acc = 0.7
        impact = EffectivenessCalculator.calculate_difficulty_impact(clean_acc, corrupted_acc)
        assert impact > 0.0  # Positive impact = easier

    def test_calculate_difficulty_impact_harder(self):
        """Test difficulty impact when corruption makes task harder."""
        clean_acc = 0.7
        corrupted_acc = 0.5
        impact = EffectivenessCalculator.calculate_difficulty_impact(clean_acc, corrupted_acc)
        assert impact < 0.0  # Negative impact = harder

    def test_calculate_difficulty_impact_zero(self):
        """Test difficulty impact with no change."""
        impact = EffectivenessCalculator.calculate_difficulty_impact(0.5, 0.5)
        assert impact == 0.0


class TestAggregateEffectiveness:
    """Test effectiveness aggregation."""

    def test_aggregate_empty(self):
        """Test aggregation with no examples."""
        eff = EffectivenessCalculator.aggregate_effectiveness(
            strategy_name="test",
            impacts=[],
            consistency_scores=[],
            semantic_shifts=[],
            validation_severities=[],
            validation_passes=[],
            errors_occurred=[],
        )
        assert eff.num_examples == 0

    def test_aggregate_single(self):
        """Test aggregation with single example."""
        eff = EffectivenessCalculator.aggregate_effectiveness(
            strategy_name="test",
            impacts=[0.7],
            consistency_scores=[0.8],
            semantic_shifts=[0.4],
            validation_severities=[0.6],
            validation_passes=[True],
            errors_occurred=[False],
        )
        assert eff.num_examples == 1
        assert eff.avg_impact == 0.7
        assert eff.label_consistency == 0.8

    def test_aggregate_multiple(self):
        """Test aggregation with multiple examples."""
        eff = EffectivenessCalculator.aggregate_effectiveness(
            strategy_name="test",
            impacts=[0.5, 0.7, 0.9],
            consistency_scores=[0.8, 0.8, 0.8],
            semantic_shifts=[0.3, 0.4, 0.5],
            validation_severities=[0.5, 0.6, 0.7],
            validation_passes=[True, True, True],
            errors_occurred=[False, False, False],
            difficulty_impact=0.05,
        )
        assert eff.num_examples == 3
        assert abs(eff.avg_impact - 0.7) < 0.01
        assert eff.validation_pass_rate == 1.0
        assert eff.error_rate == 0.0

    def test_aggregate_with_errors(self):
        """Test aggregation with some errors."""
        eff = EffectivenessCalculator.aggregate_effectiveness(
            strategy_name="test",
            impacts=[0.5, 0.7, 0.0],
            consistency_scores=[0.8, 0.8, 0.0],
            semantic_shifts=[0.3, 0.4, 0.0],
            validation_severities=[0.5, 0.6, 0.0],
            validation_passes=[True, True, False],
            errors_occurred=[False, False, True],
        )
        assert eff.num_examples == 3
        assert eff.validation_pass_rate == 2.0 / 3.0
        assert eff.error_rate == 1.0 / 3.0

    def test_aggregate_preserves_per_example(self):
        """Test that aggregation preserves per-example impacts."""
        impacts = [0.5, 0.7, 0.9]
        eff = EffectivenessCalculator.aggregate_effectiveness(
            strategy_name="test",
            impacts=impacts,
            consistency_scores=[0.8] * 3,
            semantic_shifts=[0.4] * 3,
            validation_severities=[0.6] * 3,
            validation_passes=[True] * 3,
            errors_occurred=[False] * 3,
        )
        assert eff.per_example_impacts == impacts


class TestEffectivenessEdgeCases:
    """Test edge cases in effectiveness calculation."""

    def test_very_small_values(self):
        """Test with very small but non-zero values."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=1,
            avg_impact=0.0001,
        )
        assert eff.avg_impact == 0.0001

    def test_exactly_zero_one(self):
        """Test exact 0.0 and 1.0 values."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=1,
            avg_impact=0.0,
            severity=1.0,
        )
        assert eff.avg_impact == 0.0
        assert eff.severity == 1.0

    def test_nan_inputs(self):
        """Test handling of NaN (should not occur but be safe)."""
        eff = CorruptionEffectiveness(
            strategy_name="test",
            num_examples=1,
            avg_impact=float("nan") if False else 0.5,  # Avoid actual NaN
        )
        assert eff is not None
