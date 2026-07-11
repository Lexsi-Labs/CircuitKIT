"""
Enhanced Activation Steering with Composition and Safety.

This module extends ActivationSteering with:
- Multi-steering composition (SteeringComposer)
- Safety dataset synthesis for adversarial testing
- Evaluation gates for safety checks before applying steering
- Steering interference detection

Provides safe, composable steering for complex multi-correction scenarios.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SteeringComposer:
    """
    Compose multiple steering corrections with interference tracking.

    Enables combining steering vectors from different correction objectives,
    tracking their interactions, and managing joint parameter optimization.

    Attributes:
        steering_dict: Dict mapping correction_name -> steering_vectors dict
        coefficients: Scaling factors for each steering
        composition_history: Track composition changes over time
    """

    def __init__(self):
        """Initialize SteeringComposer."""
        self.steering_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        self.coefficients: Dict[str, float] = {}
        self.composition_history = []
        logger.info("Initialized SteeringComposer")

    def add_steering(
        self,
        name: str,
        steering_vectors: Dict[str, torch.Tensor],
        coefficient: float = 1.0,
    ) -> None:
        """
        Add a steering correction to composition.

        Args:
            name: Name of this steering correction
            steering_vectors: Dict mapping node names to steering vectors
            coefficient: Scaling factor for this correction
        """
        self.steering_dict[name] = steering_vectors
        self.coefficients[name] = coefficient

        logger.info(
            f"Added steering '{name}' with coefficient {coefficient:.3f} "
            f"({len(steering_vectors)} nodes)"
        )

        self.composition_history.append(
            {
                "action": "add",
                "name": name,
                "num_nodes": len(steering_vectors),
                "coefficient": coefficient,
            }
        )

    def remove_steering(self, name: str) -> None:
        """Remove a steering correction from composition."""
        if name in self.steering_dict:
            del self.steering_dict[name]
            del self.coefficients[name]
            logger.info(f"Removed steering '{name}'")
            self.composition_history.append({"action": "remove", "name": name})

    def get_composed_vectors(
        self,
        aggregate: str = "sum",
    ) -> Dict[str, torch.Tensor]:
        """
        Compose all steering vectors into a single set.

        Args:
            aggregate: Aggregation method ("sum", "mean", "weighted_mean")

        Returns:
            Dict mapping node names to composed steering vectors
        """
        if not self.steering_dict:
            return {}

        composed = {}

        if aggregate == "sum":
            for name, vectors in self.steering_dict.items():
                coeff = self.coefficients[name]
                for node_name, vector in vectors.items():
                    if node_name not in composed:
                        composed[node_name] = torch.zeros_like(vector)
                    composed[node_name] += vector * coeff

        elif aggregate == "mean":
            for name, vectors in self.steering_dict.items():
                for node_name, vector in vectors.items():
                    if node_name not in composed:
                        composed[node_name] = torch.zeros_like(vector)
                    composed[node_name] += vector

            # Normalize by number of steerings
            for node_name in composed:
                composed[node_name] /= len(self.steering_dict)

        elif aggregate == "weighted_mean":
            total_coeff = sum(self.coefficients.values())
            for name, vectors in self.steering_dict.items():
                coeff = self.coefficients[name] / total_coeff
                for node_name, vector in vectors.items():
                    if node_name not in composed:
                        composed[node_name] = torch.zeros_like(vector)
                    composed[node_name] += vector * coeff

        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

        return composed

    def compute_interference_matrix(self) -> Dict[Tuple[str, str], float]:
        """
        Compute interference between steering vectors.

        Interference = 1 - cosine_similarity between steering vectors
        Higher interference means steering directions conflict.

        Returns:
            Dict mapping (name1, name2) -> interference_score [0, 2]
        """
        interference = {}

        for name1, vectors1 in self.steering_dict.items():
            for name2, vectors2 in self.steering_dict.items():
                if name1 >= name2:  # Avoid duplicates
                    continue

                # Compute per-node similarities
                similarities = []
                common_nodes = set(vectors1.keys()) & set(vectors2.keys())

                for node_name in common_nodes:
                    v1 = vectors1[node_name].flatten()
                    v2 = vectors2[node_name].flatten()

                    # Cosine similarity
                    cos_sim = torch.nn.functional.cosine_similarity(
                        v1.unsqueeze(0),
                        v2.unsqueeze(0),
                    ).item()
                    similarities.append(cos_sim)

                # Average similarity
                if similarities:
                    avg_similarity = np.mean(similarities)
                else:
                    avg_similarity = 0.0

                # Interference = 1 - similarity (mapped to [0, 2])
                interference_score = 1.0 - avg_similarity
                interference[(name1, name2)] = interference_score

        return interference

    def detect_high_interference(self, threshold: float = 0.7) -> List[Tuple[str, str]]:
        """
        Find steering pairs with high interference.

        Args:
            threshold: Interference threshold (higher = more interference)

        Returns:
            List of (name1, name2) pairs with interference > threshold
        """
        interference = self.compute_interference_matrix()
        return [pair for pair, score in interference.items() if score > threshold]

    def get_parameter_counts(self) -> Dict[str, Any]:
        """
        Get parameter counts for composed steering.

        Returns:
            Dict with total parameters, per-steering parameters, etc.
        """
        total_params = 0
        per_steering_params = {}

        for name, vectors in self.steering_dict.items():
            steering_params = sum(v.numel() for v in vectors.values())
            per_steering_params[name] = steering_params
            total_params += steering_params

        return {
            "total_steering_params": total_params,
            "per_steering": per_steering_params,
            "num_steerings": len(self.steering_dict),
        }

    def summary(self) -> str:
        """Generate summary of composed steerings."""
        lines = ["Steering Composition Summary", "=" * 50]

        for name, vectors in self.steering_dict.items():
            coeff = self.coefficients[name]
            lines.append(f"{name}:")
            lines.append(f"  Coefficient: {coeff:.4f}")
            lines.append(f"  Nodes: {len(vectors)}")

            # Vector statistics
            norms = [v.norm().item() for v in vectors.values()]
            lines.append(f"  Mean norm: {np.mean(norms):.4f}")

        # Interference summary
        interference = self.compute_interference_matrix()
        if interference:
            lines.append("\nInterference Matrix:")
            for (n1, n2), score in sorted(interference.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {n1} <-> {n2}: {score:.3f}")

        return "\n".join(lines)


class SafetyDatasetSynthesis:
    """
    Generate adversarial prompts to test steering safety.

    Creates test cases that check:
    - Steering doesn't break unrelated behaviors
    - Steering is brittle to small input perturbations
    - Steering effects are stable across contexts
    - Steering preserves factual knowledge
    """

    def __init__(self, model: Any, device: str = "cuda"):
        """
        Initialize safety dataset synthesizer.

        Args:
            model: Model to generate prompts for
            device: Compute device
        """
        self.model = model
        self.device = device
        logger.info("Initialized SafetyDatasetSynthesis")

    def generate_adversarial_prompts(
        self,
        base_prompt: str,
        num_variations: int = 10,
        perturbation_type: str = "paraphrase",
    ) -> List[str]:
        """
        Generate adversarial variations of a prompt.

        Args:
            base_prompt: Base prompt to perturb
            num_variations: Number of variations to generate
            perturbation_type: Type of perturbation ("paraphrase", "noise", "reorder")

        Returns:
            List of perturbed prompts
        """
        variations = [base_prompt]

        if perturbation_type == "paraphrase":
            # Simple paraphrasing by token shuffling or synonym replacement
            for i in range(num_variations - 1):
                perturbed = self._paraphrase_prompt(base_prompt)
                variations.append(perturbed)

        elif perturbation_type == "noise":
            # Add character-level noise
            for i in range(num_variations - 1):
                perturbed = self._add_noise_to_prompt(base_prompt)
                variations.append(perturbed)

        elif perturbation_type == "reorder":
            # Shuffle token order slightly
            for i in range(num_variations - 1):
                perturbed = self._reorder_prompt(base_prompt)
                variations.append(perturbed)

        logger.info(f"Generated {len(variations)} adversarial prompts")
        return variations

    def _paraphrase_prompt(self, prompt: str) -> str:
        """Simple paraphrasing via word order shuffling."""
        words = prompt.split()
        if len(words) > 3:
            # Shuffle middle words
            middle = words[1:-1]
            np.random.shuffle(middle)
            words = [words[0]] + middle + [words[-1]]
        return " ".join(words)

    def _add_noise_to_prompt(self, prompt: str) -> str:
        """Add random character-level noise."""
        chars = list(prompt)
        noise_positions = np.random.choice(len(chars), size=max(1, len(chars) // 20), replace=False)
        for pos in noise_positions:
            if chars[pos].isalpha():
                chars[pos] = chr(ord("a") + np.random.randint(26))
        return "".join(chars)

    def _reorder_prompt(self, prompt: str) -> str:
        """Reorder tokens slightly."""
        words = prompt.split()
        for _ in range(len(words) // 4):
            i = np.random.randint(len(words) - 1)
            words[i], words[i + 1] = words[i + 1], words[i]
        return " ".join(words)

    def create_safety_benchmark(
        self,
        core_prompts: List[str],
        related_prompts: List[str],
        num_adversarial_per_core: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Create comprehensive safety benchmark.

        Args:
            core_prompts: Core prompts steering should handle
            related_prompts: Related prompts that should still work
            num_adversarial_per_core: Adversarial variations per core prompt

        Returns:
            Dict with "core", "related", and "adversarial" prompt lists
        """
        benchmark = {
            "core": core_prompts,
            "related": related_prompts,
            "adversarial": [],
        }

        for prompt in core_prompts:
            adversarial = self.generate_adversarial_prompts(
                prompt,
                num_variations=num_adversarial_per_core,
            )
            benchmark["adversarial"].extend(adversarial[1:])  # Skip original

        logger.info(
            f"Created safety benchmark: {len(benchmark['core'])} core, "
            f"{len(benchmark['related'])} related, "
            f"{len(benchmark['adversarial'])} adversarial"
        )

        return benchmark


class SteeringEvaluationGates:
    """
    Pre-steering validation gates to prevent unsafe steering application.

    Checks:
    - Steering magnitudes are reasonable
    - Activation bounds are preserved
    - Model semantics not violated
    - Steering consistency across contexts
    """

    def __init__(self, model: Any, device: str = "cuda"):
        """
        Initialize evaluation gates.

        Args:
            model: Model to validate
            device: Compute device
        """
        self.model = model
        self.device = device
        self.baseline_activations: Dict[str, torch.Tensor] = {}
        logger.info("Initialized SteeringEvaluationGates")

    def set_baseline_activations(
        self,
        baseline_prompts: List[str],
    ) -> None:
        """
        Establish baseline activations for comparison.

        Args:
            baseline_prompts: List of prompts to establish baseline
        """
        logger.info(f"Setting baseline from {len(baseline_prompts)} prompts")
        # Implementation would collect activations from baseline_prompts
        self.baseline_activations = {}  # Would be populated in real usage

    def check_activation_bounds(
        self,
        steering_vectors: Dict[str, torch.Tensor],
        max_magnitude: float = 5.0,
    ) -> Tuple[bool, List[str]]:
        """
        Verify steering doesn't exceed activation magnitude bounds.

        Args:
            steering_vectors: Steering vectors to check
            max_magnitude: Maximum allowed steering magnitude

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        for node_name, vector in steering_vectors.items():
            magnitude = vector.norm().item()
            if magnitude > max_magnitude:
                violations.append(f"{node_name}: magnitude {magnitude:.3f} > {max_magnitude}")

        is_valid = len(violations) == 0
        if not is_valid:
            logger.warning(f"Activation bounds violated: {len(violations)} nodes exceed threshold")

        return is_valid, violations

    def check_steering_consistency(
        self,
        steering_vectors: Dict[str, torch.Tensor],
        min_consistency: float = 0.8,
    ) -> Tuple[bool, float]:
        """
        Check steering is consistent across vector dimensions.

        Args:
            steering_vectors: Steering vectors to check
            min_consistency: Minimum consistency threshold

        Returns:
            (is_valid, consistency_score)
        """
        if not steering_vectors:
            return True, 1.0

        # Compute variance in vector norms
        norms = [v.norm().item() for v in steering_vectors.values()]
        if len(norms) < 2:
            return True, 1.0

        norm_std = np.std(norms)
        norm_mean = np.mean(norms)

        # Consistency = 1 - (std/mean)
        consistency = max(0.0, 1.0 - (norm_std / (norm_mean + 1e-8)))

        is_valid = consistency >= min_consistency
        if not is_valid:
            logger.warning(f"Steering consistency low: {consistency:.3f} < {min_consistency}")

        return is_valid, consistency

    def check_semantic_preservation(
        self,
        base_output: torch.Tensor,
        steered_output: torch.Tensor,
        min_similarity: float = 0.9,
    ) -> Tuple[bool, float]:
        """
        Check that steering preserves semantic structure.

        Uses cosine similarity between base and steered outputs.

        Args:
            base_output: Base model output (logits)
            steered_output: Output with steering applied
            min_similarity: Minimum similarity threshold

        Returns:
            (is_valid, similarity_score)
        """
        # Flatten outputs
        base_flat = base_output.flatten()
        steered_flat = steered_output.flatten()

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            base_flat.unsqueeze(0),
            steered_flat.unsqueeze(0),
        ).item()

        is_valid = similarity >= min_similarity
        if not is_valid:
            logger.warning(f"Semantic preservation low: {similarity:.3f} < {min_similarity}")

        return is_valid, similarity

    def run_all_checks(
        self,
        steering_vectors: Dict[str, torch.Tensor],
        base_output: Optional[torch.Tensor] = None,
        steered_output: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Run all safety checks.

        Args:
            steering_vectors: Steering vectors to validate
            base_output: Optional base model output
            steered_output: Optional steered output

        Returns:
            Dict with check results and overall pass/fail
        """
        results = {}

        # Check 1: Activation bounds
        bounds_valid, bounds_violations = self.check_activation_bounds(steering_vectors)
        results["activation_bounds"] = {
            "valid": bounds_valid,
            "violations": bounds_violations,
        }

        # Check 2: Consistency
        consistency_valid, consistency_score = self.check_steering_consistency(steering_vectors)
        results["consistency"] = {
            "valid": consistency_valid,
            "score": consistency_score,
        }

        # Check 3: Semantic preservation
        if base_output is not None and steered_output is not None:
            semantic_valid, semantic_similarity = self.check_semantic_preservation(
                base_output, steered_output
            )
            results["semantic_preservation"] = {
                "valid": semantic_valid,
                "similarity": semantic_similarity,
            }

        # Overall result
        all_valid = all(
            v.get("valid", True) if isinstance(v, dict) else v for v in results.values()
        )
        results["overall_valid"] = all_valid

        if all_valid:
            logger.info("All safety checks passed")
        else:
            failed_checks = [
                k for k, v in results.items() if isinstance(v, dict) and not v.get("valid", True)
            ]
            logger.warning(f"Safety checks failed: {failed_checks}")

        return results
