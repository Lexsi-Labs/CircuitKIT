"""
Export CircuitArtifact from IBCircuit discovery.

Converts IBCircuit neuron importance scores to unified CircuitArtifact format.
"""

import logging
from typing import Dict

from circuitkit.artifacts import CircuitArtifact, ibcircuit_to_artifact

logger = logging.getLogger(__name__)


def export_circuit_artifact(
    neuron_scores: Dict[str, float],
    model_id: str,
    task: str,
    dataset: str,
    threshold: float = 0.0,
    granularity: str = "neuron",
) -> CircuitArtifact:
    """
    Export IBCircuit circuit as CircuitArtifact.

    Converts IBCircuit's neuron importance scores to the unified CircuitArtifact
    schema for compatibility with interventions and other CircuitKit modules.

    Args:
        neuron_scores: Dictionary mapping neuron identifiers to importance scores
                       (e.g., {"L0.N10": 0.75, "L1.N5": 0.85})
        model_id: HuggingFace model identifier
        task: Task name used for discovery
        dataset: Dataset name used for discovery
        threshold: Importance threshold for filtering neurons (default 0.0 = keep all)
        granularity: Node granularity ("neuron" typical for IBCircuit, or "layer")

    Returns:
        CircuitArtifact instance with neuron-level graph structure

    Example:
        >>> # After running IBCircuit discovery
        >>> from circuitkit.backends.ibcircuit.artifact_export import export_circuit_artifact
        >>> artifact = export_circuit_artifact(
        ...     neuron_scores=ibcircuit_scores,
        ...     model_id="pythia-70m",
        ...     task="greater_than",
        ...     dataset="numeric",
        ...     threshold=0.4
        ... )
        >>> artifact.save_json("circuits/ibcircuit_gt.json")
    """
    logger.info(
        f"Exporting IBCircuit circuit: model={model_id}, task={task}, "
        f"dataset={dataset}, granularity={granularity}"
    )

    # Convert using the converter
    artifact = ibcircuit_to_artifact(
        node_scores=neuron_scores,
        model_id=model_id,
        task=task,
        dataset=dataset,
        threshold=threshold,
        granularity=granularity,
    )

    # Validate export
    checks = artifact.validate()
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        logger.warning(f"Artifact validation issues: {failed}")

    logger.info(f"Successfully exported IBCircuit circuit: " f"{len(artifact.nodes)} neurons")

    return artifact


def export_and_save(
    neuron_scores: Dict[str, float],
    model_id: str,
    task: str,
    dataset: str,
    output_path: str,
    threshold: float = 0.0,
    granularity: str = "neuron",
) -> str:
    """
    Export IBCircuit circuit and save to JSON file.

    Convenience function that exports and immediately saves to disk.

    Args:
        neuron_scores: IBCircuit neuron_scores
        model_id: Model identifier
        task: Task name
        dataset: Dataset name
        output_path: Path to save JSON artifact
        threshold: Importance threshold
        granularity: Node granularity

    Returns:
        Path to saved artifact file

    Example:
        >>> path = export_and_save(
        ...     neuron_scores=scores,
        ...     model_id="pythia-70m",
        ...     task="greater_than",
        ...     dataset="numeric",
        ...     output_path="circuits/ibcircuit_gt.json"
        ... )
        >>> print(f"Saved to {path}")
    """
    artifact = export_circuit_artifact(
        neuron_scores=neuron_scores,
        model_id=model_id,
        task=task,
        dataset=dataset,
        threshold=threshold,
        granularity=granularity,
    )

    artifact.save_json(output_path)
    logger.info(f"Saved IBCircuit circuit artifact to {output_path}")

    return output_path
