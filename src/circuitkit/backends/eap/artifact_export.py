"""
Export CircuitArtifact from EAP circuit discovery.

Converts EAP node attribution scores to unified CircuitArtifact format.
"""

import logging
from typing import Any, Dict, Optional

from circuitkit.artifacts import CircuitArtifact, eap_to_artifact

logger = logging.getLogger(__name__)


def export_circuit_artifact(
    node_scores: Dict[str, float],
    model_id: str,
    task: str,
    dataset: str,
    graph: Optional[Any] = None,
    threshold: float = 0.0,
    granularity: str = "head",
) -> CircuitArtifact:
    """
    Export EAP circuit as CircuitArtifact.

    Converts EAP's node attribution scores to the unified CircuitArtifact
    schema for compatibility with interventions and other CircuitKit modules.

    Args:
        node_scores: Dictionary mapping node names to importance scores
                     (e.g., {"A0.0": 0.92, "MLP 0": 0.55})
        model_id: HuggingFace model identifier
        task: Task name used for discovery
        dataset: Dataset name used for discovery
        graph: Optional EAP Graph object for edge connectivity information
        threshold: Importance threshold for filtering nodes (default 0.0 = keep all)
        granularity: Node granularity ("head", "neuron", or "layer")

    Returns:
        CircuitArtifact instance with full graph structure

    Example:
        >>> # After running EAP discovery
        >>> from circuitkit.backends.eap.artifact_export import export_circuit_artifact
        >>> artifact = export_circuit_artifact(
        ...     node_scores=eap_node_scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset",
        ...     graph=eap_graph,
        ...     threshold=0.3
        ... )
        >>> artifact.save_json("circuits/eap_ioi.json")
    """
    logger.info(
        f"Exporting EAP circuit: model={model_id}, task={task}, "
        f"dataset={dataset}, granularity={granularity}"
    )

    # Convert using the converter
    artifact = eap_to_artifact(
        node_scores=node_scores,
        model_id=model_id,
        task=task,
        dataset=dataset,
        graph=graph,
        threshold=threshold,
        granularity=granularity,
    )

    # Validate export
    checks = artifact.validate()
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        logger.warning(f"Artifact validation issues: {failed}")

    logger.info(
        f"Successfully exported EAP circuit: "
        f"{len(artifact.nodes)} nodes, {len(artifact.edges)} edges"
    )

    return artifact


def export_and_save(
    node_scores: Dict[str, float],
    model_id: str,
    task: str,
    dataset: str,
    output_path: str,
    graph: Optional[Any] = None,
    threshold: float = 0.0,
    granularity: str = "head",
) -> str:
    """
    Export EAP circuit and save to JSON file.

    Convenience function that exports and immediately saves to disk.

    Args:
        node_scores: EAP node_scores
        model_id: Model identifier
        task: Task name
        dataset: Dataset name
        output_path: Path to save JSON artifact
        graph: Optional EAP Graph
        threshold: Importance threshold
        granularity: Node granularity

    Returns:
        Path to saved artifact file

    Example:
        >>> path = export_and_save(
        ...     node_scores=scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset",
        ...     output_path="circuits/eap_ioi.json"
        ... )
        >>> print(f"Saved to {path}")
    """
    artifact = export_circuit_artifact(
        node_scores=node_scores,
        model_id=model_id,
        task=task,
        dataset=dataset,
        graph=graph,
        threshold=threshold,
        granularity=granularity,
    )

    artifact.save_json(output_path)
    logger.info(f"Saved EAP circuit artifact to {output_path}")

    return output_path
