"""
Export CircuitArtifact from ACDC circuit discovery.

Converts ACDC prune_scores and edge information to unified CircuitArtifact format.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from circuitkit.artifacts import CircuitArtifact, acdc_to_artifact

logger = logging.getLogger(__name__)


def export_circuit_artifact(
    prune_scores: Dict[str, torch.Tensor],
    model_id: str,
    task: str,
    dataset: str,
    edges: Optional[List[Any]] = None,
    threshold: float = 0.0,
    granularity: str = "head",
) -> CircuitArtifact:
    """
    Export ACDC circuit as CircuitArtifact.

    Converts ACDC's internal prune_scores representation to the unified
    CircuitArtifact schema for compatibility with interventions and
    other CircuitKit modules.

    Args:
        prune_scores: Dictionary mapping module names to edge importance tensors
                      (from ACDC discovery algorithm)
        model_id: HuggingFace model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b")
        task: Task name used for discovery (e.g., "ioi", "sva")
        dataset: Dataset name used for discovery
        edges: Optional list of ACDC Edge objects for graph connectivity
        threshold: Importance threshold for filtering edges/nodes (default 0.0 = keep all)
        granularity: Node granularity ("head", "neuron", or "layer")

    Returns:
        CircuitArtifact instance with full graph structure

    Example:
        >>> # After running ACDC discovery
        >>> from circuitkit.backends.acdc.artifact_export import export_circuit_artifact
        >>> artifact = export_circuit_artifact(
        ...     prune_scores=acdc_prune_scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset",
        ...     edges=acdc_edges,
        ...     threshold=0.1
        ... )
        >>> artifact.save_json("circuits/acdc_ioi.json")
    """
    logger.info(
        f"Exporting ACDC circuit: model={model_id}, task={task}, "
        f"dataset={dataset}, granularity={granularity}"
    )

    # Convert using the converter
    artifact = acdc_to_artifact(
        prune_scores=prune_scores,
        model_id=model_id,
        task=task,
        dataset=dataset,
        edges=edges,
        threshold=threshold,
        granularity=granularity,
    )

    # Validate export
    checks = artifact.validate()
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        logger.warning(f"Artifact validation issues: {failed}")

    logger.info(
        f"Successfully exported ACDC circuit: "
        f"{len(artifact.nodes)} nodes, {len(artifact.edges)} edges"
    )

    return artifact


def export_and_save(
    prune_scores: Dict[str, torch.Tensor],
    model_id: str,
    task: str,
    dataset: str,
    output_path: str,
    edges: Optional[List[Any]] = None,
    threshold: float = 0.0,
    granularity: str = "head",
) -> str:
    """
    Export ACDC circuit and save to JSON file.

    Convenience function that exports and immediately saves to disk.

    Args:
        prune_scores: ACDC prune_scores
        model_id: Model identifier
        task: Task name
        dataset: Dataset name
        output_path: Path to save JSON artifact
        edges: Optional ACDC edges
        threshold: Importance threshold
        granularity: Node granularity

    Returns:
        Path to saved artifact file

    Example:
        >>> path = export_and_save(
        ...     prune_scores=scores,
        ...     model_id="gpt2",
        ...     task="ioi",
        ...     dataset="ioi_dataset",
        ...     output_path="circuits/acdc_ioi.json"
        ... )
        >>> print(f"Saved to {path}")
    """
    artifact = export_circuit_artifact(
        prune_scores=prune_scores,
        model_id=model_id,
        task=task,
        dataset=dataset,
        edges=edges,
        threshold=threshold,
        granularity=granularity,
    )

    artifact.save_json(output_path)
    logger.info(f"Saved ACDC circuit artifact to {output_path}")

    return output_path
