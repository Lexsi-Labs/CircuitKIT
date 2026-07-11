# FILE: circuitkit/analysis/scores.py

from collections import defaultdict

from ..backends.acdc.types import PruneScores
from ..backends.acdc.utils.patchable_model import PatchableModel


def calculate_node_scores_from_edges(
    p_model: PatchableModel, edge_prune_scores: PruneScores
) -> dict[str, float]:
    """
    Calculates an importance score for each source node based on its outgoing edges.

    The score for a node is the average of the absolute scores of all its
    outgoing edges. Nodes with no outgoing edges will not be included.

    Args:
        p_model: The patchable model, used to access the graph of nodes and edges.
        edge_prune_scores: A dictionary mapping destination modules to tensors of
                           edge importance scores.

    Returns:
        A dictionary mapping source node names to their calculated importance score.
    """
    # Use absolute values of scores as importance can be positive or negative in EAP
    abs_edge_scores = {mod: scores.abs() for mod, scores in edge_prune_scores.items()}

    # Group edge scores by their source node
    outgoing_scores_by_node = defaultdict(list)
    for edge in p_model.edges:
        # edge.prune_score looks up the score for this specific edge
        score = edge.prune_score(abs_edge_scores).item()
        outgoing_scores_by_node[edge.src.name].append(score)

    # Calculate the average score for each node. Use Python's built-in sum/len
    # rather than np.mean() so values stay plain Python floats — numpy scalars
    # (numpy.float64) are rejected by torch.load with weights_only=True.
    node_scores = {}
    for node_name, scores in outgoing_scores_by_node.items():
        if scores:
            node_scores[node_name] = sum(scores) / len(scores)

    return node_scores
