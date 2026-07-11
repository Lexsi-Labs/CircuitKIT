# FILE: circuitkit/analysis/metrics.py


def calculate_faithfulness(original_output, pruned_output):
    """
    Calculate faithfulness metric between original and pruned outputs.
    Uses multiple metrics for comprehensive evaluation.
    """
    import torch
    import torch.nn.functional as F

    # Ensure tensors are the same shape
    if original_output.shape != pruned_output.shape:
        min_size = min(original_output.numel(), pruned_output.numel())
        original_output = original_output.flatten()[:min_size]
        pruned_output = pruned_output.flatten()[:min_size]

    # L2 norm difference
    l2_diff = (original_output - pruned_output).pow(2).sum().item()

    # KL divergence
    try:
        kl_div = F.kl_div(
            F.log_softmax(pruned_output, dim=-1),
            F.softmax(original_output, dim=-1),
            reduction="sum",
        ).item()
    except (RuntimeError, ValueError):
        kl_div = float("inf")

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        original_output.flatten().unsqueeze(0), pruned_output.flatten().unsqueeze(0)
    ).item()

    # Relative difference
    rel_diff = torch.abs(original_output - pruned_output).sum().item() / (
        torch.abs(original_output).sum().item() + 1e-8
    )

    return {
        "l2_difference": l2_diff,
        "kl_divergence": kl_div,
        "cosine_similarity": cos_sim,
        "relative_difference": rel_diff,
        "faithfulness_score": 1.0 - min(rel_diff, 1.0),  # Higher is better
    }


def calculate_complexity(graph):
    """
    Calculates the complexity of a circuit, e.g., by node or edge count.
    """
    return {"node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges()}
