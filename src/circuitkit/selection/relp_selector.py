"""RelP selector — Relevance Patching via LRP-style detach hooks."""
import torch
import logging
from circuitkit.selection import register

logger = logging.getLogger(__name__)


@register("relp")
def relp_selector(model, task_name: str, config: dict) -> dict:
    """Run RelP attribution on an already-loaded HookedTransformer."""
    from circuitkit.backends.eap.graph import Graph, AttentionNode, MLPNode
    from circuitkit.backends.eap.attribute_node import attribute_node
    from circuitkit.tasks.registry import get_task
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    _bootstrap_builtin_tasks()

    device = next(model.parameters()).device
    task_spec = get_task(task_name)

    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True

    discovery_cfg = {
        "algorithm": "relp",
        "task": task_name,
        "level": "node",
        "mlp_hook": "mlp_out",
        "batch_size": config.get("batch_size", 4),
        "model_name": config.get("model_name", getattr(model.cfg, "model_name", "unknown")),
        "data_params": {
            "num_examples": config.get("num_examples", 128),
            "seed": config.get("seed", 42),
            # Calibrate on the "discovery" half — disjoint from the "eval" half
            # the faithfulness/accuracy eval uses, so no calibration leak.
            "data_partition": config.get("discovery_data_partition", "discovery"),
        },
    }

    task_spec.validate_discovery_config(discovery_cfg)
    dataloader = task_spec.build_dataloader(model, discovery_cfg, device)
    metric = task_spec.metric_fn()

    graph = Graph.from_model(model, node_scores=True, neuron_level=False, mlp_hook="mlp_out")

    attribute_node(model, graph, dataloader, metric, method="relp", neuron=False, quiet=False)

    scores = {}
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            fwd_idx = graph.forward_index(node, attn_slice=False)
            score = float(graph.nodes_scores[fwd_idx].item())
            scores[f"A{node.layer}.{node.head}"] = abs(score)
        elif isinstance(node, MLPNode):
            fwd_idx = graph.forward_index(node, attn_slice=False)
            score = float(graph.nodes_scores[fwd_idx].item())
            scores[f"MLP {node.layer}"] = abs(score)

    vals = list(scores.values())
    mx, mn = max(vals), min(vals)
    if mx > mn:
        for k in scores:
            scores[k] = (scores[k] - mn) / (mx - mn)

    return scores
