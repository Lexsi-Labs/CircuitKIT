"""CD-T selector — Contextual Decomposition for Transformers (gradient-free)."""
import torch
import logging
from circuitkit.selection import register

logger = logging.getLogger(__name__)


@register("cdt")
def cdt_selector(model, task_name: str, config: dict) -> dict:
    """Run CD-T attribution on an already-loaded HookedTransformer."""
    from circuitkit.backends.cdt.adapter import run_cdt_discovery
    from circuitkit.tasks.registry import get_task
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    _bootstrap_builtin_tasks()

    device = next(model.parameters()).device
    task_spec = get_task(task_name)

    # Build a simple dataloader — CD-T only needs clean strings
    discovery_cfg = {
        "algorithm": "cdt",
        "task": task_name,
        "level": "node",
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

    # CD-T expects (clean, corrupted, label) batches; extract clean strings
    # and feed them as individual examples to run_cdt_discovery.
    # Use the same num_examples that built the dataloader (single source of
    # truth) so the iteration cap can't silently diverge from the dataloader
    # size — the two prior reads used different defaults (128 vs 16).
    n_examples = discovery_cfg["data_params"]["num_examples"]
    
    # Use simplified CD-T (full propagation has architecture compatibility
    # issues with some models like GPT-2's LayerNormPre)
    scores = run_cdt_discovery(
        model, dataloader,
        device=device,
        max_seq_len=128,
        n_examples=n_examples,
        use_full_propagation=False,
    )

    # Normalize
    vals = list(scores.values())
    if len(vals) > 0:
        mx, mn = max(vals), min(vals)
        if mx > mn:
            for k in scores:
                scores[k] = (scores[k] - mn) / (mx - mn)

    return scores
