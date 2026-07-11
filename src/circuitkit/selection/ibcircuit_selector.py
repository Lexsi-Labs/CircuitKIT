"""IBCircuit selector — Information-Bottleneck circuit discovery.

Wraps ``circuitkit.backends.ibcircuit.run_ib_discovery`` (Bian, Niu, Yuan et al.,
"IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck",
ICML 2025; https://github.com/ivanniu/IBCircuit). IBCircuit trains per-component
IB-noise masks on a *single fixed batch* for ``num_epochs`` and averages them
into task-general node scores.

3B/4B-model OOM fix
-------------------
``run_ib_discovery`` holds one fixed batch on the GPU and back-props through the
frozen model on it every epoch. Memory therefore scales with that batch's
``batch_size x seq_len`` — and circuitkit flagged the algorithm ``experimental``
("single-batch training, OOM on 3B").

The OOM had two real causes, both fixed here + in the trainer:

1. Task dataloaders ignore ``batch_size`` for the IBCircuit path and pack the
   *entire* example pool (100-128 rows) into the single fixed batch. On a 3B/4B
   model with long sequences that batch alone OOMs. ``run_ib_discovery`` now
   honours a ``batch_size`` config key that truncates the fixed batch; this
   wrapper passes ``ibcircuit_batch_size`` (default 8) through to it.
2. The trainer's pre-flight ``MemoryError`` check assumed fp32 weights + 4x
   training overhead (grads + optimizer for the *model*). The model is frozen —
   only the tiny IB masks train — so that estimate over-counted by ~4x and
   could spuriously abort a 3B bf16 run. The trainer now estimates from the
   model's real dtype with frozen-model overhead.

The model stays in its loaded dtype (bf16).
"""
import logging

from circuitkit.selection import register

logger = logging.getLogger(__name__)


@register("ibcircuit")
def ibcircuit_selector(model, task_name: str, config: dict) -> dict:
    """Run IBCircuit discovery on an already-loaded HookedTransformer.

    Returns ``{"A{layer}.{head}": float, "MLP {layer}": float}`` — normalised
    node importance scores, the same shape every other selector returns.
    """
    from circuitkit.backends.ibcircuit.trainer import run_ib_discovery
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
    from circuitkit.tasks.registry import get_task

    _bootstrap_builtin_tasks()

    device = next(model.parameters()).device
    task_spec = get_task(task_name)

    # Small fixed batch — the OOM lever (see module docstring). Kept distinct
    # from the EAP discovery `batch_size` so the two paths size independently.
    ib_batch = int(config.get("ibcircuit_batch_size", 8))

    # The example pool the task draws the fixed batch from. >= ib_batch so the
    # dataloader can always fill a full batch; the trainer then truncates to
    # ib_batch. Bounding it here also keeps the (unused) tail off the device.
    pool = max(ib_batch, int(config.get("num_examples", 128)))

    discovery_cfg = {
        "algorithm": "ibcircuit",
        "task": task_name,
        "level": "node",
        "mlp_hook": "mlp_out",
        "batch_size": ib_batch,
        # Top-level num_examples — GenericTaskSpec.build_dataloader reads it
        # here (not under data_params) to cap the example pool it loads.
        "num_examples": pool,
        "model_name": config.get("model_name", getattr(model.cfg, "model_name", "unknown")),
        "data_params": {
            # IOI / other builtins read the pool size from data_params.
            "num_examples": pool,
            "seed": config.get("seed", 42),
            # Calibrate on the "discovery" half — disjoint from the "eval" half
            # the faithfulness/accuracy eval uses, so no calibration leak.
            "data_partition": config.get("discovery_data_partition", "discovery"),
        },
    }
    task_spec.validate_discovery_config(discovery_cfg)
    dataloader = task_spec.build_dataloader(model, discovery_cfg, device)

    # IB training hyper-parameters. num_epochs defaults below the paper's 1000
    # so a grid cell stays tractable; override via config for a faithful run.
    # `batch_size` here is the trainer's fixed-batch cap (the OOM lever) — it
    # truncates whatever the dataloader hands back to ib_batch rows.
    ib_config = {
        "num_epochs": int(config.get("ibcircuit_epochs", 300)),
        "batch_size": ib_batch,
        "level": "node",
        "mlp_hook": "mlp_out",
    }
    # Default scope="both" so heads AND MLPs are scored. The backend default is
    # "heads" (historical); always pass explicitly to avoid silent heads-only runs.
    ib_config["scope"] = config.get("scope", "both")
    for k in ("learning_rate", "alpha", "beta", "alpha_loss", "mask_type"):
        if k in config:
            ib_config[k] = config[k]

    node_scores, _ib_model = run_ib_discovery(model, dataloader, ib_config, str(device))

    # node_scores: {"A0.0": float, "MLP 2": float, ...}. Coerce to plain floats
    # (node level may hand back 0-d tensors) and take magnitude.
    scores = {}
    for k, v in node_scores.items():
        try:
            scores[k] = abs(float(v))
        except (TypeError, ValueError):
            scores[k] = abs(float(v.item()))

    # Min-max normalise to [0, 1] — same convention as eap/cdt selectors.
    vals = list(scores.values())
    if vals:
        mx, mn = max(vals), min(vals)
        if mx > mn:
            for k in scores:
                scores[k] = (scores[k] - mn) / (mx - mn)
    return scores
