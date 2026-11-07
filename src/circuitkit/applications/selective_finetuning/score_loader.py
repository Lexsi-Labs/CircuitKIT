"""
score_loader.py — Load and normalise circuit discovery scores for selective finetuning.

Discovery artifacts saved by circuitkit's ``discover_circuit`` come in three
distinct formats depending on which algorithm was used and at which level
(node vs neuron).  This module is the single place that knows about all three
formats and normalises them into two consistent dicts consumed by selector.py.

Output interface (always, regardless of input format)
------------------------------------------------------
head_scores : Dict[Tuple[int, int], float]
    One scalar importance score per (layer, head) pair.
    Higher absolute value means more important.
    Scores are raw attribution values and can be negative.

mlp_scores : Dict[int, Union[float, torch.Tensor]]
    One entry per MLP layer.
    - float   → node-level discovery: the whole layer has one score.
                 Selector will allow the entire down_proj to update.
    - Tensor  → neuron-level discovery: 1-D tensor of length d_mlp.
                 Each element is the attribution score for the corresponding
                 column index of down_proj.  Selector will mask to top-X%.

metadata : Dict[str, Any]
    Supplementary information for selector.py and the pipeline:
      'level'           : 'node' or 'neuron'
      'algo'            : e.g. 'eap-ig', 'ibcircuit'
      'mlp_neuron_level': bool — True when mlp_scores values are Tensors
      'mlp_hook'        : 'post_act' or 'mlp_out' (neuron-level EAP and IBCircuit);
                          'unknown' for older IBCircuit files that do not record it
      'n_heads_loaded'  : int
      'n_mlp_loaded'    : int

Format detection
----------------
api.py saves the _scores.pt file with the following keys depending on case:

  Node-level (any algo):
    {'algo', 'level'='node', 'node_scores': Dict[str, float]}

  Neuron-level EAP / EAP-IG:
    {'algo', 'level'='neuron', 'node_scores': Dict[str, float],
     'neurons_scores': Tensor(n_forward, max_d), 'total_neurons': int}
    Requires model_name to reconstruct the graph forward-index mapping.

  Neuron-level IBCircuit:
    {'algo', 'level'='neuron', 'neurons_scores': Dict[str, Tensor],
     'mlp_hook': 'post_act' | 'mlp_out', 'total_neurons': int}
    Does not require model_name.  Older files have no 'mlp_hook' key, and
    some used the singular key 'neuron_scores'; both still load.

  Both neuron-level formats use the key 'neurons_scores'.  The loader tells
  them apart by the type of the value (a dict is IBCircuit, a 2-D tensor is
  EAP / EAP-IG), not by the key.

IBCircuit scores are sigmoid mask values in (0, 1) where a higher value means
the component is kept, so they rank the same way as the EAP scores: the
selector takes the largest absolute scores first.

MLP score space and down_proj column masking
--------------------------------------------
This application masks specific columns of down_proj (input dimension = d_mlp)
to restrict gradient updates to selected neurons.  This is only possible when
MLP scores were computed in d_mlp space, i.e. with mlp_hook='post_act'.

If neuron-level EAP was run with mlp_hook='mlp_out', the MLP scores live in
d_model space and cannot be mapped to down_proj columns.  In that case this
module falls back to treating each MLP layer as node-level (a single scalar),
prints a clear warning, and sets mlp_neuron_level=False in metadata.

Usage
-----
    head_scores, mlp_scores, metadata = load_scores(
        scores_path="outputs/eap-ig_ioi_llama3_scores.pt",
        model_name="meta-llama/Meta-Llama-3-8B",   # required for EAP neuron only
    )
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple, Union

import torch
import logging


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


logger = logging.getLogger(__name__)

def load_scores(
    scores_path: str,
    model_name: Optional[str] = None,
) -> Tuple[
    Dict[Tuple[int, int], float],
    Dict[int, Union[float, torch.Tensor]],
    Dict[str, Any],
]:
    """
    Load and normalise circuit discovery scores from a _scores.pt artifact.

    Parameters
    ----------
    scores_path : Path to the ``*_scores.pt`` file saved by ``discover_circuit``.
                  This is the file whose name ends in ``_scores.pt``, not the
                  main artifact (which ends in just ``.pt``).
    model_name  : HuggingFace model identifier (e.g. "meta-llama/Meta-Llama-3-8B").
                  Required only for neuron-level EAP / EAP-IG scores, where the
                  graph forward-index structure must be reconstructed from the
                  model's architecture config.  For neuron-level IBCircuit it
                  is only used to infer the MLP hook of older files that do not
                  record one.  Ignored for node-level files.

    Returns
    -------
    head_scores : Dict[(layer, head), float]
    mlp_scores  : Dict[layer, float | Tensor]
    metadata    : Dict[str, Any]

    Raises
    ------
    FileNotFoundError : If scores_path does not exist.
    ValueError        : If the file format is unrecognised, or if model_name
                        is required but was not provided.
    """
    if not os.path.exists(scores_path):
        raise FileNotFoundError(
            f"Scores file not found: {scores_path}\n"
            f"Run discover_circuit first and point --scores-path at the "
            f"file ending in '_scores.pt'."
        )

    logger.info(f"[score_loader] Loading scores from: {scores_path}")
    # weights_only=True: a scores file may be shared/untrusted input. Its payload
    # is plain data (algo/level strings, score dicts of tensors, a metadata dict),
    # so the safe unpickler handles it while blocking pickle-based RCE (CWE-502).
    data = torch.load(scores_path, map_location="cpu", weights_only=True)

    algo = data.get("algo", "unknown")
    level = data.get("level", "node")

    logger.info(f"[score_loader] Detected: algo={algo!r}  level={level!r}")

    if level == "node":
        head_scores, mlp_scores, metadata = _load_node_level(data)

    elif level == "neuron":
        # Two neuron-level sub-formats, told apart by the type of the value.
        # IBCircuit writes 'neurons_scores' (older files: 'neuron_scores').
        neuron_scores = data.get("neurons_scores")
        if neuron_scores is None:
            neuron_scores = data.get("neuron_scores")
        if isinstance(neuron_scores, dict):
            # IBCircuit neuron: dict of {name: Tensor}
            head_scores, mlp_scores, metadata = _load_ibcircuit_neuron(
                neuron_scores, data, model_name
            )
        elif isinstance(neuron_scores, torch.Tensor) and neuron_scores.dim() == 2:
            # EAP / EAP-IG neuron: raw 2-D tensor + graph index reconstruction
            if model_name is None:
                raise ValueError(
                    "model_name is required to decode neuron-level EAP scores.\n"
                    "The graph forward-index structure must be reconstructed from "
                    "the model architecture.\n"
                    "Pass model_name=<HuggingFace model ID> to load_scores()."
                )
            head_scores, mlp_scores, metadata = _load_eap_neuron(data, model_name)
        else:
            raise ValueError(
                f"Neuron-level scores file has no usable neuron scores.\n"
                f"Expected 'neurons_scores' to be a dict of tensors (IBCircuit) or a "
                f"2-D tensor (EAP / EAP-IG); the older IBCircuit key 'neuron_scores' "
                f"is also read.\n"
                f"Found keys: {list(data.keys())}; the scores have type "
                f"{type(neuron_scores).__name__}."
            )
    else:
        raise ValueError(
            f"Unrecognised level {level!r} in scores file.\n"
            f"Expected 'node' or 'neuron'.  File keys: {list(data.keys())}"
        )

    metadata["algo"] = algo
    metadata["level"] = level

    _warn_nonfinite(head_scores, mlp_scores)
    _print_score_summary(head_scores, mlp_scores, metadata)
    return head_scores, mlp_scores, metadata


# ---------------------------------------------------------------------------
# Format handlers
# ---------------------------------------------------------------------------


def _load_node_level(
    data: Dict,
) -> Tuple[Dict, Dict, Dict]:
    """
    Handle node-level scores produced by any algorithm.

    Expected key: 'node_scores' — Dict[str, float] with keys like
    'A0.0', 'A0.1', …, 'MLP 0', 'MLP 1', …
    """
    raw: Dict[str, float] = data.get("node_scores", {})
    if not raw:
        raise ValueError("Node-level scores file has an empty 'node_scores' dict.")

    head_scores: Dict[Tuple[int, int], float] = {}
    mlp_scores: Dict[int, Union[float, torch.Tensor]] = {}

    for name, score in raw.items():
        layer, head = _parse_attn_key(name)
        if layer is not None:
            head_scores[(layer, head)] = float(score)
            continue

        layer = _parse_mlp_key(name)
        if layer is not None:
            # Wrap scalar as float — selector.py uses isinstance checks
            mlp_scores[layer] = float(score)
            continue

        # Silently skip unrecognised keys (e.g. 'logits', 'input')
        # which are present in node_scores from some algorithms.

    metadata = {
        "mlp_neuron_level": False,
        "mlp_hook": "mlp_out",  # node-level always uses mlp_out scoring
        "n_heads_loaded": len(head_scores),
        "n_mlp_loaded": len(mlp_scores),
    }
    return head_scores, mlp_scores, metadata


def _load_ibcircuit_neuron(
    raw: Dict[str, torch.Tensor],
    data: Dict,
    model_name: Optional[str],
) -> Tuple[Dict, Dict, Dict]:
    """
    Handle neuron-level IBCircuit scores.

    ``raw`` is the dict stored under 'neurons_scores' (older files:
    'neuron_scores'), ``data`` the whole file:
      Attention entries  "A{L}.{H}" → Tensor of shape [d_head]
      MLP entries        "MLP {L}"  → Tensor of shape [d_mlp] (mlp_hook='post_act')
                                      or [d_model] (mlp_hook='mlp_out', the default)

    Attention tensors are aggregated to a scalar per head (absolute sum).
    MLP tensors stay neuron-level only when they are in d_mlp space, because
    only then do they index the columns of down_proj.  The space is read from
    the file's 'mlp_hook' key; for files without one it is inferred from the
    tensor length when model_name is given.  Otherwise (mlp_hook='mlp_out', or
    it cannot be told) each MLP layer falls back to one node-level scalar.
    """
    if not raw:
        raise ValueError("IBCircuit neuron scores file has an empty neuron scores dict.")

    head_scores: Dict[Tuple[int, int], float] = {}
    mlp_scores: Dict[int, Union[float, torch.Tensor]] = {}

    for name, tensor in raw.items():
        if not isinstance(tensor, torch.Tensor):
            # Defensive: convert scalars that slipped through
            tensor = torch.tensor(float(tensor))

        tensor = tensor.float().cpu()

        layer, head = _parse_attn_key(name)
        if layer is not None:
            # Aggregate d_head-length tensor to single scalar per head.
            # Absolute sum preserves the magnitude of suppressive contributions.
            head_scores[(layer, head)] = float(tensor.abs().sum().item())
            continue

        layer = _parse_mlp_key(name)
        if layer is not None:
            mlp_scores[layer] = tensor
            continue

    # Decide whether MLP scores are genuinely neuron-level.
    # If all MLP tensors have length 1 they came from a node-level run
    # that was mislabelled — treat as node-level scalars.
    mlp_neuron_level = any(
        isinstance(v, torch.Tensor) and v.numel() > 1 for v in mlp_scores.values()
    )
    mlp_hook = data.get("mlp_hook", "unknown")
    if mlp_neuron_level:
        # Which space are the MLP tensors in?  The file says so; older files
        # are inferred from the tensor length when model_name is available.
        if mlp_hook not in ("post_act", "mlp_out") and model_name is not None:
            try:
                from transformers import AutoConfig

                hf_cfg = AutoConfig.from_pretrained(model_name)
                d_model_hf = hf_cfg.hidden_size
                d_mlp_hf = getattr(hf_cfg, "intermediate_size", 4 * d_model_hf)
                sample_len = next(
                    v.numel() for v in mlp_scores.values() if isinstance(v, torch.Tensor)
                )
                if sample_len == d_mlp_hf:
                    mlp_hook = "post_act"
                elif sample_len == d_model_hf:
                    mlp_hook = "mlp_out"
            except Exception as exc:
                logger.info(f"[score_loader] Could not infer mlp_hook for IBCircuit neuron: {exc}")

        if mlp_hook != "post_act":
            # d_model-space (or unknown-space) scores cannot be mapped to down_proj
            # columns, and masking with them would pick the wrong neurons.  Use one
            # scalar per MLP layer, which is always safe.
            if mlp_hook == "mlp_out":
                logger.info(
                    "[score_loader] WARNING: IBCircuit MLP tensors are d_model-length "
                    "(mlp_hook='mlp_out') → cannot use for down_proj column masking. "
                    "Falling back to node-level MLP treatment."
                )
            else:
                mlp_hook = "unknown"
                logger.info(
                    "[score_loader] WARNING: Cannot tell which space the IBCircuit MLP "
                    "tensors are in (no 'mlp_hook' in the file and no model_name given). "
                    "Falling back to node-level MLP treatment (no column masking)."
                )
            mlp_neuron_level = False
            mlp_scores = {
                k: float(v.abs().sum().item()) if isinstance(v, torch.Tensor) else float(v)
                for k, v in mlp_scores.items()
            }
    else:
        # Convert to floats for node-level treatment
        mlp_scores = {
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in mlp_scores.items()
        }

    metadata = {
        "mlp_neuron_level": mlp_neuron_level,
        "mlp_hook": mlp_hook,
        "n_heads_loaded": len(head_scores),
        "n_mlp_loaded": len(mlp_scores),
    }
    return head_scores, mlp_scores, metadata


def _load_eap_neuron(
    data: Dict,
    model_name: str,
) -> Tuple[Dict, Dict, Dict]:
    """
    Handle neuron-level EAP / EAP-IG scores.

    Expected key: 'neurons_scores' — Tensor of shape (n_forward, max_d).

    Each row corresponds to a source node in the computation graph.  The
    mapping from row index to node identity is determined by Graph's forward
    index formulas, which depend on the model architecture (n_layers, n_heads).
    We reconstruct a skeleton Graph from the HuggingFace model config — no
    weights are loaded, only the config JSON is fetched.

    MLP hook detection
    ------------------
    max_d = max(d_model, d_mlp)  when mlp_hook='post_act'  (wider tensor)
    max_d = d_model               when mlp_hook='mlp_out'   (narrower tensor)

    If the tensor is wider than d_model the hook was 'post_act' and MLP scores
    are in d_mlp space — suitable for down_proj column masking.
    If not, MLP scores are in d_model space and column masking is not possible;
    we fall back to treating each MLP layer as node-level.
    """
    from transformers import AutoConfig

    from circuitkit.backends.eap.graph import AttentionNode, Graph, MLPNode

    neurons_scores: torch.Tensor = data["neurons_scores"].float().cpu()

    # ── Read model architecture from HuggingFace config ───────────────────
    logger.info(f"[score_loader] Reading model config for {model_name!r} …")
    hf_cfg = AutoConfig.from_pretrained(model_name)

    n_layers = hf_cfg.num_hidden_layers
    n_heads = hf_cfg.num_attention_heads
    d_model = hf_cfg.hidden_size
    d_mlp = getattr(hf_cfg, "intermediate_size", 4 * d_model)

    # parallel_attn_mlp is a TransformerLens concept not present in standard
    # HF configs.  All LLaMA and Qwen models use sequential attention+MLP.
    parallel_attn_mlp = getattr(hf_cfg, "parallel_attn_mlp", False)

    # ── Detect which mlp_hook was used during discovery ───────────────────
    max_d = neurons_scores.shape[1]

    if d_mlp > d_model and max_d >= d_mlp:
        mlp_hook = "post_act"
        mlp_neuron_level = True
        logger.info(
            f"[score_loader] MLP hook detected: post_act " f"(tensor width {max_d} ≥ d_mlp={d_mlp})"
        )
    elif max_d == d_model:
        mlp_hook = "mlp_out"
        mlp_neuron_level = False
        logger.info(
            f"[score_loader] WARNING: MLP hook detected as mlp_out "
            f"(tensor width {max_d} == d_model={d_model}).\n"
            f"  MLP scores are in d_model space and cannot be used for "
            f"down_proj column masking.\n"
            f"  Falling back to node-level treatment for MLP layers.\n"
            f"  To enable neuron-level MLP finetuning, re-run discovery "
            f"with mlp_hook='post_act'."
        )
    else:
        # Ambiguous: d_mlp <= d_model (unusual architecture) or an unexpected
        # tensor width.  In either case we cannot safely determine the hook,
        # so fall back to node-level MLP treatment which is always safe.
        mlp_hook = "mlp_out"
        mlp_neuron_level = False
        logger.info(
            f"[score_loader] WARNING: Cannot determine mlp_hook from tensor "
            f"width={max_d} with d_model={d_model}, d_mlp={d_mlp}.\n"
            f"  Falling back to node-level MLP treatment (no column masking).\n"
            f"  If this is unexpected, verify that discovery was run with "
            f"mlp_hook='post_act' and that model_name is correct."
        )

    # ── Reconstruct skeleton Graph to get forward index mapping ───────────
    graph_cfg = {
        "n_layers": n_layers,
        "n_heads": n_heads,
        "parallel_attn_mlp": parallel_attn_mlp,
        "d_model": d_model,
        "d_mlp": d_mlp,
    }
    # Neuron-level graph needed so d_neuron properties return correct values.
    graph = Graph.from_model(
        graph_cfg,
        neuron_level=True,
        node_scores=True,
        mlp_hook=mlp_hook,
    )

    expected_n_forward = 1 + n_layers * (n_heads + 1)
    if neurons_scores.shape[0] != expected_n_forward:
        raise ValueError(
            f"neurons_scores has {neurons_scores.shape[0]} rows but the "
            f"reconstructed graph expects n_forward={expected_n_forward}.\n"
            f"Model config gives n_layers={n_layers}, n_heads={n_heads}.\n"
            f"Check that model_name matches the model used for discovery."
        )

    # ── Extract scores per node ────────────────────────────────────────────
    head_scores: Dict[Tuple[int, int], float] = {}
    mlp_scores: Dict[int, Union[float, torch.Tensor]] = {}

    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            fwd_idx = graph.forward_index(node, attn_slice=False)
            d_neuron = node.d_neuron  # always d_model for attention
            raw = neurons_scores[fwd_idx, :d_neuron]
            # Aggregate d_model-length vector to a single importance scalar.
            head_scores[(node.layer, node.head)] = float(raw.abs().sum().item())

        elif isinstance(node, MLPNode):
            fwd_idx = graph.forward_index(node, attn_slice=False)
            d_neuron = node.d_neuron  # d_mlp (post_act) or d_model (mlp_out)
            raw = neurons_scores[fwd_idx, :d_neuron].clone()

            if mlp_neuron_level:
                mlp_scores[node.layer] = raw  # Tensor[d_mlp]
            else:
                # Collapse to scalar for node-level fallback
                mlp_scores[node.layer] = float(raw.abs().sum().item())

    metadata = {
        "mlp_neuron_level": mlp_neuron_level,
        "mlp_hook": mlp_hook,
        "n_heads_loaded": len(head_scores),
        "n_mlp_loaded": len(mlp_scores),
    }
    return head_scores, mlp_scores, metadata


# ---------------------------------------------------------------------------
# Key parsers
# ---------------------------------------------------------------------------


def _parse_attn_key(name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse an attention node name into (layer, head).

    Accepts both circuitkit formats:
      'A{layer}.{head}'   — used in node_scores and neuron_scores dicts
      'a{layer}.h{head}'  — used internally in Graph.nodes (not in score files)

    Returns (None, None) if the name does not match.
    """
    # Capital A format: "A0.3", "A11.0"
    m = re.fullmatch(r"A(\d+)\.(\d+)", name.strip())
    if m:
        return int(m.group(1)), int(m.group(2))

    # Lowercase format: "a0.h3"  (defensive, should not appear in score files)
    m = re.fullmatch(r"a(\d+)\.h(\d+)", name.strip())
    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def _parse_mlp_key(name: str) -> Optional[int]:
    """
    Parse an MLP node name into a layer index.

    Accepts both circuitkit formats:
      'MLP {layer}'  — used in node_scores and neuron_scores dicts
      'm{layer}'     — used internally in Graph.nodes (not in score files)

    Returns None if the name does not match.
    """
    # "MLP 0", "MLP 11"
    m = re.fullmatch(r"MLP (\d+)", name.strip())
    if m:
        return int(m.group(1))

    # "m0", "m11"  (defensive)
    m = re.fullmatch(r"m(\d+)", name.strip())
    if m:
        return int(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_score_summary(
    head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, Union[float, torch.Tensor]],
    metadata: Dict[str, Any],
) -> None:
    """Print a concise summary of the loaded scores."""
    width = 60
    logger.info(f"\n{'='*width}")
    logger.info("  SCORE LOADER SUMMARY")
    logger.info(f"{'='*width}")
    logger.info(f"  Algorithm : {metadata.get('algo', 'unknown')}")
    logger.info(f"  Level     : {metadata.get('level', 'unknown')}")

    # Attention summary
    n_attn = len(head_scores)
    if n_attn:
        scores_abs = [abs(v) for v in head_scores.values()]
        logger.info(
            f"\n  Attention heads loaded : {n_attn}\n"
            f"    Score range : [{min(scores_abs):.4f}, {max(scores_abs):.4f}]\n"
            f"    Mean        : {sum(scores_abs)/n_attn:.4f}"
        )
        # Show unique layers touched
        layers = sorted({lyr for lyr, _ in head_scores})
        logger.info(f"    Layers      : {layers}")
    else:
        logger.info("\n  Attention heads loaded : 0 (scope may exclude heads)")

    # MLP summary
    n_mlp = len(mlp_scores)
    if n_mlp:
        mlp_neuron = metadata.get("mlp_neuron_level", False)
        if mlp_neuron:
            # Tensor values — report per-neuron stats
            all_vals = torch.cat(
                [v.abs() for v in mlp_scores.values() if isinstance(v, torch.Tensor)]
            )
            logger.info(
                f"\n  MLP layers loaded : {n_mlp}  (neuron-level)\n"
                f"    Neurons per layer : {next(v.numel() for v in mlp_scores.values() if isinstance(v, torch.Tensor))}\n"
                f"    Score range : [{all_vals.min():.4f}, {all_vals.max():.4f}]\n"
                f"    Mean        : {all_vals.mean():.4f}\n"
                f"    MLP hook    : {metadata.get('mlp_hook', 'unknown')}"
            )
        else:
            scores_abs = [
                abs(v) if isinstance(v, float) else abs(v.item()) for v in mlp_scores.values()
            ]
            logger.info(
                f"\n  MLP layers loaded : {n_mlp}  (node-level)\n"
                f"    Score range : [{min(scores_abs):.4f}, {max(scores_abs):.4f}]\n"
                f"    Mean        : {sum(scores_abs)/n_mlp:.4f}"
            )
    else:
        logger.info("\n  MLP layers loaded : 0 (scope may exclude MLP)")

    logger.info(f"\n{'='*width}\n")


def _warn_nonfinite(
    head_scores: Dict[Tuple[int, int], float],
    mlp_scores: Dict[int, Union[float, torch.Tensor]],
) -> None:
    """Warn (non-fatal) if any scores contain inf or nan values."""
    bad_heads = [(k, v) for k, v in head_scores.items() if v != v or abs(v) == float("inf")]
    if bad_heads:
        logger.info(
            f"[score_loader] WARNING: {len(bad_heads)} attention head scores "
            f"are inf/nan (e.g. {bad_heads[:3]}). These were likely pinned by "
            f"scope exclusion and will be handled by the selector."
        )

    bad_mlp = []
    for layer, val in mlp_scores.items():
        if isinstance(val, torch.Tensor):
            if not torch.isfinite(val).all():
                bad_mlp.append(layer)
        elif val != val or abs(val) == float("inf"):
            bad_mlp.append(layer)
    if bad_mlp:
        logger.info(
            f"[score_loader] WARNING: {len(bad_mlp)} MLP layer scores "
            f"are inf/nan (layers {bad_mlp[:5]}). Likely scope-pinned."
        )
