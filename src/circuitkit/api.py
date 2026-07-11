import logging
import os
import re
import warnings
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch as t

if TYPE_CHECKING:
    from .evaluation.report import FaithfulnessReport

# Suppress verbose warnings BEFORE importing heavy libraries
# This must be done early to catch warnings during imports
warnings.filterwarnings("ignore", message=".*reduced precision.*")
warnings.filterwarnings("ignore", message=".*from_pretrained_no_processing.*")
warnings.filterwarnings("ignore", message=".*pretrained.*model kwarg is not of type.*")
warnings.filterwarnings("ignore", message=".*Passed an already-initialized model.*")
warnings.filterwarnings("ignore", message=".*Overwriting default num_fewshot.*")
warnings.filterwarnings("ignore", message=".*S2 index has been computed.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress verbose loggers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("lm_eval").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

from transformer_lens import (  # noqa: E402 - import after intentional pre-import setup
    HookedTransformer,
)

# CircuitScores artifact (Workstream G)
from .artifacts.scores import (  # noqa: E402 - import after intentional pre-import setup
    CircuitScores,
)
from .utils.debug import (  # noqa: E402 - import after intentional pre-import setup
    debug_context,
    debug_function,
)
from .utils.exceptions import (  # noqa: E402 - import after intentional pre-import setup
    AlgorithmError,
    handle_errors,
    validate_discovery_algorithm,
    validate_file_exists,
    validate_model_name,
)

# CircuitKit imports
from circuitkit.utils.device import get_device, empty_cache
from .utils.logging import (  # noqa: E402 - import after intentional pre-import setup
    ProgressLogger,
    get_logger,
    log_execution_time,
)

logger = get_logger(__name__)


def _fmt_opt_score(x, spec=".4f"):
    """Format an optional score for logging. Pillar scores (patching/ablation)
    are None when the underlying metric is invalid (e.g. inverted denominator);
    formatting None with ``:.4f`` raises ``NoneType.__format__``."""
    return format(x, spec) if x is not None else "invalid"


# Task management imports
# NOTE: imported lazily inside functions to avoid a circular import between
# `circuitkit.api` and `circuitkit.tasks.registry` (task builtins pull in
# helpers re-exported from this module), which breaks when `circuitkit.api`
# is the first module imported.
def _get_task(*args, **kwargs):
    from .tasks.registry import get_task as _gt

    return _gt(*args, **kwargs)


def _register_task(*args, **kwargs):
    from .tasks.registry import register_task as _rt

    return _rt(*args, **kwargs)


import warnings as _warnings  # noqa: E402 - import after intentional pre-import setup

from .backends import (  # noqa: E402 - import after intentional pre-import setup
    DEFAULT_ALGORITHM as _DEFAULT_ALGO,
)
from .backends import (  # noqa: E402 - import after intentional pre-import setup
    EXPERIMENTAL_ALGORITHMS,
    RESEARCH_ALGORITHMS,
)

# ACDC Backend Imports
from .backends.acdc.data import (  # noqa: E402 - import after intentional pre-import setup
    load_task_data,
)
from .backends.acdc.prune_algos.ACDC import (  # noqa: E402 - import after intentional pre-import setup
    acdc_prune_scores,
)
from .backends.acdc.utils.graph_utils import (  # noqa: E402 - import after intentional pre-import setup
    patchable_model,
)
from .backends.eap.attribute_node import (  # noqa: E402 - import after intentional pre-import setup
    attribute_node,
)
# EAP Backend Imports
from .backends.eap.graph import (  # noqa: E402 - import after intentional pre-import setup
    AttentionNode,
    Graph,
    MLPNode,
)


def _log_gpu_mem(label: str, logger):
    """Log GPU memory stats at DEBUG level. No-op if CUDA unavailable."""
    import torch

    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    free_reserved = reserved - allocated
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_total = total - reserved
    logger.debug(
        f"[GPU-MEM] {label}: "
        f"alloc={allocated:.2f}GB, reserved={reserved:.2f}GB, "
        f"free_in_reserved={free_reserved:.2f}GB, free_total={free_total:.2f}GB"
    )


# EAPDiscoveryDataset is a torch Dataset — it lives in circuitkit.data, not in
# this front-door facade. Re-exported here for backward compatibility; new code
# should import it from circuitkit.data.eap_dataset.
from .data.eap_dataset import EAPDiscoveryDataset  # noqa: F401,E402


from collections import defaultdict  # noqa: E402 - import after intentional pre-import setup

from tqdm import tqdm  # noqa: E402 - import after intentional pre-import setup

# CircuitKit Core Imports
from .analysis.scores import (  # noqa: E402 - import after intentional pre-import setup
    calculate_node_scores_from_edges,
)
from .applications.pruning.node_pruner import (  # noqa: E402 - import after intentional pre-import setup
    get_nodes_to_prune,
)
from .utils.config import (  # noqa: E402 - import after intentional pre-import setup
    DEFAULT_CONFIG,
    load_and_validate_config,
)


def _correct_token_prob(logits, clean_logits, input_lengths, labels, loss=False, mean=False):
    """Correct-answer token probability at the answer position.

    A bounded [0, 1] metric suitable for clean-only evaluation (e.g.
    IBCircuit neuron-level on clean-only custom data) where no incorrect token
    is available and logit_diff would collapse to zero.

    ``labels`` shape: [batch, 2] where ``labels[:, 0]`` = correct token ID.
    The second column is ignored (may be a duplicate or a dummy).
    """
    batch = logits.size(0)
    idx = t.arange(batch, device=logits.device)
    last = (input_lengths.long() - 1).clamp_min(0)
    probs = t.softmax(logits[idx, last], dim=-1)
    correct_ids = labels[:, 0].to(logits.device)
    result = probs[idx, correct_ids]
    if loss:
        result = -result
    if mean:
        result = result.mean()
    return result


def _build_clean_only_ib_eval_dataloader(task_spec, model, num_examples: int, batch_size: int):
    """Build an EAP-format eval DataLoader for clean-only NormalizedTaskSpec.

    Duplicates the clean prompt as the corrupt side so tokenize_batch_pair
    in evaluate_baseline / evaluate_ibcircuit_neuron_circuit can run without
    real paired data. Mean- and zero-ablation paths never consume corrupt
    activations, so the duplicate is harmless.

    Labels are shaped [batch, 2] with both columns = correct-answer token ID,
    suitable for _correct_token_prob (which only reads column 0).

    Returns a torch DataLoader yielding (clean_list, corrupt_list, label_tensor)
    batches — the same EAP-format the evaluators expect.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tokenizer = model.tokenizer
    try:
        ws_probe = tokenizer.encode(" ", add_special_tokens=False)
        ws_token_id = ws_probe[0] if len(ws_probe) == 1 else None
    except Exception:
        ws_token_id = None

    clean_texts = []
    label_ids = []

    for r in task_spec.ds.records[:num_examples]:
        # Derive correct-answer token ID using joint encoding (same logic as
        # NormalizedTaskSpec._build_ibcircuit_dataloader).
        precomputed = r.meta.get("_precomputed_labels")
        if precomputed:
            ans_token = precomputed["clean_label_id"]
        else:
            prompt_ids_solo = tokenizer.encode(r.clean_prompt, add_special_tokens=False)
            full_ids = tokenizer.encode(r.clean_prompt + r.clean_answer, add_special_tokens=False)
            boundary_clean = (
                len(full_ids) > len(prompt_ids_solo)
                and full_ids[: len(prompt_ids_solo)] == prompt_ids_solo
            )
            if boundary_clean:
                first_cont = int(full_ids[len(prompt_ids_solo)])
                if (
                    ws_token_id is not None
                    and first_cont == ws_token_id
                    and len(full_ids) > len(prompt_ids_solo) + 1
                ):
                    ans_token = int(full_ids[len(prompt_ids_solo) + 1])
                else:
                    ans_token = first_cont
            else:
                ans_ids = tokenizer.encode(r.clean_answer, add_special_tokens=False)
                if not ans_ids:
                    continue
                ans_token = ans_ids[0]
                if ws_token_id is not None and ans_token == ws_token_id and len(ans_ids) > 1:
                    ans_token = ans_ids[1]

        clean_texts.append(r.clean_prompt)
        label_ids.append(ans_token)

    if not clean_texts:
        raise RuntimeError(
            "clean-only eval dataloader: no records could be built from the task spec."
        )

    label_tensor = torch.tensor([[lid, lid] for lid in label_ids], dtype=torch.long)

    # Batch into (clean_list, corrupt_list, label_chunk) tuples.
    batches = []
    for start in range(0, len(clean_texts), batch_size):
        end = start + batch_size
        chunk_clean = clean_texts[start:end]
        chunk_labels = label_tensor[start:end]
        batches.append((chunk_clean, chunk_clean, chunk_labels))

    class _TextBatchLoader:
        def __init__(self, batches, padding_side="left"):
            self._batches = batches
            self.pair_padding_side = padding_side

        def __iter__(self):
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    side = getattr(task_spec, "pair_padding_side", "left")
    return _TextBatchLoader(batches, padding_side=side)


# Backwards-compatibility re-export: legacy code (e.g. tasks/builtins/ioi_acdc.py)
# imports `_eap_logit_diff` from this module. The canonical implementation now
# lives on each TaskSpec. We forward to IOITaskSpec._ioi_logit_diff because the
# legacy importer was IOI-specific.
def _eap_logit_diff(*args, **kwargs):
    """Legacy IOI-style logit-difference metric.

    Deprecated: use ``IOITaskSpec._ioi_logit_diff`` (or the equivalent on
    your TaskSpec) instead.
    """
    from .tasks.builtins.ioi import IOITaskSpec

    return IOITaskSpec._ioi_logit_diff(*args, **kwargs)


def _eap_kl_divergence(logits, clean_logits, input_length, labels, mean=True):
    """Multi-token KL-divergence metric.

    For tasks where the answer is multi-token (long names, full
    sentences, free-form generation), the single-token logit-diff
    metric truncates to the first BPE subword and loses semantics.
    KL-divergence between the model's full distribution at the answer
    position(s) and the reference distribution captures the whole
    answer profile.

    Used as a substitute for ``_eap_logit_diff`` on tasks where
    ``clean_answer`` and ``corrupt_answer`` differ in tokens beyond
    the first subword.

    Args:
        logits (Tensor): Model logits [batch, seq_len, vocab_size].
        clean_logits (Tensor): Reference logits at the same shape.
            KL is computed as KL(softmax(logits) || softmax(clean_logits))
            at the last real token position.
        input_length (Tensor): Number of real tokens per example [batch].
        labels (Tensor): Unused for KL; kept for signature compatibility.
        mean (bool): If True, return scalar mean KL. If False, per-example.

    Returns:
        Tensor: Scalar mean KL or per-sample [batch].
    """
    if clean_logits is None:
        # KL needs the reference distribution; degrade gracefully to logit-diff.
        from .tasks.builtins.ioi import IOITaskSpec

        return IOITaskSpec._ioi_logit_diff(logits, clean_logits, input_length, labels, mean=mean)

    batch = logits.size(0)
    last = (input_length.long() - 1).clamp_min(0)
    arange = t.arange(batch, device=logits.device)
    last_logits = logits[arange, last]  # [batch, vocab]
    last_clean = clean_logits[arange, last]  # [batch, vocab]
    log_p = t.nn.functional.log_softmax(last_logits, dim=-1)
    log_q = t.nn.functional.log_softmax(last_clean, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)  # [batch]
    return kl.mean() if mean else kl


def _eap_accuracy(logits, clean_logits, input_length, labels, mean=True):
    """
    Token-prediction accuracy metric for EAP attribution.

    Selects the logit at each example's last real token position and checks
    whether the argmax matches the correct token. Handles both single-token
    labels (IOI) and multi-column label tensors (MMLU); in the latter case
    column 0 is treated as the correct answer.

    Args:
        logits (Tensor): Model logits [batch, seq_len, vocab_size].
        clean_logits (Tensor): Unused; kept for metric signature compatibility.
        input_length (Tensor): Number of real tokens per example [batch].
        labels (Tensor): Correct token indices. Shape [batch] or [batch, n_options];
            if 2-D, labels[:, 0] is used as the correct token.
        mean (bool): If True, return the batch mean accuracy scalar.
            If False, return per-sample accuracy [batch]. Defaults to True.

    Returns:
        Tensor: Scalar mean accuracy (mean=True) or per-sample float tensor [batch].
    """
    # Added Debugging here because this is a frequent crash point
    # debug_metric_shapes(logits, labels, input_length)

    batch_size = logits.size(0)
    idx = t.arange(batch_size, device=logits.device)
    logits = logits[idx, input_length - 1]

    if labels.ndim > 1:
        correct_token = labels[:, 0]
    else:
        correct_token = labels

    correct_token = correct_token.to(logits.device)
    predictions = logits.argmax(dim=-1)

    results = (predictions == correct_token).float()

    if mean:
        results = results.mean()
    return results


# def _convert_eap_scores_to_ck_format(graph: Graph) -> dict[str, float]:
#     """
#     Convert EAP node scores to CircuitKit's name-keyed score dict.

#     Maps graph node names to CircuitKit naming convention:
#     AttentionNode 'a{L}.h{H}' → 'A{L}.{H}', MLPNode 'm{L}' → 'MLP {L}'.
#     Scores are absolute values of the raw node scores.

#     Args:
#         graph (Graph): Graph with populated node scores after attribution.

#     Returns:
#         Dict[str, float]: {'A0.0': score, 'MLP 0': score, ...} for all
#             AttentionNode and MLPNode instances in the graph.
#     """
#     node_scores_dict = {}
#     for node in graph.nodes.values():
#         if isinstance(node, (AttentionNode, MLPNode)):
#             score = abs(node.score.item())
#             if isinstance(node, AttentionNode):
#                 circuit_kit_name = f"A{node.layer}.{node.head}"
#             else: # MLPNode
#                 circuit_kit_name = f"MLP {node.layer}"
#             node_scores_dict[circuit_kit_name] = score
#     return node_scores_dict

from .backends.eap.circuit_kit_adapter import (  # noqa: E402 - import after intentional pre-import setup
    convert_eap_graph_to_circuitkit_scores as _convert_eap_scores_to_ck_format,
)


def _ib_name_to_graph_name(ib_name: str) -> Optional[str]:
    """Convert IBCircuit score key ('A0.0', 'MLP 0') to EAP Graph node key ('a0.h0', 'm0')."""
    attn_match = re.match(r"A(\d+)\.(\d+)$", ib_name)
    if attn_match:
        return f"a{attn_match.group(1)}.h{attn_match.group(2)}"
    mlp_match = re.match(r"MLP (\d+)$", ib_name)
    if mlp_match:
        return f"m{mlp_match.group(1)}"
    return None


def _populate_graph_from_ib_scores(graph: Graph, ib_node_scores: dict) -> Graph:
    """
    Write IBCircuit node scores into a Graph's nodes_scores tensor.

    Converts IBCircuit naming ('A0.0', 'MLP 0') to EAP graph naming ('a0.h0',
    'm0') and records absolute scores. Any node absent from ib_node_scores
    (i.e. out-of-scope) is pinned to inf so that graph.apply_topn() always
    retains it — this is the mechanism that enforces scope constraints.

    Args:
        graph (Graph): Graph initialised with node_scores=True. Its
            nodes_scores tensor is overwritten in-place.
        ib_node_scores (Dict[str, float]): Scores keyed by IBCircuit node
            names, e.g. {'A0.0': 0.42, 'MLP 3': 0.07}.

    Returns:
        Graph: The same graph object, mutated in-place.
    """
    graph.nodes_scores = t.full((graph.n_forward,), float("nan"))
    for ib_name, score in ib_node_scores.items():
        graph_name = _ib_name_to_graph_name(ib_name)
        if graph_name and graph_name in graph.nodes:
            node = graph.nodes[graph_name]
            node.score = t.tensor(abs(float(score)))
            fwd_idx = graph.forward_index(node, attn_slice=False)
            graph.nodes_scores[fwd_idx] = abs(float(score))

    # Any node not scored by IB (nan) is pinned to inf so apply_topn
    # always keeps it. For scope='heads', this catches all MLPs.
    # For scope='mlp', this catches all attention heads.
    # For scope='both', all nodes are scored and nothing is pinned.
    for node in graph.nodes.values():
        if isinstance(node, (AttentionNode, MLPNode)):
            fwd_idx = graph.forward_index(node, attn_slice=False)
            if t.isnan(graph.nodes_scores[fwd_idx]).any():
                node.score = t.tensor(float("inf"))
                graph.nodes_scores[fwd_idx] = float("inf")

    return graph


def _validate_ibcircuit_dataloader(dataloader) -> None:
    """
    Validate that dataloader provides IBCircuit-compatible batches.

    IBCircuit requires batches with specific keys:
    - 'tokens': Input token IDs [batch_size, seq_len]
    - 'labels': Answer token IDs [batch_size]
    - 'answer_positions': Positions where answers appear [batch_size]

    Args:
        dataloader: DataLoader to validate

    Raises:
        ValueError: If dataloader format is incompatible
        StopIteration: If dataloader is empty
    """
    try:
        # Extract one batch for validation
        batch = next(iter(dataloader))
    except StopIteration:
        raise ValueError(
            "IBCircuit dataloader is empty. Ensure your task's "
            "build_dataloader() method returns a non-empty DataLoader."
        )

    # Check required keys
    required_keys = {"tokens", "labels", "answer_positions"}
    actual_keys = set(batch.keys())
    missing_keys = required_keys - actual_keys

    if missing_keys:
        raise ValueError(
            f"IBCircuit dataloader missing required keys: {missing_keys}.\n"
            f"Got keys: {list(actual_keys)}\n"
            f"Required keys: {list(required_keys)}\n\n"
            f"Your task's build_dataloader() method must return a DataLoader "
            f"that yields batches with these exact keys. See the IBCircuit "
            f"documentation for the expected batch format."
        )

    # Validate types and shapes
    if not isinstance(batch["tokens"], t.Tensor):
        raise ValueError(f"batch['tokens'] must be a torch.Tensor, got {type(batch['tokens'])}")

    if not isinstance(batch["labels"], t.Tensor):
        raise ValueError(f"batch['labels'] must be a torch.Tensor, got {type(batch['labels'])}")

    if not isinstance(batch["answer_positions"], t.Tensor):
        raise ValueError(
            f"batch['answer_positions'] must be a torch.Tensor, "
            f"got {type(batch['answer_positions'])}"
        )

    # Validate shapes are consistent
    batch_size = batch["tokens"].shape[0]

    if batch["labels"].shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: tokens has {batch_size} examples but "
            f"labels has {batch['labels'].shape[0]} examples"
        )

    if batch["answer_positions"].shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: tokens has {batch_size} examples but "
            f"answer_positions has {batch['answer_positions'].shape[0]} examples"
        )

    # Validate answer_positions are within sequence bounds
    seq_len = batch["tokens"].shape[1]
    max_pos = batch["answer_positions"].max().item()

    if max_pos >= seq_len:
        raise ValueError(
            f"Invalid answer_positions: max position {max_pos} is >= "
            f"sequence length {seq_len}. All answer positions must be "
            f"valid indices into the sequence."
        )


# ── Shared helpers used by discover_circuit and evaluate_circuit ──────────────────


def _avg_scores(scores) -> float:
    """
    Reduce per-sample metric scores to a single Python float.

    Args:
        scores (Tensor | List[Tensor]): Per-sample scores. If a list, each
            element is averaged first, then those averages are averaged.

    Returns:
        float: Mean score across all samples.
    """
    if isinstance(scores, list):
        return t.mean(t.stack([t.mean(s.float()) for s in scores])).item()
    return t.mean(scores.float()).item() if scores.numel() > 1 else scores.item()


def _make_eval_metric(task_spec):
    """
    Build a per-sample, non-loss metric callable from a TaskSpec.

    For partial-based metrics, overrides 'loss=False' and 'mean=False' so the
    metric returns raw per-sample scores suitable for faithfulness evaluation.
    Non-partial callables are returned unchanged.

    Args:
        task_spec: A registered TaskSpec with a metric_fn() method.

    Returns:
        Callable: Metric with signature
            (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
    """
    base = task_spec.metric_fn()
    if isinstance(base, partial):
        kw = base.keywords.copy()
        kw["loss"] = False
        kw["mean"] = False
        return partial(base.func, **kw)
    return base


def _compute_n_topn(graph: Graph, scope: str, sparsity: float):
    """
    Compute the apply_topn budget for node-level pruning under a given scope.

    Out-of-scope nodes are always kept, so n_topn includes their count on top
    of the in-scope budget. n_to_keep reflects only in-scope nodes and is used
    when building an equivalently-sized random baseline.

    Args:
        graph (Graph): Graph containing n_layers and n_heads in its cfg.
        scope (str): Which components are prunable — 'heads', 'mlp', or 'both'.
        sparsity (float): Fraction of in-scope nodes to remove (0.0-1.0).

    Returns:
        Tuple[int, int]: (n_topn, n_to_keep) where n_topn is passed to
            graph.apply_topn() and n_to_keep is the in-scope keep count.
    """
    n_layers = graph.cfg["n_layers"]
    n_heads = n_layers * graph.cfg["n_heads"]
    n_mlps = n_layers

    if scope == "heads":
        n_to_keep, n_always = int(n_heads * (1 - sparsity)), n_mlps
    elif scope == "mlp":
        n_to_keep, n_always = int(n_mlps * (1 - sparsity)), n_heads
    else:  # both
        n_to_keep, n_always = int((n_heads + n_mlps) * (1 - sparsity)), 0

    return n_to_keep + n_always, n_to_keep


def _build_random_node_graph(model, scope: str, n_to_keep: int, seed=None) -> Graph:
    """
    Build a randomly-pruned node-level Graph for use as a faithfulness baseline.

    Only in-scope nodes are assigned a score (1.0) so they participate in
    random selection; out-of-scope nodes keep NaN scores and are always
    retained by apply_random. This ensures a fair comparison across all
    algorithms regardless of scope.

    Args:
        model (HookedTransformer): Model whose config defines the graph structure.
        scope (str): Which components are prunable — 'heads', 'mlp', or 'both'.
        n_to_keep (int): Number of in-scope nodes to keep (from _compute_n_topn).
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
        Graph: A pruned Graph with nodes_in_graph and in_graph set randomly.
    """
    rand = Graph.from_model(model, node_scores=True, neuron_level=False)
    for node in rand.nodes.values():
        if isinstance(node, (AttentionNode, MLPNode)):
            fwd_idx = rand.forward_index(node)
            in_scope = (
                (scope == "heads" and isinstance(node, AttentionNode))
                or (scope == "mlp" and isinstance(node, MLPNode))
                or scope == "both"
            )
            if in_scope:
                rand.nodes_scores[fwd_idx] = 1.0
            # out-of-scope stays NaN → apply_random always keeps it
    rand.apply_random(n_to_keep, level="node", prune=True, seed=seed)
    return rand


def _build_random_ibcircuit_neuron_pruning_dict(
    model: HookedTransformer,
    reference_pruning_dict: dict,
    scope: str,
    seed: int = None,
) -> dict:
    """
    Build a random neuron pruning dict matching the IBCircuit discovery budget.

    Samples the same total number of neurons as the reference dict, drawn
    uniformly from the same neuron space IBCircuit searches:
    - Attention: d_head neurons per head (hook_z space).
    - MLP: d_mlp or d_model neurons per layer depending on mlp_hook.

    Used as a random baseline to contextualise faithfulness scores.

    Args:
        model (HookedTransformer): Model whose architecture defines the neuron space.
        reference_pruning_dict (dict): Pruning dict produced by IBCircuit discovery,
            used solely to count the total neurons pruned. Expected keys: 'heads',
            'mlp', '_meta'.
        scope (str): Which components to sample from — 'heads', 'mlp', or 'both'.
        seed (Optional[int]): Random seed for reproducibility. Defaults to None.

    Returns:
        Dict: Pruning dict with keys 'mlp', 'heads', '_meta', in the same
            format as the IBCircuit discovery output.
    """
    n_to_prune = sum(len(v) for v in reference_pruning_dict.get("heads", {}).values()) + sum(
        len(v) for v in reference_pruning_dict.get("mlp", {}).values()
    )

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    model.cfg.d_model

    mlp_hook = reference_pruning_dict.get("_meta", {}).get("mlp_hook", "mlp_out")
    mlp_dim = model.cfg.d_mlp if mlp_hook == "post_act" else model.cfg.d_model

    all_neurons = []
    if scope in ("heads", "both"):
        for layer in range(n_layers):
            for head in range(n_heads):
                for ni in range(d_head):
                    all_neurons.append(("attn", layer, head, ni))
    if scope in ("mlp", "both"):
        for layer in range(n_layers):
            for ni in range(mlp_dim):
                all_neurons.append(("mlp", layer, None, ni))

    if seed is not None:
        t.manual_seed(seed)
    perm = t.randperm(len(all_neurons)).tolist()
    selected = [all_neurons[i] for i in perm[:n_to_prune]]

    rand_mlp = defaultdict(list)
    rand_heads = defaultdict(list)
    for kind, layer, head, ni in selected:
        if kind == "mlp":
            rand_mlp[layer].append(ni)
        else:
            rand_heads[(layer, head)].append(ni)

    return {"mlp": dict(rand_mlp), "heads": dict(rand_heads), "_meta": {"mlp_hook": mlp_hook}}


def _build_artifact_stem(config: Dict[str, Any]) -> str:
    """
    Build a descriptive filename stem from a discovery config.

    Format: '{algo}_{task}_{model}_{scope_or_level}_sp{sparsity}[_{extras}]'
    Examples:
        eap-ig_ioi_gpt2_neuron_sp0.3
        ibcircuit_ioi_gpt2-small_heads_sp0.2_e1000
        acdc_greater-than_pythia-70m_node_sp0.5

    Args:
        config (Dict[str, Any]): Validated discovery config containing
            'model', 'discovery', and 'pruning' sub-dicts.

    Returns:
        str: Underscore-joined filename stem, safe for use in file paths.
    """
    disc = config["discovery"]
    prune = config["pruning"]
    algo = disc["algorithm"].lower()
    task = disc["task"]
    model = config["model"]["name"].split("/")[-1]  # strip org prefix
    # Use defaults from DEFAULT_CONFIG (single source of truth)
    default_discovery = DEFAULT_CONFIG["discovery"]
    default_pruning = DEFAULT_CONFIG["pruning"]
    level = disc.get("level", default_discovery.get("level"))
    scope = (
        disc.get("scope", default_discovery.get("scope"))
        if algo == "ibcircuit"
        else prune.get("scope", default_pruning.get("scope"))
    )
    sp = prune.get("target_sparsity", default_pruning.get("target_sparsity"))

    parts = [algo, task, model, scope if algo == "ibcircuit" else level, f"sp{sp}"]

    # Algo-specific differentiators
    if algo == "ibcircuit":
        parts.append(f"e{disc.get('num_epochs', default_discovery.get('num_epochs'))}")
        if disc.get("mlp_hook", default_discovery.get("mlp_hook")) != default_discovery.get(
            "mlp_hook"
        ):
            parts.append(disc["mlp_hook"])
    elif algo in ("eap", "eap-ig"):
        if disc.get("method"):
            parts.append(disc["method"].lower().replace("-", ""))

    return "_".join(str(p) for p in parts)


def _save_artifact(
    data: Any, output_path: str, suffix: str, logger, config: Dict = None
) -> Optional[str]:
    """
    Save a discovery artifact to disk as a .pt file.

    If output_path is not provided, a path is auto-generated from the config
    stem under '{cwd}/outputs/'. A suffix (e.g. '_scores') is appended to
    the stem before the extension to differentiate artifact types saved at
    the same base path.

    Args:
        data (Any): Serialisable object to save (passed to torch.save).
        output_path (str): Base output path or directory. If a directory or
            no extension, a filename is generated from the config stem.
        suffix (str): String appended to the filename stem, e.g. '_scores'
            or '_ib_weights'.
        logger: Logger instance for info messages.
        config (Optional[Dict]): Discovery config used to build the filename
            stem when output_path is absent or a directory.

    Returns:
        Optional[str]: Absolute path of the saved file, or None if no path
            could be determined.
    """
    if not output_path and config:
        stem = _build_artifact_stem(config)
        output_path = os.path.join(os.getcwd(), "outputs", stem + ".pt")
    if not output_path:
        return None
    from pathlib import Path

    p = Path(output_path)
    # If output_path is a directory, generate filename inside it
    if p.is_dir() or not p.suffix:
        stem = _build_artifact_stem(config) if config else "circuit"
        p = p / (stem + ".pt")
    dest = p.parent / (p.stem + suffix + p.suffix)
    os.makedirs(str(p.parent), exist_ok=True)
    t.save(data, str(dest))
    logger.info(f"Saved '{suffix.lstrip('_')}' → {dest}")
    return str(dest)


def _build_circuit_scores(
    task: str,
    model_name: str,
    algorithm: str,
    node_scores: Dict[str, float],
    discovery_cfg: Optional[Dict] = None,
) -> CircuitScores:
    """
    Build a CircuitScores artifact from discovered scores.

    Helper to standardize the creation of CircuitScores across all backends.

    Args:
        task: Task name (e.g., 'ioi', 'mmlu').
        model_name: Model identifier (e.g., 'gpt2').
        algorithm: Algorithm name ('eap', 'eap-ig', 'acdc', 'ibcircuit').
        node_scores: Dict mapping node names to scores.
        discovery_cfg: Optional discovery configuration.

    Returns:
        CircuitScores artifact with timestamp and metadata.
    """
    return CircuitScores(
        task=task,
        model=model_name,
        algorithm=algorithm,
        level="node",
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp(),
        version="1.0",
        discovery_cfg=discovery_cfg or {},
    )


# ────────────────────────────────

def prepare_custom_task(
    config: Dict[str, Any],
    model: HookedTransformer,
    task_name: Optional[str] = None,
) -> str:
    """
    Normalise a config["data"] block into a registered CircuitKit task.

    Must be called once before discover_circuit() and evaluate_circuit()
    when config contains a "data" block. Mutates config in-place: sets
    config["discovery"]["task"] to the registered name and removes
    config["data"] so neither downstream function re-processes it.

    Args:
        config:     Full CircuitKit config dict with a "data" block.
        model:      Loaded HookedTransformer (tokenizer used for alignment).
        task_name:  Explicit registry name. Defaults to "custom:{csv_stem}".

    Returns:
        The registered task name string.
    """
    data_cfg = config.get("data")
    if not data_cfg:
        return config["discovery"]["task"]

    data_type = data_cfg.get("type")
    if not data_type:
        raise KeyError(
            "config['data']['type'] is required ('template', 'auto', or 'clean_only')"
        )

    if task_name is None:
        task_name = f"custom:{Path(data_cfg['path']).stem}"

    logger = get_logger("circuitkit.custom_data")

    if data_type == "template":
        from .data.template import clean_only_from_template, template_normalize
        from .tasks._algorithm_families import CDT_FAMILY, IB_FAMILY

        template = data_cfg.get("template", {})
        algo = config["discovery"].get("algorithm", "").lower()
        is_clean_only_algo = algo in (IB_FAMILY | CDT_FAMILY)
        has_corrupt_keys = bool(template.get("corrupt_prompt") and template.get("corrupt_answer"))

        if is_clean_only_algo and not has_corrupt_keys:
            # Algorithm only needs the clean side; skip full pairing pipeline.
            if not template.get("clean_prompt"):
                raise ValueError(
                    "config['data']['template'] must contain 'clean_prompt'."
                )
            ds = clean_only_from_template(
                data_cfg["path"],
                template_spec=template,
                max_records=data_cfg.get("max_records"),
                name=Path(data_cfg["path"]).stem,
                source=data_cfg["path"],
            )
            logger.info(
                f"template (clean-only extraction): {len(ds)} records loaded "
                f"for algorithm '{algo}' (corrupt keys omitted, no alignment pass)"
            )
        else:
            required = ["clean_prompt", "corrupt_prompt", "clean_answer", "corrupt_answer"]
            missing = [k for k in required if not template.get(k)]
            if missing:
                raise ValueError(
                    f"config['data']['template'] is missing required fields: {missing}"
                )
            align_strategy = data_cfg.get("align_strategy", "filter")
            ds = template_normalize(
                data_cfg["path"],
                template_spec=template,
                pairing_mode=data_cfg.get("pairing_mode", "explicit"),
                align_strategy=align_strategy,
                tokenizer=model.tokenizer,
                pad_region_end=data_cfg.get("pad_region_end"),
                max_records=data_cfg.get("max_records"),
                name=Path(data_cfg["path"]).stem,
                source=data_cfg["path"],
            )
            align_meta = ds.meta.get("_alignment", {})
            logger.info(
                f"template dataset: {align_meta.get('kept')}/{align_meta.get('total_input')} "
                f"records kept after alignment (strategy={align_strategy!r}, "
                f"dropped_nondiscriminative={align_meta.get('dropped_nondiscriminative')}, "
                f"dropped_misaligned={align_meta.get('dropped_misaligned')}, "
                f"dropped_pad_failed={align_meta.get('dropped_pad_failed')}, "
                f"recommended_metric={align_meta.get('recommended_metric')!r})"
            )

    elif data_type == "auto":
        from .data.auto_detect import auto_normalize

        ds = auto_normalize(
            data_cfg["path"],
            apply_default_strategy=True,
            max_records=data_cfg.get("max_records"),
            name=data_cfg.get("name", Path(data_cfg["path"]).stem),
            source=data_cfg["path"],
        )

    elif data_type == "clean_only":
        from .data.clean_only import clean_only_normalize

        ds = clean_only_normalize(
            data_cfg["path"],
            prompt_column=data_cfg.get("prompt_column", "prompt"),
            answer_column=data_cfg.get("answer_column", "answer"),
            max_records=data_cfg.get("max_records"),
            name=data_cfg.get("name", Path(data_cfg["path"]).stem),
            source=data_cfg["path"],
        )
        logger.info(
            f"clean_only dataset: {len(ds)} records loaded "
            f"(no corrupt partner; compatible with ibcircuit, cdt)"
        )

    else:
        raise ValueError(
            f"Unknown data.type {data_type!r}. Use 'template', 'auto', or 'clean_only'."
        )

    from .data.normalized_task import NormalizedTaskSpec

    task_spec = NormalizedTaskSpec(ds, name=task_name)
    padding = data_cfg.get("pair_padding_side")
    if padding in ("left", "right"):
        task_spec.pair_padding_side = padding

    try:
        _register_task(task_spec)
    except ValueError as e:
        if "already registered" not in str(e):
            raise
        logger.info(f"Task '{task_name}' already registered, reusing.")

    config["discovery"]["task"] = task_name
    config.pop("data", None)
    logger.info(f"Custom task '{task_name}' registered ({len(ds)} records, {ds.n_paired} paired)")
    return task_name

@debug_function
@handle_errors(context={"operation": "discover_circuit"})
def discover_circuit(  # noqa: C901 - complex function, refactor out of scope for lint pass
    config: Union[str, Dict[str, Any]],
    _model: Optional[HookedTransformer] = None,
) -> Union[List[str], Dict]:
    """
    Run circuit discovery and return a pruning artifact.

    Loads the model and task, runs the specified attribution algorithm,
    applies sparsity-based pruning, and optionally evaluates faithfulness.

    Args:
        config: Path to a YAML file or a config dict with keys:
            ``model.name``, ``model.precision``, ``discovery.algorithm``,
            ``discovery.task``, ``discovery.level``, ``pruning.target_sparsity``,
            ``pruning.scope``, ``output_path``.
        _model: Internal. An already-loaded HookedTransformer to reuse.
            Leave as ``None`` for external callers.

    Returns:
        Node-level: list of node name strings. Neuron-level: dict with
        ``mlp``, ``heads``, and ``_meta`` keys.

    Raises:
        ValueError: If required config keys are missing.
        AlgorithmError: If the algorithm is not recognised.
    """
    # Bootstrap built-in tasks
    from .tasks.bootstrap import _bootstrap_builtin_tasks

    _bootstrap_builtin_tasks()

    logger = get_logger("circuitkit.discovery")
    progress = ProgressLogger(logger)

    # Snapshot of the caller's global RNG state, captured iff we seed below so
    # the ``finally`` can restore it (see the seed block).
    _rng_snapshot = None

    try:
        # Load, merge defaults, and validate the config in one step
        progress.start_operation("Circuit Discovery", 4)
        progress.step("Loading and validating configuration")
        
        config = load_and_validate_config(config)
        logger.log_config(config)

        model_cfg = config["model"]
        discovery_cfg = config["discovery"]
        pruning_cfg = config["pruning"]

        # Seed all global RNGs when the config supplies a seed, so stochastic
        # algorithms (IBCircuit, CD-T/ACDC data generation via numpy/`random`)
        # are reproducible. We snapshot the caller's global RNG state first and
        # restore it in the ``finally`` — otherwise discovery would permanently
        # reseed the whole process's numpy/random/torch RNGs as a side effect.
        _seed = discovery_cfg.get("seed", discovery_cfg.get("data_params", {}).get("seed"))
        if _seed is not None:
            import random as _random_std
            import numpy as _np_std
            _rng_snapshot = (
                t.get_rng_state(),
                _np_std.random.get_state(),
                _random_std.getstate(),
                t.cuda.get_rng_state_all() if t.cuda.is_available() else None,
            )
            t.manual_seed(_seed)
            _np_std.random.seed(_seed)
            _random_std.seed(_seed)
            if t.cuda.is_available():
                t.cuda.manual_seed_all(_seed)

        is_verbose = discovery_cfg.get("verbose", False)
        if is_verbose:
            import logging

            get_logger("circuitkit").setLevel(logging.DEBUG)
            get_logger("data").setLevel(logging.DEBUG)
            logger.debug(f"Discovery Config: {config['discovery']}")

        # Resolve and Sanitize Discovery Intervention
        # Use default from DEFAULT_CONFIG (single source of truth)
        default_discovery = DEFAULT_CONFIG["discovery"]
        discovery_intervention = discovery_cfg.get(
            "intervention", default_discovery.get("intervention")
        )

        # IG methods strictly require patching to function
        if discovery_cfg["algorithm"].lower() in [
            "eap-ig",
            "eap-ig-activations",
            "clean-corrupted",
        ]:
            if discovery_intervention != "patching":
                logger.warning(
                    f"Safety Override: {discovery_cfg['algorithm']} requires 'patching'. "
                    f"Changing discovery intervention from '{discovery_intervention}' to 'patching'."
                )
                discovery_intervention = "patching"

        if is_verbose:
            if discovery_cfg["algorithm"].lower() == "ibcircuit":
                # IBCircuit uses stochastic mean-ablation via IB Noise
                logger.debug("Discovery Phase Intervention: IB Noise (Stochastic Mean-Ablation)")
            else:
                logger.debug(f"Discovery Phase Intervention: {discovery_intervention}")

        # Validate model name
        validate_model_name(model_cfg["name"])
        if "algorithm" not in discovery_cfg:
            from .backends import DISCOVERY_ALGORITHMS

            raise ValueError(
                "Discovery config is missing the required key 'algorithm'. "
                "Add an 'algorithm' key under the discovery config. "
                "Supported discovery algorithms: "
                f"{', '.join(sorted(DISCOVERY_ALGORITHMS))}."
            )
        validate_discovery_algorithm(discovery_cfg["algorithm"])

        progress.step("Setting up model", model=model_cfg["name"])
        device = get_device()
        # Use default from DEFAULT_CONFIG (single source of truth)
        default_model = DEFAULT_CONFIG["model"]
        dtype = getattr(t, model_cfg.get("precision", default_model.get("precision")))

        if _model is not None:
            # Reuse the caller's already-loaded model instead of loading a
            # second full copy (e.g. quick.discover()/Pipeline.discover()
            # already built one via load_model()/_ensure_model()).
            model = _model
            logger.debug("discover_circuit: reusing pre-loaded model, skipping reload")
        else:
            with log_execution_time("Model loading", logger):
                model = HookedTransformer.from_pretrained(
                    model_cfg["name"], device=device, dtype=dtype
                )

        algo = discovery_cfg["algorithm"].lower()

        if hasattr(model.cfg, "ungroup_grouped_query_attention"):
            model.cfg.ungroup_grouped_query_attention = True
        
        # ── Inline data path: delegate to prepare_custom_task ───────────
        if config.get("data") and config["data"].get("type"):
            prepare_custom_task(config, model=model)
            discovery_cfg["task"] = config["discovery"]["task"]
        # ── End inline data path ─────────────────────────────────────────
        
        # Resolve and validate task spec (explicit, no defaults)
        if "task" not in discovery_cfg:
            from .tasks.registry import list_tasks

            raise ValueError(
                "Discovery config is missing the required key 'task'. "
                "Add a 'task' key under the discovery config naming the task to "
                f"discover. Registered tasks: {list_tasks()}."
            )
        task_spec = _get_task(discovery_cfg["task"])
        task_spec.validate_discovery_config(discovery_cfg)
        
        if algo in (
            "acdc",
            "eap",
            "eap-ig",
            "eap-ig-activations",
            "eap-clean-corrupted",
            "eap-exact",
            "atp-gd",
            "eap-gp",
            "relp",
            "peap",
            "eap-ifr",
        ):
            model.cfg.use_attn_result = True
            model.cfg.use_split_qkv_input = True
            model.cfg.use_hook_mlp_in = True

        # Warn about experimental / research algorithms
        if algo in RESEARCH_ALGORITHMS:
            _warnings.warn(
                f"Algorithm '{algo}' is research-quality (only validated on GPT-2 IOI). "
                f"Use '{_DEFAULT_ALGO}' for production.",
                UserWarning,
                stacklevel=2,
            )
        elif algo in EXPERIMENTAL_ALGORITHMS:
            _warnings.warn(
                f"Algorithm '{algo}' is experimental. May fail on larger models or non-IOI tasks. "
                f"Use '{_DEFAULT_ALGO}' for production.",
                UserWarning,
                stacklevel=2,
            )

        logger.log_model_info(
            model_cfg["name"],
            device=device,
            dtype=str(dtype),
            parameters=sum(p.numel() for p in model.parameters()),
        )

        progress.step("Running discovery algorithm", algorithm=algo)
        logger.info(f"Starting {algo.upper()} discovery algorithm")

        # Use defaults from DEFAULT_CONFIG (single source of truth)
        default_discovery = DEFAULT_CONFIG["discovery"]
        _ib_scope = discovery_cfg.get("scope", default_discovery.get("scope"))

        if algo == "acdc":
            with debug_context("ACDC Discovery"):
                p_model = patchable_model(
                    model,
                    factorized=True,
                    slice_output="last_seq",
                    separate_qkv=True,
                    device=device,
                )
                train_loader, _ = load_task_data(
                    task_name=discovery_cfg["task"],
                    model=model,
                    device=device,
                    **discovery_cfg.get("data_params", {}),
                )
                # ACDC sweeps one full edge pass per (base, exp) tao value.
                # With the library defaults (5 bases x 4 exps = 20 sweeps of
                # ~32k edges) a single GPT-2 run takes hours. Expose the tao
                # grid via discovery_cfg so callers can scope the search;
                # fall back to the backend defaults when unspecified.
                _acdc_kwargs = {}
                if "tao_exps" in discovery_cfg:
                    _acdc_kwargs["tao_exps"] = list(discovery_cfg["tao_exps"])
                if "tao_bases" in discovery_cfg:
                    _acdc_kwargs["tao_bases"] = list(discovery_cfg["tao_bases"])
                if "faithfulness_target" in discovery_cfg:
                    _acdc_kwargs["faithfulness_target"] = discovery_cfg["faithfulness_target"]
                # verbose=True shows tqdm bars; False (default) emits progress as
                # DEBUG log messages on circuitkit.backends.acdc.prune_algos.ACDC.
                _acdc_kwargs["verbose"] = discovery_cfg.get("verbose", False)
                edge_scores = acdc_prune_scores(
                    p_model, train_loader, official_edges=None, **_acdc_kwargs
                )
                node_scores = calculate_node_scores_from_edges(p_model, edge_scores)

                # Build unified CircuitScores artifact (Workstream G)
                circuit_scores = _build_circuit_scores(
                    task=discovery_cfg["task"],
                    model_name=model_cfg["name"],
                    algorithm=algo,
                    node_scores=node_scores,
                    discovery_cfg=discovery_cfg,
                )

                # Save CircuitScores as JSON
                if config.get("output_path"):
                    scores_path = Path(config["output_path"]).parent / (
                        Path(config["output_path"]).stem + "_scores.json"
                    )
                    circuit_scores.to_json(scores_path)
                    logger.info(f"Saved unified CircuitScores → {scores_path}")

                # Also save legacy format for compatibility
                _save_artifact(
                    {"algo": algo, "level": "node", "node_scores": node_scores},
                    config.get("output_path"),
                    "_scores",
                    logger,
                )

        elif algo in [
            "eap",
            "eap-ig",
            # Tier-0 promotions: top-level keys for the
            # 4 EAP-internal methods that previously could
            # only be selected via discovery_cfg['method'].
            "eap-ig-activations",
            "eap-clean-corrupted",
            "eap-exact",
            # AtP+GradDrop (Kramár et al. 2024) — same EAP backbone
            # with L gradient passes, one residual gradient zeroed each.
            "atp-gd",
            # EAP-GP (Zhang et al. 2025) — adaptive integration path
            # in input embedding space; replaces EAP-IG's straight line.
            "eap-gp",
            # RelP (Mohebbi et al. 2025) — LRP-style relevance
            # propagation via forward detach hooks; same EAP cost.
            "relp",
            # PEAP (Haklay et al. 2025) — per-position retention
            # of EAP scores; node-level summary preserved.
            "peap",
            # IFR / Information Flow Routes (Ferrando et al. 2024)
            # — proximity-based attribution, no metric needed.
            "eap-ifr",
        ]:
            with debug_context("EAP Discovery"):
                # Use TaskSpec for dataloader and metric
                dataloader = task_spec.build_dataloader(model, discovery_cfg, device)

                if "level" not in discovery_cfg:
                    raise ValueError(
                        "Discovery config is missing the required key 'level'. "
                        "Add a 'level' key under the discovery config set to "
                        "'node' or 'neuron'."
                    )
                is_neuron_level = discovery_cfg["level"] == "neuron"
                # Use defaults from DEFAULT_CONFIG (single source of truth)
                default_discovery = DEFAULT_CONFIG["discovery"]
                mlp_hook = discovery_cfg.get("mlp_hook", default_discovery.get("mlp_hook"))
                graph = Graph.from_model(
                    model, node_scores=True, neuron_level=is_neuron_level, mlp_hook=mlp_hook
                )

                metric = task_spec.metric_fn()

                logger.debug(
                    f"Graph initialized. Nodes: {len(graph.nodes)}. Neuron Level: {is_neuron_level}"
                )

                # Map top-level algorithm keys to internal `method` arg.
                _ALGO_METHOD_MAP = {
                    "eap": "EAP",
                    "eap-ig": "EAP-IG-inputs",
                    "eap-ig-activations": "EAP-IG-activations",
                    "eap-clean-corrupted": "clean-corrupted",
                    "eap-exact": "exact",
                    "atp-gd": "atp-gd",
                    "eap-gp": "eap-gp",
                    "relp": "relp",
                    "peap": "peap",
                    "eap-ifr": "ifr",
                }
                if algo == "eap-ig":
                    # eap-ig still supports an explicit method override
                    # (legacy behaviour) for users who want to dispatch
                    # via discovery_cfg['method'].
                    _valid_node_methods = (
                        "EAP",
                        "EAP-IG-inputs",
                        "EAP-IG-activations",
                        "exact",
                        "clean-corrupted",
                    )
                    _method = discovery_cfg.get("method", default_discovery.get("method"))
                    if _method not in _valid_node_methods:
                        raise ValueError(
                            f"discovery config key 'method' has invalid value "
                            f"{_method!r} for algorithm 'eap-ig'. "
                            f"Set 'method' to one of: {list(_valid_node_methods)}. "
                            f"Note: the algorithm name 'eap-ig' is not itself a "
                            f"valid 'method' string."
                        )
                else:
                    _method = _ALGO_METHOD_MAP[algo]
                attribute_node(
                    model,
                    graph,
                    dataloader,
                    metric,
                    method=_method,
                    ig_steps=discovery_cfg.get("ig_steps", default_discovery.get("ig_steps")),
                    neuron=is_neuron_level,
                    intervention=discovery_intervention,
                )

                if not is_neuron_level:
                    node_scores = _convert_eap_scores_to_ck_format(graph)

                    # Build unified CircuitScores artifact (Workstream G)
                    circuit_scores = _build_circuit_scores(
                        task=discovery_cfg["task"],
                        model_name=model_cfg["name"],
                        algorithm=algo,
                        node_scores=node_scores,
                        discovery_cfg=discovery_cfg,
                    )

                    # Save CircuitScores as JSON
                    if config.get("output_path"):
                        scores_path = Path(config["output_path"]).parent / (
                            Path(config["output_path"]).stem + "_scores.json"
                        )
                        circuit_scores.to_json(scores_path)
                        logger.info(f"Saved unified CircuitScores → {scores_path}")

                    # Also save legacy format for compatibility
                    _save_artifact(
                        {"algo": algo, "level": "node", "node_scores": node_scores},
                        config.get("output_path"),
                        "_scores",
                        logger,
                    )

                else:
                    # Handle neuron-level results
                    default_pruning = DEFAULT_CONFIG["pruning"]
                    effective_scope = pruning_cfg.get("scope", default_pruning.get("scope"))
                    logger.info(
                        f"Processing neuron-level scores (scope: {effective_scope}, strategy: per-layer)"
                    )
                    pruned_mlp_neurons = defaultdict(list)
                    pruned_attn_neurons = defaultdict(list)

                    all_neuron_scores = []
                    for node in tqdm(graph.nodes.values(), desc="Extracting neuron scores"):
                        if isinstance(node, (MLPNode, AttentionNode)):
                            # Filter by scope
                            if effective_scope == "mlp" and not isinstance(node, MLPNode):
                                continue
                            if effective_scope == "heads" and not isinstance(node, AttentionNode):
                                continue

                            fwd_index = graph.forward_index(node, attn_slice=False)
                            scores_tensor = graph.neurons_scores[fwd_index].clone().detach().cpu()
                            # Truncate to actual activation dimension to avoid counting padding zeros
                            valid_scores = scores_tensor[: node.d_neuron]
                            for neuron_idx, score in enumerate(valid_scores):
                                all_neuron_scores.append(
                                    (abs(score.item()), (node.name, neuron_idx))
                                )

                    all_neuron_scores.sort(key=lambda x: x[0])  # Sort by absolute score, ascending
                    num_to_prune = int(len(all_neuron_scores) * pruning_cfg["target_sparsity"])

                    logger.debug(
                        f"Total Neurons: {len(all_neuron_scores)}, Pruning: {num_to_prune}"
                    )
                    neurons_to_prune_info = all_neuron_scores[:num_to_prune]

                    for score, (node_name, neuron_idx) in neurons_to_prune_info:
                        mlp_match = re.match(r"m(\d+)", node_name)
                        attn_match = re.match(r"a(\d+)\.h(\d+)", node_name)
                        if mlp_match:
                            pruned_mlp_neurons[int(mlp_match.group(1))].append(neuron_idx)
                        elif attn_match:
                            pruned_attn_neurons[
                                (int(attn_match.group(1)), int(attn_match.group(2)))
                            ].append(neuron_idx)

                    result = {
                        "mlp": dict(pruned_mlp_neurons),
                        "heads": dict(pruned_attn_neurons),
                        "_meta": {
                            "mlp_hook": discovery_cfg.get("mlp_hook", "mlp_out"),
                            "heads_hook": "attn.hook_result",  # EAP uses attn.hook_result for heads
                        },
                    }
                    if config.get("output_path"):
                        os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
                        t.save(result, config["output_path"])
                        logger.info(f"Neuron pruning dictionary saved to {config['output_path']}")
                    _save_artifact(
                        {
                            "algo": algo,
                            "level": "neuron",
                            "neurons_scores": graph.neurons_scores.cpu(),
                            "total_neurons": len(all_neuron_scores),
                        },
                        config.get("output_path"),
                        "_scores",
                        logger,
                    )

                    # `graph` (and its GPU-resident neurons_scores tensor) is
                    # fully consumed at this point - the pruning dict and the
                    # CPU-side scores side-car are already built/saved above,
                    # and nothing below this line reads `graph` again. Free it
                    # before the optional inline evaluation, which loads/uses
                    # its own evaluation dataloaders and graph reconstruction
                    # and does not need this one.
                    del graph
                    if t.cuda.is_available():
                        empty_cache()

                    if discovery_cfg.get("evaluate", False):
                        config["_eval_result"] = evaluate_circuit(
                            config,
                            pruned_artifact_path=config.get("output_path"),
                            _model=model,
                        )

                    progress.complete(neurons_pruned=num_to_prune)
                    return result

        elif algo == "ibcircuit":
            from .backends.ibcircuit.trainer import run_ib_discovery as _run_ib

            # Build IBCircuit-format dataloader via TaskSpec.
            # TaskSpec.build_dataloader is the abstraction boundary:
            # it knows the task format, we don't need to.
            dataloader = task_spec.build_dataloader(model, discovery_cfg, device)

            # Forward only the training hyperparameters - no path/save keys.
            # Saving is api.py's responsibility, not the trainer's.
            # All defaults come from DEFAULT_CONFIG in utils/config.py (single source of truth)
            default_discovery = DEFAULT_CONFIG["discovery"]
            ib_config = {
                "num_epochs": discovery_cfg.get("num_epochs", default_discovery.get("num_epochs")),
                "learning_rate": discovery_cfg.get(
                    "learning_rate", default_discovery.get("learning_rate")
                ),
                "alpha": discovery_cfg.get("alpha", default_discovery.get("alpha")),
                "beta": discovery_cfg.get("beta", default_discovery.get("beta")),
                "alpha_loss": discovery_cfg.get("alpha_loss", default_discovery.get("alpha_loss")),
                "log_interval": discovery_cfg.get(
                    "log_interval", default_discovery.get("log_interval")
                ),
                "scope": discovery_cfg.get("scope", default_discovery.get("scope")),
                "mask_type": discovery_cfg.get("mask_type", default_discovery.get("mask_type")),
                "level": discovery_cfg.get("level", default_discovery.get("level")),
                "mlp_hook": discovery_cfg.get("mlp_hook", default_discovery.get("mlp_hook")),
                "batch_size": discovery_cfg.get("batch_size", default_discovery.get("batch_size")),
            }

            _validate_ibcircuit_dataloader(dataloader)

            # Returns {"A{layer}.{head}": float, ...}
            # Higher score = more important head.
            node_scores, ib_model = _run_ib(
                model=model, dataloader=dataloader, config=ib_config, device=device
            )

            # Save IB model weights and discovery scores (api.py owns persistence)
            _save_artifact(
                {
                    "attn_ib_weights": ib_model.attn_ib_weights.state_dict(),
                    "mlp_ib_weights": ib_model.mlp_ib_weights.state_dict(),
                    "scope": ib_model.scope,
                    "batch_size": ib_model.batch_size,
                    "n_layers": ib_model.n_layers,
                    "n_heads": ib_model.n_heads,
                    "mask_type": ib_model.mask_type,
                    "level": ib_model.level,
                    "mlp_hook": ib_model.mlp_hook,
                },
                config.get("output_path"),
                "_ib_weights",
                logger,
            )

            # Capture the one attribute this branch still needs below, then
            # free ib_model - its weights are already saved to disk above,
            # and nothing after this point references ib_model again.
            _ib_level = ib_model.level
            del ib_model
            if t.cuda.is_available():
                empty_cache()

            if _ib_level == "neuron":
                # Neuron-level: convert {IBCircuit_name: tensor} → pruning dict,
                # save in the same format as EAP neuron so evaluate_circuit works.
                pruned_mlp_neurons = defaultdict(list)
                pruned_attn_neurons = defaultdict(list)
                all_neuron_scores = []

                for ib_name, score_tensor in node_scores.items():
                    attn_match = re.match(r"A(\d+)\.(\d+)$", ib_name)
                    mlp_match = re.match(r"MLP (\d+)$", ib_name)
                    for neuron_idx, score in enumerate(score_tensor):
                        if attn_match:
                            all_neuron_scores.append(
                                (
                                    abs(score.item()),
                                    (
                                        "attn",
                                        int(attn_match.group(1)),
                                        int(attn_match.group(2)),
                                        neuron_idx,
                                    ),
                                )
                            )
                        elif mlp_match:
                            all_neuron_scores.append(
                                (
                                    abs(score.item()),
                                    ("mlp", int(mlp_match.group(1)), None, neuron_idx),
                                )
                            )

                all_neuron_scores.sort(key=lambda x: x[0])  # ascending: lowest = least important
                n_to_prune = int(len(all_neuron_scores) * pruning_cfg["target_sparsity"])

                logger.info(
                    f"IBCircuit neuron discovery: {len(all_neuron_scores)} total neurons, pruning {n_to_prune}"
                )

                for _, (kind, layer, head, neuron_idx) in all_neuron_scores[:n_to_prune]:
                    if kind == "mlp":
                        pruned_mlp_neurons[layer].append(neuron_idx)
                    else:
                        pruned_attn_neurons[(layer, head)].append(neuron_idx)

                result = {
                    "mlp": dict(pruned_mlp_neurons),
                    "heads": dict(pruned_attn_neurons),
                    "_meta": {"mlp_hook": ib_config["mlp_hook"]},
                }

                _save_artifact(
                    {
                        "algo": algo,
                        "level": "neuron",
                        "neurons_scores": node_scores,
                        "total_neurons": len(all_neuron_scores),
                    },
                    config.get("output_path"),
                    "_scores",
                    logger,
                )

                output_path = config.get("output_path")
                if output_path:
                    parent = os.path.dirname(output_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    t.save(result, output_path)
                    logger.info(f"Neuron pruning dict saved to {output_path}")

                if discovery_cfg.get("evaluate", False):
                    config["_eval_result"] = evaluate_circuit(
                        config,
                        pruned_artifact_path=output_path,
                        _model=model,
                    )

                progress.complete(neurons_pruned=n_to_prune)
                return result

            else:
                # Node-level: existing behaviour
                # Build unified CircuitScores artifact (Workstream G)
                circuit_scores = _build_circuit_scores(
                    task=discovery_cfg["task"],
                    model_name=model_cfg["name"],
                    algorithm=algo,
                    node_scores=node_scores,
                    discovery_cfg=discovery_cfg,
                )

                # Save CircuitScores as JSON
                if config.get("output_path"):
                    scores_path = Path(config["output_path"]).parent / (
                        Path(config["output_path"]).stem + "_scores.json"
                    )
                    circuit_scores.to_json(scores_path)
                    logger.info(f"Saved unified CircuitScores → {scores_path}")

                # Also save legacy format for compatibility
                _save_artifact(
                    {"algo": algo, "level": "node", "node_scores": node_scores},
                    config.get("output_path"),
                    "_scores",
                    logger,
                )

        elif algo == "cdt":
            with debug_context("CD-T Discovery"):
                
                if discovery_cfg.get("level") == "neuron":
                    raise ValueError(
                        "CD-T only supports node-level discovery in the current version. "
                        "Set discovery config key 'level' to 'node', or choose an "
                        "algorithm that supports neuron-level (e.g. eap, eap-ig, ibcircuit)."
                    )
                
                from .backends.cdt.adapter import run_cdt_discovery

                dataloader = task_spec.build_dataloader(model, discovery_cfg, device)
                node_scores = run_cdt_discovery(
                    tl_model=model,
                    dataloader=dataloader,
                    device=device,
                    n_examples=discovery_cfg.get("data_params", {}).get("num_examples", 16),
                )

                circuit_scores = _build_circuit_scores(
                    task=discovery_cfg["task"],
                    model_name=model_cfg["name"],
                    algorithm=algo,
                    node_scores=node_scores,
                    discovery_cfg=discovery_cfg,
                )
                if config.get("output_path"):
                    scores_path = Path(config["output_path"]).parent / (
                        Path(config["output_path"]).stem + "_scores.json"
                    )
                    circuit_scores.to_json(scores_path)
                _save_artifact(
                    {"algo": algo, "level": "node", "node_scores": node_scores},
                    config.get("output_path"),
                    "_scores",
                    logger,
                )
        else:
            from .backends import DISCOVERY_ALGORITHMS

            raise AlgorithmError(
                f"Unknown discovery algorithm '{algo}'. Set the discovery config "
                f"key 'algorithm' to one of: {sorted(DISCOVERY_ALGORITHMS)}."
            )

        progress.step("Identifying nodes to prune")

        effective_scope = _ib_scope if algo == "ibcircuit" else pruning_cfg.get("scope", "both")
        nodes_to_prune = get_nodes_to_prune(
            node_scores,
            target_sparsity=pruning_cfg["target_sparsity"],
            pruning_scope=effective_scope,
        )

        logger.info(f"  Pruned {len(nodes_to_prune)} nodes out of {len(node_scores)} candidates")

        if config.get("output_path"):
            parent = os.path.dirname(config["output_path"])
            if parent:
                os.makedirs(parent, exist_ok=True)
            t.save(nodes_to_prune, config["output_path"])
            logger.info(f"Pruned node list saved to {config['output_path']}")

        if discovery_cfg.get("evaluate", False):
            config["_eval_result"] = evaluate_circuit(
                config,
                pruned_artifact_path=config.get("output_path"),
                _model=model,
            )

        progress.complete(nodes_pruned=len(nodes_to_prune))
        return nodes_to_prune

    except Exception as e:
        progress.fail(str(e))
        raise
    finally:
        # Restore the caller's global RNG so a seeded discovery run doesn't
        # leak its deterministic RNG state into the surrounding process.
        if _rng_snapshot is not None:
            import random as _random_std
            import numpy as _np_std
            t.set_rng_state(_rng_snapshot[0])
            _np_std.random.set_state(_rng_snapshot[1])
            _random_std.setstate(_rng_snapshot[2])
            if _rng_snapshot[3] is not None:
                t.cuda.set_rng_state_all(_rng_snapshot[3])


def _save_evaluation_results_to_txt(
    evaluation_results: List[Dict[str, Any]],
    model_name: str,
    pruned_artifact_path: str,
    evaluation_mode: str,
    logger,
    custom_path: str = None,
) -> str:
    """
    Write lm-eval benchmark results to a plain-text file.

    Each entry in evaluation_results is rendered as a labelled block. On
    failure the error is logged and an empty string is returned rather than
    propagating the exception.

    Args:
        evaluation_results (List[Dict]): List of result dicts, each with keys
            'model_type' (str: 'original' | 'pruned') and 'results' (dict).
        model_name (str): HuggingFace model identifier, used in the filename
            when custom_path is not provided.
        pruned_artifact_path (str): Path to the pruning artifact, recorded in
            the file header for traceability.
        evaluation_mode (str): Evaluation mode label ('both', 'original', 'pruned').
        logger: Logger instance for info/error messages.
        custom_path (Optional[str]): Explicit output file path. If None, a
            timestamped file is created in the current working directory.

    Returns:
        str: Absolute path to the written file, or '' on failure.
    """
    try:
        if custom_path:
            file_path = custom_path
        else:
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{model_name.replace('/', '_')}_{timestamp}.txt"
            # Create the file path
            file_path = os.path.join(os.getcwd(), filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CIRCUITKIT EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")

            # Write metadata
            f.write(f"Model: {model_name}\n")
            f.write(f"Pruned Artifact: {pruned_artifact_path}\n")
            f.write(f"Evaluation Mode: {evaluation_mode}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Generated by: CircuitKit\n\n")

            # Write evaluation results
            for i, result in enumerate(evaluation_results, 1):
                f.write("-" * 60 + "\n")
                f.write(f"EVALUATION {i}: {result['model_type'].upper()} MODEL\n")
                f.write("-" * 60 + "\n\n")

                # Format and write the results
                results_data = result["results"]
                if isinstance(results_data, dict):
                    for task, score in results_data.items():
                        if isinstance(score, dict):
                            f.write(f"Task: {task}\n")
                            for metric, value in score.items():
                                f.write(f"  {metric}: {value}\n")
                            f.write("\n")
                        else:
                            f.write(f"{task}: {score}\n")
                else:
                    f.write(f"Results: {results_data}\n")

                f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Evaluation results saved to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Failed to save evaluation results to txt file: {e}")
        return ""


def _reconstruct_circuit_graph(
    model: HookedTransformer,
    scores_data: Dict,
    discovery_cfg: Dict[str, Any],
    pruning_cfg: Dict[str, Any],
    device: str,
) -> Graph:
    """
    Helper: Reconstruct pruned graph from scores.

    Handles all algorithms (ACDC, EAP, IBCircuit) and levels (node, neuron).
    Returns the reconstructed circuit graph with topn applied.
    """
    algo = discovery_cfg["algorithm"].lower()
    level = discovery_cfg.get("level", "node")
    scope = (
        discovery_cfg.get("scope", "heads")
        if algo == "ibcircuit"
        else pruning_cfg.get("scope", "both")
    )
    sparsity = pruning_cfg.get("target_sparsity", 0.0)

    is_neuron = level == "neuron"

    if is_neuron:
        # EAP neuron path
        mlp_hook = discovery_cfg.get("mlp_hook", "mlp_out")
        graph = Graph.from_model(model, node_scores=True, neuron_level=True, mlp_hook=mlp_hook)
        graph.neurons_scores = scores_data["neurons_scores"].to(device)
        if graph.neurons_scores.shape[1] > graph.neurons_in_graph.shape[1]:
            graph.neurons_scores = graph.neurons_scores[:, : graph.neurons_in_graph.shape[1]]

        total_to_keep_global = 0
        for node in graph.nodes.values():
            if isinstance(node, (AttentionNode, MLPNode)):
                out_of_scope = (scope == "heads" and isinstance(node, MLPNode)) or (
                    scope == "mlp" and isinstance(node, AttentionNode)
                )
                fwd_idx = graph.forward_index(node, attn_slice=False)
                if out_of_scope:
                    graph.neurons_scores[fwd_idx] = float("inf")
                    total_to_keep_global += node.d_neuron
                else:
                    num_keep_local = int(node.d_neuron * (1 - sparsity))
                    total_to_keep_global += num_keep_local
                    abs_scores = t.abs(graph.neurons_scores[fwd_idx, : node.d_neuron])
                    if num_keep_local < node.d_neuron and num_keep_local > 0:
                        # Keep exactly num_keep_local neurons by index. A
                        # threshold compare (abs_scores < kth-largest) over-keeps
                        # every neuron tied at the boundary, drifting the
                        # effective sparsity below the requested target; topk
                        # indices break ties deterministically.
                        keep_idx = t.topk(abs_scores, num_keep_local).indices
                        prune_mask = t.ones_like(abs_scores, dtype=t.bool)
                        prune_mask[keep_idx] = False
                        graph.neurons_scores[fwd_idx, : node.d_neuron][prune_mask] = -float("inf")
                    elif num_keep_local == 0:
                        graph.neurons_scores[fwd_idx, : node.d_neuron] = -float("inf")

        graph.apply_topn(total_to_keep_global, level="neuron", prune=True)
    else:
        # Node-level (ACDC, EAP)
        graph = Graph.from_model(
            model,
            node_scores=True,
            neuron_level=False,
            mlp_hook=discovery_cfg.get("mlp_hook", "mlp_out"),
        )
        _populate_graph_from_ib_scores(graph, scores_data["node_scores"])
        for node in graph.nodes.values():
            if isinstance(node, (AttentionNode, MLPNode)):
                fwd_idx = graph.forward_index(node, attn_slice=False)
                out_of_scope = (scope == "heads" and isinstance(node, MLPNode)) or (
                    scope == "mlp" and isinstance(node, AttentionNode)
                )
                if out_of_scope:
                    node.score = t.tensor(float("inf"))
                    graph.nodes_scores[fwd_idx] = float("inf")
        n_topn, n_to_keep = _compute_n_topn(graph, scope, sparsity)
        graph.apply_topn(n_topn, level="node", prune=True)

    return graph


@debug_function
@handle_errors(context={"operation": "evaluate_circuit"})
def evaluate_circuit(
    config: Union[str, Dict[str, Any]],
    pruned_artifact_path: str = None,
    scores_path: str = None,
    _model: Optional[HookedTransformer] = None,
) -> "FaithfulnessReport":
    """
    Evaluate circuit faithfulness using the 6-pillar framework.

    Thin wrapper around run_full_faithfulness(). Reconstructs the circuit
    graph from saved scores, loads the model, and runs comprehensive
    faithfulness evaluation via run_full_faithfulness().

    Args:
        config: Path to YAML config or config dict.
        pruned_artifact_path: Path to .pt pruning artifact (defaults to config['output_path']).
        scores_path: Path to _scores.pt file (auto-derived if not provided).
        _model: Optional pre-loaded HookedTransformer. Internal parameter used
            by discover_circuit() to avoid loading the model a second time
            when discovery_cfg["evaluate"]=True triggers an inline evaluation.
            When provided, this function still unconditionally (re-)asserts
            the config flags it needs (use_split_qkv_input, use_attn_result,
            use_hook_mlp_in, ungroup_grouped_query_attention) on it before
            use, since discover_circuit only sets these for EAP-family
            algorithms — algorithms like 'ibcircuit'/'cdt' may hand over a
            model that doesn't have them yet. Setting an already-true flag
            is a no-op, so this is safe either way. External callers should
            leave this as None; behavior is identical to before this
            parameter existed.

    Returns:
        FaithfulnessReport: Structured evaluation result. The two always-present
            fields are:
            - ``.patching_score``: Pillar 1 (causal patching) — original vs
              circuit performance under intervention.
            - ``.ablation_score``: Pillar 2 (ablation) — circuit sufficiency.
            The full-faithfulness path additionally populates ``.stability``,
            ``.robustness``, ``.baseline_comparison``, ``.generalization`` and
            ``.intervention_reliability``. A random-circuit baseline, when
            requested, is carried in ``.metadata["random_avg"]``.

            Prior to 1.0 the fast path returned a dict with the misleadingly
            named keys ``baseline_avg`` (= patching), ``circuit_avg``
            (= ablation) and ``random_avg``; that dict has been removed. Use the
            attributes above.
    """
    from pathlib import Path

    from .evaluation import run_full_faithfulness
    from .tasks.bootstrap import _bootstrap_builtin_tasks
    from .utils.config import load_and_validate_config

    _bootstrap_builtin_tasks()
    logger = get_logger("circuitkit.evaluate_circuit")
    progress = ProgressLogger(logger)

    try:
        progress.start_operation("Circuit Evaluation", 4)
        progress.step("Loading config and model")
        
        config = load_and_validate_config(config)
        discovery_cfg = config["discovery"]
        pruning_cfg = config["pruning"]

        # Resolve paths
        artifact_path = pruned_artifact_path or config.get("output_path")
        if not artifact_path:
            raise ValueError("Provide pruned_artifact_path or set config['output_path']")
        if scores_path is None:
            scores_path = str(
                Path(artifact_path).parent / (Path(artifact_path).stem + "_scores.pt")
            )

        validate_file_exists(artifact_path, "pruned artifact")
        validate_file_exists(scores_path, "discovery scores")

        # Load model and data
        device = get_device()
        dtype = getattr(t, config["model"].get("precision", "bfloat16"))
        if _model is not None:
            # Reuse the caller's already-loaded model (e.g. discover_circuit's
            # inline evaluate path) instead of loading a second full copy.
            model = _model
            logger.debug("evaluate_circuit: reusing pre-loaded model, skipping reload")
        else:
            with log_execution_time("Model loading", logger):
                model = HookedTransformer.from_pretrained(
                    config["model"]["name"], device=device, dtype=dtype
                )
        # These flags are required by the graph reconstruction / faithfulness
        # evaluation below regardless of model provenance. discover_circuit()
        # only sets them for the EAP-family algorithms (see its algo-dispatch
        # block); algorithms like 'ibcircuit'/'cdt' reach this function with
        # a model that may not have them set yet. Setting an already-true
        # flag is a no-op, so it's always safe to (re-)assert these here,
        # whether `model` was just loaded or reused via `_model`.
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        if hasattr(model.cfg, "ungroup_grouped_query_attention"):
            model.cfg.ungroup_grouped_query_attention = True

        scores_data = t.load(scores_path, map_location="cpu", weights_only=True)
        task_spec = _get_task(discovery_cfg["task"])

        # Build evaluation dataloader
        eval_cfg = config.get("eval", {})
        eval_num_examples = eval_cfg.get(
            "num_examples", discovery_cfg.get("data_params", {}).get("num_examples", 256)
        )
        eval_seed = eval_cfg.get(
            "seed",
            discovery_cfg.get(
                "seed",  # top-level seed (WMDP, MMLU style)
                discovery_cfg.get("data_params", {}).get("seed", 42),  # nested seed (IOI style)
            ),
        )
        dl_cfg = {
            **discovery_cfg,
            "algorithm": "eap",
            "data_params": {
                **discovery_cfg.get("data_params", {}),
                "num_examples": eval_num_examples,
                "seed": eval_seed,
            },
            "batch_size": discovery_cfg.get("batch_size", 16),
        }

        algo = discovery_cfg["algorithm"].lower()
        level = discovery_cfg.get("level", "node")

        # Detect clean-only IBCircuit neuron-level (custom data with no corrupt
        # prompts). The EAP dataloader path would crash because it requires
        # fully-paired data. Instead, build a self-paired EAP-format loader and
        # switch to the correct-token probability metric (bounded [0, 1]) which
        # doesn't need an incorrect token.
        _is_clean_only_ib = (
            algo == "ibcircuit"
            and level == "neuron"
            and hasattr(task_spec, "ds")
            and not getattr(task_spec.ds, "fully_paired", True)
        )
        if _is_clean_only_ib:
            dataloader = _build_clean_only_ib_eval_dataloader(
                task_spec,
                model,
                num_examples=eval_num_examples,
                batch_size=int(discovery_cfg.get("batch_size", 8)),
            )
        else:
            dataloader = task_spec.build_dataloader(model, dl_cfg, device)
        # eval_cfg was already set above (line 1294); use_full_faithfulness_eval resolved here.
        # For clean-only IBCircuit neuron-level we always force the fast path: only
        # sufficiency (baseline avg vs circuit avg) is computable without paired data.
        use_full_faithfulness_eval = eval_cfg.get("full_faithfulness_eval", False)
        if _is_clean_only_ib:
            use_full_faithfulness_eval = False

        # ── Shared setup for ALL algorithms ───────────────────────────────────
        # Must live here — before the IBCircuit branch — so every path has
        # access to eval_intervention, intervention_dataloader, corruption
        # dataloaders, and target task.  EAP/EAP-IG behaviour is unchanged:
        # they fall through this block and hit _reconstruct_circuit_graph below.

        eval_intervention = pruning_cfg.get("intervention", "zero")
        discovery_cfg["eval_intervention"] = (
            eval_intervention  # read by run_full_faithfulness pillars 2/4/6
        )

        intervention_dataloader = None
        if eval_intervention in ("mean", "mean-positional"):
            if _is_clean_only_ib:
                intervention_dataloader = dataloader
            else:
                intervention_dataloader = task_spec.build_dataloader(model, dl_cfg, device)

        corruption_dataloaders = {}
        pillars_to_run = eval_cfg.get("pillars")
        if pillars_to_run is None or "robustness" in pillars_to_run:
            corruption_variants = eval_cfg.get("corruption_variants", ["paraphrase"])
            for variant in corruption_variants:
                var_cfg = dl_cfg.copy()
                var_cfg["data_params"] = var_cfg.get("data_params", {}).copy()
                var_cfg["data_params"]["corruption_variant"] = variant
                try:
                    corruption_dataloaders[variant] = task_spec.build_dataloader(
                        model, var_cfg, device
                    )
                except Exception as e:
                    logger.warning(f"Could not build corruption dataloader for '{variant}': {e}")

        if not corruption_dataloaders and (
            pillars_to_run is None or "robustness" in pillars_to_run
        ):
            logger.error(
                f"Robustness pillar requested but no corruption dataloaders could be built "
                f"for variants {corruption_variants}. Skipping robustness evaluation."
            )
            # Remove 'robustness' from pillars to prevent meaningless zero-delta results
            if pillars_to_run:
                pillars_to_run = [p for p in pillars_to_run if p != "robustness"]
        target_task_name = eval_cfg.get("target_task", None)
        target_task_spec = None
        target_dataloader = None
        if target_task_name is not None:
            target_task_spec = _get_task(target_task_name)
            target_configs = eval_cfg.get("target_configs")
            if target_configs is not None:
                target_dl_cfg = {**dl_cfg, "configs": target_configs}
            else:
                target_dl_cfg = dl_cfg
            target_dataloader = target_task_spec.build_dataloader(model, target_dl_cfg, device)
            logger.info(
                f"Target task for generalization: {target_task_name}"
                + (f" (configs: {target_configs})" if target_configs else "")
            )

        # For clean-only IBCircuit neuron-level, substitute a metric that
        # doesn't need an incorrect token. _correct_token_prob is bounded [0, 1]
        # and uses only labels[:, 0] (the correct-answer token).
        if _is_clean_only_ib:
            from functools import partial as _partial

            metric = _partial(_correct_token_prob, loss=False, mean=False)
        else:
            metric = _make_eval_metric(task_spec)

        # ── IBCircuit neuron-level: special graph construction and eval ────────
        # Cannot use _reconstruct_circuit_graph because IBCircuit neuron scores
        # are stored as Dict[str, Tensor] in _scores.pt, incompatible with the
        # 2-D Tensor that the neuron branch of _reconstruct_circuit_graph expects.
        # Instead the graph is built directly from the pruning dict artifact.
        if level == "neuron" and algo == "ibcircuit":
            from .evaluation.evaluate import evaluate_baseline, evaluate_ibcircuit_neuron_circuit

            scope = discovery_cfg.get("scope", "heads")
            seed = eval_cfg.get("seed", discovery_cfg.get("data_params", {}).get("seed", 42))

            pruning_dict = t.load(artifact_path, map_location=device, weights_only=True)

            _log_gpu_mem("api.evaluate_circuit: after model+data load, before P1/P2", logger)

            # IBCircuit training uses mean-ablation; honour pruning_cfg but default to 'mean'.
            # For clean-only data, patching is unavailable (no corrupt side); keep mean/zero.
            _ib_intervention = pruning_cfg.get("intervention", "mean")
            ib_eval_intervention = (
                _ib_intervention if _ib_intervention in ("zero", "patching") else "mean"
            )
            if _is_clean_only_ib and ib_eval_intervention == "patching":
                ib_eval_intervention = "mean"  # patching needs a real corrupt side

            progress.step("Running IBCircuit neuron evaluation")

            # Pillars 1 & 2 — computed via IBCircuit-specific evaluators.
            # evaluate_graph (used inside run_full_faithfulness for P1/P2) ablates
            # via activation-difference hooks which are incompatible with the
            # per-neuron hook mechanism of evaluate_ibcircuit_neuron_circuit.
            baseline_avg = _avg_scores(evaluate_baseline(model, dataloader, metric))
            circuit_avg = _avg_scores(
                evaluate_ibcircuit_neuron_circuit(
                    model,
                    pruning_dict,
                    dataloader,
                    metric,
                    intervention=ib_eval_intervention,
                )
            )

            _log_gpu_mem("api.evaluate_circuit: after IBCircuit P1/P2 eval", logger)

            random_avg = None
            if pruning_cfg.get("random", False):
                rand_pruning_dict = _build_random_ibcircuit_neuron_pruning_dict(
                    model,
                    pruning_dict,
                    scope=scope,
                    seed=seed,
                )
                random_avg = _avg_scores(
                    evaluate_ibcircuit_neuron_circuit(
                        model,
                        rand_pruning_dict,
                        dataloader,
                        metric,
                        intervention=ib_eval_intervention,
                    )
                )

            if not use_full_faithfulness_eval:
                # Fast path: P1/P2 only, returned as a FaithfulnessReport. The
                # random-circuit baseline (when computed) is carried in metadata.
                # For clean-only IBCircuit: patching_score = full-model correct-token
                # probability (baseline), ablation_score = circuit sufficiency score.
                from .evaluation.report import FaithfulnessReport

                report = FaithfulnessReport(
                    patching_score=baseline_avg,
                    ablation_score=circuit_avg,
                )
                report.metadata = {"random_avg": random_avg} if random_avg is not None else {}
                if _is_clean_only_ib:
                    report.metadata["eval_mode"] = "clean_only_sufficiency"
                    logger.info(
                        f"Clean-only sufficiency: full-model P(correct)={_fmt_opt_score(baseline_avg)} "
                        f"| circuit P(correct)={_fmt_opt_score(circuit_avg)}"
                    )
                else:
                    logger.info(f"Original: {_fmt_opt_score(baseline_avg)} | Circuit: {_fmt_opt_score(circuit_avg)}")
                progress.complete(
                    **{
                        k: round(v, 4)
                        for k, v in {"patching_score": baseline_avg, "ablation_score": circuit_avg}.items()
                        if v is not None
                    }
                )
                return report

            # Full faithfulness path — build a proper neuron-level Graph from the
            # pruning dict so that graph-based pillars (baselines, robustness,
            # stability, generalization) receive a correctly populated graph.
            # neurons_in_graph defaults to all-ones (all in circuit); we zero
            # out the pruned neurons to match the IBCircuit discovery result.
            mlp_hook = discovery_cfg.get("mlp_hook", "mlp_out")
            graph = Graph.from_model(model, node_scores=True, neuron_level=True, mlp_hook=mlp_hook)

            _log_gpu_mem("api.evaluate_circuit: after neuron-level Graph construction", logger)

            ib_mlp_neurons = pruning_dict.get("mlp", {})  # {layer: [neuron_idx, ...]}
            ib_attn_neurons = pruning_dict.get("heads", {})  # {(layer, head): [neuron_idx, ...]}

            for node in graph.nodes.values():
                if isinstance(node, MLPNode):
                    pruned = ib_mlp_neurons.get(node.layer, [])
                    if pruned:
                        fwd_idx = graph.forward_index(node, attn_slice=False)
                        graph.neurons_in_graph[fwd_idx, pruned] = 0
                elif isinstance(node, AttentionNode):
                    pruned = ib_attn_neurons.get((node.layer, node.head), [])
                    if pruned:
                        fwd_idx = graph.forward_index(node, attn_slice=False)
                        graph.neurons_in_graph[fwd_idx, pruned] = 0

            # Run remaining pillars (baselines, robustness, stability,
            # generalization) via run_full_faithfulness.  patching and ablation
            # are intentionally excluded — they were computed above with the
            # IBCircuit-correct evaluators.
            requested_pillars = eval_cfg.get("pillars") or [
                "patching",
                "ablation",
                "baselines",
                "robustness",
                "stability",
                "generalization",
            ]
            graph_pillars = [p for p in requested_pillars if p not in ("patching", "ablation")]

            import gc

            gc.collect()
            if t.cuda.is_available():
                empty_cache()

            extra_report = None

            _log_gpu_mem("api.evaluate_circuit: before run_full_faithfulness", logger)

            if graph_pillars:
                extra_report = run_full_faithfulness(
                    model=model,
                    graph=graph,
                    task_spec=task_spec,
                    discovery_cfg=discovery_cfg,
                    pruning_cfg=pruning_cfg,
                    device=device,
                    pillars=graph_pillars,
                    n_stability_runs=eval_cfg.get("n_stability_runs", 5),
                    metric_fn=metric,
                    dataloader=dataloader,
                    intervention_dataloader=intervention_dataloader,
                    corruption_dataloaders=corruption_dataloaders,
                    target_task_spec=target_task_spec,
                    target_dataloader=target_dataloader,
                )

            # Assemble FaithfulnessReport: P1/P2 from IBCircuit evaluators,
            # remaining pillars from extra_report (if computed).
            from .evaluation.report import FaithfulnessReport

            report = FaithfulnessReport(
                patching_score=baseline_avg,
                ablation_score=circuit_avg,
            )
            if extra_report is not None:
                report.baseline_comparison = getattr(extra_report, "baseline_comparison", None)
                report.robustness = getattr(extra_report, "robustness", None)
                report.stability = getattr(extra_report, "stability", None)
                report.generalization = getattr(extra_report, "generalization", None)
                # Carry over metadata set by run_full_faithfulness; patch in our values
                report.metadata = getattr(extra_report, "metadata", {})
            else:
                report.metadata = {}

            report.metadata.update(
                {
                    "algorithm": algo,
                    "model": config["model"]["name"],
                    "task": discovery_cfg.get("task", "unknown"),
                    "level": level,
                    "scope": scope,
                    "sparsity": pruning_cfg.get("target_sparsity", 0.0),
                    "pillars_computed": requested_pillars,
                    "random_avg": random_avg,
                }
            )

            logger.info(f"Original: {_fmt_opt_score(baseline_avg)} | Circuit: {_fmt_opt_score(circuit_avg)}")
            logger.info("Full faithfulness report complete (IBCircuit neuron)")
            progress.complete()
            return report

        # ── All other algorithms (EAP, EAP-IG, ACDC, IBCircuit node-level) ────
        # Reconstruct graph from scores and run faithfulness evaluation.
        # This path is identical to the original code — no changes.
        progress.step("Reconstructing circuit")
        graph = _reconstruct_circuit_graph(model, scores_data, discovery_cfg, pruning_cfg, device)

        progress.step("Running faithfulness evaluation")

        if use_full_faithfulness_eval:
            report = run_full_faithfulness(
                model=model,
                graph=graph,
                task_spec=task_spec,
                discovery_cfg=discovery_cfg,
                pruning_cfg=pruning_cfg,
                device=device,
                pillars=eval_cfg.get("pillars", None),
                n_stability_runs=eval_cfg.get("n_stability_runs", 5),
                metric_fn=metric,
                dataloader=dataloader,
                intervention_dataloader=intervention_dataloader,
                corruption_dataloaders=corruption_dataloaders,
                baseline_types=eval_cfg.get("baseline_types", None),
                target_task_spec=target_task_spec,
                target_dataloader=target_dataloader,
            )
            logger.info("Full faithfulness report complete")
            progress.complete()
            return report
        else:
            report = run_full_faithfulness(
                model=model,
                graph=graph,
                task_spec=task_spec,
                discovery_cfg=discovery_cfg,
                pruning_cfg=pruning_cfg,
                device=device,
                pillars=["patching", "ablation"],
                metric_fn=metric,
                dataloader=dataloader,
                intervention_dataloader=intervention_dataloader,
                target_task_spec=target_task_spec,
                target_dataloader=target_dataloader,
            )
            logger.info(
                f"Original: {_fmt_opt_score(report.patching_score)} | "
                f"Circuit: {_fmt_opt_score(report.ablation_score)}"
            )
            progress.complete(
                **{
                    k: round(v, 4)
                    for k, v in {
                        "patching_score": report.patching_score,
                        "ablation_score": report.ablation_score,
                    }.items()
                    if v is not None
                }
            )
            return report

    except Exception as e:
        progress.fail(str(e))
        raise


@debug_function
@handle_errors(context={"operation": "benchmark_circuit"})
def benchmark_circuit(
    model_name: str,
    pruned_artifact_path: str,
    eval_params: Dict[str, Any],
    config_for_report: Dict[str, Any],
    report_path: str = None,
    precision: str = "bfloat16",
    use_weight_based_pruning: bool = False,
    evaluation_mode: str = "both",
    save_to_txt: bool = False,
):
    """
    Evaluate a pruned circuit on lm-eval benchmarks.

    Loads the model and pruning artifact, then runs the lm-evaluation-harness
    on the original and/or pruned model depending on evaluation_mode. Pruning
    is applied either via forward hooks (default) or by directly zeroing weights
    (use_weight_based_pruning=True, which avoids the use_attn_result overhead).

    Args:
        model_name (str): HuggingFace model identifier (e.g. 'gpt2').
        pruned_artifact_path (str): Path to the .pt pruning artifact produced
            by discover_circuit — either a List[str] of node names (node-level)
            or a Dict with 'mlp'/'heads'/'_meta' keys (neuron-level).
        eval_params (Dict[str, Any]): Evaluation configuration. Recognised
            sub-key 'lm_eval' supports:
                enabled (bool): Skip lm-eval entirely if False. Default True.
                tasks (List[str]): lm-eval task names. Default: gsm8k, mmlu,
                    truthfulqa, humaneval, hellaswag.
                fewshot (int): Number of few-shot examples. Default 0.
                limit (Optional[int]): Cap examples per task. Default None.
                max_gen_toks (int): Max generation tokens. Default 64.
                confirm_run_unsafe_code (bool): Required for code tasks. Default False.
        config_for_report (Dict[str, Any]): Original discovery config, currently
            used for logging context only.
        report_path (Optional[str]): If save_to_txt=True, write results to this
            path instead of an auto-generated timestamped file.
        precision (str): Torch dtype string for model loading
            ('bfloat16', 'float16', 'float32'). Defaults to 'bfloat16'.
        use_weight_based_pruning (bool): If True, prune by zeroing weights directly
            (more efficient, no hook overhead, does not require use_attn_result).
            If False, prune via forward hooks. Defaults to False.
        evaluation_mode (str): Which model variants to evaluate.
            'both' runs original then pruned; 'original' skips pruned;
            'pruned' skips original. Defaults to 'both'.
        save_to_txt (bool): If True, write results to a text file via
            _save_evaluation_results_to_txt. Defaults to False.

    Returns:
        None: Results are printed to stdout and optionally written to a file.

    Raises:
        ValueError: If evaluation_mode is not one of 'both', 'original', 'pruned'.
        TypeError: If the pruning artifact type is not a list or dict.
        FileNotFoundError: If pruned_artifact_path does not exist.
    """
    logger = get_logger("circuitkit.evaluation")
    progress = ProgressLogger(logger)

    try:
        progress.start_operation("Circuit Evaluation", 3)

        # Validate inputs
        validate_model_name(model_name)
        validate_file_exists(pruned_artifact_path, "pruned artifact")

        # Validate evaluation_mode
        valid_modes = ["both", "original", "pruned"]
        if evaluation_mode not in valid_modes:
            raise ValueError(
                f"evaluation_mode must be one of {valid_modes}, got '{evaluation_mode}'"
            )

        progress.step("Loading model and pruning artifacts", model=model_name)
        device = get_device()
        if not isinstance(getattr(t, precision, None), t.dtype):
            raise ValueError(
                f"Invalid precision '{precision}'. Pass 'precision' as a torch "
                f"dtype name such as 'float32', 'float16', or 'bfloat16'."
            )
        dtype = getattr(t, precision)

        with log_execution_time("Model loading", logger):
            model = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)

        # Configure model for proper hook support (only needed for hook-based pruning)
        if not use_weight_based_pruning:
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True

        with log_execution_time("Artifact loading", logger):
            pruned_artifact = t.load(pruned_artifact_path, map_location="cpu", weights_only=True)

        progress.step("Running evaluation")

        if isinstance(pruned_artifact, list):
            logger.info(f"Detected node-level pruning artifact with {len(pruned_artifact)} nodes")
        elif isinstance(pruned_artifact, dict):
            mlp_count = sum(len(neurons) for neurons in pruned_artifact.get("mlp", {}).values())
            attn_count = sum(len(neurons) for neurons in pruned_artifact.get("heads", {}).values())
            logger.info(
                f"Detected neuron-level pruning artifact: {mlp_count} MLP neurons, {attn_count} attention neurons"
            )
        else:
            raise TypeError(f"Unknown artifact type for pruning: {type(pruned_artifact)}")

        if report_path:
            logger.info(f"Report path specified: {report_path}")
            logger.warning("Report generation not yet implemented - results printed to console")

        # Initialize results collection for txt file saving
        evaluation_results = []

        # Run lm-evaluation-harness benchmarks
        lm_eval_cfg = eval_params.get("lm_eval", {}) if isinstance(eval_params, dict) else {}
        if lm_eval_cfg.get("enabled", True):
            tasks = lm_eval_cfg.get(
                "tasks",
                [
                    "gsm8k",
                    "mmlu",
                    "truthfulqa",
                    "humaneval",
                    "hellaswag",
                ],
            )
            fewshot = int(lm_eval_cfg.get("fewshot", 0))
            limit = lm_eval_cfg.get("limit", None)
            int(lm_eval_cfg.get("max_gen_toks", 64))
            confirm_unsafe = bool(lm_eval_cfg.get("confirm_run_unsafe_code", False))

            try:
                logger.info(f"Running lm-eval on tasks: {tasks}")

                if use_weight_based_pruning:
                    # Use weight-based pruning
                    from .evaluation.weight_based_eval import (
                        compare_original_vs_pruned_weight_based,
                        evaluate_lm_eval_weight_based,
                    )

                    if evaluation_mode == "both":
                        results = compare_original_vs_pruned_weight_based(
                            model,
                            pruned_artifact,
                            tasks=tasks,
                            fewshot=fewshot,
                            limit=limit,
                            confirm_run_unsafe_code=confirm_unsafe,
                            verbosity="WARNING",
                        )

                        original_results = results["original"].get("results", results["original"])
                        pruned_results = results["pruned"].get("results", results["pruned"])

                        logger.info("Original model results: %s", original_results)
                        logger.info("Weight-pruned model results: %s", pruned_results)

                        # Collect results for txt file
                        if save_to_txt:
                            evaluation_results.append(
                                {"model_type": "original", "results": original_results}
                            )
                            evaluation_results.append(
                                {"model_type": "pruned", "results": pruned_results}
                            )
                    elif evaluation_mode == "original":
                        # Only evaluate original model (empty artifact = no pruning)
                        results = evaluate_lm_eval_weight_based(
                            model,
                            tasks=tasks,
                            pruned_artifact=[],
                            fewshot=fewshot,
                            limit=limit,
                            confirm_run_unsafe_code=confirm_unsafe,
                            verbosity="WARNING",
                        )
                        original_results = results.get("results", results)
                        logger.info("Original model results: %s", original_results)

                        # Collect results for txt file
                        if save_to_txt:
                            evaluation_results.append(
                                {"model_type": "original", "results": original_results}
                            )
                    elif evaluation_mode == "pruned":
                        # Only evaluate pruned model
                        results = evaluate_lm_eval_weight_based(
                            model,
                            tasks=tasks,
                            pruned_artifact=pruned_artifact,
                            fewshot=fewshot,
                            limit=limit,
                            confirm_run_unsafe_code=confirm_unsafe,
                            verbosity="WARNING",
                        )
                        pruned_results = results.get("results", results)
                        logger.info("Weight-pruned model results: %s", pruned_results)

                        # Collect results for txt file
                        if save_to_txt:
                            evaluation_results.append(
                                {"model_type": "pruned", "results": pruned_results}
                            )

                else:
                    # Use hook-based pruning
                    from .evaluation.lm_eval_simple import evaluate_lm_eval

                    if evaluation_mode in ["both", "original"]:
                        # Original model
                        orig = evaluate_lm_eval(
                            model,
                            tasks=tasks,
                            fewshot=fewshot,
                            limit=limit,
                            confirm_run_unsafe_code=confirm_unsafe,
                            verbosity="WARNING",
                        )
                        original_results = orig.get("results", orig)
                        logger.info("Original model results: %s", original_results)

                        # Collect results for txt file
                        if save_to_txt:
                            evaluation_results.append(
                                {"model_type": "original", "results": original_results}
                            )

                    if evaluation_mode in ["both", "pruned"]:
                        # Pruned model view via hooks
                        pruned = evaluate_lm_eval(
                            model,
                            tasks=tasks,
                            pruned_artifact=pruned_artifact,
                            fewshot=fewshot,
                            limit=limit,
                            confirm_run_unsafe_code=confirm_unsafe,
                            verbosity="WARNING",
                        )
                        pruned_results = pruned.get("results", pruned)
                        logger.info("Pruned model results: %s", pruned_results)

                        # Collect results for txt file
                        if save_to_txt:
                            evaluation_results.append(
                                {"model_type": "pruned", "results": pruned_results}
                            )

            except Exception as lm_err:
                logger.warning(f"lm-eval run skipped/failed: {lm_err}")

        # Save results to txt file if requested
        if save_to_txt and evaluation_results:
            _save_evaluation_results_to_txt(
                evaluation_results,
                model_name,
                pruned_artifact_path,
                evaluation_mode,
                logger,
                report_path if report_path else None,
            )

        progress.complete()

    except Exception as e:
        progress.fail(str(e))
        raise


def load_circuit(circuit_path: str) -> Union[List[str], Dict]:
    """
    Load a saved circuit pruning artifact from disk as its **raw** form.

    This returns the low-level pruning artifact exactly as ``torch.save`` wrote
    it — a plain ``list[str]`` (node-level) or ``dict`` (neuron-level). It does
    **not** return a :class:`~circuitkit.Circuit` object and carries no scores
    or metadata. If you want a ready-to-use ``Circuit`` (with ``.scores``,
    ``.top_nodes()``, ``.task``, ...), use :func:`circuitkit.load_scores`
    instead — that is the loader most callers want. The two are not
    interchangeable: this one feeds ``prune``/``export``; ``load_scores`` feeds
    ``selective_finetune``/``Pipeline.from_scores``.

    Args:
        circuit_path (str): Path to a .pt file produced by discover_circuit.

    Returns:
        Union[List[str], Dict]: List of node name strings for node-level
            circuits, or a dict with keys 'mlp', 'heads', '_meta' for
            neuron-level circuits.

    Raises:
        FileNotFoundError: If circuit_path does not exist.

    See Also:
        circuitkit.load_scores: Load the same artifact as a rich ``Circuit``.
    """
    validate_file_exists(circuit_path, "circuit file")
    return t.load(circuit_path, map_location="cpu", weights_only=True)
