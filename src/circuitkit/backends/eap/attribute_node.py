from functools import partial
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import torch
from einops import einsum
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from ...utils.logging import get_logger
logger = get_logger("eap.attribute_node")

from .eap_utils import compute_mean_activations, tokenize_batch_pair
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph

def make_hooks_and_matrices(
    model: HookedTransformer,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores: Optional[Tensor],
    neuron: bool = False,
):
    """
    Build the activation-difference buffer and the forward/backward hooks needed for node-level attribution.

    The activation difference tensor accumulates (corrupted − clean) activations
    for each forward node. Forward-corrupted hooks add activations into it;
    forward-clean hooks subtract them out. Backward hooks read the buffer and
    the incoming gradient to update the score tensor in-place.

    Args:
        model (HookedTransformer): The model being attributed.
        graph (Graph): The computation graph defining which nodes to hook.
        batch_size (int): Number of examples in the current batch.
        n_pos (int): Sequence length (position dimension) of the current batch.
        scores (Optional[Tensor]): Score tensor to accumulate gradients into.
            Shape [n_forward, max_d] (neuron-level) or [n_forward] (node-level).
            Backward hooks are still created regardless; pass None only when the
            caller will discard them.
        neuron (bool): If True, accumulate per-neuron scores of shape
            [n_forward, max_d]; otherwise accumulate scalar node scores of
            shape [n_forward]. Defaults to False.

    Returns:
        Tuple[Tuple[List, List, List], Tensor]:
            - (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks): Hook lists to
              be passed to model.hooks(). Run fwd_hooks_corrupted on corrupted
              input, fwd_hooks_clean on clean input, bwd_hooks during the
              backward pass on clean input.
            - activation_difference: Buffer of shape
              [batch, pos, n_forward, max_d] storing accumulated
              (corrupted − clean) activations.
    """
    if neuron and scores is not None:
        # LOGICAL REQUIREMENT 1: Single Source of Truth
        # The upstream caller already calculated the correct max_d to allocate `scores`.
        # We must perfectly align with it to prevent einsum shape mismatches.
        max_d = scores.shape[-1]
    else:
        # LOGICAL REQUIREMENT 2: Safe Buffer Sizing for Scalars
        # If node-level (scores is 1D), we must ensure the buffer is wide enough
        # for intermediate MLP activations if they exist in the graph.
        d_model = model.cfg.d_model
        # Safely attempt to find d_mlp on either the graph or the model
        d_mlp = graph.cfg.get("d_mlp", getattr(model.cfg, "d_mlp", d_model))

        if graph.cfg.get("mlp_hook") == "post_act":
            max_d = max(d_model, d_mlp)
        else:
            max_d = d_model

    activation_difference = torch.zeros(
        (batch_size, n_pos, graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype
    )

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    # Fills up the activation difference matrix. In the default case (not separate_activations),
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient,
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations: torch.Tensor, hook: HookPoint, add: bool = True):

        acts = activations.detach()
        act_d = acts.shape[-1]
        try:
            if add:
                activation_difference[:, :, index, :act_d] += acts
            else:
                activation_difference[:, :, index, :act_d] -= acts

        except RuntimeError as e:
            logger.error(f"[DEBUG ERROR] Activation Hook Failed at {hook.name}")
            logger.error(f"  Target buffer shape: {activation_difference[:, :, index].shape}")
            logger.error(f"  Incoming acts shape: {acts.shape}")
            raise e

    def gradient_hook(
        fwd_index: Union[slice, int],
        bwd_index: Union[slice, int],
        gradients: torch.Tensor,
        hook: HookPoint,
    ):
        """Takes in a gradient and uses it and activation_difference
        to compute an update to the score matrix

        Args:
            fwd_index (Union[slice, int]): The forward index of the (src) node
            bwd_index (Union[slice, int]): The backward index of the (dst) node
            gradients (torch.Tensor): The gradients of this backward pass
            hook (_type_): (unused)

        """

        grads = gradients.detach()
        grad_d = grads.shape[-1]
        try:
            if neuron:
                s = einsum(
                    activation_difference[:, :, fwd_index, :grad_d],
                    grads,
                    "batch pos ... hidden, batch pos ... hidden -> ... hidden",
                )
                scores[fwd_index, :grad_d] += s
            else:
                s = einsum(
                    activation_difference[:, :, fwd_index, :grad_d],
                    grads,
                    "batch pos ... hidden, batch pos ... hidden -> ...",
                )
                scores[fwd_index] += s
        except RuntimeError as e:
            logger.error(
                hook.name,
                activation_difference.size(),
                activation_difference.device,
                grads.size(),
                grads.device,
            )
            logger.error(fwd_index, bwd_index, scores.size())
            raise e

    node = graph.nodes["input"]
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
    bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

    for layer in range(graph.cfg["n_layers"]):
        node = graph.nodes[f"a{layer}.h0"]
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))

        node = graph.nodes[f"m{layer}"]
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, fwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference

def get_scores_exact(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):
    """
    Compute node attribution scores via exact leave-one-out patching.

    For each node, temporarily disables all its outgoing edges, measures the
    performance drop relative to the full-graph baseline, and assigns that
    drop as the node's score.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph whose nodes are scored. All real edges and
            nodes are added to the graph before iteration begins.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Ablation type used when a node is removed. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Node score vector [n_forward]. Scores are also written
            in-place to each node's `.score` attribute via graph.nodes_scores.
    """
    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    graph.nodes_in_graph[:] = True
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    # LogitNode is the output and has a read-only score; skip it.
    candidate_nodes = [
        n
        for n in graph.nodes.values()
        if getattr(type(n), "score", None) is not None
        and isinstance(getattr(type(n), "score", None), property)
        and getattr(type(n), "score").fset is not None
    ]
    # evaluate_graph() calls graph.prune() internally, which only removes
    # edges/nodes — it never re-adds them. Across leave-one-out iterations
    # this corrupts the in_graph mask and every subsequent ablation reads
    # the same degenerate state. Snapshot and restore the full mask around
    # each evaluation so each node is scored against the same fully-connected baseline.
    full_in_graph = graph.in_graph.clone()
    full_nodes_in_graph = graph.nodes_in_graph.clone()
    nodes_iter = candidate_nodes if quiet else tqdm(candidate_nodes)
    for node in nodes_iter:
        # Restore to the full graph, then drop only this node's outgoing edges.
        graph.in_graph[:] = full_in_graph
        graph.nodes_in_graph[:] = full_nodes_in_graph
        for edge in node.child_edges:
            edge.in_graph = False
        intervened_performance = (
            evaluate_graph(
                model,
                graph,
                dataloader,
                metric,
                intervention=intervention,
                intervention_dataloader=intervention_dataloader,
                quiet=True,
                skip_clean=True,
            )
            .mean()
            .item()
        )
        node.score = intervened_performance - baseline
    # Leave the graph in the original fully-connected state so callers
    # downstream (CircuitScores conversion, pruning) see a consistent view.
    graph.in_graph[:] = full_in_graph
    graph.nodes_in_graph[:] = full_nodes_in_graph

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.nodes_scores

def get_scores_ifr(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention="patching",
    intervention_dataloader=None,
    quiet: bool = False,
    neuron: bool = False,
):
    """Information Flow Routes (Ferrando et al. 2024) at node level.

    The edge-level IFR scores edges by L1 proximity between each node's
    output and its upstream predecessors. We delegate to the edge-level
    ``backends.eap.attribute.get_scores_information_flow_routes`` and
    then aggregate per source node by summing over outgoing edges to
    produce the node-level score.

    Does not require a metric or corrupted inputs (clean forward only).
    The metric / intervention args are kept for dispatcher signature
    parity but are unused.
    """
    from .attribute import get_scores_information_flow_routes

    edge_scores = get_scores_information_flow_routes(
        model,
        graph,
        dataloader,
        quiet=quiet,
    )  # [n_forward, n_backward]
    # Per source node: sum (abs) over outgoing edges.
    node_scores = edge_scores.abs().sum(dim=-1)  # [n_forward]
    if neuron:
        # IFR is intrinsically a node/edge method; broadcast per-node
        # to per-neuron uniformly.
        d_model = model.cfg.d_model
        return node_scores.unsqueeze(-1).expand(-1, d_model).contiguous()
    return node_scores

def get_scores_peap(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    span_schema: Optional[Dict[str, slice]] = None,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Position-aware Edge Attribution Patching (PEAP), Haklay et al. 2025
    ACL (arxiv:2502.04577).

    Computes EAP attribution while RETAINING the position dimension.
    For each forward node, the score at position p is

        s_n[p] = E[ (a_clean[n] - a_corr[n])[p] · grad(L)[n][p] ]

    summed over the hidden dimension, averaged over batch. Per-node
    summary score (for downstream pruning) is the L2 norm over
    positions. The per-(node, position) tensor is stashed on
    ``graph._peap_per_pos_scores`` (shape ``[n_forward, max_pos]``).

    If ``span_schema`` is provided as a dict mapping span name to
    position-slice (or position-mask tensor), per-span aggregated
    scores are stored on ``graph._peap_span_scores``.

    Non-crossing edges (within-position node-to-node) are fully
    captured. Crossing edges (attention head -> attention head between
    different positions, requiring Q/K/V-specific attention-pattern
    gradients) are documented in backends/eap/PEAP_INTEGRATION.md as
    the next step — that specialisation needs the joint attention
    pattern x value gradient product which the node-level backbone
    aggregates over the head dimension.
    """
    graph.cfg["n_layers"]
    graph.cfg["n_heads"]
    d_model = graph.cfg["d_model"]

    # Per-(forward index, position) score accumulator. We allocate
    # lazily once we see the first batch's max_pos.
    per_pos: Optional[torch.Tensor] = None
    # Per-node summary (L2 norm over positions, averaged across batches).
    if neuron:
        max_d = (
            max(d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else d_model
        )
        scores_node = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores_node = torch.zeros((graph.n_forward,), device=model.cfg.device, dtype=model.cfg.dtype)

    if "mean" in intervention:
        if intervention_dataloader is None:
            raise ValueError(
                f"intervention={intervention!r} requires an "
                "'intervention_dataloader' to compute mean activations, but "
                "none was provided. Pass intervention_dataloader=<DataLoader>, "
                "or use intervention='patching' or 'zero' which need no "
                "extra data."
            )
        per_position = "positional" in intervention
        means = compute_mean_activations(
            model,
            graph,
            intervention_dataloader,
            per_position=per_position,
            padding_side=getattr(intervention_dataloader, "pair_padding_side", None),
        )
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader_iter = dataloader if quiet else tqdm(dataloader, desc="PEAP batches")
    total_items = 0

    for clean, corrupted, label in dataloader_iter:
        batch_size = len(clean)
        total_items += batch_size

        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model,
            clean,
            corrupted,
            pair_padding_side=pair_padding_side,
            templated=templated,
        )

        if per_pos is None:
            per_pos = torch.zeros(
                (graph.n_forward, n_pos),
                device=model.cfg.device,
                dtype=model.cfg.dtype,
            )
        elif per_pos.shape[1] < n_pos:
            grow = torch.zeros(
                (graph.n_forward, n_pos - per_pos.shape[1]),
                device=model.cfg.device,
                dtype=model.cfg.dtype,
            )
            per_pos = torch.cat([per_pos, grow], dim=1)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(
                model,
                graph,
                batch_size,
                n_pos,
                scores_node,
                neuron=neuron,
            )
        )

        # Per-node, per-position grad capture. For attention nodes we hook
        # `attn.hook_result` (shape [batch, pos, n_heads, d_model]) and slice
        # the relevant head; for MLPs we hook `hook_mlp_in` (shape
        # [batch, pos, d_model]); for the input node we hook the embedding
        # output. We accumulate (act_diff[:, :, idx, :d_model] * grad).sum(d_model)
        # to get the per-position score per node.
        from .graph import AttentionNode, InputNode, MLPNode

        # Group attention nodes by layer so we register one hook per layer
        # (each captures all heads' grads at once).
        attn_layers: Dict[int, list] = {}
        mlp_layers: list = []
        input_idx: Optional[int] = None
        for name, node in graph.nodes.items():
            if isinstance(node, AttentionNode):
                attn_layers.setdefault(node.layer, []).append(
                    (node.head, graph.forward_index(node, attn_slice=False)),
                )
            elif isinstance(node, MLPNode):
                mlp_layers.append((node.layer, graph.forward_index(node, attn_slice=False)))
            elif isinstance(node, InputNode):
                input_idx = graph.forward_index(node, attn_slice=False)

        captured: Dict[int, torch.Tensor] = {}

        def make_attn_layer_hook(heads_at_layer):
            def h(grad, hook):
                # grad shape: [batch, pos, n_heads, d_model] for hook_result
                detached = grad.detach()
                for head, fwd_idx in heads_at_layer:
                    # Per-head grad slice: [batch, pos, d_model]
                    captured[fwd_idx] = detached[:, :, head, :].clone()
                return None

            return h

        def make_mlp_hook(fwd_idx):
            def h(grad, hook):
                # grad shape: [batch, pos, d_model] for hook_mlp_in / mlp_out
                captured[fwd_idx] = grad.detach().clone()
                return None

            return h

        extra_bwd = []
        for layer, heads in attn_layers.items():
            extra_bwd.append((f"blocks.{layer}.attn.hook_result", make_attn_layer_hook(heads)))
        for layer, idx in mlp_layers:
            extra_bwd.append((f"blocks.{layer}.hook_mlp_out", make_mlp_hook(idx)))
        if input_idx is not None:
            extra_bwd.append(("hook_embed", make_mlp_hook(input_idx)))

        with torch.inference_mode():
            if intervention == "patching":
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            elif "mean" in intervention:
                activation_difference += means
            clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks + extra_bwd):
            logits = model(clean_tokens, attention_mask=clean_attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

        # Accumulate per-position scores from captured grads.
        # Cast to float32 to avoid dtype issues on bfloat16/float16 models.
        for fwd_idx, grad in captured.items():
            grad_d = grad.shape[-1]
            act_diff = activation_difference[:, :, fwd_idx, :grad_d].float()
            grad_f = grad.float()
            per_node_pos = (act_diff * grad_f).sum(dim=-1).mean(dim=0)  # [pos]
            cur_pos = per_node_pos.shape[0]
            per_pos[fwd_idx, :cur_pos] += per_node_pos.to(per_pos.dtype)

        captured.clear()
        model.zero_grad(set_to_none=True)

    scores_node /= total_items
    if per_pos is not None:
        per_pos /= total_items
        graph._peap_per_pos_scores = per_pos
        graph._peap_span_schema = span_schema
        if span_schema is not None:
            graph._peap_span_scores = {
                name: per_pos[:, sl].abs().sum(dim=-1) for name, sl in span_schema.items()
            }
        # Per-node summary: L2 norm over positions; preserves the per-node
        # API while reflecting the position-aware score magnitude.
        if not neuron:
            scores_node = per_pos.float().norm(dim=-1).to(per_pos.dtype)

    # Crossing edges: attention head -> attention head between different
    # positions. Computed by capturing K/V gradients per layer and
    # combining with the activation difference at the upstream head.
    # For an edge from (l_u, h_u, p_u) to (l_d, h_d, p_d):
    #     score = E_x [ (act_diff[l_u, h_u, p_u, :] · grad_v[l_d, h_d, p_u, :])
    #                   * pattern[l_d, h_d, p_d, p_u] ]
    # The output is a sparse 5-D tensor; we provide a dense
    # [n_layers, n_heads, n_layers, n_heads] aggregated-over-positions
    # tensor stashed on the graph so callers can drill in.
    if not neuron and per_pos is not None:
        graph._peap_crossing_edges = _peap_crossing_edges(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
        )

    return scores_node

def _peap_crossing_edges(
    model: HookedTransformer,
    graph: Graph,
    dataloader,
    metric,
    intervention="patching",
    intervention_dataloader=None,
):
    """Compute PEAP crossing-edge scores with all 5 aggregation methods.

    Returns a dict keyed by ``(l_u, h_u, l_d, h_d)`` ->
        {"avg", "sum", "sum_abs_pos", "sum_abs_exp", "max_abs"}
    where each value is the aggregated score across (batch, positions)
    for that (upstream, downstream) head pair, restricted to the
    upper-triangular (l_d > l_u) since attention is causal.

    Aggregation definitions (Haklay 2025 ACL §4):
      avg          : mean over positions and batch
      sum          : sum over positions, mean over batch
      sum_abs_pos  : sum over batch and positions of abs(score)
      sum_abs_exp  : abs of expected (mean) score
      max_abs      : max over positions of abs(score)

    For each batch:
      1. Run forward (clean) with cache to capture pattern, V at every layer.
      2. Run a backward pass through the metric, capturing dL/dV at every
         (layer, head, position).
      3. For each pair (l_u, l_d) with l_u < l_d, score the edge as
         pattern[l_d, h_d] · (clean_V[l_u, h_u] - corrupted_V[l_u, h_u])
         · dL/dV[l_d, h_d]. The pattern term routes the upstream
         contribution to the downstream key positions.
    """
    n_layers = graph.cfg["n_layers"]
    n_heads = graph.cfg["n_heads"]
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    edge_scores: Dict[Tuple[int, int, int, int], Dict[str, float]] = {}

    n_batches = 0

    def _accumulate(key, batch_score):
        bs = batch_score.detach().to(torch.float32)
        if key not in edge_scores:
            edge_scores[key] = {
                "avg": 0.0,
                "sum": 0.0,
                "sum_abs_pos": 0.0,
                "sum_abs_exp": 0.0,
                "max_abs": 0.0,
            }
        d = edge_scores[key]
        d["avg"] += float(bs.mean().item())
        d["sum"] += float(bs.sum(dim=-1).mean().item())
        d["sum_abs_pos"] += float(bs.abs().sum().item())
        d["sum_abs_exp"] += float(bs.mean(dim=0).abs().sum().item())
        d["max_abs"] += float(bs.abs().max().item())

    for clean, corrupted, label in dataloader:
        from .eap_utils import tokenize_batch_pair as _tok

        clean_tokens, corrupted_tokens, attn_mask, _, input_lengths, n_pos = _tok(
            model,
            clean,
            corrupted,
            pair_padding_side=pair_padding_side,
            templated=templated,
        )

        # Cache clean V per layer + the attention pattern at each layer.
        with torch.inference_mode():
            _, clean_cache = model.run_with_cache(
                clean_tokens,
                attention_mask=attn_mask,
                names_filter=lambda n: n.endswith("hook_v") or n.endswith("hook_pattern"),
            )
            corrupted_logits, corrupted_cache = model.run_with_cache(
                corrupted_tokens,
                names_filter=lambda n: n.endswith("hook_v"),
            )

        # Capture dL/dV at every (l, h, p) via backward hook.
        v_grads: Dict[int, torch.Tensor] = {}

        def make_v_grad_hook(layer_idx):
            def h(grad, hook):
                # grad shape: [batch, pos, head, d_head]
                v_grads[layer_idx] = grad.detach().clone()
                return None

            return h

        bwd = [(f"blocks.{lyr}.attn.hook_v", make_v_grad_hook(lyr)) for lyr in range(n_layers)]
        with model.hooks(bwd_hooks=bwd):
            # Fresh forward on the clean run to produce gradients on hook_v.
            logits = model(clean_tokens, attention_mask=attn_mask)
            # Use corrupted logits as the reference for the difference metric.
            metric_value = metric(logits, corrupted_logits.detach(), input_lengths, label)
            metric_value.backward()
        model.zero_grad(set_to_none=True)

        # Score each crossing edge.
        for l_u in range(n_layers - 1):
            v_u_clean = clean_cache[f"blocks.{l_u}.attn.hook_v"]  # [b, p, h, d]
            v_u_corr = corrupted_cache[f"blocks.{l_u}.attn.hook_v"]
            v_diff = v_u_corr - v_u_clean  # [b, p, h_u, d]
            for l_d in range(l_u + 1, n_layers):
                if l_d not in v_grads:
                    continue
                pattern_d = clean_cache[f"blocks.{l_d}.attn.hook_pattern"]  # [b, h_d, q, k]
                grad_v_d = v_grads[l_d]  # [b, p, h_d, d]
                # transformer_lens upcasts q/k (and hence the softmax
                # `hook_pattern`) to float32 for low-precision models,
                # while hook_v and its gradients stay in the model dtype.
                # Cast pattern to the model dtype so the einsum below has
                # matching operands (no-op when the model is already fp32).
                if pattern_d.dtype != grad_v_d.dtype:
                    pattern_d = pattern_d.to(grad_v_d.dtype)
                # Vectorized over both head dims in one einsum, replacing
                # the n_heads × n_heads Python double-loop. For GPT-2 this
                # collapses 144 einsum calls per (l_u, l_d) into one.
                # Indices: a=batch, q=query_pos, k=key_pos, u=h_u, h=h_d, d=d_head.
                # score[a, u, h, q] = sum_k sum_d pattern[a,h,q,k] * v_diff[a,k,u,d] * grad_v_d[a,q,h,d]
                score = torch.einsum(
                    "ahqk,akud,aqhd->auhq",
                    pattern_d,
                    v_diff,
                    grad_v_d,
                )  # [b, h_u, h_d, q]
                for h_u in range(n_heads):
                    for h_d in range(n_heads):
                        bs = score[:, h_u, h_d, :]
                        key = (l_u, h_u, l_d, h_d)
                        _accumulate(key, bs)
        n_batches += 1

    # Normalise accumulated sums by number of batches.
    if n_batches > 1:
        for d in edge_scores.values():
            for k in d:
                d[k] /= n_batches
    return edge_scores

def _build_relp_hooks(model: HookedTransformer):
    """
    Build forward hooks that turn standard backprop into LRP-style relevance
    propagation, following the four detach tricks in Mohebbi et al. 2025
    (RelP, NeurIPS 2025, arxiv:2508.21258):

      LN-rule:        hook_scale.detach() in every LayerNorm/RMSNorm.
      AH-rule:        hook_pattern.detach() in every attention block.
      Identity-rule:  in non-gated MLPs (GPT-2, Pythia), replace the
                      activation's gradient with z/x so it behaves as
                      identity in the backward pass.
      Half-rule:      in gated MLPs (Llama, Gemma — act(pre) * pre_linear),
                      replace hook_post output with (post/2) + (post/2).detach()
                      so only half the gradient flows through the gating
                      product. Identity-rule and Half-rule are mutually
                      exclusive — the paper uses Half-rule on gated paths
                      and Identity-rule on activation-only paths.

    The choice between Identity-rule and Half-rule is made per-model based
    on `model.cfg.gated_mlp`. All four rules together approximate LRP-ε
    relevance propagation while keeping cost identical to plain EAP.
    """
    hooks = []
    n_layers = model.cfg.n_layers
    is_gated_mlp = bool(getattr(model.cfg, "gated_mlp", False))

    def detach_hook(acts, hook):
        return acts.detach()

    # AH-rule: detach attention probabilities so gradient flows through V only.
    for layer in range(n_layers):
        hooks.append((f"blocks.{layer}.attn.hook_pattern", detach_hook))

    # LN-rule: detach the layer-norm scale divisor everywhere it is exposed.
    for layer in range(n_layers):
        ln1_key = f"blocks.{layer}.ln1.hook_scale"
        ln2_key = f"blocks.{layer}.ln2.hook_scale"
        if ln1_key in model.hook_dict:
            hooks.append((ln1_key, detach_hook))
        if ln2_key in model.hook_dict:
            hooks.append((ln2_key, detach_hook))
    if "ln_final.hook_scale" in model.hook_dict:
        hooks.append(("ln_final.hook_scale", detach_hook))

    if is_gated_mlp:
        # Half-rule: gradient through the gated product is halved. Forward
        # returns `acts` exactly; backward propagates 0.5 because the second
        # half is detached. Applied at hook_post which sits AFTER
        # act(pre) * pre_linear in TransformerLens's GatedMLP forward.
        def half_hook(acts, hook):
            return (acts / 2.0) + (acts / 2.0).detach()

        for layer in range(n_layers):
            hooks.append((f"blocks.{layer}.mlp.hook_post", half_hook))
    else:
        # Identity-rule: replace the MLP activation's gradient with z/x so the
        # activation acts identity-like in backprop while the forward value is
        # preserved. We capture `pre` at hook_pre and substitute at hook_post.
        state = {}
        for layer in range(n_layers):
            key = f"layer{layer}"

            def make_pre_hook(key=key):
                def pre_hook(acts, hook):
                    state[key] = acts
                    return acts

                return pre_hook

            def make_post_hook(key=key):
                def post_hook(acts, hook):
                    pre = state.get(key, None)
                    if pre is None:
                        return acts
                    eps = 1e-9
                    zp = pre + eps * (pre.abs() < eps).to(pre.dtype) * torch.where(
                        pre == 0, torch.ones_like(pre), pre.sign()
                    )
                    return zp * (acts / zp).detach()

                return post_hook

            hooks.append((f"blocks.{layer}.mlp.hook_pre", make_pre_hook()))
            hooks.append((f"blocks.{layer}.mlp.hook_post", make_post_hook()))

    return hooks

def get_scores_relp(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Relevance Patching (RelP), Mohebbi et al. 2025 (arxiv:2508.21258).

    Drop-in replacement for the gradient term in standard EAP / AtP. The
    forward and backward passes are the same as EAP, except a set of
    forward hooks (built by ``_build_relp_hooks``) detach attention
    probabilities and layer-norm scales and re-route the MLP-activation
    gradient through an identity-like rule. The detach tricks redirect
    gradient flow into the linear paths only, mimicking LRP-ε
    relevance propagation while keeping cost identical to EAP.

    The downstream score formula is unchanged: per-node
    contribution = E[ activation_diff * gradient ]. The gradient is what
    differs — RelP computes a relevance-style coefficient in place of
    raw ∂L/∂a.

    Half-rule (gated MLPs) is implemented; see ``_build_relp_hooks``.
    """
    get_logger("eap.attribute_node")

    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        scores = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward,), device=model.cfg.device, dtype=model.cfg.dtype)

    relp_hooks = _build_relp_hooks(model)

    total_items = 0
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader = dataloader if quiet else tqdm(dataloader, desc="RelP batches")

    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
        )

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        )

        with torch.inference_mode():
            if intervention == "patching":
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

        # The relp_hooks must be stacked alongside fwd_hooks_clean so that
        # both the activation_difference accumulator and the LRP detaches
        # are active during the gradient pass.
        with model.hooks(fwd_hooks=fwd_hooks_clean + relp_hooks, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=clean_attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
        model.zero_grad(set_to_none=True)

    scores /= total_items
    return scores

def get_scores_eap_gp(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps: int = 5,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    EAP-GP / GradPath, Zhang et al. 2025 (arxiv:2502.06852, NeurIPS 2025).

    Same per-edge formula as EAP-IG  (x_u - x'_u) * mean_j(d L / d x_v at γ_j),
    but the integration path γ replaces the straight-line interpolation
    α * (clean - corrupted) with an adaptively-constructed sequence of
    points in input-embedding space. Each step descends the L2 distance
    between the model's output at γ and at the corrupted input, taking
    unit-norm steps in the negative-gradient direction (Eq 8 of the paper):

        γ_{j+1} = γ_j - g_{j+1} / ||g_{j+1}||,
        g_{j+1} = ∂ || G(γ_j) - G(x'_u) ||² / ∂γ_j

    Path construction starts at the clean input embedding and runs for
    `steps` iterations (k=5 in the paper's main experiments). After the
    path is constructed, attribution accumulates the standard EAP
    grad * activation_diff product at each γ_j, divided by k.

    Cost: 2k forward+backward passes per batch (k path-construction +
    k attribution), so roughly 2x EAP-IG at the same `steps`.
    """
    get_logger("eap.attribute_node")

    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        scores = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward,), device=model.cfg.device, dtype=model.cfg.dtype)

    total_items = 0
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader = dataloader if quiet else tqdm(dataloader, desc="EAP-GP batches")

    input_node = graph.nodes["input"]

    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
        )

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        )

        # Capture the corrupted input embedding (γ target reference) and the
        # corrupted logits (G(x'_u)) under no_grad. The activation_difference
        # buffer is filled by the standard fwd_hooks_corrupted machinery as in EAP-IG.
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                corrupted_logits_full = model(
                    corrupted_tokens, attention_mask=corrupted_attention_mask
                )

        # Pull the corrupted input activations out of the buffer (mirrors EAP-IG).
        input_activations_corrupted = activation_difference[
            :, :, graph.forward_index(input_node)
        ].clone()

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

        # After fwd_hooks_clean, activation_difference holds (corrupted - clean),
        # so input_activations_clean = corrupted - (corrupted - clean) = clean.
        input_activations_clean = (
            input_activations_corrupted
            - activation_difference[:, :, graph.forward_index(input_node)]
        )

        # Frozen target for the path-construction loss (G(x'_u)).
        # Detach + clone so backward through gamma can't touch it.
        corrupted_logits_target = corrupted_logits_full.detach().clone()

        # Path construction in input-embedding space.
        # γ_0 = clean. After k steps we have γ_1 ... γ_k.
        gamma = input_activations_clean.detach().clone()
        path_points = []

        def make_replace_input_hook(replacement: torch.Tensor):
            def hook_fn(activations, hook):
                act_d = activations.shape[-1]
                # `+ activations * 0` keeps autograd tracking through the
                # replacement so gradients flow to `replacement`.
                return replacement[..., :act_d] + activations * 0

            return hook_fn

        for j in range(steps):
            gamma = gamma.detach().requires_grad_(True)
            replace_hook = make_replace_input_hook(gamma)
            with model.hooks(fwd_hooks=[(input_node.out_hook, replace_hook)]):
                logits_at_gamma = model(clean_tokens, attention_mask=clean_attention_mask)
            # Per-example L2 distance in logit space between γ and corrupted.
            # Sum over (seq, vocab) per example so backward gives a [batch, seq, d_model] grad.
            dist_sq = ((logits_at_gamma - corrupted_logits_target) ** 2).sum()
            grad_gamma = torch.autograd.grad(dist_sq, gamma, retain_graph=False)[0]
            # Per-example unit step: each example's gamma moves with unit
            # norm in its own embedding space, so step size is independent
            # of batch composition.
            per_ex_norm = grad_gamma.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
            per_ex_norm = per_ex_norm.view(-1, *([1] * (grad_gamma.ndim - 1)))
            with torch.no_grad():
                gamma = gamma - grad_gamma / per_ex_norm
            path_points.append(gamma.detach().clone())

        # Attribution pass: at each γ_j, run the model with γ as the
        # input-node activations and accumulate the standard EAP gradient
        # product into `scores` via bwd_hooks.
        total_steps = 0
        for gamma_j in path_points:
            total_steps += 1
            replace_hook = make_replace_input_hook(gamma_j)
            # NB: do NOT add fwd_hooks_clean here. The buffer already holds
            # (corrupted - clean) from setup; fwd_hooks_clean would subtract
            # each step's activations again, corrupting it cumulatively.
            # Mirrors EAP-IG's attribution loop, which uses only its input hook.
            with model.hooks(
                fwd_hooks=[(input_node.out_hook, replace_hook)],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean_tokens, attention_mask=clean_attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()
            model.zero_grad(set_to_none=True)

        # `scores` was accumulated with sum-over-batch via bwd_hooks; the
        # other EAP-IG functions divide by total_items at the end. We mirror that.
        # `total_steps` accumulates per batch so the per-step normalisation
        # is identical to get_scores_eap_ig: divide by k AFTER the loop ends.

    scores /= total_items
    scores /= steps

    return scores

def get_scores_atp_grad_drop(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Attribution Patching with GradDrop (AtP+GD), Kramár et al. 2024 (arxiv:2403.00745).

    Standard AtP linearises the patching effect via gradient * activation_diff.
    GradDrop reduces the bias from inter-layer gradient cancellation by averaging
    L copies of AtP, each with the gradient at one block's attention output zeroed
    so that block stops contributing to upstream gradients. The per-component
    contribution is the mean over those L passes of the absolute intervention
    effects, divided by (L - 1) to normalise for the dropped layer.

    Implementation note: this codebase's attribution backbone does not split
    Q/K from the joined attention head, so the AtP* Q-patching and K-patching
    refinements (Kramár §3.1) are not implemented here — only AtP+GD. The
    output is a single per-node score tensor, drop-in compatible with the
    rest of the EAP family.
    """
    n_layers = graph.cfg["n_layers"]
    if n_layers < 2:
        raise ValueError(f"AtP+GD requires at least 2 layers, got {n_layers}")

    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        accum = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        accum = torch.zeros((graph.n_forward,), device=model.cfg.device, dtype=model.cfg.dtype)

    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)

    layer_iter = range(n_layers) if quiet else tqdm(range(n_layers), desc="AtP+GD layers")
    for drop_layer in layer_iter:
        # Per-pass score accumulator. AtP+GD aggregates the absolute value of
        # this tensor across passes, so we allocate a fresh one per layer.
        if neuron:
            scores_l = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
        else:
            scores_l = torch.zeros((graph.n_forward,), device=model.cfg.device, dtype=model.cfg.dtype)

        # Mid-network gradient-zeroing hook: returns zero for the gradient
        # arriving at this block's attn_out, so subsequent (earlier) layers
        # see a gradient that excludes this block's residual contribution.
        drop_hook_name = f"blocks.{drop_layer}.hook_attn_out"

        def _zero_grad_hook(grads, hook):
            # TransformerLens registers grad-modifying bwd hooks via
            # register_full_backward_hook, whose return value replaces
            # grad_input and must be a tuple matching the module's input
            # arity (HookPoint.forward takes one tensor -> one-tuple).
            return (torch.zeros_like(grads),)

        total_items = 0
        for clean, corrupted, label in dataloader:
            batch_size = len(clean)
            total_items += batch_size

            (
                clean_tokens,
                corrupted_tokens,
                clean_attention_mask,
                corrupted_attention_mask,
                input_lengths,
                n_pos,
            ) = tokenize_batch_pair(
                model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
            )

            (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
                make_hooks_and_matrices(model, graph, batch_size, n_pos, scores_l, neuron=neuron)
            )

            with torch.inference_mode():
                if intervention == "patching":
                    with model.hooks(fwd_hooks_corrupted):
                        _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
                clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

            # Add the grad-drop bwd hook on top of the standard EAP bwd hooks.
            with model.hooks(
                fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks + [(drop_hook_name, _zero_grad_hook)]
            ):
                logits = model(clean_tokens, attention_mask=clean_attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

        scores_l /= total_items
        accum += scores_l.abs()

    # AtP+GD eq. 11: average of |I_AtP+GD_l| over L layers, normalised by (L-1).
    return accum / (n_layers - 1)

def get_scores_eap(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Compute node attribution scores using Edge Attribution Patching (EAP).

    Estimates each node's importance via a single linearised patching pass.
    In neuron mode, produces per-neuron scores; in node mode, collapses to a
    scalar per node by summing over the hidden dimension.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose node scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Source of reference activations for the activation difference.
            Defaults to 'patching' (corrupted inputs).
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        neuron (bool): If True, return per-neuron scores [n_forward, max_d];
            otherwise return per-node scores [n_forward]. Defaults to False.

    Returns:
        Tensor: Score tensor of shape [n_forward, max_d] (neuron=True) or
            [n_forward] (neuron=False), averaged over all examples.
    """

    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        scores = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device=model.cfg.device, dtype=model.cfg.dtype)

    if "mean" in intervention:
        if intervention_dataloader is None:
            raise ValueError(
                f"intervention={intervention!r} requires an "
                f"intervention_dataloader, but none was provided. Pass an "
                f"intervention_dataloader, or use intervention='patching' or "
                f"'zero' which do not need one."
            )
        per_position = "positional" in intervention
        means = compute_mean_activations(
            model,
            graph,
            intervention_dataloader,
            per_position=per_position,
            padding_side=getattr(intervention_dataloader, "pair_padding_side", None),
        )
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    total_items = 0
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader = dataloader if quiet else tqdm(dataloader)

    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
        )

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        )

        with torch.inference_mode():
            if intervention == "patching":
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            elif "mean" in intervention:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=clean_attention_mask)

            try:
                metric_value = metric(logits, clean_logits, input_lengths, label)
            except Exception as e:
                logger.error("[DEBUG ERROR] Metric Calculation Failed")
                raise e

            metric_value.backward()

    scores /= total_items

    return scores

def get_scores_eap_ig(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Compute node scores using Integrated Gradients over intermediate activations.

    Interpolates the output activation of each node individually between its
    corrupted and clean values across `steps` points. Scores are accumulated
    per node rather than per input embedding.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose node scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Reference activations for the activation difference.
            Defaults to 'patching'.
        steps (int): Number of integration steps per node. Defaults to 30.
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        neuron (bool): If True, return per-neuron scores [n_forward, max_d];
            otherwise return per-node scores [n_forward]. Defaults to False.

    Returns:
        Tensor: Score tensor of shape [n_forward, max_d] (neuron=True) or
            [n_forward] (neuron=False), averaged over all examples, nodes,
            and integration steps.
    """
    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        scores = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device=model.cfg.device, dtype=model.cfg.dtype)

    total_items = 0
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader = dataloader if quiet else tqdm(dataloader)

    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
        )

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)

            input_activations_corrupted = activation_difference[
                :, :, graph.forward_index(graph.nodes["input"])
            ].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

            input_activations_clean = (
                input_activations_corrupted
                - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
            )

        # + activations * 0  will cause a backwards pass on new_input
        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                act_d = activations.shape[-1]
                corrupt_sliced = input_activations_corrupted[..., :act_d]
                clean_sliced = input_activations_clean[..., :act_d]
                new_input = (
                    corrupt_sliced + (k / steps) * (clean_sliced - corrupt_sliced) + activations * 0
                )
                return new_input

            return hook_fn

        total_steps = 0
        for step in range(1, steps + 1):
            total_steps += 1
            with model.hooks(
                fwd_hooks=[(graph.nodes["input"].out_hook, input_interpolation_hook(step))],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean_tokens, attention_mask=clean_attention_mask)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    # Guard against empty/degenerate dataloaders (total_steps would be 0 if
    # the inner loop never ran). Doesn't affect our grid — every cell has a
    # populated dataloader — but is cheap insurance against silent NaN.
    scores /= max(total_steps, 1)

    return scores

def get_scores_ig_activations(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    steps=30,
    intervention_dataloader: Optional[DataLoader] = None,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Compute node scores using Integrated Gradients over intermediate activations.

    Interpolates the output activation of each node individually between its
    corrupted and clean values across `steps` points. Scores are accumulated
    per node rather than per input embedding.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose node scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Reference activations for the activation difference.
            Defaults to 'patching'.
        steps (int): Number of integration steps per node. Defaults to 30.
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        neuron (bool): If True, return per-neuron scores [n_forward, max_d];
            otherwise return per-node scores [n_forward]. Defaults to False.

    Returns:
        Tensor: Score tensor of shape [n_forward, max_d] (neuron=True) or
            [n_forward] (neuron=False), averaged over all examples, nodes,
            and integration steps.
    """
    if "mean" in intervention:
        if intervention_dataloader is None:
            raise ValueError(
                f"intervention={intervention!r} requires an "
                f"intervention_dataloader, but none was provided. Pass an "
                f"intervention_dataloader, or use intervention='patching' or "
                f"'zero' which do not need one."
            )
        per_position = "positional" in intervention
        means = compute_mean_activations(
            model,
            graph,
            intervention_dataloader,
            per_position=per_position,
            padding_side=getattr(intervention_dataloader, "pair_padding_side", None),
        )
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        scores = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device=model.cfg.device, dtype=model.cfg.dtype)

    total_items = 0
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size

        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
        )

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores, neuron=neuron
        )
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores, neuron=neuron
        )
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores, neuron=neuron
        )

        if intervention == "patching":
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)

        elif "mean" in intervention:
            activation_difference += means

        with model.hooks(fwd_hooks=fwd_hooks_clean):
            clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

            activation_difference += (
                activations_corrupted.clone().detach() - activations_clean.clone().detach()
            )

        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k / steps
                act_d = activations.shape[-1]
                clean_sliced = clean[..., :act_d]
                corrupted_sliced = corrupted[..., :act_d]
                new_output = alpha * clean_sliced + (1 - alpha) * corrupted_sliced + activations * 0
                return new_output

            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes["input"]]
        for layer in range(graph.cfg["n_layers"]):
            nodeslist.append(graph.nodes[f"a{layer}.h0"])
            nodeslist.append(graph.nodes[f"m{layer}"])

        for node in nodeslist:
            for step in range(1, steps + 1):
                total_steps += 1

                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [
                    (node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))
                ]

                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(clean_tokens, attention_mask=clean_attention_mask)
                    metric_value = metric(logits, clean_logits, input_lengths, label)

                    metric_value.backward(retain_graph=True)

    scores /= total_items
    # Guard against empty/degenerate dataloaders (total_steps would be 0 if
    # the inner loop never ran). Doesn't affect our grid — every cell has a
    # populated dataloader — but is cheap insurance against silent NaN.
    scores /= max(total_steps, 1)

    return scores

def get_scores_clean_corrupted(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Compute node scores using a two-point clean/corrupted gradient approximation.

    Runs backward passes at both the clean and corrupted inputs and averages
    the resulting node gradients. Equivalent to IG with steps=2 and faster
    than full EAP-IG. Only supports patching-style intervention.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose node scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        neuron (bool): If True, return per-neuron scores [n_forward, max_d];
            otherwise return per-node scores [n_forward]. Defaults to False.

    Returns:
        Tensor: Score tensor of shape [n_forward, max_d] (neuron=True) or
            [n_forward] (neuron=False), averaged over all examples.
    """
    if neuron:
        max_d = (
            max(model.cfg.d_model, graph.cfg.get("d_mlp", 0))
            if graph.cfg.get("mlp_hook") == "post_act"
            else model.cfg.d_model
        )
        scores = torch.zeros((graph.n_forward, max_d), device=model.cfg.device, dtype=model.cfg.dtype)
    else:
        scores = torch.zeros((graph.n_forward), device=model.cfg.device, dtype=model.cfg.dtype)

    total_items = 0
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        (
            clean_tokens,
            corrupted_tokens,
            clean_attention_mask,
            corrupted_attention_mask,
            input_lengths,
            n_pos,
        ) = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
        )

        # Here, we get our fwd / bwd hooks and the activation difference matrix
        # The forward corrupted hooks add the corrupted activations to the activation difference matrix
        # The forward clean hooks subtract the clean activations
        # The backward hooks get the gradient, and use that, plus the activation difference, for the scores
        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores, neuron=neuron)
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=corrupted_attention_mask)

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean_tokens, attention_mask=clean_attention_mask)

        total_steps = 2
        with model.hooks(bwd_hooks=bwd_hooks):
            logits = model(clean_tokens, attention_mask=clean_attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

            logits = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()
            model.zero_grad()

    scores /= total_items
    # Guard against empty/degenerate dataloaders (total_steps would be 0 if
    # the inner loop never ran). Doesn't affect our grid — every cell has a
    # populated dataloader — but is cheap insurance against silent NaN.
    scores /= max(total_steps, 1)

    return scores

allowed_aggregations = {"sum", "mean"}

def attribute_node(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    method: Literal[
        "EAP",
        "EAP-IG-inputs",
        "EAP-IG-activations",
        "exact",
        "clean-corrupted",
        "atp-gd",
        "eap-gp",
        "relp",
        "peap",
        "ifr",
    ],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    aggregation="sum",
    ig_steps: Optional[int] = None,
    intervention_dataloader: Optional[DataLoader] = None,
    quiet: bool = False,
    neuron: bool = False,
):
    """
    Compute node (or neuron) attribution scores using the specified method.

    Dispatcher that routes to the appropriate scoring function and writes the
    results into graph.nodes_scores (node-level) or graph.neurons_scores
    (neuron-level).

    Args:
        model (HookedTransformer): The model to attribute. Must have
            use_attn_result, use_split_qkv_input, and use_hook_mlp_in enabled.
        graph (Graph): Graph whose scores will be populated in-place.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
            Signature: metric(logits, clean_logits, input_lengths, label).
        method (Literal[...]): Attribution algorithm. One of:
            'EAP'               - Edge Attribution Patching.
            'EAP-IG-inputs'     - EAP with Integrated Gradients on inputs.
            'EAP-IG-activations'- IG over intermediate node activations.
            'exact'             - Leave-one-out exact patching.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Reference ablation type. Not all methods support all interventions.
            Defaults to 'patching'.
        aggregation (str): How to aggregate hidden-dim scores per node.
            'sum' keeps raw sums; 'mean' divides by d_model. Defaults to 'sum'.
        ig_steps (Optional[int]): Integration steps for IG-based methods.
            Defaults to None (each function uses its own default of 30).
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        neuron (bool): If True, compute and store per-neuron scores;
            otherwise compute per-node scalar scores. Defaults to False.

    Raises:
        ValueError: If required model config flags are not set, if aggregation
            is invalid, or an incompatible intervention is used with a method.
    """
    if not model.cfg.use_attn_result:
        raise ValueError(
            "EAP node attribution requires model.cfg.use_attn_result=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_attn_result = True before discovery."
        )
    if not model.cfg.use_split_qkv_input:
        raise ValueError(
            "EAP node attribution requires model.cfg.use_split_qkv_input=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_split_qkv_input = True before discovery."
        )
    if not model.cfg.use_hook_mlp_in:
        raise ValueError(
            "EAP node attribution requires model.cfg.use_hook_mlp_in=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_hook_mlp_in = True before discovery."
        )
    if model.cfg.n_key_value_heads is not None:
        if not model.cfg.ungroup_grouped_query_attention:
            raise ValueError(
                "This model uses grouped-query attention, so EAP node "
                "attribution requires model.cfg.ungroup_grouped_query_attention="
                "True. Load the model with circuitkit.load_model(...), which "
                "sets this flag, or set "
                "model.cfg.ungroup_grouped_query_attention = True before "
                "discovery."
            )

    if aggregation not in allowed_aggregations:
        raise ValueError(f"aggregation must be in {allowed_aggregations}, but got {aggregation}")

    # Scores are by default summed across the d_model dimension
    # This means that scores are a [n_src_nodes, n_dst_nodes] tensor
    if method == "EAP":
        scores = get_scores_eap(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
            neuron=neuron,
        )
    elif method == "EAP-IG-inputs":
        if intervention != "patching":
            raise ValueError(
                f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}"
            )
        _steps = ig_steps if ig_steps is not None else 30
        scores = get_scores_eap_ig(
            model, graph, dataloader, metric, steps=_steps, quiet=quiet, neuron=neuron
        )
    elif method == "EAP-IG-activations":
        _steps = ig_steps if ig_steps is not None else 30
        scores = get_scores_ig_activations(
            model,
            graph,
            dataloader,
            metric,
            steps=_steps,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
            neuron=neuron,
        )
    elif method == "exact":
        _steps = ig_steps if ig_steps is not None else 30
        scores = get_scores_exact(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
        )
    elif method == "clean-corrupted":
        if intervention != "patching":
            raise ValueError(
                f"intervention must be 'patching' for clean-corrupted, but got {intervention}"
            )
        scores = get_scores_clean_corrupted(
            model, graph, dataloader, metric, quiet=quiet, neuron=neuron
        )
    elif method == "atp-gd":
        scores = get_scores_atp_grad_drop(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
            neuron=neuron,
        )
    elif method == "eap-gp":
        _steps = ig_steps if ig_steps is not None else 5
        scores = get_scores_eap_gp(
            model, graph, dataloader, metric, steps=_steps, quiet=quiet, neuron=neuron
        )
    elif method == "relp":
        scores = get_scores_relp(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
            neuron=neuron,
        )
    elif method == "peap":
        scores = get_scores_peap(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
            neuron=neuron,
        )
    elif method == "ifr":
        scores = get_scores_ifr(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
            neuron=neuron,
        )
    else:
        raise ValueError(
            f"method must be in ['EAP', 'EAP-IG-inputs', 'EAP-IG-activations', 'exact', 'clean-corrupted', 'atp-gd', 'eap-gp', 'relp', 'peap', 'ifr'], but got {method}"
        )

    if aggregation == "mean":
        scores /= model.cfg.d_model

    if neuron:
        graph.neurons_scores[:] = scores.to(graph.neurons_scores.device)
    else:
        graph.nodes_scores[:] = scores.to(graph.nodes_scores.device)
