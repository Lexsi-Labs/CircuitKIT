from functools import partial
from typing import Callable, List, Literal, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .eap_utils import (
    compute_mean_activations,
    make_hooks_and_matrices,
    tokenize_batch_pair,
    tokenize_plus,
)
from .evaluate import evaluate_baseline, evaluate_graph
from .graph import Graph


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
    Compute edge attribution scores via exact leave-one-out patching.

    Iterates over every edge, temporarily removes it from the graph, measures
    the performance drop, and assigns that drop as the edge's score.
    Expensive: requires one full forward pass per edge.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): The graph whose edges are scored. All real edges are
            added to the graph before iteration begins.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Ablation type used when an edge is removed. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Edge score matrix [n_forward, n_backward]. Scores are also
            written in-place to graph.scores.
    """
    graph.in_graph |= graph.real_edge_mask  # All edges that are real are now in the graph
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    baseline = evaluate_baseline(model, dataloader, metric).mean().item()
    edges = graph.edges.values() if quiet else tqdm(graph.edges.values())
    for edge in edges:
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
                pair_padding_side=pair_padding_side,
            )
            .mean()
            .item()
        )
        edge.score = intervened_performance - baseline
        edge.in_graph = True

    # This is just to make the return type the same as all of the others; we've actually already updated the score matrix
    return graph.scores


def get_scores_eap(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):
    """
    Compute edge attribution scores using Edge Attribution Patching (EAP).

    Estimates edge importance via a single linearised patching pass: one forward
    pass on corrupted inputs (to build the activation difference), one
    forward+backward pass on clean inputs (to collect gradients). Edge scores
    are the dot product of the activation difference and the gradient, summed
    over the hidden dimension.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose edge scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Reference ablation type for building the activation difference.
            Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Edge score matrix [n_forward, n_backward], averaged over all
            examples.
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=model.cfg.device, dtype=model.cfg.dtype)

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
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
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
            metric_value = metric(logits, clean_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores


def get_scores_eap_ig(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
):
    """
    Compute edge attribution scores using EAP with Integrated Gradients on inputs.

    Interpolates the input embeddings between corrupted and clean across `steps`
    intermediate points, accumulating gradients at each step (Sundararajan et al.).
    More accurate than vanilla EAP but requires `steps` forward+backward passes
    per batch. Only supports patching-style intervention.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose edge scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        steps (int): Number of integration steps. Defaults to 30.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Edge score matrix [n_forward, n_backward], averaged over all
            examples and integration steps.
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=model.cfg.device, dtype=model.cfg.dtype)

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
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
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

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                    input_activations_clean - input_activations_corrupted
                )
                new_input.requires_grad = True
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
                if torch.isnan(metric_value).any().item():
                    raise ValueError(
                        f"Metric value is NaN.\nClean: {clean}\nCorrupted: {corrupted}\nLabel: {label}"
                    )
                metric_value.backward()

            if torch.isnan(scores).any().item():
                raise ValueError(
                    f"Scores became NaN at Step: {step}.\nClean: {clean}\nCorrupted: {corrupted}\nLabel: {label}"
                )

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_ig_activations(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    steps=30,
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):
    """
    Compute edge scores using Integrated Gradients over intermediate activations.

    Unlike EAP-IG-inputs (which interpolates only the input embeddings), this
    method interpolates the output activations of every node individually,
    giving a more granular attribution at the cost of additional forward passes.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose edge scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Reference activations used to build the activation difference.
            Defaults to 'patching'.
        steps (int): Number of integration steps per node. Defaults to 30.
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Edge score matrix [n_forward, n_backward], averaged over all
            examples, nodes, and integration steps.
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

    scores = torch.zeros((graph.n_forward, graph.n_backward), device=model.cfg.device, dtype=model.cfg.dtype)

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
            model, graph, batch_size, n_pos, scores
        )
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
        )
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(
            model, graph, batch_size, n_pos, scores
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
                new_output = alpha * clean + (1 - alpha) * corrupted
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
    scores /= total_steps

    return scores


def get_scores_clean_corrupted(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    quiet=False,
):
    """
    Compute edge scores using a two-point clean/corrupted gradient approximation.

    A lightweight alternative to full Integrated Gradients: computes gradients
    at only two endpoints (clean and corrupted inputs) and averages them,
    equivalent to IG with steps=2. Faster than EAP-IG but less accurate.
    Only supports patching-style intervention.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose edge scores will be computed.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Edge score matrix [n_forward, n_backward], averaged over
            all examples.
    """
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=model.cfg.device, dtype=model.cfg.dtype)

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
            make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
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

            corrupted_logits = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            corrupted_metric_value = metric(corrupted_logits, clean_logits, input_lengths, label)
            corrupted_metric_value.backward()
            model.zero_grad()

    scores /= total_items
    scores /= total_steps

    return scores


def get_scores_information_flow_routes(
    model: HookedTransformer, graph: Graph, dataloader: DataLoader, quiet=False
) -> torch.Tensor:
    """
    Compute edge scores using the Information Flow Routes method (Ferrando et al., 2024).

    Scores edges by measuring the L1-proximity between each node's output
    activation and its upstream predecessors' activations, normalised into an
    importance distribution. Does not require a metric or corrupted inputs;
    runs a single clean forward pass per batch.

    Note:
        The `metric` parameter is accepted for API consistency with other
        attribution functions but is not used internally.

    Args:
        model (HookedTransformer): The model to attribute.
        graph (Graph): Graph whose edge scores will be computed.
        dataloader (DataLoader): Data to attribute over (clean inputs only).
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Tensor: Edge score matrix [n_forward, n_backward], averaged over all
            examples.
    """
    # I could do some hacky overriding of make_hooks_and_matrices here but I will not
    templated = getattr(dataloader, "templated", False)
    scores = torch.zeros((graph.n_forward, graph.n_backward), device=model.cfg.device, dtype=model.cfg.dtype)

    def make_hooks(n_pos: int, input_lengths: torch.Tensor) -> List[Tuple[str, Callable]]:
        output_activations = torch.zeros(
            (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
            device=model.cfg.device,
            dtype=model.cfg.dtype,
        )

        def output_hook(index, activations, hook):
            try:
                acts = activations.detach()
                output_activations[:, :, index] = acts
            except RuntimeError as e:
                raise RuntimeError(
                    f"{hook.name} failed. Target: {output_activations[:, :, index].size()}, Acts: {output_activations.size()}"
                ) from e

        # compute the score directly, without saving the input activations
        def input_hook(prev_index, bwd_index, input_lengths, activations, hook):
            acts = activations.detach()
            try:
                if acts.ndim == 3:
                    acts = acts.unsqueeze(2)
                # acts : batch pos backward hidden
                # output acts: batch pos forward hidden
                # add forward and backwards dimensions to acts and output acts respectively
                acts = acts.unsqueeze(2)
                unsqueezed_output_activations = output_activations.unsqueeze(3)

                # acts : batch pos 1 backward hidden
                # output acts: batch pos forward 1 hidden
                proximity = torch.clamp(
                    -torch.linalg.vector_norm(
                        unsqueezed_output_activations[:, :, :prev_index] - acts, ord=1, dim=-1
                    )
                    + torch.linalg.vector_norm(acts, ord=1, dim=-1),
                    min=0,
                )
                importance = proximity / torch.sum(proximity, dim=2, keepdim=True)
                # importance: batch pos forward backward
                # aggregate over positions via sum/mean to get importance: forward backward
                # first mask out importances for padding positions
                max_len = input_lengths.max()
                mask = torch.arange(
                    max_len, device=input_lengths.device, dtype=input_lengths.dtype
                ).expand(len(input_lengths), max_len) < input_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                importance *= mask
                importance = importance.sum(1) / input_lengths.view(-1, 1, 1)  # mean over positions
                importance = importance.sum(0)

                # importance: forward backward
                # squeezing backward dim in case it isn't real (i.e. it's an MLP)
                importance = importance.squeeze(1)
                scores[:prev_index, bwd_index] += importance

            except RuntimeError as e:
                raise RuntimeError(
                    f"{hook.name} failed. Target: {unsqueezed_output_activations[:, :, prev_index].size()}, Acts: {acts.size()}"
                ) from e

        hooks = []
        node = graph.nodes["input"]
        fwd_index = graph.forward_index(node)
        hooks.append((node.out_hook, partial(output_hook, fwd_index)))

        for layer in range(graph.cfg["n_layers"]):
            node = graph.nodes[f"a{layer}.h0"]
            fwd_index = graph.forward_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            prev_index = graph.prev_index(node)
            for i, letter in enumerate("qkv"):
                bwd_index = graph.backward_index(node, qkv=letter)
                hooks.append(
                    (node.qkv_inputs[i], partial(input_hook, prev_index, bwd_index, input_lengths))
                )

            node = graph.nodes[f"m{layer}"]
            fwd_index = graph.forward_index(node)
            bwd_index = graph.backward_index(node)
            prev_index = graph.prev_index(node)
            hooks.append((node.out_hook, partial(output_hook, fwd_index)))
            hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))

        node = graph.nodes["logits"]
        prev_index = graph.prev_index(node)
        bwd_index = graph.backward_index(node)
        hooks.append((node.in_hook, partial(input_hook, prev_index, bwd_index, input_lengths)))
        return hooks

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, _, _ in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(
            model, clean, templated=templated
        )

        hooks = make_hooks(n_pos, input_lengths)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=hooks):
                _ = model(clean_tokens, attention_mask=attention_mask)

    scores /= total_items

    return scores


allowed_aggregations = {"sum", "mean"}


def attribute(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    method: Literal[
        "EAP",
        "EAP-IG-inputs",
        "clean-corrupted",
        "EAP-IG-activations",
        "information-flow-routes",
        "exact",
    ],
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    aggregation="sum",
    ig_steps: Optional[int] = None,
    intervention_dataloader: Optional[DataLoader] = None,
    quiet=False,
):
    """
    Compute edge attribution scores for a graph using the specified method.

    Dispatcher that routes to the appropriate scoring function and writes the
    resulting scores into `graph.scores`.

    Args:
        model (HookedTransformer): The model to attribute. Must have
            use_attn_result, use_split_qkv_input, and use_hook_mlp_in enabled.
        graph (Graph): Graph whose edge scores will be populated.
        dataloader (DataLoader): Data to attribute over.
        metric (Callable[[Tensor], Tensor]): Performance metric callable.
            Signature: metric(logits, clean_logits, input_lengths, label).
        method (Literal[...]): Attribution algorithm. One of:
            'EAP'                    - Edge Attribution Patching.
            'EAP-IG-inputs'          - EAP with Integrated Gradients on inputs.
            'clean-corrupted'        - Two-point IG approximation.
            'EAP-IG-activations'     - IG over intermediate node activations.
            'information-flow-routes'- Ferrando et al. (2024) proximity method.
            'exact'                  - Leave-one-out exact patching.
        intervention (Literal['patching', 'zero', 'mean', 'mean-positional']):
            Reference ablation type. Not all methods support all interventions.
            Defaults to 'patching'.
        aggregation (str): How to aggregate d_model scores into a scalar per
            edge. 'sum' keeps raw sums; 'mean' divides by d_model.
            Defaults to 'sum'.
        ig_steps (Optional[int]): Number of integration steps for IG-based
            methods ('EAP-IG-inputs', 'EAP-IG-activations'). Defaults to None
            (each function uses its own default of 30).
        intervention_dataloader (Optional[DataLoader]): Required when
            intervention is 'mean' or 'mean-positional'. Defaults to None.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Raises:
        ValueError: If required model config flags are not set, if aggregation
            is not 'sum' or 'mean', or if an incompatible intervention is used
            with a given method.
    """
    if not model.cfg.use_attn_result:
        raise ValueError(
            "EAP edge attribution requires model.cfg.use_attn_result=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_attn_result = True before discovery."
        )
    if not model.cfg.use_split_qkv_input:
        raise ValueError(
            "EAP edge attribution requires model.cfg.use_split_qkv_input=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_split_qkv_input = True before discovery."
        )
    if not model.cfg.use_hook_mlp_in:
        raise ValueError(
            "EAP edge attribution requires model.cfg.use_hook_mlp_in=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_hook_mlp_in = True before discovery."
        )
    if model.cfg.n_key_value_heads is not None:
        if not model.cfg.ungroup_grouped_query_attention:
            raise ValueError(
                "This model uses grouped-query attention, so EAP edge "
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
        )
    elif method == "EAP-IG-inputs":
        if intervention != "patching":
            raise ValueError(
                f"intervention must be 'patching' for EAP-IG-inputs, but got {intervention}"
            )
        scores = get_scores_eap_ig(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)
    elif method == "clean-corrupted":
        if intervention != "patching":
            raise ValueError(
                f"intervention must be 'patching' for clean-corrupted, but got {intervention}"
            )
        scores = get_scores_clean_corrupted(model, graph, dataloader, metric, quiet=quiet)
    elif method == "EAP-IG-activations":
        scores = get_scores_ig_activations(
            model,
            graph,
            dataloader,
            metric,
            steps=ig_steps,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
        )
    elif method == "information-flow-routes":
        scores = get_scores_information_flow_routes(model, graph, dataloader, quiet=quiet)
    elif method == "exact":
        scores = get_scores_exact(
            model,
            graph,
            dataloader,
            metric,
            intervention=intervention,
            intervention_dataloader=intervention_dataloader,
            quiet=quiet,
        )
    else:
        raise ValueError(
            f"method must be in ['EAP', 'EAP-IG-inputs', 'clean-corrupted', 'EAP-IG-activations', 'information-flow-routes', 'exact'], but got {method}"
        )

    if aggregation == "mean":
        scores /= model.cfg.d_model

    graph.scores[:] = scores.to(graph.scores.device)
