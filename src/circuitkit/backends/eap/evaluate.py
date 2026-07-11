from collections import defaultdict
from typing import Callable, List, Literal, Optional, Union

import torch
from einops import einsum
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .eap_utils import compute_mean_activations, make_hooks_and_matrices, tokenize_batch_pair
from .graph import AttentionNode, Graph, MLPNode


def evaluate_graph(  # noqa: C901 - complex function, refactor out of scope for lint pass
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metrics: Union[Callable[[Tensor], Tensor], List[Callable[[Tensor], Tensor]]],
    quiet=False,
    intervention: Literal["patching", "zero", "mean", "mean-positional"] = "patching",
    intervention_dataloader: Optional[DataLoader] = None,
    skip_clean: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate a pruned circuit's faithfulness on a task.

    Runs the model with out-of-circuit edges ablated (replaced with corrupted,
    zero, or mean activations depending on intervention). Call graph.apply_topn()
    or graph.apply_threshold() before this to define the circuit, and graph.prune()
    to ensure full connectivity (also called internally here).

    Args:
        model (HookedTransformer): Model to evaluate. Must have use_attn_result=True.
        graph (Graph): Circuit graph with in_graph flags set on edges/nodes.
        dataloader (DataLoader): Evaluation dataset yielding (clean, corrupted, label) batches.
        metrics (Union[Callable, List[Callable]]): Metric function(s) with signature
            (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            If a list, all metrics are evaluated in one pass.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        intervention (str): Ablation method for out-of-circuit edges. One of:
            - 'patching': Replace with corrupted activations (interchange intervention).
            - 'zero': Replace with zeros.
            - 'mean': Replace with dataset mean activations (requires intervention_dataloader).
            - 'mean-positional': Replace with position-specific dataset means.
            Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader]): Dataset to compute mean
            activations from. Required when intervention is 'mean' or 'mean-positional'.
            Defaults to None.
        skip_clean (bool): If True, clean_logits passed to metrics will be None.
            Set to False for metrics that require the unablated logits (e.g. KL divergence).
            Defaults to True.

    Returns:
        Union[Tensor, List[Tensor]]: Per-sample metric scores [n_samples], or a list
            of such tensors if multiple metrics were provided.

    Raises:
        ValueError: If model.cfg.use_attn_result is False, if the intervention
            is invalid, or if a mean intervention requires an
            intervention_dataloader that was not provided.
    """
    if not model.cfg.use_attn_result:
        raise ValueError(
            "Circuit evaluation requires model.cfg.use_attn_result=True. "
            "Load the model with circuitkit.load_model(...), which sets this "
            "flag, or set model.cfg.use_attn_result = True before evaluation."
        )
    if model.cfg.n_key_value_heads is not None:
        if not model.cfg.ungroup_grouped_query_attention:
            raise ValueError(
                "This model uses grouped-query attention, so circuit "
                "evaluation requires model.cfg.ungroup_grouped_query_attention="
                "True. Load the model with circuitkit.load_model(...), which "
                "sets this flag, or set "
                "model.cfg.ungroup_grouped_query_attention = True before "
                "evaluation."
            )

    _valid_interventions = ["patching", "zero", "mean", "mean-positional"]
    if intervention not in _valid_interventions:
        raise ValueError(
            f"Invalid intervention {intervention!r}. " f"Use one of: {_valid_interventions}."
        )

    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
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

    # This step cleans up the graph, removing components until it's fully connected
    graph.prune()

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

    # same thing but for neurons
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        # Each node type owns a different number of neurons: d_mlp for post_act MLP nodes,
        # d_model for everything else. Build a per-forward-index expected count so the
        # "fully in graph" check is correct regardless of mlp_hook setting.
        expected_neurons = torch.full(
            (graph.n_forward,), model.cfg.d_model,
            device=model.cfg.device, dtype=model.cfg.dtype,
        )
        if graph.cfg.get("mlp_hook") == "post_act":
            _d_mlp = graph.cfg.get("d_mlp", getattr(model.cfg, "d_mlp", model.cfg.d_model))
            for node in graph.nodes.values():
                if isinstance(node, MLPNode):
                    expected_neurons[graph.forward_index(node, attn_slice=False)] = _d_mlp
        node_fully_in_graph = (neuron_matrix.sum(-1) == expected_neurons).to(model.cfg.dtype)
        in_graph_matrix = einsum(
            in_graph_matrix, node_fully_in_graph, "forward backward, forward -> forward backward"
        )
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use it as a mask to specify
    # which edges we want to corrupt
    in_graph_matrix = 1 - in_graph_matrix
    if neuron_matrix is not None:
        neuron_matrix = 1 - neuron_matrix

    if model.cfg.use_normalization_before_and_after:
        # If the model also normalizes the outputs of attention heads, we'll need to take that into account when evaluating the graph.
        attention_head_mask = torch.zeros(
            (graph.n_forward, model.cfg.n_layers), device=model.cfg.device, dtype=model.cfg.dtype
        )
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])

    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # Case where layernorm is applied after attention (gemma only)
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]

                # get the clean outputs of the attention heads that came before
                clean_attention_results = einsum(
                    activation_matrix[1, :, :, : len(in_graph_vector)],
                    attention_head_mask[: len(in_graph_vector)],
                    "batch pos previous hidden, previous layer -> batch pos layer hidden",
                )

                # get the update corresponding to non-attention heads, and the difference between clean and corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_matrix[: len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_matrix[: len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden",
                    )
                else:
                    non_attention_update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        non_attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous -> batch pos ... hidden",
                    )
                    corrupted_attention_difference = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        attention_head_mask[: len(in_graph_vector)],
                        "batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden",
                    )

                # add the biases to the attention results, and compute the corrupted attention results using the difference
                # we process all the attention heads at once; this is how we can tell if we're doing that
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = (
                        clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    )
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += (
                        attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    )
                else:
                    corrupted_attention_results = (
                        clean_attention_results + corrupted_attention_difference
                    )
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)

                # pass both the clean and corrupted attention results through the layernorm and
                # add the difference to the update
                update = non_attention_update
                valid_layers = attention_head_mask[: len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])

            else:
                # In the non-gemma case, things are easy!
                activation_differences = activation_matrix
                # The ... here is to account for a potential head dimension, when constructing a whole attention layer's input
                if neuron_matrix is not None:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        neuron_matrix[: len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden",
                    )
                else:
                    update = einsum(
                        activation_differences[:, :, : len(in_graph_vector)],
                        in_graph_vector,
                        "batch pos previous hidden, previous ... -> batch pos ... hidden",
                    )
            # Slice update to match activations' hidden dimension. When mlp_hook='post_act'
            # the buffer's last dim is d_mlp, but attention/logit activations are d_model-wide.
            activations += update[..., : activations.shape[-1]]
            return activations

        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in the layer is in the graph, just construct the input for the entire layer
            if any(
                graph.nodes[f"a{layer}.h{head}"].in_graph for head in range(model.cfg.n_heads)
            ) and not (
                neuron_matrix is None
                and all(
                    parent_edge.in_graph
                    for head in range(model.cfg.n_heads)
                    for parent_edge in graph.nodes[f"a{layer}.h{head}"].parent_edges
                )
            ):
                for i, letter in enumerate("qkv"):
                    node = graph.nodes[f"a{layer}.h0"]
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(
                        activation_differences,
                        in_graph_matrix[:prev_index, bwd_index],
                        neuron_matrix,
                    )
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))

            # add MLP hook if MLP in graph
            if graph.nodes[f"m{layer}"].in_graph and not (
                neuron_matrix is None
                and all(
                    parent_edge.in_graph for parent_edge in graph.nodes[f"m{layer}"].parent_edges
                )
            ):
                node = graph.nodes[f"m{layer}"]
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(
                    activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix
                )
                input_construction_hooks.append((node.in_hook, input_cons_hook))

        # Always add the logits hook
        if not (
            neuron_matrix is None
            and all(parent_edge.in_graph for parent_edge in graph.nodes["logits"].parent_edges)
        ):
            node = graph.nodes["logits"]
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(
                activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix
            )
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks

    # convert metrics to list if it's not already
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]

    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
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

        # fwd_hooks_corrupted adds in corrupted acts to activation_difference
        # fwd_hooks_clean subtracts out clean acts from activation_difference
        # activation difference is of size (batch, pos, src_nodes, hidden)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(
            model, graph, len(clean), n_pos, None
        )

        input_construction_hooks = make_input_construction_hooks(
            activation_difference, in_graph_matrix, neuron_matrix
        )
        with torch.inference_mode():
            if intervention == "patching":
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    # Forward pass runs the corrupted hooks for their side effects;
                    # the logits themselves are intentionally discarded.
                    model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            else:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                if "mean" in intervention:
                    activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = (
                None if skip_clean else model(clean_tokens, attention_mask=clean_attention_mask)
            )

            with model.hooks(fwd_hooks_clean + input_construction_hooks):
                logits = model(clean_tokens, attention_mask=clean_attention_mask)

        for i, metric in enumerate(metrics):
            r = metric(logits, clean_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_baseline(
    model: HookedTransformer,
    dataloader: DataLoader,
    metrics: List[Callable[[Tensor], Tensor]],
    run_corrupted=False,
    quiet=False,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate the unmodified model on a dataset to establish a performance baseline.

    No interventions are applied. Both clean and corrupted forward passes are always
    run; which one is scored depends on run_corrupted.

    Args:
        model (HookedTransformer): Model to evaluate.
        dataloader (DataLoader): Dataset yielding (clean, corrupted, label) batches.
        metrics (Union[Callable, List[Callable]]): Metric function(s) with signature
            (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
        run_corrupted (bool): If True, score the corrupted input (passing clean logits
            as the reference). If False, score the clean input. Defaults to False.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.

    Returns:
        Union[Tensor, List[Tensor]]: Per-sample metric scores [n_samples], or a list
            of such tensors if multiple metrics were provided.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    results = [[] for _ in metrics]
    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    if not quiet:
        dataloader = tqdm(dataloader)
    for clean, corrupted, label in dataloader:
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

        with torch.inference_mode():
            corrupted_logits = model(corrupted_tokens, attention_mask=corrupted_attention_mask)
            logits = model(clean_tokens, attention_mask=clean_attention_mask)
        for i, metric in enumerate(metrics):
            if run_corrupted:
                r = metric(corrupted_logits, logits, input_lengths, label).cpu()
            else:
                r = metric(logits, corrupted_logits, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results


def evaluate_ibcircuit_neuron_circuit(  # noqa: C901 - complex function, refactor out of scope for lint pass
    model: HookedTransformer,
    pruning_dict: dict,
    dataloader: DataLoader,
    metrics,
    quiet: bool = False,
    intervention: Literal["zero", "mean", "patching"] = "mean",
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate an IBCircuit neuron-level circuit by ablating pruned neurons.

    Pruned neurons are replaced with either their position-specific dataset mean,
    zeros, or the corresponding corrupted activations, depending on the intervention.
    Mean ablation matches the IB training objective (ib_noise.py computes statistics
    over dim=0, preserving the position dimension), so 'mean' is the recommended
    intervention for faithful evaluation of IBCircuit discoveries.

    A two-phase approach is used for mean ablation:
        Phase 1: One full pass over the dataset to compute position-specific means
                 per pruned layer, with masking to exclude padding positions.
        Phase 2: Evaluation pass with static ablation hooks built from those means.

    For 'patching', corrupted activations are cached per-batch before the clean
    forward pass, and pruned neurons are overwritten with their corrupted values.

    Args:
        model (HookedTransformer): Model to evaluate.
        pruning_dict (dict): Neuron pruning specification with keys:
            - 'mlp'   (Dict[int, List[int]]): {layer: [neuron_indices]} for MLP layers.
            - 'heads' (Dict[Tuple[int,int], List[int]]): {(layer, head): [neuron_indices]}
              for attention heads. Uses hook_z space (d_head neurons per head).
            - '_meta' (dict): Optional metadata. Recognised keys:
                - 'mlp_hook' ('mlp_out' | 'post_act'): which MLP hook to target.
                - 'heads_hook': attention hook name (default: 'attn.hook_z').
        dataloader (DataLoader): Evaluation dataset yielding (clean, corrupted, label) batches.
        metrics (Union[Callable, List[Callable]]): Metric function(s) with signature
            (logits, clean_logits, input_lengths, labels) -> Tensor [batch].
            clean_logits will be None; metrics must handle this.
        quiet (bool): Suppress tqdm progress bar. Defaults to False.
        intervention (str): Ablation method for pruned neurons. One of:
            - 'mean': Replace with position-specific dataset mean (recommended).
            - 'zero': Replace with zeros.
            - 'patching': Replace with corrupted activations per example.
            Defaults to 'mean'.

    Returns:
        Union[Tensor, List[Tensor]]: Per-sample metric scores [n_samples], or a list
            of such tensors if multiple metrics were provided.

    Note:
        The mean computation is mask-aware and handles both right-padded (e.g. IOI)
        and left-padded (e.g. MMLU) batches correctly, preventing padding positions
        from contaminating the ablation values at answer positions.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    pair_padding_side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    device = model.cfg.device

    mlp_hook_name = (
        "mlp.hook_post"
        if pruning_dict.get("_meta", {}).get("mlp_hook") == "post_act"
        else "hook_mlp_out"
    )
    heads_hook_name = pruning_dict.get("_meta", {}).get("heads_hook", "attn.hook_z")

    # Build per-layer lookup of which neurons to ablate
    attn_by_layer = defaultdict(dict)  # {layer: {head: Tensor[neuron_indices]}}
    for (layer, head), neuron_indices in pruning_dict.get("heads", {}).items():
        if neuron_indices:
            attn_by_layer[layer][head] = torch.tensor(neuron_indices, device=device)

    mlp_neurons = {
        layer: torch.tensor(indices, device=device)
        for layer, indices in pruning_dict.get("mlp", {}).items()
        if indices
    }

    # ── Phase 1: Compute position-specific dataset means over clean inputs ────
    if intervention == "mean":  # skipped for zero and patching
        # Uses mask-aware weighted sum+count per position to correctly handle:
        #   - Variable-length batches (different pos across batches)
        #   - Right-padded tasks (capital_country): padding at tail
        #   - Left-padded tasks (MMLU): padding at head
        # Without masking, positions where only some sequences have real content
        # get contaminated means, which corrupt ablations at answer positions.
        attn_sum = {}  # {layer: Tensor[max_pos, n_heads, d_head/d_model], float32}
        attn_count = {}  # {layer: Tensor[max_pos], float32}
        mlp_sum = {}  # {layer: Tensor[max_pos, d_model/d_mlp], float32}
        mlp_count = {}  # {layer: Tensor[max_pos], float32}

        def _accumulate(store_sum, store_cnt, key, w_sum, w_cnt):
            """Add weighted sum+count, growing stored tensors if this batch is longer."""
            p = w_sum.shape[0]
            if key not in store_sum:
                store_sum[key] = w_sum.clone()
                store_cnt[key] = w_cnt.clone()
            else:
                cur = store_sum[key].shape[0]
                if p > cur:
                    dev = store_sum[key].device
                    store_sum[key] = torch.cat(
                        [
                            store_sum[key],
                            torch.zeros(
                                p - cur, *store_sum[key].shape[1:], device=dev, dtype=torch.float32
                            ),
                        ]
                    )
                    store_cnt[key] = torch.cat(
                        [store_cnt[key], torch.zeros(p - cur, device=dev, dtype=torch.float32)]
                    )
                store_sum[key][:p] += w_sum
                store_cnt[key][:p] += w_cnt

        with torch.inference_mode():
            for clean, corrupted, label in dataloader:
                clean_tokens, _, clean_attn_mask, _, _, _ = tokenize_batch_pair(
                    model,
                    clean,
                    corrupted,
                    pair_padding_side=pair_padding_side,
                    templated=templated,
                )
                # Real-token mask [batch, pos]: 1 = real, 0 = padding.
                # None means all tokens are real (uniform-length batch, e.g. IOI).
                if clean_attn_mask is not None:
                    pos_mask = clean_attn_mask.float().to(device)
                else:
                    pos_mask = torch.ones(
                        clean_tokens.shape[0],
                        clean_tokens.shape[1],
                        device=device,
                        dtype=torch.float32,
                    )

                # Build hooks fresh each batch so pos_mask is captured in closure.
                batch_hooks = []

                for layer in attn_by_layer:

                    def make_attn_accum_hook(lyr, m):
                        def h(z, hook):  # z: [batch, pos, n_heads, d_head]
                            p = z.shape[1]
                            mask = m[:, :p, None, None]  # [batch, pos, 1, 1]
                            w_sum = (z.detach().float() * mask).sum(dim=0)  # [pos, n_heads, d_head]
                            w_cnt = mask[:, :, 0, 0].sum(dim=0)  # [pos]
                            _accumulate(attn_sum, attn_count, lyr, w_sum, w_cnt)

                        return h

                    batch_hooks.append(
                        (f"blocks.{layer}.{heads_hook_name}", make_attn_accum_hook(layer, pos_mask))
                    )

                for layer in mlp_neurons:

                    def make_mlp_accum_hook(lyr, m):
                        def h(act, hook):  # act: [batch, pos, d_model or d_mlp]
                            p = act.shape[1]
                            mask = m[:, :p, None]  # [batch, pos, 1]
                            w_sum = (act.detach().float() * mask).sum(
                                dim=0
                            )  # [pos, d_model or d_mlp]
                            w_cnt = mask[:, :, 0].sum(dim=0)  # [pos]
                            _accumulate(mlp_sum, mlp_count, lyr, w_sum, w_cnt)

                        return h

                    batch_hooks.append(
                        (f"blocks.{layer}.{mlp_hook_name}", make_mlp_accum_hook(layer, pos_mask))
                    )

                with model.hooks(batch_hooks):
                    model(clean_tokens, attention_mask=clean_attn_mask)

        # mean = weighted_sum / count; clamp(1) safe-divides positions with zero
        # real-token count (pure padding across whole dataset → ablation value 0,
        # harmless since those positions are always masked in attention).
        attn_means = {
            lyr: (attn_sum[lyr] / attn_count[lyr].clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)).to(
                device=device, dtype=model.cfg.dtype
            )
            for lyr in attn_sum
        }  # {layer: Tensor[max_pos, n_heads, d_head]}
        mlp_means = {
            lyr: (mlp_sum[lyr] / mlp_count[lyr].clamp(min=1.0).unsqueeze(-1)).to(
                device=device, dtype=model.cfg.dtype
            )
            for lyr in mlp_sum
        }  # {layer: Tensor[max_pos, d_model or d_mlp]}
    else:
        attn_means = {}
        mlp_means = {}

    # ── Phase 2: Build static mean-ablation hooks ─────────────────────────────
    # Hooks are built once from precomputed means and reused across all batches.
    fwd_hooks = []

    for layer, heads_dict in attn_by_layer.items():
        mean_z = attn_means.get(layer)  # [max_pos, n_heads, d_head] or None for zero ablation

        def make_attn_hook(hd, mean_layer):
            def hook_fn(z, hook):  # z: [batch, pos, n_heads, d_head]
                if mean_layer is None or intervention == "zero":
                    for head_idx, neuron_idx in hd.items():
                        z[:, :, head_idx, neuron_idx] = 0
                    return z
                actual_pos = min(z.shape[1], mean_layer.shape[0])
                for head_idx, neuron_idx in hd.items():
                    z[:, :actual_pos, head_idx, neuron_idx] = mean_layer[
                        :actual_pos, head_idx, neuron_idx
                    ]
                return z

            return hook_fn

        fwd_hooks.append((f"blocks.{layer}.{heads_hook_name}", make_attn_hook(heads_dict, mean_z)))

    for layer, neuron_idx in mlp_neurons.items():
        mean_act = mlp_means.get(layer)  # [max_pos, d_model or d_mlp] or None for zero ablation

        def make_mlp_hook(ni, mean_layer):
            def hook_fn(act, hook):  # act: [batch, pos, d_model or d_mlp]
                if mean_layer is None or intervention == "zero":
                    act[:, :, ni] = 0
                    return act
                actual_pos = min(act.shape[1], mean_layer.shape[0])
                act[:, :actual_pos, ni] = mean_layer[:actual_pos, ni]
                return act

            return hook_fn

        fwd_hooks.append((f"blocks.{layer}.{mlp_hook_name}", make_mlp_hook(neuron_idx, mean_act)))

    # ── Phase 3: Evaluate ────────────────────────────────────────────────────
    results = [[] for _ in metrics]
    dl = dataloader if quiet else tqdm(dataloader)

    for clean, corrupted, label in dl:
        clean_tokens, corrupted_tokens, clean_attn_mask, corrupted_attn_mask, input_lengths, _ = (
            tokenize_batch_pair(
                model, clean, corrupted, pair_padding_side=pair_padding_side, templated=templated
            )
        )

        if intervention == "patching":
            # Cache corrupted activations at pruned hook points
            corrupt_cache = {}
            cache_hooks = []
            for layer in mlp_neurons:

                def make_mlp_cache(lyr):
                    def h(act, hook):
                        corrupt_cache[("mlp", lyr)] = act.detach().clone()

                    return h

                cache_hooks.append((f"blocks.{layer}.{mlp_hook_name}", make_mlp_cache(layer)))
            for layer in attn_by_layer:

                def make_attn_cache(lyr):
                    def h(z, hook):
                        corrupt_cache[("attn", lyr)] = z.detach().clone()

                    return h

                cache_hooks.append((f"blocks.{layer}.{heads_hook_name}", make_attn_cache(layer)))
            with torch.inference_mode():
                with model.hooks(cache_hooks):
                    model(corrupted_tokens, attention_mask=corrupted_attn_mask)

            # Build per-batch patch hooks replacing pruned neurons with corrupted activations
            batch_hooks = []
            for layer, neuron_idx in mlp_neurons.items():

                def make_mlp_patch(lyr, ni):
                    def h(act, hook):
                        corrupt = corrupt_cache[("mlp", lyr)]
                        p = min(act.shape[1], corrupt.shape[1])
                        act[:, :p, ni] = corrupt[:, :p, ni]
                        return act

                    return h

                batch_hooks.append(
                    (f"blocks.{layer}.{mlp_hook_name}", make_mlp_patch(layer, neuron_idx))
                )
            for layer, heads_dict in attn_by_layer.items():

                def make_attn_patch(lyr, hd):
                    def h(z, hook):
                        corrupt = corrupt_cache[("attn", lyr)]
                        p = min(z.shape[1], corrupt.shape[1])
                        for head_idx, neuron_idx in hd.items():
                            z[:, :p, head_idx, neuron_idx] = corrupt[:, :p, head_idx, neuron_idx]
                        return z

                    return h

                batch_hooks.append(
                    (f"blocks.{layer}.{heads_hook_name}", make_attn_patch(layer, heads_dict))
                )

            with torch.inference_mode():
                with model.hooks(batch_hooks):
                    logits = model(clean_tokens, attention_mask=clean_attn_mask)
        else:
            with torch.inference_mode():
                with model.hooks(fwd_hooks):
                    logits = model(clean_tokens, attention_mask=clean_attn_mask)

        for i, metric in enumerate(metrics):
            r = metric(logits, None, input_lengths, label).cpu()
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    if len(results) == 1:
        results = results[0]
    return results
