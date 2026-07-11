import logging
from functools import partial
from typing import List, Optional, Union

import torch
from einops import einsum
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask

from .graph import AttentionNode, Graph, LogitNode

logger = logging.getLogger(__name__)


def collate_EAP(xs):
    """
    Collate a list of EAP dataset samples into a batch.

    Args:
        xs (List[Tuple]): List of (clean, corrupted, label) samples from an EAP-style dataset.

    Returns:
        Tuple[List[str], List[str], Tensor]: A tuple of
            (clean_texts, corrupted_texts, labels) where labels has shape [batch_size]
            or [batch_size, n_labels] for multi-answer tasks (e.g. MMLU).
    """
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels


def collate_EAP_with_spans(xs):
    """
    Collate a list of EAP dataset samples into a batch with support for answer spans.

    Handles both traditional 3-tuple format and new 4-tuple format with answer spans.
    Backward compatible with existing code that returns 3-tuples.

    Args:
        xs (List[Tuple]): List of (clean, corrupted, label, answer_span) or
            (clean, corrupted, label) samples from an EAP-style dataset.

    Returns:
        Tuple: Either:
            - (clean_texts, corrupted_texts, labels, answer_spans) if any sample has spans
            - (clean_texts, corrupted_texts, labels) if no spans (backward compatible)
        Where:
            - clean_texts: List[str]
            - corrupted_texts: List[str]
            - labels: Tensor [batch_size] or [batch_size, n_labels]
            - answer_spans: List[Optional[Tuple[int, int]]] or None
    """
    clean_texts = []
    corrupted_texts = []
    labels_list = []
    answer_spans_list = []
    has_spans = False

    for item in xs:
        if len(item) == 4:
            clean, corrupted, labels, answer_span = item
            has_spans = True
        else:
            # Backward compatibility: 3-tuple without answer_span
            clean, corrupted, labels = item
            answer_span = None

        clean_texts.append(clean)
        corrupted_texts.append(corrupted)
        labels_list.append(labels)
        answer_spans_list.append(answer_span)

    labels_tensor = torch.tensor(labels_list)

    if has_spans:
        return clean_texts, corrupted_texts, labels_tensor, answer_spans_list
    else:
        return clean_texts, corrupted_texts, labels_tensor


def tokenize_plus(
    model: HookedTransformer,
    inputs: List[str],
    max_length: Optional[int] = None,
    padding_side: Optional[str] = None,
    templated: bool = False,
):
    """
    Tokenize a list of strings with attention mask and input length computation.

    Handles both left- and right-padded batches. For left-padded batches,
    input_lengths is set to n_pos uniformly so that (input_length - 1) correctly
    indexes the last real token for every sample.

    Args:
        model (HookedTransformer): Model whose tokenizer is used.
        inputs (List[str]): Raw text strings to tokenize.
        max_length (Optional[int]): If set, truncates sequences to this length
            and temporarily overrides model.cfg.n_ctx. Defaults to None.
        padding_side (Optional[str]): Override padding side ('left' or 'right').
            Defaults to model.tokenizer.padding_side.
        templated (bool): True iff ``inputs`` already carry a chat template,
            which renders its own beginning-of-text token into the string.
            Tokenizing such text with ``prepend_bos=True`` would inject a
            *second* BOS, shifting every position and corrupting attribution,
            so BOS is prepended iff ``not templated``. Defaults to False (raw
            text), which keeps ``prepend_bos=True`` — byte-identical to the
            legacy behavior for every non-templated task and base-model run.

    Returns:
        Tuple:
            - tokens (Tensor): Token IDs [batch, n_pos].
            - attention_mask (Tensor): Binary mask [batch, n_pos], 1 = real token.
            - input_lengths (Tensor): Number of real tokens per sample [batch].
                For left-padded batches, uniformly set to n_pos.
            - n_pos (int): Sequence length after padding.
    """
    effective_padding_side = padding_side or model.tokenizer.padding_side
    if max_length is not None:
        old_n_ctx = model.cfg.n_ctx
        model.cfg.n_ctx = max_length
    tokens = model.to_tokens(
        inputs,
        prepend_bos=not templated,
        padding_side=effective_padding_side,
        truncate=(max_length is not None),
    )
    if max_length is not None:
        model.cfg.n_ctx = old_n_ctx
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    # For left-padded batches all real tokens end at position n_pos-1.
    # Set input_lengths = n_pos so (input_length - 1) correctly indexes
    # the last real token for every sample, consistent with right-padded behaviour.
    if effective_padding_side == "left":
        input_lengths = torch.full_like(input_lengths, n_pos)
    return tokens, attention_mask, input_lengths, n_pos


def tokenize_batch_pair(
    model: HookedTransformer,
    clean: list[str],
    corrupted: list[str],
    pair_padding_side: Optional[str] = None,
    templated: bool = False,
):
    """
    Tokenize a clean/corrupted pair and align them to a shared sequence length.

    Each pair is tokenized independently, then cross-padded so both have the
    same n_pos. This is required because EAP computes activation differences
    between clean and corrupted inputs and they must share positional indices.
    Padding direction is controlled by pair_padding_side so that answer tokens
    remain at consistent positions across both sequences.

    Args:
        model (HookedTransformer): Model whose tokenizer is used.
        clean (List[str]): Clean input strings [batch_size].
        corrupted (List[str]): Corrupted input strings [batch_size].
        pair_padding_side (Optional[str]): Padding side for both within-batch
            and cross-pair alignment ('left' or 'right').
            Defaults to model.tokenizer.padding_side.
        templated (bool): True iff ``clean`` / ``corrupted`` already carry a
            chat template. Forwarded to :func:`tokenize_plus` so BOS is
            prepended iff ``not templated``, avoiding a double BOS on
            chat-templated text. Defaults to False (raw text), byte-identical
            to the legacy behavior.

    Returns:
        Tuple:
            - clean_tokens (Tensor): Clean token IDs [batch, n_pos].
            - corr_tokens (Tensor): Corrupted token IDs [batch, n_pos].
            - clean_mask (Tensor): Clean attention mask [batch, n_pos].
            - corr_mask (Tensor): Corrupted attention mask [batch, n_pos].
            - clean_lengths (Tensor): Real token counts for clean inputs [batch].
                Adjusted for left-padding prepend when sequences are cross-aligned.
            - n_pos (int): Shared sequence length after cross-alignment.
    """
    side = pair_padding_side or model.tokenizer.padding_side
    # Use the same padding side for within-batch and cross-pair alignment
    clean_tokens, clean_mask, clean_lengths, max_clean = tokenize_plus(
        model, clean, padding_side=side, templated=templated
    )
    corr_tokens, corr_mask, corr_lengths, max_corr = tokenize_plus(
        model, corrupted, padding_side=side, templated=templated
    )

    # Per-pair length check. EAP integrates per-position (corrupted - clean)
    # activation differences, so a pair whose clean/corrupt token counts differ
    # is only correctly aligned when the extra tokens fall at the padded end (a
    # shared prefix/suffix). A length change from a mid-sequence corruption
    # (inserted distractor, multi-token entity/role swap, respacing) shifts every
    # later position, and the end-padding below cannot restore correspondence.
    # Surface that loudly instead of silently integrating misaligned activations.
    # Logging only — the alignment/padding behaviour is unchanged.
    #
    # Throttle: warn on the 1st misaligned batch and then every 100th, with a
    # running total. (A previous permanent once-per-process latch meant the
    # first misaligned dataset silenced the warning for every later dataset in
    # the same process — e.g. a clean smoke run followed by a misaligned custom
    # dataset, or Pillar 6 evaluating several tasks back to back.)
    n_misaligned = int((clean_lengths != corr_lengths).sum().item())
    if n_misaligned:
        batches = getattr(tokenize_batch_pair, "_misaligned_batches", 0) + 1
        total = getattr(tokenize_batch_pair, "_misaligned_pairs", 0) + n_misaligned
        tokenize_batch_pair._misaligned_batches = batches
        tokenize_batch_pair._misaligned_pairs = total
        if batches % 100 == 1:
            logger.warning(
                "tokenize_batch_pair: %d of %d contrastive pair(s) have unequal clean/corrupt "
                "token counts (clean max=%d, corrupt max=%d; %d misaligned pair(s) across %d "
                "batch(es) so far this process). EAP aligns by end-padding "
                "(pair_padding_side='%s'), which preserves per-position correspondence ONLY when "
                "the length difference is at that end. Mid-sequence length-changing corruptions "
                "will misalign and the attribution for those pairs is unreliable — prefer minimal, "
                "token-length-preserving contrastive pairs. (Warning repeats every 100th "
                "misaligned batch.)",
                n_misaligned,
                len(clean),
                max_clean,
                max_corr,
                total,
                batches,
                side,
            )

    # Step 2: cross-align clean and corrupted to shared n_pos using task-controlled side
    side = pair_padding_side or model.tokenizer.padding_side
    pad_id = model.tokenizer.pad_token_id or 0

    if max_clean < max_corr:
        K = max_corr - max_clean
        pad = torch.full(
            (len(clean), K), pad_id, dtype=clean_tokens.dtype, device=clean_tokens.device
        )
        pad_mask = torch.zeros(len(clean), K, dtype=clean_mask.dtype, device=clean_mask.device)
        if side == "left":
            clean_tokens = torch.cat([pad, clean_tokens], dim=1)
            clean_mask = torch.cat([pad_mask, clean_mask], dim=1)
            clean_lengths = clean_lengths + K  # shift answer positions to reflect left-prepend
        else:
            clean_tokens = torch.cat([clean_tokens, pad], dim=1)
            clean_mask = torch.cat([clean_mask, pad_mask], dim=1)
            # clean_lengths unchanged: answer still at L_i - 1

    elif max_corr < max_clean:
        K = max_clean - max_corr
        pad = torch.full(
            (len(corrupted), K), pad_id, dtype=corr_tokens.dtype, device=corr_tokens.device
        )
        pad_mask = torch.zeros(len(corrupted), K, dtype=corr_mask.dtype, device=corr_mask.device)
        if side == "left":
            corr_tokens = torch.cat([pad, corr_tokens], dim=1)
            corr_mask = torch.cat([pad_mask, corr_mask], dim=1)
        else:
            corr_tokens = torch.cat([corr_tokens, pad], dim=1)
            corr_mask = torch.cat([corr_mask, pad_mask], dim=1)
        # corr_lengths not used by metric, no update needed

    n_pos = clean_tokens.shape[1]

    if not getattr(tokenize_batch_pair, "_pad_verified", False):
        tokenize_batch_pair._pad_verified = True
        assert (
            clean_tokens.shape[1] == corr_tokens.shape[1]
        ), f"SHAPE MISMATCH: clean={clean_tokens.shape[1]} corr={corr_tokens.shape[1]}"
        logger.debug(f"PADDING verified: pair_padding_side='{side}' n_pos={n_pos}")
        logger.debug(f"  clean[0] tail : {model.to_str_tokens(clean_tokens[0, -6:])}")
        logger.debug(f"  corr [0] tail : {model.to_str_tokens(corr_tokens[0, -6:])}")

    return clean_tokens, corr_tokens, clean_mask, corr_mask, clean_lengths, n_pos


def make_hooks_and_matrices(
    model: HookedTransformer, graph: Graph, batch_size: int, n_pos: int, scores: Optional[Tensor]
):
    """
    Build the activation-difference matrix and associated forward/backward hooks for EAP attribution.

    The activation_difference matrix accumulates (corrupted - clean) activations per
    node. Forward hooks populate it; backward hooks use it alongside gradients to
    update the edge scores tensor in-place.

    For models with post-attention LayerNorm (e.g. Gemma), activations are stored
    in two separate halves [2, batch, pos, n_forward, d_model] instead of being
    subtracted in-place, as the clean activations must be passed through LayerNorm
    before differencing.

    Args:
        model (HookedTransformer): Model being attributed.
        graph (Graph): Graph defining node structure and indexing.
        batch_size (int): Number of examples in the current batch.
        n_pos (int): Sequence length (number of token positions).
        scores (Optional[Tensor]): Edge scores tensor [n_forward, n_backward] to
            update in-place via backward hooks. Pass None for evaluation-only use
            (backward hooks are still constructed but should not be used).

    Returns:
        Tuple:
            - hooks (Tuple[List, List, List]): Three hook lists:
                (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks).
                Run fwd_hooks_corrupted on corrupted input to add activations.
                Run fwd_hooks_clean on clean input to subtract activations.
                Run bwd_hooks during the backward pass to accumulate scores.
            - activation_difference (Tensor): Buffer storing activation differences.
                Shape [batch, pos, n_forward, d_model], or
                [2, batch, pos, n_forward, d_model] for post-attn-LN models.
    """
    separate_activations = model.cfg.use_normalization_before_and_after and scores is None
    _d_model = model.cfg.d_model
    _d_mlp = graph.cfg.get("d_mlp", getattr(model.cfg, "d_mlp", _d_model))
    max_d = max(_d_model, _d_mlp) if graph.cfg.get("mlp_hook") == "post_act" else _d_model
    if separate_activations:
        activation_difference = torch.zeros(
            (2, batch_size, n_pos, graph.n_forward, max_d),
            device=model.cfg.device,
            dtype=model.cfg.dtype,
        )
    else:
        activation_difference = torch.zeros(
            (batch_size, n_pos, graph.n_forward, max_d),
            device=model.cfg.device,
            dtype=model.cfg.dtype,
        )

    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    # Fills up the activation difference matrix. In the default case (not separate_activations),
    # we add in the corrupted activations (add = True) and subtract out the clean ones (add=False)
    # In the separate_activations case, we just store them in two halves of the matrix. Less efficient,
    # but necessary for models with Gemma's architecture.
    def activation_hook(index, activations, hook, add: bool = True):
        acts = activations.detach()
        act_d = acts.shape[-1]
        try:
            if separate_activations:
                if add:
                    activation_difference[0, :, :, index, :act_d] += acts
                else:
                    activation_difference[1, :, :, index, :act_d] += acts
            else:
                if add:
                    activation_difference[:, :, index, :act_d] += acts
                else:
                    activation_difference[:, :, index, :act_d] -= acts
        except RuntimeError as e:
            logger.info(
                "%s %s %s", hook.name, activation_difference[:, :, index].size(), acts.size()
            )
            raise e

    def gradient_hook(prev_index: int, bwd_index: Union[slice, int], gradients: torch.Tensor, hook):
        """
        Accumulate edge scores using the stored activation differences and incoming gradients.

        Args:
            prev_index (int): Forward index up to which previous nodes contribute
                to this destination node's input (exclusive upper bound).
            bwd_index (Union[slice, int]): Backward index of the destination node
                in the scores tensor.
            gradients (torch.Tensor): Gradient of the metric w.r.t. this node's input.
            hook: HookedTransformer hook object (unused).
        """
        grads = gradients.detach()
        try:
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            grad_d = grads.shape[-1]
            s = einsum(
                activation_difference[:, :, :prev_index, :grad_d],
                grads,
                "batch pos forward hidden, batch pos backward hidden -> forward backward",
            )
            s = s.squeeze(1)
            scores[:prev_index, bwd_index] += s
        except RuntimeError as e:
            logger.info(
                "%s %s %s %s %s",
                hook.name,
                activation_difference.size(),
                activation_difference.device,
                grads.size(),
                grads.device,
            )
            logger.info("%s %s %s", prev_index, bwd_index, scores.size())
            raise e

    node = graph.nodes["input"]
    fwd_index = graph.forward_index(node)
    fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
    fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))

    for layer in range(graph.cfg["n_layers"]):
        node = graph.nodes[f"a{layer}.h0"]
        fwd_index = graph.forward_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        prev_index = graph.prev_index(node)
        for i, letter in enumerate("qkv"):
            bwd_index = graph.backward_index(node, qkv=letter)
            bwd_hooks.append((node.qkv_inputs[i], partial(gradient_hook, prev_index, bwd_index)))

        node = graph.nodes[f"m{layer}"]
        fwd_index = graph.forward_index(node)
        bwd_index = graph.backward_index(node)
        prev_index = graph.prev_index(node)
        fwd_hooks_corrupted.append((node.out_hook, partial(activation_hook, fwd_index)))
        fwd_hooks_clean.append((node.out_hook, partial(activation_hook, fwd_index, add=False)))
        bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    node = graph.nodes["logits"]
    prev_index = graph.prev_index(node)
    bwd_index = graph.backward_index(node)
    bwd_hooks.append((node.in_hook, partial(gradient_hook, prev_index, bwd_index)))

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def compute_mean_activations(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    per_position=False,
    padding_side: Optional[str] = None,
    templated: bool = False,
    mean_weighting: str = "token",
):
    """
    Compute mean activations for each graph node over a dataset.

    Used to construct mean-ablation baselines. One forward pass is run per batch
    with no gradient tracking. Attention layers are deduplicated so only one hook
    is registered per layer regardless of head count.

    Args:
        model (HookedTransformer): Model to run.
        graph (Graph): Graph whose node activations are collected.
        dataloader (DataLoader): Dataset to average over. Batches may be tuples
            (e.g. EAP format) or raw lists of strings.
        per_position (bool): If True, returns position-resolved means without
            collapsing across the sequence dimension. If False, returns a single
            mean vector per node averaged over all positions and examples.
            Defaults to False.
        padding_side (Optional[str]): Padding side passed to tokenize_plus.
            Defaults to model.tokenizer.padding_side.
        templated (bool): True iff the dataset's strings already carry a chat
            template (so BOS must not be re-prepended). Forwarded to
            tokenize_plus. Defaults to False (raw text), byte-identical to the
            legacy behavior.

    Returns:
        Tensor: Mean activations per node.
            - per_position=False: [n_forward, d_model], mean over all positions and examples.
            - per_position=True:  [n_pos, n_forward, d_model], mean per position.

    Note:
        Sequences are truncated to max_length=512 to bound memory.
        The logits node is excluded as it has no meaningful output activation.
    """

    def activation_hook(
        index, activations, hook, means=None, input_lengths=None, attention_mask=None
    ):
        # defining a hook that will fill up our means tensor. Means is of shape
        # (n_pos, graph.n_forward, model.cfg.d_model) if per_position is True, otherwise
        # (graph.n_forward, model.cfg.d_model)
        acts = activations.detach()

        # if you gave this hook input lengths, we mean over ALL real positions
        if input_lengths is not None:
            # Sum activations over all real (non-padding) positions per example. The
            # mask is the attention mask (1 at each real token), broadcast over the
            # optional head dim and the hidden dim. (Previously the mask selected only
            # the last token — input_lengths - 1 — then divided by the length,
            # computing (last-token activation)/seq_len rather than a mean over positions.)
            mask = attention_mask.to(dtype=acts.dtype, device=acts.device)
            while mask.dim() < acts.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(acts)

            # we need ... because there might be a head index as well
            item_sums = einsum(
                acts, mask, "batch pos ... hidden, batch pos ... hidden -> batch ... hidden"
            )

            if mean_weighting == "example":
                # Mean of per-example position-means: normalize each example by its own
                # real-token count (correct for both left- and right-padding); the
                # final divide-by-n_examples then weights each example equally.
                counts = attention_mask.to(dtype=item_sums.dtype, device=item_sums.device).sum(1)
                while counts.dim() < item_sums.dim():
                    counts = counts.unsqueeze(-1)
                means[index] += (item_sums / counts).sum(0)
            else:
                # Token-weighted grand mean ("token"): accumulate raw sums; the final
                # divide-by-total_real_tokens averages over ALL positions and examples.
                means[index] += item_sums.sum(0)
        else:
            means[:, index] += acts.sum(0)

    # we're going to get all of the out hooks / indices we need for making hooks
    # but we can't make them until we have input length masks
    processed_attn_layers = set()
    hook_points_indices = []
    for node in graph.nodes.values():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            processed_attn_layers.add(node.layer)

        if not isinstance(node, LogitNode):
            hook_points_indices.append((node.out_hook, graph.forward_index(node)))

    if mean_weighting not in ("token", "example"):
        raise ValueError(f"mean_weighting must be 'token' or 'example', got {mean_weighting!r}")
    means_initialized = False
    total = 0
    total_tokens = 0
    for batch in tqdm(dataloader, desc="Computing mean"):
        # maybe the dataset is given as a tuple, maybe its just raw strings
        batch_inputs = batch[0] if isinstance(batch, tuple) else batch
        tokens, attention_mask, input_lengths, n_pos = tokenize_plus(
            model,
            batch_inputs,
            max_length=512,
            padding_side=padding_side,
            templated=templated,
        )
        total += len(batch_inputs)
        total_tokens += int(attention_mask.sum().item())

        if not means_initialized:
            # here is where we store the means
            if per_position:
                means = torch.zeros(
                    (n_pos, graph.n_forward, model.cfg.d_model),
                    device=model.cfg.device,
                    dtype=model.cfg.dtype,
                )
            else:
                means = torch.zeros(
                    (graph.n_forward, model.cfg.d_model),
                    device=model.cfg.device,
                    dtype=model.cfg.dtype,
                )
            means_initialized = True

        if per_position:
            input_lengths = None
        add_to_mean_hooks = [
            (
                hook_point,
                partial(
                    activation_hook,
                    index,
                    means=means,
                    input_lengths=input_lengths,
                    attention_mask=attention_mask,
                ),
            )
            for hook_point, index in hook_points_indices
        ]

        with model.hooks(fwd_hooks=add_to_mean_hooks):
            model(tokens, attention_mask=attention_mask)

    # Added condition to prevent squeezing the position dimension when per_position=True
    if not per_position:
        means = means.squeeze(0)
    # per_position and example-weighting divide by the example count; token-weighting
    # divides by the total number of real tokens (mean over all positions and examples).
    denom = total if (per_position or mean_weighting == "example") else total_tokens
    means /= denom
    return means if per_position else means.mean(0)
