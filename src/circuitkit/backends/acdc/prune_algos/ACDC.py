import logging
from itertools import product
from typing import List, Literal, Optional, Set

import torch as t
from ordered_set import OrderedSet
from torch.nn.functional import log_softmax, mse_loss

from ..data import PromptDataLoader
from ..types import Edge, PruneScores
from ..utils.ablation_activations import src_ablations
from ..utils.custom_tqdm import tqdm
from ..utils.graph_utils import patch_mode, set_all_masks
from ..utils.patchable_model import PatchableModel
from ..utils.tensor_ops import multibatch_kl_div

logger = logging.getLogger(__name__)


def acdc_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
    tao_exps: List[int] = list(range(-5, -1)),
    tao_bases: List[int] = [1, 5],
    faithfulness_target: Literal["kl_div", "mse"] = "kl_div",
    verbose: bool = False,
) -> PruneScores:
    """
    Run the ACDC algorithm from "Towards Automated Circuit Discovery for
    Mechanistic Interpretability" ([Conmy et al. (2023)](https://arxiv.org/abs/2304.14997)).

    The algorithm doesn't assign continuous scores but finds a set of edges to prune for a
    given threshold (tao). We run the algorithm for several tao values (each combination
    of `tao_exps` and `tao_bases`) and assign scores based on the smallest tao value for
    which an edge is considered unimportant. Edges pruned with smaller taos (stricter
    thresholds) are less important.

    Args:
        model: The model to find the circuit for.
        dataloader: The dataloader to use for input and patches. Only the first batch
            is used.
        official_edges: Not used in this implementation.
        tao_exps: The exponents to use for the set of tao values.
        tao_bases: The bases to use for the set of tao values.
        faithfulness_target: The faithfulness metric to optimize the circuit for.
        verbose: Show tqdm progress bars. When False (default) progress is emitted
            as DEBUG log messages on the ``circuitkit.backends.acdc`` logger instead.
            Enable with ``discovery_cfg["verbose"] = True`` or by setting the logger
            level to DEBUG directly.

    Returns:
        An ordering of the edges by importance. Higher scores are more important. Edges
        not part of any discovered circuit are not included in the scores.
    """
    out_slice = model.out_slice
    edges: OrderedSet[Edge] = OrderedSet(
        sorted(model.edges, key=lambda x: x.dest.layer, reverse=True)
    )
    n_edges = len(edges)

    prune_scores = model.new_prune_scores(init_val=t.inf)
    tao_vals = sorted([a * 10**b for a, b in product(tao_bases, tao_exps)])
    n_sweeps = len(tao_vals)

    logger.debug("ACDC: %d edges, %d τ sweeps", n_edges, n_sweeps)

    for sweep_idx, tao in enumerate(tqdm(tao_vals, disable=not verbose)):
        logger.debug("ACDC sweep %d/%d  τ=%.2e", sweep_idx + 1, n_sweeps, tao)

        train_batch = next(iter(dataloader))
        clean_batch, corrupt_batch = train_batch.clean, train_batch.corrupt

        patch_outs_tensor = src_ablations(model, corrupt_batch)
        src_outs_tensor = src_ablations(model, clean_batch)

        with t.inference_mode():
            clean_out = model(clean_batch)[out_slice]
            resids = []
            if model.is_transformer:
                _, cache = model.run_with_cache(clean_batch)
                n_layers = range(model.cfg.n_layers)
                resids = [cache[f"blocks.{i}.hook_resid_pre"].clone() for i in n_layers]
                del cache

        clean_logprobs = t.nn.functional.log_softmax(clean_out, dim=-1)

        prev_faith = 0.0
        removed_edges: OrderedSet[Edge] = OrderedSet([])

        set_all_masks(model, val=0.0)
        _log_every = max(1, n_edges // 4)
        # We set curr_src_outs manually so we can skip layers before the current edge.
        with patch_mode(model, patch_outs_tensor, curr_src_outs=src_outs_tensor):
            for edge_idx, edge in enumerate(
                tqdm(edges, disable=not verbose, mininterval=5.0)
            ):
                if edge_idx % _log_every == 0:
                    logger.debug(
                        "  edge %d/%d  removed=%d",
                        edge_idx, n_edges, len(removed_edges),
                    )

                edge.patch_mask(model).data[edge.patch_idx] = 1.0
                with t.inference_mode():
                    if model.is_transformer:
                        start_layer = int(edge.dest.module_name.split(".")[1])
                        out = model(resids[start_layer], start_at_layer=start_layer)[out_slice]
                    else:
                        out = model(clean_batch)[out_slice]

                if faithfulness_target == "kl_div":
                    out_logprobs = log_softmax(out, dim=-1)
                    faith = multibatch_kl_div(out_logprobs, clean_logprobs).item()
                elif faithfulness_target == "mse":
                    faith = mse_loss(out, clean_out).item()

                if faith - prev_faith < tao:  # Edge is unimportant
                    removed_edges.add(edge)
                    curr = edge.prune_score(prune_scores)
                    prune_scores[edge.dest.module_name][edge.patch_idx] = min(tao, curr)
                    prev_faith = faith
                else:  # Edge is important - don't patch it
                    edge.patch_mask(model).data[edge.patch_idx] = 0.0

        logger.debug(
            "ACDC sweep %d/%d done — removed %d / %d edges",
            sweep_idx + 1, n_sweeps, len(removed_edges), n_edges,
        )

    # Invert scores so higher is better
    max_score = max([ps.max() for ps in prune_scores.values()])
    final_scores = model.new_prune_scores(init_val=0.0)
    for mod, ps in prune_scores.items():
        final_scores[mod] = t.where(ps != t.inf, max_score - ps, t.zeros_like(ps))
    return final_scores
