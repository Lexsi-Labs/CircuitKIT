"""Node-level EAP scores an MLP with the gradient at its own output.

The node attribution-patching score is (a_corrupt - a_clean) . dL/da, with
both factors taken at the same activation a. For an MLP node that activation
is ``hook_mlp_out``; the gradient at ``hook_mlp_in`` belongs to a different
tensor and gives a different number.
"""

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from circuitkit.backends.eap.attribute_node import make_hooks_and_matrices
from circuitkit.backends.eap.graph import Graph


def _tiny_model(gated=False):
    torch.manual_seed(0)
    # gated: a SwiGLU MLP with RMSNorm, as in Llama / Qwen. hook_mlp_out is still
    # the MLP's whole contribution to the residual stream.
    cfg = HookedTransformerConfig(
        n_layers=2, n_heads=2, d_model=16, d_head=8, d_mlp=32, d_vocab=50, n_ctx=8,
        act_fn="silu" if gated else "gelu", gated_mlp=gated,
        normalization_type="RMS" if gated else "LN", device="cpu", seed=0,
    )
    model = HookedTransformer(cfg)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.setup()
    return model


@pytest.mark.parametrize("gated", [False, True])
def test_mlp_node_score_uses_gradient_at_mlp_out(gated):
    model = _tiny_model(gated)
    graph = Graph.from_model(model, node_scores=True)
    clean = torch.tensor([[1, 2, 3, 4, 5]])
    corrupt = torch.tensor([[1, 7, 8, 9, 5]])

    def metric(logits):
        return logits[0, -1, 3] - logits[0, -1, 4]

    scores = torch.zeros(graph.n_forward)
    (fwd_corrupt, fwd_clean, bwd), _ = make_hooks_and_matrices(
        model, graph, 1, clean.shape[1], scores
    )
    with torch.no_grad(), model.hooks(fwd_corrupt):
        model(corrupt)
    with model.hooks(fwd_hooks=fwd_clean, bwd_hooks=bwd):
        metric(model(clean)).backward()

    # Reference: (corrupt - clean) at hook_mlp_out times dL/d(hook_mlp_out).
    _, corrupt_cache = model.run_with_cache(corrupt)
    clean_acts, grads = {}, {}
    names = [f"blocks.{layer}.hook_mlp_out" for layer in range(model.cfg.n_layers)]

    def save_act(act, hook):
        act.retain_grad()
        clean_acts[hook.name] = act

    with model.hooks(fwd_hooks=[(n, save_act) for n in names]):
        metric(model(clean)).backward()
    for n in names:
        grads[n] = clean_acts[n].grad

    for layer, n in enumerate(names):
        expected = ((corrupt_cache[n] - clean_acts[n].detach()) * grads[n]).sum()
        got = scores[graph.forward_index(graph.nodes[f"m{layer}"])]
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-6)
