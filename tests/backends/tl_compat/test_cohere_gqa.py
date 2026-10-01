"""Group F — runtime GQA / parallel-block gate for tiny-aya (cohere2).

Group A's ``test_cohere.py`` validates the *weight converter's* GQA slicing
(K/V sliced by ``n_key_value_heads``) at the tensor level. This file validates
the *runtime* consequences on a real — but tiny, random-init — cohere2-shaped
``HookedTransformer``: that the model actually runs as grouped-query attention,
that the activations the discovery backends read are exposed at the head count
those backends assume, and that the EAP graph is built from the ungrouped
query-head count with the parallel-block topology.

Fully offline: random-init weights, no gated checkpoints, no network, no GPU.
Numerical parity against the real weights is the separate, network-gated
``test_cohere_parity.py``.

Why these invariants matter (all traceable to Groups A–D):

* IBCircuit reads ``hook_z``, which TL always exposes at the query-head count
  (post-GQA-expansion) — so its head/MLP bookkeeping is GQA-safe by
  construction.
* CD-T's simplified path indexes ``hook_v`` per query head; TL may store
  ``hook_v`` at ``n_key_value_heads``, so the Group C fix repeat-interleaves it
  up to ``n_heads``. That is only valid when the stored head count *divides*
  ``n_heads`` — asserted here against a live forward pass.
* EAP's ``Graph.from_model`` sizes its node/edge tensors from ``cfg.n_heads``
  (the ungrouped query-head count) and reads ``cfg.parallel_attn_mlp`` for the
  cohere2 single-LN parallel block.
"""

from __future__ import annotations

import pytest
import torch

from transformer_lens import HookedTransformer
from transformer_lens import HookedTransformerConfig

from circuitkit.backends import _tl_compat  # noqa: F401 -- import-time apply_patches()
from circuitkit.backends.eap.graph import Graph

# Tiny cohere2-shaped dims: n_heads=4 query heads, n_key_value_heads=2 -> a
# genuine 2:1 grouped-query ratio (not MHA, not MQA), so the ungroup path is
# actually exercised. SWA x3 : full x1 mirrors the real layer_types pattern.
_N_LAYERS = 4
_N_HEADS = 4
_N_KV_HEADS = 2


def _build_tiny_cohere2(parallel_attn_mlp: bool = True) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=_N_LAYERS,
        d_model=16,
        n_ctx=32,
        d_head=4,
        n_heads=_N_HEADS,
        n_key_value_heads=_N_KV_HEADS,
        d_mlp=32,
        d_vocab=50,
        act_fn="silu",
        gated_mlp=True,
        normalization_type="LN",
        positional_embedding_type="rotary",
        rotary_adjacent_pairs=True,
        rotary_dim=4,
        rotary_base=50000,
        use_local_attn=True,
        window_size=4,
        attn_types=["local", "local", "local", "global"],
        parallel_attn_mlp=parallel_attn_mlp,
        use_attn_scale=True,
        original_architecture="Cohere2ForCausalLM",
        init_weights=True,
        device="cpu",
        seed=0,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return _build_tiny_cohere2()


@pytest.fixture(scope="module")
def tokens(tiny_model):
    return torch.randint(0, tiny_model.cfg.d_vocab, (2, 10))


# --------------------------------------------------------------------------- #
# The model actually runs as grouped-query attention.                         #
# --------------------------------------------------------------------------- #
class TestModelIsGQA:
    def test_kv_heads_fewer_than_query_heads(self, tiny_model):
        assert tiny_model.cfg.n_key_value_heads == _N_KV_HEADS
        assert tiny_model.cfg.n_heads == _N_HEADS
        assert _N_KV_HEADS < _N_HEADS  # genuinely grouped, not plain MHA

    def test_forward_pass_is_finite(self, tiny_model, tokens):
        with torch.no_grad():
            logits = tiny_model(tokens)
        assert logits.shape == (2, 10, tiny_model.cfg.d_vocab)
        assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# Activations are exposed at the head count each backend assumes.             #
# --------------------------------------------------------------------------- #
class TestActivationHeadCounts:
    def test_hook_z_is_per_query_head(self, tiny_model, tokens):
        """IBCircuit reads hook_z; TL always exposes it at the query-head count."""
        with torch.no_grad():
            _, cache = tiny_model.run_with_cache(tokens)
        z = cache["blocks.0.attn.hook_z"]
        # [batch, pos, n_heads, d_head] -> head axis is -2.
        assert z.shape[-2] == tiny_model.cfg.n_heads

    def test_hook_v_head_count_divides_query_heads(self, tiny_model, tokens):
        """CD-T repeat-interleaves hook_v up to n_heads; that is only valid when
        the stored head count divides n_heads. TL stores hook_v at
        n_key_value_heads when ungroup is off (the CD-T adapter default)."""
        original = getattr(tiny_model.cfg, "ungroup_grouped_query_attention", False)
        tiny_model.cfg.ungroup_grouped_query_attention = False
        try:
            with torch.no_grad():
                _, cache = tiny_model.run_with_cache(tokens)
            v_heads = cache["blocks.0.attn.hook_v"].shape[-2]
            assert v_heads in (tiny_model.cfg.n_key_value_heads, tiny_model.cfg.n_heads)
            assert tiny_model.cfg.n_heads % v_heads == 0, (
                f"hook_v head count {v_heads} does not divide n_heads "
                f"{tiny_model.cfg.n_heads}; CD-T's repeat_interleave expansion would fail"
            )
        finally:
            tiny_model.cfg.ungroup_grouped_query_attention = original

    def test_ungroup_flag_toggles_without_breaking_forward(self, tiny_model, tokens):
        """discover_circuit sets ungroup_grouped_query_attention=True; both
        settings must run and stay finite, and hook_z stays per-query-head."""
        original = getattr(tiny_model.cfg, "ungroup_grouped_query_attention", False)
        try:
            for ungroup in (True, False):
                tiny_model.cfg.ungroup_grouped_query_attention = ungroup
                with torch.no_grad():
                    logits, cache = tiny_model.run_with_cache(tokens)
                assert torch.isfinite(logits).all()
                assert cache["blocks.0.attn.hook_z"].shape[-2] == tiny_model.cfg.n_heads
        finally:
            # Restore the shared module-scoped model's state for later tests.
            tiny_model.cfg.ungroup_grouped_query_attention = original


# --------------------------------------------------------------------------- #
# EAP graph is built from the ungrouped query-head count + parallel topology.  #
# --------------------------------------------------------------------------- #
class TestEapGraphStructure:
    def test_graph_uses_query_head_count(self, tiny_model):
        graph = Graph.from_model(tiny_model, node_scores=True)
        assert graph.cfg["n_heads"] == tiny_model.cfg.n_heads  # 4, not n_kv_heads=2
        # Node/edge tensors are sized from the query-head count.
        assert graph.n_forward == 1 + _N_LAYERS * (_N_HEADS + 1)
        assert graph.n_backward == _N_LAYERS * (3 * _N_HEADS + 1) + 1

    def test_graph_reads_parallel_attn_mlp_from_cfg(self, tiny_model):
        graph = Graph.from_model(tiny_model)
        assert graph.cfg["parallel_attn_mlp"] is True

    def test_parallel_flag_is_not_hardcoded(self):
        """A sequential model must produce parallel_attn_mlp=False, proving the
        flag is read from cfg rather than assumed for cohere2."""
        seq_model = _build_tiny_cohere2(parallel_attn_mlp=False)
        seq_graph = Graph.from_model(seq_model)
        assert seq_graph.cfg["parallel_attn_mlp"] is False

    def test_graph_node_tensors_have_consistent_length(self, tiny_model):
        graph = Graph.from_model(tiny_model, node_scores=True)
        assert graph.nodes_scores.shape[0] == graph.n_forward
        assert graph.real_edge_mask.shape == (graph.n_forward, graph.n_backward)
