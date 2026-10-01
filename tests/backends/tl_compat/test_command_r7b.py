"""Offline Group A-E: Command R7B (``cohere2``) registration and conversion.

Command R7B (``CohereLabs/c4ai-command-r7b-12-2024``) shares tiny-aya's
``Cohere2ForCausalLM`` architecture and is handled by the exact same
``_convert_cohere2_config`` / ``convert_cohere2_weights`` functions
``test_cohere.py`` already covers in depth (config mapping, GQA weight
slicing, NoPE, tied embeddings, ...). This file does not re-derive those --
it covers what is actually new for this model:

* the repo ID is registered correctly;
* the config-driven path produces the right ``HookedTransformerConfig`` for
  Command R7B's *real* spec (32 layers, 4096 d_model, 32/8 GQA heads,
  d_head=128, d_mlp=14336, vocab=256000, rope_theta=50000, sliding_window=4096
  with pattern 4, tied embeddings) -- doubles as a check against the actual
  architecture, not just arbitrary numbers;
* ``logit_scale=0.25`` is folded into ``W_U`` as a genuine (non-1.0) scale --
  tiny-aya's own logit_scale=1.0 spec never exercises this as anything but a
  no-op, and the plan flags this as the one behavioural difference between
  the two cohere2 checkpoints.

Fully offline: no gated weights, no network, no GPU. Numerical parity against
the real checkpoint is the separate, network-gated ``test_command_r7b_parity.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import transformer_lens.loading_from_pretrained as tl_loading
from transformer_lens import HookedTransformerConfig

from circuitkit.backends._tl_compat import cohere as cohere_patch

MODEL_NAME = "CohereLabs/c4ai-command-r7b-12-2024"

# Command R7B's real spec (see Part 0.2 of the integration guide / the model's
# HF config): 32 layers, d_model 4096, 32 query / 8 KV heads, d_head 128,
# d_mlp 14336 (gated SiLU), vocab 256000, rope_theta 50000, sliding_window
# 4096 with a SWA x3 : full x1 pattern, logit_scale 0.25, tied embeddings.
_N_LAYERS = 32


def _make_command_r7b_hf_config(**overrides):
    layer_types = overrides.pop(
        "layer_types",
        ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(_N_LAYERS)],
    )
    defaults = dict(
        architectures=["Cohere2ForCausalLM"],
        model_type="cohere2",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=14336,
        num_hidden_layers=_N_LAYERS,
        max_position_embeddings=132096,
        layer_norm_eps=1e-5,
        vocab_size=256000,
        hidden_act="silu",
        rope_theta=50000.0,
        sliding_window=4096,
        layer_types=layer_types,
        attention_bias=False,
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg


class TestCommandR7bRegistration:
    def test_repo_id_registered(self):
        assert MODEL_NAME in tl_loading.OFFICIAL_MODEL_NAMES
        assert tl_loading.get_official_model_name(MODEL_NAME) == MODEL_NAME

    def test_repo_id_is_in_command_r7b_model_names(self):
        assert cohere_patch.COMMAND_R7B_MODEL_NAMES == [MODEL_NAME]

    def test_tiny_aya_registration_unaffected(self):
        """Registering command-r7b must not disturb tiny-aya's own list."""
        assert cohere_patch.SUPPORTED_MODEL_NAMES == [
            "CohereLabs/tiny-aya-base",
            "CohereLabs/tiny-aya-global",
            "CohereLabs/tiny-aya-earth",
            "CohereLabs/tiny-aya-fire",
            "CohereLabs/tiny-aya-water",
        ]

    def test_case_insensitive_lookup(self):
        assert tl_loading.get_official_model_name(MODEL_NAME.lower()) == MODEL_NAME


class TestCommandR7bConfigConversion:
    def test_real_spec_builds_correct_cfg_dict(self):
        fake_hf_cfg = _make_command_r7b_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        assert cfg_dict["d_model"] == 4096
        assert cfg_dict["n_heads"] == 32
        assert cfg_dict["n_key_value_heads"] == 8
        assert cfg_dict["d_head"] == 128
        assert cfg_dict["d_mlp"] == 14336
        assert cfg_dict["n_layers"] == 32
        assert cfg_dict["d_vocab"] == 256000
        assert cfg_dict["rotary_base"] == 50000
        assert cfg_dict["rotary_adjacent_pairs"] is True
        assert cfg_dict["rotary_dim"] == 128
        assert cfg_dict["use_local_attn"] is True
        assert cfg_dict["window_size"] == 4096
        assert cfg_dict["parallel_attn_mlp"] is True
        assert cfg_dict["normalization_type"] == "LN"
        assert cfg_dict["original_architecture"] == "Cohere2ForCausalLM"
        assert cfg_dict["tokenizer_name"] == MODEL_NAME
        # logit_scale is folded at weight-conversion time, not threaded
        # through the HookedTransformerConfig -- same contract as tiny-aya.
        assert "logit_scale" not in cfg_dict
        # n_ctx is capped, not the real max_position_embeddings=132096 --
        # see TestCommandR7bContextLengthCap below for why.
        assert cfg_dict["n_ctx"] == cohere_patch._MAX_SAFE_N_CTX

    def test_sliding_window_pattern_4_matches_real_layer_types(self):
        # sliding_window_pattern=4 -> SWA, SWA, SWA, full, repeating; every
        # 4th layer (1-indexed) is the full-attention "global" layer.
        fake_hf_cfg = _make_command_r7b_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        attn_types = cfg_dict["attn_types"]
        assert len(attn_types) == 32
        for i, kind in enumerate(attn_types):
            expected = "global" if (i + 1) % 4 == 0 else "local"
            assert kind == expected, f"layer {i}: expected {expected}, got {kind}"

    def test_cfg_dict_constructs_a_valid_hooked_transformer_config(self):
        fake_hf_cfg = _make_command_r7b_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        cfg_dict["model_name"] = "command-r7b"
        cfg_dict["init_weights"] = False
        cfg_dict["dtype"] = torch.float32
        cfg_dict["device"] = "cpu"

        cfg = HookedTransformerConfig.from_dict(cfg_dict)

        assert cfg.n_layers == 32
        assert cfg.n_heads == 32
        assert cfg.n_key_value_heads == 8
        assert cfg.d_head == 128
        assert cfg.attn_types[3] == "global"
        assert cfg.attn_types[0] == "local"


class TestCommandR7bContextLengthCap:
    """Command R7B's real max_position_embeddings is 132096. TL's
    AbstractAttention allocates a *dense* n_ctx x n_ctx causal mask per
    attention block -- at the real value that is a ~65 GB float32 tensor for
    a single layer, which OOMs model construction before ``ck.load_model()``
    even finishes (discovered while running the Stage 2 real-weight parity
    gate: HookedTransformer.from_pretrained silently allocated the full
    132096x132096 mask and was killed by the OOM killer). TL's own stock
    converters hit this for other long-context models and cap n_ctx in the
    config they return for exactly this reason; this pins that Command R7B
    does the same and by how much."""

    def test_n_ctx_is_capped_not_the_real_132096(self):
        fake_hf_cfg = _make_command_r7b_hf_config()
        assert fake_hf_cfg.max_position_embeddings == 132096
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["n_ctx"] == 8192
        assert cfg_dict["n_ctx"] < fake_hf_cfg.max_position_embeddings

    def test_cap_covers_the_sliding_window_boundary_test(self):
        """The parity gate's sliding-window-boundary test runs a
        window(4096) + 100 token sequence; the cap must stay comfortably
        above that or that gate becomes unrunnable too."""
        assert cohere_patch._MAX_SAFE_N_CTX > 4096 + 100

    def test_cap_is_a_floor_not_a_ceiling_on_small_models(self):
        """A hypothetical short-context cohere2 model must not be padded up
        to the cap -- min() should take the smaller of the two."""
        fake_hf_cfg = _make_command_r7b_hf_config(max_position_embeddings=2048)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["n_ctx"] == 2048


class TestCommandR7bLogitScaleFold:
    """The plan's key correctness claim for this model: logit_scale=0.25 is a
    *real* fold (tiny-aya's 1.0 never exercises this as anything but a
    no-op). The generic converter already handles arbitrary scales (see
    test_cohere.py::TestConvertCohere2Weights::test_logit_scale_folded_into_unembed);
    this pins the actual Command R7B value specifically."""

    _DIMS = dict(n_layers=2, n_heads=4, n_kv_heads=2, d_head=2, d_model=8, d_mlp=16, d_vocab=10)

    class _FakeTLCfg:
        def __init__(self, n_layers, n_heads, n_kv_heads, d_head, d_model, d_mlp, d_vocab):
            self.n_layers = n_layers
            self.n_heads = n_heads
            self.n_key_value_heads = n_kv_heads
            self.d_head = d_head
            self.d_model = d_model
            self.d_mlp = d_mlp
            self.d_vocab = d_vocab
            self.dtype = torch.float32
            self.original_architecture = "Cohere2ForCausalLM"

    def _build_fake_model(self, dims):
        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(
                    dims["d_model"], dims["n_heads"] * dims["d_head"], bias=False
                )
                self.k_proj = nn.Linear(
                    dims["d_model"], dims["n_kv_heads"] * dims["d_head"], bias=False
                )
                self.v_proj = nn.Linear(
                    dims["d_model"], dims["n_kv_heads"] * dims["d_head"], bias=False
                )
                self.o_proj = nn.Linear(
                    dims["n_heads"] * dims["d_head"], dims["d_model"], bias=False
                )

        class FakeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(dims["d_model"], dims["d_mlp"], bias=False)
                self.up_proj = nn.Linear(dims["d_model"], dims["d_mlp"], bias=False)
                self.down_proj = nn.Linear(dims["d_mlp"], dims["d_model"], bias=False)

        class FakeLN(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(dims["d_model"]))

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()
                self.mlp = FakeMLP()
                self.input_layernorm = FakeLN()

        class FakeBaseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(dims["d_vocab"], dims["d_model"])
                self.layers = nn.ModuleList([FakeLayer() for _ in range(dims["n_layers"])])
                self.norm = FakeLN()

        class FakeCohere2(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeBaseModel()
                self.lm_head = nn.Linear(dims["d_model"], dims["d_vocab"], bias=False)
                self.logit_scale = 0.25  # Command R7B's real value

        return FakeCohere2()

    def test_command_r7b_logit_scale_is_folded(self):
        model = self._build_fake_model(self._DIMS)
        cfg = self._FakeTLCfg(**self._DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        expected = model.lm_head.weight.T * 0.25
        assert torch.allclose(sd["unembed.W_U"], expected)
        # And confirm it is NOT a no-op (distinguishing it from tiny-aya's
        # logit_scale=1.0 spec, where this same assertion would trivially
        # hold for the *unscaled* weight too).
        assert not torch.equal(sd["unembed.W_U"], model.lm_head.weight.T)
