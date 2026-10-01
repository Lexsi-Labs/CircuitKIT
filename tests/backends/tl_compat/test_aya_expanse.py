"""Offline Group A-D: Aya Expanse 8B (``cohere1``, ``CohereForCausalLM``)
registration and conversion.

Aya Expanse 8B (``CohereLabs/aya-expanse-8b``) is the plain ``CohereForCausalLM``
architecture ("cohere1"), structurally a *simplification* of cohere2
(tiny-aya / Command R7B, tested in ``test_cohere.py`` / ``test_command_r7b.py``):
same LayerNorm-bias-less parallel attn+MLP block, GQA, tied embeddings and
interleaved rotary, a non-no-op ``logit_scale`` fold -- but with **no sliding
window and no NoPE**: rotary applies to every layer. This file covers what is
actually new for this architecture:

* the repo ID is registered correctly, through a second, independently
  registered ``ArchPort`` (``CohereForCausalLM``) alongside cohere2's, per the
  Stage 1 multi-arch seam;
* the config-driven path produces the right ``HookedTransformerConfig`` for
  Aya Expanse's *real* spec (32 layers, 4096 d_model, 32/8 GQA heads,
  d_head=128, d_mlp=14336, vocab=256000, rope_theta=10000, no sliding window /
  attn_types, tied embeddings);
* cohere1's HF config does not always set an explicit ``head_dim`` (unlike
  cohere2's), so the config converter's `hidden_size // num_attention_heads`
  fallback is pinned;
* the NoPE policy for this architecture always returns ``False`` -- rotary on
  every layer, regardless of the (TL-default) ``attn_type`` string, which is
  exactly the ambiguity the per-architecture registry dispatch in Stage 1
  exists to resolve (cohere2's "global" means NoPE; cohere1's default
  "global" attn_type means nothing of the sort);
* ``logit_scale=0.125`` is folded into ``W_U`` as a genuine (non-1.0,
  non-0.25) scale -- a third distinct value now exercised across the three
  cohere-family models;
* two converter guards specific to this architecture: ``attention_bias=True``
  and ``use_qk_norm=True`` (both unsupported by the shared
  ``convert_cohere2_weights`` weight converter -- see the ``cohere.py`` module
  docstring) raise ``NotImplementedError`` rather than silently producing a
  wrong model.

Fully offline: no gated weights, no network, no GPU. Numerical parity against
the real checkpoint is the separate, network-gated ``test_aya_expanse_parity.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import transformer_lens.loading_from_pretrained as tl_loading
from transformer_lens import HookedTransformerConfig

from circuitkit.backends._tl_compat import cohere as cohere_patch
from circuitkit.backends._tl_compat import registry as _registry

MODEL_NAME = "CohereLabs/aya-expanse-8b"

# Aya Expanse's real, confirmed spec (AutoConfig.from_pretrained(MODEL_NAME),
# checked directly in this venv rather than assumed -- see Part 0.2 of the
# integration guide): 32 layers, d_model 4096, 32 query / 8 KV heads, d_head
# 128 (hidden_size // n_heads, not an explicit config field), d_mlp 14336
# (gated SiLU), vocab 256000, rope_theta 10000, logit_scale 0.125, tied
# embeddings, max_position_embeddings 8192 (already <= the port's
# _MAX_SAFE_N_CTX cap, unlike Command R7B's 132096).
_N_LAYERS = 32


def _make_aya_expanse_hf_config(**overrides):
    """A ``SimpleNamespace``-backed fake HF config -- deliberately *not* a
    ``MagicMock``: cohere1's converter relies on ``getattr(cfg, "head_dim",
    None)`` falling through to a real default when the attribute is absent,
    and ``MagicMock`` auto-vivifies any attribute access (defeating that
    fallback in a test). ``SimpleNamespace`` raises ``AttributeError`` on a
    genuinely-missing attribute, like a real ``CohereConfig`` without an
    explicit ``head_dim`` does.
    """
    defaults = dict(
        architectures=["CohereForCausalLM"],
        model_type="cohere",
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        num_hidden_layers=_N_LAYERS,
        max_position_embeddings=8192,
        layer_norm_eps=1e-5,
        vocab_size=256000,
        hidden_act="silu",
        rope_theta=10000.0,
        attention_bias=False,
        tie_word_embeddings=True,
        use_qk_norm=False,
        logit_scale=0.125,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestAyaExpanseRegistration:
    def test_repo_id_registered(self):
        assert MODEL_NAME in tl_loading.OFFICIAL_MODEL_NAMES
        assert tl_loading.get_official_model_name(MODEL_NAME) == MODEL_NAME

    def test_repo_id_is_in_aya_expanse_model_names(self):
        assert cohere_patch.AYA_EXPANSE_MODEL_NAMES == [MODEL_NAME]

    def test_other_cohere_family_registrations_unaffected(self):
        """Registering cohere1 must not disturb cohere2's own lists."""
        assert cohere_patch.SUPPORTED_MODEL_NAMES == [
            "CohereLabs/tiny-aya-base",
            "CohereLabs/tiny-aya-global",
            "CohereLabs/tiny-aya-earth",
            "CohereLabs/tiny-aya-fire",
            "CohereLabs/tiny-aya-water",
        ]
        assert cohere_patch.COMMAND_R7B_MODEL_NAMES == [
            "CohereLabs/c4ai-command-r7b-12-2024",
        ]

    def test_case_insensitive_lookup(self):
        assert tl_loading.get_official_model_name(MODEL_NAME.lower()) == MODEL_NAME

    def test_resolves_to_a_distinct_architecture_port_from_cohere2(self):
        cohere1_port = _registry.resolve_by_architecture("CohereForCausalLM")
        cohere2_port = _registry.resolve_by_architecture("Cohere2ForCausalLM")
        assert cohere1_port is not None
        assert cohere2_port is not None
        assert cohere1_port is not cohere2_port
        assert cohere1_port.architecture == "CohereForCausalLM"
        assert cohere1_port.config_converter is cohere_patch._convert_cohere1_config
        # The weight converter genuinely is shared -- see the module docstring.
        assert cohere1_port.weight_converter is cohere_patch.convert_cohere2_weights


class TestAyaExpanseConfigConversion:
    def test_real_spec_builds_correct_cfg_dict(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        assert cfg_dict["d_model"] == 4096
        assert cfg_dict["n_heads"] == 32
        assert cfg_dict["n_key_value_heads"] == 8
        assert cfg_dict["d_head"] == 128
        assert cfg_dict["d_mlp"] == 14336
        assert cfg_dict["n_layers"] == 32
        assert cfg_dict["d_vocab"] == 256000
        assert cfg_dict["rotary_base"] == 10000
        assert cfg_dict["rotary_adjacent_pairs"] is True
        assert cfg_dict["rotary_dim"] == 128
        assert cfg_dict["parallel_attn_mlp"] is True
        assert cfg_dict["normalization_type"] == "LN"
        assert cfg_dict["original_architecture"] == "CohereForCausalLM"
        assert cfg_dict["tokenizer_name"] == MODEL_NAME
        # logit_scale is folded at weight-conversion time, not threaded
        # through the HookedTransformerConfig -- same contract as cohere2.
        assert "logit_scale" not in cfg_dict
        # Real max_position_embeddings (8192) already sits at the port's cap,
        # so this is a floor-not-ceiling case, not the OOM-avoidance case
        # Command R7B needed -- see TestAyaExpanseContextLength below.
        assert cfg_dict["n_ctx"] == 8192

    def test_no_local_attn_no_sliding_window_no_attn_types(self):
        """The one architectural simplification vs cohere2: no sliding
        window, no per-layer attn_types -- rotary and full attention on
        every layer."""
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        assert cfg_dict["use_local_attn"] is False
        assert "window_size" not in cfg_dict
        assert "attn_types" not in cfg_dict

    def test_head_dim_fallback_when_config_omits_it(self):
        """Unlike Cohere2Config, CohereConfig does not always set an explicit
        head_dim; CohereAttention.__init__ falls back to
        hidden_size // num_attention_heads. The fake config here has no
        head_dim attribute at all (SimpleNamespace raises AttributeError on
        access, unlike a MagicMock), so this only passes if the converter
        does the same fallback rather than assuming the attribute exists."""
        fake_hf_cfg = _make_aya_expanse_hf_config()  # no head_dim override
        assert not hasattr(fake_hf_cfg, "head_dim")
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["d_head"] == 4096 // 32 == 128
        assert cfg_dict["rotary_dim"] == 128

    def test_rejects_attention_bias(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128, attention_bias=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="attention_bias"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_use_qk_norm(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128, use_qk_norm=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="use_qk_norm"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_wrong_architecture(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(
            head_dim=128, architectures=["Cohere2ForCausalLM"]
        )
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="CohereForCausalLM"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_cfg_dict_constructs_a_valid_hooked_transformer_config(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        cfg_dict["model_name"] = "aya-expanse-8b"
        cfg_dict["init_weights"] = False
        cfg_dict["dtype"] = torch.float32
        cfg_dict["device"] = "cpu"

        cfg = HookedTransformerConfig.from_dict(cfg_dict)

        assert cfg.n_layers == 32
        assert cfg.n_heads == 32
        assert cfg.n_key_value_heads == 8
        assert cfg.d_head == 128
        assert cfg.use_local_attn is False
        assert cfg.attn_types is None


class TestAyaExpanseContextLength:
    """Aya Expanse's real max_position_embeddings is 8192 -- already exactly
    at the port's _MAX_SAFE_N_CTX cap, unlike Command R7B's 132096. This is
    the "floor, not ceiling" case: confirm the min() still behaves correctly
    on both sides for this architecture's own config converter."""

    def test_real_value_is_not_padded_up(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128)
        assert fake_hf_cfg.max_position_embeddings == 8192
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["n_ctx"] == 8192

    def test_a_hypothetically_larger_context_would_be_capped(self):
        fake_hf_cfg = _make_aya_expanse_hf_config(head_dim=128, max_position_embeddings=132096)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["n_ctx"] == cohere_patch._MAX_SAFE_N_CTX


class TestAyaExpanseRotaryAppliesToEveryLayer:
    """The plan's key architectural claim: no NoPE at all for cohere1 --
    rotary must apply on every layer regardless of the (TL-default)
    attn_type string. Unit-level check of the registered policy; the
    real-weight Group F gate (test_aya_expanse_parity.py) proves it end to
    end against actual HF outputs."""

    @pytest.mark.parametrize("layer_id", [0, 1, 15, 31])
    @pytest.mark.parametrize("layer_attn_type", ["global", "local"])
    def test_should_skip_rotary_is_always_false(self, layer_id, layer_attn_type):
        cfg = SimpleNamespace(original_architecture="CohereForCausalLM")
        assert _registry.should_skip_rotary(cfg, layer_id, layer_attn_type) is False

    def test_distinct_from_cohere2s_global_means_nope_policy(self):
        """Sanity check on the seam itself: the same attn_type string
        ("global", TL's default when a port sets no per-layer attn_types)
        must resolve to opposite NoPE behaviour depending on which
        architecture asked -- proving dispatch is genuinely per-architecture,
        not per-attn-type-string."""
        cohere1_cfg = SimpleNamespace(original_architecture="CohereForCausalLM")
        cohere2_cfg = SimpleNamespace(original_architecture="Cohere2ForCausalLM")
        assert _registry.should_skip_rotary(cohere1_cfg, 0, "global") is False
        assert _registry.should_skip_rotary(cohere2_cfg, 0, "global") is True


class TestAyaExpanseLogitScaleFold:
    """A third distinct logit_scale value now exercised across the
    cohere-family models: tiny-aya=1.0 (no-op), Command R7B=0.25, Aya
    Expanse=0.125. The generic converter (test_cohere.py) already handles
    arbitrary scales; this pins the actual Aya Expanse value specifically,
    reusing the exact same shared convert_cohere2_weights function."""

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
            self.original_architecture = "CohereForCausalLM"

    def _build_fake_model(self, dims, logit_scale):
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

        class FakeCohere1(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeBaseModel()
                self.lm_head = nn.Linear(dims["d_model"], dims["d_vocab"], bias=False)
                self.logit_scale = logit_scale

        return FakeCohere1()

    def test_aya_expanse_logit_scale_is_folded(self):
        model = self._build_fake_model(self._DIMS, logit_scale=0.125)
        cfg = self._FakeTLCfg(**self._DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        expected = model.lm_head.weight.T * 0.125
        assert torch.allclose(sd["unembed.W_U"], expected)
        # Not a no-op, and distinct from Command R7B's 0.25 fold too.
        assert not torch.equal(sd["unembed.W_U"], model.lm_head.weight.T)

    def test_gqa_weight_shapes(self):
        """GQA (32 query / 8 KV heads on the real model, 4/2 here) --
        _W_K/_W_V must carry the KV head count, not the query head count."""
        model = self._build_fake_model(self._DIMS, logit_scale=0.125)
        cfg = self._FakeTLCfg(**self._DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        assert sd["blocks.0.attn.W_Q"].shape == (
            self._DIMS["n_heads"],
            self._DIMS["d_model"],
            self._DIMS["d_head"],
        )
        assert sd["blocks.0.attn._W_K"].shape == (
            self._DIMS["n_kv_heads"],
            self._DIMS["d_model"],
            self._DIMS["d_head"],
        )
        assert sd["blocks.0.attn._W_V"].shape == (
            self._DIMS["n_kv_heads"],
            self._DIMS["d_model"],
            self._DIMS["d_head"],
        )


class TestAyaExpanseTinyRandomInitForward:
    """A tiny, random-init HookedTransformer built from cohere1's own config
    shape (as opposed to cohere2's) must produce a finite forward pass."""

    def test_forward_is_finite(self):
        cfg_dict = dict(
            d_model=32,
            d_head=8,
            n_heads=4,
            n_key_value_heads=2,
            d_mlp=64,
            n_layers=3,
            n_ctx=64,
            eps=1e-5,
            d_vocab=97,
            act_fn="silu",
            gated_mlp=True,
            normalization_type="LN",
            positional_embedding_type="rotary",
            rotary_adjacent_pairs=True,
            rotary_dim=8,
            rotary_base=10000,
            use_local_attn=False,
            parallel_attn_mlp=True,
            use_attn_scale=True,
            attention_dir="causal",
            final_rms=False,
            original_architecture="CohereForCausalLM",
            tokenizer_name="gpt2",
            model_name="tiny-cohere1",
            init_weights=True,
            dtype=torch.float32,
            device="cpu",
        )
        cfg = HookedTransformerConfig.from_dict(cfg_dict)

        from transformer_lens import HookedTransformer

        model = HookedTransformer(cfg)
        model.eval()
        x = torch.randint(0, 97, (1, 10))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 10, 97)
        assert torch.isfinite(out).all()
