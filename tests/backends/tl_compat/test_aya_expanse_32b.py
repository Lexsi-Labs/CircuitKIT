"""Offline Group A-D: Aya Expanse 32B (``cohere1``, ``CohereForCausalLM``)
registration and conversion.

Aya Expanse 32B (``CohereLabs/aya-expanse-32b``) is the same HF architecture
as Aya Expanse 8B (tested in ``test_aya_expanse.py``) -- same config
converter (``_convert_cohere1_config``), same "no NoPE at all" rotary policy,
same shared weight converter (``convert_cohere2_weights``). Only the config
*values* differ with scale:

* 8:1 GQA (64 query / 8 KV heads) vs. 8B's 4:1 (32 query / 8 KV heads);
* ``rope_theta`` 4,000,000 vs. 8B's 10,000;
* ``logit_scale`` 0.0625 vs. 8B's 0.125;
* ``head_dim`` is still absent from the HF config (same fallback:
  ``hidden_size // num_attention_heads`` = 8192 // 64 = 128, same value as
  8B's 4096 // 32 despite the different hidden_size/head-count pair);
* d_mlp 24576 (vs. 8B's 14336), 40 layers (vs. 8B's 32), d_model 8192 (vs.
  8B's 4096). vocab (256000), normalization (bias-less LayerNorm, eps 1e-5),
  tied embeddings and max_position_embeddings (8192) are unchanged from 8B.

This file covers what is new at 32B scale; it does not repeat 8B's own
coverage of the architecture-level guards (attention_bias, use_qk_norm,
wrong-architecture) beyond pinning that the same converter/guards apply
unchanged at the 32B spec.

Fully offline: no gated weights, no network, no GPU. Numerical parity
against the real checkpoint is the separate, network-gated
``test_aya_expanse_32b_parity.py``.
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

MODEL_NAME = "CohereLabs/aya-expanse-32b"

# Aya Expanse 32B's real, confirmed spec (AutoConfig.from_pretrained(MODEL_NAME)
# on the Hub copy -- see docs/advanced/experimental-models.md's spec table):
# 40 layers, d_model 8192, 64 query / 8 KV heads (8:1 GQA), d_head 128
# (hidden_size // n_heads, not an explicit config field, same fallback as 8B),
# d_mlp 24576 (gated SiLU), vocab 256000, rope_theta 4000000, logit_scale
# 0.0625, tied embeddings, max_position_embeddings 8192.
_N_LAYERS = 40


def _make_aya_expanse_32b_hf_config(**overrides):
    """A ``SimpleNamespace``-backed fake HF config for the 32B spec --
    deliberately *not* a ``MagicMock``, for the same reason as the 8B file's
    helper: the converter's ``getattr(cfg, "head_dim", None)`` fallback only
    gets genuinely exercised when the attribute is truly absent."""
    defaults = dict(
        architectures=["CohereForCausalLM"],
        model_type="cohere",
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=24576,
        num_hidden_layers=_N_LAYERS,
        max_position_embeddings=8192,
        layer_norm_eps=1e-5,
        vocab_size=256000,
        hidden_act="silu",
        rope_theta=4000000.0,
        attention_bias=False,
        tie_word_embeddings=True,
        use_qk_norm=False,
        logit_scale=0.0625,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestAyaExpanse32BRegistration:
    def test_repo_id_registered(self):
        assert MODEL_NAME in tl_loading.OFFICIAL_MODEL_NAMES
        assert tl_loading.get_official_model_name(MODEL_NAME) == MODEL_NAME

    def test_alias_resolves(self):
        """Patch 0006 registers the ``aya-expanse-32b`` alias (the Hub repo's
        short name) in TL's own name tables -- confirm it resolves to the
        same full repo ID."""
        assert tl_loading.get_official_model_name("aya-expanse-32b") == MODEL_NAME

    def test_repo_id_is_in_aya_expanse_model_names_alongside_8b(self):
        assert cohere_patch.AYA_EXPANSE_MODEL_NAMES == [
            "CohereLabs/aya-expanse-8b",
            MODEL_NAME,
        ]

    def test_other_cohere_family_registrations_unaffected(self):
        """Registering 32B must not disturb cohere2's own lists."""
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

    def test_resolves_to_the_same_cohere1_arch_port_as_8b(self):
        port_32b = _registry.resolve_by_model_name(MODEL_NAME)
        port_8b = _registry.resolve_by_model_name("CohereLabs/aya-expanse-8b")
        assert port_32b is not None
        assert port_32b is port_8b, "both sizes share one CohereForCausalLM ArchPort"
        assert port_32b.architecture == "CohereForCausalLM"
        assert port_32b.config_converter is cohere_patch._convert_cohere1_config
        assert port_32b.weight_converter is cohere_patch.convert_cohere2_weights


class TestAyaExpanse32BConfigConversion:
    def test_real_spec_builds_correct_cfg_dict(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        assert cfg_dict["d_model"] == 8192
        assert cfg_dict["n_heads"] == 64
        assert cfg_dict["n_key_value_heads"] == 8
        assert cfg_dict["d_head"] == 128
        assert cfg_dict["d_mlp"] == 24576
        assert cfg_dict["n_layers"] == 40
        assert cfg_dict["d_vocab"] == 256000
        assert cfg_dict["rotary_base"] == 4_000_000
        assert isinstance(cfg_dict["rotary_base"], int)
        assert cfg_dict["rotary_adjacent_pairs"] is True
        assert cfg_dict["rotary_dim"] == 128
        assert cfg_dict["parallel_attn_mlp"] is True
        assert cfg_dict["normalization_type"] == "LN"
        assert cfg_dict["original_architecture"] == "CohereForCausalLM"
        assert cfg_dict["tokenizer_name"] == MODEL_NAME
        # logit_scale is folded at weight-conversion time, not threaded
        # through the HookedTransformerConfig -- same contract as 8B.
        assert "logit_scale" not in cfg_dict
        assert cfg_dict["n_ctx"] == 8192

    def test_no_local_attn_no_sliding_window_no_attn_types(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        assert cfg_dict["use_local_attn"] is False
        assert "window_size" not in cfg_dict
        assert "attn_types" not in cfg_dict

    def test_head_dim_fallback_when_config_omits_it(self):
        """32B's real Hub config also has no explicit head_dim attribute (same
        as 8B); the fallback must compute hidden_size // num_attention_heads
        = 8192 // 64 = 128 -- the same numeric value as 8B's 4096 // 32, but
        from a different (n_heads, hidden_size) pair, so this is not a
        tautology carried over from the 8B test."""
        fake_hf_cfg = _make_aya_expanse_32b_hf_config()  # no head_dim override
        assert not hasattr(fake_hf_cfg, "head_dim")
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["d_head"] == 8192 // 64 == 128
        assert cfg_dict["rotary_dim"] == 128

    def test_rejects_attention_bias(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(head_dim=128, attention_bias=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="attention_bias"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_use_qk_norm(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(head_dim=128, use_qk_norm=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="use_qk_norm"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_wrong_architecture(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(
            head_dim=128, architectures=["Cohere2ForCausalLM"]
        )
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="CohereForCausalLM"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_cfg_dict_constructs_a_valid_hooked_transformer_config(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        cfg_dict["model_name"] = "aya-expanse-32b"
        cfg_dict["init_weights"] = False
        cfg_dict["dtype"] = torch.float32
        cfg_dict["device"] = "cpu"

        cfg = HookedTransformerConfig.from_dict(cfg_dict)

        assert cfg.n_layers == 40
        assert cfg.n_heads == 64
        assert cfg.n_key_value_heads == 8
        assert cfg.d_head == 128
        assert cfg.use_local_attn is False
        assert cfg.attn_types is None


class TestAyaExpanse32BRopeThetaDiffersFrom8B:
    """rope_theta is the first of two numeric values (with logit_scale) that
    genuinely differ between the two Aya Expanse sizes rather than just
    scaling dims -- pin it explicitly rather than relying on the shared
    fallback/assertion style above to catch a transposition bug."""

    def test_rope_theta_is_4e6_not_8bs_1e4(self):
        fake_hf_cfg = _make_aya_expanse_32b_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["rotary_base"] == 4_000_000
        assert cfg_dict["rotary_base"] != 10_000


class TestAyaExpanse32BRotaryAppliesToEveryLayer:
    """Same no-NoPE policy as 8B (registered once per architecture, not per
    model size) -- confirm it still resolves correctly for 32B's layer
    count/indices."""

    @pytest.mark.parametrize("layer_id", [0, 1, 20, 39])
    @pytest.mark.parametrize("layer_attn_type", ["global", "local"])
    def test_should_skip_rotary_is_always_false(self, layer_id, layer_attn_type):
        cfg = SimpleNamespace(original_architecture="CohereForCausalLM")
        assert _registry.should_skip_rotary(cfg, layer_id, layer_attn_type) is False


class TestAyaExpanse32BLogitScaleFold:
    """The second genuinely-differing numeric value: 32B's logit_scale is
    0.0625, distinct from 8B's 0.125, tiny-aya's 1.0 and Command R7B's 0.25 --
    a fourth distinct value now exercised across the cohere-family models."""

    _DIMS = dict(n_layers=2, n_heads=8, n_kv_heads=1, d_head=2, d_model=16, d_mlp=32, d_vocab=10)

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

    def test_aya_expanse_32b_logit_scale_is_folded(self):
        model = self._build_fake_model(self._DIMS, logit_scale=0.0625)
        cfg = self._FakeTLCfg(**self._DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        expected = model.lm_head.weight.T * 0.0625
        assert torch.allclose(sd["unembed.W_U"], expected)
        # Not a no-op, and distinct from every other cohere-family fold.
        assert not torch.equal(sd["unembed.W_U"], model.lm_head.weight.T)

    def test_gqa_weight_shapes_8_to_1(self):
        """32B's 8:1 GQA ratio (vs. 8B's 4:1) -- _W_K/_W_V must carry the KV
        head count (1 here), not the query head count (8 here)."""
        model = self._build_fake_model(self._DIMS, logit_scale=0.0625)
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


class TestAyaExpanse32BTinyRandomHFMatchesTL:
    """A tiny, random-init HF ``CohereForCausalLM`` with 32B's distinguishing
    values (rope_theta=4e6, logit_scale=0.0625, 8:1 GQA) loaded through the
    local-checkpoint path, matching ``test_cohere_hooked_transformer_matches_hf``'s
    pattern in ``test_tl_patchset.py``. Proves the config/weight converters
    produce a numerically-correct HookedTransformer for 32B's specific
    values, not just 8B's, entirely offline (tiny dims, no gated download)."""

    def test_tiny_32b_shaped_model_matches_hf_logits(self, tmp_path):
        import dataclasses

        import transformers
        from transformer_lens import HookedTransformer

        # Mirrors test_tl_patchset.py's own locally-defined NO_PROCESSING:
        # not a transformer_lens export, so each test file defines it itself.
        no_processing = dict(
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False,
            refactor_factored_attn_matrices=False,
        )

        hf_cfg = transformers.CohereConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=1,
            max_position_embeddings=256,
            rope_theta=4_000_000.0,
            logit_scale=0.0625,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        torch.manual_seed(0)
        hf = transformers.CohereForCausalLM(hf_cfg).float().eval()
        with torch.no_grad():
            for name, p in hf.named_parameters():
                if "norm" in name:  # off-identity norms, so a missed ln1/ln2 copy shows
                    p.copy_(1 + 0.3 * torch.randn_like(p))
        hf.save_pretrained(tmp_path)  # config.json only is needed: local-dir loader path

        cfg = tl_loading.get_pretrained_model_config(str(tmp_path))
        cfg = dataclasses.replace(cfg, tokenizer_name=None, device="cpu")
        assert cfg.original_architecture == "CohereForCausalLM"
        assert cfg.rotary_base == 4_000_000
        assert cfg.parallel_attn_mlp and cfg.rotary_adjacent_pairs
        assert cfg.use_local_attn is False and cfg.attn_types is None

        model = HookedTransformer(cfg)
        model.load_and_process_state_dict(
            cohere_patch.convert_cohere2_weights(hf, cfg), **no_processing
        )
        tokens = torch.randint(3, cfg.d_vocab, (2, 12))
        with torch.no_grad():
            ref = hf(input_ids=tokens).logits
            out = model(tokens)
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
