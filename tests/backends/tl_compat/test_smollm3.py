"""Offline Group A-E: SmolLM3-3B (``SmolLM3ForCausalLM``) registration and
conversion.

Unlike Command R7B / Aya Expanse (both extensions of the existing cohere
family port), SmolLM3-3B is a structurally different, Llama-family
architecture with its own new port module (``backends/_tl_compat/smollm3.py``):
RMSNorm (not LayerNorm), a sequential attn+MLP block (not cohere's parallel
one), standard non-interleaved Llama rotary, per-layer NoPE driven by the HF
config's ``no_rope_layers`` list, and no logit scale at all. This file covers:

* the repo ID is registered correctly, through its own independently
  registered ``ArchPort`` (``SmolLM3ForCausalLM``);
* the config-driven path produces the right ``HookedTransformerConfig`` for
  SmolLM3-3B's *real* spec (36 layers, 2048 d_model, 16/4 GQA heads,
  d_head=128, d_mlp=11008, vocab=128256, rope_theta=5000000, RMSNorm,
  sequential block, non-interleaved rotary, tied embeddings);
* SmolLM3's HF config does not always set an explicit ``head_dim`` (like
  cohere1, unlike cohere2), so the config converter's
  ``hidden_size // num_attention_heads`` fallback is pinned;
* **the per-layer NoPE mask**: the registered rotary policy skips rotary
  exactly on the layers where ``no_rope_layers[layer_id] == 0`` and applies it
  everywhere else -- the guide's #1 exit-gate item for this stage;
* **RMSNorm, not LayerNorm**: ``normalization_type == "RMS"`` /
  ``final_rms is True``, and the weight converter emits no LN bias tensors at
  all (unlike the cohere family's explicit zero-bias trick, which does not
  apply here -- RMSNorm has no bias term to satisfy in the first place);
* GQA weight slicing (``_W_K``/``_W_V`` at the KV head count, not the query
  head count);
* three converter guards specific to this architecture (``attention_bias``,
  ``mlp_bias``, ``use_sliding_window``) raise ``NotImplementedError`` rather
  than silently producing a wrong model;
* a tiny random-init forward pass is finite.

Fully offline: no gated weights, no network, no GPU. Numerical parity against
the real checkpoint is the separate, network-gated ``test_smollm3_parity.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import transformer_lens.loading_from_pretrained as tl_loading
from transformer_lens import HookedTransformerConfig

from circuitkit.backends._tl_compat import registry as _registry
from circuitkit.backends._tl_compat import smollm3 as smollm3_patch

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

# SmolLM3-3B's real, confirmed spec (AutoConfig.from_pretrained(MODEL_NAME),
# checked directly in this venv rather than assumed -- see Part 0.2 of the
# integration guide): 36 layers, d_model 2048, 16 query / 4 KV heads, d_head
# 128 (hidden_size // n_heads, not an explicit config field), d_mlp 11008
# (gated SiLU), vocab 128256, rope_theta 5000000, rms_norm_eps 1e-6, tied
# embeddings, max_position_embeddings 65536 (> the port's _MAX_SAFE_N_CTX
# cap), no_rope_layers = [1,1,1,0] * 9 (NoPE every 4th layer, 0-indexed at
# positions 3, 7, 11, ...).
_N_LAYERS = 36
_NO_ROPE_LAYERS = [1, 1, 1, 0] * (_N_LAYERS // 4)
assert len(_NO_ROPE_LAYERS) == _N_LAYERS


def _make_smollm3_hf_config(**overrides):
    """A ``SimpleNamespace``-backed fake HF config -- deliberately *not* a
    ``MagicMock``: the converter relies on ``getattr(cfg, "head_dim", None)``
    falling through to a real default when the attribute is absent, which a
    ``MagicMock`` would defeat by auto-vivifying any attribute access.
    """
    defaults = dict(
        architectures=["SmolLM3ForCausalLM"],
        model_type="smollm3",
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=11008,
        num_hidden_layers=_N_LAYERS,
        max_position_embeddings=65536,
        rms_norm_eps=1e-6,
        vocab_size=128256,
        hidden_act="silu",
        rope_theta=5000000.0,
        attention_bias=False,
        mlp_bias=False,
        use_sliding_window=False,
        no_rope_layers=list(_NO_ROPE_LAYERS),
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSmolLM3Registration:
    def test_repo_id_registered(self):
        assert MODEL_NAME in tl_loading.OFFICIAL_MODEL_NAMES
        assert tl_loading.get_official_model_name(MODEL_NAME) == MODEL_NAME

    def test_repo_id_is_in_smollm3_model_names(self):
        assert smollm3_patch.SMOLLM3_MODEL_NAMES == [MODEL_NAME]

    def test_case_insensitive_lookup(self):
        assert tl_loading.get_official_model_name(MODEL_NAME.lower()) == MODEL_NAME

    def test_resolves_to_its_own_distinct_architecture_port(self):
        smollm3_port = _registry.resolve_by_architecture("SmolLM3ForCausalLM")
        cohere2_port = _registry.resolve_by_architecture("Cohere2ForCausalLM")
        assert smollm3_port is not None
        assert cohere2_port is not None
        assert smollm3_port is not cohere2_port
        assert smollm3_port.architecture == "SmolLM3ForCausalLM"
        assert smollm3_port.config_converter is smollm3_patch._convert_smollm3_config
        assert smollm3_port.weight_converter is smollm3_patch.convert_smollm3_weights


class TestSmolLM3ConfigConversion:
    def test_real_spec_builds_correct_cfg_dict(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)

        assert cfg_dict["d_model"] == 2048
        assert cfg_dict["n_heads"] == 16
        assert cfg_dict["n_key_value_heads"] == 4
        assert cfg_dict["d_head"] == 128
        assert cfg_dict["d_mlp"] == 11008
        assert cfg_dict["n_layers"] == 36
        assert cfg_dict["d_vocab"] == 128256
        assert cfg_dict["rotary_base"] == 5000000
        assert cfg_dict["rotary_dim"] == 128
        assert cfg_dict["eps"] == 1e-6
        assert cfg_dict["original_architecture"] == "SmolLM3ForCausalLM"
        assert cfg_dict["tokenizer_name"] == MODEL_NAME
        # No logit_scale for SmolLM3 -- unlike the whole cohere family.
        assert "logit_scale" not in cfg_dict

    def test_real_context_length_is_capped(self):
        """max_position_embeddings=65536 exceeds the port's _MAX_SAFE_N_CTX
        (8192) -- must be capped, not padded up (the OOM-avoidance case, same
        as Command R7B's 132096; unlike Aya Expanse's 8192, which sits at the
        cap exactly)."""
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128)
        assert fake_hf_cfg.max_position_embeddings == 65536
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["n_ctx"] == smollm3_patch._MAX_SAFE_N_CTX == 8192

    def test_head_dim_fallback_when_config_omits_it(self):
        """SmolLM3Config does not always set an explicit head_dim (confirmed
        via AutoConfig.from_pretrained on the real checkpoint); the fake
        config here has no head_dim attribute at all (SimpleNamespace raises
        AttributeError on access, unlike a MagicMock), so this only passes if
        the converter does the hidden_size // num_attention_heads fallback
        rather than assuming the attribute exists."""
        fake_hf_cfg = _make_smollm3_hf_config()  # no head_dim override
        assert not hasattr(fake_hf_cfg, "head_dim")
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["d_head"] == 2048 // 16 == 128
        assert cfg_dict["rotary_dim"] == 128

    def test_rejects_attention_bias(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128, attention_bias=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="attention_bias"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_mlp_bias(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128, mlp_bias=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="mlp_bias"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_use_sliding_window(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128, use_sliding_window=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="use_sliding_window"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_wrong_architecture(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128, architectures=["LlamaForCausalLM"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="SmolLM3ForCausalLM"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_rejects_mismatched_no_rope_layers_length(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128, no_rope_layers=[1, 0, 1])
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="no_rope_layers"):
                tl_loading.convert_hf_model_config(MODEL_NAME)

    def test_cfg_dict_constructs_a_valid_hooked_transformer_config(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        cfg_dict["model_name"] = "smollm3-3b"
        cfg_dict["init_weights"] = False
        cfg_dict["dtype"] = torch.float32
        cfg_dict["device"] = "cpu"

        cfg = HookedTransformerConfig.from_dict(cfg_dict)

        assert cfg.n_layers == 36
        assert cfg.n_heads == 16
        assert cfg.n_key_value_heads == 4
        assert cfg.d_head == 128
        assert cfg.use_local_attn is False
        assert cfg.attn_types is None


class TestSmolLM3RmsNormNotLayerNorm:
    """The guide's second guide-mandated exit-gate item: SmolLM3 uses
    RMSNorm, not the cohere family's bias-less LayerNorm."""

    def test_normalization_type_is_rms(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        assert cfg_dict["normalization_type"] == "RMS"
        assert cfg_dict["final_rms"] is True
        # Sanity: this is genuinely different from the whole cohere family.
        assert cfg_dict["normalization_type"] != "LN"

    def test_weight_converter_emits_no_ln_bias_tensors(self):
        """convert_cohere2_weights emits explicit zero ln1.b/ln2.b/ln_final.b
        tensors (LayerNorm needs a bias TL's fold_layer_norm keys on
        unconditionally); convert_smollm3_weights must emit none at all --
        RMSNorm has no bias term to satisfy in the first place, and porting
        the cohere zero-bias trick over would be actively wrong here."""
        dims = dict(n_layers=2, n_heads=4, n_kv_heads=2, d_head=2, d_model=8, d_mlp=16, d_vocab=10)
        model = _build_fake_smollm3(dims)
        cfg = _FakeSmolLM3TLCfg(**dims)
        sd = smollm3_patch.convert_smollm3_weights(model, cfg)

        for key in sd:
            assert (
                not key.endswith(".b") or "ln" not in key
            ), f"unexpected LN bias tensor {key!r} in a RMSNorm state dict"
        assert "blocks.0.ln1.b" not in sd
        assert "blocks.0.ln2.b" not in sd
        assert "ln_final.b" not in sd
        assert "blocks.0.ln1.w" in sd
        assert "blocks.0.ln2.w" in sd
        assert "ln_final.w" in sd


class TestSmolLM3PerLayerNoPE:
    """The guide's #1 exit-gate item for this stage: NoPE layers (where
    no_rope_layers[layer_id] == 0) must skip rotary; every other layer must
    apply it. Unit-level check of the registered policy after a real
    ``_convert_smollm3_config`` call (which populates the module-level
    no_rope_layers cache the policy reads); the real-weight Group F gate
    (test_smollm3_parity.py) proves it end to end against actual HF outputs.
    """

    @pytest.fixture(autouse=True)
    def _populate_no_rope_cache(self):
        fake_hf_cfg = _make_smollm3_hf_config(head_dim=128)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            self.cfg_dict = tl_loading.convert_hf_model_config(MODEL_NAME)
        self.cfg_dict["model_name"] = "smollm3-3b"
        self.cfg_dict["init_weights"] = False
        self.cfg_dict["dtype"] = torch.float32
        self.cfg_dict["device"] = "cpu"
        self.cfg = HookedTransformerConfig.from_dict(self.cfg_dict)

    @pytest.mark.parametrize("layer_id", [3, 7, 11, 15, 19, 23, 27, 31, 35])
    def test_nope_layers_skip_rotary(self, layer_id):
        assert _NO_ROPE_LAYERS[layer_id] == 0
        assert _registry.should_skip_rotary(self.cfg, layer_id, "global") is True

    @pytest.mark.parametrize("layer_id", [0, 1, 2, 4, 5, 6, 8, 9, 10, 34])
    def test_other_layers_apply_rotary(self, layer_id):
        assert _NO_ROPE_LAYERS[layer_id] == 1
        assert _registry.should_skip_rotary(self.cfg, layer_id, "global") is False

    def test_full_mask_matches_no_rope_layers_exactly(self):
        mask = [_registry.should_skip_rotary(self.cfg, i, "global") for i in range(_N_LAYERS)]
        expected = [v == 0 for v in _NO_ROPE_LAYERS]
        assert mask == expected

    def test_unknown_model_defaults_to_never_skip(self):
        """A HookedTransformerConfig built without going through
        _convert_smollm3_config (e.g. a hand-built unit-test config) has no
        entry in the no_rope_layers cache -- must default to "apply rotary",
        stock TL behaviour, not raise or silently skip everything."""
        unseen_cfg = SimpleNamespace(
            original_architecture="SmolLM3ForCausalLM",
            tokenizer_name="some/unregistered-smollm3-variant",
        )
        for layer_id in (0, 3, 7, 35):
            assert _registry.should_skip_rotary(unseen_cfg, layer_id, "global") is False


class _FakeSmolLM3TLCfg:
    def __init__(self, n_layers, n_heads, n_kv_heads, d_head, d_model, d_mlp, d_vocab):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_key_value_heads = n_kv_heads
        self.d_head = d_head
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_vocab = d_vocab
        self.dtype = torch.float32
        self.original_architecture = "SmolLM3ForCausalLM"


def _build_fake_smollm3(dims):
    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(dims["d_model"], dims["n_heads"] * dims["d_head"], bias=False)
            self.k_proj = nn.Linear(
                dims["d_model"], dims["n_kv_heads"] * dims["d_head"], bias=False
            )
            self.v_proj = nn.Linear(
                dims["d_model"], dims["n_kv_heads"] * dims["d_head"], bias=False
            )
            self.o_proj = nn.Linear(dims["n_heads"] * dims["d_head"], dims["d_model"], bias=False)

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(dims["d_model"], dims["d_mlp"], bias=False)
            self.up_proj = nn.Linear(dims["d_model"], dims["d_mlp"], bias=False)
            self.down_proj = nn.Linear(dims["d_mlp"], dims["d_model"], bias=False)

    class FakeRMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dims["d_model"]))

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()
            self.mlp = FakeMLP()
            self.input_layernorm = FakeRMSNorm()
            self.post_attention_layernorm = FakeRMSNorm()

    class FakeBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(dims["d_vocab"], dims["d_model"])
            self.layers = nn.ModuleList([FakeLayer() for _ in range(dims["n_layers"])])
            self.norm = FakeRMSNorm()

    class FakeSmolLM3(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBaseModel()
            self.lm_head = nn.Linear(dims["d_model"], dims["d_vocab"], bias=False)

    return FakeSmolLM3()


class TestSmolLM3WeightConversion:
    _DIMS = dict(n_layers=2, n_heads=4, n_kv_heads=2, d_head=2, d_model=8, d_mlp=16, d_vocab=10)

    def test_no_logit_scale_fold(self):
        """Unlike the whole cohere family, SmolLM3 has no logit_scale at
        all -- unembed.W_U must be exactly lm_head.weight.T, no scale."""
        model = _build_fake_smollm3(self._DIMS)
        cfg = _FakeSmolLM3TLCfg(**self._DIMS)
        sd = smollm3_patch.convert_smollm3_weights(model, cfg)
        assert torch.equal(sd["unembed.W_U"], model.lm_head.weight.T)

    def test_gqa_weight_shapes(self):
        """GQA (16 query / 4 KV heads on the real model, 4/2 here) --
        _W_K/_W_V must carry the KV head count, not the query head count."""
        model = _build_fake_smollm3(self._DIMS)
        cfg = _FakeSmolLM3TLCfg(**self._DIMS)
        sd = smollm3_patch.convert_smollm3_weights(model, cfg)
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

    def test_gated_mlp_weight_names(self):
        model = _build_fake_smollm3(self._DIMS)
        cfg = _FakeSmolLM3TLCfg(**self._DIMS)
        sd = smollm3_patch.convert_smollm3_weights(model, cfg)
        assert torch.equal(sd["blocks.0.mlp.W_gate"], model.model.layers[0].mlp.gate_proj.weight.T)
        assert torch.equal(sd["blocks.0.mlp.W_in"], model.model.layers[0].mlp.up_proj.weight.T)
        assert torch.equal(sd["blocks.0.mlp.W_out"], model.model.layers[0].mlp.down_proj.weight.T)

    def test_tied_embeddings_share_storage(self):
        """tie_word_embeddings=True is handled by HF's own post_init() before
        the converter ever runs -- lm_head.weight already IS embed_tokens.
        weight, so no special tying logic is needed here (same contract as
        the cohere family)."""
        model = _build_fake_smollm3(self._DIMS)
        model.lm_head.weight = model.model.embed_tokens.weight  # simulate HF's tie_weights()
        cfg = _FakeSmolLM3TLCfg(**self._DIMS)
        sd = smollm3_patch.convert_smollm3_weights(model, cfg)
        assert torch.equal(sd["embed.W_E"], model.model.embed_tokens.weight)
        assert torch.equal(sd["unembed.W_U"], model.model.embed_tokens.weight.T)


class TestSmolLM3TinyRandomInitForward:
    """A tiny, random-init HookedTransformer built from SmolLM3's own config
    shape must produce a finite forward pass, and its per-layer rotary
    behaviour must match a hand-built no_rope_layers mask registered via the
    module-level cache (mirroring how _convert_smollm3_config populates it)."""

    def test_forward_is_finite_and_nope_layers_skip_rotary(self):
        no_rope = (1, 1, 1, 0, 1, 1, 1, 0)  # 8 layers, NoPE at index 3 and 7
        # tokenizer_name must resolve to a real, already-cacheable tokenizer
        # for HookedTransformer's own AutoTokenizer.from_pretrained call
        # (mirrors test_aya_expanse.py's tiny-forward test using "gpt2" for
        # the same reason); the NoPE cache is keyed off this same string.
        tokenizer_name = "gpt2"
        smollm3_patch._NO_ROPE_LAYERS_BY_MODEL[tokenizer_name] = no_rope

        cfg_dict = dict(
            d_model=32,
            d_head=8,
            n_heads=4,
            n_key_value_heads=2,
            d_mlp=64,
            n_layers=8,
            n_ctx=64,
            eps=1e-6,
            d_vocab=97,
            act_fn="silu",
            gated_mlp=True,
            normalization_type="RMS",
            final_rms=True,
            positional_embedding_type="rotary",
            rotary_adjacent_pairs=False,
            rotary_dim=8,
            rotary_base=5000000,
            use_local_attn=False,
            parallel_attn_mlp=False,
            use_attn_scale=True,
            attention_dir="causal",
            original_architecture="SmolLM3ForCausalLM",
            tokenizer_name=tokenizer_name,
            model_name="tiny-smollm3",
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

        xin = torch.randn(1, 5, cfg.n_heads, cfg.d_head)
        for i, block in enumerate(model.blocks):
            result = block.attn.apply_rotary(xin)
            if no_rope[i] == 0:
                assert result is xin, f"block {i}: expected rotary to be skipped (NoPE)"
            else:
                assert result is not xin, f"block {i}: expected rotary to be applied"

        del smollm3_patch._NO_ROPE_LAYERS_BY_MODEL[tokenizer_name]
