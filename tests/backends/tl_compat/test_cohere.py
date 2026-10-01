"""
Comprehensive test suite for circuitkit's tiny-aya (cohere2) TransformerLens
compatibility patch (``src/circuitkit/backends/_tl_compat/``).

Scope: this file validates the *port itself* — model registration, config
conversion, weight conversion, and the NoPE rotary patch — using synthetic /
mocked HuggingFace objects and a tiny randomly-initialized HookedTransformer.
It requires a working ``transformer_lens`` install but needs no network
access, ``HF_TOKEN``, or GPU, and never touches the real (gated) tiny-aya
checkpoints. End-to-end parity against the real weights belongs to a later,
network-gated test (the plan's Group F ``test_cohere_parity.py``).

The synthetic HF configs below intentionally mirror the real tiny-aya spec
(36 layers, d_model 2048, 16 query / 4 KV heads, d_head 128, d_mlp 11008,
vocab 262144, rope_theta 50000, sliding_window 4096, SWA x3 : full x1 layer
pattern) so that shape/config assertions double as a check against the
actual architecture, not just arbitrary small numbers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

import transformer_lens.loading_from_pretrained as tl_loading
from transformer_lens import HookedTransformerConfig
from transformer_lens.components.abstract_attention import AbstractAttention

from circuitkit.backends import _tl_compat
from circuitkit.backends._tl_compat import cohere as cohere_patch

# Importing circuitkit.backends (transitively, above) already runs
# apply_patches() as a side effect of package initialization -- this is the
# exact code path production code goes through, so no extra setup is needed.


# --------------------------------------------------------------------------
# Shared fixtures / builders
# --------------------------------------------------------------------------


def _make_fake_hf_config(**overrides):
    """Build a MagicMock standing in for a real ``Cohere2Config``, defaulting
    to the actual tiny-aya spec (36 layers, SWA x3 : full x1, etc.)."""
    n_layers = overrides.pop("num_hidden_layers", 36)
    layer_types = overrides.pop(
        "layer_types",
        ["full_attention" if (i + 1) % 4 == 0 else "sliding_attention" for i in range(n_layers)],
    )
    defaults = dict(
        architectures=["Cohere2ForCausalLM"],
        model_type="cohere2",
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        intermediate_size=11008,
        num_hidden_layers=n_layers,
        max_position_embeddings=8192,
        layer_norm_eps=1e-5,
        vocab_size=262144,
        hidden_act="silu",
        rope_theta=50000.0,
        sliding_window=4096,
        layer_types=layer_types,
        attention_bias=False,
    )
    defaults.update(overrides)
    cfg = MagicMock()
    for key, value in defaults.items():
        setattr(cfg, key, value)
    return cfg


class _FakeTLCfg:
    """Minimal stand-in for a HookedTransformerConfig, for unit-testing the
    weight converter without constructing a real HookedTransformerConfig."""

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


def _build_fake_cohere2_model(n_layers, n_heads, n_kv_heads, d_head, d_model, d_mlp, d_vocab):
    """Build a real nn.Module tree matching HF's Cohere2ForCausalLM shape
    (embed_tokens / layers[i].{self_attn,mlp,input_layernorm} / norm /
    lm_head / logit_scale), with real (non-mock) parameter tensors so einops
    reshaping and value-level assertions are exercised for real."""

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
            self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
            self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
            self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    class FakeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
            self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
            self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    class FakeLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(d_model))

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()
            self.mlp = FakeMLP()
            self.input_layernorm = FakeLN()

    class FakeBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(d_vocab, d_model)
            self.layers = nn.ModuleList([FakeLayer() for _ in range(n_layers)])
            self.norm = FakeLN()

    class FakeCohere2(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBaseModel()
            self.lm_head = nn.Linear(d_model, d_vocab, bias=False)
            self.logit_scale = 0.0625

    return FakeCohere2()


TINY_DIMS = dict(n_layers=2, n_heads=4, n_kv_heads=2, d_head=2, d_model=8, d_mlp=16, d_vocab=10)


# --------------------------------------------------------------------------
# apply_patches() lifecycle
# --------------------------------------------------------------------------


class TestApplyPatchesLifecycle:
    def test_already_applied_by_package_import(self):
        assert _tl_compat._PATCHED is True

    def test_apply_patches_is_idempotent(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cohere_patch, "register", lambda: calls.append(1))
        _tl_compat.apply_patches()
        assert calls == [], "register() must not run again once _PATCHED is True"

    def test_rejects_unsupported_tl_version(self, monkeypatch):
        monkeypatch.setattr(_tl_compat, "_PATCHED", False)
        monkeypatch.setattr(_tl_compat.transformer_lens, "__version__", raising=False, value="3.0.0")
        with pytest.raises(RuntimeError, match="2.18"):
            _tl_compat.apply_patches()
        # Aborted before completion -- must not claim to be patched.
        assert _tl_compat._PATCHED is False

    def test_accepts_matching_minor_version(self, monkeypatch):
        monkeypatch.setattr(_tl_compat, "_PATCHED", False)
        monkeypatch.setattr(_tl_compat.transformer_lens, "__version__", raising=False, value="2.18.3")
        register_calls = []
        monkeypatch.setattr(cohere_patch, "register", lambda: register_calls.append(1))
        _tl_compat.apply_patches()
        assert register_calls == [1]
        assert _tl_compat._PATCHED is True

    def test_falls_back_to_packaging_metadata_when_module_version_missing(self, monkeypatch):
        """Some installs (observed on Colab) leave transformer_lens.__version__
        empty. Confirm we still resolve the version via importlib.metadata
        rather than failing the guard on that alone."""
        monkeypatch.setattr(_tl_compat, "_PATCHED", False)
        monkeypatch.setattr(_tl_compat.transformer_lens, "__version__", raising=False, value="")
        monkeypatch.setattr(_tl_compat.importlib.metadata, "version", lambda name: "2.18.0")
        register_calls = []
        monkeypatch.setattr(cohere_patch, "register", lambda: register_calls.append(1))
        _tl_compat.apply_patches()
        assert register_calls == [1]
        assert _tl_compat._PATCHED is True

    def test_reports_unknown_when_no_version_source_available(self, monkeypatch):
        monkeypatch.setattr(_tl_compat, "_PATCHED", False)
        monkeypatch.setattr(_tl_compat.transformer_lens, "__version__", raising=False, value="")

        def _raise(_name):
            raise _tl_compat.importlib.metadata.PackageNotFoundError

        monkeypatch.setattr(_tl_compat.importlib.metadata, "version", _raise)
        with pytest.raises(RuntimeError, match="unknown"):
            _tl_compat.apply_patches()
        assert _tl_compat._PATCHED is False


# --------------------------------------------------------------------------
# Model name registration
# --------------------------------------------------------------------------


class TestModelRegistration:
    @pytest.mark.parametrize("name", cohere_patch.SUPPORTED_MODEL_NAMES)
    def test_known_repo_id_registered(self, name):
        assert name in tl_loading.OFFICIAL_MODEL_NAMES
        assert tl_loading.get_official_model_name(name) == name

    def test_registration_is_exhaustive(self):
        assert set(cohere_patch.SUPPORTED_MODEL_NAMES) == {
            "CohereLabs/tiny-aya-base",
            "CohereLabs/tiny-aya-global",
            "CohereLabs/tiny-aya-earth",
            "CohereLabs/tiny-aya-fire",
            "CohereLabs/tiny-aya-water",
        }

    @pytest.mark.parametrize("name", cohere_patch.SUPPORTED_MODEL_NAMES)
    def test_gguf_variant_not_registered(self, name):
        assert f"{name}-GGUF" not in tl_loading.OFFICIAL_MODEL_NAMES

    def test_case_insensitive_lookup(self):
        assert (
            tl_loading.get_official_model_name("coherelabs/tiny-aya-base")
            == "CohereLabs/tiny-aya-base"
        )

    def test_unknown_variant_still_rejected(self):
        with pytest.raises(ValueError):
            tl_loading.get_official_model_name("CohereLabs/tiny-aya-mars")

    def test_registering_twice_does_not_duplicate(self):
        before = tl_loading.OFFICIAL_MODEL_NAMES.count("CohereLabs/tiny-aya-base")
        cohere_patch.register()
        after = tl_loading.OFFICIAL_MODEL_NAMES.count("CohereLabs/tiny-aya-base")
        assert before == after == 1


class TestFlatLoadModelAppliesPort:
    """Regression: ``ck.load_model`` must apply the tiny-aya port *before*
    ``HookedTransformer.from_pretrained`` resolves the model name.

    ``quick.load_model`` previously imported ``circuitkit.backends`` (which
    applies the port) only when an ``algorithm=`` was passed, so
    ``load_model("CohereLabs/tiny-aya-base")`` with no algorithm called
    from_pretrained before the cohere2 registration ran and TransformerLens
    rejected the model name. This pins the import and its ordering. (Behavioural
    assertion is impossible here: importing this test module already applied the
    port, so only source ordering distinguishes the fix from the pre-applied
    state.)"""

    def test_load_model_imports_backends_before_from_pretrained(self):
        import inspect

        from circuitkit import quick

        src = inspect.getsource(quick.load_model)
        assert "import backends" in src, (
            "quick.load_model must import circuitkit.backends to apply the "
            "tiny-aya / cohere2 TransformerLens port before from_pretrained"
        )
        assert src.index("import backends") < src.index("from_pretrained("), (
            "the backends import (which applies the port) must precede the "
            "from_pretrained call that resolves the model name"
        )


# --------------------------------------------------------------------------
# convert_hf_model_config (via the live-patched transformer_lens function)
# --------------------------------------------------------------------------


class TestConvertHfModelConfig:
    def test_known_model_builds_correct_cfg_dict(self):
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg) as mock_auto:
            cfg_dict = tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")

        mock_auto.assert_called_once()
        assert cfg_dict["d_model"] == 2048
        assert cfg_dict["n_heads"] == 16
        assert cfg_dict["n_key_value_heads"] == 4
        assert cfg_dict["d_head"] == 128
        assert cfg_dict["d_mlp"] == 11008
        assert cfg_dict["n_layers"] == 36
        assert cfg_dict["d_vocab"] == 262144
        assert cfg_dict["eps"] == 1e-5
        assert cfg_dict["act_fn"] == "silu"
        assert cfg_dict["gated_mlp"] is True
        assert cfg_dict["normalization_type"] == "LN"
        assert cfg_dict["positional_embedding_type"] == "rotary"
        assert cfg_dict["rotary_adjacent_pairs"] is True
        assert cfg_dict["rotary_dim"] == 128
        assert cfg_dict["rotary_base"] == 50000
        assert cfg_dict["use_local_attn"] is True
        assert cfg_dict["window_size"] == 4096
        assert cfg_dict["parallel_attn_mlp"] is True
        assert cfg_dict["use_attn_scale"] is True
        assert cfg_dict["original_architecture"] == "Cohere2ForCausalLM"
        assert cfg_dict["tokenizer_name"] == "CohereLabs/tiny-aya-base"

    def test_layer_types_mapped_to_attn_types(self):
        # sliding_window_pattern = 4 -> SWA, SWA, SWA, full, repeating.
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")

        attn_types = cfg_dict["attn_types"]
        assert len(attn_types) == 36
        assert attn_types[:4] == ["local", "local", "local", "global"]
        assert attn_types[4:8] == ["local", "local", "local", "global"]
        # every 4th layer (1-indexed) is the full-attention "global" layer
        for i, kind in enumerate(attn_types):
            expected = "global" if (i + 1) % 4 == 0 else "local"
            assert kind == expected, f"layer {i}: expected {expected}, got {kind}"

    def test_logit_scale_not_carried_as_cfg_field(self):
        # logit_scale is folded into W_U at weight-conversion time, not
        # threaded through as a HookedTransformerConfig field.
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")
        assert "logit_scale" not in cfg_dict

    def test_wrong_architecture_raises(self):
        fake_hf_cfg = _make_fake_hf_config(architectures=["LlamaForCausalLM"])
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="Cohere2ForCausalLM"):
                tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")

    def test_attention_bias_true_raises(self):
        fake_hf_cfg = _make_fake_hf_config(attention_bias=True)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            with pytest.raises(NotImplementedError, match="attention_bias"):
                tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")

    def test_hf_token_env_var_forwarded(self, monkeypatch):
        """A real-looking ``hf_...`` env token is forwarded verbatim."""
        monkeypatch.setenv("HF_TOKEN", "hf_realish_token_value")
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg) as mock_auto:
            tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")
        _, kwargs = mock_auto.call_args
        assert kwargs["token"] == "hf_realish_token_value"

    def test_no_hf_token_passes_none(self, monkeypatch):
        """An unset ``HF_TOKEN`` yields ``token=None`` (transformers default)."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg) as mock_auto:
            tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")
        _, kwargs = mock_auto.call_args
        assert kwargs["token"] is None

    def test_placeholder_hf_token_ignored_in_favour_of_cached(self, monkeypatch):
        """A non-``hf_`` env value (e.g. an accidentally-passed placeholder like
        ``your_token_here``) must NOT be forwarded verbatim -- doing so would
        override the good cached login with a bogus token and silently 401.
        Instead the resolver returns ``token=True`` so huggingface_hub uses
        whichever cached auth is available."""
        monkeypatch.setenv("HF_TOKEN", "your_token_here")
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg) as mock_auto:
            tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")
        _, kwargs = mock_auto.call_args
        assert kwargs["token"] is True

    def test_whitespace_hf_token_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "   ")
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg) as mock_auto:
            tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")
        _, kwargs = mock_auto.call_args
        assert kwargs["token"] is None

    @pytest.mark.parametrize("name", cohere_patch.SUPPORTED_MODEL_NAMES)
    def test_all_five_variants_resolve(self, name):
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config(name)
        assert cfg_dict["tokenizer_name"] == name

    def test_unrelated_model_is_not_intercepted(self):
        # Exercise the wrapper factory directly against a mock "original" so
        # this doesn't require a real network call for e.g. "gpt2".
        sentinel = {"sentinel": True}
        mock_original = MagicMock(return_value=sentinel)
        wrapped = cohere_patch._wrap_convert_hf_model_config(mock_original)

        result = wrapped("gpt2", trust_remote_code=False)

        mock_original.assert_called_once_with("gpt2", trust_remote_code=False)
        assert result is sentinel

    def test_cfg_dict_constructs_a_valid_hooked_transformer_config(self):
        """The strongest available offline check: feed the produced cfg_dict
        into the *real* HookedTransformerConfig and let its own __post_init__
        assertions (act_fn validity, window_size/attn_types consistency,
        etc.) validate it end-to-end."""
        fake_hf_cfg = _make_fake_hf_config()
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_hf_cfg):
            cfg_dict = tl_loading.convert_hf_model_config("CohereLabs/tiny-aya-base")
        cfg_dict["model_name"] = "tiny-aya-base"
        cfg_dict["init_weights"] = False
        cfg_dict["dtype"] = torch.float32
        cfg_dict["device"] = "cpu"

        cfg = HookedTransformerConfig.from_dict(cfg_dict)

        assert cfg.n_layers == 36
        assert cfg.n_heads == 16
        assert cfg.n_key_value_heads == 4
        assert cfg.d_head == 128
        assert cfg.rotary_dim == 128
        assert cfg.parallel_attn_mlp is True
        assert cfg.attn_types[3] == "global"
        assert cfg.attn_types[0] == "local"
        assert cfg.attn_scale == pytest.approx(128**0.5)


# --------------------------------------------------------------------------
# get_pretrained_state_dict (via the live-patched transformer_lens function)
# --------------------------------------------------------------------------


class TestGetPretrainedStateDict:
    def test_delegates_for_non_cohere2_architecture(self):
        mock_original = MagicMock(return_value={"sentinel": True})
        wrapped = cohere_patch._wrap_get_pretrained_state_dict(mock_original)
        fake_cfg = MagicMock(original_architecture="LlamaForCausalLM")

        result = wrapped("meta-llama/whatever", fake_cfg, hf_model=None, dtype=torch.float32)

        mock_original.assert_called_once_with(
            "meta-llama/whatever", fake_cfg, hf_model=None, dtype=torch.float32
        )
        assert result == {"sentinel": True}

    def test_fetches_and_converts_for_cohere2(self):
        fake_model = _build_fake_cohere2_model(**TINY_DIMS)
        fake_cfg = _FakeTLCfg(**TINY_DIMS)
        mock_original = MagicMock()
        wrapped = cohere_patch._wrap_get_pretrained_state_dict(mock_original)

        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model
        ) as mock_load:
            state_dict = wrapped("CohereLabs/tiny-aya-base", fake_cfg)

        mock_original.assert_not_called()
        mock_load.assert_called_once()
        assert state_dict["embed.W_E"].shape == (TINY_DIMS["d_vocab"], TINY_DIMS["d_model"])

    def test_reuses_provided_hf_model_without_refetching(self):
        fake_model = _build_fake_cohere2_model(**TINY_DIMS)
        fake_cfg = _FakeTLCfg(**TINY_DIMS)
        wrapped = cohere_patch._wrap_get_pretrained_state_dict(MagicMock())

        with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load:
            wrapped("CohereLabs/tiny-aya-base", fake_cfg, hf_model=fake_model)

        mock_load.assert_not_called()

    def test_sets_requires_grad_false_on_all_params(self):
        fake_model = _build_fake_cohere2_model(**TINY_DIMS)
        assert all(p.requires_grad for p in fake_model.parameters())
        fake_cfg = _FakeTLCfg(**TINY_DIMS)
        wrapped = cohere_patch._wrap_get_pretrained_state_dict(MagicMock())

        wrapped("CohereLabs/tiny-aya-base", fake_cfg, hf_model=fake_model)

        assert all(not p.requires_grad for p in fake_model.parameters())

    def test_torch_dtype_kwarg_is_consumed_not_forwarded_twice(self):
        fake_model = _build_fake_cohere2_model(**TINY_DIMS)
        fake_cfg = _FakeTLCfg(**TINY_DIMS)
        wrapped = cohere_patch._wrap_get_pretrained_state_dict(MagicMock())

        with patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=fake_model
        ) as mock_load:
            wrapped(
                "CohereLabs/tiny-aya-base",
                fake_cfg,
                hf_model=None,
                dtype=torch.float32,
                torch_dtype=torch.bfloat16,
            )

        _, kwargs = mock_load.call_args
        assert kwargs["torch_dtype"] == torch.bfloat16
        assert "hf_token" not in kwargs
        assert "n_ctx" not in kwargs


# --------------------------------------------------------------------------
# convert_cohere2_weights: shapes, value correctness, real-life edge cases
# --------------------------------------------------------------------------


class TestConvertCohere2Weights:
    def test_all_expected_keys_present_with_correct_shapes(self):
        dims = TINY_DIMS
        model = _build_fake_cohere2_model(**dims)
        cfg = _FakeTLCfg(**dims)

        sd = cohere_patch.convert_cohere2_weights(model, cfg)

        expected = {
            "embed.W_E": (dims["d_vocab"], dims["d_model"]),
            "unembed.W_U": (dims["d_model"], dims["d_vocab"]),
            "unembed.b_U": (dims["d_vocab"],),
            "ln_final.w": (dims["d_model"],),
            "ln_final.b": (dims["d_model"],),
        }
        for l in range(dims["n_layers"]):
            expected.update(
                {
                    f"blocks.{l}.ln1.w": (dims["d_model"],),
                    f"blocks.{l}.ln2.w": (dims["d_model"],),
                    f"blocks.{l}.ln1.b": (dims["d_model"],),
                    f"blocks.{l}.ln2.b": (dims["d_model"],),
                    f"blocks.{l}.attn.W_Q": (dims["n_heads"], dims["d_model"], dims["d_head"]),
                    f"blocks.{l}.attn._W_K": (
                        dims["n_kv_heads"],
                        dims["d_model"],
                        dims["d_head"],
                    ),
                    f"blocks.{l}.attn._W_V": (
                        dims["n_kv_heads"],
                        dims["d_model"],
                        dims["d_head"],
                    ),
                    f"blocks.{l}.attn.b_Q": (dims["n_heads"], dims["d_head"]),
                    f"blocks.{l}.attn._b_K": (dims["n_kv_heads"], dims["d_head"]),
                    f"blocks.{l}.attn._b_V": (dims["n_kv_heads"], dims["d_head"]),
                    f"blocks.{l}.attn.W_O": (dims["n_heads"], dims["d_head"], dims["d_model"]),
                    f"blocks.{l}.attn.b_O": (dims["d_model"],),
                    f"blocks.{l}.mlp.W_gate": (dims["d_model"], dims["d_mlp"]),
                    f"blocks.{l}.mlp.W_in": (dims["d_model"], dims["d_mlp"]),
                    f"blocks.{l}.mlp.b_in": (dims["d_mlp"],),
                    f"blocks.{l}.mlp.W_out": (dims["d_mlp"], dims["d_model"]),
                    f"blocks.{l}.mlp.b_out": (dims["d_model"],),
                }
            )

        assert set(sd) == set(expected), f"key mismatch: {set(sd) ^ set(expected)}"
        for key, shape in expected.items():
            assert tuple(sd[key].shape) == shape, f"{key}: expected {shape}, got {sd[key].shape}"

    def test_ln1_and_ln2_are_the_same_tensor(self):
        """parallel_attn_mlp semantics: Cohere2 has one input_layernorm
        feeding both branches, so TL needs ln1 == ln2 exactly (same trick
        already used for GPT-J)."""
        model = _build_fake_cohere2_model(**TINY_DIMS)
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        for l in range(TINY_DIMS["n_layers"]):
            assert sd[f"blocks.{l}.ln1.w"].data_ptr() == sd[f"blocks.{l}.ln2.w"].data_ptr()
            assert torch.equal(sd[f"blocks.{l}.ln1.w"], sd[f"blocks.{l}.ln2.w"])

    def test_all_biases_are_zero(self):
        """Cohere2's q/k/v/o (attention_bias=False, enforced upstream),
        gate/up/down (hardcoded bias=False in HF), and bias-less
        CohereLayerNorm all carry no bias at all -- the converter must
        emit explicit zero tensors so TL's fold_layer_norm can key off
        them, and must not invent nonzero values."""
        model = _build_fake_cohere2_model(**TINY_DIMS)
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        for l in range(TINY_DIMS["n_layers"]):
            for key in (
                f"blocks.{l}.attn.b_Q",
                f"blocks.{l}.attn._b_K",
                f"blocks.{l}.attn._b_V",
                f"blocks.{l}.attn.b_O",
                f"blocks.{l}.mlp.b_in",
                f"blocks.{l}.mlp.b_out",
                f"blocks.{l}.ln1.b",
                f"blocks.{l}.ln2.b",
            ):
                assert torch.equal(sd[key], torch.zeros_like(sd[key])), key
        assert torch.equal(sd["unembed.b_U"], torch.zeros_like(sd["unembed.b_U"]))
        assert torch.equal(sd["ln_final.b"], torch.zeros_like(sd["ln_final.b"]))

    def test_logit_scale_folded_into_unembed(self):
        model = _build_fake_cohere2_model(**TINY_DIMS)
        model.logit_scale = 0.0625
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        expected = model.lm_head.weight.T * 0.0625
        assert torch.allclose(sd["unembed.W_U"], expected)

    def test_logit_scale_of_one_is_a_true_no_op(self):
        """tiny-aya's actual spec: logit_scale=1.0, so W_U must come through
        byte-for-byte unscaled."""
        model = _build_fake_cohere2_model(**TINY_DIMS)
        model.logit_scale = 1.0
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        assert torch.equal(sd["unembed.W_U"], model.lm_head.weight.T)

    def test_missing_logit_scale_attribute_defaults_to_one(self):
        model = _build_fake_cohere2_model(**TINY_DIMS)
        del model.logit_scale
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        assert torch.equal(sd["unembed.W_U"], model.lm_head.weight.T)

    def test_mlp_weights_are_plain_transposes(self):
        model = _build_fake_cohere2_model(**TINY_DIMS)
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        layer = model.model.layers[0]
        assert torch.equal(sd["blocks.0.mlp.W_gate"], layer.mlp.gate_proj.weight.T)
        assert torch.equal(sd["blocks.0.mlp.W_in"], layer.mlp.up_proj.weight.T)
        assert torch.equal(sd["blocks.0.mlp.W_out"], layer.mlp.down_proj.weight.T)

    def test_W_Q_head_slicing_matches_hf_view_convention(self):
        """Catches axis-order bugs in the einops rearrange: HF splits
        q_proj's output into heads via `.view(n_heads, head_dim)`, i.e. row
        block [h*d_head:(h+1)*d_head] of q_proj.weight is head h."""
        dims = TINY_DIMS
        model = _build_fake_cohere2_model(**dims)
        layer0_q = model.model.layers[0].self_attn.q_proj
        with torch.no_grad():
            layer0_q.weight.copy_(
                torch.arange(
                    dims["n_heads"] * dims["d_head"] * dims["d_model"], dtype=torch.float32
                ).reshape(dims["n_heads"] * dims["d_head"], dims["d_model"])
            )
        cfg = _FakeTLCfg(**dims)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        W_Q = sd["blocks.0.attn.W_Q"]  # (n_heads, d_model, d_head)

        for h in range(dims["n_heads"]):
            expected = layer0_q.weight[h * dims["d_head"] : (h + 1) * dims["d_head"], :].T
            assert torch.equal(W_Q[h], expected), f"head {h} sliced incorrectly"

    def test_W_K_V_head_slicing_uses_kv_head_count_not_query_head_count(self):
        """GQA regression guard: K/V must be sliced by n_kv_heads, not
        n_heads -- using the wrong count either errors on reshape or
        silently scrambles which rows belong to which KV head."""
        dims = TINY_DIMS
        model = _build_fake_cohere2_model(**dims)
        layer0_k = model.model.layers[0].self_attn.k_proj
        with torch.no_grad():
            layer0_k.weight.copy_(
                torch.arange(
                    dims["n_kv_heads"] * dims["d_head"] * dims["d_model"], dtype=torch.float32
                ).reshape(dims["n_kv_heads"] * dims["d_head"], dims["d_model"])
            )
        cfg = _FakeTLCfg(**dims)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        W_K = sd["blocks.0.attn._W_K"]  # (n_kv_heads, d_model, d_head)

        assert W_K.shape[0] == dims["n_kv_heads"]
        for h in range(dims["n_kv_heads"]):
            expected = layer0_k.weight[h * dims["d_head"] : (h + 1) * dims["d_head"], :].T
            assert torch.equal(W_K[h], expected), f"kv head {h} sliced incorrectly"

    def test_W_O_head_slicing_matches_hf_concat_convention(self):
        """HF concatenates per-head outputs along columns before o_proj, so
        column block [h*d_head:(h+1)*d_head] of o_proj.weight is head h's
        output-projection contribution."""
        dims = TINY_DIMS
        model = _build_fake_cohere2_model(**dims)
        layer0_o = model.model.layers[0].self_attn.o_proj
        with torch.no_grad():
            layer0_o.weight.copy_(
                torch.arange(
                    dims["d_model"] * dims["n_heads"] * dims["d_head"], dtype=torch.float32
                ).reshape(dims["d_model"], dims["n_heads"] * dims["d_head"])
            )
        cfg = _FakeTLCfg(**dims)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        W_O = sd["blocks.0.attn.W_O"]  # (n_heads, d_head, d_model)

        for h in range(dims["n_heads"]):
            expected = layer0_o.weight[:, h * dims["d_head"] : (h + 1) * dims["d_head"]].T
            assert torch.equal(W_O[h], expected), f"head {h} sliced incorrectly"

    def test_layers_are_not_cross_contaminated(self):
        """Each layer's weights must come from that layer, not layer 0
        (a classic off-by-one / loop-variable-capture bug)."""
        dims = dict(TINY_DIMS, n_layers=3)
        model = _build_fake_cohere2_model(**dims)
        cfg = _FakeTLCfg(**dims)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        for l in range(dims["n_layers"]):
            layer = model.model.layers[l]
            assert torch.equal(sd[f"blocks.{l}.ln1.w"], layer.input_layernorm.weight)
            assert not torch.equal(
                sd[f"blocks.{l}.ln1.w"], model.model.layers[(l + 1) % dims["n_layers"]].input_layernorm.weight
            )

    def test_embed_and_unembed_read_from_their_own_weights(self):
        """embed.W_E and unembed.W_U are read independently (not forced
        equal) -- correct because HF's own tie_weights() already made
        lm_head.weight the same tensor as embed_tokens.weight when the
        checkpoint is tied; this converter must not assume tying itself."""
        model = _build_fake_cohere2_model(**TINY_DIMS)
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        assert torch.equal(sd["embed.W_E"], model.model.embed_tokens.weight)
        assert torch.equal(sd["unembed.W_U"], model.lm_head.weight.T * model.logit_scale)

    def test_ln_final_reads_final_norm(self):
        model = _build_fake_cohere2_model(**TINY_DIMS)
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        assert torch.equal(sd["ln_final.w"], model.model.norm.weight)

    def test_single_layer_and_single_head_edge_cases(self):
        """n_layers=1, n_heads=n_kv_heads=1 (no GQA) must not error."""
        dims = dict(n_layers=1, n_heads=1, n_kv_heads=1, d_head=4, d_model=4, d_mlp=8, d_vocab=5)
        model = _build_fake_cohere2_model(**dims)
        cfg = _FakeTLCfg(**dims)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        assert sd["blocks.0.attn.W_Q"].shape == (1, 4, 4)
        assert sd["blocks.0.attn._W_K"].shape == (1, 4, 4)

    def test_output_is_finite(self):
        model = _build_fake_cohere2_model(**TINY_DIMS)
        cfg = _FakeTLCfg(**TINY_DIMS)
        sd = cohere_patch.convert_cohere2_weights(model, cfg)
        for key, tensor in sd.items():
            assert torch.isfinite(tensor).all(), f"{key} has non-finite values"


# --------------------------------------------------------------------------
# NoPE rotary patch (unit-level, via _patch_apply_rotary directly)
# --------------------------------------------------------------------------


class TestPatchApplyRotaryUnit:
    """Exercises _patch_apply_rotary's gating logic in isolation, with a
    lightweight fake attention class -- avoids needing a fully constructed
    real AbstractAttention instance for the pure branching logic."""

    class _FakeAttn:
        def __init__(self, architecture, attn_type):
            self.cfg = MagicMock(original_architecture=architecture)
            self.attn_type = attn_type
            self.rotate_calls = 0

        def apply_rotary(self, x, past_kv_pos_offset=0, attention_mask=None):
            self.rotate_calls += 1
            return x + 100  # sentinel: "rotation applied"

    def _patched_class(self):
        cls = TestPatchApplyRotaryUnit._FakeAttn
        # patch a fresh copy of the class each time so tests don't leak into
        # each other via shared class-level monkeypatching
        patched_cls = type("FakeAttnCopy", (cls,), {})
        cohere_patch._patch_apply_rotary(patched_cls)
        return patched_cls

    def test_cohere2_global_layer_skips_rotation(self):
        cls = self._patched_class()
        attn = cls("Cohere2ForCausalLM", "global")
        x = torch.zeros(3)
        result = attn.apply_rotary(x)
        assert result is x
        assert attn.rotate_calls == 0

    def test_cohere2_local_layer_still_rotates(self):
        cls = self._patched_class()
        attn = cls("Cohere2ForCausalLM", "local")
        x = torch.zeros(3)
        result = attn.apply_rotary(x)
        assert result is not x
        assert torch.equal(result, x + 100)
        assert attn.rotate_calls == 1

    @pytest.mark.parametrize(
        "architecture", ["GPTNeoXForCausalLM", "Gemma2ForCausalLM", "LlamaForCausalLM", None]
    )
    def test_other_architectures_always_rotate_on_global(self, architecture):
        """Regression guard: the patch must be a strict no-op for every
        model other than cohere2, even ones that also use "global"/"local"
        attn_type naming (e.g. Gemma-2)."""
        cls = self._patched_class()
        attn = cls(architecture, "global")
        x = torch.zeros(3)
        result = attn.apply_rotary(x)
        assert result is not x
        assert attn.rotate_calls == 1

    def test_forwards_positional_arguments_correctly(self):
        cls = self._patched_class()
        attn = cls("Cohere2ForCausalLM", "local")

        captured = {}

        def spy_apply_rotary(self, x, past_kv_pos_offset=0, attention_mask=None):
            captured["past_kv_pos_offset"] = past_kv_pos_offset
            captured["attention_mask"] = attention_mask
            return x

        cls.apply_rotary = spy_apply_rotary  # will be re-wrapped below
        cohere_patch._patch_apply_rotary(cls)
        mask = torch.ones(1, 1)
        attn.apply_rotary(torch.zeros(2), past_kv_pos_offset=7, attention_mask=mask)
        assert captured["past_kv_pos_offset"] == 7
        assert captured["attention_mask"] is mask


class TestPatchApplyRotaryRealClass:
    """Confirms the *real* AbstractAttention class (as mutated by the actual
    apply_patches() call that ran on package import) was actually patched,
    without needing to instantiate the class."""

    def test_real_class_method_was_replaced(self):
        assert AbstractAttention.apply_rotary.__name__ == "patched_apply_rotary"


# --------------------------------------------------------------------------
# End-to-end: a real, tiny, randomly-initialized HookedTransformer
# --------------------------------------------------------------------------


class TestEndToEndTinyHookedTransformer:
    """The strongest offline check available: wires a tiny cohere2-shaped
    HookedTransformerConfig into a *real* HookedTransformer (random init, no
    gated weights, no network) and confirms it runs and that NoPE is
    actually observed on the real attention modules."""

    def _build(self):
        from transformer_lens import HookedTransformer

        cfg = HookedTransformerConfig(
            n_layers=4,
            d_model=16,
            n_ctx=32,
            d_head=4,
            n_heads=4,
            n_key_value_heads=2,
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
            parallel_attn_mlp=True,
            use_attn_scale=True,
            original_architecture="Cohere2ForCausalLM",
            init_weights=True,
            device="cpu",
            seed=0,
        )
        model = HookedTransformer(cfg)
        model.eval()
        return model

    def test_forward_pass_runs_and_is_finite(self):
        model = self._build()
        tokens = torch.randint(0, model.cfg.d_vocab, (2, 10))
        with torch.no_grad():
            logits = model(tokens)
        assert logits.shape == (2, 10, model.cfg.d_vocab)
        assert torch.isfinite(logits).all()

    def test_global_layer_apply_rotary_is_a_true_no_op(self):
        model = self._build()
        local_attn = model.blocks[0].attn
        global_attn = model.blocks[-1].attn
        assert local_attn.attn_type == "local"
        assert global_attn.attn_type == "global"

        x = torch.randn(1, 5, model.cfg.n_heads, model.cfg.d_head)
        local_result = local_attn.apply_rotary(x)
        global_result = global_attn.apply_rotary(x)

        # NoPE: the global layer must return the *exact same tensor object*.
        assert global_result is x
        # The local layer must have actually rotated (different values).
        assert local_result is not x
        assert not torch.equal(local_result, x)

    def test_all_local_layers_rotate_all_global_layers_do_not(self):
        model = self._build()
        x = torch.randn(1, 5, model.cfg.n_heads, model.cfg.d_head)
        for i, block in enumerate(model.blocks):
            result = block.attn.apply_rotary(x)
            if block.attn.attn_type == "global":
                assert result is x, f"block {i} (global) should skip rotation"
            else:
                assert result is not x, f"block {i} (local) should rotate"

    def test_position_shuffle_invariance_on_global_layer_output(self):
        """A layer with genuinely no positional information (NoPE) should
        treat q/k identically regardless of position -- apply_rotary on the
        global layer must be the identity function for *any* input, not
        just zeros."""
        model = self._build()
        global_attn = model.blocks[-1].attn
        x1 = torch.randn(1, 6, model.cfg.n_heads, model.cfg.d_head)
        x2 = torch.randn(1, 6, model.cfg.n_heads, model.cfg.d_head)
        assert global_attn.apply_rotary(x1) is x1
        assert global_attn.apply_rotary(x2) is x2
