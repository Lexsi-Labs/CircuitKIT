"""The cohere port for TransformerLens 2.18 / 3.8: tiny-aya, Command R7B (cohere2)
and Aya Expanse 8B (cohere1).

Registers the tiny-aya, Command R7B and Aya Expanse repo IDs as known
TransformerLens models and teaches ``transformer_lens.loading_from_pretrained``
how to build a ``HookedTransformerConfig`` and a converted state dict for
them, plus patches ``AbstractAttention.apply_rotary`` so each architecture's
NoPE layers (if any) skip rotary, matching the corresponding HF
``modeling_cohere*.py``.

tiny-aya and Command R7B share the ``Cohere2ForCausalLM`` architecture
("cohere2") and are handled by the exact same config/weight converters below
-- everything is read off the HF config generically (sliding window,
layer_types, rope_theta, logit_scale), so Command R7B needed no new converter
logic, only its repo ID registered alongside tiny-aya's. The one behavioural
difference between them is ``logit_scale``: tiny-aya ships 1.0 (a no-op fold
into ``W_U``), Command R7B ships 0.25 (a real fold) -- see
``convert_cohere2_weights``.

Aya Expanse 8B is the plain ``CohereForCausalLM`` architecture ("cohere1") --
a simplification of cohere2, not an extension: no sliding window, no
per-layer ``attn_types``, and rotary applies to *every* layer (no NoPE at
all). It is otherwise structurally identical (parallel attn+MLP block with a
single input LayerNorm, GQA, tied embeddings, interleaved rotary, a
post-unembed ``logit_scale`` fold -- here 0.125), so it reuses
``convert_cohere2_weights`` unchanged for weight conversion; only the config
converter and NoPE policy differ. See ``_convert_cohere1_config`` /
``_cohere1_should_skip_rotary``.

Every design decision below is traceable to HF's ``transformers.models.cohere2``
/ ``transformers.models.cohere`` source (the ground truth for the
architecture) rather than assumption:

* ``Cohere2Attention.forward`` only calls ``apply_rotary_pos_emb`` when
  ``self.sliding_window is not None`` — i.e. RoPE is applied on
  sliding-window ("local") layers only; full-attention ("global") layers get
  no positional embedding at all (NoPE).
* ``rotate_half`` in ``modeling_cohere2.py`` splits ``x[..., ::2]`` /
  ``x[..., 1::2]`` (interleaved pairs) rather than first-half/second-half —
  this is GPT-J-style rotary, i.e. TL's ``rotary_adjacent_pairs=True``.
* ``Cohere2DecoderLayer`` has a single ``input_layernorm`` whose output feeds
  *both* ``self_attn`` and ``mlp``, with
  ``resid_post = resid_pre + attn_out + mlp_out``. TL models this as a
  parallel block (``parallel_attn_mlp=True``) with ``ln1`` and ``ln2`` tied to
  identical weights — the same trick TL already uses for GPT-J.
* ``Cohere2ForCausalLM.forward`` computes ``logits = lm_head(hidden) *
  logit_scale``. Since ``lm_head`` has no bias, this is a pure linear fold:
  the weight converter multiplies ``W_U`` by ``logit_scale`` at conversion
  time rather than threading a new config field through TL.
* ``tie_word_embeddings=True`` is handled by HF's own ``post_init()``/
  ``tie_weights()`` before we ever see the model, so ``lm_head.weight``
  already equals ``embed_tokens.weight`` — no special tying logic is needed
  in the converter.
* GQA weight storage follows the same ``_W_K``/``_W_V`` (underscore-prefixed)
  convention TL's own Gemma-2 converter uses; TL's
  ``GroupedQueryAttention`` module handles the ungrouping transparently.

Aya Expanse / cohere1-specific facts, from ``transformers.models.cohere.modeling_cohere``:

* ``CohereAttention.forward`` calls ``apply_rotary_pos_emb`` unconditionally
  for every layer -- there is no ``sliding_window`` attribute or per-layer
  branch at all in this architecture, unlike cohere2. Rotary therefore
  applies everywhere; the NoPE policy for this architecture always returns
  ``False``.
* ``rotate_half`` in ``modeling_cohere.py`` is byte-identical to cohere2's:
  ``x[..., ::2]`` / ``x[..., 1::2]`` interleaved pairs, i.e. also
  ``rotary_adjacent_pairs=True`` -- confirmed from source, not assumed.
* ``CohereConfig`` does not always set an explicit ``head_dim`` attribute
  (unlike ``Cohere2Config``); ``CohereAttention.__init__`` falls back to
  ``hidden_size // num_attention_heads`` via ``getattr(config, "head_dim",
  ...)`` when absent. The config converter below does the same.
* ``CohereAttention`` optionally applies QK-LayerNorm when
  ``config.use_qk_norm`` is set (a knob some cohere-family checkpoints use);
  Aya Expanse 8B ships ``use_qk_norm=False`` and ``convert_cohere2_weights``
  does not convert ``q_norm``/``k_norm`` weights, so the config converter
  raises rather than silently dropping them if a future cohere1 checkpoint
  ever sets it.
* ``CohereDecoderLayer``/``CohereModel``/``CohereForCausalLM`` have the exact
  same attribute names and parallel-block/logit_scale-fold shape as their
  cohere2 counterparts, so ``convert_cohere2_weights`` needs no changes to
  handle cohere1 -- only the config converter (no sliding window/attn_types,
  different head_dim fallback) differs.

See ``docs/advanced/tiny-aya.md`` (the original cohere2 port writeup) and
``docs/advanced/experimental-models.md`` (Command R7B + Aya Expanse) for the
user-facing architecture tables, config->TL mapping, and opt-in test matrix.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict

import einops
import torch

from . import registry as _registry

# Repo IDs confirmed by the user; the GGUF variants are llama.cpp's quantized
# single-file format, which AutoModelForCausalLM.from_pretrained (what TL
# uses internally) cannot load — they're intentionally excluded.
SUPPORTED_MODEL_NAMES = [
    "CohereLabs/tiny-aya-base",
    "CohereLabs/tiny-aya-global",
    "CohereLabs/tiny-aya-earth",
    "CohereLabs/tiny-aya-fire",
    "CohereLabs/tiny-aya-water",
]
_SUPPORTED_MODEL_NAMES_LOWER = frozenset(name.lower() for name in SUPPORTED_MODEL_NAMES)

# Repo ID confirmed by the user. Public (unlike tiny-aya) but still flagged
# "gated=auto" on the Hub -- an account needs to accept Cohere's license
# click-through once before the safetensors are downloadable, same
# _resolve_hf_token() auth path as tiny-aya covers it after that.
COMMAND_R7B_MODEL_NAMES = [
    "CohereLabs/c4ai-command-r7b-12-2024",
]

# Repo ID confirmed by the user. Public (unlike tiny-aya) but still flagged
# "gated=auto" on the Hub, same license click-through / _resolve_hf_token()
# auth path as tiny-aya and Command R7B.
AYA_EXPANSE_MODEL_NAMES = [
    "CohereLabs/aya-expanse-8b",
]

_COHERE2_ARCHITECTURE = "Cohere2ForCausalLM"
_COHERE1_ARCHITECTURE = "CohereForCausalLM"

# Command R7B ships max_position_embeddings=132096. TL's AbstractAttention
# allocates a *dense* n_ctx x n_ctx causal mask per attention block
# (torch.tril(torch.ones((n_ctx, n_ctx)))) -- at the real value that is a
# ~65 GB float32 tensor for a single layer, which OOMs on ordinary hardware
# before the model even finishes constructing. TL's own stock converters hit
# this for other long-context models (Gemma-3 up to 131K, several others)
# and cap n_ctx in the returned config for exactly this reason ("capped due
# to memory issues" in transformer_lens.loading_from_pretrained). This value
# matches Gemma-3's own safe default and comfortably covers this port's
# sliding-window parity test (past the 4096-token window). Callers that need
# more can still raise it explicitly via
# ``HookedTransformer.from_pretrained(..., n_ctx=<value>)`` --
# get_pretrained_model_config applies that override *after* this converter
# runs, so it is never silently clamped back down.
_MAX_SAFE_N_CTX = 8192


def _resolve_hf_token() -> Any:
    """Pick the best HF auth to hand ``transformers`` for a gated-repo fetch.

    ``huggingface_hub`` precedence is ``HF_TOKEN`` env > cached login. Setting
    ``HF_TOKEN`` to an obviously-fake placeholder (e.g. ``your_token_here``)
    therefore silently poisons the fetch with a bogus explicit token that
    beats the good cached login and returns 401 on files the cached account
    can read fine. Ignore anything that does not look like a real HF token
    (``hf_...``) and fall back to ``token=True`` so ``huggingface_hub`` uses
    whichever cached auth is available. An unset ``HF_TOKEN`` yields
    ``token=None`` for the same fallback via ``transformers``' default.
    """
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    if env_token.startswith("hf_"):
        return env_token
    return None if not env_token else True


_COHERE2_MODEL_NAMES = SUPPORTED_MODEL_NAMES + COMMAND_R7B_MODEL_NAMES
_COHERE1_MODEL_NAMES = list(AYA_EXPANSE_MODEL_NAMES)


def register() -> None:
    """Wire the cohere1/cohere2 support into the live ``transformer_lens`` module objects."""
    import transformer_lens.loading_from_pretrained as loading
    from transformer_lens.components.abstract_attention import AbstractAttention

    for name in _COHERE2_MODEL_NAMES + _COHERE1_MODEL_NAMES:
        if name not in loading.OFFICIAL_MODEL_NAMES:
            loading.OFFICIAL_MODEL_NAMES.append(name)

    _registry.register_port(
        _registry.ArchPort(
            architecture=_COHERE2_ARCHITECTURE,
            model_names=tuple(_COHERE2_MODEL_NAMES),
            config_converter=_convert_cohere2_config,
            weight_converter=convert_cohere2_weights,
            should_skip_rotary=_cohere2_should_skip_rotary,
        )
    )
    _registry.register_port(
        _registry.ArchPort(
            architecture=_COHERE1_ARCHITECTURE,
            model_names=tuple(_COHERE1_MODEL_NAMES),
            config_converter=_convert_cohere1_config,
            # Same weight layout as cohere2 -- see the module docstring.
            weight_converter=convert_cohere2_weights,
            should_skip_rotary=_cohere1_should_skip_rotary,
        )
    )

    loading.convert_hf_model_config = _wrap_convert_hf_model_config(loading.convert_hf_model_config)
    loading.get_pretrained_state_dict = _wrap_get_pretrained_state_dict(
        loading.get_pretrained_state_dict
    )
    _patch_apply_rotary(AbstractAttention)


def _cohere2_should_skip_rotary(cfg: Any, layer_id: Any, layer_attn_type: str) -> bool:
    """NoPE: Cohere2's periodic full-attention ("global") layers get no
    rotary at all (see modeling_cohere2.py Cohere2Attention.forward, which
    only rotates q/k `if self.sliding_window is not None`)."""
    return layer_attn_type == "global"


def _cohere1_should_skip_rotary(cfg: Any, layer_id: Any, layer_attn_type: str) -> bool:
    """No NoPE: cohere1's ``CohereAttention.forward`` calls
    ``apply_rotary_pos_emb`` unconditionally for every layer -- there is no
    ``sliding_window``/per-layer branch in this architecture at all. Every
    layer resolves to TL's default ``attn_type == "global"`` (since cohere1
    sets no ``attn_types``), which would otherwise collide with cohere2's
    policy above (where "global" means NoPE); dispatching by resolved HF
    architecture, not just the attn_type string, is exactly why this needs
    its own registered policy rather than reusing cohere2's."""
    return False


def _wrap_convert_hf_model_config(original: Callable[..., dict]) -> Callable[..., dict]:
    def wrapped(model_name: str, **kwargs: Any) -> dict:
        port = _registry.resolve_by_model_name(model_name)
        if port is not None:
            return port.config_converter(model_name, **kwargs)
        return original(model_name, **kwargs)

    return wrapped


def _convert_cohere2_config(official_model_name: str, **kwargs: Any) -> dict:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        official_model_name,
        token=_resolve_hf_token(),
        **kwargs,
    )

    architecture = hf_config.architectures[0]
    if architecture != _COHERE2_ARCHITECTURE:
        raise NotImplementedError(
            f"{official_model_name!r} is registered as a tiny-aya model but resolved "
            f"to HF architecture {architecture!r}, expected {_COHERE2_ARCHITECTURE!r}. "
            "The circuitkit TL-compat patch only supports Cohere2ForCausalLM."
        )
    if hf_config.attention_bias:
        # tiny-aya ships attention_bias=False; convert_cohere2_weights below
        # assumes it and drops q/k/v/o biases unconditionally. A future
        # checkpoint with biased attention would need that converter extended
        # first rather than silently losing its bias weights.
        raise NotImplementedError(
            f"{official_model_name!r} has attention_bias=True, which "
            "convert_cohere2_weights does not yet support."
        )

    n_heads = hf_config.num_attention_heads
    n_kv_heads = hf_config.num_key_value_heads
    d_head = hf_config.head_dim  # Cohere2Config always sets this (hidden_size // n_heads)
    attn_types = [
        "global" if layer_type == "full_attention" else "local"
        for layer_type in hf_config.layer_types
    ]

    cfg_dict: Dict[str, Any] = {
        "d_model": hf_config.hidden_size,
        "d_head": d_head,
        "n_heads": n_heads,
        "n_key_value_heads": n_kv_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": min(hf_config.max_position_embeddings, _MAX_SAFE_N_CTX),
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "gated_mlp": True,
        "normalization_type": "LN",
        "positional_embedding_type": "rotary",
        "rotary_adjacent_pairs": True,  # rope_gptj: interleaved-pair rotary
        "rotary_dim": d_head,  # rotary_pct=1.0 -> full head rotated
        "rotary_base": int(_registry.rope_theta(hf_config)),
        "use_local_attn": True,
        "window_size": hf_config.sliding_window,
        "attn_types": attn_types,
        "parallel_attn_mlp": True,
        "use_attn_scale": True,
        "attention_dir": "causal",
        "final_rms": False,
    }
    cfg_dict["original_architecture"] = architecture
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict


def _convert_cohere1_config(official_model_name: str, **kwargs: Any) -> dict:
    """Config converter for cohere1 (``CohereForCausalLM``, e.g. Aya Expanse 8B).

    Deliberately a separate function from ``_convert_cohere2_config`` rather
    than a shared one with branches: cohere1's HF config has no
    ``sliding_window``/``layer_types`` attributes at all (referencing them
    unconditionally, as the cohere2 converter does, would raise
    ``AttributeError``), and does not always set an explicit ``head_dim``
    (see the module docstring). The two converters share no code today
    because there is nothing safe left to share once those field accesses are
    removed; they do share the weight converter.
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        official_model_name,
        token=_resolve_hf_token(),
        **kwargs,
    )

    architecture = hf_config.architectures[0]
    if architecture != _COHERE1_ARCHITECTURE:
        raise NotImplementedError(
            f"{official_model_name!r} is registered as a cohere1 (Aya Expanse) model but "
            f"resolved to HF architecture {architecture!r}, expected "
            f"{_COHERE1_ARCHITECTURE!r}. The circuitkit TL-compat patch only supports "
            "CohereForCausalLM on this path."
        )
    if hf_config.attention_bias:
        # Mirrors the cohere2 converter's guard: convert_cohere2_weights drops
        # q/k/v/o biases unconditionally, which would be silently wrong if a
        # future checkpoint on this path ships attention_bias=True.
        raise NotImplementedError(
            f"{official_model_name!r} has attention_bias=True, which "
            "convert_cohere2_weights does not yet support."
        )
    if getattr(hf_config, "use_qk_norm", False):
        # CohereAttention optionally applies q_norm/k_norm (see the module
        # docstring); convert_cohere2_weights does not convert those weights,
        # so silently proceeding would drop them and produce a subtly wrong
        # model rather than a loud failure. Aya Expanse 8B itself ships
        # use_qk_norm=False.
        raise NotImplementedError(
            f"{official_model_name!r} has use_qk_norm=True, which the circuitkit "
            "cohere1 weight converter does not yet support (q_norm/k_norm weights "
            "would be silently dropped)."
        )

    n_heads = hf_config.num_attention_heads
    n_kv_heads = hf_config.num_key_value_heads
    # Unlike Cohere2Config, CohereConfig does not always set an explicit
    # head_dim -- CohereAttention.__init__ falls back to
    # `hidden_size // num_attention_heads` via getattr(..., default) when
    # absent; mirror that fallback here rather than assuming the attribute
    # exists.
    d_head = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // n_heads)

    cfg_dict: Dict[str, Any] = {
        "d_model": hf_config.hidden_size,
        "d_head": d_head,
        "n_heads": n_heads,
        "n_key_value_heads": n_kv_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": hf_config.num_hidden_layers,
        "n_ctx": min(hf_config.max_position_embeddings, _MAX_SAFE_N_CTX),
        "eps": hf_config.layer_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "gated_mlp": True,
        "normalization_type": "LN",
        "positional_embedding_type": "rotary",
        "rotary_adjacent_pairs": True,  # rotate_half is interleaved, same as cohere2
        "rotary_dim": d_head,
        "rotary_base": int(_registry.rope_theta(hf_config)),
        # No sliding window / attn_types at all for cohere1 -- rotary applies
        # to every layer (see _cohere1_should_skip_rotary) and every layer
        # attends globally. Leaving use_local_attn at TL's own default
        # (False) and window_size/attn_types unset (None) is deliberate, not
        # an oversight -- setting attn_types here would need
        # use_local_attn=True per HookedTransformerConfig's own validation.
        "use_local_attn": False,
        "parallel_attn_mlp": True,
        "use_attn_scale": True,
        "attention_dir": "causal",
        "final_rms": False,
    }
    cfg_dict["original_architecture"] = architecture
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict


def _wrap_get_pretrained_state_dict(
    original: Callable[..., Dict[str, torch.Tensor]],
) -> Callable[..., Dict[str, torch.Tensor]]:
    def wrapped(
        official_model_name: str,
        cfg: Any,
        hf_model: Any = None,
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        port = _registry.resolve_by_architecture(getattr(cfg, "original_architecture", None))
        if port is None:
            return original(official_model_name, cfg, hf_model=hf_model, dtype=dtype, **kwargs)

        from transformers import AutoModelForCausalLM

        kwargs = dict(kwargs)
        if "torch_dtype" in kwargs:
            dtype = kwargs.pop("torch_dtype")
        kwargs.pop("hf_token", None)
        kwargs.pop("n_ctx", None)

        if hf_model is None:
            # tiny-aya ships safetensors only. Forcing ``use_safetensors=True``
            # skips ``transformers``'s alternative-format probe (``tf_model.h5``,
            # ``flax_model.msgpack``), which for gated repos returns HTTP 401 on
            # a missing file rather than 404 and is then treated as a repo
            # access failure — surfacing as a spurious "gated repository"
            # OSError even when the caller's token has full access to the
            # safetensors that do exist.
            kwargs.setdefault("use_safetensors", True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                official_model_name,
                torch_dtype=dtype,
                token=_resolve_hf_token(),
                **kwargs,
            )

        for param in hf_model.parameters():
            param.requires_grad = False

        return port.weight_converter(hf_model, cfg)

    return wrapped


def convert_cohere2_weights(cohere2: Any, cfg: Any) -> Dict[str, torch.Tensor]:
    """Convert a loaded HF ``Cohere2ForCausalLM`` into a TL ``HookedTransformer`` state dict."""
    assert cfg.n_key_value_heads is not None  # keep mypy happy; GQA is mandatory for cohere2
    assert cfg.d_mlp is not None

    state_dict: Dict[str, torch.Tensor] = {}
    base_model = cohere2.model
    # All the zero bias/LN tensors below must live on the same device as the
    # real weights they sit alongside in the state dict -- torch.zeros(...)
    # with no device= defaults to CPU, which is silently correct only when
    # cohere2 itself happens to be on CPU (true for every offline unit test
    # and for tiny-aya's default load path) but breaks TL's fold_layer_norm
    # ("Expected all tensors to be on the same device") the moment a caller
    # loads the HF model onto a GPU explicitly -- as Command R7B's real-weight
    # parity gate does, to keep the HF and TL copies on separate devices.
    device = next(cohere2.parameters()).device

    state_dict["embed.W_E"] = base_model.embed_tokens.weight

    for l in range(cfg.n_layers):
        layer = base_model.layers[l]

        # Cohere2 has a single input_layernorm feeding both the attention and
        # MLP branches (parallel_attn_mlp); TL models this as tied ln1/ln2,
        # the same trick it already uses for GPT-J's analogous single-LN
        # parallel block.
        ln_w = layer.input_layernorm.weight
        state_dict[f"blocks.{l}.ln1.w"] = ln_w
        state_dict[f"blocks.{l}.ln2.w"] = ln_w
        # Cohere2's CohereLayerNorm is a bias-less LayerNorm (mean-subtract +
        # variance-normalize + weight-scale, no bias term). TL's
        # ``normalization_type="LN"`` path plus ``fold_layer_norm`` still keys
        # off ``ln{1,2}.b`` unconditionally, so emit an explicit zero bias:
        # folding a zero bias is a no-op and matches the HF math exactly.
        ln_b = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=device)
        state_dict[f"blocks.{l}.ln1.b"] = ln_b
        state_dict[f"blocks.{l}.ln2.b"] = ln_b

        W_Q = layer.self_attn.q_proj.weight
        W_K = layer.self_attn.k_proj.weight
        W_V = layer.self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m -> n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=device
        )
        state_dict[f"blocks.{l}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=device
        )
        state_dict[f"blocks.{l}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=device
        )

        W_O = layer.self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h) -> n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O
        state_dict[f"blocks.{l}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=device
        )

        # Cohere2MLP hardcodes bias=False on all three projections.
        state_dict[f"blocks.{l}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{l}.mlp.W_in"] = layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_in"] = torch.zeros(cfg.d_mlp, dtype=cfg.dtype, device=device)
        state_dict[f"blocks.{l}.mlp.W_out"] = layer.mlp.down_proj.weight.T
        state_dict[f"blocks.{l}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=device
        )

    state_dict["ln_final.w"] = base_model.norm.weight
    state_dict["ln_final.b"] = torch.zeros(cfg.d_model, dtype=cfg.dtype, device=device)

    # logit_scale is applied to the *output* logits post lm_head in HF
    # (`logits = lm_head(hidden) * logit_scale`); since lm_head has no bias
    # this is a pure linear fold into W_U rather than a separate TL hook.
    logit_scale = float(getattr(cohere2, "logit_scale", 1.0))
    state_dict["unembed.W_U"] = cohere2.lm_head.weight.T * logit_scale
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=device)

    return state_dict


# apply_rotary can only be usefully patched once per class (every port doing
# so independently would nest one near-identical check per port); the shared
# _tl_compat.registry dispatcher resolves the per-architecture NoPE policy at
# call time instead. See registry.patch_apply_rotary's docstring.
_patch_apply_rotary = _registry.patch_apply_rotary
