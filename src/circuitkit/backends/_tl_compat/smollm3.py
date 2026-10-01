"""The SmolLM3 port for TransformerLens 2.18 / 3.8: SmolLM3-3B (``SmolLM3ForCausalLM``).

Registers ``HuggingFaceTB/SmolLM3-3B`` as a known TransformerLens model and
teaches ``transformer_lens.loading_from_pretrained`` how to build a
``HookedTransformerConfig`` and a converted state dict for it, plus a
per-layer NoPE (no positional embedding) policy consulted by
``AbstractAttention.apply_rotary`` -- matching HF's
``transformers.models.smollm3.modeling_smollm3``.

Unlike ``cohere.py`` (cohere1/cohere2, both LayerNorm + parallel-block +
interleaved-rotary architectures sharing one weight converter), SmolLM3 is a
structurally different, Llama-family architecture:

* **RMSNorm**, not LayerNorm -- ``SmolLM3RMSNorm`` has a single ``weight``
  parameter and no bias at all (unlike Cohere's bias-less-but-still-LayerNorm,
  which needs an explicit zero bias emitted for TL's ``fold_layer_norm`` to
  accept it). Nothing needs to be emitted here that HF doesn't itself have.
* **Sequential block** (``parallel_attn_mlp=False``, TL's default): a
  standard ``resid -> ln1 -> attn -> +resid -> ln2 -> mlp -> +resid`` Llama
  block, not cohere's single-LN parallel block.
* **Standard (non-interleaved) Llama rotary**: ``modeling_smollm3.py``'s
  ``rotate_half`` splits ``x[..., :d/2]`` / ``x[..., d/2:]`` (first-half /
  second-half), not the GPT-J-style interleaved-pair split cohere1/cohere2
  use. This is ``rotary_adjacent_pairs=False`` -- the single highest-risk,
  silently-wrong-output choice in this port (see the module's Group F parity
  test, which is the only thing that actually proves it).
* **Per-layer NoPE**: SmolLM3's own config carries a ``no_rope_layers`` list,
  one entry per transformer block, where ``SmolLM3Attention.__init__`` reads
  ``self.use_rope = config.no_rope_layers[layer_idx]`` -- i.e. **truthy (1)
  means rotary is applied on that layer, falsy (0) means it is skipped**
  (confirmed directly from the installed ``modeling_smollm3.py`` source, not
  assumed -- the guide's own architecture table only says "0 = skip", which
  this confirms is the correct reading, not "1 = skip"). For the real
  ``HuggingFaceTB/SmolLM3-3B`` checkpoint this list is
  ``[1,1,1,0] * 9`` (36 layers, one NoPE layer every 4th, matching
  ``no_rope_layer_interval=4``), but the converter reads the concrete list
  off the loaded config rather than assuming the interval pattern, so a
  future checkpoint with a different ``no_rope_layers`` list is handled
  correctly too.
* **No logit scale.** ``SmolLM3ForCausalLM.forward`` computes
  ``logits = lm_head(hidden_states)`` with no post-multiply at all (unlike
  the cohere family's ``logit_scale``); the weight converter therefore does
  no scale fold -- ``unembed.W_U`` is exactly ``lm_head.weight.T``.
* **Tied embeddings** (``tie_word_embeddings=True``): handled by HF's own
  ``post_init()``/``tie_weights()`` before we ever see the model, exactly
  like the cohere family -- no special tying logic needed here either.

Why the NoPE list can't simply be threaded through ``HookedTransformerConfig``
------------------------------------------------------------------------------
``HookedTransformerConfig`` is a plain dataclass; ``HookedTransformerConfig.
from_dict`` is a bare ``cls(**config_dict)``, so any key in the dict returned
by this module's config converter that isn't a declared dataclass field
raises ``TypeError`` before the model is even built -- there is no "extra
data" slot to stash ``no_rope_layers`` in for ``should_skip_rotary`` to read
back off ``cfg`` later (unlike cohere2, which reuses TL's own genuine
``attn_types`` field for its per-layer sliding-window/full split; repurposing
``attn_types`` for NoPE here isn't an option either, since setting it
requires ``use_local_attn=True``, which would engage TL's real local-window
attention masking -- not what a NoPE-only, all-full-attention model wants).
Instead, the config converter stashes the resolved per-layer list in a small
module-level cache keyed by ``official_model_name`` (the same string it
already writes to ``cfg.tokenizer_name``, a genuine TL config field), and the
registered rotary policy looks it up from there via ``cfg.tokenizer_name`` at
call time. This needs no further changes to ``_tl_compat/registry.py`` --
Stage 1's ``should_skip_rotary(cfg, layer_id, layer_attn_type)`` signature
already passes the block index a per-layer policy like this one needs.

See ``docs/advanced/experimental-models.md`` for the user-facing
architecture table, config->TL mapping, NoPE explanation, and opt-in test
matrix.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

import einops
import torch

from . import registry as _registry

# Repo ID confirmed by the user.
SMOLLM3_MODEL_NAMES = [
    "HuggingFaceTB/SmolLM3-3B",
]

_SMOLLM3_ARCHITECTURE = "SmolLM3ForCausalLM"

# SmolLM3-3B ships max_position_embeddings=65536. TL's AbstractAttention
# allocates a *dense* n_ctx x n_ctx causal mask per attention block -- at the
# real value that would be a large (>16 GB float32) tensor per layer, the
# same class of OOM the cohere.py port's own _MAX_SAFE_N_CTX cap (also 8192)
# exists to avoid for Command R7B's 132096 (see cohere.py's docstring for the
# full rationale, including the reference to TL's own stock converters doing
# the same thing for other long-context models). Duplicated here rather than
# imported from cohere.py: this constant is not exposed as shared
# _tl_compat infrastructure (it lives as a private module-level constant in
# cohere.py, not in registry.py), and mirroring the same safe value in a
# self-contained new port module is preferable to reaching into another
# port's private constant. Callers needing more can still pass
# ``HookedTransformer.from_pretrained(..., n_ctx=<value>)``, applied *after*
# this converter runs.
_MAX_SAFE_N_CTX = 8192

# Per official_model_name, the resolved ``no_rope_layers`` list (one entry per
# block; 0 = skip rotary, non-zero = apply it) -- see the module docstring for
# why this can't be threaded through the HookedTransformerConfig dict itself.
_NO_ROPE_LAYERS_BY_MODEL: Dict[str, Tuple[int, ...]] = {}


def _resolve_hf_token() -> Any:
    """Pick the best HF auth to hand ``transformers`` for a fetch.

    Mirrors ``cohere.py``'s ``_resolve_hf_token`` exactly (see its docstring
    for the placeholder-token rationale); duplicated rather than imported to
    keep this port self-contained.
    """
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    if env_token.startswith("hf_"):
        return env_token
    return None if not env_token else True


def register() -> None:
    """Wire SmolLM3 support into the live ``transformer_lens`` module objects."""
    import transformer_lens.loading_from_pretrained as loading
    from transformer_lens.components.abstract_attention import AbstractAttention

    for name in SMOLLM3_MODEL_NAMES:
        if name not in loading.OFFICIAL_MODEL_NAMES:
            loading.OFFICIAL_MODEL_NAMES.append(name)

    _registry.register_port(
        _registry.ArchPort(
            architecture=_SMOLLM3_ARCHITECTURE,
            model_names=tuple(SMOLLM3_MODEL_NAMES),
            config_converter=_convert_smollm3_config,
            weight_converter=convert_smollm3_weights,
            should_skip_rotary=_smollm3_should_skip_rotary,
        )
    )

    # These wrap whatever ``loading.convert_hf_model_config`` /
    # ``get_pretrained_state_dict`` currently are (chain-of-responsibility,
    # same as cohere.py) and dispatch generically through the shared
    # ``_tl_compat.registry`` by model name / resolved architecture -- not by
    # anything SmolLM3-specific. If ``cohere.register()`` has already run
    # (true whenever ``apply_patches()`` is the entry point, as it always is
    # today), its identically-shaped wrapping already handles SmolLM3 model
    # names too, since dispatch happens dynamically against the shared
    # registry rather than being baked into which module installed the
    # wrapper. Installing this module's own copy regardless keeps this port
    # correct and self-sufficient even if that call ordering ever changes --
    # the same harmless-redundancy contract ``registry.patch_apply_rotary``'s
    # own docstring documents for multiple ports patching the same method.
    loading.convert_hf_model_config = _wrap_convert_hf_model_config(loading.convert_hf_model_config)
    loading.get_pretrained_state_dict = _wrap_get_pretrained_state_dict(
        loading.get_pretrained_state_dict
    )
    _registry.patch_apply_rotary(AbstractAttention)


def _smollm3_should_skip_rotary(cfg: Any, layer_id: Optional[int], layer_attn_type: str) -> bool:
    """Per-layer NoPE: skip rotary exactly on the layers where the loaded
    config's ``no_rope_layers[layer_id] == 0`` (see the module docstring for
    why this list is looked up via ``cfg.tokenizer_name`` rather than being a
    ``HookedTransformerConfig`` field itself). Layers past the end of a
    shorter-than-expected list, or when the list isn't found at all (e.g. a
    ``HookedTransformerConfig`` built by hand in a unit test without going
    through ``_convert_smollm3_config``), default to "apply rotary" -- stock
    behaviour, matching the shared registry's own "never skip" default.
    """
    if layer_id is None:
        return False
    tokenizer_name = getattr(cfg, "tokenizer_name", None)
    if tokenizer_name is None:
        return False
    no_rope_layers = _NO_ROPE_LAYERS_BY_MODEL.get(tokenizer_name)
    if no_rope_layers is None or layer_id >= len(no_rope_layers):
        return False
    return no_rope_layers[layer_id] == 0


def _wrap_convert_hf_model_config(original: Callable[..., dict]) -> Callable[..., dict]:
    def wrapped(model_name: str, **kwargs: Any) -> dict:
        port = _registry.resolve_by_model_name(model_name)
        if port is not None:
            return port.config_converter(model_name, **kwargs)
        return original(model_name, **kwargs)

    return wrapped


def _convert_smollm3_config(official_model_name: str, **kwargs: Any) -> dict:
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        official_model_name,
        token=_resolve_hf_token(),
        **kwargs,
    )

    architecture = hf_config.architectures[0]
    if architecture != _SMOLLM3_ARCHITECTURE:
        raise NotImplementedError(
            f"{official_model_name!r} is registered as a SmolLM3 model but resolved "
            f"to HF architecture {architecture!r}, expected {_SMOLLM3_ARCHITECTURE!r}. "
            "The circuitkit TL-compat patch only supports SmolLM3ForCausalLM."
        )
    if getattr(hf_config, "attention_bias", False):
        # SmolLM3Attention's q/k/v/o projections are built with
        # bias=config.attention_bias; convert_smollm3_weights below drops
        # biases unconditionally (the real checkpoint ships
        # attention_bias=False), so a future checkpoint setting this True
        # would silently lose its bias weights rather than failing loudly.
        raise NotImplementedError(
            f"{official_model_name!r} has attention_bias=True, which "
            "convert_smollm3_weights does not yet support."
        )
    if getattr(hf_config, "mlp_bias", False):
        # Same reasoning for SmolLM3MLP's gate/up/down projections.
        raise NotImplementedError(
            f"{official_model_name!r} has mlp_bias=True, which "
            "convert_smollm3_weights does not yet support."
        )
    if getattr(hf_config, "use_sliding_window", False):
        # The real SmolLM3-3B checkpoint ships use_sliding_window=False (every
        # layer_types entry is "full_attention"). SmolLM3Attention only
        # engages a sliding window when use_sliding_window=True *and* the
        # layer's own layer_types entry is "sliding_attention" (which,
        # per configuration_smollm3.py, only ever happens on the NoPE layers
        # in the first place) -- a mixed rotary/sliding-window pattern this
        # port's use_local_attn=False, all-full-attention config does not
        # model. Raise rather than silently ignoring a real sliding window.
        raise NotImplementedError(
            f"{official_model_name!r} has use_sliding_window=True, which this "
            "SmolLM3 port does not yet support (it assumes full attention on "
            "every layer, per the real HuggingFaceTB/SmolLM3-3B checkpoint)."
        )

    n_heads = hf_config.num_attention_heads
    n_kv_heads = hf_config.num_key_value_heads
    # SmolLM3Config does not always set an explicit head_dim (confirmed via
    # AutoConfig.from_pretrained(...) -- hasattr is False for the real
    # checkpoint); SmolLM3Attention.__init__ falls back to
    # hidden_size // num_attention_heads via the same getattr(..., default)
    # pattern cohere1's converter needed. Mirror it here rather than
    # assuming the attribute exists.
    d_head = getattr(hf_config, "head_dim", None) or (hf_config.hidden_size // n_heads)

    n_layers = hf_config.num_hidden_layers
    no_rope_layers = list(getattr(hf_config, "no_rope_layers", None) or [])
    if len(no_rope_layers) != n_layers:
        raise NotImplementedError(
            f"{official_model_name!r} has no_rope_layers of length "
            f"{len(no_rope_layers)}, expected {n_layers} (one entry per "
            "transformer block). Refusing to guess a per-layer NoPE mask."
        )
    _NO_ROPE_LAYERS_BY_MODEL[official_model_name] = tuple(int(v) for v in no_rope_layers)

    cfg_dict: Dict[str, Any] = {
        "d_model": hf_config.hidden_size,
        "d_head": d_head,
        "n_heads": n_heads,
        "n_key_value_heads": n_kv_heads,
        "d_mlp": hf_config.intermediate_size,
        "n_layers": n_layers,
        "n_ctx": min(hf_config.max_position_embeddings, _MAX_SAFE_N_CTX),
        "eps": hf_config.rms_norm_eps,
        "d_vocab": hf_config.vocab_size,
        "act_fn": hf_config.hidden_act,
        "gated_mlp": True,
        "normalization_type": "RMS",
        "final_rms": True,
        "positional_embedding_type": "rotary",
        # Standard Llama rotary (rotate_half splits first-half/second-half),
        # confirmed from the installed modeling_smollm3.py source -- NOT the
        # GPT-J-style interleaved-pair rotary the cohere family uses. This is
        # the single highest-risk choice in this port; only the real-weight
        # parity gate (test_smollm3_parity.py) actually proves it.
        "rotary_adjacent_pairs": False,
        "rotary_dim": d_head,  # rotary applied to the full head dim
        "rotary_base": int(_registry.rope_theta(hf_config)),
        "use_local_attn": False,
        "parallel_attn_mlp": False,  # sequential block, not cohere's parallel one
        "use_attn_scale": True,
        "attention_dir": "causal",
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
            # Force safetensors (SmolLM3-3B ships safetensors only) -- see
            # cohere.py's _wrap_get_pretrained_state_dict for the full 401-vs-
            # 404 rationale on gated/format-probing repos.
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


def convert_smollm3_weights(smollm3: Any, cfg: Any) -> Dict[str, torch.Tensor]:
    """Convert a loaded HF ``SmolLM3ForCausalLM`` into a TL ``HookedTransformer``
    state dict.

    Mirrors TL's own ``transformer_lens.pretrained.weight_conversions.llama.
    convert_llama_weights`` (read directly from the installed package as the
    reference for this function's shape) with two differences: RMSNorm's
    ``.w``-only weights (no ``.b`` -- Llama's own converter emits zero LN
    biases too since TL's ``normalization_type="LN"`` blocks need one, but
    SmolLM3 uses ``normalization_type="RMS"``, whose TL module has no bias
    term to satisfy in the first place, so nothing is emitted), and this
    port's fixed-device zero-tensor construction (``next(smollm3.parameters
    ()).device`` rather than TL's own converter's ``cfg.device``) -- the same
    fix ``cohere.py``'s ``convert_cohere2_weights`` needed once its parity
    gate loaded the HF model onto a GPU explicitly (a bare ``torch.zeros(...)``
    with no ``device=`` silently works only when every other tensor in the
    dict happens to already be on CPU).
    """
    assert cfg.n_key_value_heads is not None  # keep mypy happy; GQA is mandatory for SmolLM3
    assert cfg.d_mlp is not None

    state_dict: Dict[str, torch.Tensor] = {}
    base_model = smollm3.model
    device = next(smollm3.parameters()).device

    state_dict["embed.W_E"] = base_model.embed_tokens.weight

    for layer_idx in range(cfg.n_layers):
        layer = base_model.layers[layer_idx]

        state_dict[f"blocks.{layer_idx}.ln1.w"] = layer.input_layernorm.weight

        W_Q = layer.self_attn.q_proj.weight
        W_K = layer.self_attn.k_proj.weight
        W_V = layer.self_attn.v_proj.weight
        W_Q = einops.rearrange(W_Q, "(n h) m -> n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m -> n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{layer_idx}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{layer_idx}.attn._W_K"] = W_K
        state_dict[f"blocks.{layer_idx}.attn._W_V"] = W_V

        state_dict[f"blocks.{layer_idx}.attn.b_Q"] = torch.zeros(
            cfg.n_heads, cfg.d_head, dtype=cfg.dtype, device=device
        )
        state_dict[f"blocks.{layer_idx}.attn._b_K"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=device
        )
        state_dict[f"blocks.{layer_idx}.attn._b_V"] = torch.zeros(
            cfg.n_key_value_heads, cfg.d_head, dtype=cfg.dtype, device=device
        )

        W_O = layer.self_attn.o_proj.weight
        W_O = einops.rearrange(W_O, "m (n h) -> n h m", n=cfg.n_heads)
        state_dict[f"blocks.{layer_idx}.attn.W_O"] = W_O
        state_dict[f"blocks.{layer_idx}.attn.b_O"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=device
        )

        state_dict[f"blocks.{layer_idx}.ln2.w"] = layer.post_attention_layernorm.weight

        # SmolLM3MLP hardcodes bias=config.mlp_bias, guarded False above.
        state_dict[f"blocks.{layer_idx}.mlp.W_gate"] = layer.mlp.gate_proj.weight.T
        state_dict[f"blocks.{layer_idx}.mlp.W_in"] = layer.mlp.up_proj.weight.T
        state_dict[f"blocks.{layer_idx}.mlp.b_in"] = torch.zeros(
            cfg.d_mlp, dtype=cfg.dtype, device=device
        )
        state_dict[f"blocks.{layer_idx}.mlp.W_out"] = layer.mlp.down_proj.weight.T
        state_dict[f"blocks.{layer_idx}.mlp.b_out"] = torch.zeros(
            cfg.d_model, dtype=cfg.dtype, device=device
        )

    state_dict["ln_final.w"] = base_model.norm.weight
    # RMSNorm: no ln_final.b -- see the docstring above.

    # No logit_scale for SmolLM3 (unlike the whole cohere family): HF's
    # SmolLM3ForCausalLM.forward computes logits = lm_head(hidden_states)
    # with no post-multiply, so this is a plain, unscaled transpose.
    state_dict["unembed.W_U"] = smollm3.lm_head.weight.T
    state_dict["unembed.b_U"] = torch.zeros(cfg.d_vocab, dtype=cfg.dtype, device=device)

    return state_dict
