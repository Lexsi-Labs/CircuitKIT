"""Shared multi-architecture registry for circuitkit's TransformerLens
compatibility ports (``circuitkit.backends._tl_compat``).

Each port module (``cohere.py``, ``smollm3.py``, ...) owns its own repo-ID
registration and wraps ``transformer_lens.loading_from_pretrained``'s
``convert_hf_model_config`` / ``get_pretrained_state_dict`` itself, using the
chain-of-responsibility pattern already established there: each wrap receives
the previous "original" callable and falls through to it for architectures it
does not own. Both of those wrappers now resolve *which* converter to use by
looking a port up here (by model name, or by the already-resolved HF
architecture) rather than hardcoding a single owned name inline -- the seam
that lets a port module cover more than one architecture (cohere1 alongside
cohere2 in ``cohere.py``) or a new file (``smollm3.py``) register in without
touching the others.

``AbstractAttention.apply_rotary`` is the one piece that chain-of-
responsibility wrapping suits less well: every port patching it independently
would nest one near-identical architecture check per port for a single
method. Instead, ports register a per-architecture NoPE predicate here, and
``patch_apply_rotary()`` installs a dispatcher that resolves the policy by
the *calling* attention module's resolved architecture at call time. A port's
rotary policy therefore takes effect immediately, independent of
registration order between ports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

ConfigConverter = Callable[..., Dict[str, Any]]
WeightConverter = Callable[[Any, Any], Dict[str, Any]]
# (cfg, layer_id, layer_attn_type) -> True if that layer should skip rotary
# entirely. layer_id is the block index (SmolLM3's per-layer no_rope_layers
# mask); layer_attn_type is TL's "local"/"global" attn_type string (cohere2's
# sliding-window-vs-full split). A policy reads whichever it needs.
RotaryPolicy = Callable[[Any, Optional[int], str], bool]


def _never_skip(cfg: Any, layer_id: Optional[int], layer_attn_type: str) -> bool:
    return False


@dataclass(frozen=True)
class ArchPort:
    """One port module's contribution for one resolved HF architecture."""

    architecture: str
    model_names: Tuple[str, ...]
    config_converter: ConfigConverter
    weight_converter: WeightConverter
    should_skip_rotary: RotaryPolicy = field(default=_never_skip)


_PORTS_BY_MODEL_NAME: Dict[str, ArchPort] = {}
_PORTS_BY_ARCHITECTURE: Dict[str, ArchPort] = {}


def register_port(port: ArchPort) -> None:
    """Register a port's model names and architecture, additively.

    Safe to call repeatedly for the same port (e.g. a test re-invoking a
    module's ``register()``); later calls simply overwrite with the same
    data.
    """
    for name in port.model_names:
        _PORTS_BY_MODEL_NAME[name.lower()] = port
    _PORTS_BY_ARCHITECTURE[port.architecture] = port


def rope_theta(hf_config: Any) -> Any:
    """``rope_theta`` of an HF config: a top-level attribute before transformers 5,
    ``rope_parameters["rope_theta"]`` from 5 on."""
    value = getattr(hf_config, "rope_theta", None)
    if value is None:
        value = (getattr(hf_config, "rope_parameters", None) or {}).get("rope_theta")
    return value


def resolve_by_model_name(model_name: str) -> Optional[ArchPort]:
    return _PORTS_BY_MODEL_NAME.get(model_name.lower())


def resolve_by_architecture(architecture: Any) -> Optional[ArchPort]:
    return _PORTS_BY_ARCHITECTURE.get(architecture)


def should_skip_rotary(cfg: Any, layer_id: Optional[int], layer_attn_type: str) -> bool:
    """Resolve and apply the registered NoPE policy for ``cfg``'s
    architecture. Architectures with no registered port default to "never
    skip" -- stock TransformerLens behaviour."""
    port = resolve_by_architecture(getattr(cfg, "original_architecture", None))
    if port is None:
        return False
    return port.should_skip_rotary(cfg, layer_id, layer_attn_type)


def patch_apply_rotary(abstract_attention_cls: type) -> None:
    """Wrap ``abstract_attention_cls.apply_rotary`` to consult the shared
    rotary-policy registry before falling through to the original
    TransformerLens rotary application.

    Not idempotent by design: each port module's ``register()`` calls this
    once, so calling it multiple times (once per registered port) nests that
    many dispatch layers. That is harmless redundancy rather than a bug --
    every layer consults the same, fully-populated registry dynamically at
    call time, so each is independently correct regardless of port
    registration order.
    """
    original_apply_rotary = abstract_attention_cls.apply_rotary

    def patched_apply_rotary(
        self: Any,
        x: Any,
        past_kv_pos_offset: int = 0,
        attention_mask: Any = None,
    ) -> Any:
        if should_skip_rotary(self.cfg, getattr(self, "layer_id", None), self.attn_type):
            return x
        return original_apply_rotary(self, x, past_kv_pos_offset, attention_mask)

    abstract_attention_cls.apply_rotary = patched_apply_rotary
