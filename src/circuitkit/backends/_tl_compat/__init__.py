"""Compatibility patches that teach TransformerLens 2.18 / 3.8 to load tiny-aya /
Command R7B / Aya Expanse (``cohere2``/``cohere1``) and SmolLM3-3B (``smollm3``).

TransformerLens 2.18 and 3.8 have no built-in support for these architectures. This
package registers that support additively by patching
``transformer_lens.loading_from_pretrained`` and
``transformer_lens.components.abstract_attention`` at import time -- see
``cohere.py`` and ``smollm3.py`` for the per-architecture details (NoPE
policy, block topology, rotary style, weight conversion).

Every patch falls through to the original TransformerLens behavior for any
model that isn't one of the above — no other architecture is affected.
"""

from __future__ import annotations

import importlib.metadata
import logging

import transformer_lens

logger = logging.getLogger(__name__)

_PATCHED = False

# The patch touches TL internals (the convert_hf_model_config /
# get_pretrained_state_dict dispatch tables, AbstractAttention.apply_rotary)
# whose shape may change across releases. Fail loud on a TL bump rather than
# silently applying a patch written against a different internal layout.
# 3.8 keeps the same three entry points (checked against 3.8.0).
_SUPPORTED_TL_VERSION_PREFIXES = ("2.18", "3.8")


def _detect_transformer_lens_version() -> str:
    """Resolve the installed transformer_lens version, preferring the module's
    own ``__version__`` and falling back to the packaging metadata (some
    installs, including certain Colab environments, ship without
    ``__version__`` set on the module itself)."""
    version = getattr(transformer_lens, "__version__", "") or ""
    if version:
        return version
    try:
        return importlib.metadata.version("transformer_lens")
    except importlib.metadata.PackageNotFoundError:
        return ""


def apply_patches() -> None:
    """Idempotently apply the tiny-aya/cohere2 TransformerLens compatibility patches.

    Safe to call multiple times; only the first call has an effect.
    """
    global _PATCHED
    if _PATCHED:
        return

    tl_version = _detect_transformer_lens_version()
    supported = " or ".join(f"transformer_lens=={p}.*" for p in _SUPPORTED_TL_VERSION_PREFIXES)
    if not tl_version.startswith(_SUPPORTED_TL_VERSION_PREFIXES):
        version_desc = repr(tl_version) if tl_version else "unknown (no __version__ or packaging metadata)"
        raise RuntimeError(
            "circuitkit's tiny-aya/cohere2 TransformerLens compatibility patch "
            f"targets {supported}, but "
            f"transformer_lens=={version_desc} is installed. Refusing to apply "
            "it blindly since it monkeypatches internal dispatch tables and "
            "methods that may have changed shape. Pin "
            f"{supported} or update this patch."
        )

    from . import cohere as _cohere
    from . import smollm3 as _smollm3

    _cohere.register()
    _smollm3.register()

    _PATCHED = True
    logger.debug("Applied circuitkit multi-model TransformerLens compatibility patches.")
