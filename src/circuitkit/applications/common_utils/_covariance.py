"""Corpus-statistic estimation for ROME / MEMIT.

Both ROME (Eqn. 14: u = C^{-1} k) and MEMIT (Eqn. 14: ∆ = R K^T (C + KK^T)^{-1})
require C ≜ λ · E[k k^T] over a representative corpus. The original papers
estimate this from ~100k Wikipedia samples; here we accept an arbitrary
text iterable and cache the result on disk per (model, layer, n_samples).

Public surface:
    get_covariance(model, layer, hook_name, ...) -> torch.Tensor [d, d]
    solve_with_C(C, k, lam) -> torch.Tensor                       # C^{-1}-equivalent
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Iterable, List, Optional

import torch

logger = logging.getLogger(__name__)

# Tiny in-tree fallback. Used only when the caller passes no `texts` and
# datasets isn't installed. ~50 sentences is far below the paper's 100k —
# good enough for a smoke test, not for paper-grade specificity.
_FALLBACK_TEXTS: List[str] = [
    "The capital of France is Paris.",
    "Albert Einstein developed the theory of relativity.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "Shakespeare wrote Hamlet, Macbeth, and many other plays.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Mount Everest is the tallest mountain above sea level.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The Great Wall of China was built over many centuries.",
    "Marie Curie was a pioneering researcher in radioactivity.",
    "The Amazon rainforest spans several South American countries.",
    "Gravity causes objects to fall toward the Earth.",
    "DNA carries genetic information in living organisms.",
    "The Roman Empire fell in the fifth century.",
    "Mozart composed symphonies in the eighteenth century.",
    "The Sahara is the largest hot desert in the world.",
    "Programming languages let humans instruct computers.",
    "Antarctica is the coldest continent on Earth.",
    "The human brain contains about eighty-six billion neurons.",
    "Light travels faster than sound through air.",
    "The Nile is one of the longest rivers on Earth.",
    "Ancient Egyptians built pyramids as royal tombs.",
    "The internet connects billions of devices worldwide.",
    "Beethoven continued composing after losing his hearing.",
    "Penicillin was discovered by Alexander Fleming.",
    "The Earth orbits the Sun once each year.",
    "Volcanoes erupt when magma reaches the surface.",
    "Bees pollinate many of the crops humans eat.",
    "The speed of light is roughly three hundred thousand kilometres per second.",
    "Coffee originated in the highlands of Ethiopia.",
    "Birds are the closest living relatives of dinosaurs.",
    "Books have been printed since the fifteenth century.",
    "Vaccines train the immune system to fight infections.",
    "Glaciers store a large fraction of the world's fresh water.",
    "Tokyo is the most populous metropolitan area on Earth.",
    "Chess originated in India over a thousand years ago.",
    "The moon influences the tides on Earth.",
    "Carbon dioxide is a greenhouse gas in the atmosphere.",
    "Spices were once a major driver of global trade.",
    "Sound cannot travel through the vacuum of space.",
    "The cell is the basic unit of life.",
    "Atoms combine to form molecules through chemical bonds.",
    "The English language has borrowed words from many cultures.",
    "Whales are mammals that live in the ocean.",
    "Earthquakes occur along tectonic plate boundaries.",
    "Telescopes allow astronomers to see distant stars and galaxies.",
    "The wheel was one of humanity's earliest important inventions.",
    "Steam engines powered the Industrial Revolution.",
    "Antibiotics revolutionized the treatment of bacterial disease.",
    "Mathematics is used in nearly every science.",
    "The Statue of Liberty stands in New York Harbor.",
]


def _cache_dir() -> Path:
    p = Path(os.environ.get("KE_CACHE_DIR", Path.home() / ".cache" / "ke_covariance"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(model_name: str, layer: int, hook_name: str, n_samples: int, corpus_id: str) -> str:
    raw = f"{model_name}|L{layer}|{hook_name}|N{n_samples}|C{corpus_id}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def get_covariance(
    model,
    layer: int,
    hook_name: str,
    texts: Optional[Iterable[str]] = None,
    n_samples: int = 1000,
    max_seq_len: int = 256,
    device: Optional[str] = None,
    use_cache: bool = True,
    corpus_id: str = "default",
) -> Optional[torch.Tensor]:
    """Estimate C ≜ E[k k^T] at `blocks.{layer}.{hook_name}` over `texts`.

    Returns C as a [d, d] float32 tensor on `device` (defaults to model's
    device). Caches under ~/.cache/ke_covariance keyed by
    (model_name, layer, hook_name, n_samples, corpus_id). Approximate —
    paper uses ~100k samples; default here is 1k, which is enough to fix
    the worst of the magnitude problem (Bug #1) but not paper-grade.
    Increase `n_samples` for benchmarking.

    Notes:
      * accumulates in float64, stores in float32
      * uses `use_cache=False` to force recompute (e.g. after model swap)
      * silent fallback to in-tree corpus if `texts` is None
      * `corpus_id` distinguishes caches built from different `texts` at the
        same `n_samples`. Pass a stable id (e.g. "wiki100k") when supplying
        your own corpus to avoid collisions with previous runs.
    """
    model_name = getattr(getattr(model, "cfg", None), "model_name", "unknown")
    if device is None:
        device = str(next(model.parameters()).device)

    key = _cache_key(model_name, layer, hook_name, n_samples, corpus_id)
    cache_path = _cache_dir() / f"{key}.pt"
    if use_cache and cache_path.exists():
        t0 = time.time()
        try:
            C = torch.load(cache_path, map_location=device, weights_only=True).to(device)
            logger.info(
                f"C[layer={layer},hook={hook_name},n={n_samples},id={corpus_id}] "
                f"cache HIT  ({time.time() - t0:.3f}s)"
            )
            return C
        except Exception as e:
            warnings.warn(f"Failed to load cached C ({cache_path}): {e}; recomputing.")

    # Materialise sample iterable.
    if texts is None:
        texts = _FALLBACK_TEXTS
        if n_samples > len(_FALLBACK_TEXTS):
            logger.warning(
                f"No `texts` provided and n_samples={n_samples} exceeds the "
                f"in-tree fallback corpus ({len(_FALLBACK_TEXTS)} sentences). "
                "Using the fallback as-is — results will be approximate. "
                "Pass a Wikipedia sample for paper-grade C."
            )
    texts = list(texts)[:n_samples]
    if not texts:
        warnings.warn("Empty `texts`; cannot estimate C.")
        return None

    full_hook = f"blocks.{layer}.{hook_name}"
    accum: Optional[torch.Tensor] = None  # float64
    n_tokens = 0
    t_compute = time.time()

    def _hook(act, hook):
        # act: [batch, pos, d]
        nonlocal accum, n_tokens
        flat = act.detach().to(torch.float64).reshape(-1, act.shape[-1])  # [B*P, d]
        contrib = flat.T @ flat  # [d, d]
        if accum is None:
            accum = contrib
        else:
            accum = accum + contrib
        n_tokens += flat.shape[0]
        return act

    try:
        for text in texts:
            try:
                tokens = model.to_tokens(text)
                if tokens.shape[1] > max_seq_len:
                    tokens = tokens[:, :max_seq_len]
                with torch.no_grad():
                    model.run_with_hooks(tokens, fwd_hooks=[(full_hook, _hook)])
            except Exception as e:
                warnings.warn(f"Skipping sample (len={len(text)}): {e}")
                continue
    finally:
        try:
            model.reset_hooks()
        except Exception:
            pass

    if accum is None or n_tokens == 0:
        warnings.warn("Failed to accumulate any samples for C.")
        return None

    C = (accum / n_tokens).to(torch.float32).to(device)
    logger.info(
        f"C[layer={layer},hook={hook_name},n={n_samples},id={corpus_id}] "
        f"computed from {n_tokens} tokens  ({time.time() - t_compute:.2f}s)  "
        f"diag_mean={C.diag().mean().item():.4g}  norm={C.norm().item():.4g}"
    )

    if use_cache:
        try:
            torch.save(C.cpu(), cache_path)
        except Exception as e:
            warnings.warn(f"Failed to write C cache ({cache_path}): {e}")

    return C


def solve_with_C(C: torch.Tensor, k: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """
    Solve (C + λI) v = k for v. ROME's u direction (Eqn. 14) is v / |v|.

    Operates in float64 internally for stability; returns in k's original
    dtype and device. C and k may live on different devices — the solve
    runs on C.device and the result is moved back to k.device.
    """
    d = C.shape[0]
    eye = lam * torch.eye(d, dtype=torch.float64, device=C.device)
    A = C.to(torch.float64) + eye
    b = k.to(device=C.device, dtype=torch.float64).reshape(d, -1)
    v = torch.linalg.solve(A, b)
    return v.reshape(k.shape).to(dtype=k.dtype, device=k.device)
