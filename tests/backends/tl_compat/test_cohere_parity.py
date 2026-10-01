"""Group F — the real-weight parity GATE for tiny-aya (cohere2).

This is the one test that can numerically catch a wrong ``rotary_adjacent_pairs``
(rope_gptj interleaving), a broken LayerNorm fold, a mis-tied embedding, a
mis-folded ``logit_scale``, or an incorrect sliding-window / NoPE layout: it
loads the *real* gated tiny-aya checkpoint through both HuggingFace and the
circuitkit TransformerLens port and compares their next-token distributions.

It is intentionally gated. tiny-aya is a 3.35B-parameter gated checkpoint, so
this test:

* downloads gated weights (needs ``HF_TOKEN`` with access), and
* holds two ~13 GB float32 copies in memory,

so it only runs when explicitly opted in via ``CIRCUITKIT_RUN_TINY_AYA=1`` and
is marked ``slow``. Ordinary CI (and the offline unit tests) stay green without
it. Run it on a machine with the token and enough RAM/VRAM (e.g. Colab):

    CIRCUITKIT_RUN_TINY_AYA=1 HF_TOKEN=... \
        python -m pytest tests/backends/tl_compat/test_cohere_parity.py -v

Why a KL / log-prob comparison rather than raw logits: TL's default processing
folds LayerNorm and centres the unembedding, which shifts absolute logits by a
per-position constant while preserving the softmax distribution. Comparing
``log_softmax`` is therefore the fold-robust check. If the strict tolerance
fails only *slightly* (LN-fold accumulation on a 3.35B model), the documented
fallback is to load the TL model with ``from_pretrained_no_processing`` /
``fold_ln=False`` and re-compare — see the plan's LN-fold note.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

MODEL_NAME = "CohereLabs/tiny-aya-base"

# Next-token distribution tolerances (the plan's gate: KL < 1e-4).
KL_TOL = 1e-4
# Max per-vocab log-prob deviation — a stricter, position-local companion to KL.
MAX_LOGPROB_TOL = 5e-3

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_TINY_AYA", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_OPT_IN and _HAS_TOKEN),
        reason=(
            "gated tiny-aya real-weight parity: set CIRCUITKIT_RUN_TINY_AYA=1 and "
            "HF_TOKEN to run (downloads a 3.35B gated checkpoint)"
        ),
    ),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    # Prefer the cached login (huggingface_hub picks it up when ``token`` is
    # None or True) over a possibly-garbage HF_TOKEN env var. Explicit env
    # takes precedence only when it looks like a real token, so setting
    # ``HF_TOKEN=your_token_here`` as a placeholder cannot silently poison
    # the fetch with a bogus explicit token that beats the cached login in
    # huggingface_hub's precedence order.
    #
    # ``use_safetensors=True`` skips ``transformers``'s alternative-format
    # probe. On gated repos that probe returns HTTP 401 on missing files
    # (``tf_model.h5``, ``flax_model.msgpack``) and is then treated as a repo
    # access failure — see the matching guard in ``_tl_compat/cohere.py``.
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    token = env_token if env_token.startswith("hf_") else True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        token=token,
        use_safetensors=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tl_model(hf_model):
    # Reuse the already-loaded HF weights for the TL conversion so we hold two
    # copies (HF + converted TL), not three. The circuitkit port's patched
    # get_pretrained_state_dict accepts hf_model= and converts from it.
    from transformer_lens import HookedTransformer

    from circuitkit.backends import _tl_compat  # noqa: F401 -- import-time apply_patches()

    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        hf_model=hf_model,
        device=_device(),
        dtype=torch.float32,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer(tl_model):
    # TL loads the CohereTokenizer for this model; reuse it so BOS handling
    # (add_bos_token=true, BOS id 2) is exactly the checkpoint's own.
    return tl_model.tokenizer


def _next_token_logprobs(tl_model, hf_model, input_ids):
    """Return (tl_logprobs, hf_logprobs) over the vocab at the final position."""
    tl_ids = input_ids.to(next(tl_model.parameters()).device)
    hf_ids = input_ids.to(next(hf_model.parameters()).device)
    with torch.no_grad():
        tl_logits = tl_model(tl_ids)[0, -1].float().cpu()
        hf_logits = hf_model(hf_ids).logits[0, -1].float().cpu()
    return F.log_softmax(tl_logits, dim=-1), F.log_softmax(hf_logits, dim=-1)


def _kl(hf_logprobs, tl_logprobs) -> float:
    # KL(HF || TL): sum exp(hf) * (hf - tl). log_target=True since both inputs
    # are already log-probabilities.
    return float(
        F.kl_div(tl_logprobs, hf_logprobs, log_target=True, reduction="sum").item()
    )


# --------------------------------------------------------------------------- #
# Short-prompt next-token parity — the core gate.                             #
# --------------------------------------------------------------------------- #
class TestNextTokenParity:
    PROMPTS = [
        "The capital of France is",
        "Once upon a time, there was a",
        "2 + 2 =",
        "Water boils at a temperature of",
    ]

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_next_token_distribution_matches_hf(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = _next_token_logprobs(tl_model, hf_model, input_ids)

        kl = _kl(hf_lp, tl_lp)
        max_dev = float((tl_lp - hf_lp).abs().max().item())
        assert kl < KL_TOL, f"prompt {prompt!r}: KL(HF||TL)={kl:.2e} exceeds {KL_TOL:.0e}"
        assert max_dev < MAX_LOGPROB_TOL, (
            f"prompt {prompt!r}: max log-prob deviation {max_dev:.2e} exceeds "
            f"{MAX_LOGPROB_TOL:.0e} -- suspect rope_gptj / LN-fold / tie / logit_scale"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_argmax_next_token_agrees(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = _next_token_logprobs(tl_model, hf_model, input_ids)
        assert int(tl_lp.argmax()) == int(hf_lp.argmax()), (
            f"prompt {prompt!r}: TL and HF disagree on the top next token"
        )


# --------------------------------------------------------------------------- #
# Sliding-window boundary — proves exact SWA past the 4096-token window.       #
# --------------------------------------------------------------------------- #
class TestSlidingWindowBoundary:
    def test_parity_holds_past_the_window(self, tl_model, hf_model, tokenizer):
        """A prompt longer than the sliding window exercises both the windowed
        local layers and the full-attention global layers. If TL's
        use_local_attn/window_size or the NoPE global layout diverged from HF's
        sliding window, the distribution past position 4096 would blow up."""
        window = tl_model.cfg.window_size
        assert window == 4096
        seq_len = window + 100

        # A single repeated real token id keeps the forward well-defined and
        # cheap to build while still crossing the window boundary.
        base = tokenizer("the", return_tensors="pt").input_ids
        fill_id = int(base[0, -1])
        prepend_bos = bool(getattr(tl_model.cfg, "default_prepend_bos", False))
        bos = tokenizer.bos_token_id
        ids = [bos] if (prepend_bos and isinstance(bos, int)) else []
        ids += [fill_id] * (seq_len - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long)

        tl_lp, hf_lp = _next_token_logprobs(tl_model, hf_model, input_ids)
        kl = _kl(hf_lp, tl_lp)
        assert kl < KL_TOL, (
            f"KL(HF||TL)={kl:.2e} past the {window}-token window exceeds "
            f"{KL_TOL:.0e} -- sliding-window / NoPE layout mismatch"
        )


# --------------------------------------------------------------------------- #
# NoPE layout on the real config (cheap; no forward pass needed).             #
# --------------------------------------------------------------------------- #
class TestNoPELayoutOnRealModel:
    def test_global_layers_skip_rotary(self, tl_model):
        """Every full-attention ('global') layer must apply_rotary as identity;
        every sliding ('local') layer must actually rotate. This is the NoPE
        contract the parity above depends on, checked directly on the loaded
        real model's attention modules."""
        n_heads = tl_model.cfg.n_heads
        d_head = tl_model.cfg.d_head
        x = torch.randn(1, 5, n_heads, d_head, device=next(tl_model.parameters()).device)
        saw_global = saw_local = False
        for i, block in enumerate(tl_model.blocks):
            out = block.attn.apply_rotary(x)
            if block.attn.attn_type == "global":
                saw_global = True
                assert out is x, f"block {i} (global) must skip rotary (NoPE)"
            else:
                saw_local = True
                assert out is not x, f"block {i} (local) must apply rotary"
        assert saw_global and saw_local, "expected both global and local layers"
