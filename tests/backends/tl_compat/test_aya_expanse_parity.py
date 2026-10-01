"""Group F — the real-weight parity GATE for Aya Expanse 8B (``cohere1``).

Aya Expanse 8B (``CohereLabs/aya-expanse-8b``, HF architecture
``CohereForCausalLM``) reuses cohere2's shared weight converter
(``convert_cohere2_weights`` -- see ``test_cohere_parity.py`` for why a KL /
log-prob comparison is the right check) but exercises two things no other
cohere-family checkpoint in this repo does simultaneously:

* a third, distinct ``logit_scale`` value (0.125, vs. tiny-aya's 1.0 no-op
  and Command R7B's 0.25) folded into ``W_U``, and
* the "no NoPE at all" architectural choice -- rotary applied to *every*
  layer, proven by the fact that a wrong choice here (skipping rotary on any
  layer, as cohere2's NoPE policy would do) would produce a badly wrong next-
  token distribution, not a subtle numerical drift.

It is intentionally gated. Aya Expanse is a ~8B-parameter checkpoint (public,
but "gated=auto" on the Hub -- needs the license click-through accepted
once), so this test:

* downloads the checkpoint (needs an account with the license accepted), and
* holds a ~32 GB float32 HF copy (CPU) plus a ~32 GB float32 TL copy (GPU or
  CPU, depending on VRAM),

so it only runs when explicitly opted in via ``CIRCUITKIT_RUN_AYA_EXPANSE=1``
and is marked ``slow``. Ordinary CI (and the offline unit tests) stay green
without it.

    CIRCUITKIT_RUN_AYA_EXPANSE=1 HF_TOKEN=... \
        python -m pytest tests/backends/tl_compat/test_aya_expanse_parity.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

MODEL_NAME = "CohereLabs/aya-expanse-8b"

# Next-token distribution tolerances (the plan's gate: KL < 1e-4).
KL_TOL = 1e-4
MAX_LOGPROB_TOL = 5e-3

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_AYA_EXPANSE", "").strip().lower() not in (
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
            "real-weight Aya Expanse parity: set CIRCUITKIT_RUN_AYA_EXPANSE=1 and "
            "HF_TOKEN to run (downloads a ~8B checkpoint)"
        ),
    ),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    token = env_token if env_token.startswith("hf_") else True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        token=token,
        use_safetensors=True,
        # Loaded on CPU (host RAM): fold_layer_norm creates a full new set of
        # folded weight tensors alongside the originals, so keeping both HF's
        # and TL's ~32 GB float32 copies on a single 47 GB GPU does not fit --
        # same cross-device split as Command R7B's own parity gate.
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def tl_model(hf_model):
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
    return tl_model.tokenizer


def _next_token_logprobs(tl_model, hf_model, input_ids):
    tl_ids = input_ids.to(next(tl_model.parameters()).device)
    hf_ids = input_ids.to(next(hf_model.parameters()).device)
    with torch.no_grad():
        tl_logits = tl_model(tl_ids)[0, -1].float().cpu()
        hf_logits = hf_model(hf_ids).logits[0, -1].float().cpu()
    return F.log_softmax(tl_logits, dim=-1), F.log_softmax(hf_logits, dim=-1)


def _kl(hf_logprobs, tl_logprobs) -> float:
    return float(F.kl_div(tl_logprobs, hf_logprobs, log_target=True, reduction="sum").item())


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
            f"{MAX_LOGPROB_TOL:.0e} -- suspect rotary interleaving / LN-fold / tie / "
            f"logit_scale=0.125 fold"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_argmax_next_token_agrees(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = _next_token_logprobs(tl_model, hf_model, input_ids)
        assert int(tl_lp.argmax()) == int(
            hf_lp.argmax()
        ), f"prompt {prompt!r}: TL and HF disagree on the top next token"


# --------------------------------------------------------------------------- #
# logit_scale=0.125 fold — a third distinct value across the cohere family.   #
# --------------------------------------------------------------------------- #
class TestLogitScaleFold:
    def test_hf_logit_scale_is_0_125(self, hf_model):
        """Pin the real config value the KL gate above's fold-correctness
        claim depends on."""
        assert float(hf_model.logit_scale) == pytest.approx(0.125)


# --------------------------------------------------------------------------- #
# Rotary-on-every-layer — the one architectural simplification vs cohere2.    #
# --------------------------------------------------------------------------- #
class TestRotaryAppliesToEveryLayerOnRealModel:
    def test_no_layer_skips_rotary(self, tl_model):
        """Unlike cohere2 (NoPE on periodic full-attention layers), every
        cohere1 layer must apply rotary -- confirm on the real, loaded
        config/blocks, not just the unit-level registry policy check in
        test_aya_expanse.py."""
        n_heads = tl_model.cfg.n_heads
        d_head = tl_model.cfg.d_head
        x = torch.randn(1, 5, n_heads, d_head, device=next(tl_model.parameters()).device)
        for i, block in enumerate(tl_model.blocks):
            out = block.attn.apply_rotary(x)
            assert out is not x, f"block {i}: expected rotary to be applied, but it was skipped"

    def test_use_local_attn_is_false(self, tl_model):
        assert tl_model.cfg.use_local_attn is False
        assert tl_model.cfg.attn_types is None
