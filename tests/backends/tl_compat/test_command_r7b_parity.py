"""Group F — the real-weight parity GATE for Command R7B (``cohere2``).

Command R7B shares tiny-aya's ``Cohere2ForCausalLM`` architecture and the
exact same config/weight converters (see ``test_cohere_parity.py`` for why a
KL / log-prob comparison is the right check). The one thing this checkpoint
exercises that tiny-aya's own logit_scale=1.0 spec cannot: a genuine,
non-no-op ``logit_scale=0.25`` fold into ``W_U``. If that fold were wrong
(scaled the wrong way, applied twice, or skipped), the softmax distribution
itself would be distorted (not just a per-position shift LN-fold-robustness
would hide), so this parity gate is what actually proves it is correct.

It is intentionally gated. Command R7B is a ~7B-parameter checkpoint (public,
but "gated=auto" on the Hub -- needs the license click-through accepted once),
so this test:

* downloads the checkpoint (needs an account with the license accepted), and
* holds a ~28 GB float32 HF copy (CPU) plus a ~28 GB float32 TL copy (GPU),

so it only runs when explicitly opted in via ``CIRCUITKIT_RUN_COMMAND_R7B=1``
and is marked ``slow``. Ordinary CI (and the offline unit tests) stay green
without it.

    CIRCUITKIT_RUN_COMMAND_R7B=1 HF_TOKEN=... \
        python -m pytest tests/backends/tl_compat/test_command_r7b_parity.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

MODEL_NAME = "CohereLabs/c4ai-command-r7b-12-2024"

# Next-token distribution tolerances (the plan's gate: KL < 1e-4).
KL_TOL = 1e-4
MAX_LOGPROB_TOL = 5e-3

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_COMMAND_R7B", "").strip().lower() not in (
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
            "real-weight Command R7B parity: set CIRCUITKIT_RUN_COMMAND_R7B=1 and "
            "HF_TOKEN to run (downloads a ~7B checkpoint)"
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
        dtype=torch.float32,
        token=token,
        use_safetensors=True,
        # Loaded on CPU (host RAM), not GPU: fold_layer_norm below creates a
        # full new set of folded weight tensors alongside the originals, so
        # keeping both HF's and TL's ~28 GB float32 copies on a single 47 GB
        # GPU does not fit -- see convert_cohere2_weights' device= plumbing,
        # added for exactly this cross-device split. low_cpu_mem_usage avoids
        # the double-allocation from_pretrained's default loading path would
        # otherwise transiently do while converting into this dtype.
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
            f"{MAX_LOGPROB_TOL:.0e} -- suspect rope_gptj / LN-fold / tie / logit_scale=0.25 fold"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_argmax_next_token_agrees(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = _next_token_logprobs(tl_model, hf_model, input_ids)
        assert int(tl_lp.argmax()) == int(
            hf_lp.argmax()
        ), f"prompt {prompt!r}: TL and HF disagree on the top next token"


# --------------------------------------------------------------------------- #
# Sliding-window boundary — proves exact SWA past the 4096-token window.       #
# --------------------------------------------------------------------------- #
class TestSlidingWindowBoundary:
    def test_parity_holds_past_the_window(self, tl_model, hf_model, tokenizer):
        window = tl_model.cfg.window_size
        assert window == 4096
        seq_len = window + 100

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
# logit_scale=0.25 fold — the one behaviour tiny-aya's spec cannot exercise.  #
# --------------------------------------------------------------------------- #
class TestLogitScaleFold:
    def test_hf_logit_scale_is_0_25(self, hf_model):
        """Pin the real config value the KL gate above's fold-correctness
        claim depends on -- if the checkpoint ever ships a different scale,
        the "this is the one thing tiny-aya's spec cannot exercise" framing
        in this file's docstring silently stops being true."""
        assert float(hf_model.logit_scale) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# NoPE layout on the real config (cheap; no forward pass needed).             #
# --------------------------------------------------------------------------- #
class TestNoPELayoutOnRealModel:
    def test_global_layers_skip_rotary(self, tl_model):
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
