"""Group F — the real-weight parity GATE for SmolLM3-3B.

SmolLM3-3B (``HuggingFaceTB/SmolLM3-3B``, HF architecture
``SmolLM3ForCausalLM``) is the first Llama-family model this port supports,
and per the integration guide's Part 5 pitfalls list, RoPE interleaving is
"the #1 silent-wrong-output bug" -- a wrong ``rotary_adjacent_pairs`` choice,
or a wrong per-layer NoPE mask, produces a plausible-looking but numerically
wrong model that only a real-weight next-token comparison against HF
actually catches. This file is that proof:

* ``rotary_adjacent_pairs=False`` (standard, non-interleaved Llama rotary,
  confirmed from ``modeling_smollm3.py``'s ``rotate_half`` -- the opposite
  choice from the whole cohere family) -- a wrong choice here would corrupt
  every rotated layer's attention pattern.
* the per-layer NoPE mask (``no_rope_layers[layer_id] == 0`` skips rotary,
  positions 3, 7, 11, ... on the real checkpoint) -- applying rotary on a
  NoPE layer, or skipping it on a layer that should rotate, would likewise
  produce a badly wrong next-token distribution, not a subtle drift.
* no logit-scale fold (unlike the whole cohere family) -- confirms
  ``unembed.W_U`` needs no post-hoc scale to match HF's unscaled
  ``lm_head(hidden_states)``.

Unlike Command R7B / Aya Expanse (both "gated=auto" on the Hub, needing an
account with the license accepted), SmolLM3-3B is a plain public repo -- no
HF_TOKEN is required, only the opt-in env var. It is also the smallest of the
three models (~3B, ~6GB bf16 weights), so both the HF and TL float32 copies
comfortably fit in host RAM (and likely on a single 47GB GPU too, but this
test keeps the same cross-device split the larger cohere-family models
needed, for consistency and to avoid the two float32 copies competing for
the same device's memory during ``fold_layer_norm``).

    CIRCUITKIT_RUN_SMOLLM3=1 python -m pytest tests/backends/tl_compat/test_smollm3_parity.py -v
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

# Next-token distribution tolerances (the plan's gate: KL < 1e-4).
KL_TOL = 1e-4
MAX_LOGPROB_TOL = 5e-3

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_SMOLLM3", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _OPT_IN,
        reason=(
            "real-weight SmolLM3 parity: set CIRCUITKIT_RUN_SMOLLM3=1 to run "
            "(downloads a ~3B checkpoint; public, no HF_TOKEN required)"
        ),
    ),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _hf_token():
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    return env_token if env_token.startswith("hf_") else None


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        token=_hf_token(),
        use_safetensors=True,
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
            f"{MAX_LOGPROB_TOL:.0e} -- suspect rotary interleaving (should be "
            f"non-interleaved) or the per-layer NoPE mask"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_argmax_next_token_agrees(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = _next_token_logprobs(tl_model, hf_model, input_ids)
        assert int(tl_lp.argmax()) == int(
            hf_lp.argmax()
        ), f"prompt {prompt!r}: TL and HF disagree on the top next token"


# --------------------------------------------------------------------------- #
# Per-layer NoPE mask — the guide's #1 pitfall, confirmed on the real model.  #
# --------------------------------------------------------------------------- #
class TestPerLayerNoPEOnRealModel:
    # 0-indexed layers where no_rope_layers == 0 on the real checkpoint
    # (every 4th layer: no_rope_layer_interval=4).
    _NOPE_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35}

    def test_nope_layers_skip_rotary_others_apply_it(self, tl_model):
        n_heads = tl_model.cfg.n_heads
        d_head = tl_model.cfg.d_head
        x = torch.randn(1, 5, n_heads, d_head, device=next(tl_model.parameters()).device)
        for i, block in enumerate(tl_model.blocks):
            out = block.attn.apply_rotary(x)
            if i in self._NOPE_LAYERS:
                assert out is x, f"block {i}: expected NoPE (rotary skipped), but it rotated"
            else:
                assert out is not x, f"block {i}: expected rotary to be applied, but it was skipped"

    def test_use_local_attn_is_false(self, tl_model):
        assert tl_model.cfg.use_local_attn is False
        assert tl_model.cfg.attn_types is None

    def test_rotary_is_non_interleaved(self, tl_model):
        assert tl_model.cfg.rotary_adjacent_pairs is False


# --------------------------------------------------------------------------- #
# RMSNorm / no logit scale — architectural sanity on the real, loaded model.  #
# --------------------------------------------------------------------------- #
class TestArchitectureOnRealModel:
    def test_normalization_type_is_rms(self, tl_model):
        # The config converter sets normalization_type="RMS" (pinned directly
        # on the returned cfg_dict in test_smollm3.py); HookedTransformer.
        # from_pretrained's default fold_ln=True then renames it to "RMSPre"
        # on the *loaded* model (transformer_lens/HookedTransformer.py's
        # process_weights_ sets self.cfg.normalization_type = "RMSPre" after
        # folding -- the exact same rename "LN" -> "LNPre" undergoes for the
        # cohere family), since folding removes the need for a separate
        # scale step before the linear layers. Both are the RMSNorm family,
        # not LayerNorm -- the meaningful assertion is that it is one of
        # {"RMS", "RMSPre"}, never an "LN"-family value.
        assert tl_model.cfg.normalization_type in ("RMS", "RMSPre")
        assert tl_model.cfg.final_rms is True

    def test_sequential_not_parallel_block(self, tl_model):
        assert tl_model.cfg.parallel_attn_mlp is False

    def test_hf_model_has_no_logit_scale_attribute(self, hf_model):
        """Sanity check on the parity claim: HF's SmolLM3ForCausalLM applies
        no post-unembed scale at all, unlike the whole cohere family."""
        assert not hasattr(hf_model, "logit_scale")
