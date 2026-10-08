"""Group F — the real-weight parity GATE for Aya Expanse 32B (``cohere1``).

Aya Expanse 32B is the same ``CohereForCausalLM`` architecture and shares
the same config/weight converters as Aya Expanse 8B (``test_aya_expanse_
parity.py``), but exercises values no other cohere-family checkpoint in
this repo does: 8:1 GQA (vs 8B's 4:1), ``rope_theta=4,000,000`` (vs 8B's
10,000), and ``logit_scale=0.0625`` (vs 8B's 0.125) -- at 40 layers / 8192
d_model / 24576 d_mlp, a checkpoint roughly 4x 8B's parameter count.

Two tiers, both against the real checkpoint (never a tiny random model --
that coverage lives in ``test_aya_expanse_32b.py``'s offline tests):

* **Tier A** (this file's ``TestTierATruncatedExactness``): loads only the
  first ``N_LAYERS=4`` decoder layers of the real checkpoint in float32, read
  straight from the safetensors shards that hold them (see the class
  docstring for why not ``from_pretrained(..., num_hidden_layers=4)``; the
  embedding, final norm and unembed are untruncated -- those are not
  per-layer), so this exercises 32B's real
  dims, embedding and real per-layer weights for exactly the values that are
  new at this scale, at a tractable memory footprint (~20 GB HF + ~29 GB TL
  in fp32, HF on CPU / TL on one GPU, mirroring the 8B gate's device split).
  Same KL/log-prob/argmax thresholds as the 8B gate, because fp32 at this
  depth has no excuse for a looser one.

* **Tier B** (``TestTierBFullDepthBf16Sanity``): the full 40-layer checkpoint
  at bf16, one copy resident at a time (HF computes and saves log-probs to
  disk, frees, then TL loads and compares) -- a full fp32 double-copy would
  need ~129 GiB each. bf16 cannot honestly hit the 1e-4
  KL bar 8B's fp32 gate uses, so this tier measures its own noise floor
  (HF eager vs sdpa attention, same prompts, same bf16) and gates against
  that measured floor plus identical top-1 agreement, rather than importing
  Tier A's fp32 number unchanged. Skipped unless
  ``CIRCUITKIT_RUN_AYA_EXPANSE_32B_FULL=1``: it needs the full model across
  GPUs, which transformer-lens 3.8.0 cannot place correctly.

Gating mirrors the 8B file but uses its own opt-in flag
(``CIRCUITKIT_RUN_AYA_EXPANSE_32B``), deliberately distinct from 8B's
(``CIRCUITKIT_RUN_AYA_EXPANSE``) so an 8B opt-in never silently triggers a
60 GiB download:

    CIRCUITKIT_RUN_AYA_EXPANSE_32B=1 HF_TOKEN=... \
        python -m pytest tests/backends/tl_compat/test_aya_expanse_32b_parity.py -v
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

MODEL_NAME = "CohereLabs/aya-expanse-32b"
N_LAYERS = 4

# Tier A (fp32, truncated depth): same bar as the 8B gate.
KL_TOL_TIER_A = 1e-4
MAX_LOGPROB_TOL_TIER_A = 5e-3

_OPT_IN = os.environ.get("CIRCUITKIT_RUN_AYA_EXPANSE_32B", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
_FULL_DEPTH_OPT_IN = os.environ.get(
    "CIRCUITKIT_RUN_AYA_EXPANSE_32B_FULL", ""
).strip().lower() not in ("", "0", "false", "no")
# The full 40-layer model needs more memory than a single typical GPU provides, and
# transformer-lens 3.8.0's multi-GPU (n_devices) block placement is broken, so everything that
# loads it at full depth through TransformerLens fails. Kept behind a second flag so the
# documented opt-in runs only what can pass. See docs/advanced/experimental-models.md#tests.
_NEEDS_FULL_DEPTH = pytest.mark.skipif(
    not _FULL_DEPTH_OPT_IN,
    reason=(
        "needs the full 40-layer Aya Expanse 32B loaded through TransformerLens across GPUs, "
        "which transformer-lens 3.8.0 cannot do (n_devices block placement bug); set "
        "CIRCUITKIT_RUN_AYA_EXPANSE_32B_FULL=1 to run it anyway"
    ),
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_OPT_IN and _HAS_TOKEN),
        reason=(
            "real-weight Aya Expanse 32B parity: set CIRCUITKIT_RUN_AYA_EXPANSE_32B=1 "
            "and HF_TOKEN to run (downloads a ~32B/60GB checkpoint)"
        ),
    ),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, there was a",
    "2 + 2 =",
    "Water boils at a temperature of",
]

_NO_PROCESSING = dict(
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
    refactor_factored_attn_matrices=False,
)


def _env_token():
    env_token = (os.environ.get("HF_TOKEN") or "").strip()
    return env_token if env_token.startswith("hf_") else True


def _kl(hf_logprobs, tl_logprobs) -> float:
    return float(F.kl_div(tl_logprobs, hf_logprobs, log_target=True, reduction="sum").item())


# --------------------------------------------------------------------------- #
# Tier A -- truncated-depth fp32 exactness gate.                              #
# --------------------------------------------------------------------------- #
class TestTierATruncatedExactness:
    """First N_LAYERS=4 real decoder layers, fp32, HF on CPU / TL on GPU.

    The HF side is built by manually reading only the safetensors shards
    that actually contain the needed tensors (embeddings, the first
    N_LAYERS decoder layers, and the final norm -- resolved via the real
    ``model.safetensors.index.json``), rather than
    ``AutoModelForCausalLM.from_pretrained(..., num_hidden_layers=N)``.
    Verified empirically: that kwarg truncates the *model*, but
    ``from_pretrained``'s own checkpoint resolver still requires every shard
    in the index to be present/downloadable first, defeating N-layer
    truncation's entire point of avoiding a full 60GB fetch for this gate.

    Construction must NOT use ``torch.device("meta")`` + ``.to_empty()`` as
    a memory-saving shortcut: ``CohereRotaryEmbedding.inv_freq`` is a
    computed, non-persistent buffer set during normal ``__init__``, not a
    checkpoint tensor -- ``to_empty()`` reallocates it uninitialized (empirically,
    silently all-zero on CPU) and no ``load_state_dict`` call restores it, since
    it was never part of the checkpoint to begin with. That produced a real,
    large-but-silent numerical divergence (KL ~0.08-0.19, max log-prob
    deviation 2-4, 2/4 prompts' argmax wrong) that looked exactly like a
    converter bug until traced to this. Building the model with plain
    ``CohereForCausalLM(hf_cfg)`` (real, if briefly redundant, random init)
    keeps every computed buffer correct; only the checkpoint tensors are
    then overwritten from the real shards.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def hf_model(cls):
        from huggingface_hub import hf_hub_download, snapshot_download
        from safetensors.torch import safe_open
        from transformers import CohereConfig, CohereForCausalLM

        snapshot_dir = Path(
            snapshot_download(
                MODEL_NAME,
                token=_env_token(),
                allow_patterns=[
                    "config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
            )
        )
        cfg_dict = json.loads((snapshot_dir / "config.json").read_text())
        cfg_dict["num_hidden_layers"] = N_LAYERS
        hf_cfg = CohereConfig(**{k: v for k, v in cfg_dict.items() if k != "architectures"})
        model = CohereForCausalLM(hf_cfg).float()

        index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        needed_prefixes = [f"model.layers.{i}." for i in range(N_LAYERS)] + [
            "model.embed_tokens",
            "model.norm",
        ]
        needed_keys = sorted(k for k in index if any(k.startswith(p) for p in needed_prefixes))
        by_shard: dict = {}
        for k in needed_keys:
            by_shard.setdefault(index[k], []).append(k)

        state_dict = {}
        for shard, keys in by_shard.items():
            # Downloads only the specific shard(s) covering the truncated
            # model's tensors -- N_LAYERS=4 needs shards 1-3 + the last
            # (final norm), never the full 14-shard set.
            shard_path = hf_hub_download(MODEL_NAME, shard, token=_env_token())
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key).float()

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert missing == ["lm_head.weight"], f"unexpected missing keys: {missing}"
        assert not unexpected, f"unexpected keys in state dict: {unexpected}"
        model.tie_weights()
        model.eval()
        return model

    @pytest.fixture(scope="class")
    @classmethod
    def tl_model(cls, hf_model):
        import transformer_lens.loading_from_pretrained as tl_loading
        from transformer_lens import HookedTransformer

        from circuitkit.backends import _tl_compat  # noqa: F401 -- apply_patches()
        from circuitkit.backends._tl_compat import cohere as cohere_patch

        # Auth flows via the HF_TOKEN env var (_resolve_hf_token() in cohere.py)
        # -- get_pretrained_model_config forwards **kwargs straight through to
        # the registered config converter's AutoConfig.from_pretrained call,
        # which does not accept an "hf_token" kwarg (only "token"), so passing
        # one here would raise a TypeError rather than authenticate anything.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = tl_loading.get_pretrained_model_config(
            MODEL_NAME, dtype=torch.float32, first_n_layers=N_LAYERS
        )
        cfg = dataclasses.replace(cfg, device="cpu")
        model = HookedTransformer(cfg)
        model.load_and_process_state_dict(
            cohere_patch.convert_cohere2_weights(hf_model, cfg), **_NO_PROCESSING
        )
        model = model.to(device)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_NAME, token=_env_token())

    def _next_token_logprobs(self, tl_model, hf_model, input_ids):
        tl_ids = input_ids.to(next(tl_model.parameters()).device)
        hf_ids = input_ids.to(next(hf_model.parameters()).device)
        with torch.no_grad():
            tl_logits = tl_model(tl_ids)[0, -1].float().cpu()
            hf_logits = hf_model(hf_ids).logits[0, -1].float().cpu()
        return F.log_softmax(tl_logits, dim=-1), F.log_softmax(hf_logits, dim=-1)

    def test_real_config_pins_32b_distinguishing_values(self, hf_model):
        """The gate's correctness claim depends on these real values, not
        the offline tests' fakes -- pin them against the actual Hub config.

        ``rope_theta`` is read via the same ``_registry.rope_theta()`` helper
        ``cohere.py`` itself uses: ``CohereConfig`` has no top-level
        ``rope_theta`` attribute on this transformers version (it moved under
        ``rope_parameters``), so asserting on it directly would raise
        ``AttributeError`` rather than test anything.
        """
        from circuitkit.backends._tl_compat import registry as _registry

        assert float(_registry.rope_theta(hf_model.config)) == 4_000_000
        assert float(hf_model.logit_scale) == pytest.approx(0.0625)
        assert hf_model.config.num_attention_heads == 64
        assert hf_model.config.num_key_value_heads == 8

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_next_token_distribution_matches_hf(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = self._next_token_logprobs(tl_model, hf_model, input_ids)

        kl = _kl(hf_lp, tl_lp)
        max_dev = float((tl_lp - hf_lp).abs().max().item())
        assert (
            kl < KL_TOL_TIER_A
        ), f"prompt {prompt!r}: KL(HF||TL)={kl:.2e} exceeds {KL_TOL_TIER_A:.0e}"
        assert max_dev < MAX_LOGPROB_TOL_TIER_A, (
            f"prompt {prompt!r}: max log-prob deviation {max_dev:.2e} exceeds "
            f"{MAX_LOGPROB_TOL_TIER_A:.0e} -- suspect 8:1 GQA grouping / rope_theta=4e6 / "
            f"logit_scale=0.0625 fold"
        )

    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_argmax_next_token_agrees(self, tl_model, hf_model, tokenizer, prompt):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        tl_lp, hf_lp = self._next_token_logprobs(tl_model, hf_model, input_ids)
        assert int(tl_lp.argmax()) == int(
            hf_lp.argmax()
        ), f"prompt {prompt!r}: TL and HF disagree on the top next token"

    def test_no_layer_skips_rotary(self, tl_model):
        n_heads = tl_model.cfg.n_heads
        d_head = tl_model.cfg.d_head
        x = torch.randn(1, 5, n_heads, d_head, device=next(tl_model.parameters()).device)
        for i, block in enumerate(tl_model.blocks):
            out = block.attn.apply_rotary(x)
            assert out is not x, f"block {i}: expected rotary to be applied, but it was skipped"


# --------------------------------------------------------------------------- #
# Tier B -- full-depth bf16 sanity, one full copy resident at a time.        #
# --------------------------------------------------------------------------- #
@_NEEDS_FULL_DEPTH
class TestTierBFullDepthBf16Sanity:
    """Full 40-layer checkpoint at bf16. HF computes+saves log-probs then is
    freed; TL then loads (unprocessed, matching ``load_model``'s own
    fold_ln=False/center_writing_weights=False/center_unembed=False/
    fold_value_biases=False) and is compared against the saved HF log-probs
    -- never two full bf16 copies resident simultaneously.

    The measured noise floor (HF eager vs sdpa attention, identical prompts,
    identical bf16) sets this tier's KL tolerance, rather than reusing Tier
    A's fp32 1e-4 -- see the class docstring's rationale. Both the floor and
    the TL-vs-HF numbers are asserted and printed (captured in -s runs / the
    final report) so the actual measured tolerance is on record, not assumed.

    **Status: blocked by a transformer-lens bug, not a cohere1/32B bug.**
    The full bf16 checkpoint (~60 GiB of weights, plus TL's separate
    ``W_U``, about 64 GiB in total) needs more memory than a single typical
    GPU provides, so this tier needs TransformerLens's ``n_devices``
    multi-GPU loading. That path is broken in transformer-lens==3.8.0:
    ``HookedTransformer.move_model_modules_to_device`` places each block
    with ``get_best_available_device(cfg)`` (the device with the most free
    memory, ignoring the block index), while the forward pass moves the
    residual stream per block with the index-based
    ``get_device_for_block_index(i, cfg)``. The two disagree. Symptoms:
    ``RuntimeError: Expected all tensors to be on the same device`` inside
    ``layer_norm.py``, or nearly the whole model placed on one GPU followed
    by an out-of-memory error. The HF side loads with
    ``device_map="balanced"`` across GPUs; the TL load needs a single GPU
    large enough to hold the full bf16 model unprocessed.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def device(cls):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture(scope="class")
    @classmethod
    def n_gpus(cls):
        # The full bf16 checkpoint (~60.2 GiB weights, +TL's extra W_U on the
        # TL side) does not fit on most single GPUs -- see "GPU memory
        # reality" in docs/advanced/experimental-models.md. Shard across
        # however many CUDA devices this process can see (the caller controls
        # that via CUDA_VISIBLE_DEVICES, per CircuitKIT's own GPU-budget
        # discipline); single-GPU/CPU runs fall back to one copy as before.
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL_NAME, token=_env_token())

    @pytest.fixture(scope="class")
    @classmethod
    def noise_floor_and_hf_logprobs(cls, device, n_gpus, tokenizer):
        """Load HF bf16 twice (eager, then sdpa attention) on the same
        prompts to measure the bf16 noise floor, saving the eager run's
        log-probs (CPU tensors) for the TL comparison below, then free both
        HF copies before TL ever loads."""
        from transformers import AutoModelForCausalLM

        def _logprobs(attn_impl):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.bfloat16,
                device_map="balanced" if n_gpus > 1 else device,
                token=_env_token(),
                attn_implementation=attn_impl,
            )
            model.eval()
            # With device_map="balanced" the embedding layer (where input ids
            # must land) is not necessarily cuda:0 -- read the real input
            # device from the loaded model instead of assuming `device`.
            input_device = model.get_input_embeddings().weight.device
            out = {}
            with torch.no_grad():
                for prompt in PROMPTS:
                    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(input_device)
                    logits = model(ids).logits[0, -1].float().cpu()
                    out[prompt] = F.log_softmax(logits, dim=-1)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return out

        eager_lp = _logprobs("eager")
        sdpa_lp = _logprobs("sdpa")

        floor_kl = {p: _kl(eager_lp[p], sdpa_lp[p]) for p in PROMPTS}
        floor_max_dev = {p: float((eager_lp[p] - sdpa_lp[p]).abs().max().item()) for p in PROMPTS}
        measured_floor_kl = max(floor_kl.values())
        print(
            f"\n[Tier B] bf16 noise floor (HF eager vs sdpa): KL={floor_kl}, max_dev={floor_max_dev}"
        )

        return {
            "hf_logprobs": eager_lp,
            "measured_floor_kl": measured_floor_kl,
            "floor_kl_per_prompt": floor_kl,
        }

    @pytest.fixture(scope="class")
    @classmethod
    def tl_logprobs(cls, device, tokenizer):
        from circuitkit import load_model

        # Auth via HF_TOKEN env var, same reasoning as the Tier A fixture
        # above -- load_model forwards kwargs to HookedTransformer.from_
        # pretrained, which has no dedicated hf_token parameter either.
        #
        # Deliberately NOT passing n_devices here: transformer-lens==3.8.0's
        # multi-GPU block placement is broken (see the class docstring for
        # the root cause), so n_devices>1 fails with a device mismatch
        # rather than sharding the model. This fixture needs a single GPU
        # large enough to hold the full bf16 checkpoint unprocessed, and
        # runs out of memory on anything smaller.
        model = load_model(MODEL_NAME, dtype="bfloat16", device=device)
        model.eval()
        out = {}
        with torch.no_grad():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                logits = model(ids)[0, -1].float().cpu()
                out[prompt] = F.log_softmax(logits, dim=-1)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def test_bf16_kl_within_measured_noise_floor(self, noise_floor_and_hf_logprobs, tl_logprobs):
        hf_lp = noise_floor_and_hf_logprobs["hf_logprobs"]
        floor = noise_floor_and_hf_logprobs["measured_floor_kl"]
        # Generous margin over the measured floor (TL vs HF crosses more
        # code paths -- a full reimplementation, not just an attention
        # kernel swap -- than eager-vs-sdpa does within HF itself); the
        # actual numbers are printed above and belong in the PR body, not
        # hardcoded here as if they were a priori known.
        tol = max(floor * 10, 1e-2)
        for prompt in PROMPTS:
            kl = _kl(hf_lp[prompt], tl_logprobs[prompt])
            print(
                f"[Tier B] prompt={prompt!r} KL(HF||TL)={kl:.2e} (floor={floor:.2e}, tol={tol:.2e})"
            )
            assert kl < tol, (
                f"prompt {prompt!r}: KL(HF||TL)={kl:.2e} exceeds {tol:.2e} "
                f"(measured bf16 noise floor was {floor:.2e})"
            )

    def test_bf16_top1_agrees_on_every_prompt(self, noise_floor_and_hf_logprobs, tl_logprobs):
        hf_lp = noise_floor_and_hf_logprobs["hf_logprobs"]
        for prompt in PROMPTS:
            assert int(tl_logprobs[prompt].argmax()) == int(
                hf_lp[prompt].argmax()
            ), f"prompt {prompt!r}: TL and HF disagree on the top next token at bf16/full depth"
