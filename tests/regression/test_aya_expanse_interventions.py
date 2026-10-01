"""Group F — gated real-weight interventions regression for Aya Expanse 8B (cohere1).

Stage 5 (interventions & evaluation validation): proves the third of the
three required CircuitKIT surfaces -- discovery (Stage 3) and evaluation
(``test_aya_expanse_evaluation.py``) are the other two -- by exercising all
three intervention types against the real ``CohereLabs/aya-expanse-8b``
checkpoint, resolved through the ``"cohere"`` ``arch_registry`` family
(which also covers tiny-aya and Command R7B -- see
``src/circuitkit/applications/arch_registry.py``):

1. **Pruning** (``applications/pruning/score_extractor.build_importance_dict``):
   resolves real ``k_proj``/``gate_proj`` ``nn.Linear`` modules on a small
   layer subset and produces correctly-shaped importance tensors.
2. **Quantization** (``applications/quantization/quant_utils``): confirms
   ``detect_model_architecture``/``get_arch_config``/``validate_model_paths``
   resolve on the real loaded model, and that ``build_patterns`` (the
   fnmatch-pattern builder ``circuit_quantize`` hands to optimum-quanto's
   ``include=``) actually matches real submodule names. ``optimum-quanto``
   is an optional dependency not installed in this environment, so only
   target-module *resolution* is exercised here, per the guide's explicit
   allowance.
3. **Weight steering** (``applications/steering/weight_steering.py``):
   confirms ``CircuitWeightSteering`` accepts a circuit (a dict of
   ``"A{layer}.{head}"`` -> score, the module's own contract format -- a
   different naming convention from the discovery ``Graph``'s
   ``"a{layer}.h{head}"`` node names, pre-existing and unrelated to this
   integration) and a real ``HookedTransformer``, resolves real per-head
   weight slices (exercising the GQA query-head -> kv-head floor-div
   mapping), and that ``apply_steering`` produces a distinct model with the
   targeted slices actually changed and a still-finite forward pass. The
   expensive contrastive fine-tune is skipped (out of scope for a smoke
   test) by constructing the steering vector directly, which still
   exercises the real ``apply_steering`` code path.

Gating mirrors ``test_aya_expanse_discovery.py``: opt in with
``CIRCUITKIT_RUN_AYA_EXPANSE=1`` and provide ``HF_TOKEN``. Marked ``slow``.

    CIRCUITKIT_RUN_AYA_EXPANSE=1 HF_TOKEN=... \
        python -m pytest tests/regression/test_aya_expanse_interventions.py -v
"""

from __future__ import annotations

import fnmatch
import os

import pytest
import torch

MODEL_NAME = "CohereLabs/aya-expanse-8b"
FAMILY = "cohere"

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
            "real-weight Aya Expanse interventions: set "
            "CIRCUITKIT_RUN_AYA_EXPANSE=1 and HF_TOKEN to run"
        ),
    ),
]

# Small layer subset -- interventions are smoke-tested, not exhaustively run
# across every layer.
_LAYER_SUBSET = [0, 1]

_N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
_HF_DEVICE = "cuda:0" if _N_GPUS >= 1 else "cpu"
_TL_DEVICE = "cuda:1" if _N_GPUS >= 2 else _HF_DEVICE


@pytest.fixture(scope="module")
def hf_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=_HF_DEVICE,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    yield model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestPruningScoreExtraction:
    """score_extractor.build_importance_dict resolves real modules via arch_registry."""

    def test_build_importance_dict_resolves_real_modules(self, hf_model):
        from circuitkit.applications.arch_utils import (
            get_arch_config,
            get_attn_proj,
            get_layers,
            get_mlp_proj,
        )
        from circuitkit.applications.pruning.score_extractor import build_importance_dict

        n_kv = hf_model.config.num_key_value_heads
        d_mlp = hf_model.config.intermediate_size

        kv_head_scores = {
            (layer, kv): float(kv + 1) for layer in _LAYER_SUBSET for kv in range(n_kv)
        }
        mlp_scores = {layer: torch.randn(d_mlp) for layer in _LAYER_SUBSET}

        scores_dict = build_importance_dict(
            hf_model,
            kv_head_scores,
            mlp_scores,
            attn_layers=_LAYER_SUBSET,
            mlp_layers=_LAYER_SUBSET,
        )

        arch_cfg = get_arch_config(FAMILY)
        layers = get_layers(hf_model, arch_cfg)

        assert len(scores_dict) == 2 * len(_LAYER_SUBSET), (
            f"expected one k_proj + one gate_proj entry per layer in the subset, "
            f"got {len(scores_dict)} entries"
        )
        for layer_idx in _LAYER_SUBSET:
            k_proj = get_attn_proj(layers[layer_idx], arch_cfg, "k_proj")
            assert k_proj in scores_dict, f"layer {layer_idx}: k_proj not in importance dict"
            assert scores_dict[k_proj].shape == (k_proj.weight.shape[0],)

            gate_proj = get_mlp_proj(layers[layer_idx], arch_cfg, "gate_proj")
            assert gate_proj in scores_dict, f"layer {layer_idx}: gate_proj not in importance dict"
            assert scores_dict[gate_proj].shape == (d_mlp,)


class TestQuantizationTargetResolution:
    """quant_utils target-module resolution against the real loaded model."""

    def test_arch_resolution_and_patterns_match_real_modules(self, hf_model):
        from circuitkit.applications import (
            detect_model_architecture,
            get_arch_config,
            validate_model_paths,
        )
        from circuitkit.applications.quantization.quant_utils import (
            build_patterns,
            build_quantization_plan,
            compute_layer_scores,
        )

        family = detect_model_architecture(hf_model)
        assert family == FAMILY
        arch_cfg = get_arch_config(family)
        validate_model_paths(hf_model, arch_cfg)  # should not raise

        named = dict(hf_model.named_modules())
        module_names = list(named.keys())
        for component in ("attn", "mlp"):
            patterns = build_patterns(_LAYER_SUBSET, component, arch_cfg)
            assert patterns, f"{component}: build_patterns produced no patterns"
            for pattern in patterns:
                matches = fnmatch.filter(module_names, pattern)
                assert matches, (
                    f"{component}: pattern {pattern!r} matched no real submodule "
                    f"of {MODEL_NAME} -- target-module resolution is broken"
                )
                linear_matches = [m for m in matches if isinstance(named[m], torch.nn.Linear)]
                assert linear_matches, (
                    f"{component}: pattern {pattern!r} matched {len(matches)} module(s) "
                    f"but none are nn.Linear -- nothing quantizable would be targeted"
                )

        n_layers = hf_model.config.num_hidden_layers
        q_head_scores = {
            (layer, head): float(layer + head) for layer in range(n_layers) for head in range(2)
        }
        mlp_scores = {
            layer: torch.randn(hf_model.config.intermediate_size) for layer in range(n_layers)
        }
        attn_scores, mlp_layer_scores = compute_layer_scores(q_head_scores, mlp_scores, n_layers)
        plan = build_quantization_plan(attn_scores, mlp_layer_scores, high_fraction=0.3)
        assert set(plan.keys()) == {"attn", "mlp"}
        assert len(plan["attn"]) == n_layers
        assert len(plan["mlp"]) == n_layers
        assert set(plan["attn"].values()) <= {"high", "mid", "low"}


class TestWeightSteeringSetup:
    """CircuitWeightSteering accepts a circuit + real model and applies steering."""

    def test_steering_resolves_heads_and_applies_on_real_weights(self):
        from transformer_lens import HookedTransformer

        from circuitkit.applications.steering.weight_steering import (
            CircuitWeightSteering,
            get_head_weight_info,
            get_head_weight_slice,
        )

        model = HookedTransformer.from_pretrained(
            MODEL_NAME, device=_TL_DEVICE, dtype=torch.bfloat16
        )
        try:
            n_layers = model.cfg.n_layers
            n_heads = model.cfg.n_heads
            n_kv = getattr(model.cfg, "n_key_value_heads", n_heads) or n_heads
            assert n_kv < n_heads, "expected Aya Expanse's GQA ratio"

            circuit_scores = {
                f"A{layer}.{head}": float((layer + 1) * (head + 1))
                for layer in range(n_layers)
                for head in range(n_heads)
            }

            cws = CircuitWeightSteering(model, circuit_scores, top_k_frac=0.02)
            assert len(cws.head_names) > 0, "no attention heads selected for steering"

            probe_heads = cws.head_names[:3]
            for name in probe_heads:
                info = get_head_weight_info(model, name)
                for key in ("W_Q", "W_K", "W_V", "W_O"):
                    assert key in info, f"{name}: {key} slice not resolved"
                    param, idx = info[key]
                    assert 0 <= idx < param.shape[0]

            steering = {}
            for name in probe_heads:
                slices = get_head_weight_slice(model, name)
                steering[name] = {k: torch.full_like(v, 1e-3) for k, v in slices.items()}
            cws.head_names = probe_heads
            cws._steering_vector = steering

            steered = cws.apply_steering(k=1.0)
            try:
                assert steered is not model
                for name in probe_heads:
                    orig = get_head_weight_slice(model, name)
                    new = get_head_weight_slice(steered, name)
                    for key in orig:
                        assert not torch.allclose(
                            orig[key], new[key]
                        ), f"{name}.{key} unchanged after apply_steering"

                tokens = steered.to_tokens("The quick brown fox jumps")
                with torch.inference_mode():
                    logits = steered(tokens)
                assert torch.isfinite(logits).all(), "steered model produced non-finite logits"
            finally:
                del steered
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
