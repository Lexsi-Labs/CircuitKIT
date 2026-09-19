# Algorithm Overview

CircuitKit ships **13 discovery algorithms** across 4 backends. This page explains how to choose the right one. Note that "ships" is not "validated". Six are Stable (`eap`, `eap-ig`, `eap-gp`, `acdc`, `ibcircuit`, `cdt`) and have been tested across the GPT-2, Llama, Gemma, and Qwen families. The remaining seven are Research: implemented and validated on GPT-2/IOI but not yet exercised at scale or across architectures. Three of the Stable algorithms carry documented caveats: `acdc` is node-only by construction and slow, `ibcircuit` has a memory ceiling on multi-billion-parameter models at aggressive settings, and `cdt` scores on RoPE models are approximate.

<div class="grid cards" markdown>

-   :material-chart-timeline-variant:{ .lg .middle } **EAP Family**  Stable

    ---

    Gradient-based edge attribution patching. Default `eap-ig` is validated across GPT-2 through Llama-3B and Gemma-4B.

    [:octicons-arrow-right-24: EAP variants](eap.md)

-   :material-scissors-cutting:{ .lg .middle } **ACDC**  Stable

    ---

    Greedy edge-pruning. Produces minimal circuits. Node-only and slow.

    [:octicons-arrow-right-24: ACDC details](acdc.md)

-   :material-bottle-tonic:{ .lg .middle } **IBCircuit**  Stable

    ---

    Information-bottleneck noise model. No paired corruption data needed.

    [:octicons-arrow-right-24: IBCircuit details](ibcircuit.md)

-   :material-puzzle:{ .lg .middle } **CD-T**  Stable

    ---

    Contextual decomposition through transformers. Clean inputs only. Scores on RoPE models are approximate.

    [:octicons-arrow-right-24: CD-T details](cdt.md)

</div>

## Quick selection guide

| Goal | Algorithm | Why |
|---|---|---|
| New to CircuitKit, any model | `eap-ig` | Stable, fast, validated across model families |
| Speed over precision | `eap` | ~30% faster; slightly noisier |
| Minimal circuit | `acdc` | Greedy edge-pruning. Node-only and slow |
| Information-flow analysis | `ibcircuit` | No paired data needed |
| Large model (3B+) | `eap-ig` | Validated at this scale. `ibcircuit` has a memory ceiling here and `acdc` is slow |

## All 13 algorithms

### EAP family

| Algorithm | Tier | Description |
|---|---|---|
| `eap-ig` |  Stable | EAP + Integrated Gradients — **default** |
| `eap` |  Stable | Vanilla EAP — fast baseline |
| `eap-gp` |  Stable | EAP-GP / GradPath |
| `eap-ig-activations` |  Research | IG over node activations |
| `eap-clean-corrupted` |  Research | EAP with both clean/corrupted passes |
| `eap-exact` |  Research | Exact EAP (quadratic cost) |
| `atp-gd` |  Research | Attribution Patching with GradDrop (AtP+GD) |
| `relp` |  Research | Relevance Patching (LRP-style) |
| `peap` |  Research | Position-aware EAP (PEAP) |
| `eap-ifr` |  Research | Information Flow Routes (IFR) |

**When to use:** Start with `eap-ig`. Use `eap` if speed is the bottleneck. Use Research variants only for algorithm comparison studies.

### ACDC

| Algorithm | Tier | Description |
|---|---|---|
| `acdc` |  Stable | Greedy edge-pruning; node-only by construction (edge search); slow |

When you want a minimal circuit and can afford a slow search.

### IBCircuit

| Algorithm | Tier | Description |
|---|---|---|
| `ibcircuit` |  Stable | Noise-model approach; clean-only data |

When paired (clean, corrupted) examples are hard to construct.

!!! warning "Memory ceiling on multi-billion-parameter models"
    IBCircuit trains a noise model end-to-end, doubling memory. At aggressive settings it can OOM on models above ~3B parameters.

### CD-T

| Algorithm | Tier | Description |
|---|---|---|
| `cdt` |  Stable | Clean inputs only; frozen-RoPE attention approximation |

CD-T uses a frozen-RoPE attention approximation (Q/K are not decomposed) and a 50/50 gated-MLP cross-term split, so its scores on RoPE models are approximate.

## Model compatibility

| Algorithm | GPT-2 | Llama 3.x | Gemma 2/3 | Qwen 2.5 |
|---|---|---|---|---|
| `eap-ig` | ✅ | ✅ | ✅ | ✅ |
| `eap` | ✅ | ✅ | ✅ | ✅ |
| `eap-gp` | ✅ | ✅ | ✅ | ✅ |
| `acdc` | ✅ | ⚠️ | ⚠️ | ⚠️ |
| `ibcircuit` | ✅ | ⚠️ | ⚠️ | ⚠️ |
| `cdt` | ✅ | ⚠️ | ⚠️ | ⚠️ |
| `eap-ig-activations` | ✅ | ❌ | ❌ | ❌ |
| `eap-clean-corrupted` | ✅ | ❌ | ❌ | ❌ |
| Research tier | ✅ | ❌ | ❌ | ❌ |

✅ Tested  ⚠️ Tested, with a documented caveat  ❌ Not validated

Marks outside the GPT-2 column for `eap-gp`, `acdc`, `ibcircuit`, and `cdt` restate the stable-tier statement at family level. Faithfulness scores beyond GPT-2 are published only for `eap` and `eap-ig` (see [Audit Results](../trust/results.md)).

Caveats: `acdc` is slow above GPT-2 scale, `ibcircuit` has a memory ceiling on multi-billion-parameter models at aggressive settings, and `cdt` scores on RoPE models (Llama, Gemma, Qwen) are approximate.

## Algorithm-specific config keys

| Algorithm | Key | Default | Description |
|---|---|---|---|
| `eap-ig` | `ig_steps` | `5` | Integration steps for IG |
| `acdc` | `tao_bases` | `[1, 3, 5, 7, 9]` | Bases for the tao threshold sweep |
| `acdc` | `tao_exps` | `[-5, -4, -3, -2]` | Exponents for the tao threshold sweep |
| `acdc` | `faithfulness_target` | `kl_div` | Metric optimized during pruning (`kl_div` or `mse`) |
| `ibcircuit` | `num_epochs` | `1000` | Training epochs for noise model |
| `ibcircuit` | `beta` | `0.001` | IB regularization weight |

## Next steps

- [:octicons-arrow-right-24: Stability Tiers](stability-tiers.md)
- [:octicons-arrow-right-24: EAP Family](eap.md)
- [:octicons-arrow-right-24: ACDC](acdc.md)
- [:octicons-arrow-right-24: Capability Matrix](capability-matrix.md)
