# Algorithm Overview

CircuitKit ships **13 discovery algorithms** across 4 backends. This page explains how to choose the right one. Note that "ships" is not "validated": only 2 are validated at production scale — `eap` and `eap-ig` (Stable). `acdc` and `ibcircuit` are Experimental (GPT-2 scale; `ibcircuit` OOMs above ~3B), and the remaining 9 are Research (validated only on GPT-2 IOI).

<div class="grid cards" markdown>

-   :material-chart-timeline-variant:{ .lg .middle } **EAP Family**  Stable

    ---

    Gradient-based edge attribution patching. Default `eap-ig` is validated across GPT-2 through Llama-3B and Gemma-4B.

    [:octicons-arrow-right-24: EAP variants](eap.md)

-   :material-scissors-cutting:{ .lg .middle } **ACDC**  Experimental

    ---

    Greedy edge-pruning. Produces minimal circuits at GPT-2 scale.

    [:octicons-arrow-right-24: ACDC details](acdc.md)

-   :material-bottle-tonic:{ .lg .middle } **IBCircuit**  Experimental

    ---

    Information-bottleneck noise model. No paired corruption data needed.

    [:octicons-arrow-right-24: IBCircuit details](ibcircuit.md)

-   :material-puzzle:{ .lg .middle } **CD-T**  Research

    ---

    Contextual decomposition through transformers. GPT-2 only.

    [:octicons-arrow-right-24: CD-T details](cdt.md)

</div>

## Quick selection guide

| Goal | Algorithm | Why |
|---|---|---|
| New to CircuitKit, any model | `eap-ig` | Stable, fast, validated across model families |
| Speed over precision | `eap` | ~30% faster; slightly noisier |
| GPT-2 exploratory | `acdc` | Minimal circuits via greedy edge-pruning |
| Information-flow analysis | `ibcircuit` | No paired data needed |
| Large model (3B+) | `eap-ig` | Only Stable-tier validated at this scale |

## All 13 algorithms

### EAP family

| Algorithm | Tier | Description |
|---|---|---|
| `eap-ig` |  Stable | EAP + Integrated Gradients — **default** |
| `eap` |  Stable | Vanilla EAP — fast baseline |
| `eap-ig-activations` |  Research | IG over node activations |
| `eap-clean-corrupted` |  Research | EAP with both clean/corrupted passes |
| `eap-exact` |  Research | Exact EAP (quadratic cost) |
| `atp-gd` |  Research | Attribution Patching with GradDrop (AtP+GD) |
| `eap-gp` |  Research | EAP-GP / GradPath |
| `relp` |  Research | Relevance Patching (LRP-style) |
| `peap` |  Research | Position-aware EAP (PEAP) |
| `eap-ifr` |  Research | Information Flow Routes (IFR) |

**When to use:** Start with `eap-ig`. Use `eap` if speed is the bottleneck. Use Research variants only for algorithm comparison studies.

### ACDC

| Algorithm | Tier | Description |
|---|---|---|
| `acdc` |  Experimental | Greedy edge-pruning; GPT-2 scale |

When you want a minimal circuit and you're working at GPT-2 scale.

### IBCircuit

| Algorithm | Tier | Description |
|---|---|---|
| `ibcircuit` |  Experimental | Noise-model approach; clean-only data |

When paired (clean, corrupted) examples are hard to construct.

!!! warning "OOM risk above ~3B"
    IBCircuit trains a noise model end-to-end, doubling memory. Known to OOM above ~3B parameters.

### CD-T

| Algorithm | Tier | Description |
|---|---|---|
| `cdt` |  Research | Frozen-RoPE attention approximation; GPT-2 IOI only |

Only for research replication.

## Model compatibility

| Algorithm | GPT-2 | Llama 3.x | Gemma 2/3 | Qwen 2.5 |
|---|---|---|---|---|
| `eap-ig` | ✅ | ✅ | ✅ | ✅ |
| `eap` | ✅ | ✅ | ✅ | ✅ |
| `eap-ig-activations` | ✅ | ❌ | ❌ | ❌ |
| `eap-clean-corrupted` | ✅ | ❌ | ❌ | ❌ |
| `acdc` | ✅ | ⚠️ | ⚠️ | ⚠️ |
| `ibcircuit` | ✅ | ⚠️ | ❌ | ❌ |
| `cdt` | ✅ | ❌ | ❌ | ❌ |
| Research tier | ✅ | ❌ | ❌ | ❌ |

✅ Validated  ⚠️ Experimental  ❌ Not validated

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
