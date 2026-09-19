# Stability Tiers

Every algorithm has an explicit **stability tier** describing how broadly it has been validated. The tier is enforced at runtime — experimental and research algorithms emit a `UserWarning`.

```python
from circuitkit.backends import STABILITY, is_stable, is_experimental, is_research

print(is_stable("eap-ig"))      # True
print(is_stable("acdc"))        # True
print(is_experimental("acdc"))  # False
print(is_research("relp"))      # True
print(STABILITY["eap-ig"])      # "stable"
```

Of the 13 discovery algorithms, 6 are Stable and 7 are Research. No discovery algorithm is in the Experimental tier at the moment.

##  Stable

Tested across the GPT-2, Llama, Gemma, and Qwen families.

| Algorithm | Notes |
|---|---|
| `eap-ig` | Default. Validated on GPT-2, Llama-3.2-1B/3B, Gemma-2-2B, Gemma-3-4B, Qwen2.5-1.5B. |
| `eap` | Validated on GPT-2 and small Llama/Gemma models. |
| `eap-gp` | EAP-GP / GradPath. |
| `acdc` | Node-only by construction (edge search). Slow. |
| `ibcircuit` | Memory ceiling on multi-billion-parameter models at aggressive settings. |
| `cdt` | Works from clean inputs only. Uses a frozen-RoPE attention approximation (Q/K are not decomposed) and a 50/50 gated-MLP cross-term split, so its scores on RoPE models are approximate. |

##  Experimental

None currently for discovery. The tier, `EXPERIMENTAL_ALGORITHMS`, and `is_experimental` remain part of the API.

##  Research

Implemented and validated on GPT-2/IOI but not yet exercised at scale or across architectures.

| Algorithm |
|---|
| `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `relp`, `peap`, `eap-ifr` |

## Why tiers matter

Attribution methods are not universally portable. A method validated on GPT-2 IOI may produce meaningless circuits on Llama-3 because of:
- Different attention patterns (GQA vs. MHA)
- Different positional encodings (RoPE vs. absolute)
- Different MLP structures (SwiGLU vs. GELU)
- Different scales (what works at 124M may OOM at 7B)

## Runtime warnings

When you request a non-Stable algorithm, CircuitKit emits a `UserWarning`:
```python
UserWarning: Algorithm 'relp' is research-quality (only validated on GPT-2 IOI). Use 'eap-ig' for production.
```

This is expected behaviour. Suppress with `warnings.filterwarnings("ignore")` only after you have verified the algorithm works for your use case.

## Checking before discovery

```python
from circuitkit.backends import is_stable, STABILITY

algo = "my_algorithm"
if not is_stable(algo):
    print(f"WARNING: {algo} is {STABILITY.get(algo, 'unknown')} tier")
```

## Recommendation

Start with `eap-ig` (Stable) for any new experiment. Only switch to Research algorithms when you have a specific reason (e.g., comparing methods for a paper).
