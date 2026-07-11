# Stability Tiers

Every algorithm has an explicit **stability tier** describing how broadly it has been validated. The tier is enforced at runtime — experimental and research algorithms emit a `UserWarning`.

```python
from circuitkit.backends import STABILITY, is_stable, is_experimental, is_research

print(is_stable("eap-ig"))      # True
print(is_experimental("acdc"))  # True
print(is_research("cdt"))       # True
print(STABILITY["eap-ig"])      # "stable"
```

##  Stable

Validated across multiple model families and tasks. Suitable for production use.

| Algorithm | Validated on |
|---|---|
| `eap-ig` | GPT-2, Llama-3.2-1B/3B, Gemma-2-2B, Gemma-3-4B, Qwen2.5-1.5B |
| `eap` | GPT-2, small Llama/Gemma |

##  Experimental

Validated on GPT-2 IOI. May fail or OOM on larger models.

| Algorithm | Known limitation |
|---|---|
| `acdc` | Slow and memory-intensive above GPT-2 scale |
| `ibcircuit` | OOM risk above ~3B parameters |

##  Research

Validated only on GPT-2 IOI. Do not rely on for non-GPT-2 models.

| Algorithm |
|---|
| `eap-ig-activations`, `eap-clean-corrupted`, `eap-exact`, `atp-gd`, `eap-gp`, `relp`, `peap`, `eap-ifr`, `cdt` |

## Why tiers matter

Attribution methods are not universally portable. A method validated on GPT-2 IOI may produce meaningless circuits on Llama-3 because of:
- Different attention patterns (GQA vs. MHA)
- Different positional encodings (RoPE vs. absolute)
- Different MLP structures (SwiGLU vs. GELU)
- Different scales (what works at 124M may OOM at 7B)

## Runtime warnings

When you request a non-Stable algorithm, CircuitKit emits a `UserWarning`:
```python
UserWarning: Algorithm 'acdc' is experimental. May fail on larger models or non-IOI tasks. Use 'eap-ig' for production.
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

Start with `eap-ig` (Stable) for any new experiment. Only switch to Experimental or Research algorithms when you have a specific reason (e.g., comparing methods for a paper).
