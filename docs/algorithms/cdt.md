# CD-T

**Contextual Decomposition through Transformers** (CD-T) adapts the classical Contextual Decomposition (CD) approach to transformer attention layers. It decomposes a layer's contribution into components attributable to specific input tokens.

**Stability tier:**  Research — validated on GPT-2 IOI only. Uses a frozen-RoPE approximation and a 50/50 gated-MLP cross-term split that are not valid for GQA or SwiGLU architectures.

---

## Important Caveats

!!! warning "Research-tier only"
    CD-T is implemented for algorithm comparison and research purposes. It uses a frozen RoPE attention approximation (Q/K are not decomposed) and a simplified MLP cross-term split. These approximations are specific to GPT-2's attention and MLP structure and are **not valid for modern architectures** (Llama, Gemma, Qwen, etc.).
    
    For non-GPT-2 models: use `eap-ig` (Stable).

---

## Usage

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "gpt2", "precision": "float32"},
    "discovery": {
        "algorithm": "cdt",
        "task": "ioi",
        "level": "node",     # CD-T supports node-level only
        "data_params": {"num_examples": 64},
    },
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

!!! note "Node-level only"
    CD-T only supports `level="node"`. If `level="neuron"` is requested, CircuitKit raises a `ValueError` — pick `level="node"`, or use an algorithm that supports neuron-level discovery (e.g. `eap`, `eap-ig`, `ibcircuit`).

---

## CD-T vs. EAP-IG

| Aspect | CD-T | EAP-IG |
|--------|------|--------|
| Approach | Contextual decomposition | Gradient attribution |
| Data | Clean only (or paired) | Paired |
| Architecture scope | GPT-2 only | GPT-2 through 4B |
| Level | Node only | Node or neuron |
| Tier |  Research |  Stable |

---

## When to Use CD-T

- You are **replicating a CD-based paper result** on GPT-2
- You are contributing to CircuitKit's algorithm diversity study
- You specifically need a **non-gradient decomposition** approach

For any other use case: use `eap-ig`.

---

## Implementation Notes

CD-T in CircuitKit is invoked through `discover_circuit({"algorithm": "cdt", ...})` rather than the lower-level `run_cdt_discovery()` function directly. The package re-exports `circuitkit.backends.cdt.wrappers`, `circuitkit.backends.cdt.core`, and `circuitkit.backends.cdt.basic` for research use.

The `cdt` extra is not required for CD-T (it is bundled with the core). The `cdt` extra installs `captum`, `lime`, and `shap` for experimental attribution comparison work.

---

## Next Steps

- [EAP Family](eap.md) — the recommended alternative
- [Stability Tiers](stability-tiers.md) — full tier table
- [Algorithms: Overview](overview.md) — algorithm selection guide
