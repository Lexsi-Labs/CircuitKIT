# IBCircuit

**Information-Bottleneck Circuit** (IBCircuit) takes a different approach to circuit discovery: instead of computing gradients with respect to a metric, it trains a **noise model** that learns which activations are necessary for the task.

**Stability tier:**  Experimental — validated on GPT-2 IOI; OOM risk above ~3B parameters.

---

## How It Works

IBCircuit adds learned noise gates to every component in the model. It then minimizes an Information Bottleneck objective:

$$\mathcal{L}_{IB} = \mathcal{L}_{task} + \beta \cdot I(\text{activations}; \text{noise mask})$$

Components whose gates converge to "pass" (low noise) are in the circuit. Components whose gates converge to "block" (high noise) are not.

**Key difference from EAP:** IBCircuit does not require a **corrupted input** — it discovers the circuit from clean examples only. The corruption is learned (the noise model learns what to suppress).

---

## Usage

```python
from circuitkit.api import discover_circuit

circuit = discover_circuit({
    "model": {"name": "gpt2", "precision": "float32"},
    "discovery": {
        "algorithm": "ibcircuit",
        "task": "ioi",
        "level": "node",
        "data_params": {"num_examples": 64},
        "num_epochs": 1000,    # training epochs for the noise model
        "beta": 0.001,         # information-bottleneck regularization strength (default)
    },
    "pruning": {"target_sparsity": 0.3, "scope": "heads"},
    "output_path": "./circuit.pt",
})
```

!!! warning "IBCircuit emits a UserWarning"
    Since IBCircuit is Experimental-tier, `discover_circuit` emits a `UserWarning` when you request it. This is expected.

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | `1000` | Training epochs. More epochs → more refined circuit. |
| `beta` | `0.001` | IB regularization weight. Higher → smaller circuit (more compression). |
| `n_examples` | from `data_params` | Clean examples (no corruption needed). |

---

## Unpaired Data (No Corruption Needed)

IBCircuit works with **clean-only** datasets. You can use it with `Pipeline.from_custom_data()` without providing `corrupt_prompt` / `corrupt_answer`:

```python
from circuitkit import Pipeline

pipe = Pipeline.from_custom_data(
    model_name="gpt2",
    data_path="clean_only.csv",
    clean_prompt="{text}",
    clean_answer="{label}",
    # no corrupt_prompt or corrupt_answer — IBCircuit doesn't need them
)
pipe.discover(algorithm="ibcircuit", n_examples=64, num_epochs=500)
```

---

## IBCircuit vs. EAP-IG

| Aspect | IBCircuit | EAP-IG |
|--------|-----------|--------|
| Data requirement | Clean only | Paired (clean + corrupted) |
| Mechanism | Learned noise gates | Gradient attribution |
| Runtime | Slower (trains a model) | Fast (gradient passes) |
| Memory | 2× model size (OOM risk) | ~1.5× model size |
| Model size | GPT-2 scale (~124M) | Validated to 4B |
| Tier |  Experimental |  Stable |

---

## When to Use IBCircuit

- You **cannot construct paired (clean, corrupted) examples** for your task
- You're doing **information-flow analysis** (IBCircuit reveals which components transmit task-relevant information)
- You're **studying algorithm diversity** at GPT-2 scale

---

## Known Limitations

- **OOM above ~3B** — the noise model doubles memory requirements
- **GPT-2 validated only** — GQA and SwiGLU architectures are untested
- **Slower** than EAP — requires training the noise model, not just a gradient pass

---

## Next Steps

- [EAP Family](eap.md) — faster, stable alternative
- [Custom Data](../user-guide/custom-data.md) — clean-only dataset setup
- [Stability Tiers](stability-tiers.md) — understanding experimental-tier risks
