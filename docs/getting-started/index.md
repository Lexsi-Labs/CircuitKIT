# Getting started

CircuitKit runs a three-stage loop: **discover** the circuit driving a behaviour, **evaluate** how faithful that circuit is, then **act on it** — prune, quantize, edit, steer, or fine-tune — and export a reloadable HuggingFace checkpoint.

This page just orients you. Two links get you moving:

- **[Installation](installation.md)** — `pip install circuitkit`, plus GPU extras and troubleshooting.
- **[Quick Start](quickstart.md)** — discover, evaluate, prune, and export your first circuit on GPT-2 in a few minutes, no GPU required.

## Which tool for which job

| I want to… | Start with | Namespace |
|---|---|---|
| find a circuit | `discover_circuit({...})` | `circuitkit.api` |
| score faithfulness | `evaluate_circuit({...})` | `circuitkit.api` |
| prune model to circuit | `ck.prune(model, circuit, ...)` | `circuitkit.quick` |
| quantize with circuit guidance | `ck.quantize(model, circuit, ...)` | `circuitkit.quick` |
| benchmark a compressed model | `ck.benchmark(path, tasks)` | `circuitkit.quick` |
| run a stateful, chained workflow | `Pipeline(model, task)` | `circuitkit` |
| use the CLI | `circuitkit discover --help` | CLI |

## Where to go next

| | |
|---|---|
| **[Installation](installation.md)** | Install, GPU extras, troubleshooting |
| **[Quick Start](quickstart.md)** | Your first circuit: discover → evaluate → prune → export → benchmark |
| **[Core Concepts](core-concepts.md)** | Circuits, tasks, faithfulness, interventions |
| **[Configuration](configuration.md)** | Dict-config, Flat API, Pipeline, CLI, YAML — which interface when |
| **[Taxonomy](taxonomy.md)** | The end-to-end workflow at a glance |
