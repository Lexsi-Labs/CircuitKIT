# User Guide

These guides go one level deeper than the [Quick Start](../getting-started/quickstart.md). Each covers one stage of the workflow with runnable snippets and the config knobs that actually matter.

If you're reading start to finish, this is the spine:

1. **[Pipeline](../user-guide/pipeline-overview.md)** ties the loop together — one object that discovers, evaluates, and intervenes. Start here.
2. **[Data](../user-guide/data.md)** — what discovery needs, the built-in tasks, and how to bring your own CSV, JSONL, or HuggingFace dataset.
3. **[Evaluation](../user-guide/evaluation.md)** — the 6-pillar faithfulness framework, run as a subset or a full audit.
4. **[Applications](../user-guide/applications.md)** — prune, quantize, edit, steer, or fine-tune once you trust the circuit.

The cards below link every guide, including the deeper references (Selectors, Visualization, Memory Optimization, Circuit Artifacts).

<div class="grid cards" markdown>

-   :octicons-workflow-24:{ .lg .middle } **Pipeline**

    ---

    The stateful `Pipeline` orchestrator — discover, evaluate, prune, export in chained calls.

    [:octicons-arrow-right-24: Pipeline guide](../user-guide/pipeline-overview.md)

-   :material-database:{ .lg .middle } **Data**

    ---

    What discovery needs, the 16 built-in tasks, and bringing your own CSV, JSONL, or HuggingFace dataset.

    [:octicons-arrow-right-24: Data overview](../user-guide/data.md)

-   :material-scale-balance:{ .lg .middle } **Evaluation**

    ---

    Run the 6-pillar faithfulness framework. Choose a subset or run the full audit.

    [:octicons-arrow-right-24: Evaluation guide](../user-guide/evaluation.md)

-   :material-toolbox:{ .lg .middle } **Applications**

    ---

    Prune, quantize, edit, steer, or fine-tune based on your discovered circuit.

    [:octicons-arrow-right-24: Applications guide](../user-guide/applications.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Visualization**

    ---

    Circuit graphs, comparison dashboards, score histograms, and Jupyter widgets.

    [:octicons-arrow-right-24: Visualization guide](../user-guide/visualization.md)

-   :material-filter-variant:{ .lg .middle } **Selectors**

    ---

    The selector registry — extend CircuitKit with custom scoring methods.

    [:octicons-arrow-right-24: Selectors guide](../user-guide/selectors.md)

-   :material-list-box:{ .lg .middle } **Built-in tasks**

    ---

    The 16 registered tasks, their metrics, and chat-template defaults.

    [:octicons-arrow-right-24: Built-in tasks](../user-guide/tasks.md)

-   :material-help-circle:{ .lg .middle } **Troubleshooting**

    ---

    Common issues and their solutions.

    [:octicons-arrow-right-24: Solutions](../user-guide/troubleshooting.md)

-   :material-steering:{ .lg .middle } **Steering**

    ---

    Activation steering and contrastive weight steering — details and configuration.

    [:octicons-arrow-right-24: Steering guide](steering.md)

-   :material-chip:{ .lg .middle } **Memory Optimization**

    ---

    Run EAP-IG on large models within limited VRAM.

    [:octicons-arrow-right-24: Memory guide](../advanced/memory-optimization.md)

-   :material-file-code:{ .lg .middle } **Circuit Artifacts**

    ---

    The `.pt` / `_scores.json` / `_scores.pt` format specification.

    [:octicons-arrow-right-24: Artifacts guide](../advanced/circuit-artifacts.md)

</div>
