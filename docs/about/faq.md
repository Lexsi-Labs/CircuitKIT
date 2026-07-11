# FAQ

---

## General

**What is CircuitKit?**

CircuitKit is a mechanistic interpretability toolkit for discovering, evaluating, and intervening on circuits in Transformer language models. A circuit is the minimal subgraph of attention heads and MLP layers that explains a model's behavior on a specific task.

**What models does CircuitKit support?**

For circuit discovery, any model supported by TransformerLens (GPT-2, GPT-Neo, Pythia, Llama, Gemma, Qwen, Mistral, Phi, Falcon, and more). For applications (pruning, quantization), the architecture registry covers production-validated families: `llama`, `qwen`, `gemma`. See [Architecture Registry](../advanced/architecture-registry.md).

**Is CircuitKit production-ready?**

The Stable-tier discovery path (EAP, EAP-IG) is validated. Experimental and research tier backends — including EAP-IG-Activations and EAP-Clean-Corrupted, which are Research tier despite the similar name — are not production-ready. Applications are validated on production families but have known limitations (see release notes).

---

## Discovery

**Which algorithm should I use?**

Start with `eap-ig` (the default). It is the most accurate stable algorithm. For speed over accuracy, use `eap`. For memory-constrained settings, reduce `ig_steps` to 3. See [Algorithm Selection Guide](../algorithms/overview.md).

**How many examples do I need?**

The practical minimum is 32 examples for a reasonable circuit estimate. 128 (the default) gives good results for most tasks. For production audits, use 256+. Built-in tasks like `ioi` generate examples on demand from their templates, so you can request as many as you need.

**Why is my circuit empty?**

Common causes: (1) `sparsity` is too high — try 0.1 or 0.15; (2) the task has very low signal — check `validate_token_alignment`; (3) you're using a chat model with `chat_template_mode="off"` for a downstream task — try `"auto"`.

**How do I run discovery on an instruction-tuned model?**

Use `chat_template_mode="auto"` (or `"on"`). The model's chat template will be applied to the prompts. This is the default for downstream tasks like `mmlu`, `boolq`, and `truthfulqa`. See [Chat Templates](../advanced/chat-templates.md).

---

## Evaluation

**What is a "good" circuit quality score?**

A patching score (P1) ≥ 0.75 is generally considered a strong circuit. GPT-2 IOI with the default `eap-ig` reaches ~0.9 in our audit (see [Audit Results](../trust/results.md)). Below 0.60, the circuit is not faithfully explaining the behavior. See [Causal Patching](../evaluation/causal-patching.md) for interpretation guidance.

**Why is Pillar 6 (Generalization) marked preliminary?**

Pillar 6 measures whether a circuit transfers to related tasks. The implementation is complete but has not been validated at scale across multiple model families and task pairs. The score may be unreliable. It will be promoted to full status in v1.1.

**Can I run only a subset of pillars?**

Yes. `ck.faithfulness(model, circuit, "ioi", pillars=["patching", "ablation"])` or `run_full_faithfulness(..., pillars=["patching", "ablation", "baselines"])`. The fast subset (P1 + P2 + P5) takes ~2 minutes on GPT-2.

---

## Applications

**Why doesn't `applications.selective_finetune` exist?**

`selective_finetune` is intentionally NOT exported from `applications.__all__`. Access it via `ck.selective_finetune()` or `Pipeline.selective_finetune()`. This is by design to prevent direct import confusion.

**Can I prune and then quantize?**

Yes. The typical workflow is `discover → prune → quantize → export`. Pruning first reduces the model size; quantization then applies mixed precision to the remaining components.

**Which model families support pruning?**

Production: `llama`, `qwen`, `gemma`. Ready: `mistral`, `phi`, `falcon`, `gpt2`. See [Architecture Registry](../advanced/architecture-registry.md).

---

## Performance

**How do I reduce memory usage for large models?**

Set `model.precision="bfloat16"`, `ig_steps=3`, `batch_size=1`, and lower `n_examples`. Combined, these reduce peak VRAM by about 80%. There is no `memory_efficient` or `use_half_precision_activations` config key — those are not read anywhere in the discovery backend. See [Memory Optimization](../advanced/memory-optimization.md).

**How long does discovery take?**

| Model | Algorithm | n_examples | Time |
|-------|-----------|------------|------|
| GPT-2 | EAP-IG | 128 | ~2 min |
| Llama-1B | EAP-IG | 128 | ~8 min |
| Gemma-4B | EAP-IG | 64 | ~20 min |

---

## Bugs and Support

**How do I report a bug?**

Open a GitHub issue at [github.com/Lexsi-Labs/circuitkit/issues](https://github.com/Lexsi-Labs/circuitkit/issues) with a minimal reproduction case, your CircuitKit version (`circuitkit.__version__`), and the full error traceback.

**How do I request a feature?**

Open a GitHub issue with the label `enhancement`. See [Roadmap](roadmap.md) for planned features.
