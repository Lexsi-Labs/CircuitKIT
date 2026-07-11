# Release Notes

---

## v1.0.0 — 2026-05-16

First stable release. This release closes a correctness-hardening cycle: an audit found and fixed 10+ serious bugs in code that had previously been marked "done." The public API is stable and the Stable-tier discovery path is validated. Experimental- and research-tier backends remain explicitly marked as such.

### Highlights

**Stable discovery path validated**

- EAP and EAP-IG are validated across GPT-2, Llama-3.2-1B/3B, Gemma-2-2B, Gemma-3-4B, and Qwen2.5-1.5B
- EAP-IG-Activations and EAP-Clean-Corrupted remain Research tier — validated only on GPT-2 IOI
- 13 discovery algorithms across 4 backends (EAP, ACDC, IBCircuit, CD-T) — but only 2 are validated at production scale (`eap`, `eap-ig`); `acdc` and `ibcircuit` are Experimental (GPT-2 scale, `ibcircuit` OOMs above ~3B) and the other 9 are Research (GPT-2 IOI only)
- 14 registered selectors; 16 built-in tasks
- 6-pillar faithfulness framework with Pillar 6 marked preliminary

**bf16 / GQA correctness**

Discovery backends (PEAP, CD-T, ACDC) and all applications (knowledge editing, `CircuitWeightSteering`, circuit-localized fine-tuning) now run correctly on bf16 models and grouped-query-attention architectures.

**Chat-template handling**

Per-task `chat_template_mode` (`auto`/`on`/`off`), BOS consistency across all components, auto chat-model detection, and discovery freezing the resolved policy into artifact metadata.

**Checkpoint export protocol**

Intervened (pruned/quantized) checkpoints are exported as standard reloadable HuggingFace checkpoints and are vLLM-evaluable via lm-eval. 8 checkpoint export/eval correctness bugs fixed.

**API surfaces**

- Flat typed API (`ck.discover`, `ck.prune`, `ck.quantize`, `ck.faithfulness`, `ck.export_checkpoint`, `ck.benchmark`, `ck.load_scores`, `ck.selective_finetune`, `ck.visualize_circuit`)
- Pipeline class with method chaining and `from_artifact` / `from_scores` constructors
- Unified `scope` / `protect_layers` component selection across all applications

**Removed**

- `lrp` pruning selector removed (orphan — replaced by `taylor` and `wanda`)
- Old Sphinx documentation replaced by MkDocs

### Known Issues

- Pillar 6 (Generalization) is implemented but not validated at scale — treat scores as preliminary
- IBCircuit OOM above ~3B parameters on single-GPU
- ROME knowledge editing: MLP-only target selection (attention target selection is not yet implemented)

---

For the full bug-fix history, see `CHANGELOG.md` in the repository root.
