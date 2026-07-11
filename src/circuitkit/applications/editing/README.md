# editing

Knowledge editing for transformer models via ROME, MEMIT, and circuit-guided methods (usable via `circuitkit.applications.editing`, not part of the flat public API).

## Key modules

- `rome_wrapper.py` — `RomeHandler`, `RomeWrapper`, `RomeEditVectors`: ROME (Rank-One Model Editing), editing facts via rank-one perturbations to MLP weight matrices.
- `memit_wrapper.py` — `MemitHandler`, `MemitBatchEdit`: MEMIT, extending ROME to edit many facts simultaneously in one update.
- `knowledge_editing.py` — `CircuitKnowledgeEditor`, `EditResult`, `UnlearningReport`: circuit-aware knowledge editing that integrates ROME/MEMIT with circuit discovery.
- `knowledge_editing_enhanced.py` — `BatchKnowledgeEditor`, `UnlearningVerifier` (results `BatchEditResult`, `LeakageReport`): batch editing, enhanced unlearning verification, and edit-interference detection.
- `circuit_guided_editing.py` — `CircuitGuidedEditor`, `CircuitVerificationResult`: end-to-end pipeline that targets edits via circuits and verifies edits don't break them.
- `cake.py` — `CaKEEditor` (edit `CaKEEdit`): Circuit-Aware Knowledge Editing, restricting updates to MLP layers strong in the discovered circuit.
- `mcircke.py` — `MCircKEEditor`, `MultiHopEdit` (`Hop`): multi-hop circuit-guided editing that decomposes queries and applies ROME per hop.
- `fine_tune_editing.py` — `FineTuneEditHandler`, `FineTuneEditResult`: architecture-agnostic gradient fine-tuning fallback when ROME-style editing is too architecture-dependent.

## Public API / entry points

`__all__`: `CircuitKnowledgeEditor`, `BatchKnowledgeEditor`, `RomeHandler`, `RomeWrapper`, `MemitHandler`, `CircuitGuidedEditor`, `MCircKEEditor`, `CaKEEditor`, `FineTuneEditHandler`, `EditResult`, `MultiHopEdit`, `CircuitVerificationResult`, `FineTuneEditResult`, `UnlearningReport`, `UnlearningVerifier`, `MemitBatchEdit`, `RomeEditVectors`.

## How it fits

One of the intervention applications. It combines model-editing algorithms with circuit discovery to target and verify factual edits.
