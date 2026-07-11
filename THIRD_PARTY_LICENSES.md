# Third-Party Licenses

CircuitKit is distributed under the Lexsi Labs Source Available License (LSAL)
v1.1 (see [LICENSE.md](LICENSE.md)). It
incorporates or adapts source code from the third-party projects listed below.
Each remains under its own license, reproduced or linked here as required.

CircuitKit also depends on many packages at runtime (PyTorch, Transformers,
TransformerLens, etc.) that are installed via `pip` and are governed by their
own licenses; only projects whose **source code is vendored or adapted directly
into this repository** are listed here.

---

## 1. Automatic Circuit DisCovery (ACDC)

- **Upstream:** https://github.com/ArthurConmy/Automatic-Circuit-Discovery
- **License:** MIT
- **Copyright:** © 2023 Arthur Conmy, Adrià Garriga-Alonso
- **Used in CircuitKit:** the ACDC search algorithm, task data, and metric
  utilities are adapted from the ACDC repository:
  - `src/circuitkit/backends/acdc/data.py` ← `acdc/data.py`
  - `src/circuitkit/backends/acdc/tasks/ioi_dataset.py` ← `acdc/ioi/ioi_dataset.py`
  - `src/circuitkit/backends/acdc/tasks/docstring_prompts.py` ← `acdc/docstring/prompts.py`
  - `src/circuitkit/backends/acdc/tasks/ioi_utils.py`,
    `tasks/induction_utils.py` ← `acdc/acdc_utils.py` and related task helpers
  - `src/circuitkit/backends/acdc/prune.py`, `prune_algos/ACDC.py`,
    `prune_algos/mask_gradient.py`, `utils/graph_utils.py`,
    `utils/tensor_ops.py` ← the ACDC edge-pruning search and its graph/tensor
    utilities (the hard-concrete sampling in `tensor_ops.py` originates with
    Subnetwork Probing; see section 5)
  - `src/circuitkit/data/task_data/core/` (`acdc_utils.py`,
    `TLACDCExperiment.py`, `TLACDCCorrespondence.py`, `TLACDCEdge.py`,
    `TLACDCInterpNode.py`) ← `acdc/TLACDC*.py` and `acdc/acdc_utils.py`
- The IOI task data originates from Wang et al. (2022), "Interpretability in the
  Wild" (arXiv:2211.00593), redistributed via ACDC.

```
MIT License

Copyright (c) 2023 Arthur Conmy, Adrià Garriga-Alonso

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. LLM-Pruner

- **Upstream:** https://github.com/horseee/LLM-Pruner
- **License:** Apache License 2.0
- **Used in CircuitKit:** `src/circuitkit/applications/pruning/finetune_utils.py`
  — the Prompter / tokenization logic for post-pruning LoRA recovery is adapted
  from LLM-Pruner's `post_training.py` to stay compatible with the Alpaca
  instruction format. Modifications have been made to integrate with CircuitKit's
  pruning artifacts.
- A full copy of the Apache License 2.0 is available at
  https://www.apache.org/licenses/LICENSE-2.0. Per Section 4, the above notice
  records the origin and the fact that changes were made.

---

## 3. CD-T (Contextual Decomposition for Transformers)

- **Upstream:** https://github.com/adelaidehsu/CD_Circuit (CD-T, ICLR 2025)
- **License:** none published — the upstream repository has no `LICENSE` file at
  the time of vendoring (all rights reserved by default). **Flagged for legal
  review before public release** (see note below).
- **Used in CircuitKit:** the CD-T backend vendors the reference
  implementation under `src/circuitkit/backends/cdt/pyfunctions/`
  (`cdt_core.py`, `cdt_basic.py`, `cdt_ablations.py`, `cdt_from_source_nodes.py`,
  `cdt_source_to_target.py`, `toy_model.py`, `general.py`, and the two files
  attributed separately in sections 4 and 6); `backends/cdt/propagation.py` and
  `backends/cdt/adapter.py` are CircuitKit's TransformerLens-native port of that
  method.

---

## 4. Easy-Transformer / EasyTransformer (IOI dataset)

- **Upstream:** https://github.com/redwoodresearch/Easy-Transformer
- **License:** MIT — Copyright (c) 2022 neelnanda-io (full text reproduced in the
  file headers below)
- **Used in CircuitKit:**
  - `src/circuitkit/backends/cdt/pyfunctions/ioi_dataset.py` — copied verbatim
    from `easy_transformer/ioi_dataset.py`; the MIT license text is retained in
    the file header.
  - `src/circuitkit/data/task_data/tasks/ioi/ioi_dataset.py` — a very slightly
    edited version of the same file, obtained via the Redwood `mlab2`
    (`arthur/induction`) branch.

---

## 5. Subnetwork Probing

- **Upstream:** https://github.com/stevenxcao/subnetwork-probing (Cao et al.,
  "Low-Complexity Probing via Finding Subnetworks")
- **License:** none published — the upstream repository has no `LICENSE` file at
  the time of vendoring. **Flagged for legal review before public release.**
- **Used in CircuitKit:** the hard-concrete sampling routine in
  `src/circuitkit/backends/acdc/utils/tensor_ops.py` (`sample_hard_concrete` and
  the `left/right/temp` constants) is copied from the Subnetwork Probing
  reference code, as noted in the file header.

---

## 6. ARENA 3.0 (IOI faithfulness ablations)

- **Upstream:** https://github.com/callummcdougall/ARENA_3.0 (ARENA 3.0 IOI
  material, by Callum McDougall)
- **License:** none published — the upstream repository has no `LICENSE` file at
  the time of vendoring. The source file records that we are not aware of any
  formal attribution or license requirement and invites correction.
  **Flagged for legal review before public release.**
- **Used in CircuitKit:**
  `src/circuitkit/backends/cdt/pyfunctions/faithfulness_ablations.py` — taken
  from the ARENA 3.0 notebook on the IOI task.

---

## 7. IBCircuit

- **Upstream:** https://github.com/ivanniu/IBCircuit (IBCircuit, Niu et al.)
- **License:** none published — the upstream repository has no `LICENSE` file at
  the time of vendoring. **Flagged for legal review before public release.**
- **Used in CircuitKit:** `src/circuitkit/backends/ibcircuit/` reimplements the
  Information Bottleneck circuit-discovery method with reference to the upstream
  repository (see `ib_noise.py`).

---

> **Note for release review.** Sections 3, 5, 6, and 7 vendor or adapt code from
> repositories that publish **no license file**. Absent an explicit grant, such
> code is all-rights-reserved by default. Obtain permission from the upstream
> authors, or replace these components with independently licensed
> implementations, before distributing CircuitKit publicly.

---

## Runtime dependencies (not vendored)

These are imported, not copied, and installed through packaging metadata:

- **TransformerLens** — MIT — https://github.com/TransformerLensOrg/TransformerLens
- **PyTorch** — BSD-3-Clause — https://github.com/pytorch/pytorch
- **Hugging Face Transformers** — Apache-2.0 — https://github.com/huggingface/transformers
- **lm-evaluation-harness** — MIT — https://github.com/EleutherAI/lm-evaluation-harness

See each project's repository for full license text.
