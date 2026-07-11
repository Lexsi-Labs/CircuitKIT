# License

CircuitKit is released under the **Lexsi Labs Source Available License (LSAL) v1.1**, a source-available (not open-source) license. It grants free use for research, evaluation, education, and audit; commercial use requires a separate license from Lexsi Labs; and responsible-use conditions apply to safety-relevant behaviors. The full terms are in [LICENSE.md](https://github.com/Lexsi-Labs/circuitkit/blob/main/LICENSE.md).

In short:

- **Free** for noncommercial research, evaluation, education, and auditing.
- **No commercial use** (SaaS, hosted, embedded, or paid support/consulting) without a separate commercial license.
- **Responsible use:** do not use CircuitKit to locate, remove, or weaken the safety behaviors of a model for deployment, and re-evaluate any intervention-exported checkpoint's safety before deploying it.
- **© 2026 Lithasa Technologies Pvt. Ltd.** Contact **support@lexsi.ai** for commercial licensing.

---

## Third-Party Licenses

CircuitKit depends on and integrates with several open-source projects:

| Library | License | Usage |
|---------|---------|-------|
| [TransformerLens](https://github.com/neelnanda-io/TransformerLens) | MIT | Circuit discovery backend |
| [PyTorch](https://pytorch.org) | BSD-3-Clause | Tensor computation |
| [Transformers (HuggingFace)](https://github.com/huggingface/transformers) | Apache 2.0 | Model loading and checkpoint export |
| [PEFT (HuggingFace)](https://github.com/huggingface/peft) | Apache 2.0 | LoRA fine-tuning |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT | Downstream benchmarking |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | MIT | ROME baseline in validation examples |
| [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 | Optional fast evaluation backend |
| [MkDocs](https://www.mkdocs.org) | BSD-2-Clause | Documentation |
| [Click](https://click.palletsprojects.com) | BSD-3-Clause | CLI framework |
| [Rich](https://github.com/Textualize/rich) | MIT | CLI output formatting |

See each library's repository for their full license text.

---

## Citation

If you use CircuitKit in academic work, please cite:

```bibtex
@software{circuitkit2026,
  title   = {CircuitKit: Circuit Discovery, Evaluation, and Application Toolkit
             for Mechanistic Interpretability},
  author  = {Seth, Pratinav and Gosalia, Hem and Kasliwal, Aditya
             and Sankarapu, Vinay Kumar},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Lexsi-Labs/circuitkit},
}
```
