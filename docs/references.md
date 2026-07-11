# References

This page collects the key papers and resources referenced throughout the CircuitKit documentation.

> For a machine-checkable map of every `arXiv:` reference cited **inside the source code**
> (with Hugging Face Papers links), see the [Citation Map](citation-map.md). Vendored
> third-party code and its licenses are recorded in
> [THIRD_PARTY_LICENSES.md](https://github.com/Lexsi-Labs/circuitkit/blob/main/THIRD_PARTY_LICENSES.md).

## Circuit Discovery

- **Conmy, A., Nanda, N., Bloom, J., & others.** (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability." *NeurIPS 2023*. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)
- **Zhang, F. & Nanda, N.** (2023). "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods." *ICLR 2024*. [arXiv:2309.16042](https://arxiv.org/abs/2309.16042)
- **Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J.** (2022). "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small." *ICLR 2023*. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593)

## Mechanistic Interpretability Foundations

- **Elhage, N., Nanda, N., Olsson, C., & others.** (2021). "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*. [Link](https://transformer-circuits.pub/2021/framework/index.html)
- **Olsson, C., Elhage, N., Nanda, N., & others.** (2022). "In-context Learning and Induction Heads." *Transformer Circuits Thread*. [Link](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- **Nanda, N., Bloom, J., & others.** (2023). "TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models." [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- **Marks, S., Rager, C., Michaud, E. J., & others.** (2024). "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." [arXiv:2403.19647](https://arxiv.org/abs/2403.19647)

## Faithfulness Evaluation

- **Zhang, F. & Nanda, N.** (2023). "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods." *ICLR 2024*. [arXiv:2309.16042](https://arxiv.org/abs/2309.16042)
- **Miller, J., Chughtai, B. & Saunders, W.** (2024). "Transformer Circuit Faithfulness Metrics are not Robust." [arXiv:2407.08734](https://arxiv.org/abs/2407.08734) — faithfulness measurements are highly sensitive to ablation methodology.
- **Hanna, M., Pezzelle, S. & Belinkov, Y.** (2024). "Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms." (EAP-IG) [arXiv:2403.17806](https://arxiv.org/abs/2403.17806)
- **Seth, P.\*, Gosalia, H.\*, Kasliwal, A.\*, & Sankarapu, V. K.** (2026). "Faithfulness Is Not Actionability: An Extrinsic Audit of Circuit Discovery for Model Compression." *EMNLP Findings 2026*. <br> <sub>\* Equal co-first authorship. Corresponding author: Pratinav Seth (pratinav.seth@lexsi.ai).</sub>

## Software

- **CircuitKit** — Seth, P.\*, Gosalia, H.\*, Kasliwal, A.\*, & Sankarapu, V. K. (2026). "CircuitKit: Circuit Discovery, Evaluation, and Application Toolkit for Mechanistic Interpretability." [GitHub](https://github.com/Lexsi-Labs/circuitkit) <br> <sub>\* Equal co-first authorship. Corresponding author: Pratinav Seth (pratinav.seth@lexsi.ai).</sub>
- **TransformerLens** — Nanda, N., & Bloom, J. (2022). [GitHub](https://github.com/TransformerLensOrg/TransformerLens)
- **lm-evaluation-harness** — EleutherAI. [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- **PyTorch** — Paszke, A., & others. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS 2019*.
