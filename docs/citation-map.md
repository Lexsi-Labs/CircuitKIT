# Citation Map

Every paper referenced by an `arXiv:` identifier **inside the CircuitKit source
tree** (`src/`), mapped to its verified title, authors, arXiv page, and Hugging
Face Papers page. Titles were checked against arXiv — if you add a new `arXiv:`
reference in code, add a row here too.

For the broader documentation bibliography (foundational MI papers, tooling), see
[references.md](references.md).

| arXiv | Title | Authors | Cited by (module) | Links |
|-------|-------|---------|-------------------|-------|
| 2211.00593 | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small | Wang, Variengien, Conmy, Shlegeris, Steinhardt | `backends/acdc/tasks/ioi_dataset.py` | [arXiv](https://arxiv.org/abs/2211.00593) · [HF](https://huggingface.co/papers/2211.00593) |
| 2309.16042 | Towards Best Practices of Activation Patching in Language Models | Zhang, Nanda | `data/corruption/resample.py` | [arXiv](https://arxiv.org/abs/2309.16042) · [HF](https://huggingface.co/papers/2309.16042) |
| 2403.00745 | AtP*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components | Kramár, Lieberum, Shah, Nanda | `backends/eap/attribute_node.py` | [arXiv](https://arxiv.org/abs/2403.00745) · [HF](https://huggingface.co/papers/2403.00745) |
| 2502.04577 | Position-aware Automatic Circuit Discovery | Haklay, Orgad, Bau, Mueller, Belinkov | `backends/eap/attribute_node.py`, `data/dataset_schema.py` | [arXiv](https://arxiv.org/abs/2502.04577) · [HF](https://huggingface.co/papers/2502.04577) |
| 2508.21258 | RelP: Faithful and Efficient Circuit Discovery in Language Models via Relevance Patching | Rezaei Jafari, Eberle, Khakzar, Nanda | `backends/eap/attribute_node.py` | [arXiv](https://arxiv.org/abs/2508.21258) · [HF](https://huggingface.co/papers/2508.21258) |
| 2502.06852 | EAP-GP: Mitigating Saturation Effect in Gradient-based Automated Circuit Identification | Zhang, Dong, Zhang, Yang, Hu, Liu, Zhou, Wang | `backends/eap/attribute_node.py` | [arXiv](https://arxiv.org/abs/2502.06852) · [HF](https://huggingface.co/papers/2502.06852) |
| 2504.07389 | Task-Circuit Quantization: Leveraging Knowledge Localization and Interpretability for Compression | Xiao, Sung, Stengel-Eskin, Bansal | `applications/quantization/selectors/tacq_selector.py` | [arXiv](https://arxiv.org/abs/2504.07389) · [HF](https://huggingface.co/papers/2504.07389) |
| 2604.05876 | Mechanistic Circuit-Based Knowledge Editing in Large Language Models | Zhao, He, Zheng, Chen | `applications/editing/cake.py`, `applications/editing/mcircke.py` | [arXiv](https://arxiv.org/abs/2604.05876) · [HF](https://huggingface.co/papers/2604.05876) |
| 2010.00133 | CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models | Nangia, Vania, Bhalerao, Bowman | `data/adapters/pairwise.py` | [arXiv](https://arxiv.org/abs/2010.00133) · [HF](https://huggingface.co/papers/2010.00133) |
| 2603.23268 | SafeSeek: Universal Attribution of Safety Circuits in Language Models | Yu, Fu, Aloqaily, Zhou, Otoum, Fan, Wang, Guo, Wen | `data/corruption/benign_rewrite.py` | [arXiv](https://arxiv.org/abs/2603.23268) · [HF](https://huggingface.co/papers/2603.23268) |
| 2406.11717 | Refusal in Language Models Is Mediated by a Single Direction | Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda | `data/corruption/benign_rewrite.py` | [arXiv](https://arxiv.org/abs/2406.11717) · [HF](https://huggingface.co/papers/2406.11717) |
| 2312.15710 | Alleviating Hallucinations of Large Language Models through Induced Hallucinations | Zhang, Cui, Bi, Shi | `data/corruption/llm_counterfactual.py` | [arXiv](https://arxiv.org/abs/2312.15710) · [HF](https://huggingface.co/papers/2312.15710) |
| 2404.12010 | ParaFusion: A Large-Scale LLM-Driven English Paraphrase Dataset | Jayawardena, Yapa | `data/corruption/llm_counterfactual.py` | [arXiv](https://arxiv.org/abs/2404.12010) · [HF](https://huggingface.co/papers/2404.12010) |
| 2305.15054 | A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis | Stolfo, Belinkov, Sachan | `data/corruption/operand_swap.py` | [arXiv](https://arxiv.org/abs/2305.15054) · [HF](https://huggingface.co/papers/2305.15054) |
| 2004.12265 | Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias | Vig, Gehrmann, Belinkov, Qian, Nevo, Sakenis, Huang, Singer, Shieber | `data/corruption/profession_swap.py` | [arXiv](https://arxiv.org/abs/2004.12265) · [HF](https://huggingface.co/papers/2004.12265) |
| 2411.16105 | Adaptive Circuit Behavior and Generalization in Mechanistic Interpretability | Nainani, Vaidyanathan, Yeung, Gupta, Jensen | `tasks/builtins/double_io.py`, `data/task_data/tasks/double_io/` | [arXiv](https://arxiv.org/abs/2411.16105) · [HF](https://huggingface.co/papers/2411.16105) |

## References without an arXiv identifier

| Title | Authors / Venue | Cited by (module) | Link |
|-------|-----------------|-------------------|------|
| IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck | Bian, Niu, Yuan, Piao, Wu, Huang, Rong, Xu, Cheng, Li — ICML 2025 | `backends/ibcircuit/`, `selection/ibcircuit_selector.py` | [Code](https://github.com/ivanniu/IBCircuit) |

> **Note:** earlier revisions cited `arXiv:2408.05520` for IBCircuit; that
> identifier actually resolves to an unrelated quantum-physics paper ("On
> stability issues of the HEOM method") and has been corrected to the ICML 2025
> reference above. IBCircuit's noise-injection mechanism builds on the
> information-bottleneck attribution method of Schulz et al. (2020),
> [arXiv:2001.00396](https://arxiv.org/abs/2001.00396).
