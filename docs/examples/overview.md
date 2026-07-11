# Examples Overview

CircuitKit ships three sets of runnable examples: **Python scripts** in `examples/` (CI-testable, CPU-friendly tutorials on GPT-2), **Jupyter notebooks** in `examples/notebooks/` (Colab-ready, with GPU tracks on Gemma / Qwen / Llama), and **[case studies](case-studies.md)** in `examples/case-studies/` (domain-framed end-to-end walkthroughs — compliance, safety steering, unlearning, edge deployment — on domain-appropriate models).

---

## Quick Navigation

The table below maps common goals to the core workflow scripts. The `examples/` directory ships **13** numbered scripts in total (`01`–`13`); the advanced ones (`09`–`13`: load-and-reuse, steering, knowledge editing, custom corruption, transfer matrix) are listed in the [`examples/README.md`](https://github.com/Lexsi-Labs/circuitkit/tree/main/examples).

| Goal | Resource |
|------|---------|
| First circuit in 5 minutes | [`examples/01-quickstart.py`](scripts.md) or [Notebook 01](notebooks.md) |
| Discover via Python API | [`examples/02-discover-python-api.py`](scripts.md) |
| Discover via CLI | [`examples/03-discover-cli.sh`](scripts.md) |
| Custom data / YAML task | [`examples/08-custom-data.py`](scripts.md) or [Notebook 03](notebooks.md) |
| 6-pillar evaluation | [`examples/05-evaluate-faithfulness.py`](scripts.md) or [Notebook 04](notebooks.md) |
| Prune / quantize / finetune | [`examples/06-applications.py`](scripts.md) or [Notebook 06](notebooks.md) |
| Pipeline chaining | [`examples/07-pipeline.py`](scripts.md) or [Notebook 01](notebooks.md) |
| Algorithm comparison | [Notebook 02](notebooks.md) |
| Visualization | [Notebook 05](notebooks.md) |
| CLI + YAML workflow | [Notebook 07](notebooks.md) or [`examples/03-discover-cli.sh`](scripts.md) |
| Real-world scenario (compliance, safety, unlearning, edge) | [Case Studies 14–24](case-studies.md) |
| Safety steering / jailbreak defense | [Case Study 18 or 23](case-studies.md) |

---

## Scripts vs Notebooks

| | Scripts (`examples/`) | Notebooks (`examples/notebooks/`) |
|--|----------------------|--------------------------|
| Runtime | Python process | Colab / Jupyter |
| GPU required | No (GPT-2) | Some notebooks (T4+) |
| CI testable | Yes | No |
| Output | Terminal | Inline cells |
| Best for | Automation, scripting | Interactive exploration |

---

## Detailed Pages

- [Python Scripts](scripts.md) — the example scripts with descriptions and usage
- [Notebooks](notebooks.md) — all 9 Colab notebooks with badges and runtime estimates
- [Case Studies](case-studies.md) — 11 domain-framed end-to-end walkthroughs (scripts 14–20, 24; notebooks 21–23)
