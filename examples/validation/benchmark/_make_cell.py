"""Helper to render one benchmark cell file from a (script_name, model, task) tuple.

Used to generate the per-task benchmark scripts without copy/paste drift.
Run as: python _make_cell.py
"""
from __future__ import annotations
from pathlib import Path

TEMPLATE = '''"""Benchmark: all 9 algorithms on {model} + {task}.

{description}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_common import (  # noqa: E402
    ALL_ALGOS, run_benchmark_cell, render_summary, make_results_dir, write_status,
)

SCRIPT_NAME = "{script_name}"
MODEL = "{model}"
TASK = "{task}"
NEEDS_CUSTOM_REGISTRATION = {needs_custom}


def main() -> int:
    out_dir = make_results_dir(SCRIPT_NAME)
    if NEEDS_CUSTOM_REGISTRATION:
        from _custom_tasks import register_all
        register_all()
    rows = []

    for algo in ALL_ALGOS:
        if algo == "eap-exact":
            num_examples = 8
        elif algo in ("atp-gd", "eap-gp"):
            num_examples = 12
        else:
            num_examples = {default_num_examples}

        row = run_benchmark_cell(
            algorithm=algo, model=MODEL, task=TASK,
            num_examples=num_examples, batch_size=1, ig_steps=3,
            target_sparsity=0.1, scope="heads",
            ibcircuit_epochs=200,
        )
        rows.append(row)
        msg = (
            f"{{algo:25s}}  "
            f"discover={{row.get('discovery_seconds', '?'):>6}}s  "
            f"eval={{row.get('eval_seconds', '?'):>6}}s  "
            f"patch={{row.get('patching')}}  "
            f"abl={{row.get('ablation')}}"
        )
        if "error" in row:
            msg += f"  ERR: {{row['error'][:80]}}"
        print(msg, flush=True)

    md = render_summary(rows)
    (out_dir / "table.md").write_text(md)
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, default=str))

    n_ok = sum(1 for r in rows if "error" not in r)
    write_status(out_dir, {{
        "script": SCRIPT_NAME,
        "module": "benchmark[{model_short}-{task}]",
        "input": {{"model": MODEL, "task": TASK, "n_algos": len(ALL_ALGOS)}},
        "output": {{
            "table_md": str(out_dir / "table.md"),
            "rows_json": str(out_dir / "rows.json"),
        }},
        "metrics": {{"n_algos_ok": n_ok, "n_algos_total": len(ALL_ALGOS)}},
        "status": "WORKING" if n_ok == len(ALL_ALGOS) else "NEEDS-FIX",
    }})

    print(f"\\n{{MODEL}}/{{TASK}}: {{n_ok}}/{{len(ALL_ALGOS)}} algorithms succeeded")
    print(md)
    return 0 if n_ok == len(ALL_ALGOS) else 1


if __name__ == "__main__":
    sys.exit(main())
'''


CELLS = [
    # 7-tuple: (script_name, model, task, description, default_num_examples, needs_custom_registration)
    # Built-in TaskSpecs (no custom registration needed)
    ("07_gpt2_gender_bias",     "gpt2", "gender_bias",
     "Built-in TaskSpec for gender-coreference bias detection (paired counterfactual).",
     32, False),
    ("08_gpt2_hypernymy",       "gpt2", "hypernymy",
     "Built-in TaskSpec for hypernym vs hyponym word relations.",
     32, False),
    ("09_gpt2_sva",             "gpt2", "sva",
     "Built-in TaskSpec for subject-verb agreement (Goldberg/Linzen-style).",
     32, False),
    ("10_gpt2_capital_country", "gpt2", "capital_country",
     "Built-in TaskSpec for capital-country factual recall pairs.",
     32, False),
    ("11_gpt2_boolq",           "gpt2", "boolq",
     "Built-in TaskSpec for BoolQ yes/no questions (real HF data).",
     24, False),
    ("12_gpt2_wmdp",            "gpt2", "wmdp",
     "Built-in TaskSpec for WMDP dangerous-knowledge proxy (real HF data).",
     16, False),
    # Custom HF-dataset TaskSpecs (registered via _custom_tasks.py)
    ("13_gpt2_arc_easy",        "gpt2", "arc_easy",
     "ARC-Easy MCQ (allenai/ai2_arc) via NormalizedTaskSpec + MCQChoiceSwap.",
     24, True),
    ("14_gpt2_arc_challenge",   "gpt2", "arc_challenge",
     "ARC-Challenge MCQ (allenai/ai2_arc) via NormalizedTaskSpec + MCQChoiceSwap.",
     24, True),
    ("15_gpt2_hellaswag",       "gpt2", "hellaswag",
     "HellaSwag commonsense MCQ (Rowan/hellaswag) via NormalizedTaskSpec.",
     24, True),
    ("16_gpt2_crows_pairs",     "gpt2", "crows_pairs",
     "CrowS-Pairs bias dataset (nyu-mll) via NormalizedTaskSpec (natively paired).",
     24, True),
    ("17_gpt2_tofu",            "gpt2", "tofu",
     "TOFU forget/retain unlearning dataset (locuslab/TOFU) via NormalizedTaskSpec.",
     24, True),
    ("18_gpt2_gsm8k",           "gpt2", "gsm8k",
     "GSM8K math word problems (openai/gsm8k) via MathAdapter + FinalAnswerSwap.",
     24, True),
]


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    for script, model, task, desc, n, needs_custom in CELLS:
        path = out_dir / f"{script}.py"
        body = TEMPLATE.format(
            script_name=script, model=model, task=task,
            description=desc,
            default_num_examples=n,
            model_short=model.split("/")[-1],
            needs_custom=needs_custom,
        )
        path.write_text(body)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
