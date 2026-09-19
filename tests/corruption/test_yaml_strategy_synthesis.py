"""
A YAML task with only ``corruption.strategy`` set (no strategy config, no
corrupted columns) must synthesize a usable corrupt half from a clean-only
dataset. Exercises the same per-example path GenericTaskSpec uses.
"""

import pytest

from circuitkit.tasks.yaml_loader import YAMLTaskLoader

pytest.importorskip("spacy")
try:
    import spacy

    spacy.load("en_core_web_sm")
except Exception:  # pragma: no cover - model not installed
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)

CAPITALS = [
    ("France", "Paris"),
    ("Italy", "Rome"),
    ("Japan", "Tokyo"),
    ("Egypt", "Cairo"),
    ("Canada", "Ottawa"),
    ("Spain", "Madrid"),
]


def _task(tmp_path, strategy):
    csv = tmp_path / "clean.csv"
    csv.write_text(
        "question,answer\n" + "".join(f"The capital of {c} is,{a}\n" for c, a in CAPITALS)
    )
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        f"name: capitals_{strategy}\n"
        f"source:\n  type: csv\n  path: {csv}\n"
        "schema:\n  prompt: question\n  answer: answer\n"
        f"corruption:\n  strategy: {strategy}\n"
        "metric: logit_diff\n"
    )
    return YAMLTaskLoader.load(task_yaml)


@pytest.mark.parametrize("strategy", ["entity_swap", "token_swap"])
def test_strategy_synthesizes_counterfactual_pairs(tmp_path, strategy):
    task = _task(tmp_path, strategy)
    clean = [{"prompt": f"The capital of {c} is", "answer": a} for c, a in CAPITALS]

    corrupted = task._apply_corruptions(clean, {"seed": 42})

    changed = [(c, x) for c, x in zip(clean, corrupted) if c["prompt"] != x["prompt"]]
    assert len(changed) == len(clean), "every prompt should be rewritten"
    answers = dict((f"The capital of {c} is", a) for c, a in CAPITALS)
    for c, x in changed:
        # The rewrite lands on another row, so it carries that row's answer:
        # the pair is answer-changing, not a degenerate (Paris, Paris) pair.
        assert x["answer"] == answers[x["prompt"]]
        assert x["answer"] != c["answer"]


def test_role_swap_flips_an_answer_that_names_a_role():
    import random

    from circuitkit.corruption import RoleSwapCorruption

    clean = {
        "prompt": "The doctor called the nurse. The person who made the call was the",
        "answer": " doctor",
    }
    out = RoleSwapCorruption().corrupt(clean, rng=random.Random(0))
    assert out["prompt"] != clean["prompt"]
    assert out["answer"] == " nurse"


def test_role_swap_runs_through_the_yaml_task_path(tmp_path):
    csv = tmp_path / "roles.csv"
    csv.write_text(
        "question,answer\n"
        '"The doctor called the nurse. The person who made the call was the",doctor\n'
        '"The pilot thanked the judge. The person who gave the thanks was the",pilot\n'
    )
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        f"name: roles\nsource:\n  type: csv\n  path: {csv}\n"
        "schema:\n  prompt: question\n  answer: answer\n"
        "corruption:\n  strategy: role_swap\nmetric: logit_diff\n"
    )
    task = YAMLTaskLoader.load(task_yaml)
    clean = [
        {"prompt": "The doctor called the nurse. The person who made the call was the", "answer": "doctor"},
        {"prompt": "The pilot thanked the judge. The person who gave the thanks was the", "answer": "pilot"},
    ]
    corrupted = task._apply_corruptions(clean, {"seed": 42})
    assert [x["answer"] for x in corrupted] == ["nurse", "judge"]
