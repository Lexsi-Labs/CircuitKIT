"""Regression tests for the (fixed) capital_country circuit-discovery task.

The capital_country task previously shipped with two fatal bugs that made
circuit discovery meaningless:

  1. ``incorrect_idx`` was set equal to ``correct_idx`` on every CSV row, so
     the discovery metric ``logit(correct) - logit(incorrect)`` was identically
     0 with a zero gradient -> EAP / EAP-IG produced all-zero, all-equal scores.
  2. The prompt placed the answer entity mid-sentence
     ("Vienna is the capital of Austria"), but the metric scores the position
     ``input_length - 1`` (whose prediction is the *next* token), so the scored
     position had nothing to do with the answer.

The redesigned task (mirroring boolq.py / ioi.py) ends the prompt right before
the answer -- ``"The capital of Austria is"`` -> the model predicts
``" Vienna"`` -- and pairs each clean prompt with a corrupt prompt that flips
the country to a length-matched different country, so the corrupt prompt's
capital is a genuinely different ``incorrect_idx``.

These tests pin that contract WITHOUT requiring a model where possible, and
verify end-to-end non-degeneracy with gpt2 where a model is needed.
"""

import math

import pytest
import torch

from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
from circuitkit.tasks.builtins.capital_country import CapitalCountryTaskSpec
from circuitkit.tasks.registry import get_task, list_tasks


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _bootstrap_builtin_tasks()


@pytest.fixture(scope="module")
def gpt2():
    """Shared gpt2 HookedTransformer; skip the module if it cannot load."""
    try:
        from transformer_lens import HookedTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HookedTransformer.from_pretrained("gpt2", device=device)
    except Exception as e:  # pragma: no cover - offline CI
        pytest.skip(f"gpt2 unavailable: {e}")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def test_capital_country_is_registered():
    assert "capital_country" in list_tasks()
    spec = get_task("capital_country")
    assert spec.name == "capital_country"


# --------------------------------------------------------------------------
# Metric: differentiable, non-degenerate logit-difference
# --------------------------------------------------------------------------
def test_logit_diff_metric_is_differentiable_and_nondegenerate():
    """The metric must produce a real, non-zero gradient w.r.t. the logits.

    With a genuinely different incorrect token per row, gathering
    logit(correct) - logit(incorrect) yields a gradient at exactly the two
    distinct token positions per row -- never the all-cancelling 0 the buggy
    (correct_idx == incorrect_idx) version produced.
    """
    spec = CapitalCountryTaskSpec()
    metric = spec.metric_fn("logit_diff")

    batch, n_pos, vocab = 4, 6, 200
    logits = torch.randn(batch, n_pos, vocab, requires_grad=True)
    input_length = torch.full((batch,), n_pos, dtype=torch.long)
    # correct != incorrect on every row.
    labels = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80]])

    loss = metric(logits, None, input_length, labels)
    loss.backward()

    assert logits.grad is not None
    grad = logits.grad
    assert grad.norm().item() > 0, "gradient must be non-zero"
    # exactly 2 distinct gathered tokens per row -> 2 * batch nonzero entries.
    assert (grad != 0).sum().item() == 2 * batch


def test_logit_diff_degenerate_labels_kill_the_gradient():
    """Sanity check pinning the ROOT CAUSE: when incorrect_idx == correct_idx
    the metric is identically 0 with a zero gradient. The redesigned data
    generator must therefore never emit such rows (see CSV invariant test)."""
    spec = CapitalCountryTaskSpec()
    metric = spec.metric_fn("logit_diff")

    batch, n_pos, vocab = 4, 6, 200
    logits = torch.randn(batch, n_pos, vocab, requires_grad=True)
    input_length = torch.full((batch,), n_pos, dtype=torch.long)
    # The OLD BUG: incorrect == correct on every row.
    labels = torch.tensor([[10, 10], [30, 30], [50, 50], [70, 70]])

    loss = metric(logits, None, input_length, labels)
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad.norm().item() == 0.0


def test_logit_diff_scores_the_prediction_position():
    """The metric reads ``input_length - 1`` -- the redesigned prompt ends
    right before the answer so that position predicts the answer token."""
    spec = CapitalCountryTaskSpec()
    metric = spec.metric_fn("logit_diff")

    batch, n_pos, vocab = 3, 8, 50
    logits = torch.zeros(batch, n_pos, vocab)
    input_length = torch.tensor([8, 8, 8])
    labels = torch.tensor([[1, 2], [3, 4], [5, 6]])
    # Make the correct token dominate ONLY at the prediction position.
    for b, (c, _) in enumerate(labels.tolist()):
        logits[b, n_pos - 1, c] = 5.0

    per_row = metric.func(logits, None, input_length, labels, mean=False, loss=False)
    assert torch.all(per_row > 0), "correct token must win at input_length-1"


# --------------------------------------------------------------------------
# Data generation contract (requires gpt2)
# --------------------------------------------------------------------------
def test_generated_csv_invariants(gpt2):
    """The generated clean/corrupt pairs must satisfy the EAP contract:
    correct_idx != incorrect_idx, single-token answers, length-aligned."""
    from circuitkit.data.task_data.tasks.capital_country.utils import generate_capital_country_data

    df = generate_capital_country_data(n_samples=40, seed=42, model=gpt2)
    assert len(df) > 0

    # Invariant 1: non-degenerate logit-diff (the root-cause fix).
    assert (df["correct_idx"] != df["incorrect_idx"]).all()

    # Invariant 2: clean/corrupt token-length aligned.
    for clean, corrupt in zip(df["clean"], df["corrupted"]):
        cl = gpt2.to_tokens(clean, prepend_bos=False).shape[1]
        co = gpt2.to_tokens(corrupt, prepend_bos=False).shape[1]
        assert cl == co

    # Invariant 3: prompt ends right before the answer (no trailing answer).
    for clean in df["clean"]:
        assert clean.endswith(" is")

    # Invariant 4: single-token answers.
    for idx in df["correct_idx"]:
        assert isinstance(int(idx), int)


def test_base_accuracy_above_chance(gpt2):
    """gpt2 must favour the correct capital over the corrupt one well above
    chance -- the logit-diff signal the circuit discovery optimizes."""
    from circuitkit.data.task_data.tasks.capital_country.utils import generate_capital_country_data

    df = generate_capital_country_data(n_samples=30, seed=42, model=gpt2)
    wins = 0
    for _, row in df.iterrows():
        toks = gpt2.to_tokens(row["clean"])
        with torch.no_grad():
            logits = gpt2(toks)[0, -1]
        ci, ii = int(row["correct_idx"]), int(row["incorrect_idx"])
        wins += int(logits[ci].item() > logits[ii].item())
    frac = wins / len(df)
    # Chance = 0.5; the model should be clearly above it.
    assert frac > 0.7, f"base accuracy {frac:.3f} not above chance"


# --------------------------------------------------------------------------
# End-to-end: EAP / EAP-IG discovery must be NON-DEGENERATE
# --------------------------------------------------------------------------
@pytest.mark.parametrize("algorithm", ["eap", "eap-ig"])
def test_discovery_is_nondegenerate(gpt2, tmp_path, algorithm):
    """Run real circuit discovery and confirm scores span a real range and
    are not the all-zero / all-equal output the buggy task produced."""
    from circuitkit.api import discover_circuit

    discovery = {
        "algorithm": algorithm,
        "task": "capital_country",
        "level": "node",
        "batch_size": 4,
        "model_name": "gpt2",
        "seed": 42,
        "cache_dir": str(tmp_path / "cache"),
        "data_params": {"n_samples": 12, "seed": 42},
    }
    if algorithm == "eap-ig":
        discovery["ig_steps"] = 5
        discovery["method"] = "EAP-IG-inputs"

    cfg = {
        "model": {"name": "gpt2", "precision": "float32"},
        "discovery": discovery,
        "pruning": {"target_sparsity": 0.15, "scope": "both"},
        "output_path": str(tmp_path / f"cc_{algorithm}.pt"),
    }
    discover_circuit(cfg)

    scores_path = tmp_path / f"cc_{algorithm}_scores.pt"
    assert scores_path.exists()
    payload = torch.load(scores_path, map_location="cpu")
    node_scores = payload["node_scores"]
    values = list(node_scores.values())

    assert len(values) > 1
    # Non-degenerate: not all zero, not all equal, real spread.
    assert any(v != 0 for v in values), "all-zero scores -> degenerate"
    assert not math.isclose(min(values), max(values)), "all-equal scores -> degenerate"
    assert max(values) - min(values) > 1e-3, "score range collapsed"
    assert max(values) > 0
