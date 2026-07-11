"""Focused tests for the WinoGrande-MC (multiple-choice) circuit-discovery task.

``winogrande_mc`` reformulates each WinoGrande item as an explicit
multiple-choice comprehension question and scores it with a single-token
logit-difference metric on the " A" / " B" answer-letter tokens. Unlike the
cloze ``winogrande`` task it has a real question/answer turn structure, so it
is ``chat_template_mode = "auto"`` and CAN be chat-templated. These tests pin
that contract:

  * task registration / discoverability
  * ``chat_template_mode`` is "auto"
  * ``validate_discovery_config`` accepts eap/node, rejects bad input
  * the single-token logit-diff metric is differentiable and sign-correct
  * the EAP dataset builds token-aligned clean/corrupt pairs

The CSV-generation test hits the HuggingFace ``winogrande`` dataset and is
skipped automatically if it is unavailable (offline CI).
"""

import pytest
import torch

from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
from circuitkit.tasks.builtins.winogrande_mc import (
    WinoGrandeMCTaskSpec,
    _format_body,
    _format_prompt,
)
from circuitkit.tasks.registry import get_task, list_tasks


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _bootstrap_builtin_tasks()


# --------------------------------------------------------------------------
# Registration / discoverability
# --------------------------------------------------------------------------
def test_winogrande_mc_is_registered():
    assert "winogrande_mc" in list_tasks()
    spec = get_task("winogrande_mc")
    assert spec.name == "winogrande_mc"
    assert spec.pair_padding_side == "left"


def test_winogrande_mc_is_distinct_from_winogrande():
    """winogrande_mc is additive -- the cloze winogrande task is untouched."""
    tasks = list_tasks()
    assert "winogrande" in tasks
    assert "winogrande_mc" in tasks
    assert get_task("winogrande").chat_template_mode == "off"


# --------------------------------------------------------------------------
# Chat-template mode: this variant is templatable
# --------------------------------------------------------------------------
def test_chat_template_mode_is_auto():
    spec = WinoGrandeMCTaskSpec()
    assert spec.chat_template_mode == "auto"


# --------------------------------------------------------------------------
# Prompt format
# --------------------------------------------------------------------------
def test_prompt_format_has_mc_structure():
    body = _format_body("the trophy did not fit because _ was large.", "trophy", "suitcase")
    assert "Sentence:" in body
    assert "which word fills the blank?" in body
    assert "A) trophy" in body
    assert "B) suitcase" in body
    # The body is the user turn -- it must NOT include the answer tail.
    assert not body.rstrip().endswith("Answer:")

    prompt = _format_prompt("the trophy did not fit because _ was large.", "trophy", "suitcase")
    assert prompt.rstrip().endswith("Answer:")
    assert prompt.startswith(body)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------
def test_validate_discovery_config_accepts_eap():
    spec = WinoGrandeMCTaskSpec()
    # Should not raise.
    spec.validate_discovery_config({"algorithm": "eap", "level": "node", "batch_size": 8})


def test_validate_discovery_config_rejects_acdc():
    spec = WinoGrandeMCTaskSpec()
    with pytest.raises(ValueError, match="does not support"):
        spec.validate_discovery_config({"algorithm": "acdc", "level": "node"})


def test_validate_discovery_config_rejects_bad_level():
    spec = WinoGrandeMCTaskSpec()
    with pytest.raises(ValueError, match="level"):
        spec.validate_discovery_config({"algorithm": "eap", "level": "edge"})


def test_validate_discovery_config_rejects_bad_batch_size():
    spec = WinoGrandeMCTaskSpec()
    with pytest.raises(ValueError, match="batch_size"):
        spec.validate_discovery_config({"algorithm": "eap", "level": "node", "batch_size": 0})


# --------------------------------------------------------------------------
# Metric: single-token logit difference
# --------------------------------------------------------------------------
def test_logit_diff_metric_is_differentiable():
    """The discovery metric must be a differentiable function of the logits."""
    spec = WinoGrandeMCTaskSpec()
    metric = spec.metric_fn()  # loss=True, mean=True

    batch, n_pos, vocab = 4, 10, 100
    logits = torch.randn(batch, n_pos, vocab, requires_grad=True)
    clean_logits = torch.randn(batch, n_pos, vocab)
    input_length = torch.tensor([n_pos] * batch)
    # labels[:, 0] = correct letter token, labels[:, 1] = incorrect letter token
    labels = torch.tensor([[10, 20], [21, 11], [12, 22], [23, 13]])

    loss = metric(logits, clean_logits, input_length, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_logit_diff_metric_is_sign_correct():
    """logit_diff = logit(correct) - logit(incorrect): positive when the model
    favours the correct letter, negative otherwise. loss=True negates it."""
    spec = WinoGrandeMCTaskSpec()

    batch, n_pos, vocab = 1, 6, 50
    correct_tok, incorrect_tok = 7, 9
    labels = torch.tensor([[correct_tok, incorrect_tok]])
    input_length = torch.tensor([n_pos])

    # Model favours the correct letter at the answer position (n_pos - 1).
    favour_correct = torch.zeros(batch, n_pos, vocab)
    favour_correct[:, n_pos - 1, correct_tok] = 5.0
    val = spec._logit_diff(
        favour_correct, favour_correct, input_length, labels, mean=False, loss=False
    )
    assert val.item() == pytest.approx(5.0, abs=1e-4)
    # loss=True negates -> a correct prediction is a low (negative) loss.
    val_loss = spec._logit_diff(
        favour_correct, favour_correct, input_length, labels, mean=False, loss=True
    )
    assert val_loss.item() == pytest.approx(-5.0, abs=1e-4)

    # Model favours the incorrect letter -> negative logit diff.
    favour_incorrect = torch.zeros(batch, n_pos, vocab)
    favour_incorrect[:, n_pos - 1, incorrect_tok] = 4.0
    val_neg = spec._logit_diff(
        favour_incorrect, favour_incorrect, input_length, labels, mean=False, loss=False
    )
    assert val_neg.item() == pytest.approx(-4.0, abs=1e-4)


def test_logit_diff_metric_left_padding_safe():
    """The metric scores logits at input_length-1; with left padding the
    answer position is the last index."""
    spec = WinoGrandeMCTaskSpec()
    vocab = 40
    labels = torch.tensor([[3, 8]])

    base = torch.zeros(1, 8, vocab)
    base[:, 8 - 1, 3] = 6.0
    v_short = spec._logit_diff(base, base, torch.tensor([8]), labels, mean=False, loss=False)

    padded = torch.zeros(1, 12, vocab)
    padded[:, 4:, :] = base  # 4 pad positions prepended
    v_pad = spec._logit_diff(padded, padded, torch.tensor([12]), labels, mean=False, loss=False)
    assert v_short.item() == pytest.approx(v_pad.item(), abs=1e-4)


# --------------------------------------------------------------------------
# Artifact metadata records the resolved chat-template mode
# --------------------------------------------------------------------------
def test_artifact_metadata_records_chat_template_mode():
    spec = WinoGrandeMCTaskSpec()
    md = spec.artifact_metadata({"algorithm": "eap", "level": "node"})
    assert md["task"] == "winogrande_mc"
    assert md["chat_template_mode"] == "auto"
    assert md["metric"] == "logit_diff"
    assert md["corruption_mode"] == "option_swap"

    md2 = spec.artifact_metadata({"algorithm": "eap", "level": "node", "chat_template_mode": "on"})
    assert md2["chat_template_mode"] == "on"


# --------------------------------------------------------------------------
# CSV generation (network) -- clean/corrupt are token-aligned (option swap is
# a minimal change), which is what the logit-diff metric requires.
# --------------------------------------------------------------------------
def test_generate_csv_pairs_are_token_aligned(tmp_path):
    """clean and corrupt differ only by the swapped option spans, so they
    tokenize to the same length and the answer-letter tokens are distinct."""
    transformer_lens = pytest.importorskip("transformer_lens")

    try:
        model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cpu")
    except (OSError, ConnectionError) as e:
        pytest.skip(f"gpt2 unavailable (offline): {e}")

    spec = WinoGrandeMCTaskSpec()
    out = tmp_path / "wg_mc_gen.csv"
    try:
        df = spec._generate_winogrande_mc_csv(
            n_samples=12, output_path=out, seed=42, model=model, apply=False
        )
    except (OSError, ConnectionError) as e:
        pytest.skip(f"winogrande dataset unavailable (offline): {e}")
    except (TypeError, RuntimeError) as e:
        if "pickle" in str(e) or "RLock" in str(e):
            pytest.skip(f"datasets/dill cross-test pickle pollution: {e}")
        raise

    assert len(df) > 0
    for _, row in df.iterrows():
        assert row["clean"] != row["corrupted"]
        # Option swap is a minimal change -> token-length aligned pair.
        ct = model.to_tokens(row["clean"], prepend_bos=True)
        kt = model.to_tokens(row["corrupted"], prepend_bos=True)
        assert ct.shape[1] == kt.shape[1]
        # correct/incorrect letter tokens are distinct so logit-diff is defined.
        assert row["correct_idx"] != row["incorrect_idx"]
