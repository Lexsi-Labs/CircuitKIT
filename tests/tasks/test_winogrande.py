"""Focused tests for the (fixed) WinoGrande circuit-discovery task.

WinoGrande's disambiguating cue lies AFTER the blank, so the task is scored
with a *suffix log-likelihood* metric: fill the blank with each option and
compare the model's log-likelihood of the text following the blank. These
tests pin that contract:

  * task registration / discoverability
  * the suffix-LL metric is a real differentiable log-likelihood that scores
    the suffix span encoded in ``labels``
  * the metric is length-normalised and sign-correct (higher suffix-LL when
    the model confidently predicts the suffix tokens)
  * the custom EAP dataset packs ``labels`` as [n_suffix, suffix_tokens...]
    and yields token-aligned clean/corrupt pairs
  * ``validate_discovery_config`` rejects unsupported algorithms / levels

The CSV-generation test hits the HuggingFace ``winogrande`` dataset and is
skipped automatically if it is unavailable (offline CI).
"""

import json
import math

import pytest
import torch

from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks
from circuitkit.tasks.builtins.winogrande import WinoGrandeTaskSpec, _WinoGrandeEAPDataset
from circuitkit.tasks.registry import get_task, list_tasks


@pytest.fixture(scope="module", autouse=True)
def _bootstrap():
    _bootstrap_builtin_tasks()


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
def test_winogrande_is_registered():
    assert "winogrande" in list_tasks()
    spec = get_task("winogrande")
    assert spec.name == "winogrande"
    assert spec.pair_padding_side == "left"


# --------------------------------------------------------------------------
# Metric: differentiable suffix log-likelihood
# --------------------------------------------------------------------------
def test_suffix_loglik_metric_is_differentiable():
    """The discovery metric must be a differentiable function of the logits."""
    spec = WinoGrandeTaskSpec()
    metric = spec.metric_fn()  # loss=True, mean=True

    batch, n_pos, vocab = 4, 12, 200
    logits = torch.randn(batch, n_pos, vocab, requires_grad=True)
    clean_logits = torch.randn(batch, n_pos, vocab)
    input_length = torch.tensor([n_pos] * batch)
    # labels: [n_suffix, suffix_tok_0, suffix_tok_1, suffix_tok_2]
    labels = torch.tensor(
        [
            [3, 10, 11, 12],
            [2, 20, 21, 0],
            [3, 30, 31, 32],
            [1, 40, 0, 0],
        ]
    )

    loss = metric(logits, clean_logits, input_length, labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_suffix_loglik_scores_the_suffix_span():
    """A model that confidently predicts the suffix tokens scores ~0;
    a uniform model scores log(1/vocab) per suffix token."""
    spec = WinoGrandeTaskSpec()

    batch, n_pos, vocab = 2, 10, 50
    labels = torch.tensor([[2, 7, 9], [2, 7, 9]])  # n_suffix=2, toks 7 & 9

    # Confident: place huge logit mass on each suffix token at its predicting
    # position. Suffix tokens occupy positions n_pos-2 and n_pos-1; the token
    # at position p is predicted from logits[p-1].
    confident = torch.zeros(batch, n_pos, vocab)
    confident[:, n_pos - 2 - 1, 7] = 30.0
    confident[:, n_pos - 1 - 1, 9] = 30.0
    val_conf = spec._suffix_loglik(
        confident,
        confident,
        torch.tensor([n_pos, n_pos]),
        labels,
        mean=True,
        loss=False,
    )
    assert val_conf.item() == pytest.approx(0.0, abs=1e-3)

    # Uniform logits -> per-token mean LL is log(1/vocab).
    uniform = torch.zeros(batch, n_pos, vocab)
    val_uni = spec._suffix_loglik(
        uniform,
        uniform,
        torch.tensor([n_pos, n_pos]),
        labels,
        mean=True,
        loss=False,
    )
    assert val_uni.item() == pytest.approx(math.log(1.0 / vocab), abs=1e-3)

    # loss=True negates: NLL of a uniform model is positive.
    val_loss = spec._suffix_loglik(
        uniform,
        uniform,
        torch.tensor([n_pos, n_pos]),
        labels,
        mean=True,
        loss=True,
    )
    assert val_loss.item() == pytest.approx(-math.log(1.0 / vocab), abs=1e-3)


def test_suffix_loglik_is_length_normalised():
    """Per-token mean -> a 1-token and a 3-token suffix of equal per-token
    confidence yield the same metric value (no length bias)."""
    spec = WinoGrandeTaskSpec()
    n_pos, vocab = 12, 60

    # Example A: 1 suffix token; Example B: 3 suffix tokens. Both uniform.
    labels = torch.tensor([[1, 5, 0, 0], [3, 5, 6, 7]])
    uniform = torch.zeros(2, n_pos, vocab)
    per_ex = spec._suffix_loglik(
        uniform,
        uniform,
        torch.tensor([n_pos, n_pos]),
        labels,
        mean=False,
        loss=False,
    )
    assert per_ex[0].item() == pytest.approx(per_ex[1].item(), abs=1e-4)


def test_suffix_loglik_left_padding_safe():
    """Indexing the suffix from the sequence END keeps it correct when the
    batch is left-padded (n_pos is the shared padded length)."""
    spec = WinoGrandeTaskSpec()
    vocab = 40
    labels = torch.tensor([[2, 3, 8]])

    # Short sequence, then the same sequence left-padded by 4 positions.
    base = torch.zeros(1, 8, vocab)
    base[:, 8 - 2 - 1, 3] = 25.0
    base[:, 8 - 1 - 1, 8] = 25.0
    v_short = spec._suffix_loglik(base, base, torch.tensor([8]), labels, mean=False, loss=False)

    padded = torch.zeros(1, 12, vocab)
    padded[:, 4:, :] = base  # 4 pad positions prepended
    v_pad = spec._suffix_loglik(padded, padded, torch.tensor([12]), labels, mean=False, loss=False)
    assert v_short.item() == pytest.approx(v_pad.item(), abs=1e-4)


# --------------------------------------------------------------------------
# Custom EAP dataset: labels carry the suffix span
# --------------------------------------------------------------------------
def test_eap_dataset_packs_suffix_labels(tmp_path):
    import pandas as pd

    csv = tmp_path / "wg.csv"
    pd.DataFrame(
        [
            {
                "clean": "the cat sat because it was tired.",
                "corrupted": "the dog sat because it was tired.",
                "correct_idx": 100,
                "incorrect_idx": 200,
                "suffix_tokens": json.dumps([11, 12, 13]),
            },
            {
                "clean": "a then b.",
                "corrupted": "a then c.",
                "correct_idx": 101,
                "incorrect_idx": 201,
                "suffix_tokens": json.dumps([14]),
            },
        ]
    ).to_csv(csv, index=False)

    ds = _WinoGrandeEAPDataset(str(csv))
    assert len(ds) == 2
    # max suffix length across the dataset is 3.
    assert ds.max_suffix == 3

    clean, corrupted, labels = ds[0]
    assert clean != corrupted
    # labels = [n_suffix] + suffix_tokens padded to max_suffix.
    assert labels[0] == 3
    assert labels[1:4] == [11, 12, 13]
    assert len(labels) == 1 + ds.max_suffix

    clean1, _, labels1 = ds[1]
    assert labels1[0] == 1  # n_suffix
    assert labels1[1] == 14  # the single suffix token
    assert labels1[2:] == [0, 0]  # padded with the inert sentinel
    assert len(labels1) == len(labels)  # uniform length -> collatable


def test_eap_dataset_collates_into_batch(tmp_path):
    """collate_EAP must turn the per-example label lists into a 2-D tensor."""
    import pandas as pd

    from circuitkit.backends.eap.eap_utils import collate_EAP

    csv = tmp_path / "wg.csv"
    pd.DataFrame(
        [
            {
                "clean": "x because y.",
                "corrupted": "x because z.",
                "correct_idx": 1,
                "incorrect_idx": 2,
                "suffix_tokens": json.dumps([5, 6]),
            },
            {
                "clean": "p so q.",
                "corrupted": "p so r.",
                "correct_idx": 3,
                "incorrect_idx": 4,
                "suffix_tokens": json.dumps([7]),
            },
        ]
    ).to_csv(csv, index=False)

    ds = _WinoGrandeEAPDataset(str(csv))
    clean, corrupted, labels = collate_EAP([ds[0], ds[1]])
    assert len(clean) == 2 and len(corrupted) == 2
    assert isinstance(labels, torch.Tensor)
    assert labels.shape == (2, 1 + ds.max_suffix)
    assert labels[0, 0].item() == 2 and labels[1, 0].item() == 1


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------
def test_validate_discovery_config_rejects_acdc():
    spec = WinoGrandeTaskSpec()
    with pytest.raises(ValueError, match="does not support"):
        spec.validate_discovery_config({"algorithm": "acdc", "level": "node"})


def test_validate_discovery_config_rejects_bad_level():
    spec = WinoGrandeTaskSpec()
    with pytest.raises(ValueError, match="level"):
        spec.validate_discovery_config({"algorithm": "eap", "level": "edge"})


def test_validate_discovery_config_accepts_eap():
    spec = WinoGrandeTaskSpec()
    # Should not raise.
    spec.validate_discovery_config({"algorithm": "eap", "level": "node", "batch_size": 8})


# --------------------------------------------------------------------------
# CSV generation (network) — clean/corrupt are token-aligned with a shared
# suffix span, which is exactly what the suffix-LL metric requires.
# --------------------------------------------------------------------------
def test_generate_csv_pairs_are_token_aligned(tmp_path):
    """clean and corrupt differ only at the single-token option position, so
    they share the suffix at identical trailing positions."""
    transformer_lens = pytest.importorskip("transformer_lens")

    try:
        model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device="cpu")
    except (OSError, ConnectionError) as e:
        pytest.skip(f"gpt2 unavailable (offline): {e}")

    spec = WinoGrandeTaskSpec()
    out = tmp_path / "wg_gen.csv"
    try:
        df = spec._generate_winogrande_csv(n_samples=12, output_path=out, seed=42, model=model)
    except (OSError, ConnectionError) as e:
        pytest.skip(f"winogrande dataset unavailable (offline): {e}")
    except (TypeError, RuntimeError) as e:
        if "pickle" in str(e) or "RLock" in str(e):
            pytest.skip(f"datasets/dill cross-test pickle pollution: {e}")
        raise

    assert len(df) > 0
    for _, row in df.iterrows():
        assert row["clean"] != row["corrupted"]
        ct = model.to_tokens(row["clean"], prepend_bos=True)
        kt = model.to_tokens(row["corrupted"], prepend_bos=True)
        # single-token options -> token-length aligned pair.
        assert ct.shape[1] == kt.shape[1]
        # the precomputed suffix tokens are the actual trailing tokens.
        suffix = json.loads(row["suffix_tokens"])
        assert len(suffix) > 0
        assert ct.squeeze(0)[-len(suffix) :].tolist() == suffix
        # clean and corrupt share that suffix span verbatim.
        assert kt.squeeze(0)[-len(suffix) :].tolist() == suffix
