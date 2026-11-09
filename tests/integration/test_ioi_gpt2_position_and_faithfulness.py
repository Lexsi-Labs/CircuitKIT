"""GPT-2 checks for the ``ioi`` scored position and for good-versus-bad circuit separation.

Slow tier: loads the real ``gpt2`` weights (from the local Hugging Face cache)
and runs one EAP discovery and two faithfulness evaluations on CPU.
Run with ``pytest -m slow tests/integration/test_ioi_gpt2_position_and_faithfulness.py``.
"""

import pytest
import torch

import circuitkit as ck
from circuitkit.circuit import Circuit

pytestmark = pytest.mark.slow

N = 16
SEED = 0


@pytest.fixture(scope="module")
def gpt2():
    return ck.load_model("gpt2", dtype="float32", device="cpu")


def test_ioi_logit_difference_is_read_where_the_name_is_predicted(gpt2, tmp_path):
    from circuitkit.api import _make_eval_metric
    from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
    from circuitkit.data.task_data.tasks.ioi.ioi_dataset import IOIDataset
    from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks

    spec = _bootstrap_builtin_tasks()["ioi"]
    cfg = {
        "algorithm": "eap",
        "level": "node",
        "batch_size": 8,
        "data_params": {"num_examples": N, "seed": SEED, "cache_dir": str(tmp_path)},
    }
    dl = spec.build_dataloader(gpt2, cfg, "cpu")
    metric = _make_eval_metric(spec)
    task_values = []
    for clean, corrupted, label in dl:
        tokens, _, mask, _, lengths, _ = tokenize_batch_pair(
            gpt2, clean, corrupted, pair_padding_side=dl.pair_padding_side
        )
        assert all(gpt2.to_string(tokens[i, lengths[i] - 1]) == " to" for i in range(len(clean)))
        with torch.no_grad():
            logits = gpt2(tokens, attention_mask=mask)
        task_values += metric(logits, None, lengths, label).tolist()

    # Reference, computed without the task code: the same sentences with the
    # answer name removed, read at the last token.
    data = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1, model=gpt2, seed=SEED)
    reference = []
    for i, prompt in enumerate(data.ioi_prompts[:N]):
        text = prompt["text"][: -len(" " + prompt["IO"])]
        with torch.no_grad():
            last = gpt2(gpt2.to_tokens(text))[0, -1]
        reference.append(float(last[data.io_tokenIDs[i]] - last[data.s_tokenIDs[i]]))

    task_mean = sum(task_values) / N
    assert task_mean == pytest.approx(sum(reference) / N, abs=1e-3)
    # At the answer name itself the same prompts give about 1.0.
    assert task_mean > 2.5


def test_reversed_control_circuit_scores_clearly_below_the_discovered_one(
    gpt2, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # the task cache and discovery outputs default to the cwd
    circuit = ck.discover(
        gpt2,
        "ioi",
        algorithm="eap",
        n_examples=N,
        batch_size=8,
        sparsity=0.3,
        seed=SEED,
        device="cpu",
        output_path=str(tmp_path / "circuit.pt"),
    )
    # Control: the same number of components removed, but ranks reversed, so
    # the components EAP scored highest are the ones taken out.
    top = max(abs(v) for v in circuit.scores.values())
    control = Circuit(
        circuit.nodes,
        {k: top - abs(v) for k, v in circuit.scores.items()},
        task="ioi",
        algorithm="eap",
        level="node",
    )
    kw = dict(n_examples=N, batch_size=8, seed=SEED, device="cpu")
    good = ck.faithfulness(gpt2, circuit, "ioi", **kw)
    bad = ck.faithfulness(gpt2, control, "ioi", **kw)

    assert good.patching_raw_ratio - bad.patching_raw_ratio > 0.5
    assert good.patching_score > 0.7 and bad.patching_score < 0.3
    # Zero ablation takes GPT-2 below the corrupt baseline for both circuits, so
    # both clipped scores are 0.0. The raw ratio is reported and says so.
    for report in (good, bad):
        assert report.ablation_raw_ratio is not None
        if not 0.0 <= report.ablation_raw_ratio <= 1.0:
            assert "OUT OF RANGE" in repr(report)
