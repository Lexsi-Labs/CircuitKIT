"""float16 is refused where weights are trained in place with Adam/AdamW, and
FineTuneEditHandler honours its weight_decay.

Adam on float16 parameters turns them into NaN/inf on the first step: eps=1e-8
and the squared gradient of any gradient below about 5e-3 round to 0 in float16,
so the update is 0/0 or x/0. bfloat16 and float32 are fine.
"""

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformer_lens")

from transformer_lens import HookedTransformer, HookedTransformerConfig  # noqa: E402

from circuitkit.applications.common_utils._dtype import reject_float16  # noqa: E402
from circuitkit.applications.editing import fine_tune_editing as fte  # noqa: E402


def _tiny_model(dtype):
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=16,
        d_head=4,
        n_heads=4,
        d_mlp=32,
        d_vocab=50,
        n_ctx=16,
        act_fn="gelu",
        dtype=dtype,
        device="cpu",
    )
    return HookedTransformer(cfg)


class _Linear(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.lin = nn.Linear(8, 8).to(dtype)
        self.cfg = types.SimpleNamespace(device="cpu")

    def forward(self, x):
        return self.lin(x)


def test_helper_rejects_any_float16_parameter():
    mixed = [nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2, dtype=torch.float16))]
    with pytest.raises(ValueError, match="float16"):
        reject_float16(mixed, "Thing")
    with pytest.raises(ValueError, match="^Thing "):
        reject_float16(mixed, "Thing")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_helper_accepts_float32_and_bfloat16(dtype):
    reject_float16([nn.Parameter(torch.zeros(2, dtype=dtype))], "Thing")


def test_fine_tune_edit_handler_rejects_float16_model():
    with pytest.raises(ValueError, match="FineTuneEditHandler"):
        fte.FineTuneEditHandler(_Linear(torch.float16))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fine_tune_edit_handler_accepts_float32_and_bfloat16(dtype):
    fte.FineTuneEditHandler(_Linear(dtype))


@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_fine_tune_edit_handler_passes_weight_decay_to_adamw(monkeypatch, weight_decay):
    """The configured weight_decay reaches the optimiser; AdamW's own default is 0.01."""
    seen = []
    real_adamw = torch.optim.AdamW

    class Spy(real_adamw):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self.param_groups[0]["weight_decay"])

    monkeypatch.setattr(torch.optim, "AdamW", Spy)
    net = _Linear(torch.float32)
    handler = fte.FineTuneEditHandler(net, steps=1, lr=1e-3, weight_decay=weight_decay)
    handler._fact_loss = lambda prompt, subject, target: net(torch.ones(1, 8)).pow(2).sum()

    handler._fine_tune([("p", "s", "t")], steps=1, lr=1e-3, batch_size=1, max_grad_norm=1.0)

    assert seen == [weight_decay]


def test_circuit_tuner_rejects_float16_model():
    from circuitkit.applications.finetuning.circuit_tuning import CircuitTuner

    model = _tiny_model(torch.float16)
    before = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError, match="CircuitTuner"):
        CircuitTuner(model, node_scores={"MLP 0": 1.0}).fit(prompts=["a"], targets=["b"])
    assert [p.requires_grad for p in model.parameters()] == before


def test_cure_clue_rejects_float16_model():
    from circuitkit.applications.common_utils.cure_clue import CureClueUnlearner

    model = _tiny_model(torch.float16)
    before = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError, match="CureClueUnlearner"):
        CureClueUnlearner(model, node_scores={"MLP 0": 1.0}).unlearn(
            forget_prompts=["a"], forget_targets=["b"]
        )
    assert [p.requires_grad for p in model.parameters()] == before


def test_circuit_lora_finetune_rejects_float16_model():
    from circuitkit.applications.finetuning.soft_healing import CircuitLoRA

    model = _tiny_model(torch.float16)
    lora = CircuitLoRA(model, {"MLP 0": 1.0}, lora_rank=2)
    with pytest.raises(ValueError, match="CircuitLoRA.finetune"):
        lora.finetune(train_loader=[], epochs=1)
