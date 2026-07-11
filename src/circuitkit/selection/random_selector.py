"""Random selector — wraps circuitkit RandomBaseline."""
import torch
from circuitkit.selection import register

@register("random")
def random_selector(model, task_name: str, config: dict) -> dict:
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    seed = config.get("seed", 42)
    gen = torch.Generator()
    gen.manual_seed(seed)
    scores = {}
    for L in range(nL):
        for h in range(nH):
            scores[f"A{L}.{h}"] = float(torch.rand(1, generator=gen).item())
        scores[f"MLP {L}"] = float(torch.rand(1, generator=gen).item())
    return scores
