# prune_algos

Edge-importance scoring algorithms for the ACDC backend — each takes a
`PatchableModel` + dataloader and returns `PruneScores`.

## Key modules

- `ACDC.py` — `acdc_prune_scores`: the ACDC threshold-sweep algorithm (Conmy et
  al. 2023); runs over a grid of tau values and scores edges by the smallest tau
  at which they become unimportant. Optimizes KL-divergence or MSE faithfulness.
- `mask_gradient.py` — `mask_gradient_prune_scores`: scores edges by the gradient
  of interpolation masks between clean and ablated activations; equivalent to EAP
  under specific arguments, with optional integrated-gradients approximation.
- `prune_algos.py` — `PruneAlgo`, a frozen dataclass wrapping a named scoring
  function so algorithms can be registered and compared by key.

## How it fits

`api.discover_circuit` imports `acdc_prune_scores` from `ACDC.py` for the `acdc`
algorithm; the other scorers are reusable building blocks for ACDC/EAP variants.
