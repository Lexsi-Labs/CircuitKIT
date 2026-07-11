# ibcircuit

Backend for IBCircuit — circuit discovery via an Information Bottleneck. Learns
per-component importance weights that inject noise inversely proportional to
importance, then averages them into task-general component scores.

## Key modules

- `ib_noise.py` — `apply_ib_noise` (the core IB formula) and weight initialisers
  (`initialize_attn_ib_weights`, `initialize_mlp_ib_weights`) for node/neuron
  granularity.
- `model_wrapper.py` — `IBHookedTransformer`: wraps a frozen `HookedTransformer`
  and injects IB noise into attention/MLP activations via hooks in
  `forward_with_ib`.
- `trainer.py` — `run_ib_discovery` (and `train_ib_epoch`): the training loop that
  optimizes importance weights on a fixed batch and returns component scores.
- `ib_utils.py` — training helpers: logit extraction at positions, task/baseline
  loss, batch validation.
- `artifact_export.py` — `export_circuit_artifact`: neuron importance scores →
  `CircuitArtifact`.

## How it fits

Dispatched from `api.discover_circuit` (`ibcircuit`, experimental tier) via
`trainer.run_ib_discovery`. Reference: Bian, Niu, Yuan et al., "IBCircuit:
Towards Holistic Circuit Discovery with Information Bottleneck" (ICML 2025),
https://github.com/ivanniu/IBCircuit. Note: single-batch training, may OOM on 3B models.
