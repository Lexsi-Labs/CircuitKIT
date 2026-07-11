# steering

Circuit-guided steering: activation steering, contrastive weight steering, and composition/safety utilities (usable via `circuitkit.applications.steering`, not part of the flat public API).

## Key modules

- `steering.py` — `ActivationSteering`: activation steering via runtime hooks at circuit nodes; the steering vector is the target-minus-source activation difference, leaving weights untouched (reversible, the standard CAA-style baseline).
- `weight_steering.py` — `CircuitWeightSteering` (type `HeadSlice`): C-ΔΘ contrastive weight steering; fine-tune positive/negative copies, take the per-head weight difference (θ_pos − θ_neg), and add it to the target with a coefficient (the paper-faithful method).
- `steering_enhanced.py` — `SteeringComposer` and `SafetyDatasetSynthesis`: utilities on top of activation steering for composing multiple steering vectors, synthesizing safety/adversarial datasets, and detecting interference.

## Public API / entry points

`__all__`: `ActivationSteering`, `SteeringComposer`, `SafetyDatasetSynthesis`, `CircuitWeightSteering`. The methods are not interchangeable: pick `ActivationSteering` for a reversible inference-time baseline, or `CircuitWeightSteering` for a permanent weight edit.

## How it fits

One of the intervention applications. It targets circuit nodes and heads to steer model behavior, either at inference time or by editing weights.
