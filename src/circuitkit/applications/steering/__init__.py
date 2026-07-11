"""Steering: three distinct circuit-guided steering methods.

These are NOT interchangeable — pick by what you want to change:

- ``ActivationSteering`` (steering.py) — *activation steering*. Adds runtime
  hooks at circuit nodes; the steering vector is the (target − source)
  activation difference. Weights are untouched and the effect is fully
  reversible. This is the standard literature method (CAA-style); use it as
  the inference-time baseline.

- ``CircuitWeightSteering`` (weight_steering.py) — *C-ΔΘ contrastive weight
  steering*. Permanently edits per-head weight slices (W_Q/K/V/O): fine-tune
  M_pos and M_neg, take θ_pos − θ_neg, add to the target with coefficient k.
  This is the paper-faithful method (the canonical C-ΔΘ recipe) — the one to
  use when you want the contribution method, not the baseline.

- ``SteeringComposer`` / ``SafetyDatasetSynthesis`` (steering_enhanced.py) —
  utilities layered on activation steering: compose multiple steering
  vectors, synthesize safety/adversarial datasets, detect interference.
"""

from .steering import ActivationSteering
from .steering_enhanced import SafetyDatasetSynthesis, SteeringComposer
from .weight_steering import CircuitWeightSteering

__all__ = [
    "ActivationSteering",
    "SteeringComposer",
    "SafetyDatasetSynthesis",
    "CircuitWeightSteering",
]
