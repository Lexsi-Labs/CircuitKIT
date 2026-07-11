"""
CD-T (Contextual Decomposition for Transformers) backend.

Sourced from the canonical CircuitKit fork. CD-T uses contextual
decomposition to attribute predictions to source nodes via a
forward-pass linearisation of every transformer component, rather
than the gradient-based EAP family. Different paradigm: produces
node-level relevance scores, not edge-level patches.

Status in this fork: backend code present (pyfunctions/), but the
discover_circuit() dispatcher in api.py does not yet wire CD-T
through the TransformerLens path that the EAP family uses. Full
integration needs:

  1. An adapter that runs CD-T on a HuggingFace-transformers model
     view of the same checkpoint (CD-T is built around HF Bert /
     GPT layer wrappers in `pyfunctions.wrappers`, not TL hooks).
  2. A Graph reconstruction that converts the resulting
     TargetNodeDecompositionList into the CircuitScores format used
     downstream by pruning / evaluate_circuit.

This module exposes the raw functions for users who want to invoke
CD-T directly outside the discover_circuit pipeline.
"""

from .pyfunctions import cdt_basic as basic
from .pyfunctions import cdt_core as core
from .pyfunctions import wrappers as wrappers  # re-export

__all__ = ["wrappers", "core", "basic"]
