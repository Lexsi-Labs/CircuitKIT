"""Unit smoke tests for the pruner module."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformer_lens")


@pytest.fixture(scope="module")
def gpt2_model():
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HookedTransformer.from_pretrained("gpt2", device=device, dtype=torch.float32)


def test_pruner_module_imports():
    """The pruner module + its public classes are importable."""
    from circuitkit.applications import pruner

    for name in ("__name__",):
        assert hasattr(pruner, name)


def test_structural_pruner_imports():
    from circuitkit.applications.pruning.pruner import StructuralPruner

    assert StructuralPruner is not None


def test_node_pruner_returns_list():
    """Sanity: NodePruner constructed from a tiny scores dict returns something."""
    from circuitkit.applications.pruning.node_pruner import NodePruner

    scores = {"A0.0": 0.9, "A1.0": 0.6, "MLP 0": 0.5, "A5.5": 0.4, "A7.3": 0.3}
    pruner_obj = NodePruner()
    out = pruner_obj.prune(scores, target_sparsity=0.4, scope="heads")
    assert isinstance(out, (list, tuple, set)), f"unexpected return type {type(out)}"
    assert len(out) > 0
