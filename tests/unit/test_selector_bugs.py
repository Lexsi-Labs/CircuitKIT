"""Regression tests for three selector bugs:

  Bug 1 — wanda zeroed all attention heads because it read
          blocks.{L}.attn.hook_result without enabling use_attn_result.
  Bug 2 — awq had the identical bug.
  Bug 3 — multi_granular never scored MLPs (scores.get("MLP {L}", 0.0)),
          so all MLPs tied at the global minimum after normalization.

These tests run the real selectors on gpt2 and assert the outputs are
non-degenerate: attention heads are not all 0.0 (wanda/awq) and MLP
scores are not all-equal (multi_granular).
"""

import unittest

import pytest

torch = pytest.importorskip("torch")
HookedTransformer = pytest.importorskip("transformer_lens").HookedTransformer

import circuitkit.selection as S  # noqa: E402  import after importorskip guards
import circuitkit.selection.wanda_selector  # noqa: F401,E402  registers "wanda"
from circuitkit.tasks.bootstrap import _bootstrap_builtin_tasks  # noqa: E402

_bootstrap_builtin_tasks()

_CFG = {
    "model_name": "gpt2",
    "num_examples": 16,
    "max_batches": 3,
    "batch_size": 4,
}


def _fresh_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    # Precondition for the bug: flag starts disabled on a fresh model.
    assert model.cfg.use_attn_result is False
    return model


def _split(scores):
    attn = {k: v for k, v in scores.items() if k.startswith("A")}
    mlp = {k: v for k, v in scores.items() if k.startswith("MLP")}
    return attn, mlp


class TestSelectorBugs(unittest.TestCase):
    def _assert_finite(self, values):
        for v in values:
            self.assertTrue(torch.isfinite(torch.tensor(float(v))))

    def test_wanda_attention_heads_not_all_zero(self):
        scores = S.get_selector("wanda")(_fresh_model(), "ioi", _CFG)
        attn, mlp = _split(scores)
        self.assertEqual(len(attn), 144)
        self._assert_finite(attn.values())
        self._assert_finite(mlp.values())
        self.assertFalse(
            all(v == 0.0 for v in attn.values()),
            "wanda: all attention-head scores are 0.0",
        )
        # Non-degenerate: heads are genuinely distinct.
        self.assertGreater(len({round(v, 6) for v in attn.values()}), 10)

    def test_awq_attention_heads_not_all_zero(self):
        # Fresh model proves the fix is not order-dependent.
        scores = S.get_selector("awq")(_fresh_model(), "ioi", _CFG)
        attn, mlp = _split(scores)
        self.assertEqual(len(attn), 144)
        self._assert_finite(attn.values())
        self._assert_finite(mlp.values())
        self.assertFalse(
            all(v == 0.0 for v in attn.values()),
            "awq: all attention-head scores are 0.0",
        )
        self.assertGreater(len({round(v, 6) for v in attn.values()}), 10)

    def test_multi_granular_mlps_not_all_equal(self):
        scores = S.get_selector("multi_granular")(_fresh_model(), "ioi", _CFG)
        attn, mlp = _split(scores)
        self.assertEqual(len(mlp), 12)
        self._assert_finite(mlp.values())
        self.assertGreater(
            len({round(v, 6) for v in mlp.values()}),
            1,
            "multi_granular: all MLP scores are tied",
        )
        # MLPs are genuinely ranked, not just two buckets.
        self.assertGreaterEqual(len({round(v, 4) for v in mlp.values()}), 5)
        # Heads remain non-degenerate too.
        self.assertFalse(all(v == 0.0 for v in attn.values()))


if __name__ == "__main__":
    unittest.main()
