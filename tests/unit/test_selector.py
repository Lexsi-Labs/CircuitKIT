import unittest

import torch

from circuitkit.applications.selective_finetuning.selector import (
    _count_selected_heads,
    _filter_by_layer_exclusion,
    _resolve_gqa_indices,
    _resolve_mlp_neuron,
    _resolve_mlp_node,
    build_baseline_selection,
    random_selection,
    select_components,
)


class TestSelector(unittest.TestCase):
    def setUp(self):
        # Common test constants
        self.n_layers = 4
        self.n_q_heads = 4
        self.n_kv_heads = 2
        self.head_dim = 16

        # Mock node-level scores
        self.head_scores_node = {
            (0, 0): 0.9,
            (0, 1): 0.1,
            (0, 2): 0.5,
            (0, 3): 0.2,
            (1, 0): 0.8,
            (1, 1): 0.85,
            (1, 2): 0.3,
            (1, 3): 0.4,
            (2, 0): 0.1,
            (2, 1): 0.2,
            (2, 2): 0.95,
            (2, 3): 0.6,
            (3, 0): 0.4,
            (3, 1): 0.5,
            (3, 2): 0.6,
            (3, 3): 0.7,
        }
        self.mlp_scores_node = {0: 1.5, 1: 0.5, 2: 2.0, 3: 0.1}
        self.metadata_node = {"level": "node", "mlp_neuron_level": False}

        # Mock neuron-level scores
        self.head_scores_neuron = self.head_scores_node.copy()
        self.mlp_scores_neuron = {
            0: torch.tensor([0.1, 0.9, 0.2]),  # Neurons for layer 0
            1: torch.tensor([0.8, 0.3, 0.4]),  # Neurons for layer 1
            2: torch.tensor([0.2, 0.7, 0.6]),  # Neurons for layer 2
            3: torch.tensor([0.5, 0.1, 0.9]),  # Neurons for layer 3
        }
        self.metadata_neuron = {"level": "neuron", "mlp_neuron_level": True}

    # -----------------------------------------------------------------------
    # 1. Layer Exclusion Tests
    # -----------------------------------------------------------------------
    def test_filter_by_layer_exclusion(self):
        head, mlp = _filter_by_layer_exclusion(
            self.head_scores_node,
            self.mlp_scores_node,
            exclude_first_n=1,
            exclude_last_n=1,
            n_layers=self.n_layers,
        )
        # Should only contain layers 1 and 2
        for lyr, h in head.keys():
            self.assertIn(lyr, [1, 2])
        for lyr in mlp.keys():
            self.assertIn(lyr, [1, 2])

    def test_filter_exclusion_bounds_error(self):
        with self.assertRaises(ValueError):
            _filter_by_layer_exclusion({}, {}, exclude_first_n=2, exclude_last_n=3, n_layers=4)

    # -----------------------------------------------------------------------
    # 2. GQA Index Resolution Tests
    # -----------------------------------------------------------------------
    def test_resolve_gqa_indices_node_level(self):
        # group_size = 4 // 2 = 2.
        # q_heads 2 and 3 belong to kv_head 1.
        pairs = [(0, 2), (0, 3)]
        res = _resolve_gqa_indices(
            pairs, self.n_q_heads, self.n_kv_heads, self.head_dim, neuron_level=False
        )

        self.assertIn("attn_0", res)
        attn = res["attn_0"]

        # q and o should cover indices for heads 2 and 3 (2*16 to 4*16-1 -> 32 to 63)
        expected_qo = list(range(32, 64))
        self.assertEqual(attn["q"], expected_qo)
        self.assertEqual(attn["o"], expected_qo)

        # k and v should cover kv_head 1 (1*16 to 2*16-1 -> 16 to 31)
        expected_kv = list(range(16, 32))
        self.assertEqual(attn["k"], expected_kv)
        self.assertEqual(attn["v"], expected_kv)

    def test_resolve_gqa_indices_neuron_level(self):
        pairs = [(0, 2)]
        res = _resolve_gqa_indices(
            pairs, self.n_q_heads, self.n_kv_heads, self.head_dim, neuron_level=True
        )
        self.assertIn("attn_0", res)
        self.assertNotIn("q", res["attn_0"])
        self.assertIn("o", res["attn_0"])
        self.assertEqual(len(res["attn_0"]["o"]), self.head_dim)

    # -----------------------------------------------------------------------
    # 3. MLP Resolution Tests
    # -----------------------------------------------------------------------
    def test_resolve_mlp_node(self):
        res = _resolve_mlp_node([1, 3])
        self.assertEqual(res, {"mlp_1": None, "mlp_3": None})

    def test_resolve_mlp_neuron(self):
        res = _resolve_mlp_neuron(
            self.mlp_scores_neuron, candidate_layers=[0, 1, 2, 3], top_frac=0.25
        )
        # 4 layers * 3 neurons = 12 total. Top 25% = 3 neurons.
        # Top 3 abs scores are: layer 0, col 1 (0.9); layer 3, col 2 (0.9); layer 1, col 0 (0.8)
        self.assertIn("mlp_0", res)
        self.assertEqual(res["mlp_0"], [1])
        self.assertIn("mlp_3", res)
        self.assertEqual(res["mlp_3"], [2])
        self.assertIn("mlp_1", res)
        self.assertEqual(res["mlp_1"], [0])
        self.assertNotIn("mlp_2", res)

    # -----------------------------------------------------------------------
    # 4. End-to-End Selection Tests (Circuit)
    # -----------------------------------------------------------------------
    def test_select_components_node_both(self):
        res = select_components(
            self.head_scores_node,
            self.mlp_scores_node,
            self.metadata_node,
            top_frac=0.5,
            scope="both",
            n_layers=self.n_layers,
            n_q_heads=self.n_q_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            exclude_first_n=0,
            exclude_last_n=0,
        )
        # MLP node selection: Top 50% of 4 layers = 2 layers (layers 2 and 0)
        self.assertEqual(res.mlp, {"mlp_2": None, "mlp_0": None})

        # Attention node selection: Top 50% of 16 heads = 8 heads.
        self.assertTrue(len(res.attn) > 0)
        total_heads = _count_selected_heads(res.attn, self.head_dim)
        self.assertEqual(total_heads, 8)

    # -----------------------------------------------------------------------
    # 5. Random Parity Tests
    # -----------------------------------------------------------------------
    def test_random_selection_parity(self):
        circuit = select_components(
            self.head_scores_neuron,
            self.mlp_scores_neuron,
            self.metadata_neuron,
            top_frac=0.3,
            scope="both",
            n_layers=self.n_layers,
            n_q_heads=self.n_q_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
        )

        random_res = random_selection(
            self.head_scores_neuron,
            self.mlp_scores_neuron,
            self.metadata_neuron,
            circuit_result=circuit,
            n_layers=self.n_layers,
            n_q_heads=self.n_q_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            seed=42,
        )

        # Ensure random matches circuit count for attention
        circ_heads = _count_selected_heads(circuit.attn, self.head_dim)
        rand_heads = _count_selected_heads(random_res.attn, self.head_dim)
        self.assertEqual(circ_heads, rand_heads)

        # Ensure random matches circuit count for neurons
        circ_neurons = sum(len(v) for v in circuit.mlp.values() if v is not None)
        rand_neurons = sum(len(v) for v in random_res.mlp.values() if v is not None)
        self.assertEqual(circ_neurons, rand_neurons)

    # -----------------------------------------------------------------------
    # 6. Baseline Selection Tests
    # -----------------------------------------------------------------------
    def test_baseline_selection(self):
        base = build_baseline_selection(
            self.head_scores_node,
            self.mlp_scores_node,
            self.metadata_node,
            scope="both",
            n_layers=self.n_layers,
            exclude_first_n=1,
            exclude_last_n=1,
        )
        # Should only have layers 1 and 2
        self.assertIn("attn_1", base.attn)
        self.assertIn("attn_2", base.attn)
        self.assertNotIn("attn_0", base.attn)

        # Everything should be unmasked (None)
        self.assertIsNone(base.attn["attn_1"]["q"])
        self.assertIsNone(base.mlp["mlp_1"])


if __name__ == "__main__":
    unittest.main()
