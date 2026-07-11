import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the modules being tested
from circuitkit.applications.selective_finetuning import finetune_utils
from circuitkit.applications.selective_finetuning.selector import SelectionResult


# ---------------------------------------------------------------------------
# MOCK LLAMA-LIKE ARCHITECTURE
# ---------------------------------------------------------------------------
class MockProj(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_feat, in_feat))


class MockAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim):
        super().__init__()
        self.q_proj = MockProj(d_model, n_heads * head_dim)
        self.k_proj = MockProj(d_model, n_heads * head_dim)
        self.v_proj = MockProj(d_model, n_heads * head_dim)
        self.o_proj = MockProj(n_heads * head_dim, d_model)


class MockMLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.down_proj = MockProj(d_mlp, d_model)
        self.c_proj = MockProj(d_mlp, d_model)


class MockLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_mlp):
        super().__init__()
        self.self_attn = MockAttention(d_model, n_heads, head_dim)
        self.mlp = MockMLP(d_model, d_mlp)


class MockLlamaModel(nn.Module):
    def __init__(self, n_layers=2, d_model=16, n_heads=4, head_dim=4, d_mlp=32):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MockLayer(d_model, n_heads, head_dim, d_mlp) for _ in range(n_layers)]
        )

        self.config = MagicMock()
        self.config.tokenizer_padding_side = "right"

    def forward(self, input_ids, attention_mask, labels=None):
        # Dummy forward pass returning a scalar loss to allow .backward()
        out = torch.ones(input_ids.shape[0], input_ids.shape[1], 16, requires_grad=True)
        # Simple loss: sum of all elements, so base gradients are all 1.0
        loss = out.sum() if labels is None else (out * labels.unsqueeze(-1).float()).sum()

        outputs = MagicMock()
        outputs.loss = loss
        return outputs


# ---------------------------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------------------------
class TestWeightAccessors(unittest.TestCase):
    def setUp(self):
        self.model = MockLlamaModel()

    def test_get_q_proj(self):
        weight = finetune_utils.get_q_proj(self.model, 0)
        self.assertIsInstance(weight, nn.Parameter)

    def test_missing_attribute_raises_error(self):
        del self.model.model.layers[0].self_attn.k_proj
        with self.assertRaises(AttributeError):
            finetune_utils.get_k_proj(self.model, 0)

    def test_down_proj_fallback(self):
        del self.model.model.layers[0].mlp.down_proj
        weight = finetune_utils.get_down_proj(self.model, 0)
        self.assertIsInstance(weight, nn.Parameter)


class TestGradientMaskingAndSetup(unittest.TestCase):
    def setUp(self):
        self.model = MockLlamaModel()
        self.device = torch.device("cpu")
        self.selection = SelectionResult(attn={"attn_0": {"q": [0], "o": [0]}}, mlp={"mlp_0": [0]})

    def test_actual_gradient_math(self):
        """Verify that the hook actually zero-outs gradients during backward pass."""
        hooks, trainable, masks = finetune_utils.setup_selective_training(
            self.model, self.selection, self.device
        )

        # Trigger a dummy backward pass
        q_proj = finetune_utils.get_q_proj(self.model, 0)
        loss = q_proj.sum()
        loss.backward()

        # Because we only selected index [0] for q_proj (row mask),
        # Row 0 should have gradient 1.0, and all other rows should be 0.0.
        self.assertTrue(torch.all(q_proj.grad[0, :] == 1.0))
        self.assertTrue(torch.all(q_proj.grad[1:, :] == 0.0))

        for h in hooks:
            h.remove()

    def test_verify_gradient_masking_passes(self):
        """Test the verifier correctly identifies good gradients."""
        hooks, trainable, masks = finetune_utils.setup_selective_training(
            self.model, self.selection, self.device
        )

        # Trigger valid backward pass
        loss = sum(p.sum() for p in trainable)
        loss.backward()

        # verify_gradient_masking logs (not prints) its verdict.
        with self.assertLogs(level="INFO") as cm:
            finetune_utils.verify_gradient_masking(trainable, masks)

        output_str = "\n".join(cm.output)
        self.assertIn("Verification PASSED", output_str)

        for h in hooks:
            h.remove()


class TestDataUtilities(unittest.TestCase):
    def test_dataset_edge_case(self):
        """Test fallback when query_length > real_length (non-padded sequence)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = [
            # full_enc (5 tokens total, only 2 real tokens, rest are padding)
            {"input_ids": torch.ones(1, 5), "attention_mask": torch.tensor([[1, 1, 0, 0, 0]])},
            # query_enc (Wait, somehow query length is calculated as 3)
            {"input_ids": torch.ones(1, 3)},
        ]

        dataset = finetune_utils.LanguageModelingDataset(
            tokenizer=mock_tokenizer, clean_texts=["text"], query_strings=["query"], max_length=5
        )

        # The dataset should clamp query_length down to 2 (real_length)
        self.assertEqual(dataset[0]["query_length"].item(), 2)

    def test_build_finetune_dataloader(self):
        """build_finetune_dataloader delegates to task_spec.build_finetuning_dataset()
        and wraps the (clean_texts, query_strings) pair in a DataLoader."""

        # Real API: build_finetuning_dataset returns two parallel string lists.
        def _build_finetuning_dataset(tokenizer, model_name, n_examples, discovery_cfg, seed):
            return (["The capital of France is Paris"], ["The capital of France is"])

        mock_task = MagicMock()
        mock_task.build_finetuning_dataset.side_effect = _build_finetuning_dataset
        # build_finetune_dataloader resolves the chat-template decision via
        # resolve_chat_template_from_tokenizer, which validates chat_template_mode
        # against VALID_MODES. A bare MagicMock attribute is not a valid mode, so
        # declare an explicit one ("off" => raw text, byte-identical tokenization).
        mock_task.chat_template_mode = "off"

        # LanguageModelingDataset.__getitem__ calls tokenizer(text, ...) twice:
        # once for the padded full text, once for the unpadded query prefix.
        def _tokenize(text, **kwargs):
            if kwargs.get("padding") == "max_length":
                return {
                    "input_ids": torch.ones(1, 5, dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
                }
            return {"input_ids": torch.ones(1, 3, dtype=torch.long)}

        mock_tokenizer = MagicMock(side_effect=_tokenize)

        dl = finetune_utils.build_finetune_dataloader(
            task_spec=mock_task,
            tokenizer=mock_tokenizer,
            model_name="gpt2",
            discovery_cfg={},
            device=torch.device("cpu"),
            n_examples=1,
            max_length=5,
        )

        self.assertIsInstance(dl, DataLoader)
        mock_task.build_finetuning_dataset.assert_called_once()
        batch = next(iter(dl))
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("query_length", batch)
        # query (3 tokens) clamped to real_length (3) — both equal here.
        self.assertEqual(int(batch["query_length"][0]), 3)


class TestFinetuningEndToEnd(unittest.TestCase):
    def test_run_finetuning_padding_logic(self):
        """Ensure labels are properly padded with -100 for cross-entropy ignore."""
        model = MockLlamaModel()
        selection = SelectionResult(attn={"attn_0": {"q": None}}, mlp={})  # Baseline selection

        # 1 batch, sequence length 4, query length 2
        batch = {
            "input_ids": torch.tensor([[5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),  # Last token is padding
            "query_length": torch.tensor([2]),
        }

        # We need to spy on the model's forward pass to see the `labels` argument
        with patch.object(model, "forward", wraps=model.forward) as spy_forward:
            finetune_utils.run_finetuning(
                model=model,
                selection=selection,
                finetune_dataloader=[batch],
                device=torch.device("cpu"),
                n_epochs=1,
            )

            # Extract the labels passed to the model
            kwargs = spy_forward.call_args.kwargs
            labels = kwargs["labels"]

            # The first 2 tokens (query) should be -100.
            # The 3rd token (completion) should be unchanged (7).
            # The 4th token (padding) should be -100.
            self.assertTrue(torch.equal(labels, torch.tensor([[-100, -100, 7, -100]])))


if __name__ == "__main__":
    unittest.main()
