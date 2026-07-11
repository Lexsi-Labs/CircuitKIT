"""
Shared pytest fixtures for EAP/EAP-IG backend tests.

All fixtures that touch a real model require CUDA and are marked accordingly.
Tests that only exercise pure-Python / pure-tensor logic run on CPU.
"""

import pytest
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# CUDA guard
# ---------------------------------------------------------------------------
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Minimal model config dict (no real model needed for graph-only tests)
# ---------------------------------------------------------------------------
TINY_CFG = {
    "n_layers": 2,
    "n_heads": 2,
    "d_model": 64,
    "d_mlp": 128,
    "d_head": 32,
    "parallel_attn_mlp": False,
}


@pytest.fixture(scope="session")
def tiny_cfg():
    return dict(TINY_CFG)


# ---------------------------------------------------------------------------
# Real HookedTransformer (only instantiated when CUDA is available)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_model():
    """
    A 2-layer, 2-head GPT-2-style HookedTransformer with all EAP flags set.
    """
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformers import AutoTokenizer

    # 1. Load and configure the tokenizer FIRST
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build config dynamically based on the tokenizer to prevent IndexErrors
    cfg = HookedTransformerConfig(
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_mlp=128,
        d_head=32,
        d_vocab=tokenizer.vocab_size,  # Strictly bind vocab size to tokenizer
        n_ctx=128,
        act_fn="gelu",
        normalization_type="LN",
    )

    # 3. Initialize the raw model
    model = HookedTransformer(cfg)

    # 4. Use the library's internal method to attach and register the tokenizer
    model.set_tokenizer(tokenizer)

    # 5. Override config flags required by attribute / attribute_node
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.n_key_value_heads = None

    if torch.cuda.is_available():
        model = model.cuda()

    return model


# ---------------------------------------------------------------------------
# Graph fixture built from cfg dict (no model required)
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_graph(tiny_cfg):
    from circuitkit.backends.eap.graph import Graph

    return Graph.from_model(tiny_cfg)


@pytest.fixture
def tiny_graph_with_node_scores(tiny_cfg):
    from circuitkit.backends.eap.graph import Graph

    return Graph.from_model(tiny_cfg, node_scores=True)


# ---------------------------------------------------------------------------
# Minimal EAP-format dataloader (raw string lists)
# ---------------------------------------------------------------------------
CLEAN_TEXTS = [
    "The cat sat on the mat",
    "A dog ran in the park",
    "She opened the red door",
    "He read the old book",
]
CORRUPTED_TEXTS = [
    "The cat sat on a hat",
    "A dog ran in the yard",
    "She closed the blue door",
    "He wrote the new book",
]
LABELS = [0, 1, 0, 1]


def _make_raw_dataloader(batch_size=2):
    """Yields (clean_list, corrupted_list, label_tensor) batches."""
    dataset = list(zip(CLEAN_TEXTS, CORRUPTED_TEXTS, LABELS))

    def collate(xs):
        clean, corrupted, labels = zip(*xs)
        return list(clean), list(corrupted), torch.tensor(labels)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate)


@pytest.fixture
def tiny_dataloader():
    return _make_raw_dataloader(batch_size=2)


@pytest.fixture
def single_batch_dataloader():
    """All examples in one batch — exercises batch_size=4 path."""
    return _make_raw_dataloader(batch_size=4)


# ---------------------------------------------------------------------------
# Simple metric: logit difference at last token
# ---------------------------------------------------------------------------
@pytest.fixture
def logit_diff_metric():
    """
    Returns a metric function compatible with the EAP signature:
        metric(logits, clean_logits, input_lengths, labels) -> Tensor[scalar]
    """

    def metric(logits, clean_logits, input_lengths, labels):
        # logits: [batch, pos, vocab]
        last = input_lengths - 1  # index of last real token
        last_logits = logits[torch.arange(len(last)), last]  # [batch, vocab]

        # Change .mean() to .sum() so gradients aren't scaled by batch size
        return last_logits.sum()

    return metric
