"""The paired-text tasks must score the position that predicts the answer.

``ioi`` and ``double_io`` build their prompts from generated token rows. The
cached prompt text has to stop before the answer name: the metric reads
``logits[input_length - 1]``, so a prompt that still ends with the answer scores
the logits at the answer itself. These tests run on a tiny random model with a
word-level tokenizer: CPU, offline, no download.
"""

import re

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from circuitkit.backends.eap.eap_utils import tokenize_batch_pair
from circuitkit.tasks.specs import prompt_text_before_answer

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _words():
    from circuitkit.data.task_data.tasks.double_io import double_io_dataset as dio
    from circuitkit.data.task_data.tasks.ioi import ioi_dataset as ioi

    text = " ".join(
        ioi.NAMES + ioi.PLACES + ioi.OBJECTS + ioi.ABBA_TEMPLATES + ioi.BABA_TEMPLATES
        + dio.NAMES + dio.PLACES + dio.OBJECTS + dio.IO_FILLER_CLAUSES
        + dio.DOUBLE_IO_ABBA_TEMPLATES + dio.DOUBLE_IO_BABA_TEMPLATES
    )  # fmt: skip
    return sorted(set(re.findall(r"\w+|[^\w\s]", text)))


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 1-layer random HookedTransformer whose tokenizer has one token per word."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    vocab = {w: i for i, w in enumerate(SPECIALS + _words())}
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    # Saved and reloaded: TransformerLens needs a tokenizer that knows its own path.
    path = tmp_path_factory.mktemp("tiny-words-tokenizer")
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    ).save_pretrained(path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    cfg = HookedTransformerConfig(
        n_layers=1, n_heads=2, d_model=16, d_head=8, d_mlp=32, n_ctx=64,
        d_vocab=len(vocab), act_fn="gelu", model_name="tiny-words",
    )  # fmt: skip
    torch.manual_seed(0)
    return HookedTransformer(cfg, tokenizer=tokenizer)


def _spec(name):
    if name == "ioi":
        from circuitkit.tasks.builtins.ioi import IOITaskSpec

        return IOITaskSpec()
    if name == "ioi_legacy":
        from circuitkit.tasks.builtins.ioi_legacy import IOITaskSpec

        return IOITaskSpec()
    from circuitkit.tasks.builtins.double_io import DoubleIOTaskSpec

    return DoubleIOTaskSpec()


def _cfg(tmp_path, n=12):
    return {
        "algorithm": "eap",
        "level": "node",
        "batch_size": 4,
        "data_params": {"num_examples": n, "seed": 0, "cache_dir": str(tmp_path)},
    }


@pytest.mark.parametrize("task", ["ioi", "ioi_legacy", "double_io"])
def test_scored_position_predicts_the_answer(task, tiny_model, tmp_path):
    """For every row the scored token is the last prompt word, never the answer."""
    model = tiny_model
    dl = _spec(task).build_dataloader(model, _cfg(tmp_path), "cpu")
    bos, pad = model.tokenizer.bos_token_id, model.tokenizer.pad_token_id
    rows = 0
    for clean, corrupted, label in dl:
        tokens, corrupt_tokens, mask, _, lengths, _ = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=dl.pair_padding_side
        )
        for i, text in enumerate(clean):
            rows += 1
            scored = int(tokens[i, lengths[i] - 1])
            correct = int(label[i][0])
            assert scored != correct, f"{task}: scores the answer itself in {text!r}"
            assert scored not in (bos, pad), f"{task}: scores a special token in {text!r}"
            assert int(corrupt_tokens[i, lengths[i] - 1]) == scored  # same last prompt word
            assert "<bos>" not in text and "<pad>" not in text
            real = tokens[i][mask[i].bool()].tolist()
            assert real.count(bos) == 1, f"{task}: {real.count(bos)} BOS tokens in {text!r}"
            assert model.to_string(torch.tensor([scored])) in ("to", "said")
    assert rows == 12


@pytest.mark.parametrize("task,prefix", [("ioi", "ioi"), ("double_io", "double_io")])
def test_cache_written_before_the_fix_is_not_reused(task, prefix, tiny_model, tmp_path):
    """A cache file under the old name (text ending with the answer) is ignored."""
    spec = _spec(task)
    old = tmp_path / f"{prefix}_tiny-words_12_seed0.csv"
    old.write_text("clean,corrupted,correct_idx,incorrect_idx\nstale Mary,stale John,5,6\n")
    dl = spec.build_dataloader(tiny_model, _cfg(tmp_path), "cpu")
    assert all("stale" not in text for clean, _, _ in dl for text in clean)
    new = tmp_path / f"{prefix}_prompt_tiny-words_12_seed0.csv"
    assert new.exists()
    # The finetuning lookup takes the largest match; it must not see the old file.
    old.write_text(old.read_text() + "x" * 100_000)
    prompts_with_answer, prompts = spec.build_finetuning_dataset(
        tiny_model.tokenizer, "tiny-words", 4, {"cache_dir": str(tmp_path)}, seed=0
    )
    assert len(prompts) == 4 and all("stale" not in p for p in prompts)
    # prompt + answer is the full sentence once, not the answer name twice
    for full, prompt in zip(prompts_with_answer, prompts):
        answer = full[len(prompt) :].strip()
        assert answer and not prompt.rstrip().endswith(answer)


def test_prompt_text_before_answer_drops_bos_answer_and_padding(tiny_model):
    model = tiny_model
    row = model.to_tokens("Then , Mary and John went to the store . John gave a drink to Mary")[0]
    padded = torch.cat([row, torch.full((3,), model.tokenizer.pad_token_id)])
    end = len(row) - 2  # the final "to"
    text = prompt_text_before_answer(model, padded, end)
    assert text.split()[-1] == "to" and text.split()[0] == "Then"
    assert "<" not in text
    # a row generated without BOS keeps its first word
    assert prompt_text_before_answer(model, row[1:], end - 1) == text
    # tensor indices are accepted
    assert prompt_text_before_answer(model, padded, torch.tensor(end)) == text


@pytest.mark.parametrize("bad", [-1, 99])
def test_prompt_text_before_answer_rejects_an_index_outside_the_row(bad, tiny_model):
    row = tiny_model.to_tokens("John gave a drink to Mary")[0]
    with pytest.raises(ValueError, match="outside a row"):
        prompt_text_before_answer(tiny_model, row, bad)
