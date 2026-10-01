"""The whole ``ck`` stack in the single TransformerLens 3.8 environment, plus the interop contract.

Tiny random GPT-2, Aya Expanse (``cohere``) and Tiny Aya (``cohere2``) models are built
from config classes and saved to a local folder with a word-level tokenizer, so this
runs on CPU with no downloads and no gated models. Covered: ``Pipeline``,
``ck.discover``, ``ck.prune``, a ROME edit, activation steering, ``ck.export_checkpoint``
(current weights, round-tripped through ``AutoModelForCausalLM``), the SafeTune keys in
``*_scores.json`` and ``lexsi_provenance.json`` lineage.
"""

import json

import pytest
import torch
import transformers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import circuitkit as ck
from circuitkit.provenance import FILENAME

WORDS = (
    "the capital of france is paris spain madrid italy rome germany berlin "
    "cat dog sat ran on in mat park red blue"
).split()
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
VOCAB = {w: i for i, w in enumerate(SPECIALS + WORDS)}
FACTS = [("france", "paris"), ("spain", "madrid"), ("italy", "rome"), ("germany", "berlin")]
PARENT = {
    "schema": "lexsi.provenance/1",
    "library": "safetune",
    "version": "0.2.0",
    "base_model": "CohereLabs/aya-expanse-8b",
    "method": "harden.SafeGrad",
    "inputs": [],
    "params": {},
}


def _tiny(arch, path):
    shape = dict(pad_token_id=0, bos_token_id=1, eos_token_id=2, vocab_size=len(VOCAB))
    if arch == "gpt2":
        cfg = transformers.GPT2Config(n_embd=64, n_layer=2, n_head=4, n_positions=64, n_ctx=64, **shape)
        cls = transformers.GPT2LMHeadModel
    elif arch == "llama":
        cfg = transformers.LlamaConfig(
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            **shape,
        )
        cls = transformers.LlamaForCausalLM
    else:
        shape.update(
            hidden_size=64,
            intermediate_size=96,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            logit_scale=0.25,
        )
        if arch == "cohere":
            cfg, cls = transformers.CohereConfig(num_hidden_layers=2, **shape), transformers.CohereForCausalLM
        else:  # 3 sliding-window layers, then 1 full-attention layer
            cfg = transformers.Cohere2Config(num_hidden_layers=4, head_dim=16, sliding_window=4, **shape)
            cls = transformers.Cohere2ForCausalLM
    torch.manual_seed(0)
    cls(cfg).eval().save_pretrained(path)
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=_word_tokenizer(),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    ).save_pretrained(path)
    (path / FILENAME).write_text(json.dumps(PARENT))
    return str(path)


def _word_tokenizer():
    tok = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return tok


def _task(tmp_path, name):
    from circuitkit.tasks.generic import GenericTaskSpec
    from circuitkit.tasks.registry import register_task

    rows = [
        {
            "prompt": f"the capital of {a} is",
            "answer": x,
            "corrupted_prompt": f"the capital of {b} is",
            "corrupted_answer": y,
        }
        for (a, x), (b, y) in zip(FACTS, FACTS[1:] + FACTS[:1])
    ]
    data = tmp_path / f"{name}.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows * 2))
    register_task(
        GenericTaskSpec.from_jsonl(
            str(data),
            schema={
                "prompt": "prompt",
                "answer": "answer",
                "corrupted_prompt": "corrupted_prompt",
                "corrupted_answer": "corrupted_answer",
            },
            name=name,
            chat_template_mode="off",
        )
    )
    return name


def _down_weight(hf, arch):
    if arch == "gpt2":
        return hf.transformer.h[0].mlp.c_proj.weight  # Conv1D [d_mlp, d_model] == W_out
    return hf.model.layers[0].mlp.down_proj.weight.T  # nn.Linear [d_model, d_mlp]


ARCHS = ["gpt2", "cohere", "cohere2"]


def test_eap_ig_activations_llama_has_nonzero_scores():
    """Activation IG must retain a gradient path through rotary decoder blocks."""
    import tempfile
    from pathlib import Path

    # Keep the local checkpoint path neutral: TransformerLens uses path-name
    # heuristics for Llama/Gemma and rejects ambiguous local directory names.
    with tempfile.TemporaryDirectory(prefix="ck-model-") as directory:
        root = Path(directory)
        src = _tiny("llama", root / "model")
        task = _task(root, "tl3_activation_ig_llama")
        model = ck.load_model(src, dtype="float32", device="cpu")
        circuit = ck.discover(
            model,
            task,
            algorithm="eap-ig-activations",
            n_examples=4,
            batch_size=2,
            ig_steps=2,
            output_path=str(root / "eap-ig-activations.pt"),
        )

    assert circuit.scores
    assert any(abs(float(score)) > 0 for score in circuit.scores.values())


@pytest.mark.parametrize("arch", ARCHS)
def test_export_writes_edited_weights(arch, tmp_path):
    """Edit W_out and W_U, export, reload with HF: the edits are there (audit fix 4)."""
    src = _tiny(arch, tmp_path / "in")
    model = ck.load_model(src, dtype="float32", device="cpu")
    torch.manual_seed(1)
    with torch.no_grad():
        model.blocks[0].mlp.W_out.add_(0.1 * torch.randn_like(model.blocks[0].mlp.W_out))
        model.unembed.W_U.add_(0.01 * torch.randn_like(model.unembed.W_U))  # unties tied heads

    out = ck.export_checkpoint(model, None, str(tmp_path / "out"))

    hf = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    assert torch.equal(_down_weight(hf, arch), model.blocks[0].mlp.W_out)
    tokens = torch.tensor([[1] + [VOCAB[w] for w in "the capital of france is".split()]])
    with torch.no_grad():
        ref = model(tokens).log_softmax(-1)
        got = hf(tokens).logits.log_softmax(-1)
    assert torch.allclose(got, ref, atol=1e-4)

    prov = json.loads((tmp_path / "out" / FILENAME).read_text())
    assert prov["schema"] == "lexsi.provenance/1" and prov["library"] == "circuitkit"
    assert prov["version"] == ck.__version__
    assert prov["params"]["weights"] == "current"
    assert prov["inputs"][0]["ref"] == src and prov["inputs"][0]["provenance"] == PARENT
    assert prov["base_model"] == PARENT["base_model"]


def test_export_of_folded_model_warns_and_keeps_original_weights(tmp_path):
    src = _tiny("gpt2", tmp_path / "in")
    model = ck.load_model(src, dtype="float32", device="cpu", fold_ln=True)
    with pytest.raises(ValueError, match="fold_ln"):
        ck.export_checkpoint(model, None, str(tmp_path / "x"), weights="current")
    with pytest.warns(UserWarning, match="original Hugging Face weights"):
        out = ck.export_checkpoint(model, ["MLP 0"], str(tmp_path / "out"))
    hf = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert not hf.transformer.h[0].mlp.c_proj.weight.any()  # the pruning still lands
    assert json.loads((tmp_path / "out" / FILENAME).read_text())["params"]["weights"] == "original"


@pytest.mark.parametrize("arch", ARCHS)
def test_stack_runs(arch, tmp_path):
    """Pipeline, discover, prune, ROME, steering and export on one tiny model."""
    from circuitkit.applications.editing import RomeHandler
    from circuitkit.applications.steering import ActivationSteering

    src = _tiny(arch, tmp_path / "in")
    task = _task(tmp_path, f"tl3_capitals_{arch}")
    disco = dict(n_examples=4, batch_size=2, ig_steps=2)

    pipe = ck.Pipeline(src, task=task, precision="float32", device="cpu", output_dir=str(tmp_path / "pipe"))
    pipe.discover(**disco).prune(sparsity=0.3)
    pipe.export(str(tmp_path / "pipe_ckpt"))
    assert (tmp_path / "pipe_ckpt" / FILENAME).exists()

    model = ck.load_model(src, dtype="float32", device="cpu")
    circuit = ck.discover(model, task, output_path=str(tmp_path / "c.pt"), **disco)
    assert circuit.scores

    # SafeTune's core/circuit_kit/adapter.py reads these keys.
    blob = json.loads((tmp_path / "c_scores.json").read_text())
    units, sugg = blob["safety_units"], blob["layer_suggestions"]
    assert units["unit_ids"] and set(units["unit_ids"]) <= set(blob["node_scores"])
    prefix = "transformer.h." if arch == "gpt2" else "model.layers."
    assert all(m.startswith(prefix) for m in units["module_names"])
    known = {"c_attn", "c_proj", "c_fc"} if arch == "gpt2" else {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    assert sugg["target_modules"] and set(sugg["target_modules"]) <= known
    assert blob["provenance"]["inputs"][0]["provenance"] == PARENT
    assert ck.load_scores(str(tmp_path / "c.pt")).scores == circuit.scores  # old readers still work

    pruned = ck.prune(model, circuit, sparsity=0.3)

    before = model.blocks[0].mlp.W_out.detach().clone()
    RomeHandler(model).edit_single_fact(
        prompt="the capital of france is",
        subject="france",
        target="rome",
        target_layer=0,
        use_corpus_C=False,
    )
    assert not torch.equal(model.blocks[0].mlp.W_out, before)

    steering = ActivationSteering(model, circuit.scores)
    steering.compute_steering_vector([{"text": "the cat sat on the mat"}], [{"text": "the dog ran in the park"}])
    assert steering.steer("the cat sat on the mat") is not None

    out = ck.export_checkpoint(model, None, str(tmp_path / "edited"))
    hf = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert torch.equal(_down_weight(hf, arch), model.blocks[0].mlp.W_out)  # the ROME edit
    ck.export_checkpoint(pruned, circuit, str(tmp_path / "pruned"))
