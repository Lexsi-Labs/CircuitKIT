"""Unit tests for HF checkpoint export/eval protocol.

Covers the audit fixes:
* head-pruning zeroes the correct ``o_proj`` axis per architecture
  (Llama-style nn.Linear: axis 1 / columns; GPT-2 Conv1D: axis 0 / rows);
* GQA shared K/V heads are NOT zeroed for a single-head node;
* neuron-level MLP pruning honours indices; neuron-level attention raises;
* quantized checkpoints genuinely round-trip (quanto modules present on
  reload) — gated with importorskip.
"""

import pytest
import torch as t

from circuitkit.evaluation.hf_checkpoint import (
    _prune_hf_state_dict,
    expected_zero_keys,
)

# ----------------------------------------------------------------------------
# Architecture configs (mirrors hf_checkpoint.ARCH_CONFIGS minimal subset).
# ----------------------------------------------------------------------------
_LLAMA_CFG = {
    "attn_merged_qkv": False,
    "has_gate": True,
    "prefix_layer": "model.layers.{}",
    "key_attn_q": "self_attn.q_proj",
    "key_attn_k": "self_attn.k_proj",
    "key_attn_v": "self_attn.v_proj",
    "key_attn_o": "self_attn.o_proj",
    "key_mlp_gate": "mlp.gate_proj",
    "key_mlp_up": "mlp.up_proj",
    "key_mlp_down": "mlp.down_proj",
}
_GPT2_CFG = {
    "attn_merged_qkv": True,
    "has_gate": False,
    "prefix_layer": "transformer.h.{}",
    "key_attn_qkv": "attn.c_attn",
    "key_attn_o": "attn.c_proj",
    "key_mlp_in": "mlp.c_fc",
    "key_mlp_down": "mlp.c_proj",
}


def _fake_llama_state_dict(n_heads, n_kv_heads, d_head, d_model, d_mlp, layer=0):
    """Llama-shaped weights, all-ones so a zeroed slice is unambiguous."""
    p = f"model.layers.{layer}"
    return {
        f"{p}.self_attn.q_proj.weight": t.ones(n_heads * d_head, d_model),
        f"{p}.self_attn.k_proj.weight": t.ones(n_kv_heads * d_head, d_model),
        f"{p}.self_attn.v_proj.weight": t.ones(n_kv_heads * d_head, d_model),
        # o_proj is nn.Linear: weight [d_model, n_heads*d_head].
        f"{p}.self_attn.o_proj.weight": t.ones(d_model, n_heads * d_head),
        f"{p}.mlp.gate_proj.weight": t.ones(d_mlp, d_model),
        f"{p}.mlp.up_proj.weight": t.ones(d_mlp, d_model),
        f"{p}.mlp.down_proj.weight": t.ones(d_model, d_mlp),
    }


def _fake_gpt2_state_dict(n_heads, d_head, d_model, d_mlp, layer=0):
    """GPT-2 Conv1D-shaped weights, all-ones."""
    p = f"transformer.h.{layer}"
    return {
        # c_attn Conv1D: weight [d_model, 3*d_model], bias [3*d_model].
        f"{p}.attn.c_attn.weight": t.ones(d_model, 3 * n_heads * d_head),
        f"{p}.attn.c_attn.bias": t.ones(3 * n_heads * d_head),
        # c_proj Conv1D: weight [d_model, d_model].
        f"{p}.attn.c_proj.weight": t.ones(d_model, d_model),
        f"{p}.mlp.c_fc.weight": t.ones(d_model, d_mlp),
        f"{p}.mlp.c_proj.weight": t.ones(d_mlp, d_model),
    }


# ----------------------------------------------------------------------------
# BLOCKER 1 — Llama o_proj per-head slice must be COLUMNS (axis 1).
# ----------------------------------------------------------------------------
def test_llama_head_pruning_zeroes_oproj_columns():
    n_heads, n_kv_heads, d_head, d_model, d_mlp = 8, 8, 4, 32, 64
    sd = _fake_llama_state_dict(n_heads, n_kv_heads, d_head, d_model, d_mlp)
    # Prune head 2 of layer 0.
    _prune_hf_state_dict(sd, ["A0.2"], _LLAMA_CFG, n_heads, d_head, d_model, n_kv_heads)

    o = sd["model.layers.0.self_attn.o_proj.weight"]  # [d_model, n_heads*d_head]
    hs, he = 2 * d_head, 3 * d_head
    # The head-2 COLUMNS must be exactly zero ...
    assert float(o[:, hs:he].abs().sum()) == 0.0, "o_proj columns of head not zeroed"
    # ... and nothing else.
    assert float(o[:, :hs].abs().sum()) > 0.0
    assert float(o[:, he:].abs().sum()) > 0.0
    # The corresponding ROW slice must be UNTOUCHED (the old buggy axis).
    assert float(o[hs:he, :].abs().sum()) > 0.0, "rows zeroed — wrong axis (BLOCKER 1 regression)"

    # Q rows for this head are zeroed (axis 0 of q_proj).
    q = sd["model.layers.0.self_attn.q_proj.weight"]
    assert float(q[hs:he, :].abs().sum()) == 0.0
    assert float(q[:hs, :].abs().sum()) > 0.0


def test_gpt2_head_pruning_zeroes_cproj_rows():
    """GPT-2 Conv1D c_proj: per-head slice is ROWS (axis 0) — must stay correct."""
    n_heads, d_head, d_model, d_mlp = 12, 4, 48, 96
    sd = _fake_gpt2_state_dict(n_heads, d_head, d_model, d_mlp)
    _prune_hf_state_dict(sd, ["A0.3"], _GPT2_CFG, n_heads, d_head, d_model, n_heads)

    cproj = sd["transformer.h.0.attn.c_proj.weight"]  # [d_model, d_model]
    hs, he = 3 * d_head, 4 * d_head
    assert float(cproj[hs:he, :].abs().sum()) == 0.0, "GPT-2 c_proj rows not zeroed"
    assert float(cproj[:hs, :].abs().sum()) > 0.0
    assert float(cproj[he:, :].abs().sum()) > 0.0

    # c_attn Q/K/V columns for head 3 are zeroed.
    c_attn = sd["transformer.h.0.attn.c_attn.weight"]
    d_q = n_heads * d_head
    for blk in (0, d_q, 2 * d_q):
        assert float(c_attn[:, blk + hs : blk + he].abs().sum()) == 0.0


# ----------------------------------------------------------------------------
# SHOULD-FIX 4 — GQA: shared K/V heads must NOT be zeroed for a single head.
# ----------------------------------------------------------------------------
def test_gqa_single_head_does_not_zero_shared_kv():
    n_heads, n_kv_heads, d_head, d_model, d_mlp = 8, 2, 4, 32, 64  # group_size=4
    sd = _fake_llama_state_dict(n_heads, n_kv_heads, d_head, d_model, d_mlp)
    _prune_hf_state_dict(sd, ["A0.1"], _LLAMA_CFG, n_heads, d_head, d_model, n_kv_heads)

    # K and V are SHARED across the Q-head group — must be completely untouched.
    k = sd["model.layers.0.self_attn.k_proj.weight"]
    v = sd["model.layers.0.self_attn.v_proj.weight"]
    assert float(k.abs().sum()) == float(k.numel()), "shared K head wrongly zeroed"
    assert float(v.abs().sum()) == float(v.numel()), "shared V head wrongly zeroed"
    # Q rows + o_proj columns for head 1 are still zeroed.
    q = sd["model.layers.0.self_attn.q_proj.weight"]
    assert float(q[d_head : 2 * d_head, :].abs().sum()) == 0.0


# ----------------------------------------------------------------------------
# SHOULD-FIX 5 — neuron-level MLP pruning honours indices; attention raises.
# ----------------------------------------------------------------------------
def test_neuron_level_mlp_pruning_honours_indices():
    n_heads, n_kv_heads, d_head, d_model, d_mlp = 8, 8, 4, 32, 64
    sd = _fake_llama_state_dict(n_heads, n_kv_heads, d_head, d_model, d_mlp)
    neurons = [1, 5, 9]
    _prune_hf_state_dict(
        sd, {"mlp": {0: neurons}}, _LLAMA_CFG, n_heads, d_head, d_model, n_kv_heads
    )
    gate = sd["model.layers.0.mlp.gate_proj.weight"]  # [d_mlp, d_model]
    down = sd["model.layers.0.mlp.down_proj.weight"]  # [d_model, d_mlp]
    for n in neurons:
        assert float(gate[n, :].abs().sum()) == 0.0
        assert float(down[:, n].abs().sum()) == 0.0
    # Untouched neurons still non-zero (NOT upgraded to full-node wipe).
    assert float(gate[0, :].abs().sum()) > 0.0
    assert float(gate[2, :].abs().sum()) > 0.0
    assert float(gate.abs().sum()) > 0.0


def test_neuron_level_attention_pruning_raises():
    n_heads, n_kv_heads, d_head, d_model, d_mlp = 8, 8, 4, 32, 64
    sd = _fake_llama_state_dict(n_heads, n_kv_heads, d_head, d_model, d_mlp)
    with pytest.raises(NotImplementedError):
        _prune_hf_state_dict(
            sd,
            {"attn": {(0, 0): [1, 2]}},
            _LLAMA_CFG,
            n_heads,
            d_head,
            d_model,
            n_kv_heads,
        )


def test_expected_zero_keys_axis():
    """expected_zero_keys reports the right o_proj axis per architecture."""
    llama = expected_zero_keys(["A0.2"], _LLAMA_CFG)
    o = [e for e in llama if e["kind"] == "o_proj"][0]
    assert o["axis"] == 1, "Llama o_proj expectation must be axis 1 (columns)"

    gpt2 = expected_zero_keys(["A0.2"], _GPT2_CFG)
    o2 = [e for e in gpt2 if e["kind"] == "o_proj"][0]
    assert o2["axis"] == 0, "GPT-2 c_proj expectation must be axis 0 (rows)"


# ----------------------------------------------------------------------------
# BLOCKER 2 — quantized checkpoints genuinely round-trip.
# ----------------------------------------------------------------------------
def test_quantized_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("optimum.quanto")
    pytest.importorskip("transformers")

    from optimum.quanto import qint8, quantize
    from transformers import AutoModelForCausalLM

    from circuitkit.evaluation.hf_checkpoint import (
        _has_quanto_modules,
        is_quantized_checkpoint,
        load_quantized_checkpoint,
        save_quantized_checkpoint,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as e:  # pragma: no cover - offline / no model cache
        pytest.skip(f"gpt2 not available offline: {e}")

    quantize(model, weights=qint8)
    assert _has_quanto_modules(model), "quantize() did not produce quanto modules"

    ckpt = str(tmp_path / "qckpt")
    save_quantized_checkpoint(model, ckpt, tokenizer_name="gpt2", overwrite=True)
    assert is_quantized_checkpoint(ckpt), "quanto qmap marker not written"

    reloaded = load_quantized_checkpoint(ckpt)
    assert _has_quanto_modules(reloaded), (
        "reloaded checkpoint has no quanto modules — quantization did not " "round-trip"
    )

    # A tiny forward pass must run.
    with t.no_grad():
        out = reloaded(input_ids=t.tensor([[101, 102, 103]]))
    assert out.logits.shape[-1] > 0


def test_save_quantized_rejects_unquantized_model(tmp_path):
    pytest.importorskip("optimum.quanto")
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM

    from circuitkit.evaluation.hf_checkpoint import save_quantized_checkpoint

    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"gpt2 not available offline: {e}")

    # Not quantized -> must raise rather than silently write an fp checkpoint.
    with pytest.raises(RuntimeError, match="not quantized"):
        save_quantized_checkpoint(model, str(tmp_path / "bad"), overwrite=True)


# ----------------------------------------------------------------------------
# vLLM backend — a quanto checkpoint is not vLLM-loadable; the protocol routes
# it through a dequantized fp checkpoint instead. Verify that export.
# ----------------------------------------------------------------------------
def test_export_dequantized_checkpoint(tmp_path):
    """A quanto checkpoint dequantizes to a plain fp HF checkpoint with no
    quanto modules, weights == the quanto-dequantized values, and is loadable
    by a stock AutoModelForCausalLM (hence by vLLM)."""
    pytest.importorskip("optimum.quanto")
    pytest.importorskip("transformers")

    from optimum.quanto import QModuleMixin, qint8, quantize
    from transformers import AutoModelForCausalLM

    from circuitkit.evaluation.hf_checkpoint import (
        _has_quanto_modules,
        export_dequantized_checkpoint,
        is_quantized_checkpoint,
        load_quantized_checkpoint,
        save_quantized_checkpoint,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as e:  # pragma: no cover - offline / no model cache
        pytest.skip(f"gpt2 not available offline: {e}")

    quantize(model, weights=qint8)
    qckpt = str(tmp_path / "qckpt")
    save_quantized_checkpoint(model, qckpt, tokenizer_name="gpt2", overwrite=True)

    # Reference: a quanto QModule weight dequantized directly.
    q_reloaded = load_quantized_checkpoint(qckpt)
    ref = {}
    for name, mod in q_reloaded.named_modules():
        if isinstance(mod, QModuleMixin):
            w = mod.weight
            ref[name] = (w.dequantize() if hasattr(w, "dequantize") else w).detach().clone()

    fpckpt = str(tmp_path / "fpckpt")
    export_dequantized_checkpoint(qckpt, fpckpt, overwrite=True)

    # The dequantized checkpoint is a plain HF checkpoint — no quanto marker.
    assert not is_quantized_checkpoint(
        fpckpt
    ), "dequantized checkpoint must not carry the quanto qmap marker"

    fp_model = AutoModelForCausalLM.from_pretrained(fpckpt)
    assert not _has_quanto_modules(
        fp_model
    ), "dequantized checkpoint still has quanto modules — vLLM cannot load it"

    # Weights are the exact dequantized quantized values (quantization error
    # baked in), so vLLM eval numbers match the quanto/HFLM path.
    fp_sd = fp_model.state_dict()
    for name, ref_w in ref.items():
        key = name + ".weight"
        assert key in fp_sd, f"missing dequantized weight {key}"
        assert t.equal(
            fp_sd[key].cpu(), ref_w.cpu()
        ), f"dequantized weight {key} differs from the quanto-dequantized value"

    # A tiny forward pass must run on the plain fp checkpoint.
    with t.no_grad():
        out = fp_model(input_ids=t.tensor([[101, 102, 103]]))
    assert out.logits.shape[-1] > 0


def test_export_dequantized_rejects_non_quanto_checkpoint(tmp_path):
    """export_dequantized_checkpoint must reject a plain (non-quanto)
    checkpoint with a clear error rather than mangling it."""
    from circuitkit.evaluation.hf_checkpoint import export_dequantized_checkpoint

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not a quanto"):
        export_dequantized_checkpoint(str(plain), str(tmp_path / "out"))


def test_tied_head_roundtrips_against_original_model(tmp_path):
    """The reloaded quantized head must match the PRE-SAVE model's values.

    Regression for the tied-embedding roundtrip bug: saving an UNFROZEN
    quantized GPT-2 dropped lm_head.weight from the file (it shared the wte
    tensor), and reload rebuilt the head over uninitialized memory — garbage
    logits, allocator-dependent and silent. The sibling tests couldn't catch
    it because they compared the reload against itself (two wrongs compare
    equal). This test anchors on the original model:
    save now freezes (packing the head as a distinct tensor), so the reloaded
    head's dequantized values must exactly equal the frozen original's.
    """
    pytest.importorskip("optimum.quanto")
    pytest.importorskip("transformers")

    from optimum.quanto import QModuleMixin, qint8, quantize
    from transformers import AutoModelForCausalLM

    from circuitkit.evaluation.hf_checkpoint import (
        load_quantized_checkpoint,
        save_quantized_checkpoint,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"gpt2 not available offline: {e}")

    assert model.config.tie_word_embeddings, "gpt2 must be weight-tied for this test"
    quantize(model, weights=qint8)  # deliberately NOT frozen — save must handle it

    ckpt = str(tmp_path / "q")
    save_quantized_checkpoint(model, ckpt, tokenizer_name="gpt2", overwrite=True)

    # save froze the model in place; capture the ground-truth head values.
    w = model.lm_head.weight
    original_head = (w.dequantize() if hasattr(w, "dequantize") else w).detach().clone()

    reloaded = load_quantized_checkpoint(ckpt)
    rw = reloaded.lm_head.weight
    reloaded_head = (rw.dequantize() if hasattr(rw, "dequantize") else rw).detach()

    assert t.equal(reloaded_head.cpu(), original_head.cpu()), (
        "reloaded quantized head differs from the pre-save model — the tied "
        "head did not genuinely round-trip"
    )
    # And it must not be the all-zeros uninitialized signature.
    assert reloaded_head.abs().sum() > 0
