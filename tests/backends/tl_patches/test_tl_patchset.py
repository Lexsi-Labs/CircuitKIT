"""TransformerLens patch series (``circuitkit/_tl_patches``): Gemma-4, Sarvam-MoE, Cohere, Ministral-3.

``import circuitkit`` applies the series to transformer-lens 3.8.0 in memory (see
``src/circuitkit/_tl_patches/README.md``). Every model is a tiny random-weight model
shaped by the real ``config.json`` in ``configs/`` (fetched from the Hugging Face Hub,
main branch, 2026-09-26). CPU only, no downloads.
"""

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import circuitkit  # noqa: F401  - installs the patch hook before transformer_lens loads
from circuitkit._tl_patches import is_applied

if not is_applied():
    pytest.fail("transformer_lens was loaded without CircuitKIT's patch series.", pytrace=False)

import transformer_lens.loading_from_pretrained as tl_loading  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402
from transformer_lens.components.mlps.moe import MoE  # noqa: E402
from transformer_lens.pretrained.weight_conversions import (  # noqa: E402
    convert_cohere_weights,
    convert_gemma4_weights,
    convert_sarvam_moe_weights,
)

CONFIGS = Path(__file__).parent / "configs"
NO_PROCESSING = dict(
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False,
    refactor_factored_attn_matrices=False,
)


def _config(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------- Gemma-4


def test_gemma4_registry_cfg_matches_config_json():
    tc = _config("gemma-4-31B-it.json")["text_config"]
    full, sliding = (
        tc["rope_parameters"]["full_attention"],
        tc["rope_parameters"]["sliding_attention"],
    )
    cfg = tl_loading.get_pretrained_model_config("google/gemma-4-31B-it")
    expected = {
        "d_model": tc["hidden_size"],
        "n_layers": tc["num_hidden_layers"],
        "n_heads": tc["num_attention_heads"],
        "d_head": tc["global_head_dim"],
        "d_head_local": tc["head_dim"],
        "n_key_value_heads": tc["num_global_key_value_heads"],
        "n_key_value_heads_local": tc["num_key_value_heads"],
        "d_mlp": tc["intermediate_size"],
        "d_vocab": tc["vocab_size"],
        "n_ctx": tc["max_position_embeddings"],
        "eps": tc["rms_norm_eps"],
        "act_fn": tc["hidden_activation"],
        "window_size": tc["sliding_window"],
        "rotary_base": full["rope_theta"],
        "rotary_base_local": sliding["rope_theta"],
        "rotary_dim": tc["global_head_dim"],
        "rotary_dim_local": tc["head_dim"],
        "proportional_rotary_factor": full["partial_rotary_factor"],
        "output_logits_soft_cap": tc["final_logit_softcapping"],
        "attn_scale": 1.0,
    }
    assert {k: getattr(cfg, k) for k in expected} == expected
    assert cfg.attn_types == [
        "local" if t == "sliding_attention" else "global" for t in tc["layer_types"]
    ]
    # Gemma-4 features the port does not model must stay off for this checkpoint.
    assert tc["attention_k_eq_v"] and full["rope_type"] == "proportional"
    assert tc["num_kv_shared_layers"] == 0 and tc["hidden_size_per_layer_input"] == 0
    assert not tc["enable_moe_block"]


def _tiny_gemma4(perturb):
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    raw = _config("gemma-4-31B-it.json")
    d, heads, hd_local, hd_global, kv_local, kv_global, mlp, vocab, n_layers, window = (
        64,
        4,
        16,
        32,
        2,
        1,
        96,
        128,
        6,
        4,
    )
    raw["text_config"].update(
        hidden_size=d,
        num_attention_heads=heads,
        head_dim=hd_local,
        global_head_dim=hd_global,
        num_key_value_heads=kv_local,
        num_global_key_value_heads=kv_global,
        intermediate_size=mlp,
        num_hidden_layers=n_layers,
        layer_types=raw["text_config"]["layer_types"][:n_layers],  # 5 sliding + 1 full
        vocab_size=vocab,
        vocab_size_per_layer_input=vocab,
        sliding_window=window,
        max_position_embeddings=256,
    )
    raw["vision_config"].update(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
    )
    for key in raw:
        if key.endswith(("_token_id", "_token_index")):
            raw[key] = vocab - 1
    raw["eos_token_id"] = 1

    torch.manual_seed(0)
    hf_cfg = Gemma4Config(**raw)
    hf_cfg.text_config._attn_implementation = "eager"
    hf = Gemma4ForConditionalGeneration(hf_cfg).float().eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:  # off-identity norms, so a wrong norm convention shows
                p.copy_(1 + 0.3 * torch.randn_like(p))
        if perturb:
            for layer in hf.model.language_model.layers:
                layer.layer_scalar.uniform_(0.5, 1.5)
            hf.model.language_model.norm.weight.mul_(400)  # logits reach the 30.0 softcap

    real = tl_loading.get_pretrained_model_config("google/gemma-4-31B-it")
    cfg = dataclasses.replace(
        real,
        d_model=d,
        n_heads=heads,
        d_head=hd_global,
        d_head_local=hd_local,
        n_key_value_heads=kv_global,
        n_key_value_heads_local=kv_local,
        d_mlp=mlp,
        n_layers=n_layers,
        attn_types=real.attn_types[:n_layers],
        d_vocab=vocab,
        d_vocab_out=vocab,
        window_size=window,
        n_ctx=64,
        tokenizer_name=None,
        rotary_dim=hd_global,
        rotary_dim_local=hd_local,
    )
    return hf, cfg


@pytest.mark.parametrize("perturb", [False, True], ids=["plain", "layer_scalar+softcap"])
def test_gemma4_hooked_transformer_matches_hf(perturb):
    hf, cfg = _tiny_gemma4(perturb)
    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(convert_gemma4_weights(hf, cfg), **NO_PROCESSING)
    # HookedTransformer moves itself to cfg.device (CUDA when available); compare there.
    hf = hf.to(model.cfg.device)
    tokens = torch.randint(2, cfg.d_vocab - 2, (2, 12), device=model.cfg.device)  # 12 > sliding window of 4
    with torch.no_grad():
        ref = hf(input_ids=tokens).logits
        out = model(tokens)
    assert out.shape == ref.shape
    assert (out - ref).abs().max() < 2e-3
    assert torch.equal(out.argmax(-1), ref.argmax(-1))
    # Per-layer geometry: local layers carry the local head size and KV-head count.
    assert model.blocks[0].attn.cfg.d_head == 16 and model.blocks[0].attn.cfg.n_key_value_heads == 2
    assert model.blocks[5].attn.cfg.d_head == 32 and model.blocks[5].attn.cfg.n_key_value_heads == 1


# ------------------------------------------------------------------------- Sarvam-MoE


def _tiny_sarvam_cfg(monkeypatch, dtype=torch.float32):
    """The patched loader's own sarvam cfg_dict, fed a shrunk config.json offline."""
    from transformers import PretrainedConfig

    raw = _config("sarvam-30b.json")
    raw.update(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        moe_intermediate_size=16,
        intermediate_size=24,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=3,
        vocab_size=100,
        max_position_embeddings=64,
    )

    class OfflineAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return PretrainedConfig(**raw)

    monkeypatch.setattr(tl_loading, "AutoConfig", OfflineAutoConfig)
    cfg = tl_loading.get_pretrained_model_config("sarvamai/sarvam-30b", dtype=dtype)
    return dataclasses.replace(cfg, tokenizer_name=None)


def _lin(out_f, in_f):
    return nn.Linear(in_f, out_f, bias=False)


def _mock_sarvam_hf(cfg):
    """Module tree with SarvamMoEForCausalLM's names and shapes (modeling_sarvam_moe.py)."""

    def mlp(width):
        m = nn.Module()
        m.gate_proj, m.up_proj, m.down_proj = (
            _lin(width, cfg.d_model),
            _lin(width, cfg.d_model),
            _lin(cfg.d_model, width),
        )
        return m

    def norm(n):
        m = nn.Module()
        m.weight = nn.Parameter(1 + 0.3 * torch.randn(n))
        return m

    def layer(i):
        n_q, n_kv = cfg.n_heads * cfg.d_head, cfg.n_key_value_heads * cfg.d_head
        m = nn.Module()
        m.input_layernorm, m.post_attention_layernorm = norm(cfg.d_model), norm(cfg.d_model)
        m.attention = nn.Module()
        m.attention.query_key_value = _lin(n_q + 2 * n_kv, cfg.d_model)
        m.attention.dense = _lin(cfg.d_model, n_q)
        m.attention.query_layernorm, m.attention.key_layernorm = norm(cfg.d_head), norm(cfg.d_head)
        if i < cfg.first_k_dense_replace:
            m.mlp = mlp(cfg.d_mlp_dense)
        else:
            m.mlp = nn.Module()
            m.mlp.gate = nn.Module()
            m.mlp.gate.weight = nn.Parameter(torch.randn(cfg.num_experts, cfg.d_model))
            m.mlp.gate.expert_bias = nn.Parameter(0.5 * torch.randn(cfg.num_experts))
            m.mlp.experts = nn.ModuleList(mlp(cfg.d_mlp) for _ in range(cfg.num_experts))
            m.mlp.shared_experts = mlp(cfg.d_mlp * cfg.num_shared_experts)
        return m

    hf = nn.Module()
    hf.model = nn.Module()
    hf.model.word_embeddings = nn.Embedding(cfg.d_vocab, cfg.d_model)
    hf.model.layers = nn.ModuleList(layer(i) for i in range(cfg.n_layers))
    hf.model.norm = norm(cfg.d_model)
    hf.lm_head = _lin(cfg.d_vocab, cfg.d_model)
    return hf


def test_sarvam_registry_cfg_reads_config_json(monkeypatch):
    cfg = _tiny_sarvam_cfg(monkeypatch)
    raw = _config("sarvam-30b.json")
    assert cfg.original_architecture == "SarvamMoEForCausalLM"
    assert (cfg.moe_router_score_function, cfg.use_expert_bias) == (raw["score_function"], True)
    assert cfg.num_shared_experts == raw["num_shared_experts"]
    assert cfg.routed_scaling_factor == raw["routed_scaling_factor"]
    assert (cfg.n_group, cfg.topk_group) == (raw["n_group"], raw["topk_group"])
    assert cfg.first_k_dense_replace == raw["first_k_dense_replace"]
    assert (cfg.d_mlp, cfg.d_mlp_dense, cfg.num_experts) == (16, 24, 8)
    assert cfg.use_qk_norm and cfg.rotary_base == raw["rope_theta"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sarvam_hooked_transformer_forward(monkeypatch, dtype):
    torch.manual_seed(0)
    cfg = _tiny_sarvam_cfg(monkeypatch, dtype)
    hf = _mock_sarvam_hf(cfg)
    state_dict = convert_sarvam_moe_weights(hf, cfg)
    model = HookedTransformer(cfg)
    expected = {k for k in model.state_dict() if not k.endswith(("IGNORE", "mask", "_sin", "_cos"))}
    assert set(state_dict) == expected  # every checkpoint tensor has exactly one TL home
    model.load_and_process_state_dict(state_dict, **NO_PROCESSING)
    assert type(model.blocks[0].mlp).__name__ == "GatedMLP"  # first_k_dense_replace=1
    assert all(isinstance(b.mlp, MoE) for b in model.blocks[1:])
    assert all(p.dtype == dtype for p in model.parameters())  # MoE honours cfg.dtype
    tokens = torch.randint(0, cfg.d_vocab, (2, 7))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (2, 7, cfg.d_vocab) and torch.isfinite(logits).all()


def _reference_sarvam_moe(moe, x):
    """Per-token port of SarvamMoEGate.forward + SarvamMoESparseMoeBlock (n_group=1)."""
    cfg = moe.cfg
    scores = torch.sigmoid(x.float() @ moe.W_gate.weight.float().T)
    idx = torch.topk(scores + moe.expert_bias.float(), cfg.experts_per_token, dim=-1).indices
    weights = scores.gather(-1, idx)  # the bias picks experts; weights use unbiased scores
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor

    def expert(e, v):
        return e.W_out(F.silu(e.W_gate(v)) * e.W_in(v))

    out = torch.stack(
        [
            sum(w * expert(moe.experts[int(i)], xt) for w, i in zip(wt, it))
            for xt, wt, it in zip(x, weights, idx)
        ]
    )
    return out + expert(moe.shared_experts, x), idx, scores


def test_sarvam_moe_routing_matches_reference(monkeypatch):
    torch.manual_seed(0)
    cfg = _tiny_sarvam_cfg(monkeypatch)
    moe = MoE(cfg).eval()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(std=0.3)
        moe.expert_bias.normal_(std=0.5)
    x = torch.randn(1, 9, cfg.d_model)
    with torch.no_grad():
        out = moe(x)
        ref, idx, scores = _reference_sarvam_moe(moe, x[0])
    assert torch.allclose(out[0], ref, atol=1e-5)
    unbiased = torch.topk(scores, cfg.experts_per_token, dim=-1).indices
    assert not torch.equal(
        idx.sort(-1).values, unbiased.sort(-1).values
    ), "bias never changed routing"


def test_sarvam_moe_component_hooks_decompose_output(monkeypatch):
    torch.manual_seed(0)
    moe = MoE(_tiny_sarvam_cfg(monkeypatch)).eval()
    x = torch.randn(2, 5, moe.cfg.d_model)
    cache = {}
    for name in ("expert_contributions", "routed_out", "shared_out"):
        getattr(moe, f"hook_{name}").add_hook(lambda t, hook, n=name: cache.__setitem__(n, t))
    with torch.no_grad():
        out = moe(x)
    for name in cache:
        getattr(moe, f"hook_{name}").remove_hooks()
    assert cache["expert_contributions"].shape == (2, 5, moe.cfg.experts_per_token, moe.cfg.d_model)
    assert torch.allclose(cache["expert_contributions"].sum(2), cache["routed_out"], atol=1e-5)
    assert torch.allclose(cache["routed_out"] + cache["shared_out"], out, atol=1e-5)


# ---------------------------------------------------------------------- Cohere / Aya


def _tiny_cohere(arch):
    """Tiny random Aya Expanse (``cohere``) / Tiny Aya (``cohere2``) with off-identity norms."""
    import transformers

    shape = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        logit_scale=0.25,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    if arch == "cohere":
        hf_cfg = transformers.CohereConfig(num_hidden_layers=2, **shape)
        model_cls = transformers.CohereForCausalLM
    else:  # 3 sliding-window layers with RoPE, then 1 full-attention layer without
        hf_cfg = transformers.Cohere2Config(
            num_hidden_layers=4, head_dim=16, sliding_window=4, **shape
        )
        model_cls = transformers.Cohere2ForCausalLM
    torch.manual_seed(0)
    hf = model_cls(hf_cfg).float().eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:  # off-identity norms, so a missed ln1/ln2 copy shows
                p.copy_(1 + 0.3 * torch.randn_like(p))
    return hf


@pytest.mark.parametrize("arch", ["cohere", "cohere2"])
def test_cohere_hooked_transformer_matches_hf(arch, tmp_path):
    hf = _tiny_cohere(arch)
    hf.save_pretrained(tmp_path)  # config.json only is needed: exercises the local-dir loader

    cfg = tl_loading.get_pretrained_model_config(str(tmp_path))
    cfg = dataclasses.replace(cfg, tokenizer_name=None, device="cpu")
    assert cfg.original_architecture == type(hf).__name__
    assert cfg.parallel_attn_mlp and cfg.rotary_adjacent_pairs
    if arch == "cohere2":
        assert cfg.attn_types == ["local", "local", "local", "global"] and cfg.window_size == 4

    model = HookedTransformer(cfg)
    model.load_and_process_state_dict(convert_cohere_weights(hf, cfg), **NO_PROCESSING)
    tokens = torch.randint(3, cfg.d_vocab, (2, 12))  # 12 > sliding window of 4
    with torch.no_grad():
        ref = hf(input_ids=tokens).logits
        out = model(tokens)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_cohere2_loads_from_local_dir_via_load_model(tmp_path):
    """ck.load_model on a saved Tiny-Aya-shaped checkpoint dir (e.g. a SafeTune output)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from circuitkit import load_model

    hf = _tiny_cohere("cohere2")
    hf.save_pretrained(tmp_path)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(128)}, unk_token="t0")),
        bos_token="t1",
        eos_token="t2",
        pad_token="t0",
        unk_token="t0",
    ).save_pretrained(tmp_path)

    model = load_model(str(tmp_path), dtype="float32", device="cpu")
    tokens = torch.randint(3, 128, (2, 12))
    with torch.no_grad():  # log-probs: from_pretrained folds LN and centres the unembed
        ref = hf(input_ids=tokens).logits.log_softmax(-1)
        out = model(tokens).log_softmax(-1)
    assert model.cfg.original_architecture == "Cohere2ForCausalLM"
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max()


def test_cohere_registry_names():
    for name in (
        "CohereLabs/aya-expanse-8b",
        "CohereLabs/aya-expanse-32b",
        "CohereLabs/tiny-aya-global",
    ):
        assert name in tl_loading.OFFICIAL_MODEL_NAMES
        assert tl_loading.get_official_model_name(name.split("/")[1]) == name


# ------------------------------------------------------------------------ Ministral-3


def test_ministral3_runs_on_upstream_bridge_without_patches():
    """No patch covers Ministral-3: upstream 3.8.0 supports it only via TransformerBridge."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformer_lens.model_bridge import TransformerBridge
    from transformers import Ministral3Config, Ministral3ForCausalLM, PreTrainedTokenizerFast

    tc = _config("Ministral-3-3B-Instruct-2512.json")["text_config"]
    tc.update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=256,
    )
    hf_cfg = Ministral3Config(**tc)
    hf_cfg.architectures = ["Ministral3ForCausalLM"]
    torch.manual_seed(0)
    hf = Ministral3ForCausalLM(hf_cfg).float().eval()
    vocab = {f"t{i}": i for i in range(128)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="t0")),
        bos_token="t1",
        eos_token="t2",
        pad_token="t0",
        unk_token="t0",
    )
    with tempfile.TemporaryDirectory() as tmp:  # the bridge reloads it by name_or_path
        tokenizer.save_pretrained(tmp)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tmp)
        bridge = TransformerBridge.boot_transformers(
            "mistralai/Ministral-3-3B-Instruct-2512", hf_model=hf, tokenizer=tokenizer
        )
    assert type(bridge.adapter).__name__ == "Ministral3ArchitectureAdapter"
    tokens = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        assert torch.allclose(bridge(tokens), hf(tokens).logits, atol=1e-5)
    assert (
        "mistralai/Ministral-3-3B-Instruct-2512" not in tl_loading.OFFICIAL_MODEL_NAMES
    )  # no HookedTransformer path
