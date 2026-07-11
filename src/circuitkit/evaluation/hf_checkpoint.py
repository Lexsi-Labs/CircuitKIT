"""Save pruned HookedTransformer as a HuggingFace checkpoint for lm-eval.

Applies the pruning artifact directly to HF model weights (bypassing
TransformerLens preprocessing), then saves as a standard HF checkpoint
loadable by ``lm_eval.models.huggingface.HFLM`` or any HF tool.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Union

import torch as t
from safetensors.torch import save_file as safe_save
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def _hf_weight_key(arch_cfg: dict, layer_idx: int, param: str) -> str:
    """Build the HF state-dict key for a parameter in a given layer."""
    prefix = arch_cfg["prefix_layer"].format(layer_idx)
    key_map = {
        "ln1_weight": f"{prefix}.{arch_cfg['key_ln1']}.weight",
        "ln1_bias": f"{prefix}.{arch_cfg['key_ln1']}.bias",
        "ln2_weight": f"{prefix}.{arch_cfg['key_ln2']}.weight",
        "ln2_bias": f"{prefix}.{arch_cfg['key_ln2']}.bias",
    }

    if arch_cfg.get("attn_merged_qkv"):
        key_map["attn_qkv_weight"] = f"{prefix}.{arch_cfg['key_attn_qkv']}.weight"
        key_map["attn_qkv_bias"] = f"{prefix}.{arch_cfg['key_attn_qkv']}.bias"
    else:
        key_map["attn_q_weight"] = f"{prefix}.{arch_cfg['key_attn_q']}.weight"
        key_map["attn_k_weight"] = f"{prefix}.{arch_cfg['key_attn_k']}.weight"
        key_map["attn_v_weight"] = f"{prefix}.{arch_cfg['key_attn_v']}.weight"
        key_map["attn_q_bias"] = f"{prefix}.{arch_cfg['key_attn_q']}.bias"
        key_map["attn_k_bias"] = f"{prefix}.{arch_cfg['key_attn_k']}.bias"
        key_map["attn_v_bias"] = f"{prefix}.{arch_cfg['key_attn_v']}.bias"

    key_map["attn_o_weight"] = f"{prefix}.{arch_cfg['key_attn_o']}.weight"
    key_map["attn_o_bias"] = f"{prefix}.{arch_cfg['key_attn_o']}.bias"

    if arch_cfg.get("has_gate"):
        key_map["mlp_gate_weight"] = f"{prefix}.{arch_cfg['key_mlp_gate']}.weight"
        key_map["mlp_up_weight"] = f"{prefix}.{arch_cfg['key_mlp_up']}.weight"
    else:
        key_map["mlp_in_weight"] = f"{prefix}.{arch_cfg['key_mlp_in']}.weight"
        key_map["mlp_in_bias"] = f"{prefix}.{arch_cfg['key_mlp_in']}.bias"

    key_map["mlp_down_weight"] = f"{prefix}.{arch_cfg['key_mlp_down']}.weight"
    key_map["mlp_down_bias"] = f"{prefix}.{arch_cfg['key_mlp_down']}.bias"

    return key_map.get(param, param)


# Architecture-specific configs: maps TL's original_architecture → HF weight keys
ARCH_CONFIGS: dict = {
    "GPT2LMHeadModel": {
        "type": "gpt2",
        "attn_merged_qkv": True,
        "has_ln_bias": True,
        "has_mlp_bias": True,
        "has_gate": False,
        "has_pos_embed": True,
        "prefix_layer": "transformer.h.{}",
        "key_attn_qkv": "attn.c_attn",
        "key_attn_o": "attn.c_proj",
        "key_mlp_in": "mlp.c_fc",
        "key_mlp_down": "mlp.c_proj",
        "key_ln1": "ln_1",
        "key_ln2": "ln_2",
        "prefix_embed": "transformer.wte.weight",
        "prefix_final_norm": "transformer.ln_f",
        "prefix_lm_head": "lm_head.weight",
    },
    "LlamaForCausalLM": {
        "type": "llama",
        "attn_merged_qkv": False,
        "has_ln_bias": False,
        "has_mlp_bias": False,
        "has_gate": True,
        "has_pos_embed": False,
        "prefix_layer": "model.layers.{}",
        "key_attn_q": "self_attn.q_proj",
        "key_attn_k": "self_attn.k_proj",
        "key_attn_v": "self_attn.v_proj",
        "key_attn_o": "self_attn.o_proj",
        "key_mlp_gate": "mlp.gate_proj",
        "key_mlp_up": "mlp.up_proj",
        "key_mlp_down": "mlp.down_proj",
        "key_ln1": "input_layernorm",
        "key_ln2": "post_attention_layernorm",
        "prefix_embed": "model.embed_tokens.weight",
        "prefix_final_norm": "model.norm",
        "prefix_lm_head": "lm_head.weight",
    },
    "GemmaForCausalLM": {
        "type": "gemma",
        "attn_merged_qkv": False,
        "has_ln_bias": False,
        "has_mlp_bias": False,
        "has_gate": True,
        "has_pos_embed": False,
        "prefix_layer": "model.layers.{}",
        "key_attn_q": "self_attn.q_proj",
        "key_attn_k": "self_attn.k_proj",
        "key_attn_v": "self_attn.v_proj",
        "key_attn_o": "self_attn.o_proj",
        "key_mlp_gate": "mlp.gate_proj",
        "key_mlp_up": "mlp.up_proj",
        "key_mlp_down": "mlp.down_proj",
        "key_ln1": "input_layernorm",
        "key_ln2": "post_attention_layernorm",
        "prefix_embed": "model.embed_tokens.weight",
        "prefix_final_norm": "model.norm",
        "prefix_lm_head": "lm_head.weight",
    },
}


# Architectures that use the same key naming as Llama
_LLAMA_LIKE = {
    "MistralForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "StableLmForCausalLM",
    "CohereForCausalLM",
    "OlmoForCausalLM",
    "FalconForCausalLM",
}


def _load_causal_lm(model_name: str, dtype: Any = None) -> Any:
    """Load an HF causal LM, preferring the text-only class for multimodal repos.

    Multimodal models such as Gemma 3 resolve under ``AutoModelForCausalLM`` to
    a conditional-generation class whose state dict nests the text weights under
    ``model.language_model.`` and bundles a SigLIP vision tower. The text-only
    pruning / export path keys into a flat ``model.layers.*`` layout and cannot
    handle that (it would build ``model.vision_tower.…`` keys and ``KeyError``).
    When the repo config exposes a ``text_config`` we load the matching
    text-only ``*ForCausalLM`` class instead, yielding a plain text checkpoint
    (no vision tower) that prunes, serves and benchmarks cleanly.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    kw: Dict[str, Any] = {} if dtype is None else {"torch_dtype": dtype}
    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:  # noqa: BLE001 - let AutoModel surface the real error
        return AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if getattr(cfg, "text_config", None) is not None:
        import transformers as _tf

        # Multimodal config -> use the text-only causal-LM class for this family.
        text_cls = {"gemma3": "Gemma3ForCausalLM"}.get(getattr(cfg, "model_type", ""))
        if text_cls and hasattr(_tf, text_cls):
            return getattr(_tf, text_cls).from_pretrained(model_name, **kw)
    return AutoModelForCausalLM.from_pretrained(model_name, **kw)


def _resolve_arch(original_architecture: str, model_name: str) -> dict:
    """Get architecture config, auto-detecting for unknown architectures."""
    if original_architecture in ARCH_CONFIGS:
        return ARCH_CONFIGS[original_architecture]

    if original_architecture in _LLAMA_LIKE:
        return ARCH_CONFIGS["LlamaForCausalLM"]

    # Fallback: load HF model and auto-detect key naming. Use the text-only
    # loader so a multimodal repo (Gemma 3) does not contaminate the detected
    # layer prefix with vision-tower keys.
    hf_model = _load_causal_lm(model_name)
    sd = hf_model.state_dict()

    # Try to find the layer prefix
    cfg = dict(ARCH_CONFIGS["LlamaForCausalLM"])  # start with Llama defaults
    first_layer_key = [k for k in sd if ".0." in k]
    if first_layer_key:
        key = first_layer_key[0]
        prefix = key.split(".0.")[0] + ".{}"  # e.g. "model.layers.{}" or "transformer.h.{}"
        cfg["prefix_layer"] = prefix

    # Detect merged QKV (GPT-2 style)
    sample_layer_key = cfg["prefix_layer"].format(0)
    merged_keys = [k for k in sd if sample_layer_key in k and "c_attn" in k]
    if merged_keys:
        cfg["attn_merged_qkv"] = True
        cfg["key_attn_qkv"] = "attn.c_attn"
        cfg["key_attn_o"] = "attn.c_proj"
        cfg["key_mlp_in"] = "mlp.c_fc"
        cfg["key_mlp_down"] = "mlp.c_proj"
        cfg["key_ln1"] = "ln_1"
        cfg["key_ln2"] = "ln_2"
        cfg["has_gate"] = False
        cfg["prefix_embed"] = f"{prefix.split('.')[0]}.wte.weight"
        cfg["prefix_final_norm"] = f"{prefix.split('.')[0]}.ln_f"
        return cfg

    # Detect GPT-NeoX style
    neox_keys = [k for k in sd if sample_layer_key in k and "attention.query_key_value" in k]
    if neox_keys:
        cfg["attn_merged_qkv"] = True
        cfg["key_attn_qkv"] = "attention.query_key_value"
        cfg["key_attn_o"] = "attention.dense"
        cfg["key_mlp_in"] = "mlp.dense_h_to_4h"
        cfg["key_mlp_down"] = "mlp.dense_4h_to_h"
        cfg["key_ln1"] = "input_layernorm"
        cfg["key_ln2"] = "post_attention_layernorm"
        cfg["has_gate"] = False
        return cfg

    # Detect separate QKV (Llama-style)
    has_separate = [
        k
        for k in sd
        if "self_attn.q_proj" in k or "self_attn.k_proj" in k or "self_attn.v_proj" in k
    ]
    if has_separate:
        cfg["attn_merged_qkv"] = False
        if any("gate_proj" in k for k in sd if ".0." in k):
            cfg["has_gate"] = True
        else:
            cfg["has_gate"] = False
            cfg["key_mlp_up"] = cfg.get("key_mlp_in", "mlp.up_proj")
            cfg.pop("key_mlp_gate", None)
        return cfg

    # Last resort: return Llama defaults (most modern models follow this)
    return cfg


def expected_zero_keys(
    pruned_artifact: Union[List[str], Dict[str, Any]],
    arch_cfg: dict,
) -> List[Dict[str, Any]]:
    """Derive the HF state-dict slices a node-level pruning should drive to zero.

    Returns a list of expectation records, each:
        {"key": <state-dict key>, "axis": <0|1|None>, "start": int, "stop": int}
    ``axis=None`` means the whole tensor is expected zero.  This is consumed by
    ``verify_checkpoint_weights`` to assert the *specific* per-head o_proj
    sub-slice (and Q rows / MLP tensors) actually persisted.

    Only node-level (list) artifacts are supported here; neuron-level dicts
    zero arbitrary index sets and are verified by the caller differently.
    """
    head_pat = re.compile(r"A(\d+)\.(\d+)")
    mlp_pat = re.compile(r"MLP (\d+)")
    out: List[Dict[str, Any]] = []
    if not isinstance(pruned_artifact, list):
        return out

    merged = bool(arch_cfg.get("attn_merged_qkv"))
    for node in pruned_artifact:
        m_head = head_pat.match(node)
        if m_head:
            layer = int(m_head.group(1))
            key = arch_cfg["prefix_layer"].format(layer)
            o_key = f"{key}.{arch_cfg['key_attn_o']}.weight"
            # GPT-2 Conv1D c_proj weight is [d_model_in, d_model_out] so a
            # per-head slice lives on axis 0; Llama-style nn.Linear o_proj
            # weight is [d_model, n_heads*d_head] so a per-head slice is axis 1.
            o_axis = 0 if merged else 1
            # start/stop are filled by verify (needs d_head); record axis only.
            out.append({"key": o_key, "axis": o_axis, "node": node, "kind": "o_proj"})
            continue
        m_mlp = mlp_pat.match(node)
        if m_mlp:
            layer = int(m_mlp.group(1))
            key = arch_cfg["prefix_layer"].format(layer)
            if arch_cfg.get("has_gate"):
                out.append(
                    {
                        "key": f"{key}.{arch_cfg['key_mlp_gate']}.weight",
                        "axis": None,
                        "node": node,
                        "kind": "mlp",
                    }
                )
                out.append(
                    {
                        "key": f"{key}.{arch_cfg['key_mlp_up']}.weight",
                        "axis": None,
                        "node": node,
                        "kind": "mlp",
                    }
                )
            else:
                out.append(
                    {
                        "key": f"{key}.{arch_cfg['key_mlp_in']}.weight",
                        "axis": None,
                        "node": node,
                        "kind": "mlp",
                    }
                )
            out.append(
                {
                    "key": f"{key}.{arch_cfg['key_mlp_down']}.weight",
                    "axis": None,
                    "node": node,
                    "kind": "mlp",
                }
            )
    return out


def _prune_hf_state_dict(
    state_dict: Dict[str, t.Tensor],
    pruned_artifact: Union[List[str], Dict[str, Any]],
    arch_cfg: dict,
    n_heads: int,
    d_head: int,
    d_model: int,
    n_kv_heads: int,
) -> None:
    """Zero HF weights in-place according to the pruning artifact.

    Node-level artifact (list of ``"A<l>.<h>"`` / ``"MLP <l>"``) zeros a whole
    head or MLP.  Neuron-level artifact (dict) zeros only the listed neuron
    indices of the MLP up/gate/down matrices; neuron-level *attention* indices
    are not weight-decomposable and raise ``NotImplementedError``.
    """
    import logging

    _log = logging.getLogger("circuitkit.evaluation.hf_checkpoint")
    head_pat = re.compile(r"A(\d+)\.(\d+)")
    mlp_pat = re.compile(r"MLP (\d+)")

    # group_size: how many Q heads share one K/V head under GQA. The KV head
    # owning a Q head is `head // group_size` (NOT `head % n_kv_heads`).
    group_size = max(1, n_heads // max(1, n_kv_heads))

    def _zero_head(layer: int, head: int) -> None:
        """Zero one attention head's Q rows + o_proj columns (the head's own
        contribution).  K/V are SHARED across a Q-head group under GQA, so a
        single-head node must NOT zero them — doing so would silently kill
        every other Q head in the same group."""
        key = arch_cfg["prefix_layer"].format(layer)
        hs = head * d_head
        he = hs + d_head

        if arch_cfg.get("attn_merged_qkv"):
            # GPT-2: c_attn is a Conv1D, weight [d_model, 3*d_model], the Q/K/V
            # blocks are columns. c_proj is a Conv1D, weight [d_model, d_model]
            # — a per-head slice of its *input* is on axis 0.
            d_q = n_heads * d_head
            qkv_w = state_dict[f"{key}.{arch_cfg['key_attn_qkv']}.weight"]
            qkv_b = state_dict.get(f"{key}.{arch_cfg['key_attn_qkv']}.bias")
            # GPT-2 MHA: n_kv_heads == n_heads, so K/V slices are this head's
            # own and zeroing them is correct (no group sharing).
            qkv_w[:, hs:he] = 0
            qkv_w[:, d_q + hs : d_q + he] = 0
            qkv_w[:, 2 * d_q + hs : 2 * d_q + he] = 0
            if qkv_b is not None:
                qkv_b[hs:he] = 0
                qkv_b[d_q + hs : d_q + he] = 0
                qkv_b[2 * d_q + hs : 2 * d_q + he] = 0
            # o_proj (c_proj) per-head slice is on axis 0 for Conv1D.
            o_w = state_dict[f"{key}.{arch_cfg['key_attn_o']}.weight"]
            o_w[hs:he, :] = 0
        else:
            # Llama / Mistral / Qwen / Gemma: q_proj is nn.Linear with weight
            # [n_heads*d_head, d_model] -> per-head slice is rows (axis 0).
            q_w = state_dict[f"{key}.{arch_cfg['key_attn_q']}.weight"]
            q_w[hs:he, :] = 0
            q_b_key = f"{key}.{arch_cfg['key_attn_q']}.bias"
            if q_b_key in state_dict:
                state_dict[q_b_key][hs:he] = 0

            # K/V (k_proj/v_proj) weight is [n_kv_heads*d_head, d_model]. Under
            # GQA each K/V head is shared by `group_size` Q heads. Zeroing the
            # shared K/V head for a single Q-head node is WRONG (it would zero
            # the other Q heads in the group too) — so we deliberately leave
            # K/V untouched. The Q rows + o_proj columns fully remove this
            # head's read/write path. (Mapping for reference:
            # kv_head = head // group_size.)

            # o_proj is nn.Linear with weight [d_model, n_heads*d_head] -> the
            # per-head slice is COLUMNS (axis 1). This differs from GPT-2's
            # Conv1D c_proj (axis 0) above.
            o_w = state_dict[f"{key}.{arch_cfg['key_attn_o']}.weight"]
            o_w[:, hs:he] = 0
            if group_size > 1:
                _log.debug(
                    "layer %d head %d: GQA group_size=%d, shared KV head=%d "
                    "left intact (Q rows + o_proj cols zeroed)",
                    layer,
                    head,
                    group_size,
                    head // group_size,
                )
        # o_proj bias is not per-head — leave it.

    def _zero_mlp_full(layer: int) -> None:
        key = arch_cfg["prefix_layer"].format(layer)
        if arch_cfg.get("has_gate"):
            state_dict[f"{key}.{arch_cfg['key_mlp_gate']}.weight"].zero_()
            state_dict[f"{key}.{arch_cfg['key_mlp_up']}.weight"].zero_()
        else:
            state_dict[f"{key}.{arch_cfg['key_mlp_in']}.weight"].zero_()
            b_key = f"{key}.{arch_cfg['key_mlp_in']}.bias"
            if b_key in state_dict:
                state_dict[b_key].zero_()
        state_dict[f"{key}.{arch_cfg['key_mlp_down']}.weight"].zero_()
        b_key = f"{key}.{arch_cfg['key_mlp_down']}.bias"
        if b_key in state_dict:
            state_dict[b_key].zero_()

    def _zero_mlp_neurons(layer: int, neurons: List[int]) -> None:
        """Zero only the listed MLP intermediate neurons (rows of up/gate,
        columns of down). This honours neuron-level pruning instead of
        upgrading it to a full-MLP wipe."""
        key = arch_cfg["prefix_layer"].format(layer)
        idx = t.as_tensor(sorted(set(int(n) for n in neurons)), dtype=t.long)
        if idx.numel() == 0:
            return
        if arch_cfg.get("has_gate"):
            # gate_proj / up_proj: nn.Linear [d_mlp, d_model] -> neuron = row.
            state_dict[f"{key}.{arch_cfg['key_mlp_gate']}.weight"][idx, :] = 0
            state_dict[f"{key}.{arch_cfg['key_mlp_up']}.weight"][idx, :] = 0
        else:
            in_w = state_dict[f"{key}.{arch_cfg['key_mlp_in']}.weight"]
            in_b_key = f"{key}.{arch_cfg['key_mlp_in']}.bias"
            if arch_cfg.get("attn_merged_qkv"):
                # GPT-2 mlp.c_fc is a Conv1D, weight [d_model, d_mlp] ->
                # neuron = column.
                in_w[:, idx] = 0
            else:
                in_w[idx, :] = 0
            if in_b_key in state_dict:
                state_dict[in_b_key][idx] = 0
        # down_proj / c_proj: maps d_mlp -> d_model. For nn.Linear weight is
        # [d_model, d_mlp] (neuron = column); for GPT-2 Conv1D c_proj weight is
        # [d_mlp, d_model] (neuron = row).
        down_w = state_dict[f"{key}.{arch_cfg['key_mlp_down']}.weight"]
        if arch_cfg.get("attn_merged_qkv"):
            down_w[idx, :] = 0
        else:
            down_w[:, idx] = 0
        # down_proj bias is per-d_model output, not per-neuron — leave it.

    if isinstance(pruned_artifact, list):
        for node in pruned_artifact:
            m_head = head_pat.match(node)
            if m_head:
                _zero_head(int(m_head.group(1)), int(m_head.group(2)))
                continue
            m_mlp = mlp_pat.match(node)
            if m_mlp:
                _zero_mlp_full(int(m_mlp.group(1)))
                continue
    elif isinstance(pruned_artifact, dict):
        # Neuron-level MLP pruning: honour the indices.
        for layer_idx, neurons in pruned_artifact.get("mlp", {}).items():
            _zero_mlp_neurons(int(layer_idx), list(neurons))
        # Neuron-level attention pruning: the "neurons" of an attention head
        # are not a weight axis (they are per-position attention outputs), so
        # there is no sound weight-zeroing for an attention neuron subset.
        attn = pruned_artifact.get("attn", {})
        if attn:
            raise NotImplementedError(
                "Neuron-level (index-subset) attention pruning cannot be "
                "expressed as a static weight mask: attention 'neurons' are "
                "per-token attention outputs, not a fixed weight axis. Pass "
                "attention nodes as full heads (node list 'A<l>.<h>') instead. "
                f"Offending entries: {sorted(attn.keys())}"
            )
    else:
        raise TypeError(f"Unsupported pruned_artifact type: {type(pruned_artifact)}")


def _copy_processor_configs(repo_id: str, output_path: str) -> None:
    """Copy multimodal processor/preprocessor configs into the checkpoint dir.

    Multimodal models (e.g. Gemma 3) keep a conditional-generation
    architecture in ``config.json`` even when only their text tower is used.
    A server such as vLLM then tries to load an image processor from the
    checkpoint and aborts if ``preprocessor_config.json`` is missing —
    failing the benchmark of an otherwise-correct text checkpoint. These
    files are tiny, so copy them best-effort. Text-only models (Llama, …)
    have no such file and are skipped silently.
    """
    if not repo_id:
        return
    import shutil

    try:
        from huggingface_hub import hf_hub_download
    except Exception:  # noqa: BLE001 - huggingface_hub always present in practice
        return
    for fname in ("preprocessor_config.json", "processor_config.json"):
        dst = os.path.join(output_path, fname)
        if os.path.exists(dst):
            continue
        try:
            src = hf_hub_download(repo_id, fname)
            shutil.copy(src, dst)
        except Exception:  # noqa: BLE001 - text-only repos have no such file
            pass


def save_pruned_checkpoint(
    model: HookedTransformer,
    pruned_artifact: Union[List[str], Dict[str, Any]],
    output_path: str,
    *,
    overwrite: bool = False,
) -> str:
    """Apply pruning, convert to HF format, save safetensors + config.

    The HF checkpoint is a real ``AutoModelForCausalLM``-compatible directory,
    loadable by ``HFLM(pretrained=output_path, tokenizer=...)`` or any HF
    inference tool.

    Note on TransformerLens preprocessing: TL folds LayerNorm into attention
    weights (``fold_ln``) by default, which alters weight values. This function
    loads the **original** HuggingFace weights (identical to what
    ``HookedTransformer.from_pretrained_no_processing`` returns) and prunes
    those directly. The result is a checkpoint with the same components
    (heads / MLPs) zeroed, evaluated at the HF level. For exact parity with
    TL's pruned model, use ``from_pretrained_no_processing`` in your discovery
    pipeline as well, or compare faithfulness deltas (which are directionally
    consistent).

    Args:
        model: The HookedTransformer model. Used only for architecture config
            (``model.cfg.original_architecture``) and tokenizer name.
        pruned_artifact: Node-level ``["A0.0", "MLP 1", ...]`` or neuron-level
            ``{"mlp": {0: [1,2,3]}, "attn": {(0,0): [4,5,6]}}``.
        output_path: Directory to write the checkpoint into.
        overwrite: If True, remove ``output_path`` first if it exists.

    Returns:
        ``output_path`` for chaining.
    """
    if os.path.exists(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to remove it."
            )
    os.makedirs(output_path, exist_ok=True)

    arch = model.cfg.original_architecture
    # TransformerLens stores ``cfg.model_name`` as the *short alias*
    # (e.g. "Llama-3.2-1B-Instruct"), which AutoModelForCausalLM cannot resolve
    # — especially offline. ``cfg.tokenizer_name`` carries the full HF repo id
    # (e.g. "meta-llama/Llama-3.2-1B-Instruct"); prefer it for HF loads.
    hf_repo_id = getattr(model.cfg, "tokenizer_name", None) or model.cfg.model_name
    arch_cfg = _resolve_arch(arch, hf_repo_id)

    # Load the original HF model and its state dict. Preserve the model's
    # dtype: a bf16 HookedTransformer must export a bf16 checkpoint, otherwise
    # save_pretrained writes fp32 and reload silently up-casts (4x size, and a
    # numeric mismatch vs the discovery model).
    model_dtype = getattr(model.cfg, "dtype", None)
    # Text-only loader: a multimodal repo (Gemma 3) is loaded via its text-only
    # *ForCausalLM class so the state dict is a flat model.layers.* layout with
    # no vision tower — exactly what the pruning surgery below expects.
    hf_model = _load_causal_lm(hf_repo_id, dtype=model_dtype)
    state_dict = dict(hf_model.state_dict())

    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_model = model.cfg.d_model
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", n_heads) or n_heads

    _prune_hf_state_dict(
        state_dict, pruned_artifact, arch_cfg, n_heads, d_head, d_model, n_kv_heads
    )

    # Save config, recording the export dtype so the checkpoint round-trips
    # at the same precision as the discovery model. Use the loaded model's own
    # config (the text-only config for a multimodal repo) so it matches the
    # weights actually written — no stray vision_config / multimodal arch.
    config = hf_model.config
    # A text-only sub-config lifted from a multimodal repo (Gemma3TextConfig)
    # carries no ``architectures`` field, and this path writes config.json
    # directly rather than via model.save_pretrained — so set it explicitly
    # or vLLM's ModelConfig rejects the checkpoint ("No model architectures").
    if not getattr(config, "architectures", None):
        config.architectures = [type(hf_model).__name__]
    if model_dtype is not None:
        # str(torch.bfloat16) == "torch.bfloat16"; HF expects the bare name.
        dtype_str = str(model_dtype).replace("torch.", "")
        # transformers >= 4.57 serializes the field as ``dtype``; older
        # versions use ``torch_dtype``. Set both so the checkpoint records the
        # export precision regardless of the installed transformers version.
        config.dtype = dtype_str
        config.torch_dtype = dtype_str
    config.save_pretrained(output_path)

    # Clone any shared-weight aliases (GPT-2 ties embed ↔ lm_head)
    deduped = {}
    seen_data = {}
    for k, v in state_dict.items():
        data_ptr = v.data_ptr()
        if data_ptr in seen_data:
            deduped[k] = v.clone()
        else:
            seen_data[data_ptr] = k
            deduped[k] = v

    # Save weights
    safe_save(deduped, os.path.join(output_path, "model.safetensors"))

    # Save the tokenizer so the checkpoint is self-contained for lm-eval / HF.
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(hf_repo_id).save_pretrained(output_path)
    except Exception:  # noqa: BLE001 - tokenizer is best-effort
        pass

    # Multimodal-derived checkpoints (Gemma 3, …) need their processor configs
    # so a downstream server (vLLM) can load them; no-op for text-only models.
    _copy_processor_configs(hf_repo_id, output_path)

    return output_path


# Marker file written into a quantized checkpoint dir so the matching
# reload path (load_quantized_checkpoint) can be auto-detected.
_QUANTO_MARKER = "quanto_qmap.json"


def _has_quanto_modules(model: Any) -> bool:
    """True iff the model currently contains any optimum-quanto quantized
    module (``QModuleMixin`` — QLinear / QConv2d / ...)."""
    try:
        from optimum.quanto import QModuleMixin
    except ImportError:
        return False
    return any(isinstance(m, QModuleMixin) for m in model.modules())


def _has_compressed_tensors_modules(model: Any) -> bool:
    """True iff the model carries ``compressed-tensors`` quantized modules
    (produced by the llm-compressor backend — modules tagged with a
    ``quantization_scheme``)."""
    for mod in model.modules():
        if getattr(mod, "quantization_scheme", None) is not None:
            return True
    return False


def save_compressed_tensors_checkpoint(
    hf_model: Any,
    output_path: str,
    *,
    tokenizer_name: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Persist a ``compressed-tensors`` (llm-compressor) quantized model.

    A model quantized by the ``llmcompressor`` backend
    (:func:`circuitkit.applications.quantization.llmcompressor_circuit_quantize`)
    carries ``compressed-tensors`` quantized modules. Unlike optimum-quanto,
    ``compressed-tensors`` is a first-class HuggingFace quantization format:
    plain ``model.save_pretrained`` writes the packed weights **and** a
    ``quantization_config`` block into ``config.json``, so the checkpoint
    round-trips through ``AutoModelForCausalLM.from_pretrained`` and is served
    natively by vLLM — **no dequantization step is needed**.

    Args:
        hf_model: A HuggingFace model with compressed-tensors quantized modules.
        output_path: Directory to write the checkpoint into.
        tokenizer_name: Optional HF tokenizer id to also save alongside the
            weights (so the checkpoint is self-contained for lm-eval / vLLM).
        overwrite: Remove ``output_path`` first if it exists.

    Returns:
        ``output_path`` for chaining.

    Raises:
        RuntimeError: if ``hf_model`` carries no compressed-tensors modules.
    """
    if os.path.exists(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to remove it."
            )

    if not _has_compressed_tensors_modules(hf_model):
        raise RuntimeError(
            "save_compressed_tensors_checkpoint was called on a model with no "
            "compressed-tensors quantized modules — it is not quantized. Run "
            "circuitkit.quantize(..., backend='llmcompressor') first."
        )

    os.makedirs(output_path, exist_ok=True)

    # compressed-tensors is HF-native: save_pretrained writes the packed
    # safetensors + a quantization_config into config.json. save_compressed=True
    # stores the weights in the genuinely compressed layout.
    #
    # quantization_format="pack-quantized": for weight-only integer
    # quantization (3 / 4-bit) the packed-int32 layout is the format vLLM
    # expects. The auto-inferred default for sub-4-bit widths can be
    # "naive-quantized", which vLLM's compressed-tensors loader treats as an
    # *activation*-quant format and then asserts FLOAT weights — breaking the
    # vLLM load of an int weight-only checkpoint. Forcing "pack-quantized"
    # keeps the checkpoint vLLM-loadable for every supported bit width.
    try:
        hf_model.save_pretrained(
            output_path,
            save_compressed=True,
            quantization_format="pack-quantized",
        )
    except TypeError:
        # Older llm-compressor save_pretrained without quantization_format.
        hf_model.save_pretrained(output_path, save_compressed=True)

    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_path)
        # Multimodal-derived checkpoints (Gemma 3, …) need their processor
        # configs so a downstream server (vLLM) can load them.
        _copy_processor_configs(tokenizer_name, output_path)

    return output_path


def is_compressed_tensors_checkpoint(checkpoint_path: str) -> bool:
    """True iff ``checkpoint_path`` is a compressed-tensors quantized checkpoint.

    Detected by a ``quantization_config`` block in ``config.json`` whose
    ``quant_method`` is ``compressed-tensors``. Such a checkpoint reloads with
    plain ``AutoModelForCausalLM.from_pretrained`` and is served by vLLM
    natively — it must NOT go through the quanto reload / dequantization path.
    """
    import json as _json

    cfg_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        cfg = _json.loads(open(cfg_path, "r").read())
    except (OSError, ValueError):
        return False
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    method = str(qc.get("quant_method", "")).lower()
    return method in ("compressed-tensors", "compressed_tensors")


def load_compressed_tensors_checkpoint(checkpoint_path: str, **kwargs: Any) -> Any:
    """Reload a checkpoint saved by :func:`save_compressed_tensors_checkpoint`.

    compressed-tensors is HF-native, so this is a plain
    ``AutoModelForCausalLM.from_pretrained`` — the ``quantization_config`` in
    ``config.json`` drives the dequant/compressed-linear reconstruction. Provided
    as a named helper for symmetry with :func:`load_quantized_checkpoint`.
    """
    from transformers import AutoModelForCausalLM

    if not is_compressed_tensors_checkpoint(checkpoint_path):
        raise RuntimeError(
            f"{checkpoint_path} carries no compressed-tensors quantization_config; "
            "it is not a compressed-tensors checkpoint."
        )
    return AutoModelForCausalLM.from_pretrained(checkpoint_path, **kwargs)


def save_quantized_checkpoint(
    hf_model: Any,
    output_path: str,
    *,
    tokenizer_name: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Persist an already-quantized HuggingFace model as a reloadable checkpoint.

    The model passed in must already have had ``circuit_quantize`` (or any
    ``optimum.quanto.quantize``) applied so its modules are ``QModuleMixin``
    instances.

    A plain ``hf_model.save_pretrained()`` does **not** persist quantization:
    it writes no ``quantization_config`` and the float/packed state in a
    standalone-``quantize``d model does not round-trip — reload gives a plain
    fp model (or fails). To genuinely round-trip we use quanto's HF-integrated
    serializer, :class:`optimum.quanto.QuantizedModelForCausalLM`, which writes
    the safetensors weights **and** a ``quanto_qmap.json`` quantization map.
    Reload with :func:`load_quantized_checkpoint`.

    Args:
        hf_model: A HuggingFace ``AutoModelForCausalLM`` with quanto modules.
        output_path: Directory to write the checkpoint into.
        tokenizer_name: Optional HF tokenizer id to also save alongside the
            weights (so the checkpoint is self-contained for lm-eval).
        overwrite: Remove ``output_path`` first if it exists.

    Returns:
        ``output_path`` for chaining.

    The model may also be a ``compressed-tensors`` model produced by the
    ``llmcompressor`` quantization backend; this is auto-detected and dispatched
    to :func:`save_compressed_tensors_checkpoint`, which writes a vLLM-native,
    ``from_pretrained``-reloadable checkpoint (no quanto qmap, no dequantization).

    Raises:
        RuntimeError: if ``hf_model`` is not actually quantized (neither quanto
            nor compressed-tensors modules) — saving it as a "quantized
            checkpoint" would silently produce a plain fp checkpoint.
        ImportError: if optimum-quanto is not installed (quanto models only).
    """
    # llm-compressor / compressed-tensors models serialize natively — dispatch
    # before touching optimum-quanto so a compressed-tensors model never falls
    # into the quanto path.
    if _has_compressed_tensors_modules(hf_model):
        return save_compressed_tensors_checkpoint(
            hf_model,
            output_path,
            tokenizer_name=tokenizer_name,
            overwrite=overwrite,
        )

    if os.path.exists(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to remove it."
            )

    try:
        from optimum.quanto import QuantizedModelForCausalLM
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "optimum-quanto is required to save a quantized checkpoint: "
            "pip install optimum-quanto"
        ) from e

    if not _has_quanto_modules(hf_model):
        raise RuntimeError(
            "save_quantized_checkpoint was called on a model with no "
            "optimum-quanto modules — it is not quantized. Run "
            "circuit_quantize (or optimum.quanto.quantize) on the model "
            "first, otherwise the checkpoint would silently be plain fp."
        )

    os.makedirs(output_path, exist_ok=True)

    # quanto's canonical flow is quantize -> (calibrate) -> freeze -> save:
    # freeze() materialises each QModule's weight as a packed QTensor that is
    # a DISTINCT tensor. Saving an UNFROZEN quantized model is subtly broken
    # for weight-tied models (GPT-2 and most small LLMs): the unfrozen
    # lm_head still shares the input embedding's tensor, so safetensors drops
    # lm_head.weight from the file entirely, and the reload rebuilds a
    # QModule head over UNINITIALIZED memory — garbage logits, order/allocator
    # dependent, silent. Freezing before save unties the head into real packed
    # data (lm_head.weight._data/_scale) so the checkpoint genuinely
    # round-trips. Freezing an already-frozen model is a no-op.
    from optimum.quanto import QModuleMixin, freeze

    def _has_unfrozen_qmodule(m: Any) -> bool:
        for mod in m.modules():
            if isinstance(mod, QModuleMixin) and not getattr(mod, "frozen", True):
                return True
        return False

    if _has_unfrozen_qmodule(hf_model):
        logger.info(
            "save_quantized_checkpoint: model has unfrozen quanto modules — "
            "running optimum.quanto.freeze() so the packed weights serialize "
            "(required for weight-tied heads to round-trip)."
        )
        freeze(hf_model)

    # QuantizedModelForCausalLM.quantize would re-quantize; we already have a
    # quantized model, so wrap it directly and serialize. The wrapper writes
    # model.safetensors + config.json + quanto_qmap.json (the quantization map
    # needed to rebuild the QModules on reload).
    qmodel = QuantizedModelForCausalLM(hf_model)
    qmodel.save_pretrained(output_path)

    # Sanity: the qmap marker must exist or reload cannot reconstruct QModules.
    if not os.path.exists(os.path.join(output_path, _QUANTO_MARKER)):
        raise RuntimeError(
            f"quanto serialization did not write {_QUANTO_MARKER} into "
            f"{output_path}; the checkpoint would not reload as quantized."
        )

    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer_name).save_pretrained(output_path)
        # Multimodal-derived checkpoints (Gemma 3, …) need their processor
        # configs so a downstream server (vLLM) can load them.
        _copy_processor_configs(tokenizer_name, output_path)

    return output_path


def is_quantized_checkpoint(checkpoint_path: str) -> bool:
    """True iff ``checkpoint_path`` was written by :func:`save_quantized_checkpoint`
    (i.e. it carries a quanto quantization map and must be reloaded the quanto
    way, not via plain ``AutoModelForCausalLM.from_pretrained``)."""
    return os.path.exists(os.path.join(checkpoint_path, _QUANTO_MARKER))


def _checkpoint_tensor_keys(checkpoint_path: str) -> set:
    """Tensor key names present in a checkpoint's safetensors file(s)."""
    import glob
    import json

    keys: set = set()
    index = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, "r", encoding="utf-8") as f:
            keys.update(json.load(f).get("weight_map", {}).keys())
        return keys
    from safetensors import safe_open

    for fname in glob.glob(os.path.join(checkpoint_path, "*.safetensors")):
        with safe_open(fname, framework="pt") as sf:
            keys.update(sf.keys())
    return keys


def load_quantized_checkpoint(checkpoint_path: str, **kwargs: Any) -> Any:
    """Reload a checkpoint saved by :func:`save_quantized_checkpoint`.

    Uses ``optimum.quanto.QuantizedModelForCausalLM.from_pretrained``, which
    rebuilds the quanto ``QModule`` layers from the saved quantization map.

    Asserts the reloaded model genuinely contains quanto modules — if quanto
    silently produced a plain fp model, this raises rather than letting a
    corrupted "quantized" result through.

    Args:
        checkpoint_path: Directory written by :func:`save_quantized_checkpoint`.
        **kwargs: Forwarded to ``from_pretrained`` (e.g. ``device_map``).

    Returns:
        The reloaded quantized ``nn.Module`` (the underlying HF model).
    """
    try:
        from optimum.quanto import QModuleMixin, QuantizedModelForCausalLM
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("optimum-quanto is required to load a quantized checkpoint.") from e

    if not is_quantized_checkpoint(checkpoint_path):
        raise RuntimeError(
            f"{checkpoint_path} has no {_QUANTO_MARKER}; it is not a quanto "
            "quantized checkpoint. Use AutoModelForCausalLM.from_pretrained."
        )

    qmodel = QuantizedModelForCausalLM.from_pretrained(checkpoint_path, **kwargs)
    # QuantizedModelForCausalLM is a thin wrapper exposing the HF model via
    # attribute delegation; pull out the genuine nn.Module for downstream use.
    inner = getattr(qmodel, "_wrapped", None)
    model = inner if inner is not None else qmodel

    if not _has_quanto_modules(model):
        raise RuntimeError(
            f"Reloaded checkpoint {checkpoint_path} contains NO optimum-quanto "
            "modules — quanto reload silently produced a plain fp model. The "
            "quantized checkpoint did not round-trip; results would be wrong."
        )

    # Weight-tied models (e.g. GPT-2, most small LLMs): at save time the
    # output embedding shares the input embedding's tensor, so
    # save_pretrained/safetensors DROPS lm_head.weight from the file — while
    # quanto's serializer simultaneously writes tie_word_embeddings=False
    # into the saved config (incoherent upstream state). quanto's
    # from_pretrained then rebuilds lm_head as a QModule whose packed weight
    # was never in the file, leaving UNINITIALIZED memory: depending on
    # allocator state the logits are zeros or garbage, order-dependent and
    # silent. The saved config flag is therefore untrustworthy — detect the
    # condition from the ground truth instead: an output-embedding QModule
    # whose weight key is absent from the saved tensors. Restore the original
    # tied semantics by replacing the orphaned head with a plain Linear
    # sharing the input-embedding weight.
    out_emb = model.get_output_embeddings()
    in_emb = model.get_input_embeddings()
    if isinstance(out_emb, QModuleMixin) and in_emb is not None:
        out_name = next((n for n, m_ in model.named_modules() if m_ is out_emb), None)
        saved_keys = _checkpoint_tensor_keys(checkpoint_path)
        # The weight may be stored plain (lm_head.weight) or as quanto's
        # frozen packed representation (lm_head.weight._data / ._scale) —
        # both mean the head's data genuinely survived the save.
        head_key_saved = out_name is not None and any(
            k == f"{out_name}.weight" or k.startswith(f"{out_name}.weight.") for k in saved_keys
        )
        if not head_key_saved:
            new_head = t.nn.Linear(
                in_emb.weight.shape[1],
                in_emb.weight.shape[0],
                bias=getattr(out_emb, "bias", None) is not None,
                dtype=in_emb.weight.dtype,
            )
            new_head.weight = in_emb.weight  # re-tie: share the tensor
            if new_head.bias is not None and out_emb.bias is not None:
                b = out_emb.bias
                b = b.dequantize() if hasattr(b, "dequantize") else b
                with t.no_grad():
                    new_head.bias.copy_(b.detach().to(new_head.bias.dtype))
            model.set_output_embeddings(new_head)
            logger.info(
                "load_quantized_checkpoint: re-tied the output embedding "
                f"({out_name}) to the input embedding — its weight was dropped "
                "from the saved file (weight tying at save time), so the "
                "reloaded quanto head held uninitialized data."
            )
    return model


def export_dequantized_checkpoint(
    quantized_checkpoint_path: str,
    output_path: str,
    *,
    dtype: Optional[Any] = None,
    overwrite: bool = False,
) -> str:
    """Materialise a quanto-quantized checkpoint as a plain fp HF checkpoint.

    A checkpoint written by :func:`save_quantized_checkpoint` is an
    ``optimum.quanto`` artifact (``quanto_qmap.json`` + quanto-packed tensors).
    vLLM has no quanto loader (it supports GPTQ / AWQ / bitsandbytes / fp8 /
    compressed-tensors / gguf / ...), so such a checkpoint cannot be served by
    the vLLM backend directly.

    This helper reloads the quanto checkpoint, **dequantizes every QModule
    weight back to a floating-point tensor**, and re-saves a standard
    ``AutoModelForCausalLM`` checkpoint (``model.safetensors`` + ``config.json``,
    no ``quanto_qmap.json``).  vLLM then runs the *exact* quantized weights —
    the quantization error is already baked into the values; only the packed
    int kernel is dropped.  For **evaluation** the numbers are identical to the
    quanto checkpoint, just computed in fp.

    Note: this is a dequantized fp checkpoint, not a vLLM-native quantized
    format. It is the pragmatic path for *benchmarking* a quantized model under
    vLLM. For a genuinely packed deployment artifact, re-quantize into a
    vLLM-supported format (GPTQ / AWQ / compressed-tensors).

    Args:
        quantized_checkpoint_path: Directory written by
            :func:`save_quantized_checkpoint`.
        output_path: Directory to write the plain fp checkpoint into.
        dtype: Optional torch dtype for the exported weights (default: keep
            each dequantized tensor's dtype, typically the model's fp dtype).
        overwrite: Remove ``output_path`` first if it exists.

    Returns:
        ``output_path`` for chaining.

    Raises:
        RuntimeError: if ``quantized_checkpoint_path`` is not a quanto
            checkpoint (no ``quanto_qmap.json``).
    """
    if not is_quantized_checkpoint(quantized_checkpoint_path):
        raise RuntimeError(
            f"{quantized_checkpoint_path} has no {_QUANTO_MARKER}; it is not a "
            "quanto quantized checkpoint and does not need dequantizing. Use it "
            "directly (plain HF checkpoints already load under vLLM)."
        )

    if os.path.exists(output_path):
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"{output_path} already exists. Pass overwrite=True to remove it."
            )

    try:
        from optimum.quanto import QModuleMixin
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("optimum-quanto is required to dequantize a quantized checkpoint.") from e

    # Reload the quanto checkpoint — this rebuilds the QModule layers.
    model = load_quantized_checkpoint(quantized_checkpoint_path)

    # Replace every QModule with a plain nn.Linear holding the dequantized
    # weight, so save_pretrained writes a standard fp checkpoint.
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, QModuleMixin):
            continue
        w = mod.weight
        # quanto QTensors expose .dequantize(); a plain tensor passes through.
        dq_w = w.dequantize() if hasattr(w, "dequantize") else w
        dq_w = dq_w.detach()
        if dtype is not None:
            dq_w = dq_w.to(dtype)
        linear = t.nn.Linear(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            dtype=dq_w.dtype,
        )
        with t.no_grad():
            linear.weight.copy_(dq_w)
            if mod.bias is not None:
                b = mod.bias
                b = b.dequantize() if hasattr(b, "dequantize") else b
                linear.bias.copy_(b.detach().to(linear.bias.dtype))
        # Splice the plain Linear back into the parent module.
        parent = model
        *parents, leaf = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, leaf, linear)

    if _has_quanto_modules(model):
        raise RuntimeError(
            "export_dequantized_checkpoint failed to remove all quanto modules; "
            "the exported checkpoint would still be quanto-only."
        )

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)

    # Carry the tokenizer over so the fp checkpoint is self-contained.
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        src = os.path.join(quantized_checkpoint_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))

    return output_path


def benchmark_on_checkpoint(
    checkpoint_path: str,
    tokenizer_name: str,
    task_names: List[str],
    *,
    fewshot: int = 0,
    limit: Optional[int] = None,
    apply_chat_template: Union[bool, str] = "auto",
    **eval_kwargs,
) -> Dict[str, Any]:
    """Evaluate a saved HF checkpoint on lm-eval tasks using stock ``HFLM``.

    Use this when you already have a checkpoint from a previous
    ``save_pruned_checkpoint`` call — avoids re-pruning the model.

    Args:
        checkpoint_path: Directory containing ``config.json`` + ``model.safetensors``.
        tokenizer_name: HuggingFace tokenizer ID (e.g. ``"gpt2"``).
        task_names: List of lm-eval task names (e.g. ``["mmlu", "lambada_openai"]``).
        fewshot: Number of few-shot examples.
        limit: Max examples to evaluate (useful for quick smoke tests).
        apply_chat_template: Wrap each task prompt in the model's chat template
            before scoring. Instruction-tuned checkpoints score badly on raw
            task text (MMLU/BoolQ/GSM8K/...), so this must be on for them.
            ``"auto"`` (default) enables it iff the checkpoint's tokenizer
            ships a ``chat_template``; pass ``True``/``False`` to force it.

    Returns:
        Raw lm-eval results dict.  Task metrics are under
        ``results["results"]["<task_name>"]`` — use keys like
        ``"acc,none"``, ``"acc_norm,none"``, ``"exact_match,none"`` etc.
    """
    import inspect

    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("lm-eval is required. Install: pip install lm-eval")

    hflm = HFLM(pretrained=checkpoint_path, tokenizer=tokenizer_name)

    # Instruction-tuned checkpoints must receive prompts wrapped in their chat
    # template — raw task text (MMLU/BoolQ/GSM8K/...) confuses them. "auto"
    # triggers iff the checkpoint's tokenizer ships a chat_template.
    if apply_chat_template == "auto":
        tok_obj = getattr(hflm, "tokenizer", None)
        chat_tmpl = getattr(tok_obj, "chat_template", None)
        if chat_tmpl is None and tok_obj is None:
            from transformers import AutoTokenizer

            chat_tmpl = getattr(
                AutoTokenizer.from_pretrained(tokenizer_name), "chat_template", None
            )
        apply_chat_template = chat_tmpl is not None

    _sig = inspect.signature(evaluator.simple_evaluate)
    if "apply_chat_template" in _sig.parameters:
        eval_kwargs["apply_chat_template"] = apply_chat_template
        if "fewshot_as_multiturn" in _sig.parameters:
            eval_kwargs["fewshot_as_multiturn"] = bool(apply_chat_template) and fewshot > 0
    elif apply_chat_template:
        import warnings

        warnings.warn(
            "Installed lm-eval has no apply_chat_template support; upgrade "
            "lm-eval to score instruction-tuned checkpoints correctly."
        )

    return evaluator.simple_evaluate(
        model=hflm,
        tasks=task_names,
        num_fewshot=fewshot,
        limit=limit,
        **eval_kwargs,
    )
