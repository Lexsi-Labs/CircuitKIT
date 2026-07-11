"""
CircuitKit's flat, typed, Pythonic front-door API.

This module is a *thin facade* over the existing CircuitKit engine. It exists
so researchers can run the common Discover -> Evaluate -> Intervene workflow
with ordinary keyword arguments instead of hand-writing nested ``dict`` configs.

Every function here builds the appropriate dict-config internally and calls the
real engine (:mod:`circuitkit.api`, :mod:`circuitkit.evaluation`,
:mod:`circuitkit.applications`). Nothing in the engine is modified.

The flat API is a **strict superset shortcut**: it covers the common path.
Anything it cannot express (custom corruption variants, exotic algorithm
sub-methods, bespoke pruning scopes, ...) is still reachable by dropping to the
dict-config functions :func:`circuitkit.api.discover_circuit` and
:func:`circuitkit.api.evaluate_circuit` directly — see each function's
docstring for the exact escape hatch.

Public surface:
    * :func:`load_model`        — load a TransformerLens model with discovery flags set.
    * :func:`discover`          — run circuit discovery, return a :class:`~circuitkit.circuit.Circuit`.
    * :func:`faithfulness`      — score a circuit with the 6/7-pillar framework.
    * :func:`prune`             — structurally prune a model using a circuit.
    * :func:`quantize`          — circuit-guided mixed-precision quantization.
    * :func:`export_checkpoint` — write an intervened model as a HF checkpoint.
    * :func:`benchmark`         — run lm-evaluation-harness on a checkpoint.

All heavy imports (torch, transformer_lens, ...) are performed lazily inside
the functions so ``import circuitkit`` stays fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .circuit import Circuit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformer_lens import HookedTransformer

    from .evaluation.report import FaithfulnessReport
    
    from pathlib import Path

__all__ = [
    "load_model",
    "discover",
    "faithfulness",
    "prune",
    "quantize",
    "export_checkpoint",
    "benchmark",
    "load_scores",
    "selective_finetune",
    "visualize_circuit",
]


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
# Algorithms that consume the EAP graph and therefore need the four hook flags.
_EAP_FLAG_ALGORITHMS = frozenset(
    {
        "acdc",
        "eap",
        "eap-ig",
        "eap-ig-activations",
        "eap-clean-corrupted",
        "eap-exact",
        "atp-gd",
        "eap-gp",
        "relp",
        "peap",
        "eap-ifr",
        "cdt",
    }
)


def _valid_discovery_algorithms() -> List[str]:
    """Return the sorted list of known discovery algorithm names."""
    from .backends import DISCOVERY_ALGORITHMS

    return sorted(DISCOVERY_ALGORITHMS)


def _check_algorithm(algorithm: str) -> str:
    """Validate a discovery algorithm name, returning it lower-cased.

    Raises:
        ValueError: With the full list of valid names if ``algorithm`` is unknown.
    """
    algo = algorithm.lower()
    valid = _valid_discovery_algorithms()
    if algo not in valid:
        raise ValueError(
            f"Unknown discovery algorithm {algorithm!r}. "
            f"Valid algorithms: {valid}. "
            f"(Default and recommended: 'eap-ig'.)"
        )
    return algo


def _check_level(level: str) -> str:
    """Validate a discovery level (``'node'`` or ``'neuron'``)."""
    if level not in ("node", "neuron"):
        raise ValueError(f"level must be 'node' or 'neuron', got {level!r}")
    return level


# --------------------------------------------------------------------------- #
# load_model                                                                  #
# --------------------------------------------------------------------------- #
def load_model(
    name: str,
    *,
    dtype: str = "bfloat16",
    device: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> "HookedTransformer":
    """Load a TransformerLens model with the flags circuit discovery needs.

    Plain ``HookedTransformer.from_pretrained`` does not enable the hook points
    EAP-family algorithms rely on, which leads researchers into cryptic crashes
    ("must ungroup grouped attention", attention-result shape errors). This
    helper loads the model **and** sets the four config flags so discovery
    just works:

    * ``use_attn_result``
    * ``use_split_qkv_input``
    * ``use_hook_mlp_in``
    * ``ungroup_grouped_query_attention`` (when the model exposes it)

    Args:
        name: HuggingFace / TransformerLens model id (e.g. ``"gpt2"``,
            ``"pythia-70m"``, ``"Qwen/Qwen2-0.5B"``).
        dtype: Torch dtype string — ``"bfloat16"`` (default), ``"float32"``,
            ``"float16"``, ...
        device: Target device. ``None`` (default) auto-selects ``"cuda"`` when
            available, else ``"cpu"``.
        algorithm: If given, only the flags that algorithm needs are set. When
            omitted, the safe superset (all four flags) is enabled so the model
            works with any algorithm.

    Returns:
        A configured :class:`~transformer_lens.HookedTransformer`, ready to pass
        straight to :func:`discover`.

    Raises:
        ValueError: If ``algorithm`` is given but not a known discovery algorithm.

    Example:
        >>> import circuitkit as ck
        >>> model = ck.load_model("gpt2", dtype="float32")
        >>> circuit = ck.discover(model, "ioi", n_examples=16)
    """
    import torch
    from transformer_lens import HookedTransformer

    if algorithm is not None:
        algorithm = _check_algorithm(algorithm)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        torch_dtype = getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown dtype {dtype!r}. Use a torch dtype name such as "
            f"'bfloat16', 'float32' or 'float16'."
        ) from exc

    model = HookedTransformer.from_pretrained(name, device=device, dtype=torch_dtype)

    # Enable the hook points discovery relies on. When an algorithm is given we
    # still set the safe superset for EAP-family algorithms (they all need the
    # same three flags); ungroup is always safe to enable when present.
    needs_eap_flags = algorithm is None or algorithm in _EAP_FLAG_ALGORITHMS
    if needs_eap_flags:
        model.cfg.use_attn_result = True
        model.cfg.use_split_qkv_input = True
        model.cfg.use_hook_mlp_in = True
    if hasattr(model.cfg, "ungroup_grouped_query_attention"):
        model.cfg.ungroup_grouped_query_attention = True

    return model


# --------------------------------------------------------------------------- #
# discover                                                                    #
# --------------------------------------------------------------------------- #
def build_discovery_config(
    model: "HookedTransformer",
    task: str,
    *,
    algorithm: str = "eap-ig",
    level: str = "node",
    n_examples: int = 128,
    batch_size: int = 4,
    sparsity: float = 0.3,
    scope: str = "both",
    output_path: str = "./circuit_discovery_results.pt",
    **kw: Any,
) -> Dict[str, Any]:
    """Build the dict-config that :func:`circuitkit.api.discover_circuit` expects.

    Exposed publicly so researchers can inspect — or tweak — the exact config
    the flat :func:`discover` shortcut generates before handing it to the
    engine. Extra keyword arguments are merged into the ``discovery`` block.

    Args:
        model: A configured :class:`~transformer_lens.HookedTransformer`.
        task: Registered task name (e.g. ``"ioi"``, ``"mmlu"``).
        algorithm: Discovery algorithm. Defaults to ``"eap-ig"``.
        level: ``"node"`` or ``"neuron"``.
        n_examples: Number of examples to attribute over.
        batch_size: Discovery batch size.
        sparsity: Target pruning sparsity (fraction of components removed).
        scope: Pruning scope — ``"heads"``, ``"mlp"`` or ``"both"``.
        output_path: Where the engine writes the artifact + score side-cars.
        **kw: Extra keys merged into the ``discovery`` block (e.g.
            ``ig_steps``, ``intervention``, ``method``, ``num_epochs``).

    Returns:
        A dict with ``model``, ``discovery``, ``pruning`` and ``output_path``
        keys — directly consumable by :func:`circuitkit.api.discover_circuit`.

    Example:
        >>> cfg = build_discovery_config(model, "ioi", n_examples=16)
        >>> cfg["discovery"]["algorithm"]
        'eap-ig'
    """
    algorithm = _check_algorithm(algorithm)
    level = _check_level(level)
    if scope not in ("heads", "mlp", "both"):
        raise ValueError(f"scope must be 'heads', 'mlp' or 'both', got {scope!r}")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0.0, 1.0], got {sparsity}")
    if n_examples <= 0:
        raise ValueError(f"n_examples must be a positive int, got {n_examples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive int, got {batch_size}")

    # Prefer the full HF repo name (``cfg.tokenizer_name`` /
    # ``tokenizer.name_or_path``) over ``cfg.model_name``: TransformerLens
    # stores ``model_name`` as a stripped alias (e.g. "Llama-3.2-1B-Instruct"),
    # which ``HookedTransformer.from_pretrained`` no longer accepts for
    # Llama-3.x — it requires the "meta-llama/..." repo path. The discovery
    # engine re-loads the model from this config name, so a bare alias here
    # makes discovery fail to re-load the model.
    model_name = (
        getattr(model.cfg, "tokenizer_name", None)
        or getattr(getattr(model, "tokenizer", None), "name_or_path", None)
        or getattr(model.cfg, "model_name", None)
        or "gpt2"
    )
    precision = str(getattr(model.cfg, "dtype", "float32"))
    precision = precision.replace("torch.", "")

    data_params: Dict[str, Any] = {"num_examples": n_examples, "batch_size": batch_size}
    # Propagate the seed into data_params so it reaches data generation (e.g.
    # IOI, which reads discovery.data_params.seed and otherwise defaults to 42).
    # Without this, ck.discover(seed=...) left the data seed fixed, so multi-seed
    # runs produced identical circuits and degenerate (zero-variance) error bars.
    if kw.get("seed") is not None:
        data_params["seed"] = kw["seed"]

    discovery: Dict[str, Any] = {
        "algorithm": algorithm,
        "task": task,
        "level": level,
        "batch_size": batch_size,
        "data_params": data_params,
    }
    discovery.update(kw)

    return {
        "model": {"name": model_name, "precision": precision},
        "discovery": discovery,
        "pruning": {"target_sparsity": sparsity, "scope": scope},
        "output_path": output_path,
    }


def discover(
    model: "HookedTransformer",
    task: str,
    *,
    algorithm: str = "eap-ig",
    level: str = "node",
    n_examples: int = 128,
    batch_size: int = 4,
    sparsity: float = 0.3,
    scope: str = "both",
    output_path: str = "./circuit_discovery_results.pt",
    **kw: Any,
) -> Circuit:
    """Run circuit discovery and return a :class:`~circuitkit.circuit.Circuit`.

    This is the flat shortcut for :func:`circuitkit.api.discover_circuit`: it
    assembles the nested dict-config from keyword arguments, runs the discovery
    engine, and wraps the resulting artifact + score side-cars in a typed
    :class:`Circuit` object.

    For anything this shortcut cannot express — custom ``intervention``,
    algorithm sub-``method``, IBCircuit training hyper-parameters — pass it as
    an extra keyword (it is merged into the ``discovery`` block) or drop to
    :func:`circuitkit.api.discover_circuit` with a hand-written dict.

    Args:
        model: A configured :class:`~transformer_lens.HookedTransformer`,
            ideally from :func:`load_model`.
        task: Registered task name (e.g. ``"ioi"``, ``"mmlu"``). See
            :func:`circuitkit.list_tasks`.
        algorithm: Discovery algorithm. Defaults to the stable ``"eap-ig"``.
        level: ``"node"`` (heads / MLPs) or ``"neuron"``.
        n_examples: Number of examples to attribute over. Defaults to 128.
        batch_size: Discovery batch size. Defaults to 4.
        sparsity: Target pruning sparsity. Defaults to 0.3.
        scope: Pruning scope — ``"heads"``, ``"mlp"`` or ``"both"``.
        output_path: Where the engine writes the ``.pt`` artifact and the
            ``_scores.json`` / ``_scores.pt`` side-cars.
        **kw: Extra keys forwarded into the ``discovery`` block (e.g.
            ``ig_steps=5``, ``intervention="patching"``).

    Returns:
        A :class:`Circuit` wrapping the discovered nodes and their scores.

    Raises:
        ValueError: If ``algorithm``, ``level``, ``scope`` or numeric args are invalid.

    Example:
        >>> import circuitkit as ck
        >>> model = ck.load_model("gpt2", dtype="float32")
        >>> circuit = ck.discover(model, "ioi", algorithm="eap-ig",
        ...                       n_examples=16, ig_steps=2)
        >>> print(circuit)
        Circuit(level=node, n_nodes=..., algorithm='eap-ig', task='ioi', ...)
        >>> circuit.top_nodes(3)
        {...}
    """
    from .api import discover_circuit

    config = build_discovery_config(
        model,
        task,
        algorithm=algorithm,
        level=level,
        n_examples=n_examples,
        batch_size=batch_size,
        sparsity=sparsity,
        scope=scope,
        output_path=output_path,
        **kw,
    )

    # Reuse the caller's already-loaded `model` instead of letting
    # discover_circuit() load a second copy from config["model"]["name"].
    # discover_circuit() still applies all its config flags unconditionally
    # after this, so this is safe regardless of how `model` was loaded.
    nodes = discover_circuit(config, _model=model)
    algorithm = _check_algorithm(algorithm)

    # discover_circuit writes the score side-cars next to output_path; load
    # them so the Circuit carries scores + metadata. Fall back gracefully if
    # the artifact / side-car is missing.
    from pathlib import Path

    if Path(output_path).exists():
        circuit = Circuit.from_artifact(output_path)
        # Prefer the in-memory return value of discover_circuit; it is always
        # present even when no output directory was configured.
        circuit.nodes = nodes
    else:
        circuit = Circuit(nodes, level=level, artifact_path=output_path)

    circuit.level = level
    circuit.task = circuit.task or task
    circuit.algorithm = circuit.algorithm or algorithm
    circuit.model_name = circuit.model_name or config["model"]["name"]
    return circuit


# --------------------------------------------------------------------------- #
# faithfulness                                                                #
# --------------------------------------------------------------------------- #
def faithfulness(
    model: "HookedTransformer",
    circuit: Circuit,
    task: str,
    *,
    pillars: Optional[List[str]] = None,
    n_examples: int = 256,
    batch_size: int = 16,
    device: Optional[str] = None,
    configs: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
    **kw: Any,
) -> "FaithfulnessReport":
    """Score a discovered circuit with the multi-pillar faithfulness framework.

    Flat wrapper around :func:`circuitkit.evaluation.run_full_faithfulness`. It
    resolves the task spec and metric function, reconstructs the circuit graph
    from the discovered scores, builds an evaluation dataloader, and runs the
    requested faithfulness pillars.

    For full control over corruption variants, baseline types, stability runs
    or cross-task generalization, call
    :func:`circuitkit.evaluation.run_full_faithfulness` directly.

    Args:
        model: A configured :class:`~transformer_lens.HookedTransformer`.
        circuit: A :class:`Circuit` from :func:`discover`. It must carry node
            scores (run :func:`discover` with an ``output_path``).
        task: Registered task name the circuit was discovered for.
        pillars: Which pillars to compute. ``None`` runs all. Pass a subset to
            skip expensive pillars, e.g. ``["patching", "ablation"]``.
        n_examples: Number of evaluation examples.
        batch_size: Evaluation batch size.
        device: Target device. ``None`` auto-selects.
        configs: Task-specific config/subject subset to restrict the
            evaluation dataloader to (e.g. ``"wmdp-bio"``). Only meaningful
            for tasks that partition their data into named configs — pass the
            same value used for ``discover(..., configs=...)`` so the circuit
            is evaluated on the distribution it was discovered on. ``None``
            (default) leaves the task's own default in effect, which for
            multi-config tasks like WMDP is *all* configs.
        **kw: Extra keyword arguments forwarded to
            :func:`run_full_faithfulness` (e.g. ``n_stability_runs``).

    Returns:
        A :class:`~circuitkit.evaluation.report.FaithfulnessReport` with per-pillar scores.

    Raises:
        ValueError: If the circuit carries no scores, or the task is unknown.

    Example:
        >>> report = ck.faithfulness(model, circuit, "ioi",
        ...                          pillars=["patching", "ablation"])
        >>> report.patching_score
        0.87
    """
    import torch

    from .evaluation import run_full_faithfulness
    from .tasks.bootstrap import _bootstrap_builtin_tasks
    from .tasks.registry import get_task as _get_task

    if not circuit.scores:
        raise ValueError(
            "Circuit has no node scores — faithfulness needs them. "
            "Re-run discover() with an output_path so scores are saved."
        )

    from .api import _make_eval_metric

    _bootstrap_builtin_tasks()
    task_spec = _get_task(task)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    algorithm = circuit.algorithm or "eap-ig"
    discovery_cfg: Dict[str, Any] = {
        "algorithm": algorithm,
        "task": task,
        "level": circuit.level,
        "batch_size": batch_size,
        "data_params": {"num_examples": n_examples, "batch_size": batch_size},
    }
    if configs is not None:
        discovery_cfg["configs"] = configs
    # Propagate the seed into data_params so Pillar 3 (Stability) rediscovers
    # from it (run_full_faithfulness reads discovery.data_params.seed for
    # seed_start, else 42). Without this, stability was measured over the same
    # fixed neighbourhood for every circuit, so per-seed stability collapsed to
    # a constant (zero-variance) — breaking any stability-vs-outcome analysis.
    if seed is not None:
        discovery_cfg["data_params"]["seed"] = seed

    # Reconstruct at the circuit's ACTUAL sparsity. circuit.nodes is the PRUNED
    # (removed) node list, so the discovery sparsity is that fraction of the
    # scoreable nodes. A hardcoded 0.0 here (the previous behaviour) is wrong —
    # combined with the bespoke reconstructor below it left the circuit's edge
    # matrix empty, so every circuit evaluated identically to the corrupt
    # baseline (faithfulness ~0) regardless of what was discovered.
    n_scoreable = int(model.cfg.n_layers) * (int(model.cfg.n_heads) + 1)
    if isinstance(circuit.nodes, dict):
        n_pruned = len(circuit.nodes.get("heads", {})) + len(circuit.nodes.get("mlp", {}))
    else:
        n_pruned = len(circuit.nodes)
    target_sparsity = min(1.0, max(0.0, n_pruned / n_scoreable)) if n_scoreable else 0.0
    pruning_cfg: Dict[str, Any] = {"target_sparsity": target_sparsity, "scope": "both"}

    # Rebuild the circuit graph using the SAME reconstruction the discovery/eval
    # path uses (populates graph.nodes_scores, pins out-of-scope nodes, then
    # apply_topn to set node membership AND the 2-D edge matrix that
    # evaluate_graph reads, with prune() for connectivity).
    from .api import _reconstruct_circuit_graph

    scores_data = {"node_scores": circuit.scores}
    graph = _reconstruct_circuit_graph(model, scores_data, discovery_cfg, pruning_cfg, device)
    dataloader = task_spec.build_dataloader(model, discovery_cfg, device)

    return run_full_faithfulness(
        model=model,
        graph=graph,
        task_spec=task_spec,
        discovery_cfg=discovery_cfg,
        device=device,
        pillars=pillars,
        # Reward-oriented, per-sample metric (loss=False) so clean > corrupt and
        # the faithfulness ratio is well-defined. task_spec.metric_fn() is the
        # loss-style *discovery* metric, which would invert the denominator and
        # make Pillars 1/2 spuriously report status='invalid' on canonical tasks.
        metric_fn=_make_eval_metric(task_spec),
        dataloader=dataloader,
        pruning_cfg=pruning_cfg,
        **kw,
    )


# --------------------------------------------------------------------------- #
# prune                                                                       #
# --------------------------------------------------------------------------- #
def prune(
    model: "HookedTransformer",
    circuit: Circuit,
    *,
    sparsity: float = 0.3,
    scope: str = "both",
    protect_layers: Optional[List[int]] = None,
    inplace: bool = False,
    dry_run: Optional[bool] = None,
    **kw: Any,
) -> "HookedTransformer":
    """Structurally mask a model using a discovered circuit's scores.

    Flat wrapper around :class:`circuitkit.applications.pruning.StructuralPruner`.
    Masks (zeroes the weights of) the lowest-scoring attention heads / MLP
    layers to reach the target sparsity, returning the masked model. This is
    structured *masking* — tensor shapes are unchanged and no parameters are
    physically removed; genuine removal happens at checkpoint export via
    :func:`export_checkpoint`.

    Args:
        model: The :class:`~transformer_lens.HookedTransformer` to mask.
        circuit: A node-level :class:`Circuit` carrying node scores.
        sparsity: Target fraction of nodes to mask. Defaults to 0.3.
        scope: Which component type to mask — ``"heads"`` (only attention
            heads), ``"mlp"`` (only MLP layers) or ``"both"`` (default). Uses
            the same vocabulary as :func:`discover`'s ``scope``. The sparsity
            budget is taken within the chosen component type.
        protect_layers: Layer indices that must never be masked — any head or
            MLP in those layers is excluded. ``None`` (default) protects
            nothing.
        inplace: When ``False`` (default, safe) masking operates on a deep
            copy and the original ``model`` is left untouched. Pass ``True``
            to mask ``model`` in place.
        dry_run: Deprecated alias for ``not inplace`` (``dry_run=True`` ==
            ``inplace=False``). Kept for back-compatibility; emits a
            ``DeprecationWarning``. ``inplace`` wins if both are passed.
        **kw: Reserved for forward compatibility.

    Returns:
        The masked :class:`~transformer_lens.HookedTransformer`.

    Raises:
        ValueError: If the circuit carries no scores, is not node-level, or
            ``scope`` is invalid.

    Example:
        >>> pruned = ck.prune(model, circuit, sparsity=0.4, scope="mlp")
    """
    from .applications.pruning import StructuralPruner
    from .artifacts.scores import CircuitScores

    if circuit.level != "node":
        raise ValueError(
            f"prune() supports node-level circuits only; got level={circuit.level!r}. "
            f"For neuron-level pruning use circuitkit.applications.pruning directly."
        )
    if not circuit.scores:
        raise ValueError(
            "Circuit has no node scores — prune() needs them. "
            "Re-run discover() with an output_path so scores are saved."
        )

    circuit_scores = circuit.circuit_scores
    if circuit_scores is None:
        circuit_scores = CircuitScores(
            task=circuit.task or "unknown",
            model=circuit.model_name or "unknown",
            algorithm=circuit.algorithm or "eap-ig",
            level="node",
            node_scores=circuit.scores,
            timestamp=CircuitScores.create_timestamp(),
        )

    pruner = StructuralPruner()
    return pruner.prune(
        model,
        circuit_scores,
        sparsity=sparsity,
        scope=scope,
        protect_layers=protect_layers,
        inplace=inplace,
        dry_run=dry_run,
    )


# --------------------------------------------------------------------------- #
# quantize                                                                    #
# --------------------------------------------------------------------------- #
def quantize(
    model: Any,
    circuit: Circuit,
    *,
    n_layers: Optional[int] = None,
    high_fraction: float = 0.3,
    protect_layers: Optional[List[int]] = None,
    backend: str = "quanto",
    bits: int = 3,
    tokenizer: Any = None,
    **kw: Any,
) -> Dict[str, Any]:
    """Apply circuit-guided mixed-precision quantization.

    Ranks layers by circuit importance and keeps important layers at high
    precision while quantizing the rest. Model is modified in place.

    Two backends: ``"quanto"`` (default, integer 2/4/8-bit tiers) or
    ``"llmcompressor"`` (true low-bit with GPTQ calibration, vLLM-compatible).

    Args:
        model: HuggingFace ``AutoModelForCausalLM``.
        circuit: Node-level :class:`Circuit` with scores.
        n_layers: Number of layers (inferred from model if omitted).
        high_fraction: Fraction of layers to protect.
        protect_layers: Specific layer indices to leave at native precision.
        backend: ``"quanto"`` or ``"llmcompressor"``.
        bits: Bit-width for ``"llmcompressor"`` backend. Ignored for
            ``"quanto"``, which is controlled via ``low_weights``/
            ``high_weights``/``mid_weights`` (see ``**kw``) and otherwise
            defaults to ``qint4`` for the low tier.
        tokenizer: HF tokenizer for GPTQ calibration (``"llmcompressor"`` only).
        **kw: Extra args forwarded to the backend — for ``"quanto"`` this
            includes ``low_weights``, ``high_weights``, ``mid_weights``
            (quanto qtypes, e.g. ``optimum.quanto.qint8``).

    Returns:
        Tier-assignment plan (``"quanto"``) or summary dict (``"llmcompressor"``).
    """
    import re

    if backend not in ("quanto", "llmcompressor"):
        raise ValueError(
            f"backend must be 'quanto' or 'llmcompressor', got {backend!r}"
        )

    if not circuit.scores:
        raise ValueError(
            "Circuit has no node scores — quantize() needs them. "
            "Re-run discover() with an output_path so scores are saved."
        )

    # Derive per-head and per-MLP scores from the circuit's node scores.
    # Node names: attention heads 'A{layer}.{head}', MLPs 'MLP {layer}'.
    q_head_scores: Dict[Any, float] = {}
    mlp_scores: Dict[int, float] = {}
    max_layer = -1
    for name, score in circuit.scores.items():
        attn = re.match(r"[Aa](\d+)\.[hH]?(\d+)$", name)
        mlp = re.match(r"MLP\s*(\d+)$", name)
        if attn:
            layer, head = int(attn.group(1)), int(attn.group(2))
            q_head_scores[(layer, head)] = float(score)
            max_layer = max(max_layer, layer)
        elif mlp:
            layer = int(mlp.group(1))
            mlp_scores[layer] = float(score)
            max_layer = max(max_layer, layer)

    if n_layers is None:
        cfg = getattr(model, "config", None)
        n_layers = getattr(cfg, "num_hidden_layers", None)
        if n_layers is None and cfg is not None:
            # Gemma-3 / multimodal configs nest the decoder under text_config.
            n_layers = getattr(
                getattr(cfg, "text_config", None), "num_hidden_layers", None
            )
        if n_layers is None and max_layer >= 0:
            n_layers = max_layer + 1
    if not n_layers:
        raise ValueError("Could not determine n_layers; pass n_layers explicitly to quantize().")

    if backend == "llmcompressor":
        from .applications.quantization import llmcompressor_circuit_quantize

        if tokenizer is None:
            from transformers import AutoTokenizer

            repo_id = getattr(
                getattr(model, "config", None), "_name_or_path", None
            ) or circuit.model_name
            if not repo_id:
                raise ValueError(
                    "quantize(backend='llmcompressor') needs a tokenizer; pass "
                    "tokenizer= explicitly (the model carries no resolvable "
                    "repo id to auto-load one)."
                )
            tokenizer = AutoTokenizer.from_pretrained(repo_id)

        return llmcompressor_circuit_quantize(
            model,
            tokenizer,
            q_head_scores=q_head_scores,
            mlp_scores=mlp_scores,
            n_layers=int(n_layers),
            bits=bits,
            high_fraction=high_fraction,
            protect_layers=protect_layers,
            **kw,
        )

    from .applications.quantization import circuit_quantize

    return circuit_quantize(
        model,
        q_head_scores=q_head_scores,
        mlp_scores=mlp_scores,
        n_layers=int(n_layers),
        high_fraction=high_fraction,
        protect_layers=protect_layers,
        **kw,
    )


# --------------------------------------------------------------------------- #
# export_checkpoint                                                           #
# --------------------------------------------------------------------------- #
def export_checkpoint(
    model: Any,
    artifact: Union[Circuit, List[str], Dict[str, Any], None],
    path: str,
    *,
    intervention: str = "pruning",
    overwrite: bool = True,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    hub_private: bool = True,
    **kw: Any,
) -> str:
    """Export an intervened model as a HuggingFace checkpoint on disk.

    Flat wrapper around :func:`circuitkit.evaluation.save_pruned_checkpoint` /
    :func:`circuitkit.evaluation.save_quantized_checkpoint`. The resulting
    directory holds ``config.json`` + weights + tokenizer and is directly
    consumable by :func:`benchmark` or lm-evaluation-harness.

    Args:
        model: For ``intervention="pruning"`` a TransformerLens
            ``HookedTransformer``; for ``intervention="quantization"`` an
            already-quantized HF ``AutoModelForCausalLM``.
        artifact: For pruning, the pruning artifact — a :class:`Circuit`, a
            node-name list, or a neuron dict. Ignored for quantization
            (pass ``None``).
        path: Destination directory for the HF checkpoint.
        intervention: ``"pruning"`` (default) or ``"quantization"``.
        overwrite: Overwrite ``path`` if it already exists.
        push_to_hub: When ``True``, after the local checkpoint is written it is
            uploaded to the HuggingFace Hub via ``huggingface_hub``. Default
            ``False`` (purely opt-in — nothing is pushed unless asked).
        hub_repo: Target Hub repo id (``"org/name"``). Required when
            ``push_to_hub`` is ``True``.
        hub_private: Create the Hub repo as private (default ``True``).
        **kw: Extra keyword arguments forwarded to the underlying ``save_*``
            helper (e.g. ``tokenizer_name`` for quantization).

    Returns:
        The checkpoint directory path.

    Raises:
        ValueError: If ``intervention`` is unknown, a pruning export is
            requested without an ``artifact``, or ``push_to_hub`` is set
            without ``hub_repo``.

    Example:
        >>> ck.export_checkpoint(pruned_model, circuit, "ckpt/ioi_pruned")
        'ckpt/ioi_pruned'
        >>> # archive a headline checkpoint to the Hub:
        >>> ck.export_checkpoint(pruned_model, circuit, "ckpt/headline",
        ...                      push_to_hub=True, hub_repo="my-org/headline-1")
    """
    from .evaluation import save_pruned_checkpoint, save_quantized_checkpoint

    if intervention == "pruning":
        nodes = artifact.nodes if isinstance(artifact, Circuit) else artifact
        if nodes is None:
            raise ValueError("export_checkpoint(intervention='pruning') needs an artifact.")
        save_pruned_checkpoint(model, nodes, path, overwrite=overwrite, **kw)
    elif intervention == "quantization":
        save_quantized_checkpoint(model, path, overwrite=overwrite, **kw)
    else:
        raise ValueError(f"intervention must be 'pruning' or 'quantization', got {intervention!r}")

    if push_to_hub:
        _push_checkpoint_to_hub(path, hub_repo, hub_private)

    return path


def _push_checkpoint_to_hub(
    path: str, hub_repo: Optional[str], hub_private: bool
) -> None:
    """Upload a written checkpoint directory to the HuggingFace Hub.

    Opt-in archival path for :func:`export_checkpoint`. The local checkpoint is
    already on disk; this only uploads it. Requires ``huggingface_hub`` and an
    authenticated token (``huggingface-cli login`` or ``HF_TOKEN``).
    """
    if not hub_repo:
        raise ValueError(
            "export_checkpoint(push_to_hub=True) needs hub_repo='org/name'."
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "push_to_hub=True needs huggingface_hub: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=hub_repo, private=hub_private, exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=hub_repo, repo_type="model")


# --------------------------------------------------------------------------- #
# benchmark                                                                   #
# --------------------------------------------------------------------------- #
def benchmark(
    checkpoint_path: str,
    tasks: Union[str, List[str]],
    *,
    backend: str = "hf",
    limit: Optional[int] = None,
    fewshot: int = 0,
    device: Optional[str] = None,
    dtype: str = "float32",
    **kw: Any,
) -> Dict[str, Dict[str, float]]:
    """Run lm-evaluation-harness on a saved HF checkpoint.

    Flat wrapper around :func:`circuitkit.evaluation.run_lm_eval`. Use it to score
    a checkpoint produced by :func:`export_checkpoint` on standard benchmarks.

    Args:
        checkpoint_path: Directory containing ``config.json`` + weights, e.g.
            the output of :func:`export_checkpoint`.
        tasks: lm-eval task name or list of names (e.g. ``"boolq"`` or
            ``["boolq", "winogrande"]``).
        backend: ``"hf"`` (HFLM, default) or ``"vllm"`` if installed.
        limit: Cap examples per task — handy for quick smoke tests.
        fewshot: Few-shot example count.
        device: Torch device for the ``hf`` backend. ``None`` auto-selects.
        dtype: Model dtype string for the ``hf`` backend.
        **kw: Extra keyword arguments forwarded to ``run_lm_eval`` (e.g.
            ``tokenizer``, ``batch_size``, ``apply_chat_template``).

    Returns:
        ``{task: {metric: value}}`` — finite leaderboard metrics per task.

    Example:
        >>> scores = ck.benchmark("ckpt/ioi_pruned", ["boolq"], limit=50)
        >>> scores["boolq"]["acc"]
        0.62
    """
    import torch

    from .evaluation import run_lm_eval

    if isinstance(tasks, str):
        tasks = [tasks]
    if not tasks:
        raise ValueError("benchmark() needs at least one lm-eval task.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return run_lm_eval(
        checkpoint_path,
        tasks,
        backend=backend,
        limit=limit,
        fewshot=fewshot,
        device=device,
        dtype=dtype,
        **kw,
    )
    
# --------------------------------------------------------------------------- #
# load_scores                                                                  #
# --------------------------------------------------------------------------- #
def load_scores(
    path: Union[str, "Path"],
    *,
    scores_path: Optional[Union[str, "Path"]] = None,
) -> Circuit:
    """Load a previously saved circuit from disk as a rich :class:`Circuit`.

    Wrapper around :meth:`Circuit.from_artifact`. Loads the pruning artifact
    and its score side-car, returning a ready-to-use :class:`Circuit` (with
    ``.scores``, ``.top_nodes()``, ``.task``, ``.model_name``, ...). This is the
    loader most callers want. Contrast with :func:`circuitkit.load_circuit`,
    which returns the **raw** ``list``/``dict`` pruning artifact and no scores.

    Args:
        path: Path to the ``.pt`` pruning artifact.
        scores_path: Optional explicit path to the scores side-car.
            Auto-derived from ``path`` when omitted.

    Returns:
        A :class:`Circuit` populated from disk.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Example:
        >>> circuit = ck.load_scores("results/ioi_circuit.pt")
        >>> circuit.top_nodes(5)

    See Also:
        circuitkit.load_circuit: Load the raw pruning artifact (``list``/``dict``).
    """
    return Circuit.from_artifact(path, scores_path=scores_path)


# --------------------------------------------------------------------------- #
# selective_finetune                                                           #
# --------------------------------------------------------------------------- #
def selective_finetune(
    circuit: Circuit,
    *,
    model_name: Optional[str] = None,
    top_fraction: float = 0.2,
    scope: str = "both",
    exclude_first_n: int = 0,
    exclude_last_n: int = 0,
    n_layers: Optional[int] = None,
    n_q_heads: Optional[int] = None,
    n_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Any:
    """Select components for circuit-guided selective finetuning.

    Wrapper around :mod:`circuitkit.applications.selective_finetuning`.
    Uses the circuit's scores to identify the top-X% most important
    components (attention heads and/or MLP layers/neurons).

    The model architecture parameters (``n_layers``, ``n_q_heads``,
    ``n_kv_heads``, ``head_dim``) are auto-loaded from the HuggingFace config
    when ``model_name`` is provided and any parameter is omitted.

    Args:
        circuit: A :class:`Circuit` with scores (node or neuron level).
        model_name: HF model name, used to auto-load architecture params and
            for neuron-level EAP scores. Defaults to ``circuit.model_name``.
        top_fraction: Fraction of components to select (0.0-1.0).
        scope: ``"attn"``, ``"mlp"``, or ``"both"``.
        exclude_first_n: Exclude the first N layers from selection.
        exclude_last_n: Exclude the last N layers from selection.
        n_layers: Total transformer layer count. Auto-loaded when omitted.
        n_q_heads: Q heads per layer. Auto-loaded when omitted.
        n_kv_heads: KV heads per layer (equal to n_q_heads for MHA).
            Auto-loaded when omitted.
        head_dim: Dimension per attention head. Auto-loaded when omitted.

    Returns:
        A :class:`~circuitkit.applications.selective_finetuning.selector.SelectionResult`
        with ``.attn`` and ``.mlp`` dicts mapping component keys to index lists.

    Raises:
        ValueError: If the circuit has no ``artifact_path`` and no scores
            side-car can be located, or if architecture params cannot be
            resolved.

    Example:
        >>> result = ck.selective_finetune(circuit, top_fraction=0.15)
        >>> print(result.attn.keys())
    """
    from pathlib import Path as _Path

    from .applications.selective_finetuning.score_loader import load_scores as _load_scores
    from .applications.selective_finetuning.selector import select_components

    resolved_model = model_name or circuit.model_name

    if not circuit.artifact_path:
        raise ValueError(
            "Circuit has no artifact_path. Save the circuit first with circuit.save(), "
            "or use ck.load_scores() to load from disk."
        )

    artifact = _Path(circuit.artifact_path)
    # score_loader.load_scores expects the _scores.pt side-car specifically
    scores_pt = artifact.parent / f"{artifact.stem}_scores.pt"
    if not scores_pt.exists():
        raise FileNotFoundError(
            f"Scores side-car not found at {scores_pt}. "
            "Re-run discovery with an output_path to generate it."
        )

    head_scores, mlp_scores, metadata = _load_scores(
        str(scores_pt),
        model_name=resolved_model,
    )

    # Auto-load architecture params from HF config when any are missing
    if any(p is None for p in (n_layers, n_q_heads, n_kv_heads, head_dim)):
        if resolved_model is None:
            raise ValueError(
                "model_name is required to auto-load architecture params "
                "(n_layers, n_q_heads, n_kv_heads, head_dim). "
                "Pass them explicitly or set circuit.model_name."
            )
        try:
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(resolved_model)
            if n_layers is None:
                n_layers = hf_cfg.num_hidden_layers
            if n_q_heads is None:
                n_q_heads = hf_cfg.num_attention_heads
            if n_kv_heads is None:
                n_kv_heads = getattr(hf_cfg, "num_key_value_heads", n_q_heads)
            if head_dim is None:
                head_dim = hf_cfg.hidden_size // n_q_heads
        except Exception as exc:
            raise ValueError(
                f"Could not auto-load architecture params from {resolved_model!r}: {exc}. "
                "Pass n_layers, n_q_heads, n_kv_heads, head_dim explicitly."
            ) from exc

    return select_components(
        head_scores,
        mlp_scores,
        metadata,
        top_frac=top_fraction,
        scope=scope,
        n_layers=n_layers,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        exclude_first_n=exclude_first_n,
        exclude_last_n=exclude_last_n,
    )


# --------------------------------------------------------------------------- #
# visualize                                                                    #
# --------------------------------------------------------------------------- #
def visualize_circuit(
    circuit: Circuit,
    *,
    mode: str = "graph",
    output: Optional[str] = None,
    second_circuit: Optional[Circuit] = None,
    **kw: Any,
) -> Any:
    """Visualize a circuit.

    Dispatches to the appropriate visualizer from :mod:`circuitkit.visualize`.

    Args:
        circuit: The :class:`Circuit` to visualize.
        mode: Visualization mode:

            ``"graph"`` — Interactive node/edge graph (default). Delegates to
            :meth:`Circuit.plot`.

            ``"comparison"`` — Side-by-side heatmap / correlation comparison of
            two circuits. Requires ``second_circuit``. Uses
            :class:`~circuitkit.visualize.ComparisonDashboard`.

            ``"dashboard"`` — Launch interactive Streamlit dashboard (if
            available).

        output: Path to save HTML export. ``None`` returns the widget/figure
            inline.
        second_circuit: Second :class:`Circuit` for ``mode="comparison"``.

    Returns:
        For ``"graph"``: HTML string or Jupyter widget (from :meth:`Circuit.plot`).
        For ``"comparison"``: ``None`` (writes HTML to ``output``) or the
            :class:`~circuitkit.visualize.ComparisonDashboard` instance.
        For ``"dashboard"``: ``None`` (launches subprocess).

    Raises:
        ValueError: If ``mode`` is unknown, or ``mode="comparison"`` is used
            without ``second_circuit``.

    Example:
        >>> ck.visualize_circuit(circuit, mode="graph", output="circuit.html")
        >>> ck.visualize_circuit(circuit, mode="comparison",
        ...              second_circuit=circuit2, output="compare.html")
    """
    if mode == "graph":
        return circuit.plot(output)

    if mode == "comparison":
        from .visualize.comparison import ComparisonDashboard

        if second_circuit is None:
            raise ValueError("mode='comparison' requires the second_circuit argument.")

        if not circuit.scores or not second_circuit.scores:
            raise ValueError(
                "Both circuits must carry node scores for comparison. "
                "Re-run discovery with an output_path."
            )

        label_a = circuit.task or circuit.algorithm or "circuit_a"
        label_b = second_circuit.task or second_circuit.algorithm or "circuit_b"
        # Deduplicate labels if identical
        if label_a == label_b:
            label_a, label_b = f"{label_a}_1", f"{label_b}_2"

        dashboard = ComparisonDashboard(
            circuits={label_a: circuit.scores, label_b: second_circuit.scores},
            comparison_type=kw.get("comparison_type", "stability"),
            labels=kw.get("labels"),
            metadata=kw.get("metadata"),
        )

        if output:
            dashboard.export_to_html(output)
            return None
        return dashboard

    if mode == "dashboard":
        import subprocess
        import sys
        from pathlib import Path as _Path

        streamlit_path = str(_Path(__file__).parent / "visualize" / "streamlit_app.py")
        if not _Path(streamlit_path).exists():
            raise FileNotFoundError(
                f"Streamlit app not found at {streamlit_path}. "
                "The dashboard is not available in this installation."
            )
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", streamlit_path])
        return None

    raise ValueError(
        f"Unknown visualization mode: {mode!r}. "
        "Use 'graph', 'comparison', or 'dashboard'."
    )
