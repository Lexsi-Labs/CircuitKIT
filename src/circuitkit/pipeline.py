"""Stateful Discover → Evaluate → Intervene pipeline for CircuitKit.

The Pipeline class is a stateful orchestrator that carries model, circuit,
and evaluation state across method calls. It delegates to ``quick.*`` functions
for applications and calls ``api.*`` directly for discovery/evaluation/benchmarking.

Use Pipeline for multi-step workflows; use ``circuitkit.quick.*`` functions
for single one-shot calls.

Example::

    from circuitkit import Pipeline

    pipe = Pipeline("gpt2", task="ioi")
    pipe.discover(algorithm="eap-ig", level="node", sparsity=0.3, n_examples=32)
    pipe.evaluate(pillars=["patching", "ablation"], n_examples=64)
    pipe.prune(sparsity=0.3)
    pipe.export("./output/checkpoint")
    pipe.summary()
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from circuitkit.evaluation.report import FaithfulnessReport

import logging

logger = logging.getLogger(__name__)

__all__ = ["Pipeline"]


def _safe_path_token(value: Any) -> str:
    """Reduce an arbitrary value to a single, path-safe filename token.

    Replaces every character that isn't ``[A-Za-z0-9._-]`` (path separators
    included) with ``_``, so an attacker-influenced ``model_name``/``task``
    can't inject ``/`` or ``..`` to escape the output directory when it is
    interpolated into an artifact filename.
    """
    token = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    # Collapse any leading dots so a bare ".."/"." can't act as a path segment.
    return token.lstrip(".") or "unnamed"


# Numbered pillars (1-6) and their canonical string names. Pillar 7
# ("intervention_reliability") has no number and is passed through by name.
_PILLAR_NAMES = {
    "patching": 1,
    "ablation": 2,
    "stability": 3,
    "robustness": 4,
    "baselines": 5,
    "generalization": 6,
}
_PILLAR_BY_ID = {v: k for k, v in _PILLAR_NAMES.items()}


def _resolve_pillars(pillars: Optional[Union[List, str]]) -> Optional[List[str]]:
    """Normalise a pillar spec to the canonical string names the evaluator uses.

    Integers 1-6 map to their pillar name; string names pass through unchanged
    (validated downstream by ``run_full_faithfulness``, which also knows the
    unnumbered ``intervention_reliability`` pillar). ``None`` or ``"all"``
    returns ``None`` (run every pillar).
    """
    if pillars is None or pillars == "all":
        return None
    result: List[str] = []
    for p in pillars:
        # bool is a subclass of int — reject it explicitly rather than treat
        # True/False as pillar 1/0.
        if isinstance(p, bool):
            raise TypeError(f"Pillar must be a name or an int 1-6, got bool {p!r}")
        if isinstance(p, int):
            if p not in _PILLAR_BY_ID:
                raise ValueError(
                    f"Unknown pillar id {p}. Valid ids: {sorted(_PILLAR_BY_ID)} "
                    f"(or use names: {sorted(_PILLAR_NAMES)})."
                )
            result.append(_PILLAR_BY_ID[p])
        elif isinstance(p, str):
            result.append(p)
        else:
            raise TypeError(f"Pillar must be a name or an int 1-6, got {type(p).__name__}")
    return result


class Pipeline:
    """Stateful Discover → Evaluate → Intervene orchestrator.

    Carries model name, task, and discovered circuit state across method calls.
    The model is loaded lazily on the first call that needs it.

    Args:
        model_name: HuggingFace / TransformerLens model identifier.
        task: Registered task name (e.g. ``"ioi"``, ``"sva"``). Required for
            discovery. Can be omitted when constructing via :meth:`from_scores`
            or :meth:`from_artifact`.
        precision: Torch dtype string for model loading. Defaults to
            ``"bfloat16"``.
        device: Target device. ``None`` auto-selects ``"cuda"`` when available.
        output_dir: Directory for all artifact outputs. Defaults to
            ``"./pipeline_output"``.

    Example::

        pipe = Pipeline("gpt2", task="ioi", output_dir="./results")
        pipe.discover(algorithm="eap-ig", n_examples=128)
        pipe.prune(sparsity=0.3)
        pipe.export("./results/checkpoint")
    """

    def __init__(
        self,
        model_name: str,
        *,
        task: Optional[str] = None,
        precision: str = "bfloat16",
        device: Optional[str] = None,
        output_dir: str = "./pipeline_output",
    ) -> None:
        self.model_name = model_name
        self.task = task
        self.precision = precision
        self.output_dir = str(output_dir)
        self._device: Optional[str] = device

        # Lazily populated state
        self._model: Optional[Any] = None
        self._hf_model: Optional[Any] = None  # HF AutoModelForCausalLM, for quantize()
        self._circuit: Optional[Any] = None  # Circuit instance
        self._artifact_path: Optional[str] = None
        self._eval_report: Optional["FaithfulnessReport"] = None
        self._pruned_model: Optional[Any] = None  # model after prune()/quantize()
        self._last_intervention: Optional[str] = None  # "pruning" | "quantization"
        self._history: List[str] = []  # steps executed
        # Custom data config (set by from_custom_data)
        self._custom_data_cfg: Optional[Dict[str, Any]] = None
        self._task_name_override: Optional[str] = None

        # Discovery config snapshot (used by evaluate/benchmark as config template)
        self._discovery_cfg: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Alternative constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_artifact(
        cls,
        artifact_path: Union[str, Path],
        model_name: str,
        *,
        task: Optional[str] = None,
        precision: str = "bfloat16",
        device: Optional[str] = None,
        output_dir: str = "./pipeline_output",
    ) -> "Pipeline":
        """Construct a Pipeline from a pre-existing circuit artifact.

        Loads the circuit from disk so you can immediately call
        :meth:`prune`, :meth:`quantize`, :meth:`visualize`, etc.

        Args:
            artifact_path: Path to the ``.pt`` pruning artifact.
            model_name: HF model name (required for applications).
            task: Task name for metadata / evaluation.
            precision: Torch dtype string.
            device: Target device.
            output_dir: Directory for outputs.

        Returns:
            A :class:`Pipeline` with ``._circuit`` populated.
        """
        from .circuit import Circuit

        pipe = cls(
            model_name,
            task=task,
            precision=precision,
            device=device,
            output_dir=output_dir,
        )
        pipe._circuit = Circuit.from_artifact(str(artifact_path))
        pipe._artifact_path = str(artifact_path)
        pipe.task = pipe.task or pipe._circuit.task
        pipe._history.append("from_artifact")
        return pipe

    @classmethod
    def from_scores(
        cls,
        scores_path: Union[str, Path],
        model_name: str,
        *,
        task: Optional[str] = None,
        precision: str = "bfloat16",
        device: Optional[str] = None,
        output_dir: str = "./pipeline_output",
    ) -> "Pipeline":
        """Construct a Pipeline by loading pre-saved circuit scores.

        Uses :func:`circuitkit.load_scores` to hydrate a :class:`Circuit`
        from a ``_scores.pt`` file, then stores it for downstream use.

        Args:
            scores_path: Path to a ``_scores.pt`` or ``_scores.json`` side-car.
            model_name: HF model name.
            task: Task name for metadata / evaluation.
            precision: Torch dtype string.
            device: Target device.
            output_dir: Directory for outputs.

        Returns:
            A :class:`Pipeline` with ``._circuit`` populated.
        """
        from . import quick

        pipe = cls(
            model_name,
            task=task,
            precision=precision,
            device=device,
            output_dir=output_dir,
        )
        pipe._circuit = quick.load_scores(str(scores_path))
        if not pipe._circuit.model_name:
            pipe._circuit.model_name = model_name
        pipe.task = pipe.task or pipe._circuit.task
        pipe._history.append("from_scores")
        return pipe

    @classmethod
    def from_custom_data(
        cls,
        model_name: str,
        data_path: Union[str, Path],
        *,
        clean_prompt: str,
        clean_answer: str,
        corrupt_prompt: Optional[str] = None,
        corrupt_answer: Optional[str] = None,
        task_name: Optional[str] = None,
        precision: str = "bfloat16",
        device: Optional[str] = None,
        output_dir: str = "./pipeline_output",
    ) -> "Pipeline":
        """Construct a Pipeline backed by custom CSV data.

        Registers the CSV as a task and returns a Pipeline ready for
        :meth:`discover`. Supports both paired (EAP/ACDC/CDT) and
        unpaired (IBCircuit/CDT clean-only) data.

        Args:
            model_name: HF model name.
            data_path: Path to a CSV file.
            clean_prompt: Template string for clean prompts (e.g. ``"{question}"``).
            clean_answer: Template string for clean answers.
            corrupt_prompt: Template for corrupt prompts. Omit for IBCircuit/CDT.
            corrupt_answer: Template for corrupt answers. Omit for IBCircuit/CDT.
            task_name: Registry name. Auto-derived from CSV stem if omitted.
            precision: Torch dtype string.
            device: Target device.
            output_dir: Directory for outputs.

        Returns:
            A :class:`Pipeline` with :attr:`task` set to the registered task name.
        """
        data_path = str(data_path)
        derived_name = task_name or f"custom:{Path(data_path).stem}"

        pipe = cls(
            model_name,
            task=derived_name,
            precision=precision,
            device=device,
            output_dir=output_dir,
        )
        # Store data config for use in discover()
        pipe._custom_data_cfg = {
            "type": "template",
            "path": data_path,
            "template": {
                "clean_prompt": clean_prompt,
                "clean_answer": clean_answer,
            },
        }
        if corrupt_prompt and corrupt_answer:
            pipe._custom_data_cfg["template"]["corrupt_prompt"] = corrupt_prompt
            pipe._custom_data_cfg["template"]["corrupt_answer"] = corrupt_answer
        pipe._task_name_override = derived_name
        pipe._history.append("from_custom_data")
        return pipe

    # ------------------------------------------------------------------ #
    # Lazy model loading                                                  #
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> Any:
        """Load model lazily, reusing the cached instance."""
        if self._model is None:
            from . import quick

            self._model = quick.load_model(
                self.model_name,
                dtype=self.precision,
                device=self._device,
            )
        return self._model

    def _ensure_hf_model(self) -> Any:
        """Load a HuggingFace ``AutoModelForCausalLM`` lazily, reusing the
        cached instance.

        Quantization operates on the model's native HF module tree via
        architecture-specific layer paths (``circuit_quantize`` calls
        ``detect_model_architecture``, which reads ``model.config``), so it
        needs a plain ``AutoModelForCausalLM`` rather than the TransformerLens
        ``HookedTransformer`` that :meth:`_ensure_model` loads for
        discovery/pruning.
        """
        if self._hf_model is None:
            import torch
            from transformers import AutoModelForCausalLM

            try:
                torch_dtype = getattr(torch, self.precision)
            except AttributeError as exc:
                raise ValueError(
                    f"Unknown dtype {self.precision!r}. Use a torch dtype name "
                    f"such as 'bfloat16', 'float32' or 'float16'."
                ) from exc

            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else self.device,
            )
        return self._hf_model

    @property
    def device(self) -> str:
        """Resolved device string (auto-detects if not set)."""
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def report(self) -> Optional["FaithfulnessReport"]:
        """The most recent faithfulness report, or ``None`` before ``evaluate()``.

        Populated by :meth:`evaluate`. Access pillar scores directly, e.g.
        ``pipe.report.patching_score`` or ``pipe.report.stability``.
        """
        return self._eval_report

    @property
    def circuit(self) -> Optional[Any]:
        """The discovered circuit, or ``None`` before ``discover()`` / ``from_artifact()``."""
        return self._circuit

    @property
    def model(self) -> Optional[Any]:
        """The loaded ``HookedTransformer``, or ``None`` until first use.

        Also returns ``None`` after ``quantize(release_original=True)``, which
        discards the original model to free memory.
        """
        return self._model

    @property
    def pruned_model(self) -> Optional[Any]:
        """The masked/quantized model, or ``None`` before ``prune()`` / ``quantize()``."""
        return self._pruned_model

    @property
    def artifact_path(self) -> Optional[str]:
        """Path the circuit artifact was written to, or ``None`` if not yet saved."""
        return self._artifact_path

    @property
    def history(self) -> List[str]:
        """Names of steps run so far, e.g. ``["discover", "prune"]`` (a copy)."""
        return list(self._history)

    # ------------------------------------------------------------------ #
    # Discovery                                                           #
    # ------------------------------------------------------------------ #

    def discover(
        self,
        *,
        algorithm: str = "eap-ig",
        level: str = "node",
        sparsity: float = 0.3,
        n_examples: int = 128,
        batch_size: int = 4,
        scope: str = "both",
        seed: Optional[int] = None,
        **kw: Any,
    ) -> "Pipeline":
        """Run circuit discovery.

        Builds a discovery config, calls :func:`circuitkit.api.discover_circuit`,
        and stores the result as ``._circuit``.

        Args:
            algorithm: Discovery algorithm. Default ``"eap-ig"``.
            level: ``"node"`` or ``"neuron"``.
            sparsity: Target pruning sparsity.
            n_examples: Number of examples to attribute over.
            batch_size: Discovery batch size.
            scope: ``"heads"``, ``"mlp"``, or ``"both"``.
            seed: Random seed for discovery. Propagated into
                ``discovery.data_params.seed`` so it reaches data generation
                (e.g. IOI). Vary it across runs for genuine multi-seed error
                bars; leave ``None`` to use the data default.
            **kw: Extra keys forwarded into the discovery config block
                (e.g. ``ig_steps=5``).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If :attr:`task` is not set.
        """
        if self.task is None:
            raise ValueError(
                "Pipeline.task must be set before calling discover(). "
                "Pass task= to the constructor or use from_custom_data()."
            )

        from .api import discover_circuit, prepare_custom_task
        from .circuit import Circuit

        os.makedirs(self.output_dir, exist_ok=True)
        # Sanitize model_name and task before interpolating them into the
        # artifact filename: a value containing path separators or ".." (e.g. a
        # task name from a shared pipeline config) would otherwise let the
        # written file escape output_dir (path traversal / arbitrary write).
        safe_model = _safe_path_token(self.model_name)
        safe_task = _safe_path_token(self.task)
        output_path = os.path.join(
            self.output_dir,
            f"{algorithm}_{safe_model}_{safe_task}_{level}.pt",
        )

        data_params: Dict[str, Any] = {"num_examples": n_examples, "batch_size": batch_size}
        # Propagate the seed into data_params so it actually reaches data
        # generation (e.g. IOI, which reads discovery.data_params.seed and
        # otherwise silently defaults to 42). Without this, discover(seed=...)
        # left the data seed fixed, so multi-seed runs produced identical
        # circuits and degenerate (zero-variance) error bars.
        if seed is not None:
            data_params["seed"] = seed

        discovery_block: Dict[str, Any] = {
            "algorithm": algorithm,
            "task": self.task,
            "level": level,
            "batch_size": batch_size,
            "data_params": data_params,
        }
        if seed is not None:
            discovery_block["seed"] = seed
        discovery_block.update(kw)

        config: Dict[str, Any] = {
            "model": {"name": self.model_name, "precision": self.precision},
            "discovery": discovery_block,
            "pruning": {"target_sparsity": sparsity, "scope": scope},
            "output_path": output_path,
        }

        # Attach custom data block if the pipeline was built from custom data
        if hasattr(self, "_custom_data_cfg"):
            config["data"] = self._custom_data_cfg
            model = self._ensure_model()
            task_name = getattr(self, "_task_name_override", None)
            self.task = prepare_custom_task(config, model, task_name=task_name)

            # prepare_custom_task() consumed `model` for its tokenizer only
            # and has already popped config["data"], so discover_circuit()
            # below will NOT re-run the custom-task setup - it loads its own
            # model from config["model"]["name"] regardless. Release our copy
            # now rather than holding two full models in GPU RAM simultaneously.
            # Any later pipeline step that needs self._model will reload it
            # transparently via _ensure_model().
            self._model = None
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # CDT warning: node-level only
        if algorithm.lower() == "cdt" and level != "node":
            import warnings

            warnings.warn(
                "CD-T supports node-level discovery only. Overriding level to 'node'.",
                UserWarning,
                stacklevel=2,
            )
            config["discovery"]["level"] = "node"
            level = "node"

        # If self._model is already in memory (e.g. a prior pipeline step
        # populated it, or the non-custom-data path loaded it earlier), pass
        # it through so discover_circuit() can skip reloading. In the common
        # case self._model is None here (either the custom-data branch above
        # just released it, or nothing pre-loaded it), and discover_circuit()
        # loads exactly as before.
        nodes = discover_circuit(config, _model=self._model)
        self._artifact_path = output_path
        self._discovery_cfg = config

        # Wrap in Circuit
        if Path(output_path).exists():
            circuit = Circuit.from_artifact(output_path)
            circuit.nodes = nodes
        else:
            circuit = Circuit(nodes, level=level, artifact_path=output_path)

        circuit.level = level
        circuit.task = circuit.task or self.task
        circuit.algorithm = circuit.algorithm or algorithm
        circuit.model_name = circuit.model_name or self.model_name
        self._circuit = circuit
        self._history.append("discover")
        return self

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        *,
        pillars: Optional[Union[List, str]] = None,
        n_examples: int = 256,
        n_stability_runs: int = 5,
        target_task: Optional[str] = None,
        **kw: Any,
    ) -> "Pipeline":
        """Run circuit faithfulness evaluation (6-pillar framework).

        Requires :meth:`discover` to have been called first (or the pipeline
        to have been loaded from an artifact/scores).

        Args:
            pillars: Subset of pillars to run. ``None`` or ``"all"`` runs all.
                Pass a list of ints (1-6) or names:
                ``"patching"``, ``"ablation"``, ``"stability"``,
                ``"robustness"``, ``"baselines"``, ``"generalization"``.
            n_examples: Number of evaluation examples.
            n_stability_runs: Stability pillar rediscovery count.
            target_task: Override task for cross-task generalization pillar.
            **kw: Forwarded to :func:`circuitkit.api.evaluate_circuit`.

        Returns:
            ``self`` for chaining.
        """
        self._require_circuit("evaluate")
        if not self._artifact_path or not Path(self._artifact_path).exists():
            raise RuntimeError(
                "evaluate() requires a saved artifact. "
                "Call discover() first, or use from_artifact() / from_scores() "
                "and ensure the artifact file exists on disk."
            )

        from .api import evaluate_circuit

        eval_cfg: Dict[str, Any] = {
            "num_examples": n_examples,
            "full_faithfulness_eval": True,
            "n_stability_runs": n_stability_runs,
        }
        resolved_pillars = _resolve_pillars(pillars)
        if resolved_pillars is not None:
            eval_cfg["pillars"] = resolved_pillars
        if target_task is not None:
            eval_cfg["target_task"] = target_task

        # Build config from discovery snapshot or a minimal fallback
        config = self._build_eval_config(eval_cfg)
        config.update(kw)

        self._eval_report = evaluate_circuit(config, pruned_artifact_path=self._artifact_path)
        self._history.append("evaluate")
        return self

    def evaluate_advanced(self, mode: str, **kw: Any) -> Any:
        """Run advanced evaluation modes.

        Args:
            mode: One of:
                ``"transfer"`` — Build a transfer matrix across tasks.
                    Requires ``tasks`` kwarg (list of task names).
                ``"master_grid"`` — Run the full methods × wrappers × seeds grid.
                    Requires ``methods``, ``wrappers``, ``seeds`` kwargs.
                ``"intervention_faithfulness"`` — LOO cross-validated IF metric.
                    Requires ``cells`` kwarg
                    (``Dict[(method, wrapper), dict]`` of scores).
            **kw: Mode-specific arguments.

        Returns:
            Mode-specific result object.

        Raises:
            ValueError: If ``mode`` is unknown.
        """
        if mode == "transfer":
            from .evaluation.transfer import TransferMatrix

            tasks = kw.pop("tasks")
            template = self._discovery_cfg or {
                "model": {"name": self.model_name, "precision": self.precision},
                "discovery": {"algorithm": "eap-ig", "level": "node"},
            }
            model = self._ensure_model()
            return TransferMatrix(tasks).build(model, template, **kw)

        if mode == "master_grid":
            from .evaluation.master_grid import MasterGrid

            return MasterGrid(
                kw.pop("methods"),
                self.model_name,
                kw.pop("wrappers"),
                kw.pop("seeds"),
            ).run(kw.pop("out_dir", self.output_dir))

        if mode == "intervention_faithfulness":
            from .evaluation.intervention_faithfulness import IF

            return IF().fit_loo(kw.pop("cells"))

        raise ValueError(
            f"Unknown evaluate_advanced mode: {mode!r}. "
            "Use 'transfer', 'master_grid', or 'intervention_faithfulness'."
        )

    # ------------------------------------------------------------------ #
    # Applications (delegate to quick.*)                                  #
    # ------------------------------------------------------------------ #

    def prune(
        self,
        sparsity: float = 0.3,
        scope: str = "both",
        protect_layers: Optional[List[int]] = None,
        release_original: bool = False,
        **kw: Any,
    ) -> "Pipeline":
        """Structurally prune the model using the discovered circuit.

        Delegates to :func:`circuitkit.quick.prune`. Stores the masked model
        as ``._pruned_model`` for use by :meth:`export`.

        Args:
            sparsity: Target fraction of nodes to mask.
            scope: ``"heads"``, ``"mlp"``, or ``"both"``.
            protect_layers: Layer indices to never mask.
            release_original: When True, drop the pipeline's reference to the
                unpruned ``self._model`` after the masked copy is built and
                clear the CUDA cache. Defaults to False (current behaviour:
                both the original and pruned model stay resident in GPU RAM).
                Only enable this if you don't need the unpruned model again —
                a later call that needs it (e.g. a second :meth:`prune` with
                different sparsity, or :meth:`evaluate_advanced`) will
                transparently reload it via :meth:`_ensure_model`, at the
                cost of a re-initialisation.

        Returns:
            ``self`` for chaining.
        """
        self._require_circuit("prune")
        from . import quick

        model = self._ensure_model()
        self._pruned_model = quick.prune(
            model,
            self._circuit,
            sparsity=sparsity,
            scope=scope,
            protect_layers=protect_layers,
            inplace=False,
        )

        if release_original:
            # quick.prune(inplace=False) returns a deep copy; self._model is
            # now a separate object from self._pruned_model. Only release it
            # when the caller explicitly opts in - it will be lazily reloaded
            # by _ensure_model() if any later step needs the unpruned model.
            self._model = None
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._last_intervention = "pruning"
        self._history.append("prune")
        return self

    def quantize(
        self,
        bits: int = 4,
        high_fraction: float = 0.3,
        backend: str = "quanto",
        release_original: bool = False,
        **kw: Any,
    ) -> "Pipeline":
        """Apply circuit-guided mixed-precision quantization.

        Delegates to :func:`circuitkit.quick.quantize`. Quantization modifies
        the model in place; the result is stored as ``._pruned_model``.

        Unlike :meth:`prune`, this loads a HuggingFace ``AutoModelForCausalLM``
        (cached separately as ``._hf_model``) rather than reusing the
        TransformerLens ``HookedTransformer`` from :meth:`discover` — the
        quantization backends key off ``model.config.model_type`` and the
        model's native HF module tree. That means BOTH models are resident
        after this call unless ``release_original=True``.

        Args:
            bits: Quantization bit-width for ``"llmcompressor"`` (one of
                3, 4, or 8). Ignored for ``"quanto"``, which defaults its low
                tier to ``qint4`` — pass ``low_weights=<quanto qtype>`` via
                ``**kw`` to override.
            high_fraction: Fraction of top-scoring layers to keep at high precision.
            backend: ``"quanto"`` (default) or ``"llmcompressor"``.
            release_original: When True, drop the pipeline's reference to the
                TransformerLens ``self._model`` from :meth:`discover` before
                loading the HF model, and clear the CUDA cache — halving peak
                model memory. Defaults to False (both models stay resident).
                Mirrors :meth:`prune`'s flag of the same name; a later step
                that needs the TL model reloads it via :meth:`_ensure_model`.

        Returns:
            ``self`` for chaining.
        """
        self._require_circuit("quantize")
        from . import quick

        if release_original and self._model is not None:
            # The HF model loaded below is an independent copy of the weights;
            # holding the discovery-time HookedTransformer alongside it doubles
            # resident model memory (OOM territory for multi-B models).
            self._model = None
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model = self._ensure_hf_model()
        self._quantization_plan = quick.quantize(
            model,
            self._circuit,
            bits=bits,
            high_fraction=high_fraction,
            backend=backend,
            **kw,
        )
        self._pruned_model = model
        self._last_intervention = "quantization"
        self._history.append("quantize")
        return self

    def selective_finetune(
        self,
        top_fraction: float = 0.2,
        scope: str = "both",
        **kw: Any,
    ) -> Any:
        """Select components for circuit-guided selective finetuning.

        Delegates to :func:`circuitkit.quick.selective_finetune`.

        Args:
            top_fraction: Fraction of top-scoring components to select.
            scope: ``"attn"``, ``"mlp"``, or ``"both"``.

        Returns:
            A ``SelectionResult`` with ``.attn`` and ``.mlp`` dicts.
        """
        self._require_circuit("selective_finetune")
        from . import quick

        result = quick.selective_finetune(
            self._circuit,
            model_name=self.model_name,
            top_fraction=top_fraction,
            scope=scope,
        )
        self._history.append("selective_finetune")
        return result

    def export(self, path: str, intervention: Optional[str] = None) -> str:
        """Export the intervened model as a HuggingFace checkpoint.

        Delegates to :func:`circuitkit.quick.export_checkpoint`.

        Args:
            path: Destination directory for the HF checkpoint.
            intervention: ``"pruning"`` or ``"quantization"``. ``None``
                (default) uses whichever intervention ran last — so
                ``pipe.quantize().export(path)`` exports through the
                quantization path automatically. (Before this default,
                export always assumed "pruning", which fed the quantized
                HF ``AutoModelForCausalLM`` into ``save_pruned_checkpoint``
                — a function that requires a TransformerLens model — and
                crashed.) Falls back to ``"pruning"`` if no intervention
                has been recorded.

        Returns:
            The checkpoint directory path.

        Raises:
            RuntimeError: If :meth:`prune` or :meth:`quantize` has not been
                called yet.
        """
        from . import quick

        model = self._pruned_model
        if model is None:
            raise RuntimeError(
                "export() requires a pruned/quantized model. " "Call prune() or quantize() first."
            )
        if intervention is None:
            intervention = self._last_intervention or "pruning"
        artifact = self._circuit if intervention == "pruning" else None
        result = quick.export_checkpoint(model, artifact, path, intervention=intervention)
        self._history.append("export")
        return result

    def benchmark(
        self,
        tasks: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kw: Any,
    ) -> None:
        """Run lm-evaluation-harness benchmarks on the artifact.

        Delegates to :func:`circuitkit.api.benchmark_circuit`.

        Args:
            tasks: lm-eval task names. Defaults to api default set.
            limit: Cap examples per task.
            **kw: Forwarded to ``benchmark_circuit``.
        """
        if not self._artifact_path or not Path(self._artifact_path).exists():
            raise RuntimeError("benchmark() requires a saved artifact. Call discover() first.")

        from .api import benchmark_circuit

        lm_eval_cfg: Dict[str, Any] = {"enabled": True}
        if tasks:
            lm_eval_cfg["tasks"] = tasks
        if limit is not None:
            lm_eval_cfg["limit"] = limit

        eval_params = {"lm_eval": lm_eval_cfg}
        config_for_report = self._discovery_cfg or {
            "model": {"name": self.model_name},
        }

        benchmark_circuit(
            self.model_name,
            self._artifact_path,
            eval_params,
            config_for_report,
            precision=self.precision,
            **kw,
        )
        self._history.append("benchmark")

    def visualize(self, mode: str = "graph", output: Optional[str] = None, **kw: Any) -> Any:
        """Visualize the discovered circuit.

        Delegates to :func:`circuitkit.quick.visualize_circuit`.

        Args:
            mode: ``"graph"`` (default), ``"comparison"``, or ``"dashboard"``.
            output: Path to save HTML export. ``None`` returns inline widget.
            **kw: Forwarded (e.g. ``second_circuit``, ``comparison_type``).

        Returns:
            See :func:`circuitkit.quick.visualize_circuit`.
        """
        self._require_circuit("visualize")
        from . import quick

        return quick.visualize_circuit(self._circuit, mode=mode, output=output, **kw)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        """Print a human-readable summary of the pipeline state."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Pipeline Summary")
            table.add_column("Property", style="cyan")
            table.add_column("Value")

            table.add_row("Model", self.model_name)
            table.add_row("Task", self.task or "—")
            table.add_row("Precision", self.precision)
            table.add_row("Output Dir", self.output_dir)
            table.add_row("Steps", " → ".join(self._history) if self._history else "—")

            if self._circuit is not None:
                table.add_row("Circuit Level", self._circuit.level)
                table.add_row("Circuit Size", str(len(self._circuit)))
                table.add_row("Algorithm", self._circuit.algorithm or "—")

            if self._eval_report is not None:
                # _eval_report is always a FaithfulnessReport (evaluate_circuit
                # returns one for every path as of 1.0).
                baseline = self._eval_report.patching_score
                circuit_score = self._eval_report.ablation_score
                if baseline is not None:
                    table.add_row("Pillar 1 (Patching)", f"{baseline:.4f}")
                if circuit_score is not None:
                    table.add_row("Pillar 2 (Ablation)", f"{circuit_score:.4f}")

            console.print(table)
        except ImportError:
            # Fallback without rich
            logger.info(f"Pipeline: model={self.model_name}, task={self.task}")
            logger.info(f"  Steps: {' → '.join(self._history) if self._history else 'none'}")
            if self._circuit is not None:
                logger.info(f"  Circuit: {self._circuit!r}")
            if self._eval_report is not None:
                logger.info(f"  Eval report: {self._eval_report!r}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _require_circuit(self, caller: str) -> None:
        """Raise a helpful error if no circuit is loaded yet."""
        if self._circuit is None:
            raise RuntimeError(
                f"{caller}() requires a circuit. "
                "Call discover() first, or construct via from_artifact() / from_scores()."
            )

    def _build_eval_config(self, eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build a config dict for evaluate_circuit from stored discovery state."""
        base: Dict[str, Any]
        if self._discovery_cfg is not None:
            import copy

            base = copy.deepcopy(self._discovery_cfg)
        else:
            # Minimal fallback when pipeline was loaded from artifact
            base = {
                "model": {"name": self.model_name, "precision": self.precision},
                "discovery": {
                    "algorithm": (self._circuit.algorithm if self._circuit else "eap-ig"),
                    "task": self.task or "unknown",
                    "level": (self._circuit.level if self._circuit else "node"),
                    "batch_size": 4,
                    "data_params": {"num_examples": 256},
                },
                "pruning": {"target_sparsity": 0.3, "scope": "both"},
                "output_path": self._artifact_path or "",
            }
        base["eval"] = eval_cfg
        return base

    def __repr__(self) -> str:
        circuit_info = repr(self._circuit) if self._circuit else "None"
        return (
            f"Pipeline(model={self.model_name!r}, task={self.task!r}, "
            f"circuit={circuit_info}, steps={self._history})"
        )
