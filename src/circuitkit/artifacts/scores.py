"""
Unified CircuitScores artifact schema across all CircuitKIT backends.

All algorithms (EAP, EAP-IG, ACDC, IBCircuit) emit this format for
node-level circuit discovery. This provides a single contract for:
- JSON serialization/deserialization
- Score access and normalization
- Metadata tracking (algorithm, task, model, timestamp)
"""

import json
import re
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_HEAD = re.compile(r"A(\d+)\.(\d+)$")
_MLP = re.compile(r"(?:MLP |L)(\d+)(?:\.\d+)?$")  # "MLP 3" (node) or "L3.42" (neuron)
# Hugging Face projection names per registry family, minus the non-module keys.
_NON_MODULE_KEYS = {"module", "head_dim"}


def _model_type(model: Optional[str]) -> Optional[str]:
    """HF ``model_type`` of a TL alias, local dir or cached Hub id; ``None`` if unknown. No network."""
    if not model:
        return None
    try:
        from transformer_lens.loading_from_pretrained import get_official_model_name

        model = get_official_model_name(model)
    except Exception:  # noqa: BLE001 - a local dir or unregistered id stays as is
        pass
    try:
        from transformers import PretrainedConfig

        return PretrainedConfig.get_config_dict(model, local_files_only=True)[0].get("model_type")
    except Exception:  # noqa: BLE001 - not a local dir and not in the HF cache
        return None


def interop_fields(
    node_scores: Dict[str, float],
    model: Optional[str] = None,
    *,
    method: str = "discover",
    top_fraction: float = 0.2,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Keys written next to ``node_scores`` in every ``*_scores.json``.

    * ``safety_units`` / ``layer_suggestions``: the format SafeTune's
      ``core/circuit_kit/adapter.py`` reads, derived from the top ``top_fraction``
      of nodes by score. ``unit_ids`` are the node names, ``module_names`` their HF
      modules (``model.layers.3.self_attn``), ``activation_correlation`` the scores
      divided by the largest one, and ``target_modules`` the HF projection names of
      the node kinds selected (attention heads and/or MLPs), named per
      :mod:`circuitkit.applications.arch_registry` (Llama names when the family is
      unknown).
    * ``provenance``: a ``lexsi.provenance/1`` record (see :mod:`circuitkit.provenance`).
    """
    from ..applications.arch_registry import MODEL_ARCH_REGISTRY, get_model_family
    from ..provenance import make_provenance, read_provenance

    try:
        arch = MODEL_ARCH_REGISTRY[get_model_family(_model_type(model))]
    except KeyError:
        arch = MODEL_ARCH_REGISTRY["llama"]
    ranked = sorted(node_scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranked[: max(1, round(len(ranked) * top_fraction))] if ranked else []
    peak = max((v for _, v in top), default=0.0) or 1.0

    units, layers, kinds = [], set(), {}
    for name, score in top:
        m_head, m_mlp = _HEAD.match(name), _MLP.match(name)
        if not (m_head or m_mlp):
            continue
        layer, kind = int((m_head or m_mlp).group(1)), "attn" if m_head else "mlp"
        module = "mlp" if kind == "mlp" else arch["attn"]["module"]
        units.append((name, f"{arch['layers_path'][0]}.{layer}.{module}", score / peak))
        layers.add(layer)
        kinds[kind] = max(kinds.get(kind, 0.0), score / peak)
    targets = {
        proj: weight
        for kind, weight in kinds.items()
        for key, proj in arch[kind].items()
        if key not in _NON_MODULE_KEYS and proj
    }
    meta = {"source": "circuitkit node_scores", "top_fraction": top_fraction}
    return {
        "safety_units": {
            "layer_indices": sorted(layers),
            "module_names": sorted({u[1] for u in units}),
            "unit_ids": [u[0] for u in units],
            "activation_correlation": {u[0]: u[2] for u in units},
            "metadata": meta,
        },
        "layer_suggestions": {
            "target_modules": list(targets),
            "layer_subset": sorted(layers),
            "priority": targets,
            "metadata": meta,
        },
        "provenance": make_provenance(
            method,
            inputs=[{"kind": "model", "ref": model, "provenance": read_provenance(model)}],
            params=params,
        ),
    }


@dataclass
class CircuitScores:
    """
    Unified scores artifact across all CircuitKIT algorithms.

    Represents node-level importance scores from circuit discovery.
    All backends (EAP, ACDC, IBCircuit) convert their outputs to this
    format for consistent downstream processing.

    Attributes:
        task (str): Task name (e.g., 'ioi', 'mmlu', 'capital_country').
        model (str): Model identifier (e.g., 'gpt2', 'pythia-70m').
        algorithm (str): Discovery algorithm used ('eap', 'eap-ig', 'acdc', 'ibcircuit').
        level (str): Granularity of scores ('node' or 'neuron').

        node_scores (Dict[str, float]): Importance scores keyed by node name.
            - Attention heads: 'A{layer}.{head}' (e.g., 'A0.1')
            - MLPs: 'MLP {layer}' (e.g., 'MLP 3')
            - Values are absolute importance scores (non-negative).

        timestamp (str): ISO 8601 datetime when scores were generated.
        version (str): Schema version for backward compatibility (default '1.0').
        discovery_cfg (Optional[Dict]): Full discovery configuration for reproducibility.
    """

    task: str
    model: str
    algorithm: str
    level: str
    node_scores: Dict[str, float]
    timestamp: str
    version: str = "1.0"
    discovery_cfg: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate schema constraints."""
        from ..utils.exceptions import SUPPORTED_ALGORITHMS

        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {sorted(SUPPORTED_ALGORITHMS)}, "
                f"got {self.algorithm!r}"
            )
        if self.level not in {"node", "neuron"}:
            raise ValueError(f"level must be 'node' or 'neuron', got {self.level!r}")
        # Neuron-level scores use the same node_scores dict with keys of
        # the form "L{layer}.{neuron_idx}" (e.g. "L3.42"). No additional
        # schema validation is required beyond the numeric check below.

        # Validate node_scores format
        for name, score in self.node_scores.items():
            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"All scores must be numeric; got {type(score).__name__} " f"for node {name!r}"
                )
            if score < 0:
                raise ValueError(
                    f"All scores must be non-negative; got {score} " f"for node {name!r}"
                )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitScores":
        """
        Construct CircuitScores from a dictionary.

        Handles schema evolution: if version is absent, assumes '1.0'.
        """
        # Drop keys that are not fields (interop_fields: safety_units, provenance, ...).
        names = {f.name for f in fields(cls)}
        data_copy = {k: v for k, v in data.items() if k in names}
        if "version" not in data_copy:
            data_copy["version"] = "1.0"
        if "discovery_cfg" not in data_copy:
            data_copy["discovery_cfg"] = {}
        return cls(**data_copy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, path: Path, *, top_fraction: float = 0.2) -> None:
        """
        Save CircuitScores to JSON file, plus the :func:`interop_fields` keys.

        Args:
            path (Path): Output path (typically .json).
            top_fraction (float): Share of nodes (by score) listed as SafeTune
                ``safety_units`` / ``layer_suggestions``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data.update(
            interop_fields(
                self.node_scores,
                self.model,
                method=f"discover.{self.algorithm}",
                top_fraction=top_fraction,
                params={"task": self.task, "level": self.level},
            )
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "CircuitScores":
        """
        Load CircuitScores from JSON file.

        Args:
            path (Path): Input path (typically .json).

        Returns:
            CircuitScores instance.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def normalize_scores(self, method: str = "minmax") -> Dict[str, float]:
        """
        Return normalized scores (original scores unchanged).

        Args:
            method (str): Normalization method.
                - 'minmax': Rescale to [0, 1] using min-max normalization.
                - 'zscore': Standardize to mean=0, std=1.

        Returns:
            Dict[str, float]: Normalized scores by node name.
        """
        if not self.node_scores:
            return {}

        scores = list(self.node_scores.values())

        if method == "minmax":
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                # All scores equal; return 1.0
                return {name: 1.0 for name in self.node_scores.keys()}
            return {
                name: (score - min_score) / (max_score - min_score)
                for name, score in self.node_scores.items()
            }

        elif method == "zscore":
            import numpy as np

            mean = np.mean(scores)
            std = np.std(scores)
            if std == 0:
                # All scores equal; return 0.0
                return {name: 0.0 for name in self.node_scores.keys()}
            return {name: (score - mean) / std for name, score in self.node_scores.items()}

        else:
            raise ValueError(f"method must be 'minmax' or 'zscore', got {method!r}")

    def top_k_nodes(self, k: int) -> Dict[str, float]:
        """
        Return the top-k highest-scoring nodes.

        Args:
            k (int): Number of top nodes to return.

        Returns:
            Dict[str, float]: Top-k nodes by importance (descending).
        """
        sorted_nodes = sorted(self.node_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_nodes[:k])

    def bottom_k_nodes(self, k: int) -> Dict[str, float]:
        """
        Return the bottom-k lowest-scoring nodes (candidates for pruning).

        Args:
            k (int): Number of bottom nodes to return.

        Returns:
            Dict[str, float]: Bottom-k nodes by importance (ascending).
        """
        sorted_nodes = sorted(self.node_scores.items(), key=lambda x: x[1], reverse=False)
        return dict(sorted_nodes[:k])

    @staticmethod
    def create_timestamp() -> str:
        """Generate ISO 8601 timestamp."""
        return datetime.utcnow().isoformat() + "Z"
