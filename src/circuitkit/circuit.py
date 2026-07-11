"""
The ``Circuit`` result class for CircuitKit's flat front-door API.

A :class:`Circuit` is a thin, typed wrapper around the artifacts produced by
:func:`circuitkit.discover` (and the lower-level
:func:`circuitkit.api.discover_circuit`):

* the **pruning artifact** — a list of node names (node-level discovery) or a
  ``{"mlp": ..., "heads": ..., "_meta": ...}`` dict (neuron-level discovery);
* the **node scores** — a :class:`circuitkit.artifacts.scores.CircuitScores`
  artifact mapping node names to importance scores.

The class is deliberately small: it gives researchers attribute access
(``.nodes``, ``.scores``, ``.graph``), persistence (``.save``), a graceful
``.plot()`` and Pythonic dunders (``len``, ``repr``) without forcing them to
poke at raw dicts and ``.pt`` files.

Anything this class cannot express is still reachable through the
dict-config API (:func:`circuitkit.api.discover_circuit` /
:func:`circuitkit.api.load_circuit`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union


import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids heavy imports
    from .artifacts.scores import CircuitScores

__all__ = ["Circuit"]

# A pruning artifact is either a flat list of node names (node-level) or a
# nested neuron dict (neuron-level discovery).
PruningArtifact = Union[List[str], Dict[str, Any]]


class Circuit:
    """A discovered circuit: pruning artifact + node importance scores.

    Returned by :func:`circuitkit.discover`. Wraps the graph nodes selected by
    a discovery algorithm together with their importance scores so researchers
    can inspect, persist, plot and re-use a circuit without hand-managing the
    underlying ``.pt`` / ``.json`` files.

    Attributes:
        nodes: The pruning artifact. For node-level discovery this is a
            ``list[str]`` of node names (e.g. ``["A0.1", "MLP 3"]``). For
            neuron-level discovery it is a ``dict`` with ``"mlp"``, ``"heads"``
            and ``"_meta"`` keys.
        scores: A ``dict[str, float]`` mapping every candidate node name to its
            importance score. Empty if the discovery run produced no score
            side-car (e.g. a bare neuron-level artifact).
        graph: The underlying EAP ``Graph`` object if one was supplied,
            otherwise ``None``. Most flat-API runs do not retain the live
            graph; drop to :func:`circuitkit.api.discover_circuit` if you need
            it.
        circuit_scores: The raw :class:`CircuitScores` artifact when available,
            carrying metadata (algorithm, task, model, timestamp).
        task: Task name the circuit was discovered for, if known.
        algorithm: Discovery algorithm used, if known.
        model_name: Model the circuit was discovered on, if known.
        level: ``"node"`` or ``"neuron"``.

    Example:
        >>> import circuitkit as ck
        >>> model = ck.load_model("gpt2")
        >>> circuit = ck.discover(model, "ioi", algorithm="eap-ig", n_examples=16)
        >>> len(circuit)               # number of nodes in the circuit
        12
        >>> circuit.scores["A0.1"]     # importance of a specific head
        0.0731
        >>> circuit.save("ioi_circuit.pt")
        >>> circuit.plot("ioi_circuit.html")   # interactive HTML, deps permitting
    """

    def __init__(
        self,
        nodes: PruningArtifact,
        scores: Optional[Dict[str, float]] = None,
        *,
        graph: Optional[Any] = None,
        circuit_scores: Optional["CircuitScores"] = None,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        model_name: Optional[str] = None,
        level: str = "node",
        artifact_path: Optional[str] = None,
    ) -> None:
        """Construct a Circuit. Researchers normally get one from :func:`discover`.

        Args:
            nodes: Pruning artifact — a node-name list or a neuron dict.
            scores: Optional mapping of node name to importance score.
            graph: Optional underlying EAP ``Graph`` object.
            circuit_scores: Optional :class:`CircuitScores` artifact (carries
                metadata and is preferred over ``scores`` for ``.plot()``).
            task: Task name, for metadata / ``repr``.
            algorithm: Discovery algorithm name, for metadata / ``repr``.
            model_name: Model name, for metadata / ``repr``.
            level: ``"node"`` or ``"neuron"``.
            artifact_path: Path the pruning artifact was loaded from / saved to.
        """
        self.nodes: PruningArtifact = nodes
        self.scores: Dict[str, float] = dict(scores) if scores else {}
        self.graph: Optional[Any] = graph
        self.circuit_scores: Optional["CircuitScores"] = circuit_scores
        self.task: Optional[str] = task
        self.algorithm: Optional[str] = algorithm
        self.model_name: Optional[str] = model_name
        self.level: str = level
        self.artifact_path: Optional[str] = artifact_path

        # Backfill metadata from the CircuitScores artifact when available.
        if circuit_scores is not None:
            self.task = self.task or getattr(circuit_scores, "task", None)
            self.algorithm = self.algorithm or getattr(circuit_scores, "algorithm", None)
            self.model_name = self.model_name or getattr(circuit_scores, "model", None)
            if not self.scores:
                self.scores = dict(getattr(circuit_scores, "node_scores", {}) or {})

    # ------------------------------------------------------------------ #
    # Constructors                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_artifact(
        cls,
        artifact_path: Union[str, Path],
        *,
        scores_path: Optional[Union[str, Path]] = None,
        graph: Optional[Any] = None,
    ) -> "Circuit":
        """Load a Circuit from a discovery artifact written to disk.

        Reconstructs a :class:`Circuit` from the ``.pt`` pruning artifact and
        its ``_scores.json`` side-car (auto-derived if not given), exactly the
        files :func:`circuitkit.api.discover_circuit` writes.

        Args:
            artifact_path: Path to the ``.pt`` pruning artifact.
            scores_path: Optional explicit path to the ``_scores.json`` (or
                ``_scores.pt``) side-car. Auto-derived from ``artifact_path``
                when omitted.
            graph: Optional underlying EAP ``Graph`` to attach.

        Returns:
            A populated :class:`Circuit`.

        Raises:
            FileNotFoundError: If ``artifact_path`` does not exist.

        Example:
            >>> circuit = Circuit.from_artifact("results/ioi_circuit.pt")
            >>> len(circuit)
            12
        """
        import torch  # local import keeps `import circuitkit` light

        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Circuit artifact not found: {artifact_path}")

        # weights_only=True: circuit artifacts are shared between users, so an
        # artifact is untrusted input. The payload is only a list of node names
        # or a dict of primitives + tensors, all of which the safe unpickler
        # supports — this blocks pickle-based RCE (CWE-502) without losing data.
        nodes = torch.load(artifact_path, map_location="cpu", weights_only=True)
        level = "neuron" if isinstance(nodes, dict) and "_meta" in nodes else "node"

        circuit_scores = None
        scores: Dict[str, float] = {}
        # Resolve the scores side-car: explicit path, then JSON, then .pt.
        candidates: List[Path] = []
        if scores_path is not None:
            candidates.append(Path(scores_path))
        else:
            stem = artifact_path.stem
            candidates.append(artifact_path.parent / f"{stem}_scores.json")
            candidates.append(artifact_path.parent / f"{stem}_scores.pt")

        for cand in candidates:
            if not cand.exists():
                continue
            if cand.suffix == ".json":
                import json as _json

                with open(cand, encoding="utf-8") as f:
                    blob = _json.load(f)
                # A full CircuitScores artifact carries metadata; a bare
                # side-car (written by Circuit.save without metadata) only
                # has {"node_scores": {...}}.
                if "algorithm" in blob and "task" in blob:
                    from .artifacts.scores import CircuitScores

                    circuit_scores = CircuitScores.from_dict(blob)
                    scores = dict(circuit_scores.node_scores)
                else:
                    scores = {k: float(v) for k, v in blob.get("node_scores", {}).items()}
            else:
                # Untrusted side-car (auto-discovered next to the artifact);
                # weights_only=True blocks pickle RCE. Payload is a score dict.
                blob = torch.load(cand, map_location="cpu", weights_only=True)
                if isinstance(blob, dict) and "node_scores" in blob:
                    raw = blob["node_scores"]
                    scores = {
                        k: float(v.item() if hasattr(v, "item") else v) for k, v in raw.items()
                    }
            break

        return cls(
            nodes,
            scores,
            graph=graph,
            circuit_scores=circuit_scores,
            level=level,
            artifact_path=str(artifact_path),
        )

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> Path:
        """Save the circuit's pruning artifact (and a scores side-car).

        Writes the pruning artifact to ``path`` and, when scores are present, a
        ``<stem>_scores.json`` side-car next to it — the same layout
        :func:`circuitkit.api.discover_circuit` produces, so the saved files
        round-trip through :meth:`from_artifact`, :func:`circuitkit.api.load_circuit`
        and :func:`circuitkit.api.evaluate_circuit`.

        Args:
            path: Destination path for the ``.pt`` pruning artifact.

        Returns:
            The resolved :class:`~pathlib.Path` the artifact was written to.

        Example:
            >>> circuit.save("results/ioi_circuit.pt")
            PosixPath('results/ioi_circuit.pt')
        """
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.nodes, path)
        self.artifact_path = str(path)

        if self.circuit_scores is not None:
            self.circuit_scores.to_json(path.parent / f"{path.stem}_scores.json")
        elif self.scores:
            import json

            with open(path.parent / f"{path.stem}_scores.json", "w", encoding="utf-8") as f:
                json.dump({"node_scores": self.scores}, f, indent=2)
        return path

    # ------------------------------------------------------------------ #
    # Visualisation                                                      #
    # ------------------------------------------------------------------ #
    def plot(self, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Render the circuit as an interactive graph.

        Delegates to :class:`circuitkit.visualize.graph_viz.CircuitGraphVisualizer`.
        If the visualization stack's optional dependencies are missing, the
        method degrades gracefully: it prints a short textual summary and
        returns ``None`` instead of raising.

        Args:
            output_path: Optional path for an HTML export. When omitted, an
                in-Jupyter interactive widget is returned where supported.

        Returns:
            The HTML string when ``output_path`` is given and rendering
            succeeded, otherwise ``None``.

        Example:
            >>> circuit.plot("ioi_circuit.html")   # writes an interactive HTML
        """
        if not self.scores:
            logger.info(
                "Circuit has no node scores to plot. "
                "Re-run discovery with an output_path to capture scores."
            )
            return None
        try:
            from .artifacts.scores import CircuitScores
            from .visualize.graph_viz import CircuitGraphVisualizer


            cs = self.circuit_scores
            if cs is None:
                cs = CircuitScores(
                    task=self.task or "unknown",
                    model=self.model_name or "unknown",
                    algorithm=self.algorithm or "eap-ig",
                    level="node",
                    node_scores=self.scores,
                    timestamp=CircuitScores.create_timestamp(),
                )

            # circuit.nodes is the *pruned* (removed) node list produced by
            # discover_circuit(). The "in-circuit" nodes are scores.keys() minus
            # this pruned set. We pass the pruned list so the visualizer can
            # correctly distinguish kept vs removed nodes.
            pruned_nodes: list = []
            if isinstance(self.nodes, list):
                pruned_nodes = self.nodes
            elif isinstance(self.nodes, dict):
                # neuron-level artifact: flatten mlp + heads keys
                pruned_nodes = (
                    list(self.nodes.get("mlp", {}).keys())
                    + list(self.nodes.get("heads", {}).keys())
                )

            graph_dict = {"nodes": {name: {} for name in self.scores}, "edges": []}
            viz = CircuitGraphVisualizer(graph_dict, cs, pruned_nodes=pruned_nodes)

            if output_path is not None:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                title = f"Circuit ({self.task or 'unknown'} / {self.algorithm or 'eap-ig'})"
                return viz.to_html(str(out), title=title)
            return viz.interactive_widget()
        except Exception as exc:  # pragma: no cover - depends on optional deps
            logger.info(
                f"Plotting unavailable ({type(exc).__name__}: {exc}). " f"Circuit summary: {self!r}"
            )
            return None

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #
    def top_nodes(self, k: int = 10) -> Dict[str, float]:
        """Return the ``k`` highest-scoring nodes as a name -> score dict.

        Args:
            k: Number of top nodes to return.

        Returns:
            An ordered ``dict`` of the ``k`` most important nodes. Empty if no
            scores are available.

        Example:
            >>> circuit.top_nodes(3)
            {'A9.6': 0.91, 'A9.9': 0.88, 'MLP 10': 0.74}
        """
        if not self.scores:
            return {}
        ranked = sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)
        return dict(ranked[:k])

    # ------------------------------------------------------------------ #
    # Dunders                                                            #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Number of nodes/components in the circuit."""
        if isinstance(self.nodes, dict):
            mlp = self.nodes.get("mlp", {}) or {}
            heads = self.nodes.get("heads", {}) or {}
            return len(mlp) + len(heads)
        return len(self.nodes)

    def __iter__(self):
        """Iterate over node names (node-level circuits only)."""
        if isinstance(self.nodes, dict):
            return iter(list(self.nodes.get("mlp", {})) + list(self.nodes.get("heads", {})))
        return iter(self.nodes)

    def __contains__(self, node: object) -> bool:
        """Membership test against the circuit's node names."""
        if isinstance(self.nodes, dict):
            return node in self.nodes.get("mlp", {}) or node in self.nodes.get("heads", {})
        return node in self.nodes

    def __repr__(self) -> str:
        """Concise one-line summary of the circuit."""
        bits = [f"level={self.level}", f"n_nodes={len(self)}"]
        if self.algorithm:
            bits.append(f"algorithm={self.algorithm!r}")
        if self.task:
            bits.append(f"task={self.task!r}")
        if self.model_name:
            bits.append(f"model={self.model_name!r}")
        bits.append(f"n_scored={len(self.scores)}")
        return f"Circuit({', '.join(bits)})"
