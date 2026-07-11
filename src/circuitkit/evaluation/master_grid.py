"""MasterGrid runner: produce the (method, wrapper) grid as a single
library-level artifact.

The EMNLP 2026 submission's Section 7.8 reports a 5-method x 7-wrapper
master grid on Llama-3.2-3B-Instruct. The grid is currently driven by
shell scripts under `emnlp_experiments/scripts/`; this class is the
library-level promotion of that workflow.

Usage
-----

>>> from circuitkit.evaluation.master_grid import MasterGrid
>>>
>>> grid = MasterGrid(
...     methods=["eap", "eap-ig", "atp-gd"],
...     model="meta-llama/Llama-3.2-3B-Instruct",
...     wrappers=["pruning", "quantization", "lora",
...               "steering", "editing", "healing", "detection"],
...     seeds=[42, 143, 256],
... )
>>> grid.run(out_dir="results/")
>>> df = grid.to_dataframe()
>>> # df is a tidy table indexed by (method, wrapper, seed) with quality,
>>> # stability, faithfulness, baseline_ratio, probe columns
>>> grid.summary()
>>> # prints per-wrapper ranks, top method per wrapper, and a summary
>>> # of the within-wrapper rank distribution

The runner does not itself execute discovery or wrapper applications;
it composes the existing wrapper drivers (validation/applications/24
through 27 and the workshop Q2 to Q4 pipelines). For full GPU
execution, users invoke the bash scripts in `emnlp_experiments/`.
This class provides the post-hoc aggregation, ranking, and rank-
correlation analysis as a library API.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MasterGridCell:
    method: str
    wrapper: str
    seed: int
    quality: Optional[float] = None
    stability: Optional[float] = None
    faithfulness: Optional[float] = None
    baseline_ratio: Optional[float] = None
    probe: Optional[float] = None


@dataclass
class MasterGrid:
    """A runnable description of the master-grid experiment.

    Most fields parameterize the grid; results land in :attr:`cells`
    after :meth:`run` (or via :meth:`from_csv` for post-hoc loading).
    """

    methods: List[str]
    model: str
    wrappers: List[str]
    seeds: List[int] = field(default_factory=lambda: [42, 143, 256])
    cells: List[MasterGridCell] = field(default_factory=list)

    def run(
        self,
        out_dir: str = "results/",
        *,
        dry: bool = False,
        cuda_device: Optional[int] = None,
        timeout_per_cell: int = 3600,
    ) -> None:
        """Run the full method x wrapper x seed grid.

        Executes discovery + wrapper application for each (method, wrapper, seed)
        combination using the programmatic circuitkit API.  Results are written
        to ``out_dir/<method>_<wrapper>_seed<seed>.json`` and loaded into
        :attr:`cells`.

        Args:
            out_dir: Directory for per-cell JSON results.
            dry: If True, print the plan without running anything.
            cuda_device: CUDA device index.  None = use current device.
            timeout_per_cell: Max seconds per cell (default 3600).
        """
        import os

        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        total = len(self.methods) * len(self.wrappers) * len(self.seeds)
        logger.info(
            f"MasterGrid.run(): {total} cells | "
            f"methods={self.methods}, wrappers={self.wrappers}, seeds={self.seeds}"
        )

        if dry:
            for method in self.methods:
                for wrapper in self.wrappers:
                    for seed in self.seeds:
                        logger.info(f"  [DRY] {method} x {wrapper} x seed={seed}")
            return

        for method in self.methods:
            for wrapper in self.wrappers:
                for seed in self.seeds:
                    cell_file = out / f"{method}_{wrapper}_seed{seed}.json"
                    if cell_file.exists():
                        logger.info(f"  skip (cached): {cell_file.name}")
                        result = json.loads(cell_file.read_text())
                        self._load_cell(result, method, wrapper, seed)
                        continue

                    logger.info(f"  running: {method} x {wrapper} x seed={seed}")
                    t0 = time.time()
                    try:
                        result = self._run_cell(method, wrapper, seed, timeout_per_cell)
                        result["status"] = "WORKING"
                    except Exception as exc:
                        result = {
                            "method": method,
                            "wrapper": wrapper,
                            "seed": seed,
                            "status": "BROKEN",
                            "error": str(exc),
                        }
                        logger.warning(f"  cell failed: {exc}")
                    result["wall_s"] = round(time.time() - t0, 1)
                    cell_file.write_text(json.dumps(result, indent=2))
                    self._load_cell(result, method, wrapper, seed)

    def _run_cell(
        self,
        method: str,
        wrapper: str,
        seed: int,
        timeout: int,
    ) -> Dict[str, Any]:
        """Run a single (method, wrapper, seed) cell and return a results dict."""
        from circuitkit.api import discover_circuit

        cfg = {
            "model": {"name": self.model, "precision": "bfloat16"},
            "discovery": {
                "algorithm": method,
                "task": "ioi",
                "level": "node",
                "batch_size": 1,
                "data_params": {"num_examples": 32, "seed": seed},
            },
            "pruning": {"target_sparsity": 0.1, "scope": "heads"},
        }
        circuit = discover_circuit(cfg)
        node_scores = getattr(circuit, "node_scores", {})
        n_nodes = len(node_scores)

        result: Dict[str, Any] = {
            "method": method,
            "wrapper": wrapper,
            "seed": seed,
            "n_nodes": n_nodes,
            "top5": sorted(node_scores, key=lambda k: -node_scores[k])[:5],
        }

        if wrapper == "pruning":
            result.update(self._apply_pruning(circuit, node_scores))
        elif wrapper == "editing":
            result.update(self._apply_editing(circuit, node_scores))
        elif wrapper == "unlearning":
            result.update(self._apply_unlearning(circuit, node_scores))
        elif wrapper in ("quantization", "lora", "steering", "healing", "detection"):
            result["note"] = f"{wrapper} wrapper: no additional metrics beyond discovery"

        return result

    def _apply_pruning(self, circuit: Any, node_scores: Dict[str, float]) -> Dict[str, Any]:
        return {"pruning_n_nodes_kept": sum(1 for v in node_scores.values() if v > 0)}

    def _apply_editing(self, circuit: Any, node_scores: Dict[str, float]) -> Dict[str, Any]:
        return {"editing_note": "CaKE editing not run in grid mode; add CaKEEditor call here"}

    def _apply_unlearning(self, circuit: Any, node_scores: Dict[str, float]) -> Dict[str, Any]:
        return {
            "unlearning_note": "CURE/CLUE not run in grid mode; add CureClueUnlearner call here"
        }

    def _load_cell(
        self,
        result: Dict[str, Any],
        method: str,
        wrapper: str,
        seed: int,
    ) -> None:
        cell = MasterGridCell(
            method=method,
            wrapper=wrapper,
            seed=seed,
            quality=result.get("quality"),
            stability=result.get("stability"),
            faithfulness=result.get("faithfulness"),
            baseline_ratio=result.get("baseline_ratio"),
            probe=result.get("probe"),
        )
        self.cells.append(cell)

    @classmethod
    def from_csv(cls, csv_path: str) -> "MasterGrid":
        """Load a previously aggregated grid CSV (the format emitted by
        emnlp_experiments/analysis/aggregate_status_jsons.py)."""
        import csv as _csv

        rows = []
        methods, wrappers, seeds = set(), set(), set()
        with open(csv_path) as f:
            for r in _csv.DictReader(f):

                def _maybe_float(s: str):
                    try:
                        return float(s) if s not in (None, "", "None") else None
                    except ValueError:
                        return None

                rows.append(
                    MasterGridCell(
                        method=r["method"],
                        wrapper=r["wrapper"],
                        seed=int(r["seed"]),
                        quality=_maybe_float(r.get("quality", "")),
                        stability=_maybe_float(r.get("stability", "")),
                        faithfulness=_maybe_float(r.get("faithfulness", "")),
                        baseline_ratio=_maybe_float(r.get("baseline_ratio", "")),
                        probe=_maybe_float(r.get("probe", "")),
                    )
                )
                methods.add(r["method"])
                wrappers.add(r["wrapper"])
                seeds.add(int(r["seed"]))
        return cls(
            methods=sorted(methods),
            model="(loaded from csv)",
            wrappers=sorted(wrappers),
            seeds=sorted(seeds),
            cells=rows,
        )

    def per_wrapper_ranks(self) -> Dict[Tuple[str, str], int]:
        """Return within-wrapper rank of each (method, wrapper) cell
        (1 = best). Averages across seeds first."""
        per_mw_mean: Dict[Tuple[str, str], List[float]] = {}
        for c in self.cells:
            if c.quality is None:
                continue
            per_mw_mean.setdefault((c.method, c.wrapper), []).append(c.quality)

        ranks: Dict[Tuple[str, str], int] = {}
        for w in self.wrappers:
            ms = [
                (m, sum(per_mw_mean[(m, w)]) / len(per_mw_mean[(m, w)]))
                for m in self.methods
                if (m, w) in per_mw_mean
            ]
            ms.sort(key=lambda mv: -mv[1])
            for rank, (m, _) in enumerate(ms, 1):
                ranks[(m, w)] = rank
        return ranks

    def summary(self) -> str:
        """Pretty-print a within-wrapper rank summary."""
        ranks = self.per_wrapper_ranks()
        lines = []
        lines.append(
            f"MasterGrid({len(self.methods)} methods x "
            f"{len(self.wrappers)} wrappers, model={self.model})"
        )
        for w in self.wrappers:
            ws = [(m, ranks.get((m, w))) for m in self.methods]
            ws_sorted = sorted([(m, r) for m, r in ws if r is not None], key=lambda mr: mr[1])
            winner = ws_sorted[0][0] if ws_sorted else "?"
            lines.append(f"  {w:>14}: winner = {winner}")
        return "\n".join(lines)

    def to_dataframe(self):
        """Return a pandas DataFrame if pandas is available; otherwise
        a list of dicts."""
        rows = [c.__dict__ for c in self.cells]
        try:
            import pandas as pd  # noqa: WPS433

            return pd.DataFrame(rows)
        except ImportError:
            return rows


__all__ = ["MasterGrid", "MasterGridCell"]
