"""
Cross-task circuit transfer matrix builder and analyzer.

Discovers circuits on a source task and evaluates them on target tasks,
building an NxN transfer matrix showing circuit effectiveness across tasks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from circuitkit.utils.device import get_device, empty_cache
from ..utils.logging import get_logger


class TransferMatrix:
    """
    Build and analyze cross-task circuit transfer.

    A TransferMatrix discovers a circuit on one source task and evaluates
    it on multiple target tasks, building an NxN matrix of transfer scores.
    This reveals which circuits generalize across tasks and which are
    task-specific.
    """

    def __init__(self, task_names: List[str]):
        """
        Initialize transfer matrix for given tasks.

        Args:
            task_names (List[str]): List of task names (e.g., ['ioi', 'sva', 'greater_than']).
                Tasks must be registered in CircuitKit's task registry.
        """
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.matrix = None
        self.logger = get_logger("circuitkit.transfer")
        self._circuits = {}  # Store discovered circuits for reuse
        self._eval_scores = {}  # Store evaluation scores for analysis

    def build(
        self,
        model,
        discovery_cfg_template: Dict[str, Any],
        eval_cfg_template: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        save_dir: Optional[str] = None,
        skip_diagonal: bool = False,
    ) -> np.ndarray:
        """
        Build NxN cross-task transfer matrix.

        For each source task i:
            1. Discover circuit on task i using discovery_cfg_template
            2. For each target task j:
               - Evaluate discovered circuit on task j
               - Record transfer score in matrix[i, j]

        Args:
            model: HookedTransformer model instance.
            discovery_cfg_template (Dict[str, Any]): Discovery config template.
                Will be used for each source task, with 'task' field replaced.
                Must have keys: 'algorithm', 'task', 'pruning', etc.
            eval_cfg_template (Optional[Dict[str, Any]]): Evaluation config template.
                If None, uses discovery_cfg_template.
            device (str): Device to run on ('cuda', 'cpu'). Defaults to 'cuda'.
            save_dir (Optional[str]): If provided, save circuits and results to this dir.
            skip_diagonal (bool): If True, skip diagonal (source=target) evaluations.
                Defaults to False.

        Returns:
            np.ndarray: NxN transfer matrix where matrix[i, j] is the transfer
                score when circuit discovered on task i is evaluated on task j.

        Raises:
            ValueError: If task_names not found in registry.
            RuntimeError: If discovery or evaluation fails.
        """
        from ..api import discover_circuit, evaluate_circuit
        from ..tasks.bootstrap import _bootstrap_builtin_tasks
        from ..tasks.registry import get_task

        # Built-in task registration is lazy; the raw registry.get_task below
        # does not trigger it (and load_model() doesn't either), so bootstrap
        # explicitly here or every task lookup fails with "registered tasks: []".
        _bootstrap_builtin_tasks()

        self.matrix = np.zeros((self.n_tasks, self.n_tasks))

        # Create save directory if needed. Each source circuit MUST be
        # persisted to a distinct path so the evaluation phase can load the
        # right per-source artifact: evaluate_circuit restricts to the circuit
        # named by pruned_artifact_path, and if that path is None it silently
        # falls back to a single shared auto-saved ./outputs/*.pt (overwritten
        # by the last discovery) — making every matrix cell source-independent
        # (identical rows). When the caller gives no save_dir, use a private
        # temp dir so artifacts are still per-source.
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            import tempfile

            save_path = Path(tempfile.mkdtemp(prefix="ck_transfer_"))

        self.logger.info(
            f"Building {self.n_tasks}x{self.n_tasks} transfer matrix "
            f"for tasks: {self.task_names}"
        )

        # Validate all tasks exist
        for task_name in self.task_names:
            try:
                get_task(task_name)
            except Exception as e:
                raise ValueError(f"Task '{task_name}' not found: {e}")

        # Discovery phase: for each source task
        for src_idx, source_task in enumerate(tqdm(self.task_names, desc="Discovery")):
            self.logger.info(f"[{src_idx + 1}/{self.n_tasks}] Discovering circuit on {source_task}")

            # Build discovery config for this source task. discover_circuit
            # expects the nested schema (model / discovery / pruning), with the
            # task under `discovery`. Rebuild the `discovery` sub-dict per
            # iteration so we don't mutate the caller's template.
            discovery_cfg = {**discovery_cfg_template}
            discovery_cfg["discovery"] = {
                **discovery_cfg_template.get("discovery", {}),
                "task": source_task,
            }

            # Define output path for circuit artifact. Always set (save_path is
            # a temp dir when the caller gave no save_dir) so each source gets
            # its own artifact and the eval phase loads the correct circuit.
            circuit_path = str(save_path / f"circuit_{source_task}_discovery.pt")
            discovery_cfg["output_path"] = circuit_path

            try:
                # Run discovery
                result = discover_circuit(discovery_cfg)
                self._circuits[source_task] = {
                    "artifact": result,
                    "path": circuit_path,
                    "config": discovery_cfg,
                }
                self.logger.debug(f"Circuit discovered on {source_task}, saved to {circuit_path}")
            except Exception as e:
                self.logger.error(f"Discovery failed on {source_task}: {e}")
                raise RuntimeError(f"Failed to discover circuit on {source_task}: {e}")

        # Evaluation phase: for each source-target pair
        total_evals = (
            self.n_tasks * self.n_tasks if not skip_diagonal else self.n_tasks * (self.n_tasks - 1)
        )
        eval_bar = tqdm(total=total_evals, desc="Evaluation")

        for src_idx, source_task in enumerate(self.task_names):
            for tgt_idx, target_task in enumerate(self.task_names):
                if skip_diagonal and src_idx == tgt_idx:
                    self.logger.debug(f"Skipping diagonal: {source_task} -> {source_task}")
                    continue

                eval_bar.update(1)

                self.logger.info(
                    f"Evaluating {source_task} circuit on {target_task} "
                    f"({src_idx * self.n_tasks + tgt_idx + 1}/{total_evals})"
                )

                # Build evaluation config. Like the discovery phase,
                # evaluate_circuit needs the task under `discovery`, not at the
                # top level, or every cell errors with "requires discovery.task"
                # and the whole matrix comes back NaN.
                eval_cfg = {**discovery_cfg_template}
                if eval_cfg_template:
                    eval_cfg.update(eval_cfg_template)
                eval_cfg["discovery"] = {
                    **eval_cfg.get("discovery", {}),
                    "task": target_task,
                }

                try:
                    # Evaluate circuit from source_task on target_task
                    circuit_artifact_path = self._circuits[source_task]["path"]

                    # Use the saved circuit artifact
                    result = evaluate_circuit(
                        config=eval_cfg,
                        pruned_artifact_path=circuit_artifact_path,
                    )

                    # Transfer score = circuit (ablation) performance on the target task.
                    transfer_score = result.ablation_score or 0.0
                    self.matrix[src_idx, tgt_idx] = transfer_score

                    self._eval_scores[(source_task, target_task)] = {
                        "score": transfer_score,
                        "full_result": result,
                    }

                    self.logger.debug(
                        f"Transfer {source_task}->{target_task}: {transfer_score:.4f}"
                    )
                except Exception as e:
                    self.logger.error(f"Evaluation failed {source_task}->{target_task}: {e}")
                    # Set score to NaN to indicate failure, but continue
                    self.matrix[src_idx, tgt_idx] = np.nan

        eval_bar.close()

        # Save matrix if requested
        if save_dir:
            matrix_path = str(save_path / "transfer_matrix.npy")
            np.save(matrix_path, self.matrix)
            self.logger.info(f"Transfer matrix saved to {matrix_path}")

        return self.matrix

    def analyze(self) -> Dict[str, Any]:
        """
        Compute statistics from the transfer matrix.

        Computes:
        - Per-source averages (how well each circuit generalizes)
        - Per-target averages (which tasks are easiest to transfer to)
        - Best/worst transfers
        - Overall transfer statistics

        Returns:
            Dict[str, Any] with keys:
                'source_avg': Dict[str, float] - Mean score per source task
                'target_avg': Dict[str, float] - Mean score per target task
                'best_transfer': Tuple[str, str, float] - (source, target, score)
                'worst_transfer': Tuple[str, str, float] - (source, target, score)
                'overall_mean': float - Mean of all non-NaN transfers
                'overall_std': float - Std of all non-NaN transfers
                'num_successful': int - Count of non-NaN transfers
                'high_transfer_pairs': List[Tuple[str, str, float]] - Transfers >= threshold
        """
        if self.matrix is None:
            raise RuntimeError("Matrix not yet built. Call build() first.")

        # Mask out NaN values for statistics
        valid_mask = ~np.isnan(self.matrix)

        # Per-source averages (ignore NaN)
        source_avg = {}
        for src_idx, src_task in enumerate(self.task_names):
            valid_scores = self.matrix[src_idx, valid_mask[src_idx]]
            if len(valid_scores) > 0:
                source_avg[src_task] = float(np.mean(valid_scores))
            else:
                source_avg[src_task] = np.nan

        # Per-target averages
        target_avg = {}
        for tgt_idx, tgt_task in enumerate(self.task_names):
            valid_scores = self.matrix[valid_mask[:, tgt_idx], tgt_idx]
            if len(valid_scores) > 0:
                target_avg[tgt_task] = float(np.mean(valid_scores))
            else:
                target_avg[tgt_task] = np.nan

        # Find best and worst transfers
        valid_scores = self.matrix[valid_mask]
        best_idx = np.unravel_index(np.argmax(self.matrix), self.matrix.shape)
        worst_idx = np.nanargmin(self.matrix) if np.any(~np.isnan(self.matrix)) else None

        best_transfer = (
            self.task_names[best_idx[0]],
            self.task_names[best_idx[1]],
            float(self.matrix[best_idx]),
        )

        worst_transfer = None
        if worst_idx is not None:
            worst_idx = np.unravel_index(worst_idx, self.matrix.shape)
            worst_transfer = (
                self.task_names[worst_idx[0]],
                self.task_names[worst_idx[1]],
                float(self.matrix[worst_idx]),
            )

        # Find high-transfer pairs (>= 0.5)
        high_transfer_pairs = []
        for src_idx, src_task in enumerate(self.task_names):
            for tgt_idx, tgt_task in enumerate(self.task_names):
                score = self.matrix[src_idx, tgt_idx]
                if not np.isnan(score) and score >= 0.5:
                    high_transfer_pairs.append((src_task, tgt_task, float(score)))

        high_transfer_pairs.sort(key=lambda x: x[2], reverse=True)

        result = {
            "source_avg": source_avg,
            "target_avg": target_avg,
            "best_transfer": best_transfer,
            "worst_transfer": worst_transfer,
            "overall_mean": float(np.mean(valid_scores)) if len(valid_scores) > 0 else np.nan,
            "overall_std": float(np.std(valid_scores)) if len(valid_scores) > 0 else np.nan,
            "num_successful": int(np.sum(valid_mask)),
            "high_transfer_pairs": high_transfer_pairs,
        }

        return result

    def summary(self, threshold: float = 0.5) -> str:
        """
        Generate a human-readable summary of transfer analysis.

        Args:
            threshold (float): Report transfers >= threshold. Defaults to 0.5.

        Returns:
            str: Formatted summary text.
        """
        if self.matrix is None:
            return "Transfer matrix not built yet."

        analysis = self.analyze()

        lines = [
            "=" * 70,
            "CROSS-TASK TRANSFER MATRIX ANALYSIS",
            "=" * 70,
            "",
            "Transfer Matrix (rows=source, cols=target):",
            str(np.round(self.matrix, 4)),
            "",
            "Per-Source Task Generalization (avg transfer score):",
        ]

        for task, avg in sorted(analysis["source_avg"].items()):
            if np.isnan(avg):
                lines.append(f"  {task:20s}: NaN")
            else:
                lines.append(f"  {task:20s}: {avg:.4f}")

        lines.extend(
            [
                "",
                "Per-Target Task Difficulty (avg incoming transfer score):",
            ]
        )

        for task, avg in sorted(analysis["target_avg"].items()):
            if np.isnan(avg):
                lines.append(f"  {task:20s}: NaN")
            else:
                lines.append(f"  {task:20s}: {avg:.4f}")

        if analysis["best_transfer"]:
            src, tgt, score = analysis["best_transfer"]
            lines.append(f"\nBest Transfer:  {src:20s} -> {tgt:20s} ({score:.4f})")

        if analysis["worst_transfer"]:
            src, tgt, score = analysis["worst_transfer"]
            lines.append(f"Worst Transfer: {src:20s} -> {tgt:20s} ({score:.4f})")

        lines.append(f"\nOverall Mean:   {analysis['overall_mean']:.4f}")
        lines.append(f"Overall Std:    {analysis['overall_std']:.4f}")
        lines.append(f"Success Rate:   {analysis['num_successful']}/{self.n_tasks**2}")

        if analysis["high_transfer_pairs"]:
            lines.append(f"\nHigh-Transfer Pairs (>= {threshold}):")
            for src, tgt, score in analysis["high_transfer_pairs"]:
                lines.append(f"  {src:20s} -> {tgt:20s} ({score:.4f})")

        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize transfer matrix and analysis to dict.

        Useful for saving results or integration with other tools.

        Returns:
            Dict with keys:
                'task_names': List[str]
                'matrix': np.ndarray (NxN)
                'analysis': Dict from analyze()
        """
        return {
            "task_names": self.task_names,
            "matrix": self.matrix,
            "analysis": self.analyze(),
        }

    def to_json(self, path: Path) -> None:
        """
        Save transfer matrix and analysis to JSON file.

        Converts NaN to null and numpy types to JSON-serializable formats.

        Args:
            path (Path): Path to save JSON file.
        """
        import json

        def _convert_for_json(obj):
            """Convert numpy types and NaN to JSON-serializable formats."""
            if isinstance(obj, np.ndarray):
                # Recurse so NaN entries inside the array become null.
                return [_convert_for_json(v) for v in obj]
            elif isinstance(obj, (float, np.floating)):
                # Return None for NaN, otherwise return the float value
                return None if np.isnan(obj) else float(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: _convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_convert_for_json(v) for v in obj]
            elif obj is None or isinstance(obj, (str, bool)):
                return obj
            else:
                return str(obj)

        # Build JSON-serializable structure
        data = {
            "task_names": self.task_names,
            "matrix": _convert_for_json(self.matrix),
            "analysis": _convert_for_json(self.analyze()),
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Transfer matrix and analysis saved to {path}")

    def visualize(
        self,
        output_dir: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> Dict[str, str]:
        """
        Generate visualizations of the transfer matrix.

        Requires matplotlib and seaborn.

        Args:
            output_dir (Optional[str]): Directory to save visualizations.
                If None, visualizations are shown in-place (Jupyter).
            figsize (Tuple[int, int]): Figure size for plots.

        Returns:
            Dict[str, str]: Paths to saved visualization files.

        Raises:
            RuntimeError: If matrix not built yet.
            ImportError: If visualization dependencies missing.
        """
        if self.matrix is None:
            raise RuntimeError("Matrix not yet built. Call build() first.")

        try:
            from .transfer_visualizer import TransferMatrixVisualizer
        except ImportError:
            raise ImportError(
                "Visualization requires matplotlib and seaborn. "
                "Install with: pip install matplotlib seaborn"
            )

        visualizer = TransferMatrixVisualizer(self.task_names, figsize=figsize)
        analysis = self.analyze()

        if output_dir:
            return {
                "saved_files": visualizer.save_all_visualizations(self.matrix, analysis, output_dir)
            }
        else:
            visualizer.heatmap(self.matrix)
            visualizer.per_task_averages(self.matrix, analysis)
            visualizer.distribution_plot(self.matrix, analysis)
            return {"message": "Visualizations displayed"}

    def statistical_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on the transfer matrix.

        Computes task similarity, clustering, correlations, etc.

        Returns:
            Dict with statistical analysis results.

        Raises:
            RuntimeError: If matrix not built yet.
            ImportError: If analysis dependencies missing.
        """
        if self.matrix is None:
            raise RuntimeError("Matrix not yet built. Call build() first.")

        try:
            from .transfer_analysis import TransferMatrixAnalyzer

        except ImportError:
            raise ImportError(
                "Statistical analysis requires scipy and scikit-learn. "
                "Install with: pip install scipy scikit-learn"
            )

        analyzer = TransferMatrixAnalyzer(self.task_names)
        return analyzer.comprehensive_analysis(self.matrix)
