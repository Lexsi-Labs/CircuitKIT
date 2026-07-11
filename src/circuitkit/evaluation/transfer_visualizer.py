"""
Visualization tools for cross-task transfer matrices.

Provides heatmap visualization, comparison plots, and
interactive visualizations for transfer matrix analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import get_logger


def _sns():
    """Lazy-import seaborn so a fresh install of circuitkit doesn't fail
    when this module is imported transitively (e.g. via evaluate.full).
    seaborn is only needed for the heatmap rendering inside specific
    methods of TransferMatrixVisualizer."""
    import seaborn as _seaborn  # noqa: WPS433

    return _seaborn


class TransferMatrixVisualizer:
    """
    Visualize cross-task transfer matrices as heatmaps and analysis plots.

    Provides:
    - Heatmap visualization of transfer scores
    - Per-task averages plot
    - Best/worst transfer highlights
    - Comparison visualizations
    """

    def __init__(self, task_names: List[str], figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualizer for a transfer matrix.

        Args:
            task_names (List[str]): Names of tasks in the matrix.
            figsize (Tuple[int, int]): Figure size for plots. Defaults to (10, 8).
        """
        self.task_names = task_names
        self.figsize = figsize
        self.logger = get_logger("circuitkit.transfer_visualizer")

    def heatmap(
        self,
        matrix: np.ndarray,
        output_path: Optional[str] = None,
        title: str = "Cross-Task Transfer Matrix",
        cmap: str = "RdYlGn",
        vmin: float = 0.0,
        vmax: float = 1.0,
        annot: bool = True,
        fmt: str = ".3f",
        cbar_label: str = "Transfer Score",
    ) -> Optional[Any]:
        """
        Create a heatmap visualization of the transfer matrix.

        Args:
            matrix (np.ndarray): NxN transfer matrix.
            output_path (Optional[str]): Path to save figure. If None, displays in notebook.
            title (str): Title for the heatmap. Defaults to "Cross-Task Transfer Matrix".
            cmap (str): Colormap name. Defaults to "RdYlGn".
            vmin (float): Minimum value for color scaling. Defaults to 0.0.
            vmax (float): Maximum value for color scaling. Defaults to 1.0.
            annot (bool): Whether to annotate cells with values. Defaults to True.
            fmt (str): Format string for annotations. Defaults to ".3f".
            cbar_label (str): Label for colorbar. Defaults to "Transfer Score".

        Returns:
            Optional[Any]: Matplotlib figure object if output_path is provided, else None.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create heatmap (lazy-imported seaborn).
        sns = _sns()
        sns.heatmap(
            matrix,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": cbar_label},
            xticklabels=self.task_names,
            yticklabels=self.task_names,
            ax=ax,
            square=True,
            linewidths=0.5,
            linecolor="gray",
        )

        # Labels
        ax.set_xlabel("Target Task", fontsize=12, fontweight="bold")
        ax.set_ylabel("Source Task", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Heatmap saved to {output_path}")
            plt.close(fig)
            return fig
        else:
            return fig

    def per_task_averages(
        self,
        matrix: np.ndarray,
        analysis: Dict[str, Any],
        output_path: Optional[str] = None,
        title: str = "Per-Task Transfer Statistics",
    ) -> Optional[Any]:
        """
        Create side-by-side bar plots of per-task averages.

        Args:
            matrix (np.ndarray): NxN transfer matrix.
            analysis (Dict[str, Any]): Analysis dict from TransferMatrix.analyze().
            output_path (Optional[str]): Path to save figure.
            title (str): Title for the plot.

        Returns:
            Optional[Any]: Matplotlib figure object.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Source averages
        src_avgs = analysis["source_avg"]
        src_tasks = list(src_avgs.keys())
        src_scores = [src_avgs[t] for t in src_tasks]

        axes[0].bar(src_tasks, src_scores, color="steelblue", alpha=0.7, edgecolor="black")
        axes[0].set_ylabel("Average Transfer Score", fontsize=11, fontweight="bold")
        axes[0].set_title("Per-Source Task Generalization", fontsize=12, fontweight="bold")
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis="y", alpha=0.3)
        for i, v in enumerate(src_scores):
            if not np.isnan(v):
                axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

        # Target averages
        tgt_avgs = analysis["target_avg"]
        tgt_tasks = list(tgt_avgs.keys())
        tgt_scores = [tgt_avgs[t] for t in tgt_tasks]

        axes[1].bar(tgt_tasks, tgt_scores, color="darkorange", alpha=0.7, edgecolor="black")
        axes[1].set_ylabel("Average Transfer Score", fontsize=11, fontweight="bold")
        axes[1].set_title("Per-Target Task Difficulty", fontsize=12, fontweight="bold")
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis="y", alpha=0.3)
        for i, v in enumerate(tgt_scores):
            if not np.isnan(v):
                axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Per-task averages plot saved to {output_path}")
            plt.close(fig)
            return fig
        else:
            return fig

    def distribution_plot(
        self,
        matrix: np.ndarray,
        analysis: Dict[str, Any],
        output_path: Optional[str] = None,
        title: str = "Transfer Score Distribution",
    ) -> Optional[Any]:
        """
        Create visualization of transfer score distribution.

        Args:
            matrix (np.ndarray): NxN transfer matrix.
            analysis (Dict[str, Any]): Analysis dict from TransferMatrix.analyze().
            output_path (Optional[str]): Path to save figure.
            title (str): Title for the plot.

        Returns:
            Optional[Any]: Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Flatten matrix and remove NaN values
        scores = matrix[~np.isnan(matrix)].flatten()

        ax.hist(scores, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
        ax.axvline(
            analysis["overall_mean"],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {analysis['overall_mean']:.3f}",
        )
        ax.axvline(
            np.median(scores),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(scores):.3f}",
        )
        ax.set_xlabel("Transfer Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Distribution plot saved to {output_path}")
            plt.close(fig)
            return fig
        else:
            return fig

    def comparison_plot(
        self,
        matrices: Dict[str, np.ndarray],
        output_path: Optional[str] = None,
        title: str = "Transfer Matrix Comparison",
    ) -> Optional[Any]:
        """
        Create side-by-side comparison of multiple transfer matrices.

        Args:
            matrices (Dict[str, np.ndarray]): Dictionary of {name: matrix}.
            output_path (Optional[str]): Path to save figure.
            title (str): Title for the plot.

        Returns:
            Optional[Any]: Matplotlib figure object.
        """
        n_matrices = len(matrices)
        fig, axes = plt.subplots(1, n_matrices, figsize=(8 * n_matrices, 8))

        if n_matrices == 1:
            axes = [axes]

        sns = _sns()
        for idx, (name, matrix) in enumerate(matrices.items()):
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
                xticklabels=self.task_names,
                yticklabels=self.task_names,
                ax=axes[idx],
                square=True,
                cbar_kws={"label": "Transfer Score"},
            )
            axes[idx].set_title(name, fontsize=12, fontweight="bold")

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Comparison plot saved to {output_path}")
            plt.close(fig)
            return fig
        else:
            return fig

    def save_all_visualizations(
        self,
        matrix: np.ndarray,
        analysis: Dict[str, Any],
        output_dir: str,
    ) -> List[str]:
        """
        Generate and save all standard visualizations.

        Args:
            matrix (np.ndarray): NxN transfer matrix.
            analysis (Dict[str, Any]): Analysis dict from TransferMatrix.analyze().
            output_dir (str): Directory to save all visualizations.

        Returns:
            List[str]: Paths to all saved files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        # Heatmap
        heatmap_path = output_dir / "transfer_matrix_heatmap.png"
        self.heatmap(matrix, str(heatmap_path))
        saved_files.append(str(heatmap_path))
        self.logger.info(f"Heatmap saved to {heatmap_path}")

        # Per-task averages
        avg_path = output_dir / "per_task_averages.png"
        self.per_task_averages(matrix, analysis, str(avg_path))
        saved_files.append(str(avg_path))
        self.logger.info(f"Per-task averages plot saved to {avg_path}")

        # Distribution
        dist_path = output_dir / "score_distribution.png"
        self.distribution_plot(matrix, analysis, str(dist_path))
        saved_files.append(str(dist_path))
        self.logger.info(f"Distribution plot saved to {dist_path}")

        return saved_files
