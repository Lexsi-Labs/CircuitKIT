"""
Statistical analysis for cross-task transfer matrices.

Provides correlation analysis, clustering, and other
statistical insights into transfer patterns.
"""

from typing import Any, Dict, List

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from ..utils.logging import get_logger


class TransferMatrixAnalyzer:
    """
    Statistical analysis of cross-task transfer matrices.

    Computes:
    - Task similarity metrics
    - Transfer pattern clustering
    - Correlation analysis
    - Task difficulty/transferability rankings
    """

    def __init__(self, task_names: List[str]):
        """
        Initialize analyzer for a transfer matrix.

        Args:
            task_names (List[str]): Names of tasks in the matrix.
        """
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.logger = get_logger("circuitkit.transfer_analysis")

    def task_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute task-task similarity from transfer matrix.

        Uses Pearson correlation of transfer profiles:
        - Rows represent source transfer profile
        - Columns represent target transfer profile
        Similarity is the correlation between these profiles.

        Args:
            matrix (np.ndarray): NxN transfer matrix.

        Returns:
            np.ndarray: NxN similarity matrix (values in [-1, 1]).
        """
        n = len(self.task_names)
        similarity = np.zeros((n, n))

        # Remove NaN for correlation computation
        valid_mask = ~np.isnan(matrix)

        for i in range(n):
            for j in range(n):
                # Get valid indices for both rows
                valid_i = valid_mask[i]
                valid_j = valid_mask[j]

                # Compute correlation only on commonly valid entries
                common_valid = valid_i & valid_j
                if np.sum(common_valid) > 1:  # Need at least 2 valid entries
                    corr = np.corrcoef(
                        matrix[i, common_valid],
                        matrix[j, common_valid],
                    )[0, 1]
                    # Handle NaN correlation (can occur if std is 0)
                    similarity[i, j] = corr if not np.isnan(corr) else 0.0
                else:
                    # Not enough valid entries for correlation
                    similarity[i, j] = 0.0

        return similarity

    def transfer_clustering(
        self,
        matrix: np.ndarray,
        method: str = "ward",
    ) -> Dict[str, Any]:
        """
        Perform hierarchical clustering on transfer patterns.

        Groups tasks based on similar transfer patterns.

        Args:
            matrix (np.ndarray): NxN transfer matrix.
            method (str): Linkage method for clustering.
                One of: 'ward', 'complete', 'average', 'single'.
                Defaults to 'ward'.

        Returns:
            Dict with keys:
                'linkage_matrix': scipy linkage matrix
                'clusters': Dict[str, List[str]] - cluster assignments
                'dendrogram_data': Dict for matplotlib dendrogram plotting
        """
        # Compute distances from rows (source transfer profiles)
        ~np.isnan(matrix)

        # Replace NaN with 0 for distance computation
        matrix_clean = np.nan_to_num(matrix, nan=0.0)

        # Compute pairwise distances
        distances = pdist(matrix_clean, metric="euclidean")
        linkage_matrix = linkage(distances, method=method)

        # Perform clustering (using ward method as default)
        from scipy.cluster.hierarchy import fcluster

        # Cut the dendrogram at a reasonable height
        # Use a distance threshold based on the linkage matrix
        max_d = np.max(linkage_matrix[:, 2])
        threshold = 0.7 * max_d

        cluster_ids = fcluster(linkage_matrix, threshold, criterion="distance")

        # Build cluster assignments
        clusters = {}
        for cluster_id in np.unique(cluster_ids):
            cluster_tasks = [
                self.task_names[i] for i in range(len(cluster_ids)) if cluster_ids[i] == cluster_id
            ]
            clusters[f"Cluster {cluster_id}"] = cluster_tasks

        self.logger.info(f"Clustering identified {len(clusters)} clusters")

        return {
            "linkage_matrix": linkage_matrix,
            "clusters": clusters,
            "cluster_ids": cluster_ids,
            "threshold": threshold,
        }

    def transferability_score(
        self,
        matrix: np.ndarray,
        use_source: bool = True,
    ) -> Dict[str, float]:
        """
        Rank tasks by how well they transfer to other tasks.

        Scores task generalizability or learnability from other tasks.

        Args:
            matrix (np.ndarray): NxN transfer matrix.
            use_source (bool): If True, rank by source (generalization);
                if False, rank by target (learnability). Defaults to True.

        Returns:
            Dict[str, float]: {task_name: transferability_score}
        """
        scores = {}

        if use_source:
            # Source transferability: how well does this circuit generalize?
            for idx, task in enumerate(self.task_names):
                valid_scores = matrix[idx, ~np.isnan(matrix[idx])]
                if len(valid_scores) > 0:
                    # Mean + variance (high mean + high variance = unpredictable but potentially good)
                    score = float(np.mean(valid_scores) * (1 + 0.5 * np.std(valid_scores)))
                    scores[task] = score
                else:
                    scores[task] = 0.0
        else:
            # Target learnability: which tasks are easy to transfer to?
            for idx, task in enumerate(self.task_names):
                valid_scores = matrix[~np.isnan(matrix[:, idx]), idx]
                if len(valid_scores) > 0:
                    score = float(np.mean(valid_scores))
                    scores[task] = score
                else:
                    scores[task] = 0.0

        return scores

    def correlation_structure(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze correlation structure of the transfer matrix.

        Identifies systematic patterns in transfer performance.

        Args:
            matrix (np.ndarray): NxN transfer matrix.

        Returns:
            Dict with keys:
                'diagonal_strength': float - how much do diagonal elements dominate?
                'off_diagonal_mean': float - average off-diagonal transfer
                'diagonal_mean': float - average diagonal transfer (self-transfer)
                'symmetry': float - how symmetric is the matrix? (0=asymmetric, 1=symmetric)
                'sparsity': float - fraction of NaN values
        """
        valid_mask = ~np.isnan(matrix)

        # Diagonal elements (self-transfer)
        diagonal = np.diag(matrix)
        diag_valid = diagonal[~np.isnan(diagonal)]
        diag_mean = float(np.mean(diag_valid)) if len(diag_valid) > 0 else 0.0

        # Off-diagonal elements
        off_diag = matrix.copy()
        np.fill_diagonal(off_diag, np.nan)
        off_diag_valid = off_diag[~np.isnan(off_diag)]
        off_diag_mean = float(np.mean(off_diag_valid)) if len(off_diag_valid) > 0 else 0.0

        # Diagonal strength
        diagonal_strength = diag_mean - off_diag_mean

        # Symmetry
        symmetric_error = 0
        symmetric_count = 0
        for i in range(len(self.task_names)):
            for j in range(i + 1, len(self.task_names)):
                if valid_mask[i, j] and valid_mask[j, i]:
                    symmetric_error += np.abs(matrix[i, j] - matrix[j, i])
                    symmetric_count += 1

        if symmetric_count > 0:
            avg_error = symmetric_error / symmetric_count
            symmetry = float(1.0 - (avg_error / (diag_mean + 1e-6)))  # Normalize by typical scale
            symmetry = float(np.clip(symmetry, 0, 1))
        else:
            symmetry = 0.0

        # Sparsity
        total_elements = self.n_tasks * self.n_tasks
        nan_count = np.sum(~valid_mask)
        sparsity = float(nan_count / total_elements)

        return {
            "diagonal_strength": float(diagonal_strength),
            "diagonal_mean": diag_mean,
            "off_diagonal_mean": off_diag_mean,
            "symmetry": symmetry,
            "sparsity": sparsity,
        }

    def effect_sizes(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compute effect sizes and practical significance metrics.

        Args:
            matrix (np.ndarray): NxN transfer matrix.

        Returns:
            Dict with keys:
                'pairs_above_threshold': Dict[float, int] - count of pairs >= threshold
                'practical_significance': float - % of transfers >= 0.5
                'effect_size_range': Tuple[float, float] - (min, max)
                'largest_improvements': List - best transfers
                'largest_regressions': List - worst transfers
        """
        valid = matrix[~np.isnan(matrix)].flatten()

        pairs_above_threshold = {}
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            count = np.sum(valid >= threshold)
            pairs_above_threshold[threshold] = int(count)

        practical_significance = float(np.sum(valid >= 0.5) / len(valid) if len(valid) > 0 else 0)

        effect_range = (float(np.min(valid)), float(np.max(valid)))

        # Find best and worst transfers for each source
        largest_improvements = []
        largest_regressions = []

        for src_idx, src_task in enumerate(self.task_names):
            # Best transfer from this source
            valid_targets = ~np.isnan(matrix[src_idx])
            if np.any(valid_targets):
                best_idx = np.argmax(matrix[src_idx])
                best_val = matrix[src_idx, best_idx]
                largest_improvements.append((src_task, self.task_names[best_idx], float(best_val)))

            # Worst transfer from this source
            if np.any(valid_targets):
                worst_idx = np.nanargmin(matrix[src_idx])
                worst_val = matrix[src_idx, worst_idx]
                largest_regressions.append((src_task, self.task_names[worst_idx], float(worst_val)))

        largest_improvements.sort(key=lambda x: x[2], reverse=True)
        largest_regressions.sort(key=lambda x: x[2])

        return {
            "pairs_above_threshold": pairs_above_threshold,
            "practical_significance": practical_significance,
            "effect_size_range": effect_range,
            "largest_improvements": largest_improvements[:5],  # Top 5
            "largest_regressions": largest_regressions[:5],  # Bottom 5
        }

    def comprehensive_analysis(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Run comprehensive statistical analysis on transfer matrix.

        Args:
            matrix (np.ndarray): NxN transfer matrix.

        Returns:
            Dict with all analysis results.
        """
        return {
            "task_similarity": self.task_similarity(matrix),
            "clustering": self.transfer_clustering(matrix),
            "transferability": self.transferability_score(matrix),
            "correlation_structure": self.correlation_structure(matrix),
            "effect_sizes": self.effect_sizes(matrix),
        }
