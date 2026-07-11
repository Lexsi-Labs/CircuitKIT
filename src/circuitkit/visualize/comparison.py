"""
Comparison dashboards for visualizing circuits across different conditions.

Supports side-by-side comparison of circuits across:
- Multiple seeds (stability visualization)
- Different corruptions (robustness visualization)
- Cross-task transfer (generalization visualization)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .theme import PALETTE, get_plotly_layout
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ComparisonDashboard:
    """
    Comparison dashboard for visualizing circuits across multiple dimensions.

    Supports:
    - Multi-seed stability comparison
    - Corruption robustness comparison
    - Cross-task transfer comparison
    - Interactive Plotly with tabs
    - Export to HTML
    """

    def __init__(
        self,
        circuits: Dict[str, Dict[str, float]],
        comparison_type: str = "stability",
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize comparison dashboard.

        Args:
            circuits: Dictionary mapping condition names to node score dicts.
                Format: {condition_name: {node_name: score}}
            comparison_type: Type of comparison ('stability', 'robustness', 'generalization').
            labels: Optional list of labels for circuits (default: circuit names).
            metadata: Optional metadata about the comparison.
        """
        self.circuits = circuits
        self.comparison_type = comparison_type
        self.labels = labels or list(circuits.keys())
        self.metadata = metadata or {}

        # Validate inputs
        if len(self.circuits) < 2:
            raise ValueError("At least 2 circuits required for comparison")

        # Extract common nodes
        self.common_nodes = self._get_common_nodes()
        if not self.common_nodes:
            raise ValueError("No common nodes found across circuits")

        # Normalize all circuits
        self.normalized_circuits = self._normalize_circuits()

    def _get_common_nodes(self) -> List[str]:
        """Get nodes that appear in all circuits."""
        all_node_sets = [set(circuit.keys()) for circuit in self.circuits.values()]
        common = set.intersection(*all_node_sets) if all_node_sets else set()
        return sorted(list(common))

    def _normalize_circuits(self) -> Dict[str, Dict[str, float]]:
        """Normalize all circuits to [0, 1] range."""
        normalized = {}

        for condition_name, circuit in self.circuits.items():
            scores = np.array([circuit[node] for node in self.common_nodes], dtype=np.float32)

            # Normalize to [0, 1]
            score_min, score_max = scores.min(), scores.max()
            if score_max > score_min:
                normalized_scores = (scores - score_min) / (score_max - score_min)
            else:
                normalized_scores = np.zeros_like(scores)

            normalized[condition_name] = {
                node: float(score) for node, score in zip(self.common_nodes, normalized_scores)
            }

        return normalized

    def plot_stability_heatmap(
        self,
        top_k: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        colorscale: str = "Viridis",
    ) -> go.Figure:
        """
        Create heatmap showing how scores vary across different seeds/runs.

        Args:
            top_k: Show only top-k nodes (by average score). If None, shows all.
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name.

        Returns:
            Plotly Figure object.
        """
        # Stack normalized scores
        circuit_names = list(self.circuits.keys())
        scores_matrix = np.array(
            [
                [self.normalized_circuits[circuit][node] for node in self.common_nodes]
                for circuit in circuit_names
            ]
        )  # (num_circuits, num_nodes)

        # Sort by mean score if top_k specified
        if top_k:
            mean_scores = scores_matrix.mean(axis=0)
            top_indices = np.argsort(mean_scores)[-top_k:][::-1]
            scores_matrix = scores_matrix[:, top_indices]
            nodes_to_show = [self.common_nodes[i] for i in top_indices]
        else:
            nodes_to_show = self.common_nodes

        fig = go.Figure(
            data=go.Heatmap(
                z=scores_matrix,
                x=nodes_to_show,
                y=circuit_names,
                colorscale=colorscale,
                colorbar=dict(title="Normalized<br>Score"),
                hovertemplate="Condition: %{y}<br>Node: %{x}<br>Score: %{z:.3f}<extra></extra>",
            )
        )

        title_suffix = f" (Top {top_k})" if top_k else " (All Nodes)"
        fig.update_layout(**get_plotly_layout(
        title=f"Stability Analysis{title_suffix}",
        width=int(figsize[0] * 100),
        height=int(figsize[1] * 100),
        xaxis=dict(showgrid=False, zeroline=False, tickangle=-45,
                title="Node", color=PALETTE.text_secondary),
        yaxis=dict(showgrid=False, zeroline=False, title="Seed/Run",
                color=PALETTE.text_secondary),
    ))


        return fig

    def plot_correlation_matrix(
        self,
        figsize: Tuple[int, int] = (10, 10),
        colorscale: str = "RdBu",
    ) -> go.Figure:
        """
        Create correlation matrix showing how similar circuits are across conditions.

        Args:
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name.

        Returns:
            Plotly Figure object.
        """
        circuit_names = list(self.circuits.keys())
        num_circuits = len(circuit_names)

        # Compute pairwise correlations
        correlation_matrix = np.zeros((num_circuits, num_circuits))

        for i, circuit1_name in enumerate(circuit_names):
            for j, circuit2_name in enumerate(circuit_names):
                scores1 = np.array(
                    [self.normalized_circuits[circuit1_name][node] for node in self.common_nodes]
                )
                scores2 = np.array(
                    [self.normalized_circuits[circuit2_name][node] for node in self.common_nodes]
                )

                # Pearson correlation
                correlation = np.corrcoef(scores1, scores2)[0, 1]
                correlation_matrix[i, j] = correlation if not np.isnan(correlation) else 1.0

        fig = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=circuit_names,
                y=circuit_names,
                colorscale=colorscale,
                zmid=0,
                colorbar=dict(title="Correlation"),
                hovertemplate="Pair: %{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
                text=np.round(correlation_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(**get_plotly_layout(
            title="Circuit Similarity Across Conditions",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            xaxis=dict(showgrid=False, zeroline=False, title="Circuit",
                    color=PALETTE.text_secondary),
            yaxis=dict(showgrid=False, zeroline=False, title="Circuit",
                    color=PALETTE.text_secondary),
        ))

        return fig

    def plot_robustness_comparison(
        self,
        top_k: Optional[int] = 10,
        figsize: Tuple[int, int] = (12, 6),
    ) -> go.Figure:
        """
        Create bar chart comparing node importance across different corruption types.

        Args:
            top_k: Show only top-k nodes by average score across corruptions.
            figsize: Figure size (width, height) in inches.

        Returns:
            Plotly Figure object.
        """
        circuit_names = list(self.circuits.keys())

        # Get average scores for each node
        avg_scores = {}
        for node in self.common_nodes:
            scores = [self.normalized_circuits[circuit][node] for circuit in circuit_names]
            avg_scores[node] = np.mean(scores)

        # Sort and select top-k
        sorted_nodes = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        if top_k:
            sorted_nodes = sorted_nodes[:top_k]

        nodes_to_show = [node for node, _ in sorted_nodes]

        # Create grouped bar chart
        fig = go.Figure()

        for circuit_name in circuit_names:
            scores = [self.normalized_circuits[circuit_name][node] for node in nodes_to_show]
            fig.add_trace(
                go.Bar(
                    name=circuit_name,
                    x=nodes_to_show,
                    y=scores,
                    hovertemplate="Condition: "
                    + circuit_name
                    + "<br>Node: %{x}<br>Score: %{y:.3f}<extra></extra>",
                )
            )

        fig.update_layout(**get_plotly_layout(
            title="Robustness: Node Importance Across Conditions",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            barmode="group",
            xaxis=dict(showgrid=False, zeroline=False, tickangle=-45,
                    title="Node", color=PALETTE.text_secondary),
            yaxis=dict(showgrid=False, zeroline=False, title="Normalized Score",
                    color=PALETTE.text_secondary),
        ))

        return fig

    def plot_transfer_matrix(
        self,
        figsize: Tuple[int, int] = (10, 8),
        colorscale: str = "RdYlGn",
    ) -> go.Figure:
        """
        Create transfer matrix showing generalization across tasks.

        Args:
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name.

        Returns:
            Plotly Figure object.
        """
        circuit_names = list(self.circuits.keys())
        num_circuits = len(circuit_names)

        # Compute pairwise similarity using Jaccard index for overlap
        transfer_matrix = np.zeros((num_circuits, num_circuits))

        for i, circuit1_name in enumerate(circuit_names):
            for j, circuit2_name in enumerate(circuit_names):
                scores1 = self.normalized_circuits[circuit1_name]
                scores2 = self.normalized_circuits[circuit2_name]

                # Use threshold-based overlap as transfer measure
                threshold = 0.5
                nodes1 = {node for node, score in scores1.items() if score >= threshold}
                nodes2 = {node for node, score in scores2.items() if score >= threshold}

                if nodes1 or nodes2:
                    jaccard = len(nodes1 & nodes2) / len(nodes1 | nodes2)
                else:
                    jaccard = 1.0 if i == j else 0.0

                transfer_matrix[i, j] = jaccard

        fig = go.Figure(
            data=go.Heatmap(
                z=transfer_matrix,
                x=circuit_names,
                y=circuit_names,
                colorscale=colorscale,
                colorbar=dict(title="Transfer<br>Score"),
                hovertemplate="From: %{y} to %{x}<br>Transfer: %{z:.3f}<extra></extra>",
                text=np.round(transfer_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(**get_plotly_layout(
            title="Cross-Task Transfer Matrix",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            xaxis=dict(showgrid=False, zeroline=False, title="Target Task",
                    color=PALETTE.text_secondary),
            yaxis=dict(showgrid=False, zeroline=False, title="Source Task",
                    color=PALETTE.text_secondary),
        ))

        return fig

    def plot_distribution_comparison(
        self,
        figsize: Tuple[int, int] = (12, 6),
    ) -> go.Figure:
        """
        Create violin/box plot comparing score distributions across circuits.

        Args:
            figsize: Figure size (width, height) in inches.

        Returns:
            Plotly Figure object.
        """
        circuit_names = list(self.circuits.keys())

        fig = go.Figure()

        for circuit_name in circuit_names:
            scores = list(self.normalized_circuits[circuit_name].values())
            fig.add_trace(
                go.Box(
                    y=scores,
                    name=circuit_name,
                    boxmean="sd",
                    hovertemplate="Score: %{y:.3f}<extra></extra>",
                )
            )

        fig.update_layout(**get_plotly_layout(
            title="Score Distribution Across Conditions",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            showlegend=True,
            yaxis=dict(showgrid=False, zeroline=False, title="Normalized Score",
                    color=PALETTE.text_secondary),
        ))

        return fig

    def export_to_html(
        self,
        output_path: str,
        plot_type: str = "all",
        **kwargs,
    ) -> None:
        """
        Export visualizations to HTML file(s).

        Args:
            output_path: Path to save HTML file(s).
            plot_type: Type of plot ('stability', 'correlation', 'robustness',
                'transfer', 'distribution', 'all').
            **kwargs: Additional arguments passed to plotting methods.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plots = {}

        if plot_type in ("stability", "all"):
            plots["Stability"] = self.plot_stability_heatmap(**kwargs)

        if plot_type in ("correlation", "all"):
            plots["Correlation"] = self.plot_correlation_matrix(**kwargs)

        if plot_type in ("robustness", "all"):
            plots["Robustness"] = self.plot_robustness_comparison(**kwargs)

        if plot_type in ("transfer", "all"):
            plots["Transfer"] = self.plot_transfer_matrix(**kwargs)

        if plot_type in ("distribution", "all"):
            plots["Distribution"] = self.plot_distribution_comparison(**kwargs)

        if not plots:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        # Create tabbed interface if multiple plots
        if len(plots) > 1:
            make_subplots(rows=1, cols=1)

            # Create figure with tabs
            buttons = []
            for plot_name, plot_fig in plots.items():
                visible = [plot_name == list(plots.keys())[0]] * len(plot_fig.data)
                buttons.append(
                    dict(
                        label=plot_name,
                        method="update",
                        args=[{"visible": visible}, {"title": plot_fig.layout.title}],
                    )
                )

            # Combine all traces
            combined_fig = go.Figure()
            for plot_name, plot_fig in plots.items():
                for trace in plot_fig.data:
                    visible = plot_name == list(plots.keys())[0]
                    trace.visible = visible
                    combined_fig.add_trace(trace)

            # Add buttons for tabs
            combined_fig.update_layout(
                updatemenus=[
                    dict(buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True)
                ]
            )

            combined_fig.write_html(str(output_path))
        else:
            # Single plot
            list(plots.values())[0].write_html(str(output_path))

    def export_to_json(self, output_path: str) -> None:
        """
        Export comparison data to JSON.

        Args:
            output_path: Path to save JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "comparison_type": self.comparison_type,
            "circuits": {
                name: {
                    "original": self.circuits[name],
                    "normalized": self.normalized_circuits[name],
                }
                for name in self.circuits.keys()
            },
            "common_nodes": self.common_nodes,
            "metadata": self.metadata,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all circuits.

        Returns:
            Dictionary with statistics for each circuit.
        """
        summary = {
            "comparison_type": self.comparison_type,
            "num_circuits": len(self.circuits),
            "num_common_nodes": len(self.common_nodes),
            "circuits": {},
        }

        for circuit_name in self.circuits.keys():
            scores = np.array(list(self.normalized_circuits[circuit_name].values()))
            summary["circuits"][circuit_name] = {
                "mean": float(scores.mean()),
                "median": float(np.median(scores)),
                "std": float(scores.std()),
                "max": float(scores.max()),
                "min": float(scores.min()),
            }

        return summary
