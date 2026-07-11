"""
Feature saliency maps for circuit visualization.

Visualizes which circuit nodes most contribute to the model output using
attribution-based methods (gradient or patching). Supports comparative views
of original vs corrupted activations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .theme import PALETTE, get_plotly_layout
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FeatureSaliencyVisualizer:
    """
    Visualizes feature saliency maps showing which circuit nodes contribute
    most to the model output.

    Supports:
    - Gradient-based attribution
    - Patching-based attribution
    - Comparative visualization (original vs corrupted)
    - Node-level importance ranking
    - Export to HTML with interactive Plotly
    """

    def __init__(
        self,
        node_attributions: Dict[str, float],
        node_names: Optional[List[str]] = None,
        original_activations: Optional[Dict[str, np.ndarray]] = None,
        corrupted_activations: Optional[Dict[str, np.ndarray]] = None,
        method: str = "gradient",
    ):
        """
        Initialize the feature saliency visualizer.

        Args:
            node_attributions: Dictionary mapping node names to attribution scores.
                Represents how much each node contributes to output.
            node_names: Optional list of node names. If not provided,
                uses keys from node_attributions.
            original_activations: Optional dict of original layer activations
                for comparison visualization.
            corrupted_activations: Optional dict of corrupted layer activations
                for comparison visualization.
            method: Attribution method used ('gradient', 'patching', etc.).
        """
        # Zip the custom names with the existing attribution values if custom names are provided
        if node_names is not None:
            if len(node_names) != len(node_attributions):
                raise ValueError("Length of node_names must match number of node_attributions")
            self.node_attributions = dict(zip(node_names, node_attributions.values()))
            self.node_names = node_names
        else:
            self.node_attributions = node_attributions
            self.node_names = list(node_attributions.keys())

        self.original_activations = original_activations or {}
        self.corrupted_activations = corrupted_activations or {}
        self.method = method

        # Normalize attributions
        self.normalized_attributions = self._normalize_attributions()

        # Validate comparison activations
        if self.original_activations or self.corrupted_activations:
            self._validate_comparison_data()

    def _normalize_attributions(self) -> Dict[str, float]:
        """Normalize attribution scores to [0, 1] range."""
        attributions = [abs(self.node_attributions[node]) for node in self.node_names]

        attr_array = np.array(attributions, dtype=np.float32)
        attr_min, attr_max = attr_array.min(), attr_array.max()

        if attr_max > attr_min:
            normalized = (attr_array - attr_min) / (attr_max - attr_min)
        else:
            normalized = np.zeros_like(attr_array)

        return {node: float(norm) for node, norm in zip(self.node_names, normalized)}

    def _validate_comparison_data(self) -> None:
        """Validate that comparison activations have compatible structure."""
        if self.original_activations and self.corrupted_activations:
            if set(self.original_activations.keys()) != set(self.corrupted_activations.keys()):
                raise ValueError(
                    "original_activations and corrupted_activations must have "
                    "the same keys (layers)"
                )

    def plot_importance_bar(
        self,
        top_k: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6),
        colorscale: str = "Blues",
    ) -> go.Figure:
        """
        Create bar chart of node importance scores.

        Args:
            top_k: Show only top-k nodes. If None, shows all.
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name.

        Returns:
            Plotly Figure object.
        """
        # Sort by attribution
        sorted_items = sorted(
            self.normalized_attributions.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k:
            sorted_items = sorted_items[:top_k]

        nodes, scores = zip(*sorted_items)

        # Create color array with gradient
        [f"rgba(31, 119, 180, {score})" for score in scores]

        fig = go.Figure(
            data=go.Bar(
                y=list(nodes),
                x=list(scores),
                orientation="h",
                marker=dict(color=list(scores), colorscale=colorscale, showscale=True),
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
                hovertemplate="Node: %{y}<br>Attribution: %{x:.3f}<extra></extra>",
            )
        )

        title = f"Feature Attribution ({'Top ' + str(top_k) if top_k else 'All Nodes'})"
        fig.update_layout(**get_plotly_layout(
            title=title,
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            margin={"b": 40, "l": 150, "r": 20, "t": 60},
            xaxis=dict(showgrid=False, zeroline=False,
                    title="Normalized Attribution Score",
                    color=PALETTE.text_secondary),
            yaxis=dict(showgrid=False, zeroline=False,
                    title="Node", color=PALETTE.text_secondary),
        ))


        return fig

    def plot_network_saliency(
        self,
        figsize: Tuple[int, int] = (12, 10),
        colorscale: str = "Reds",
    ) -> go.Figure:
        """
        Create network-style visualization showing node importance as node size/color.

        Args:
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name.

        Returns:
            Plotly Figure object.
        """
        # Create simple positions for nodes (circular layout)
        num_nodes = len(self.node_names)
        angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        attributions = [self.normalized_attributions[node] for node in self.node_names]

        # Node sizes based on attribution
        node_sizes = [5 + 50 * attr for attr in attributions]

        fig = go.Figure(
            data=go.Scatter(
                x=x_pos,
                y=y_pos,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=attributions,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Attribution"),
                    line=dict(width=2, color="white"),
                ),
                text=self.node_names,
                textposition="middle center",
                hovertemplate="Node: %{text}<br>Attribution: %{marker.color:.3f}<extra></extra>",
            )
        )

        fig.update_layout(**get_plotly_layout(
            title="Feature Saliency Network",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            margin={"b": 20, "l": 5, "r": 5, "t": 40},
        ))

        return fig

    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (14, 6),
        colorscale: str = "RdBu",
    ) -> go.Figure:
        """
        Create side-by-side comparison of original vs corrupted activations.

        Args:
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name for difference visualization.

        Returns:
            Plotly Figure object.

        Raises:
            ValueError: If comparison activations not provided.
        """
        if not self.original_activations or not self.corrupted_activations:
            raise ValueError(
                "original_activations and corrupted_activations required for comparison"
            )

        # Get common layers
        common_layers = set(self.original_activations.keys()) & set(
            self.corrupted_activations.keys()
        )
        if not common_layers:
            raise ValueError("No common layers between original and corrupted activations")

        layers = sorted(list(common_layers))

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("Original Activations", "Corrupted Activations", "Difference"),
            specs=[[{"type": "heatmap"} for _ in range(3)]],
        )

        # Process each layer pair
        original_stack = []
        corrupted_stack = []

        for layer in layers:
            orig = np.asarray(self.original_activations[layer], dtype=np.float32)
            corr = np.asarray(self.corrupted_activations[layer], dtype=np.float32)

            # Flatten if multi-dimensional
            if orig.ndim > 1:
                orig = np.max(np.abs(orig), axis=tuple(range(1, orig.ndim)))
            if corr.ndim > 1:
                corr = np.max(np.abs(corr), axis=tuple(range(1, corr.ndim)))

            original_stack.append(orig)
            corrupted_stack.append(corr)

        # Stack and normalize
        original_data = np.stack(original_stack, axis=0)  # (num_layers, seq_len)
        corrupted_data = np.stack(corrupted_stack, axis=0)

        # Normalize
        orig_min, orig_max = original_data.min(), original_data.max()
        if orig_max > orig_min:
            original_data = (original_data - orig_min) / (orig_max - orig_min)

        corr_min, corr_max = corrupted_data.min(), corrupted_data.max()
        if corr_max > corr_min:
            corrupted_data = (corrupted_data - corr_min) / (corr_max - corr_min)

        # Compute difference
        difference_data = np.abs(original_data - corrupted_data)

        # Add heatmaps
        fig.add_trace(
            go.Heatmap(
                z=original_data,
                colorscale="Viridis",
                colorbar=dict(title="Activation", x=0.32),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                z=corrupted_data,
                colorscale="Viridis",
                colorbar=dict(title="Activation", x=0.65),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Heatmap(
                z=difference_data,
                colorscale="Reds",
                colorbar=dict(title="Difference", x=0.98),
            ),
            row=1,
            col=3,
        )

        fig.update_layout(**get_plotly_layout(
            title="Activation Saliency Comparison",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
        ))

        return fig

    def export_to_html(
        self,
        output_path: str,
        plot_type: str = "importance_bar",
        **kwargs,
    ) -> None:
        """
        Export visualization to HTML file.

        Args:
            output_path: Path to save HTML file.
            plot_type: Type of plot ('importance_bar', 'network', 'comparison').
            **kwargs: Additional arguments passed to plotting method.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if plot_type == "importance_bar":
            fig = self.plot_importance_bar(**kwargs)
        elif plot_type == "network":
            fig = self.plot_network_saliency(**kwargs)
        elif plot_type == "comparison":
            fig = self.plot_comparison(**kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        fig.write_html(str(output_path))

    def export_to_json(self, output_path: str) -> None:
        """
        Export attributions to JSON.

        Args:
            output_path: Path to save JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "node_attributions": self.normalized_attributions,
            "method": self.method,
            "num_nodes": len(self.node_names),
            "metadata": {
                "has_comparison": bool(self.original_activations and self.corrupted_activations),
            },
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def get_top_nodes(self, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k nodes by attribution score.

        Args:
            k: Number of top nodes to return.

        Returns:
            List of (node, attribution) tuples sorted by attribution descending.
        """
        sorted_items = sorted(
            self.normalized_attributions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_items[:k]

    def get_attribution_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for attributions.

        Returns:
            Dictionary with summary statistics.
        """
        attributions_array = np.array(list(self.normalized_attributions.values()), dtype=np.float32)

        return {
            "mean": float(attributions_array.mean()),
            "median": float(np.median(attributions_array)),
            "max": float(attributions_array.max()),
            "min": float(attributions_array.min()),
            "std": float(attributions_array.std()),
            "method": self.method,
            "num_nodes": len(self.node_names),
        }
