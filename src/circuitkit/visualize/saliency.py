"""
Activation saliency heatmaps for circuit visualization.

Visualizes which tokens/positions have the strongest activations across
different layers of a neural network, supporting layer-by-layer and
aggregate visualizations with color intensity indicating activation magnitude.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from .theme import PALETTE, get_plotly_layout
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ActivationSaliencyVisualizer:
    """
    Visualizes activation saliency heatmaps showing which tokens/positions
    have the strongest activations across layers.

    Supports:
    - Layer-by-layer visualization
    - Aggregate across layers
    - Token-level and position-level analysis
    - Color intensity = activation magnitude
    - Export to HTML with interactive Plotly
    """

    def __init__(
        self,
        activations: Dict[str, np.ndarray],
        tokens: Optional[List[str]] = None,
        layer_names: Optional[List[str]] = None,
    ):
        """
        Initialize the activation saliency visualizer.

        Args:
            activations: Dictionary mapping layer names to activation tensors.
                Shape per layer: (seq_len,) or (seq_len, hidden_dim) or similar.
                Can also be 2D for position-level aggregation.
            tokens: Optional list of token strings for x-axis labels.
                If not provided, uses token indices.
            layer_names: Optional list of layer names. If not provided,
                uses keys from activations dict.
        """
        self.activations = activations
        self.tokens = tokens
        self.layer_names = layer_names or list(activations.keys())

        # Validate activations
        self._validate_activations()

        # Normalize activations per layer
        self.normalized_activations = self._normalize_activations()

    def _validate_activations(self) -> None:
        """Validate activation tensor shapes and values."""
        seq_len = None
        for layer_name, activation in self.activations.items():
            if not isinstance(activation, (np.ndarray, list)):
                raise TypeError(
                    f"Activation for {layer_name} must be numpy array or list, "
                    f"got {type(activation)}"
                )
            if isinstance(activation, list):
                activation = np.array(activation)

            if activation.ndim < 1:
                raise ValueError(
                    f"Activation for {layer_name} must be at least 1D, "
                    f"got shape {activation.shape}"
                )

            # Track and enforce consistent sequence length across layers
            layer_seq_len = activation.shape[0]
            if seq_len is None:
                seq_len = layer_seq_len
            elif layer_seq_len != seq_len:
                raise ValueError(
                    f"Activation for {layer_name} has sequence length {layer_seq_len}, "
                    f"expected {seq_len} (all layers must have the same sequence length)"
                )

        # Validate tokens length matches sequence length
        if self.tokens is not None and seq_len is not None:
            if len(self.tokens) != seq_len:
                raise ValueError(
                    f"Length of tokens ({len(self.tokens)}) must match activation "
                    f"sequence length ({seq_len})"
                )

    def _normalize_activations(self) -> Dict[str, np.ndarray]:
        """Normalize activations per layer to [0, 1] range."""
        normalized = {}
        for layer_name, activation in self.activations.items():
            activation = np.asarray(activation, dtype=np.float32)

            # Handle multi-dimensional activations (e.g., (seq_len, hidden_dim))
            # by taking max or mean across last dimensions
            if activation.ndim > 1:
                activation = np.max(np.abs(activation), axis=tuple(range(1, activation.ndim)))

            # Normalize to [0, 1]
            act_min, act_max = activation.min(), activation.max()
            if act_max > act_min:
                normalized[layer_name] = (activation - act_min) / (act_max - act_min)
            else:
                normalized[layer_name] = np.zeros_like(activation)

        return normalized

    def plot_layer_heatmaps(
        self,
        layers: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        colorscale: str = "Viridis",
    ) -> go.Figure:
        """
        Create layer-by-layer heatmap visualization.

        Args:
            layers: List of layer names to visualize. If None, uses all layers.
            figsize: Figure size (width, height) in inches.
            colorscale: Plotly colorscale name (e.g., 'Viridis', 'Reds', 'Blues').

        Returns:
            Plotly Figure object with subplots.
        """
        layers = layers or self.layer_names
        num_layers = len(layers)

        # Create subplots grid
        rows = (num_layers + 1) // 2
        cols = 2 if num_layers > 1 else 1

        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Layer {layer}" for layer in layers],
            specs=[[{"type": "heatmap"} for _ in range(cols)] for _ in range(rows)],
        )

        for idx, layer_name in enumerate(layers):
            row = (idx // cols) + 1
            col = (idx % cols) + 1

            activation = self.normalized_activations[layer_name]

            # Create heatmap data
            heatmap_data = activation.reshape(1, -1)  # (1, seq_len)

            # Create heatmap trace
            heatmap = go.Heatmap(
                z=heatmap_data,
                colorscale=colorscale,
                showscale=(idx == 0),  # Only show colorbar for first subplot
                colorbar=(
                    dict(
                        title="Activation<br>Magnitude",
                        x=1.02,
                        len=0.9,
                    )
                    if idx == 0
                    else None
                ),
                hovertemplate="Token %{x}: %{z:.3f}<extra></extra>",
            )

            fig.add_trace(heatmap, row=row, col=col)

            # Update axes
            x_labels = self.tokens if self.tokens else [str(i) for i in range(len(activation))]
            fig.update_xaxes(
                ticktext=x_labels,
                tickvals=list(range(len(activation))),
                row=row,
                col=col,
            )
            fig.update_yaxes(showticklabels=False, row=row, col=col)

        # Update layout
        height = int(figsize[1] * 100)  # Convert inches to approximate pixels
        width = int(figsize[0] * 100)

        fig.update_layout(**get_plotly_layout(
            title="Activation Saliency by Layer",
            width=width,
            height=height,
        ))

        return fig

    def plot_aggregate_heatmap(
        self,
        aggregation: str = "mean",
        colorscale: str = "Reds",
        figsize: Tuple[int, int] = (12, 6),
    ) -> go.Figure:
        """
        Create aggregate heatmap across all layers.

        Args:
            aggregation: How to aggregate across layers ('mean', 'max', 'sum').
            colorscale: Plotly colorscale name.
            figsize: Figure size (width, height) in inches.

        Returns:
            Plotly Figure object.
        """
        # Stack all normalized activations
        activations_list = [self.normalized_activations[layer] for layer in self.layer_names]
        stacked = np.stack(activations_list, axis=0)  # (num_layers, seq_len)

        # Aggregate across layers
        if aggregation == "mean":
            aggregated = stacked.mean(axis=0)
        elif aggregation == "max":
            aggregated = stacked.max(axis=0)
        elif aggregation == "sum":
            aggregated = stacked.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Normalize aggregated
        agg_min, agg_max = aggregated.min(), aggregated.max()
        if agg_max > agg_min:
            aggregated = (aggregated - agg_min) / (agg_max - agg_min)

        heatmap_data = aggregated.reshape(1, -1)

        # Create figure
        x_labels = self.tokens if self.tokens else [str(i) for i in range(len(aggregated))]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data,
                x=x_labels,
                colorscale=colorscale,
                colorbar=dict(title="Activation<br>Magnitude"),
                hovertemplate="Token %{x}: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(**get_plotly_layout(
            title=f"Aggregate Activation Saliency ({aggregation.capitalize()})",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            xaxis=dict(showgrid=False, zeroline=False, title="Token",
                    showticklabels=True, color=PALETTE.text_secondary),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ))

        fig.update_yaxes(showticklabels=False)

        return fig

    def plot_layer_comparison(
        self,
        colorscale: str = "Viridis",
        figsize: Tuple[int, int] = (14, 8),
    ) -> go.Figure:
        """
        Create stacked heatmap showing all layers at once.

        Args:
            colorscale: Plotly colorscale name.
            figsize: Figure size (width, height) in inches.

        Returns:
            Plotly Figure object.
        """
        # Stack all normalized activations
        activations_list = [self.normalized_activations[layer] for layer in self.layer_names]
        stacked = np.stack(activations_list, axis=0)  # (num_layers, seq_len)

        x_labels = self.tokens if self.tokens else [str(i) for i in range(stacked.shape[1])]

        fig = go.Figure(
            data=go.Heatmap(
                z=stacked,
                y=self.layer_names,
                x=x_labels,
                colorscale=colorscale,
                colorbar=dict(title="Activation<br>Magnitude"),
                hovertemplate="Layer %{y}, Token %{x}: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(**get_plotly_layout(
            title="Activation Saliency: All Layers",
            width=int(figsize[0] * 100),
            height=int(figsize[1] * 100),
            xaxis=dict(showgrid=False, zeroline=False, title="Token",
                    showticklabels=True, color=PALETTE.text_secondary),
            yaxis=dict(showgrid=False, zeroline=False, title="Layer",
                    showticklabels=True, color=PALETTE.text_secondary),
        ))

        return fig

    def export_to_html(
        self,
        output_path: str,
        plot_type: str = "comparison",
        **kwargs,
    ) -> None:
        """
        Export visualization to HTML file.

        Args:
            output_path: Path to save HTML file.
            plot_type: Type of plot ('layer_heatmaps', 'aggregate', 'comparison').
            **kwargs: Additional arguments passed to plotting method.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if plot_type == "layer_heatmaps":
            fig = self.plot_layer_heatmaps(**kwargs)
        elif plot_type == "aggregate":
            fig = self.plot_aggregate_heatmap(**kwargs)
        elif plot_type == "comparison":
            fig = self.plot_layer_comparison(**kwargs)
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        fig.write_html(str(output_path))

    def export_to_json(self, output_path: str) -> None:
        """
        Export normalized activations to JSON.

        Args:
            output_path: Path to save JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "layers": {},
            "tokens": self.tokens,
            "metadata": {
                "num_layers": len(self.layer_names),
                "num_tokens": len(self.normalized_activations[self.layer_names[0]]),
            },
        }

        for layer_name, activation in self.normalized_activations.items():
            export_data["layers"][layer_name] = activation.tolist()

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def get_top_tokens(
        self,
        layer: str,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k tokens by activation magnitude for a specific layer.

        Args:
            layer: Layer name.
            k: Number of top tokens to return.

        Returns:
            List of (token, activation) tuples sorted by activation descending.
        """
        if layer not in self.normalized_activations:
            raise ValueError(f"Layer {layer} not found in activations")

        activation = self.normalized_activations[layer]
        top_indices = np.argsort(activation)[-k:][::-1]

        results = []
        for idx in top_indices:
            token = self.tokens[idx] if self.tokens else str(idx)
            results.append((token, float(activation[idx])))

        return results

    def get_saliency_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all activations.

        Returns:
            Dictionary with summary stats per layer.
        """
        summary = {}
        for layer_name in self.layer_names:
            activation = self.normalized_activations[layer_name]
            summary[layer_name] = {
                "mean": float(activation.mean()),
                "max": float(activation.max()),
                "min": float(activation.min()),
                "std": float(activation.std()),
            }
        return summary
