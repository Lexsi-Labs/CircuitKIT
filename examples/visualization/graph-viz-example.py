"""
Example: Circuit Graph Visualization

Demonstrates how to visualize discovered circuits using CircuitGraphVisualizer.

This example shows:
1. Creating CircuitScores from discovery results
2. Building a graph structure
3. Generating interactive HTML visualizations
4. Using threshold filtering
5. Exporting graph data
"""

import os
from pathlib import Path
from circuitkit.artifacts.scores import CircuitScores
from circuitkit.visualize.graph_viz import CircuitGraphVisualizer


def create_sample_ioi_circuit():
    """
    Create a sample IOI circuit structure.

    IOI (Indirect Object Identification) is a common benchmark for circuit discovery.
    """
    # Define node scores (example values)
    node_scores = {
        # Layer 0: Early detection
        'A0.0': 0.92,
        'A0.1': 0.88,
        'A0.7': 0.85,
        'MLP 0': 0.45,

        # Layer 1: Processing
        'A1.4': 0.91,
        'A1.9': 0.87,
        'A1.10': 0.83,
        'MLP 1': 0.48,

        # Layer 2: Integration
        'A2.2': 0.89,
        'A2.5': 0.86,
        'A2.11': 0.82,
        'MLP 2': 0.51,

        # Layer 3: Output prediction
        'A3.0': 0.94,
        'A3.1': 0.90,
        'MLP 3': 0.55,
    }

    # Define node metadata
    nodes_dict = {}
    for node_name, score in node_scores.items():
        if node_name.startswith('A'):
            parts = node_name[1:].split('.')
            layer = int(parts[0])
            head = int(parts[1])
            nodes_dict[node_name] = {'layer': layer, 'type': 'attn_head', 'head': head}
        elif node_name.startswith('MLP'):
            parts = node_name.split()
            layer = int(parts[1])
            nodes_dict[node_name] = {'layer': layer, 'type': 'mlp'}

    # Define edges (signal flow through circuit)
    edges = [
        # Layer 0 -> Layer 1
        ('A0.0', 'A1.4'),
        ('A0.1', 'A1.9'),
        ('A0.7', 'A1.10'),
        ('MLP 0', 'A1.4'),

        # Layer 1 -> Layer 2
        ('A1.4', 'A2.2'),
        ('A1.9', 'A2.5'),
        ('A1.10', 'A2.11'),
        ('MLP 1', 'A2.2'),

        # Layer 2 -> Layer 3
        ('A2.2', 'A3.0'),
        ('A2.5', 'A3.1'),
        ('A2.11', 'A3.0'),
        ('MLP 2', 'A3.1'),

        # Cross-layer connections
        ('A0.0', 'A2.2'),
        ('A1.4', 'A3.0'),
    ]

    # Define edge scores (attribution strengths)
    edge_scores = {
        ('A0.0', 'A1.4'): 0.88,
        ('A0.1', 'A1.9'): 0.85,
        ('A0.7', 'A1.10'): 0.80,
        ('MLP 0', 'A1.4'): 0.45,
        ('A1.4', 'A2.2'): 0.87,
        ('A1.9', 'A2.5'): 0.84,
        ('A1.10', 'A2.11'): 0.79,
        ('MLP 1', 'A2.2'): 0.48,
        ('A2.2', 'A3.0'): 0.90,
        ('A2.5', 'A3.1'): 0.86,
        ('A2.11', 'A3.0'): 0.81,
        ('MLP 2', 'A3.1'): 0.52,
        ('A0.0', 'A2.2'): 0.75,
        ('A1.4', 'A3.0'): 0.85,
    }

    return node_scores, nodes_dict, edges, edge_scores


def example_basic_visualization():
    """Example 1: Basic HTML visualization."""
    print("Example 1: Basic HTML Visualization")
    print("=" * 50)

    # Create sample circuit
    node_scores, nodes_dict, edges, edge_scores = create_sample_ioi_circuit()

    # Create CircuitScores artifact
    scores = CircuitScores(
        task='ioi',
        model='gpt2',
        algorithm='eap',
        level='node',
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp(),
        discovery_cfg={
            'batch_size': 10,
            'num_examples': 100,
            'threshold': 0.1,
        }
    )

    # Create graph structure
    graph = {
        'nodes': nodes_dict,
        'edges': edges,
    }

    # Initialize visualizer
    viz = CircuitGraphVisualizer(graph, scores, edge_scores)

    # Generate HTML
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ioi_circuit_viz.html'

    viz.to_html(
        str(output_path),
        width=1400,
        height=900,
        node_size_scale=12.0,
        title='IOI Circuit Discovery - EAP'
    )

    print(f"Visualization saved to: {output_path}")
    print(f"Open in browser to view interactive graph")
    print()


def example_threshold_filtering():
    """Example 2: Create visualizations at different thresholds."""
    print("Example 2: Threshold Filtering")
    print("=" * 50)

    node_scores, nodes_dict, edges, edge_scores = create_sample_ioi_circuit()

    scores = CircuitScores(
        task='ioi',
        model='gpt2',
        algorithm='eap',
        level='node',
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp()
    )

    graph = {'nodes': nodes_dict, 'edges': edges}
    viz = CircuitGraphVisualizer(graph, scores, edge_scores)

    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)

    # Generate at different thresholds
    thresholds = [0.0, 0.3, 0.6, 0.8]

    for threshold in thresholds:
        filtered_viz = viz.filter_by_threshold(threshold=threshold)
        output_path = output_dir / f'ioi_circuit_threshold_{threshold:.1f}.html'

        filtered_viz.to_html(
            str(output_path),
            title=f'IOI Circuit (Min Score: {threshold:.2f})'
        )

        num_nodes = len(filtered_viz.node_scores)
        print(f"Threshold {threshold:.1f}: {num_nodes} nodes included")

    print(f"Visualizations saved to {output_dir}")
    print()


def example_analysis():
    """Example 3: Analyze circuit structure."""
    print("Example 3: Circuit Analysis")
    print("=" * 50)

    node_scores, nodes_dict, edges, edge_scores = create_sample_ioi_circuit()

    scores = CircuitScores(
        task='ioi',
        model='gpt2',
        algorithm='eap',
        level='node',
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp()
    )

    graph = {'nodes': nodes_dict, 'edges': edges}
    viz = CircuitGraphVisualizer(graph, scores, edge_scores)

    # Get top-k nodes
    top_5 = viz.get_top_k_nodes(k=5)
    print("Top 5 Important Nodes:")
    for i, (node, score) in enumerate(top_5.items(), 1):
        print(f"  {i}. {node}: {score:.3f}")
    print()

    # Get degree statistics
    degrees = viz.get_node_degree_stats()
    print("Node Connectivity:")
    high_connectivity = [
        (node, deg['in_degree'] + deg['out_degree'])
        for node, deg in degrees.items()
        if deg['in_degree'] + deg['out_degree'] > 0
    ]
    high_connectivity.sort(key=lambda x: x[1], reverse=True)
    for node, total_degree in high_connectivity[:5]:
        print(f"  {node}: {total_degree} connections")
    print()


def example_export_data():
    """Example 4: Export graph data for external processing."""
    print("Example 4: Export Graph Data")
    print("=" * 50)

    node_scores, nodes_dict, edges, edge_scores = create_sample_ioi_circuit()

    scores = CircuitScores(
        task='ioi',
        model='gpt2',
        algorithm='eap',
        level='node',
        node_scores=node_scores,
        timestamp=CircuitScores.create_timestamp()
    )

    graph = {'nodes': nodes_dict, 'edges': edges}
    viz = CircuitGraphVisualizer(graph, scores, edge_scores)

    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'ioi_circuit_data.json'

    viz.export_graph_data(str(output_path))
    print(f"Graph data exported to: {output_path}")
    print("This JSON can be used for:")
    print("  - Visualization in other tools (D3.js, Cytoscape)")
    print("  - Further analysis and processing")
    print("  - Publication/sharing of circuit structure")
    print()


def example_jupyter_widget():
    """Example 5: Interactive Jupyter widget."""
    print("Example 5: Jupyter Interactive Widget")
    print("=" * 50)
    print("""
In a Jupyter notebook, you can create an interactive widget:

```python
from circuitkit.visualize.graph_viz import CircuitGraphVisualizer
from circuitkit.artifacts.scores import CircuitScores

# ... create graph and scores ...

viz = CircuitGraphVisualizer(graph, scores, edge_scores)
widget = viz.interactive_widget(width=1200, height=800)
```

This creates a slider to dynamically filter nodes by minimum score.
The visualization updates in real-time as you move the slider.

Requirements:
  - ipywidgets: pip install ipywidgets
  - Jupyter environment (notebook, lab, or colab)
    """)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CircuitGraphVisualizer Examples")
    print("=" * 70 + "\n")

    try:
        example_basic_visualization()
        example_threshold_filtering()
        example_analysis()
        example_export_data()
        example_jupyter_widget()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"Error running examples: {e}")
        raise


if __name__ == '__main__':
    main()
