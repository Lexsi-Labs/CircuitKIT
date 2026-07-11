"""
Streamlit dashboard for circuit visualization and analysis.

Multi-page application with:
- Page 1: Upload circuit JSON, display graph
- Page 2: Saliency heatmaps
- Page 3: Feature attribution
- Page 4: Comparison across circuits
- Page 5: Interactive editor

Self-contained, runnable script: streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

try:
    from ..artifacts.scores import CircuitScores
    from .comparison import ComparisonDashboard
    from .feature_saliency import FeatureSaliencyVisualizer
    from .graph_viz import CircuitGraphVisualizer
    from .saliency import ActivationSaliencyVisualizer
except ImportError:
    # Support running as standalone script
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from artifacts.scores import CircuitScores
    from visualize.comparison import ComparisonDashboard
    from visualize.feature_saliency import FeatureSaliencyVisualizer
    from visualize.graph_viz import CircuitGraphVisualizer
    from visualize.saliency import ActivationSaliencyVisualizer


class StreamlitCircuitDashboard:
    """
    Streamlit-based circuit visualization dashboard.

    Features:
    - Multi-page navigation
    - Circuit upload and visualization
    - Saliency analysis
    - Feature attribution
    - Circuit comparison
    - Interactive editor
    """

    def __init__(self):
        """Initialize the dashboard."""
        self._setup_page()

    @staticmethod
    def _setup_page() -> None:
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="CircuitKit Dashboard",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Custom CSS for better styling
        st.markdown(
            """
        <style>
        .main {
            padding: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    def run(self) -> None:
        """Run the main dashboard."""
        st.sidebar.title("CircuitKit Dashboard")
        page = st.sidebar.radio(
            "Select Page",
            [
                "Home",
                "Circuit Visualization",
                "Activation Saliency",
                "Feature Attribution",
                "Circuit Comparison",
                "Interactive Editor",
            ],
        )

        if page == "Home":
            self.page_home()
        elif page == "Circuit Visualization":
            self.page_circuit_viz()
        elif page == "Activation Saliency":
            self.page_activation_saliency()
        elif page == "Feature Attribution":
            self.page_feature_attribution()
        elif page == "Circuit Comparison":
            self.page_circuit_comparison()
        elif page == "Interactive Editor":
            self.page_interactive_editor()

    @staticmethod
    def page_home() -> None:
        """Home page with overview and instructions."""
        st.title("⚡ CircuitKit Dashboard")

        st.markdown(
            """
        Welcome to the CircuitKit visualization dashboard!

        ### Features
        - **Circuit Visualization**: Visualize circuit graphs with node importance and edge attribution
        - **Activation Saliency**: Analyze activation patterns across layers
        - **Feature Attribution**: Understand which nodes most contribute to output
        - **Circuit Comparison**: Compare circuits across different conditions
        - **Interactive Editor**: Edit circuits and preview changes in real-time

        ### Getting Started
        1. Upload a circuit JSON file in the "Circuit Visualization" page
        2. Explore different visualizations and analyses
        3. Edit circuits in the "Interactive Editor" page
        4. Compare circuits across conditions in "Circuit Comparison" page

        ### Input Format
        Circuits should be JSON with the following structure:
        ```json
        {
            "nodes": {
                "A0.1": {"layer": 0, "type": "attention"},
                "MLP 3": {"layer": 3, "type": "mlp"}
            },
            "edges": [
                ["A0.1", "MLP 3"],
                ["A1.0", "MLP 3"]
            ]
        }
        ```
        """
        )

    @staticmethod
    def page_circuit_viz() -> None:
        """Circuit visualization page."""
        st.title("Circuit Visualization")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload Circuit")
            uploaded_file = st.file_uploader("Choose a circuit JSON file", type="json")

            if uploaded_file:
                try:
                    circuit = json.load(uploaded_file)
                    st.session_state.circuit = circuit
                    st.success("Circuit loaded successfully!")

                    # Display circuit stats
                    num_nodes = len(circuit.get("nodes", {}))
                    num_edges = len(circuit.get("edges", []))

                    st.metric("Number of Nodes", num_nodes)
                    st.metric("Number of Edges", num_edges)

                except Exception as e:
                    st.error(f"Error loading circuit: {e}")

        with col2:
            st.subheader("Circuit Structure")
            if "circuit" in st.session_state:
                st.json(st.session_state.circuit)

        # Visualization
        if "circuit" in st.session_state:
            st.subheader("Graph Visualization")

            # Create dummy scores for visualization
            node_names = list(st.session_state.circuit.get("nodes", {}).keys())
            node_scores = {node: np.random.rand() for node in node_names}

            try:
                scores = CircuitScores(
                    task="visualization",
                    model="unknown",
                    algorithm="manual",
                    level="node",
                    node_scores=node_scores,
                    timestamp="",
                )

                viz = CircuitGraphVisualizer(st.session_state.circuit, scores)
                fig = viz.plot_graph()

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error visualizing circuit: {e}")

    @staticmethod
    def page_activation_saliency() -> None:
        """Activation saliency page."""
        st.title("Activation Saliency Analysis")

        st.markdown(
            """
        Analyze activation patterns across layers. Upload activation data
        to visualize which tokens have the strongest activations.
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Upload activation data (JSON)", type="json", key="activations"
            )

            if uploaded_file:
                try:
                    activation_data = json.load(uploaded_file)
                    st.session_state.activations = activation_data
                    st.success("Activation data loaded!")
                except Exception as e:
                    st.error(f"Error loading activations: {e}")

        with col2:
            st.subheader("Options")
            plot_type = st.radio(
                "Visualization type",
                ["Layer comparison", "Aggregate", "Individual layers"],
            )

        if "activations" in st.session_state:
            try:
                activations = st.session_state.activations.get("layers", {})
                tokens = st.session_state.activations.get("tokens", None)

                viz = ActivationSaliencyVisualizer(activations, tokens)

                if plot_type == "Layer comparison":
                    fig = viz.plot_layer_comparison()
                elif plot_type == "Aggregate":
                    fig = viz.plot_aggregate_heatmap()
                else:
                    layers = st.multiselect(
                        "Select layers",
                        list(activations.keys()),
                        default=list(activations.keys())[:3],
                    )
                    fig = viz.plot_layer_heatmaps(layers=layers)

                st.plotly_chart(fig, use_container_width=True)

                # Summary statistics
                st.subheader("Summary Statistics")
                summary = viz.get_saliency_summary()
                st.json(summary)

            except Exception as e:
                st.error(f"Error visualizing activations: {e}")

    @staticmethod
    def page_feature_attribution() -> None:
        """Feature attribution page."""
        st.title("Feature Attribution Analysis")

        st.markdown(
            """
        Analyze which circuit nodes most contribute to the model output.
        """
        )

        uploaded_file = st.file_uploader(
            "Upload attribution data (JSON)", type="json", key="attributions"
        )

        if uploaded_file:
            try:
                attribution_data = json.load(uploaded_file)
                node_attributions = attribution_data.get("attributions", {})

                viz = FeatureSaliencyVisualizer(node_attributions)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Nodes")
                    top_k = st.slider("Show top-k nodes", 5, len(node_attributions), 10)
                    fig = viz.plot_importance_bar(top_k=top_k)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Network View")
                    fig = viz.plot_network_saliency()
                    st.plotly_chart(fig, use_container_width=True)

                # Summary
                st.subheader("Attribution Summary")
                summary = viz.get_attribution_summary()
                st.json(summary)

            except Exception as e:
                st.error(f"Error visualizing attributions: {e}")

    @staticmethod
    def page_circuit_comparison() -> None:
        """Circuit comparison page."""
        st.title("Circuit Comparison")

        st.markdown(
            """
        Compare circuits across different conditions (seeds, corruptions, tasks).
        """
        )

        uploaded_file = st.file_uploader(
            "Upload comparison data (JSON)", type="json", key="comparison"
        )

        if uploaded_file:
            try:
                comparison_data = json.load(uploaded_file)
                circuits = comparison_data.get("circuits", {})

                if len(circuits) < 2:
                    st.warning("Need at least 2 circuits for comparison")
                else:
                    comparison_type = st.radio(
                        "Comparison type",
                        ["Stability", "Robustness", "Generalization"],
                    )

                    dashboard = ComparisonDashboard(
                        circuits, comparison_type=comparison_type.lower()
                    )

                    # Select visualization
                    viz_type = st.selectbox(
                        "Visualization",
                        [
                            "Stability Heatmap",
                            "Correlation",
                            "Robustness",
                            "Transfer",
                            "Distribution",
                        ],
                    )

                    if viz_type == "Stability Heatmap":
                        fig = dashboard.plot_stability_heatmap()
                    elif viz_type == "Correlation":
                        fig = dashboard.plot_correlation_matrix()
                    elif viz_type == "Robustness":
                        fig = dashboard.plot_robustness_comparison()
                    elif viz_type == "Transfer":
                        fig = dashboard.plot_transfer_matrix()
                    else:
                        fig = dashboard.plot_distribution_comparison()

                    st.plotly_chart(fig, use_container_width=True)

                    # Summary stats
                    st.subheader("Comparison Statistics")
                    summary = dashboard.get_summary_stats()
                    st.json(summary)

            except Exception as e:
                st.error(f"Error comparing circuits: {e}")

    @staticmethod
    def page_interactive_editor() -> None:
        """Interactive editor page."""
        st.title("Interactive Circuit Editor")

        st.markdown(
            """
        Edit circuits directly in the dashboard. Add/remove nodes and edges,
        and save your changes.
        """
        )

        if "circuit" in st.session_state:
            circuit = st.session_state.circuit

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Current Circuit")
                st.json(circuit)

            with col2:
                st.subheader("Edit Operations")

                operation = st.radio(
                    "Select operation", ["Add Node", "Remove Node", "Add Edge", "Remove Edge"]
                )

                if operation == "Add Node":
                    node_name = st.text_input("New node name")
                    if st.button("Add Node"):
                        if "nodes" not in circuit:
                            circuit["nodes"] = {}
                        circuit["nodes"][node_name] = {}
                        st.session_state.circuit = circuit
                        st.success(f"Added node: {node_name}")

                elif operation == "Remove Node":
                    nodes = list(circuit.get("nodes", {}).keys())
                    if nodes:
                        node = st.selectbox("Select node to remove", nodes)
                        if st.button("Remove Node"):
                            del circuit["nodes"][node]
                            if "edges" in circuit:
                                circuit["edges"] = [e for e in circuit["edges"] if node not in e]
                            st.session_state.circuit = circuit
                            st.success(f"Removed node: {node}")

                elif operation == "Add Edge":
                    nodes = list(circuit.get("nodes", {}).keys())
                    if nodes:
                        src = st.selectbox("Source node", nodes)
                        tgt = st.selectbox("Target node", nodes)
                        if st.button("Add Edge"):
                            if "edges" not in circuit:
                                circuit["edges"] = []
                            if [src, tgt] not in circuit["edges"]:
                                circuit["edges"].append([src, tgt])
                                st.session_state.circuit = circuit
                                st.success(f"Added edge: {src} → {tgt}")

                elif operation == "Remove Edge":
                    edges = circuit.get("edges", [])
                    if edges:
                        [f"{e[0]} → {e[1]}" for e in edges]
                        edge_idx = st.selectbox("Select edge to remove", range(len(edges)))
                        if st.button("Remove Edge"):
                            del circuit["edges"][edge_idx]
                            st.session_state.circuit = circuit
                            st.success("Removed edge")

            # Export
            st.subheader("Export")
            if st.button("Export Circuit"):
                circuit_json = json.dumps(circuit, indent=2)
                st.download_button(
                    "Download Circuit",
                    circuit_json,
                    "circuit_edited.json",
                    "application/json",
                )

        else:
            st.info("Please upload a circuit in the 'Circuit Visualization' page first.")


def main():
    """Main entry point."""
    dashboard = StreamlitCircuitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
