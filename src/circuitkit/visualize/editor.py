"""
Interactive circuit editor for Jupyter notebooks using ipywidgets.

Allows users to interactively add/remove nodes and edges, toggle edges,
preview effects on model output, and save edited circuits to JSON.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


import logging

logger = logging.getLogger(__name__)

try:
    import ipywidgets as widgets
    from IPython.display import display

    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


class CircuitEditor:
    """
    Interactive Jupyter widget for editing circuits.

    Allows users to:
    - Add/remove nodes
    - Toggle edges on/off
    - Preview effect on model output in real-time (if callback provided)
    - Save edited circuit to JSON
    """

    def __init__(
        self,
        initial_circuit: Dict[str, Any],
        node_names: Optional[List[str]] = None,
        output_callback: Optional[Callable[[Dict], Any]] = None,
    ):
        """
        Initialize the circuit editor.

        Args:
            initial_circuit: Initial circuit structure with 'nodes' and 'edges'.
            node_names: Optional list of available node names for adding new nodes.
            output_callback: Optional callable that takes the edited circuit
                and returns output data for visualization. Called on circuit changes.
        """
        if not IPYWIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for CircuitEditor. Install with: pip install ipywidgets"
            )

        self.initial_circuit = initial_circuit
        self.current_circuit = self._deep_copy(initial_circuit)
        self.node_names = node_names or self._extract_node_names(initial_circuit)
        self.output_callback = output_callback

        # Track change history
        self.change_history: List[Dict[str, Any]] = []

        # Build UI
        self._build_ui()

    @staticmethod
    def _deep_copy(obj: Dict) -> Dict:
        """Deep copy a dictionary."""
        import json


        return json.loads(json.dumps(obj))

    def _extract_node_names(self, circuit: Dict) -> List[str]:
        """Extract all unique node names from circuit."""
        nodes = set()
        if "nodes" in circuit and isinstance(circuit["nodes"], dict):
            nodes.update(circuit["nodes"].keys())
        if "edges" in circuit:
            for edge in circuit["edges"]:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        return sorted(list(nodes))

    def _build_ui(self) -> None:
        """Build the interactive UI components."""
        # Title
        self.title = widgets.HTML("<h3>Interactive Circuit Editor</h3>")

        # Node management section
        self.node_dropdown = widgets.Combobox(
            options=tuple(self.node_names),  # Combobox requires a tuple or list
            value=self.node_names[0] if self.node_names else "",
            description="Node Name:",
            placeholder="Type a new node name...",
            ensure_option=False,  # Crucial: allows values not in the options list
            disabled=False,
        )
        self.add_node_button = widgets.Button(description="Add Node")
        self.remove_node_button = widgets.Button(description="Remove Node")
        self.add_node_button.on_click(self._on_add_node)
        self.remove_node_button.on_click(self._on_remove_node)

        node_control_box = widgets.HBox(
            [
                widgets.VBox([self.node_dropdown, self.add_node_button]),
                self.remove_node_button,
            ]
        )

        # Edge management section
        self.source_dropdown = widgets.Dropdown(
            options=self.node_names,
            description="Source:",
            disabled=False,
        )
        self.target_dropdown = widgets.Dropdown(
            options=self.node_names,
            description="Target:",
            disabled=False,
        )
        self.add_edge_button = widgets.Button(description="Add Edge")
        self.remove_edge_button = widgets.Button(description="Remove Edge")
        self.add_edge_button.on_click(self._on_add_edge)
        self.remove_edge_button.on_click(self._on_remove_edge)

        edge_control_box = widgets.VBox(
            [
                widgets.HBox([self.source_dropdown, self.target_dropdown]),
                widgets.HBox([self.add_edge_button, self.remove_edge_button]),
            ]
        )

        # Display current circuit structure
        self.circuit_display = widgets.HTML(self._format_circuit_display())

        # Export section
        self.export_button = widgets.Button(description="Export to JSON")
        self.export_button.on_click(self._on_export)
        self.export_path = widgets.Text(
            value="circuit_edited.json",
            description="Save to:",
            disabled=False,
        )

        export_box = widgets.HBox([self.export_button, self.export_path])

        # Reset button
        self.reset_button = widgets.Button(description="Reset to Initial")
        self.reset_button.on_click(self._on_reset)

        # Undo button
        self.undo_button = widgets.Button(description="Undo")
        self.undo_button.on_click(self._on_undo)
        self.undo_button.disabled = True

        history_box = widgets.HBox([self.reset_button, self.undo_button])

        # Statistics display
        self.stats_display = widgets.HTML(self._format_stats_display())

        # Output display
        self.output_display = widgets.HTML("")

        # Assemble main UI
        self.ui = widgets.VBox(
            [
                self.title,
                widgets.HTML("<h4>Nodes</h4>"),
                node_control_box,
                widgets.HTML("<h4>Edges</h4>"),
                edge_control_box,
                widgets.HTML("<h4>Circuit Structure</h4>"),
                self.circuit_display,
                widgets.HTML("<h4>Statistics</h4>"),
                self.stats_display,
                widgets.HTML("<h4>Export</h4>"),
                export_box,
                history_box,
                widgets.HTML("<h4>Output Preview</h4>"),
                self.output_display,
            ]
        )

    def _format_circuit_display(self) -> str:
        """Format current circuit as HTML table."""
        html = "<p><strong>Nodes:</strong> "
        if "nodes" in self.current_circuit and self.current_circuit["nodes"]:
            nodes = list(self.current_circuit["nodes"].keys())
            html += ", ".join(f"<code>{n}</code>" for n in nodes)
        else:
            html += "None"
        html += "</p>"

        html += "<p><strong>Edges:</strong> "
        if "edges" in self.current_circuit and self.current_circuit["edges"]:
            edges = self.current_circuit["edges"]
            edge_strs = []
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    edge_strs.append(f"{edge[0]} → {edge[1]}")
            html += ", ".join(f"<code>{e}</code>" for e in edge_strs)
        else:
            html += "None"
        html += "</p>"

        return html

    def _format_stats_display(self) -> str:
        """Format statistics about current circuit."""
        nodes = self.current_circuit.get("nodes", {})
        edges = self.current_circuit.get("edges", [])

        num_nodes = len(nodes) if isinstance(nodes, dict) else 0
        num_edges = len(edges) if isinstance(edges, list) else 0

        return f"""
        <p>
            <strong>Nodes:</strong> {num_nodes} |
            <strong>Edges:</strong> {num_edges} |
            <strong>Changes:</strong> {len(self.change_history)}
        </p>
        """

    def _on_add_node(self, button) -> None:
        """Handle add node button click."""
        node = self.node_dropdown.value
        if not node:
            return  # Prevent adding empty strings

        if "nodes" not in self.current_circuit:
            self.current_circuit["nodes"] = {}

        if node not in self.current_circuit["nodes"]:
            self.change_history.append(
                {
                    "action": "add_node",
                    "node": node,
                    "circuit_before": self._deep_copy(self.current_circuit),
                }
            )
            self.current_circuit["nodes"][node] = {}

            # Update the list of valid node names for the entire UI
            new_names = sorted(list(set(list(self.node_names) + [node])))
            self.node_names = new_names
            self.node_dropdown.options = tuple(new_names)
            self.source_dropdown.options = tuple(new_names)
            self.target_dropdown.options = tuple(new_names)

            self._update_display()

    def _on_remove_node(self, button) -> None:
        """Handle remove node button click."""
        node = self.node_dropdown.value
        if "nodes" in self.current_circuit and node in self.current_circuit["nodes"]:
            self.change_history.append(
                {
                    "action": "remove_node",
                    "node": node,
                    "circuit_before": self._deep_copy(self.current_circuit),
                }
            )
            del self.current_circuit["nodes"][node]

            # Also remove associated edges
            if "edges" in self.current_circuit:
                self.current_circuit["edges"] = [
                    e
                    for e in self.current_circuit["edges"]
                    if not (
                        isinstance(e, (list, tuple))
                        and len(e) >= 2
                        and (e[0] == node or e[1] == node)
                    )
                ]

            self._update_display()

    def _on_add_edge(self, button) -> None:
        """Handle add edge button click."""
        src = self.source_dropdown.value
        tgt = self.target_dropdown.value

        if src == tgt:
            self.output_display.value = (
                "<p style='color:red'>Source and target must be different.</p>"
            )
            return

        if "edges" not in self.current_circuit:
            self.current_circuit["edges"] = []

        edge = [src, tgt]
        if edge not in self.current_circuit["edges"]:
            self.change_history.append(
                {
                    "action": "add_edge",
                    "edge": edge,
                    "circuit_before": self._deep_copy(self.current_circuit),
                }
            )
            self.current_circuit["edges"].append(edge)
            self._update_display()

    def _on_remove_edge(self, button) -> None:
        """Handle remove edge button click."""
        src = self.source_dropdown.value
        tgt = self.target_dropdown.value

        if "edges" in self.current_circuit:
            edge = [src, tgt]
            if edge in self.current_circuit["edges"]:
                self.change_history.append(
                    {
                        "action": "remove_edge",
                        "edge": edge,
                        "circuit_before": self._deep_copy(self.current_circuit),
                    }
                )
                self.current_circuit["edges"].remove(edge)
                self._update_display()

    def _on_export(self, button) -> None:
        """Handle export button click."""
        try:
            path = Path(self.export_path.value)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.current_circuit, f, indent=2)
            self.output_display.value = f"<p style='color:green'>Exported to {path}</p>"
        except Exception as e:
            self.output_display.value = f"<p style='color:red'>Export failed: {str(e)}</p>"

    def _on_reset(self, button) -> None:
        """Handle reset button click."""
        self.current_circuit = self._deep_copy(self.initial_circuit)
        self.change_history = []
        self._update_display()

    def _on_undo(self, button) -> None:
        """Handle undo button click."""
        if self.change_history:
            last_change = self.change_history.pop()
            self.current_circuit = last_change["circuit_before"]
            self._update_display()

    def _update_display(self) -> None:
        """Update all display elements."""
        self.circuit_display.value = self._format_circuit_display()
        self.stats_display.value = self._format_stats_display()
        self.undo_button.disabled = len(self.change_history) == 0

        # Call output callback if provided
        if self.output_callback:
            try:
                output = self.output_callback(self.current_circuit)
                self.output_display.value = f"<p><strong>Output:</strong> {output}</p>"
            except Exception as e:
                self.output_display.value = f"<p style='color:orange'>Callback error: {str(e)}</p>"

    def display(self) -> None:
        """Display the editor widget in Jupyter."""
        if not IPYWIDGETS_AVAILABLE:
            logger.info("ipywidgets is required. Install with: pip install ipywidgets")
            return
        display(self.ui)

    def get_circuit(self) -> Dict[str, Any]:
        """Get the current edited circuit."""
        return self._deep_copy(self.current_circuit)

    def get_changes(self) -> List[Dict[str, Any]]:
        """Get the change history."""
        return self.change_history.copy()

    def save_circuit(self, path: str) -> None:
        """Save the current circuit to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.current_circuit, f, indent=2)
