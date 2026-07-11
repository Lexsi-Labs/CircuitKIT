"""
Tests for interactive circuit editor.
"""

import json

import pytest

from circuitkit.visualize.editor import CircuitEditor


class TestCircuitEditor:
    """Test circuit editor."""

    @pytest.fixture
    def sample_circuit(self):
        """Create sample circuit."""
        return {
            "nodes": {
                "A0.0": {"layer": 0, "type": "attention"},
                "A0.1": {"layer": 0, "type": "attention"},
                "MLP0": {"layer": 0, "type": "mlp"},
            },
            "edges": [
                ["A0.0", "MLP0"],
                ["A0.1", "MLP0"],
            ],
        }

    def test_initialization(self, sample_circuit):
        """Test editor initialization."""
        editor = CircuitEditor(sample_circuit)
        assert editor.current_circuit == sample_circuit

    def test_extract_node_names(self, sample_circuit):
        """Test node name extraction."""
        editor = CircuitEditor(sample_circuit)
        node_names = editor.node_names
        assert "A0.0" in node_names
        assert "A0.1" in node_names
        assert "MLP0" in node_names

    def test_deep_copy(self):
        """Test that deep copy works correctly."""
        original = {"nodes": {"A": {}}, "edges": [["A", "B"]]}
        editor = CircuitEditor(original)

        # Modify current circuit
        editor.current_circuit["nodes"]["C"] = {}

        # Original should be unchanged
        assert "C" not in editor.initial_circuit["nodes"]

    def test_add_node(self, sample_circuit):
        """Test adding a node."""
        editor = CircuitEditor(sample_circuit)
        initial_count = len(editor.current_circuit["nodes"])

        # Simulate add node
        editor.current_circuit["nodes"]["NEW_NODE"] = {}

        assert len(editor.current_circuit["nodes"]) == initial_count + 1
        assert "NEW_NODE" in editor.current_circuit["nodes"]

    def test_remove_node(self, sample_circuit):
        """Test removing a node."""
        editor = CircuitEditor(sample_circuit)

        # Remove a node
        if "A0.0" in editor.current_circuit["nodes"]:
            del editor.current_circuit["nodes"]["A0.0"]

        assert "A0.0" not in editor.current_circuit["nodes"]

    def test_add_edge(self, sample_circuit):
        """Test adding an edge."""
        editor = CircuitEditor(sample_circuit)
        initial_count = len(editor.current_circuit.get("edges", []))

        # Add edge
        editor.current_circuit["edges"].append(["A0.0", "A0.1"])

        assert len(editor.current_circuit["edges"]) == initial_count + 1

    def test_remove_edge(self, sample_circuit):
        """Test removing an edge."""
        editor = CircuitEditor(sample_circuit)
        initial_count = len(editor.current_circuit["edges"])

        # Remove first edge
        if editor.current_circuit["edges"]:
            editor.current_circuit["edges"].pop(0)

        assert len(editor.current_circuit["edges"]) == initial_count - 1

    def test_change_history(self, sample_circuit):
        """Test change history tracking."""
        editor = CircuitEditor(sample_circuit)

        # Make a change
        editor.current_circuit["nodes"]["NEW"] = {}
        editor.change_history.append(
            {
                "action": "add_node",
                "node": "NEW",
            }
        )

        assert len(editor.change_history) > 0

    def test_save_circuit(self, sample_circuit, tmp_path):
        """Test saving circuit to file."""
        editor = CircuitEditor(sample_circuit)
        output_file = tmp_path / "circuit.json"

        editor.save_circuit(str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            saved = json.load(f)
            assert saved == editor.current_circuit

    def test_get_circuit(self, sample_circuit):
        """Test getting current circuit."""
        editor = CircuitEditor(sample_circuit)
        circuit = editor.get_circuit()

        # Should be a copy
        circuit["nodes"]["NEW"] = {}
        assert "NEW" not in editor.current_circuit["nodes"]

    def test_get_changes(self, sample_circuit):
        """Test getting change history."""
        editor = CircuitEditor(sample_circuit)
        editor.change_history.append({"action": "test"})

        changes = editor.get_changes()
        assert len(changes) > 0

    def test_empty_circuit(self):
        """Test editor with empty circuit."""
        empty_circuit = {"nodes": {}, "edges": []}
        editor = CircuitEditor(empty_circuit)
        assert len(editor.current_circuit["nodes"]) == 0

    def test_circuit_without_edges(self):
        """Test circuit without edges."""
        circuit = {
            "nodes": {
                "A0.0": {},
                "A0.1": {},
            }
        }
        editor = CircuitEditor(circuit)
        assert editor.current_circuit is not None

    def test_format_circuit_display(self, sample_circuit):
        """Test circuit display formatting."""
        editor = CircuitEditor(sample_circuit)
        display = editor._format_circuit_display()
        assert isinstance(display, str)
        assert "A0.0" in display or "Nodes:" in display

    def test_format_stats_display(self, sample_circuit):
        """Test stats display formatting."""
        editor = CircuitEditor(sample_circuit)
        stats = editor._format_stats_display()
        assert isinstance(stats, str)
        assert "Nodes:" in stats or "Edges:" in stats

    def test_circuit_with_node_metadata(self):
        """Test circuit with node metadata."""
        circuit = {
            "nodes": {
                "A0.0": {"layer": 0, "head": 0, "importance": 0.8},
                "MLP0": {"layer": 0, "neurons": 512},
            },
            "edges": [],
        }
        editor = CircuitEditor(circuit)
        assert editor.current_circuit == circuit

    def test_extraction_from_edges_only(self):
        """Test node extraction when edges are present but nodes aren't."""
        circuit = {
            "edges": [
                ["A0.0", "MLP0"],
                ["A1.0", "MLP1"],
            ]
        }
        editor = CircuitEditor(circuit)
        assert "A0.0" in editor.node_names
        assert "MLP0" in editor.node_names

    def test_callback_execution(self, sample_circuit):
        """Test output callback execution."""
        callback_results = []

        def dummy_callback(circuit):
            callback_results.append(circuit)
            return "test_output"

        editor = CircuitEditor(sample_circuit, output_callback=dummy_callback)
        assert editor.output_callback is not None

    def test_handler_add_node_free_text(self, sample_circuit):
        """Verify that we can now add nodes by typing arbitrary text."""
        editor = CircuitEditor(sample_circuit)

        # This previously failed with TraitError; now it succeeds[cite: 4]
        editor.node_dropdown.value = "CUSTOM_NODE_99"
        editor._on_add_node(None)

        assert "CUSTOM_NODE_99" in editor.current_circuit["nodes"]
        # Check that the UI updated its internal options list
        assert "CUSTOM_NODE_99" in editor.node_dropdown.options

    def test_handler_remove_node_with_edges(self, sample_circuit):
        """Test that removing a node also cleans up associated edges."""
        editor = CircuitEditor(sample_circuit)
        editor.node_dropdown.value = "A0.0"

        editor._on_remove_node(None)

        # Verify node is gone
        assert "A0.0" not in editor.current_circuit["nodes"]
        # Verify edges involving A0.0 are automatically removed
        for edge in editor.current_circuit["edges"]:
            assert "A0.0" not in edge

    def test_prevent_self_loop_edge(self, sample_circuit):
        """Verify that edges cannot have the same source and target."""
        editor = CircuitEditor(sample_circuit)
        editor.source_dropdown.value = "MLP0"
        editor.target_dropdown.value = "MLP0"

        initial_edge_count = len(editor.current_circuit["edges"])
        editor._on_add_edge(None)

        # Edge should not be added
        assert len(editor.current_circuit["edges"]) == initial_edge_count
        # Error message should be displayed in the UI
        assert "Source and target must be different" in editor.output_display.value

    def test_add_duplicate_edge(self, sample_circuit):
        """Ensure duplicate edges are not added to the history or circuit."""
        editor = CircuitEditor(sample_circuit)
        editor.source_dropdown.value = "A0.0"
        editor.target_dropdown.value = "MLP0"  # This edge already exists

        editor._on_add_edge(None)
        # History should not increase because the edge already exists
        assert len(editor.change_history) == 0

    def test_undo_button_state_sync(self, sample_circuit):
        """Test if the undo button enables/disables correctly based on history."""
        editor = CircuitEditor(sample_circuit)
        assert editor.undo_button.disabled is True

        # Perform action
        editor.node_dropdown.value = "A0.0"
        editor._on_remove_node(None)
        assert editor.undo_button.disabled is False

        # Perform undo
        editor._on_undo(None)
        assert editor.undo_button.disabled is True
        assert "A0.0" in editor.current_circuit["nodes"]

    def test_callback_trigger_on_change(self, sample_circuit):
        """Verify the callback triggers for nodes added via free-text input."""
        result = {"called": False}

        def mock_callback(circuit):
            result["called"] = True
            return "ok"

        editor = CircuitEditor(sample_circuit, output_callback=mock_callback)
        editor.node_dropdown.value = "CALLBACK_TEST_NODE"
        editor._on_add_node(None)

        assert result["called"] is True
