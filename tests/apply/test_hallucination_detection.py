"""
Tests for hallucination detection module.

Tests LinearProbe training, HallucinationDetector initialization,
probe training, and hallucination detection on mock data.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from circuitkit.applications.common_utils.hallucination_detection import (
    HallucinationDataset,
    HallucinationDetector,
)
from circuitkit.applications.common_utils.linear_probe import LinearProbe, ProbeTrainer
from circuitkit.artifacts import CircuitArtifact, Node, NodeType

# Fixtures


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""

    class MockConfig:
        hidden_size = 128
        num_attention_heads = 4
        num_layers = 2
        intermediate_size = 512

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Linear(128, 128)
            self.mlp = nn.Linear(128, 128)

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.layers = nn.ModuleList([MockLayer() for _ in range(2)])

        def forward(self, x):
            return torch.randn(1, 128)

    return MockModel()


@pytest.fixture
def mock_activation_hook():
    """Custom activation hook that returns synthetic activations for circuit layers."""

    def hook_fn(model, text, circuit):
        circuit_layers = set(n.layer_idx for n in circuit.nodes.values())
        hidden_size = model.config.hidden_size
        return {layer_idx: torch.randn(hidden_size) for layer_idx in circuit_layers}

    return hook_fn


@pytest.fixture
def mock_circuit():
    """Create a mock circuit artifact."""
    artifact = CircuitArtifact(
        model_id="gpt2",
        discovery_method="eap",
        task="test",
        dataset="test",
    )

    # Add nodes
    artifact.add_node("L0H0", Node(0, NodeType.ATTENTION_HEAD, 0, 0.9))
    artifact.add_node("L0H1", Node(0, NodeType.ATTENTION_HEAD, 1, 0.7))
    artifact.add_node("L1H0", Node(1, NodeType.ATTENTION_HEAD, 0, 0.8))

    return artifact


@pytest.fixture
def mock_arch_cfg():
    """Mock architecture config."""
    return {
        "name": "gpt2",
        "layers_path": ["h"],
        "attn": {"module": "attn"},
        "mlp": {},
    }


@pytest.fixture
def mock_training_data():
    """Create mock training data."""
    return [
        {"text": "Paris is the capital of France", "is_hallucination": False},
        {"text": "Paris is the capital of Germany", "is_hallucination": True},
        {"text": "2 + 2 equals 4", "is_hallucination": False},
        {"text": "2 + 2 equals 5", "is_hallucination": True},
        {"text": "The Earth orbits the Sun", "is_hallucination": False},
        {"text": "The Sun orbits the Earth", "is_hallucination": True},
    ]


@pytest.fixture
def mock_val_data():
    """Create mock validation data."""
    return [
        {"text": "Einstein discovered relativity", "is_hallucination": False},
        {"text": "Einstein discovered magic", "is_hallucination": True},
        {"text": "Water boils at 100 C", "is_hallucination": False},
    ]


# Test LinearProbe


class TestLinearProbe:
    """Test LinearProbe class."""

    def test_probe_creation(self):
        """Test creating a linear probe."""
        probe = LinearProbe(input_dim=256)
        assert probe.input_dim == 256
        assert isinstance(probe.linear, nn.Linear)

    def test_probe_with_dropout(self):
        """Test probe with dropout."""
        probe = LinearProbe(input_dim=128, dropout=0.5)
        assert probe.dropout is not None
        assert isinstance(probe.dropout, nn.Dropout)

    def test_probe_forward_shape(self):
        """Test probe forward pass output shape."""
        probe = LinearProbe(input_dim=256)

        # Input: batch_size=4, seq_len=10, hidden=256
        x = torch.randn(4, 10, 256)
        output = probe(x)

        # Output should be [..., 1] with values in [0, 1]
        assert output.shape == (4, 10, 1)
        assert (output >= 0).all() and (output <= 1).all()

    def test_probe_forward_1d_input(self):
        """Test probe with 1D input."""
        probe = LinearProbe(input_dim=128)
        x = torch.randn(128)

        output = probe(x)
        assert output.shape == (1,)
        assert 0 <= output <= 1

    def test_probe_get_logits(self):
        """Test getting raw logits."""
        probe = LinearProbe(input_dim=256)
        x = torch.randn(4, 256)

        logits = probe.get_logits(x)
        assert logits.shape == (4, 1)
        # Logits are unbounded (not sigmoid)
        assert logits.min() < 0 or logits.max() > 1

    def test_probe_eval_mode(self):
        """Test that probe works in eval mode."""
        probe = LinearProbe(input_dim=128, dropout=0.5)
        probe.eval()

        x = torch.randn(8, 128)
        output = probe(x)

        assert output.shape == (8, 1)
        # In eval mode, dropout disabled
        assert (output >= 0).all() and (output <= 1).all()


class TestProbeTrainer:
    """Test ProbeTrainer class."""

    def test_trainer_creation(self):
        """Test creating a probe trainer."""
        probe = LinearProbe(input_dim=128)
        trainer = ProbeTrainer(probe, device="cpu", learning_rate=1e-3)

        assert trainer.probe is probe
        assert trainer.device == "cpu"
        assert trainer.best_val_auroc == -1.0

    def test_trainer_on_gpu_if_available(self):
        """Test trainer on GPU if available."""
        probe = LinearProbe(input_dim=128)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = ProbeTrainer(probe, device=device)

        assert trainer.device == device

    def test_trainer_get_probe(self):
        """Test getting probe from trainer."""
        probe = LinearProbe(input_dim=128)
        trainer = ProbeTrainer(probe)

        retrieved = trainer.get_probe()
        assert retrieved is probe

    def test_trainer_get_metrics(self):
        """Test getting trainer metrics."""
        probe = LinearProbe(input_dim=128)
        trainer = ProbeTrainer(probe)

        metrics = trainer.get_metrics()
        assert "best_val_auroc" in metrics
        assert metrics["best_val_auroc"] == -1.0


class TestHallucinationDataset:
    """Test HallucinationDataset class."""

    def test_dataset_creation(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test creating a hallucination dataset."""
        data = [{"text": "test", "is_hallucination": False}]
        dataset = HallucinationDataset(data, mock_model, mock_circuit, mock_arch_cfg)

        assert len(dataset) == 1

    def test_dataset_getitem(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test getting item from dataset."""
        data = [
            {"text": "test1", "is_hallucination": False},
            {"text": "test2", "is_hallucination": True},
        ]
        dataset = HallucinationDataset(data, mock_model, mock_circuit, mock_arch_cfg)

        activations, label = dataset[0]
        assert isinstance(activations, dict)
        assert label.shape == (1,)
        assert label.item() == 0.0  # is_hallucination=False

        activations, label = dataset[1]
        assert label.item() == 1.0  # is_hallucination=True

    def test_dataset_missing_label(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test dataset with missing label defaults to False."""
        data = [{"text": "test"}]  # No is_hallucination key
        dataset = HallucinationDataset(data, mock_model, mock_circuit, mock_arch_cfg)

        _, label = dataset[0]
        assert label.item() == 0.0


class TestHallucinationDetector:
    """Test HallucinationDetector class."""

    def test_detector_creation(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test creating detector."""
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")

        assert detector.model is mock_model
        assert detector.circuit is mock_circuit
        assert len(detector.probes) == 0

    def test_detector_device(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test detector device handling."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device=device)

        assert detector.device == device

    def test_detector_without_probes(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test detection without trained probes."""
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")

        result = detector.detect_hallucinations("test text")

        assert result["text"] == "test text"
        assert result["hallucination_prob"] == 0.0
        assert "error" in result

    def test_detector_get_probes(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test getting probes from detector."""
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")

        probes = detector.get_model_probes()
        assert isinstance(probes, dict)
        assert len(probes) == 0

    def test_detector_get_stats(self, mock_model, mock_circuit, mock_arch_cfg):
        """Test getting detector statistics."""
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")

        stats = detector.get_probe_stats()
        assert "num_probes" in stats
        assert "trained_layers" in stats
        assert stats["num_probes"] == 0

    def test_detector_get_activations(
        self, mock_model, mock_circuit, mock_arch_cfg, mock_activation_hook
    ):
        """Test getting activation profile."""
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")
        # Override the internal activation extraction with our mock hook
        detector._get_circuit_activations = lambda text: mock_activation_hook(
            mock_model, text, mock_circuit
        )

        activations = detector.get_activation_profile("test text")

        # Should return dict with layer indices as keys
        assert isinstance(activations, dict)
        # Should have entries for circuit layers
        circuit_layers = set(n.layer_idx for n in mock_circuit.nodes.values())
        for layer_idx in circuit_layers:
            assert layer_idx in activations
            assert isinstance(activations[layer_idx], torch.Tensor)


class TestProbeTraining:
    """Test probe training functionality."""

    def test_train_epoch_structure(self):
        """Test that train_epoch returns expected structure."""
        from torch.utils.data import DataLoader, TensorDataset

        probe = LinearProbe(input_dim=128)
        trainer = ProbeTrainer(probe, device="cpu")

        # Create dummy dataloader
        X = torch.randn(16, 128)
        y = torch.randint(0, 2, (16, 1)).float()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)

        train_loss, train_auroc, val_loss, val_auroc = trainer.train_epoch(loader, loader)

        assert isinstance(train_loss, float)
        assert isinstance(train_auroc, float)
        assert isinstance(val_loss, float)
        assert isinstance(val_auroc, float)

    def test_full_training_loop(self):
        """Test complete training loop."""
        from torch.utils.data import DataLoader, TensorDataset

        probe = LinearProbe(input_dim=128)
        trainer = ProbeTrainer(probe, device="cpu", learning_rate=1e-2)

        # Create training data with clear separation
        X_train = torch.cat(
            [
                torch.randn(8, 128) - 1,  # Class 0
                torch.randn(8, 128) + 1,  # Class 1
            ]
        )
        y_train = torch.cat(
            [
                torch.zeros(8, 1),
                torch.ones(8, 1),
            ]
        )

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=4)

        history = trainer.train(
            loader,
            loader,
            epochs=3,
            patience=10,
            verbose=False,
        )

        assert "train_loss" in history
        assert "val_auroc" in history
        assert len(history["train_loss"]) > 0


class TestDetectorTraining:
    """Test HallucinationDetector probe training."""

    def _make_detector_with_hook(
        self, mock_model, mock_circuit, mock_arch_cfg, mock_activation_hook
    ):
        """Create a detector whose activation extraction uses the mock hook."""
        detector = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")
        original_get = detector._get_circuit_activations

        def hooked_get(text):
            acts = original_get(text)
            if acts:
                return acts
            return mock_activation_hook(mock_model, text, mock_circuit)

        detector._get_circuit_activations = hooked_get
        return detector

    def test_detector_train_probes_with_mock_data(
        self,
        mock_model,
        mock_circuit,
        mock_arch_cfg,
        mock_training_data,
        mock_val_data,
        mock_activation_hook,
    ):
        """Test training detector probes."""
        detector = self._make_detector_with_hook(
            mock_model, mock_circuit, mock_arch_cfg, mock_activation_hook
        )

        # Train probes
        result = detector.train_probes(
            mock_training_data,
            mock_val_data,
            batch_size=2,
            epochs=2,
            patience=5,
        )

        assert "num_probes" in result
        assert "probes" in result
        assert "circuit_layers" in result
        # Should have trained probes for circuit layers
        assert len(result["probes"]) > 0

    def test_detector_detection_after_training(
        self,
        mock_model,
        mock_circuit,
        mock_arch_cfg,
        mock_training_data,
        mock_val_data,
        mock_activation_hook,
    ):
        """Test detection after training probes."""
        detector = self._make_detector_with_hook(
            mock_model, mock_circuit, mock_arch_cfg, mock_activation_hook
        )

        # Train probes
        detector.train_probes(
            mock_training_data,
            mock_val_data,
            batch_size=2,
            epochs=2,
            patience=5,
        )

        # Now detect
        result = detector.detect_hallucinations("test text")

        assert result["text"] == "test text"
        assert "hallucination_prob" in result
        assert "per_token_probs" in result
        assert 0 <= result["hallucination_prob"] <= 1

    def test_detector_save_load_probes(
        self,
        mock_model,
        mock_circuit,
        mock_arch_cfg,
        mock_training_data,
        mock_val_data,
        mock_activation_hook,
    ):
        """Test saving and loading probes."""
        detector1 = self._make_detector_with_hook(
            mock_model, mock_circuit, mock_arch_cfg, mock_activation_hook
        )

        # Train
        detector1.train_probes(
            mock_training_data,
            mock_val_data,
            batch_size=2,
            epochs=2,
            patience=5,
        )

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probes.pt"
            detector1.save_probes(str(path))

            # Create new detector and load
            detector2 = HallucinationDetector(mock_model, mock_circuit, mock_arch_cfg, device="cpu")
            detector2.load_probes(str(path))

            # Check probes loaded
            assert len(detector2.get_model_probes()) == len(detector1.get_model_probes())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
