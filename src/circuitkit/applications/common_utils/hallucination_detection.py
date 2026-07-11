"""
Hallucination Detection Module

Monitor circuit activations during generation to detect hallucination signals.
Trains linear probes on clean/corrupted activation pairs and applies them
to flag probable hallucinations in real-time.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from circuitkit.artifacts import CircuitArtifact

from .linear_probe import LinearProbe, ProbeTrainer

logger = logging.getLogger(__name__)


class HallucinationDataset(Dataset):
    """
    Dataset for hallucination probe training.

    Requires list of examples with:
    - text: Input prompt or context
    - is_hallucination: Boolean label (True = hallucination, False = factual)
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        model: nn.Module,
        circuit: CircuitArtifact,
        arch_cfg: Dict[str, Any],
        activation_hook_fn: Optional[Callable] = None,
    ):
        """
        Initialize dataset.

        Args:
            data: List of {"text": str, "is_hallucination": bool}
            model: Model to extract activations from
            circuit: CircuitArtifact defining which activations to collect
            arch_cfg: Architecture configuration
            activation_hook_fn: Optional custom hook function for activation extraction
        """
        self.data = data
        self.model = model
        self.circuit = circuit
        self.arch_cfg = arch_cfg
        self.activation_hook_fn = activation_hook_fn

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Get a single example.

        Returns:
            (activations, label) where:
            - activations: Dict[layer_idx, Tensor] of shape [seq_len, hidden_dim]
            - label: Tensor of shape [1] (0 or 1)
        """
        item = self.data[idx]
        text = item["text"]
        label = torch.tensor([float(item.get("is_hallucination", False))])

        # Extract circuit activations
        activations = self._extract_activations(text)

        return activations, label

    def _extract_activations(
        self,
        text: str,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations for circuit nodes from text.

        Args:
            text: Input text to process

        Returns:
            Dict mapping layer_idx to activation tensors
        """
        if self.activation_hook_fn:
            return self.activation_hook_fn(self.model, text, self.circuit)

        circuit_layers = set(n.layer_idx for n in self.circuit.nodes.values())
        activations_by_layer: Dict[int, torch.Tensor] = {}
        if not circuit_layers:
            return activations_by_layer

        # TransformerLens path
        if hasattr(self.model, "run_with_cache"):
            tok = getattr(self.model, "tokenizer", None)
            if tok is None:
                return activations_by_layer
            device = next(self.model.parameters()).device
            in_ids = tok(text, return_tensors="pt", truncation=True, max_length=512)[
                "input_ids"
            ].to(device)
            target_pos = in_ids.shape[1] - 1
            with torch.inference_mode():
                _, cache = self.model.run_with_cache(in_ids)
            for layer_idx in circuit_layers:
                key = f"blocks.{layer_idx}.hook_resid_post"
                if key in cache:
                    activations_by_layer[layer_idx] = cache[key][0, target_pos].detach()
            return activations_by_layer

        # HF path: register forward hooks on transformer layers
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                activations_by_layer[layer_idx] = tensor[0, -1].detach()

            return hook_fn

        try:
            transformer = getattr(self.model, "transformer", getattr(self.model, "model", None))
            layers_list = None
            if transformer is not None:
                layers_list = getattr(transformer, "h", getattr(transformer, "layers", None))
            if layers_list is None:
                return activations_by_layer
            for layer_idx in circuit_layers:
                if 0 <= layer_idx < len(layers_list):
                    h = layers_list[layer_idx].register_forward_hook(make_hook(layer_idx))
                    hooks.append(h)
            tok = getattr(self.model, "tokenizer", None)
            if tok is None:
                return activations_by_layer
            device = next(self.model.parameters()).device
            in_ids = tok(text, return_tensors="pt", truncation=True, max_length=512)[
                "input_ids"
            ].to(device)
            with torch.inference_mode():
                self.model(in_ids)
        finally:
            for h in hooks:
                h.remove()
        return activations_by_layer


class HallucinationDetector:
    """
    Detect hallucinations by monitoring circuit activations.

    Trains linear probes on circuit activations to distinguish between
    factual and hallucinated generations. Can be applied during model
    generation to flag probable hallucinations in real-time.

    Attributes:
        model: The language model to monitor
        circuit: CircuitArtifact defining the circuit
        arch_cfg: Architecture configuration
        probes: Trained linear probes per layer
        device: Compute device
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitArtifact,
        arch_cfg: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize hallucination detector.

        Args:
            model: HuggingFace model instance
            circuit: CircuitArtifact to monitor
            arch_cfg: Architecture configuration
            device: Device for computation ("cuda" or "cpu")
        """
        self.model = model
        self.circuit = circuit
        self.arch_cfg = arch_cfg
        self.device = device

        # Probes per layer
        self.probes: Dict[int, LinearProbe] = {}

        # Activation hooks
        self.activation_hooks = {}
        self.activations_cache = {}

        logger.info(
            f"Initialized HallucinationDetector for {circuit.model_id} " f"on {circuit.task} task"
        )

    def train_probes(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        patience: int = 3,
    ) -> Dict[str, Any]:
        """
        Train linear probes on circuit activations.

        Args:
            train_data: Training examples with {"text": str, "is_hallucination": bool}
            val_data: Validation examples (same format)
            batch_size: Batch size for training
            epochs: Maximum training epochs
            learning_rate: Adam learning rate
            patience: Early stopping patience

        Returns:
            Dictionary with training results:
            {
                "best_val_auroc": float,   # max validation AUROC across layer probes
                "num_probes": int,
                "probes": Dict[int, LinearProbe],
                "circuit_layers": List[int],
            }
        """
        logger.info(
            f"Training probes on {len(train_data)} examples "
            f"with {len(val_data)} validation examples"
        )

        # Create datasets
        train_dataset = HallucinationDataset(train_data, self.model, self.circuit, self.arch_cfg)
        val_dataset = HallucinationDataset(val_data, self.model, self.circuit, self.arch_cfg)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # Initialize probes for layers in circuit
        circuit_layers = set(n.layer_idx for n in self.circuit.nodes.values())
        hidden_dim = self.model.config.hidden_size

        per_layer_val_auroc = []
        for layer_idx in circuit_layers:
            probe = LinearProbe(hidden_dim).to(self.device)
            trainer = ProbeTrainer(probe, device=self.device, learning_rate=learning_rate)

            # Train this layer's probe
            trainer.train(
                train_loader,
                val_loader,
                epochs=epochs,
                patience=patience,
                verbose=True,
            )

            self.probes[layer_idx] = trainer.get_probe()
            # Capture the real validation AUROC from the trainer that actually
            # ran the training loop.
            per_layer_val_auroc.append(trainer.best_val_auroc)

            logger.info(
                f"Layer {layer_idx} probe trained: " f"best_val_auroc={trainer.best_val_auroc:.4f}"
            )

        # Return summary. best_val_auroc is the max across layer probes, taken
        # from the trainers that actually trained — NOT from freshly
        # constructed ProbeTrainer(p) wrappers, whose best_val_auroc would be
        # the untrained sentinel (-1.0) and silently mask the real score.
        return {
            "best_val_auroc": max(per_layer_val_auroc, default=0.0),
            "num_probes": len(self.probes),
            "probes": self.probes,
            "circuit_layers": list(circuit_layers),
        }

    def detect_hallucinations(
        self,
        text: str,
        generate_fn: Optional[Callable] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in generated text.

        Monitors circuit activations during generation and flags tokens
        with high hallucination probability.

        Args:
            text: Input prompt
            generate_fn: Optional generation function
                         If provided, generates from text with monitoring
                         If None, analyzes provided text as-is
            threshold: Probability threshold for flagging hallucinations

        Returns:
            Dictionary with detection results:
            {
                "text": str,
                "is_hallucination": bool,
                "hallucination_prob": float,
                "per_token_probs": List[float],
                "flagged_tokens": List[int],
                "circuit_activations": Dict[int, Tensor],
                "explanation": str,
            }
        """
        self.model.eval()

        if not self.probes:
            logger.warning("No trained probes available. Run train_probes() first.")
            return {
                "text": text,
                "is_hallucination": False,
                "hallucination_prob": 0.0,
                "per_token_probs": [],
                "flagged_tokens": [],
                "error": "No trained probes",
            }

        # If generation function provided, generate with monitoring
        if generate_fn:
            try:
                # Generate from the model
                # This would hook into the generation process to collect activations
                generate_fn(text)
            except Exception as e:
                logger.error(f"Generation failed: {e}")

        # Extract activations (mock implementation)
        # In real use, this would hook into model forward pass
        activations = self._get_circuit_activations(text)

        # Run probes on activations
        per_token_probs = []
        flagged_tokens = []

        for layer_idx, probe in self.probes.items():
            if layer_idx in activations:
                with torch.no_grad():
                    layer_acts = activations[layer_idx].to(self.device)
                    # Ensure 2D input: [seq_len, hidden] or [1, hidden]
                    if layer_acts.dim() == 1:
                        layer_acts = layer_acts.unsqueeze(0)
                    probs = probe(layer_acts)  # [seq_len, 1]

                    per_token_probs.extend(probs.squeeze(-1).cpu().numpy().tolist())

                    # Flag tokens above threshold
                    flagged = (probs.squeeze(-1) > threshold).nonzero(as_tuple=True)[0]
                    flagged_tokens.extend(flagged.cpu().numpy().tolist())

        # Aggregate results
        avg_prob = sum(per_token_probs) / max(len(per_token_probs), 1)
        is_hallucination = avg_prob > threshold

        # Create explanation
        if is_hallucination:
            explanation = (
                f"High hallucination probability ({avg_prob:.2%}). "
                f"Flagged {len(flagged_tokens)} tokens with p > {threshold}."
            )
        else:
            explanation = f"Low hallucination probability ({avg_prob:.2%}). Likely factual."

        return {
            "text": text,
            "is_hallucination": is_hallucination,
            "hallucination_prob": float(avg_prob),
            "per_token_probs": per_token_probs,
            "flagged_tokens": flagged_tokens,
            "circuit_activations": activations,
            "explanation": explanation,
            "num_probes": len(self.probes),
        }

    def get_activation_profile(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Extract circuit activations for analysis.

        Args:
            text: Input text

        Returns:
            Dict mapping layer_idx to activation tensors
        """
        return self._get_circuit_activations(text)

    def _get_circuit_activations(
        self,
        text: str,
    ) -> Dict[int, torch.Tensor]:
        """Extract activations for circuit nodes via TransformerLens cache.

        Supports HookedTransformer (preferred) and falls back to HF forward hooks.
        Returns a dict mapping layer_idx to the mean residual-stream representation
        at that layer's circuit nodes, averaged over the sequence's last position.
        Shape: {layer_idx: Tensor[d_model]} for each layer in the circuit.
        """
        circuit_layers = set(n.layer_idx for n in self.circuit.nodes.values())
        if not circuit_layers:
            return {}

        # --- TransformerLens path (preferred) ---
        if hasattr(self.model, "run_with_cache"):
            tok = getattr(self.model, "tokenizer", None)
            if tok is None:
                return {}
            in_ids = tok(text, return_tensors="pt", truncation=True, max_length=512)[
                "input_ids"
            ].to(self.device)
            target_pos = in_ids.shape[1] - 1
            with torch.inference_mode():
                _, cache = self.model.run_with_cache(in_ids)
            activations = {}
            for layer_idx in circuit_layers:
                key = f"blocks.{layer_idx}.hook_resid_post"
                if key in cache:
                    activations[layer_idx] = cache[key][0, target_pos].detach()
            return activations

        # --- HuggingFace model path (register forward hooks) ---
        activations: Dict[int, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                activations[layer_idx] = tensor[0, -1].detach()

            return hook_fn

        try:
            transformer = getattr(self.model, "transformer", getattr(self.model, "model", None))
            if transformer is None:
                return activations
            layers = getattr(transformer, "h", getattr(transformer, "layers", None))
            if layers is None:
                return activations

            for layer_idx in circuit_layers:
                if 0 <= layer_idx < len(layers):
                    h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
                    hooks.append(h)

            tok = getattr(self.model, "tokenizer", getattr(self.model, "_tokenizer", None))
            if tok is None:
                return activations
            in_ids = tok(text, return_tensors="pt", truncation=True, max_length=512)[
                "input_ids"
            ].to(self.device)
            with torch.inference_mode():
                self.model(in_ids)
        finally:
            for h in hooks:
                h.remove()
        return activations

    def get_model_probes(self) -> Dict[int, LinearProbe]:
        """Get trained probes."""
        return self.probes

    def get_probe_stats(self) -> Dict[str, Any]:
        """Get statistics about trained probes."""
        return {
            "num_probes": len(self.probes),
            "trained_layers": list(self.probes.keys()),
            "device": self.device,
        }

    def save_probes(self, path: str) -> None:
        """Save trained probes to file."""
        state_dict = {layer_idx: probe.state_dict() for layer_idx, probe in self.probes.items()}
        torch.save(state_dict, path)
        logger.info(f"Saved {len(self.probes)} probes to {path}")

    def load_probes(self, path: str) -> None:
        """Load trained probes from file."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)

        for layer_idx, weights in state_dict.items():
            probe = LinearProbe(self.model.config.hidden_size).to(self.device)
            probe.load_state_dict(weights)
            self.probes[layer_idx] = probe

        logger.info(f"Loaded {len(self.probes)} probes from {path}")
