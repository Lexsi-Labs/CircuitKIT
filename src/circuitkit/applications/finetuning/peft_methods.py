"""
Circuit-Aware PEFT Methods

Flexible parameter-efficient fine-tuning methods that can be targeted to
specific circuit neurons identified during discovery. Includes:
- LoRA (Low-Rank Adaptation) - already exists, provided for completeness
- Adapter (bottleneck) modules
- Prefix-tuning (learnable prefix tokens)
- BitFit (bias-only tuning)
"""

import logging
from circuitkit.utils.device import get_device, empty_cache
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn

from circuitkit.artifacts import CircuitArtifact

logger = logging.getLogger(__name__)


class CircuitPEFT(ABC):
    """
    Base class for circuit-aware parameter-efficient fine-tuning.

    All PEFT methods should:
    1. Initialize with model and circuit
    2. Apply modifications to specific layers/neurons
    3. Support training on new data
    4. Allow weight merging into base model
    5. Track parameter efficiency (num_params / total_params)
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitArtifact,
        device: str = "auto",
        freeze_base: bool = True,
    ):
        """
        Initialize circuit-aware PEFT module.

        Args:
            model: Model to adapt
            circuit: CircuitArtifact identifying target neurons
            device: Device for computation
            freeze_base: Freeze base model weights (only train PEFT params)
        """
        self.model = model
        self.circuit = circuit
        self.device = device
        self.freeze_base = freeze_base

        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False

        logger.info(
            f"Initialized {self.__class__.__name__} for {circuit.model_id} "
            f"on {circuit.task} task (freeze_base={freeze_base})"
        )

    @abstractmethod
    def apply_to_model(self) -> None:
        """
        Apply PEFT modifications to the model.

        Modifies model in-place by adding trainable parameters.
        """

    @abstractmethod
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""

    @abstractmethod
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get parameter counts.

        Returns:
            {
                "peft_params": int,
                "base_params": int,
                "trainable_params": int,
                "efficiency": float (trainable / total)
            }
        """

    @abstractmethod
    def merge_weights(self, alpha: float = 1.0) -> nn.Module:
        """
        Merge PEFT weights into base model.

        Args:
            alpha: Merging coefficient (1.0 = full merge)

        Returns:
            Model with merged weights
        """

    def get_circuit_nodes(self) -> Dict[int, List[int]]:
        """Get circuit nodes organized by layer."""
        nodes_by_layer = {}
        for node in self.circuit.nodes.values():
            if node.layer_idx not in nodes_by_layer:
                nodes_by_layer[node.layer_idx] = []
            nodes_by_layer[node.layer_idx].append(node.index)
        return nodes_by_layer

    def get_circuit_layers(self) -> List[int]:
        """Get unique layers in circuit."""
        return sorted(set(n.layer_idx for n in self.circuit.nodes.values()))


class CircuitLoRA(CircuitPEFT):
    """
    Circuit-aware Low-Rank Adaptation.

    Adds low-rank weight updates specifically to circuit neurons.
    Rank is applied to attention heads and MLP neurons identified in circuit.
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitArtifact,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.1,
        device: str = "auto",
        freeze_base: bool = True,
    ):
        """
        Initialize LoRA.

        Args:
            model: Model to adapt
            circuit: CircuitArtifact with target nodes
            rank: LoRA rank
            alpha: Scaling factor for LoRA updates
            dropout: Dropout in LoRA layers
            device: Device for computation
            freeze_base: Freeze base weights
        """
        super().__init__(model, circuit, device, freeze_base)

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_modules = {}

        self.apply_to_model()

    def apply_to_model(self) -> None:
        """Apply LoRA to circuit attention heads."""
        from circuitkit.applications import get_arch_config, get_layers
        from circuitkit.applications.arch_utils import detect_model_architecture

        try:
            # Detect the architecture family from the loaded model's config
            # (model_type, e.g. "llama"), not by parsing circuit.model_id —
            # a HuggingFace repo id like "meta-llama/Llama-3.2-1B-Instruct" is
            # not a registry key and silently disables PEFT.
            family = detect_model_architecture(self.model)
            arch_cfg = get_arch_config(family)
            layers = get_layers(self.model, arch_cfg)
        except Exception as e:
            logger.warning(f"Could not auto-detect architecture: {e}")
            return

        nodes_by_layer = self.get_circuit_nodes()
        self.model.config.hidden_size

        for layer_idx in nodes_by_layer:
            if layer_idx >= len(layers):
                continue

            layer = layers[layer_idx]
            node_indices = nodes_by_layer[layer_idx]

            # Add LoRA to attention in this layer
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                if hasattr(attn, "v_proj"):
                    self._add_lora_to_projection(
                        attn.v_proj,
                        f"layer_{layer_idx}_attn_v",
                        node_indices,
                    )

    def _add_lora_to_projection(
        self,
        projection: nn.Linear,
        name: str,
        node_indices: List[int],
    ) -> None:
        """Add LoRA to a specific projection layer."""
        in_dim = projection.in_features
        out_dim = projection.out_features

        # LoRA: W_new = W_old + (A @ B) * (alpha / rank)
        # Match the base projection's dtype so the LoRA forward does not raise
        # a dtype mismatch on bf16 / fp16 models.
        dtype = projection.weight.dtype
        lora_a = nn.Linear(in_dim, self.rank, bias=False).to(device=self.device, dtype=dtype)
        lora_b = nn.Linear(self.rank, out_dim, bias=False).to(device=self.device, dtype=dtype)

        # Initialize B to zero (start with identity)
        nn.init.normal_(lora_a.weight, std=1 / self.rank)
        nn.init.zeros_(lora_b.weight)

        self.lora_modules[name] = {
            "lora_a": lora_a,
            "lora_b": lora_b,
            "nodes": node_indices,
        }

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all LoRA parameters."""
        params = []
        for module_info in self.lora_modules.values():
            params.extend(module_info["lora_a"].parameters())
            params.extend(module_info["lora_b"].parameters())
        return params

    def get_parameter_count(self) -> Dict[str, int]:
        """Count LoRA parameters."""
        peft_params = sum(p.numel() for p in self.get_trainable_params())
        sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "peft_params": peft_params,
            "base_params": total_params - peft_params,
            "trainable_params": peft_params,
            "efficiency": peft_params / total_params if total_params > 0 else 0,
        }

    def merge_weights(self, alpha: float = 1.0) -> nn.Module:
        """Merge LoRA into base model."""
        logger.info(f"Merging LoRA with alpha={alpha}")
        # Implementation would merge weights
        return self.model


class CircuitAdapterTuning(CircuitPEFT):
    """
    Circuit-aware Adapter Tuning.

    Adds bottleneck adapter modules to circuit-identified layers.
    Adapters are small feed-forward networks inserted in residual connections.
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitArtifact,
        hidden_dim: int = 64,
        device: str = "auto",
        freeze_base: bool = True,
    ):
        """
        Initialize Adapter tuning.

        Args:
            model: Model to adapt
            circuit: CircuitArtifact with target nodes
            hidden_dim: Hidden dimension of adapter bottleneck
            device: Device for computation
            freeze_base: Freeze base weights
        """
        super().__init__(model, circuit, device, freeze_base)

        self.adapter_hidden_dim = hidden_dim
        self.adapters = {}

        self.apply_to_model()

    def apply_to_model(self) -> None:
        """Apply adapters to circuit layers."""
        from circuitkit.applications import get_arch_config, get_layers
        from circuitkit.applications.arch_utils import detect_model_architecture

        try:
            # Detect the architecture family from the loaded model's config
            # (model_type, e.g. "llama"), not by parsing circuit.model_id.
            family = detect_model_architecture(self.model)
            arch_cfg = get_arch_config(family)
            layers = get_layers(self.model, arch_cfg)
        except Exception as e:
            logger.warning(f"Could not auto-detect architecture: {e}")
            return

        circuit_layers = self.get_circuit_layers()
        hidden_dim = self.model.config.hidden_size
        # Match the base model dtype so the adapter forward does not raise a
        # dtype mismatch on bf16 / fp16 models.
        dtype = next(self.model.parameters()).dtype

        for layer_idx in circuit_layers:
            if layer_idx >= len(layers):
                continue

            # Create adapter for this layer
            adapter = self._create_adapter(hidden_dim)
            self.adapters[layer_idx] = adapter
            adapter.to(device=self.device, dtype=dtype)

    def _create_adapter(self, input_dim: int) -> nn.Module:
        """
        Create bottleneck adapter module.

        Structure: input → linear_down → activation → linear_up → output
        """
        return nn.Sequential(
            nn.Linear(input_dim, self.adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.adapter_hidden_dim, input_dim),
        )

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all adapter parameters."""
        params = []
        for adapter in self.adapters.values():
            params.extend(adapter.parameters())
        return params

    def get_parameter_count(self) -> Dict[str, int]:
        """Count adapter parameters."""
        peft_params = sum(p.numel() for p in self.get_trainable_params())
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "peft_params": peft_params,
            "base_params": total_params - peft_params,
            "trainable_params": peft_params,
            "efficiency": peft_params / total_params if total_params > 0 else 0,
        }

    def merge_weights(self, alpha: float = 1.0) -> nn.Module:
        """Merge adapters into base model."""
        logger.info(f"Merging adapters with alpha={alpha}")
        # Implementation would merge weights
        return self.model


class CircuitPrefixTuning(CircuitPEFT):
    """
    Circuit-aware Prefix-Tuning.

    Prepends learnable prefix tokens to the key/value cache.
    Only affects circuit-identified layers.
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitArtifact,
        prefix_length: int = 10,
        device: str = "auto",
        freeze_base: bool = True,
    ):
        """
        Initialize Prefix-tuning.

        Args:
            model: Model to adapt
            circuit: CircuitArtifact with target nodes
            prefix_length: Length of prefix tokens
            device: Device for computation
            freeze_base: Freeze base weights
        """
        super().__init__(model, circuit, device, freeze_base)

        self.prefix_length = prefix_length
        self.prefix_embeddings = {}

        self.apply_to_model()

    def apply_to_model(self) -> None:
        """Initialize learnable prefix embeddings for circuit layers."""
        circuit_layers = self.get_circuit_layers()
        hidden_dim = self.model.config.hidden_size
        # Match the base model dtype so prefixes can be concatenated into a
        # bf16 / fp16 forward without a dtype mismatch.
        dtype = next(self.model.parameters()).dtype

        for layer_idx in circuit_layers:
            # Create learnable prefix for this layer
            # Shape: [prefix_length, hidden_dim]
            prefix = nn.Parameter(
                torch.randn(self.prefix_length, hidden_dim, device=self.device, dtype=dtype)
            )
            nn.init.normal_(prefix, std=0.1)
            self.prefix_embeddings[layer_idx] = prefix

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get all prefix parameters."""
        return list(self.prefix_embeddings.values())

    def get_parameter_count(self) -> Dict[str, int]:
        """Count prefix parameters."""
        peft_params = sum(p.numel() for p in self.get_trainable_params())
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "peft_params": peft_params,
            "base_params": total_params - peft_params,
            "trainable_params": peft_params,
            "efficiency": peft_params / total_params if total_params > 0 else 0,
        }

    def merge_weights(self, alpha: float = 1.0) -> nn.Module:
        """Merge prefixes into model cache (not possible post-hoc)."""
        logger.warning("Prefix-tuning merging requires model modification during forward pass")
        return self.model


class CircuitBitFit(CircuitPEFT):
    """
    Circuit-aware BitFit.

    Only fine-tunes bias parameters, particularly in circuit-identified layers.
    Very parameter-efficient (only biases trained).
    """

    def __init__(
        self,
        model: nn.Module,
        circuit: CircuitArtifact,
        device: str = "auto",
        freeze_base: bool = True,
    ):
        """
        Initialize BitFit.

        Args:
            model: Model to adapt
            circuit: CircuitArtifact with target nodes
            device: Device for computation
            freeze_base: Freeze base weights
        """
        super().__init__(model, circuit, device, freeze_base)

        self.trainable_biases = []

        self.apply_to_model()

    def apply_to_model(self) -> None:
        """Enable gradients only for biases in circuit layers."""
        from circuitkit.applications import get_arch_config, get_layers
        from circuitkit.applications.arch_utils import detect_model_architecture


        try:
            # Detect the architecture family from the loaded model's config
            # (model_type, e.g. "llama"), not by parsing circuit.model_id.
            family = detect_model_architecture(self.model)
            arch_cfg = get_arch_config(family)
            layers = get_layers(self.model, arch_cfg)
        except Exception as e:
            logger.warning(f"Could not auto-detect architecture: {e}")
            return

        circuit_layers = self.get_circuit_layers()

        for layer_idx in circuit_layers:
            if layer_idx >= len(layers):
                continue

            layer = layers[layer_idx]

            # Enable bias gradients
            for module in layer.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    if hasattr(module, "bias") and module.bias is not None:
                        module.bias.requires_grad = True
                        self.trainable_biases.append(module.bias)

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable bias parameters."""
        return self.trainable_biases

    def get_parameter_count(self) -> Dict[str, int]:
        """Count bias parameters."""
        peft_params = sum(p.numel() for p in self.get_trainable_params())
        total_params = sum(p.numel() for p in self.model.parameters())

        return {
            "peft_params": peft_params,
            "base_params": total_params - peft_params,
            "trainable_params": peft_params,
            "efficiency": peft_params / total_params if total_params > 0 else 0,
        }

    def merge_weights(self, alpha: float = 1.0) -> nn.Module:
        """BitFit weights are already in-place."""
        logger.info("BitFit weights already merged (in-place training)")
        return self.model


class PEFTComposer:
    """
    Compose multiple PEFT methods.

    Allows combining different PEFT approaches (e.g., LoRA + BitFit).
    Tracks total parameter efficiency and provides joint training interface.
    """

    def __init__(self):
        """Initialize PEFT composer."""
        self.methods: Dict[str, CircuitPEFT] = {}

    def add_method(self, name: str, method: CircuitPEFT) -> None:
        """Add a PEFT method."""
        self.methods[name] = method
        logger.info(f"Added {name} to composition")

    def get_total_parameters(self) -> Dict[str, int]:
        """Get total parameter counts across all methods."""
        total_peft = 0
        total_params = 0

        for method in self.methods.values():
            counts = method.get_parameter_count()
            total_peft += counts["peft_params"]
            total_params = max(total_params, counts["base_params"] + counts["peft_params"])

        return {
            "total_peft_params": total_peft,
            "total_base_params": total_params - total_peft,
            "total_params": total_params,
            "efficiency": total_peft / total_params if total_params > 0 else 0,
        }

    def get_all_trainable_params(self) -> List[nn.Parameter]:
        """Get all trainable parameters from all methods."""
        all_params = []
        for method in self.methods.values():
            all_params.extend(method.get_trainable_params())
        return all_params

    def summary(self) -> str:
        """Print summary of all composed methods."""
        lines = ["PEFT Composition Summary", "=" * 50]

        for name, method in self.methods.items():
            counts = method.get_parameter_count()
            lines.append(f"{name}:")
            lines.append(f"  PEFT params: {counts['peft_params']:,}")
            lines.append(f"  Efficiency: {counts['efficiency']:.4%}")

        total = self.get_total_parameters()
        lines.append("\nTotal:")
        lines.append(f"  PEFT params: {total['total_peft_params']:,}")
        lines.append(f"  Base params: {total['total_base_params']:,}")
        lines.append(f"  Efficiency: {total['efficiency']:.4%}")

        return "\n".join(lines)
