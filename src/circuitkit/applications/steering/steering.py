# FILE: circuitkit/applications/steering/steering.py
"""
Activation Steering: Steer model behavior via activation patching.

This module implements ActivationSteering, which modifies activations at
circuit-relevant nodes to steer model behavior toward desired outputs without
fine-tuning weights.

The steering vector is computed as the difference between activations in
target and source distributions, then applied via activation patching during inference.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import logging

logger = logging.getLogger(__name__)

import torch
from transformer_lens import HookedTransformer


class ActivationSteering:
    """
    Steer model behavior via activation patching.

    This class learns steering vectors from source/target example pairs and
    applies them to modify activations at circuit-relevant nodes.

    Attributes:
        model: The transformer model (HookedTransformer)
        circuit_scores: Dict mapping node names to importance scores
        steering_vectors: Dict mapping node names to steering vectors
        steering_hooks: List of (hook_point, hook_fn) tuples for patching
    """

    def __init__(
        self,
        model: HookedTransformer,
        circuit_scores: Dict[str, float],
        score_threshold: float = 0.0,
    ):
        """
        Initialize ActivationSteering.

        Args:
            model: HookedTransformer model
            circuit_scores: Dict mapping node names (e.g., "A0.0", "MLP 1") to importance scores.
                           High-score nodes will have steering applied.
            score_threshold: Only compute steering for nodes with score >= threshold.

        Examples:
            >>> circuit_scores = {"A0.0": 0.95, "A0.1": 0.92, "MLP 1": 0.88}
            >>> steering = ActivationSteering(model, circuit_scores)
        """
        self.model = model
        self.circuit_scores = circuit_scores
        self.score_threshold = score_threshold

        # Steering hooks attach to attn.hook_result, which only exists when the
        # model is configured to materialise per-head results.
        if hasattr(model, "set_use_attn_result"):
            model.set_use_attn_result(True)
        elif hasattr(model, "cfg") and hasattr(model.cfg, "use_attn_result"):
            model.cfg.use_attn_result = True

        # Filter high-score nodes
        self.high_score_nodes = {
            name: score for name, score in circuit_scores.items() if score >= score_threshold
        }

        logger.info(
            f"Initializing ActivationSteering with {len(self.high_score_nodes)} "
            f"circuit-relevant nodes (threshold: {score_threshold})"
        )

        # Storage for steering vectors and related metadata
        self.steering_vectors = {}
        self.steering_metadata = {}
        self.steering_hooks = []

    def _get_hook_point_from_node(self, node_name: str) -> Optional[str]:
        """
        Convert circuit node name to transformer_lens hook point.

        Args:
            node_name: Node name like "A0.0" (attention layer 0, head 0) or
                      "MLP 1" (MLP layer 1)

        Returns:
            Hook point string, e.g., "blocks.0.attn.hook_result"
        """
        import re

        # Attention head node: "A<layer>.<head>"
        attn_match = re.match(r"A(\d+)\.(\d+)", node_name)
        if attn_match:
            layer_idx = int(attn_match.group(1))
            # Hook to the attention result (after all heads are combined)
            return f"blocks.{layer_idx}.attn.hook_result"

        # MLP node: "MLP <layer>"
        mlp_match = re.match(r"MLP\s+(\d+)", node_name)
        if mlp_match:
            layer_idx = int(mlp_match.group(1))
            # Hook to the MLP output
            return f"blocks.{layer_idx}.hook_mlp_out"

        return None

    @staticmethod
    def _node_head_index(node_name: str) -> Optional[int]:
        """Return the head index for an attention node name ("A<layer>.<head>")."""
        import re

        attn_match = re.match(r"A(\d+)\.(\d+)", node_name)
        if attn_match:
            return int(attn_match.group(2))
        return None

    def compute_steering_vector(
        self,
        source_examples: List[Dict],
        target_examples: List[Dict],
        batch_size: int = 32,
        return_all_positions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute steering vectors as mean(target_activations) - mean(source_activations).

        The steering vector represents the direction to push activations to steer
        the model toward target behavior.

        Args:
            source_examples: List of source examples. Each should be a dict with:
                            - 'text': str, the input text
                            OR
                            - 'input_ids': torch.Tensor, tokenized input
            target_examples: List of target examples with same format as source
            batch_size: Batch size for computing activations
            return_all_positions: If True, return vectors for all sequence positions.
                                 If False, return only the mean across positions.

        Returns:
            Dict mapping node names to steering vectors:
            {
                "A0.0": torch.Tensor of shape [d_model] or [seq_len, d_model],
                "A0.1": torch.Tensor,
                ...
            }

        Examples:
            >>> source = [{'text': 'The subject is'}, {'text': 'The object is'}]
            >>> target = [{'text': 'The object is'}, {'text': 'The subject is'}]
            >>> vectors = steering.compute_steering_vector(source, target)
        """
        logger.info(
            f"\nComputing steering vectors from {len(source_examples)} source "
            f"and {len(target_examples)} target examples..."
        )

        self.model.cfg.device

        # Collect activations from source distribution
        source_acts = self._collect_activations(
            source_examples, batch_size=batch_size, return_all_positions=return_all_positions
        )

        # Collect activations from target distribution
        target_acts = self._collect_activations(
            target_examples, batch_size=batch_size, return_all_positions=return_all_positions
        )

        # Compute steering vectors
        steering_vectors = {}
        for node_name in self.high_score_nodes:
            if node_name in source_acts and node_name in target_acts:
                source_activation = source_acts[
                    node_name
                ]  # [n_examples, d_model] or [n_examples, seq_len, d_model]
                target_activation = target_acts[node_name]

                # Compute mean across examples
                source_mean = source_activation.mean(dim=0)
                target_mean = target_activation.mean(dim=0)

                # Steering vector is the difference
                steering_vector = target_mean - source_mean
                steering_vectors[node_name] = steering_vector

                # Store metadata
                self.steering_metadata[node_name] = {
                    "source_mean": source_mean,
                    "target_mean": target_mean,
                    "shape": steering_vector.shape,
                    "norm": steering_vector.norm().item(),
                }

                logger.info(f"  {node_name}: steering_norm={steering_vectors[node_name].norm():.4f}")

        self.steering_vectors = steering_vectors
        logger.info(f"Computed steering vectors for {len(steering_vectors)} nodes\n")
        return steering_vectors

    def _collect_activations(
        self, examples: List[Dict], batch_size: int = 32, return_all_positions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activations for given examples at circuit nodes.

        Args:
            examples: List of example dicts with 'text' or 'input_ids'
            batch_size: Batch size for processing
            return_all_positions: If True, keep position dimension; else average

        Returns:
            Dict mapping node names to stacked activations
        """
        device = self.model.cfg.device
        collected_activations = {node: [] for node in self.high_score_nodes}

        # Convert examples to input_ids if needed
        input_ids_list = []
        for example in examples:
            if "input_ids" in example:
                input_ids = example["input_ids"]
                if isinstance(input_ids, list):
                    input_ids = torch.tensor(input_ids)
            elif "text" in example:
                # Simple tokenization (requires model to have tokenizer)
                text = example["text"]
                if hasattr(self.model, "tokenizer"):
                    tokens = self.model.tokenizer.encode(text, return_tensors="pt").squeeze(0)
                else:
                    # Fallback: try basic string tokenization
                    tokens = torch.tensor([ord(c) for c in text[:256]])
                input_ids = tokens
            else:
                raise ValueError("Example must have 'text' or 'input_ids' key")

            input_ids_list.append(input_ids)

        # Process in batches
        for batch_start in range(0, len(input_ids_list), batch_size):
            batch_end = min(batch_start + batch_size, len(input_ids_list))
            batch_input_ids = input_ids_list[batch_start:batch_end]

            # Pad batch
            max_len = max(ids.shape[0] for ids in batch_input_ids)
            padded_batch = []
            for ids in batch_input_ids:
                if ids.shape[0] < max_len:
                    ids = torch.cat([ids, torch.zeros(max_len - ids.shape[0], dtype=ids.dtype)])
                padded_batch.append(ids)

            batch_tensor = torch.stack(padded_batch).to(device)

            # Collect activations for this batch
            batch_acts = self._collect_batch_activations(batch_tensor)

            # Aggregate
            for node_name in self.high_score_nodes:
                if node_name in batch_acts:
                    acts = batch_acts[node_name]  # [batch, seq_len, d_model] or [batch, d_model]

                    # Average over sequence positions if needed
                    if not return_all_positions and len(acts.shape) == 3:
                        acts = acts.mean(dim=1)  # [batch, d_model]

                    collected_activations[node_name].append(acts)

        # Stack all batches
        result = {}
        for node_name in self.high_score_nodes:
            if collected_activations[node_name]:
                stacked = torch.cat(collected_activations[node_name], dim=0)
                result[node_name] = stacked

        return result

    def _collect_batch_activations(self, batch_input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Collect activations for a batch using hooks.

        Args:
            batch_input_ids: [batch_size, seq_len] tensor of input IDs

        Returns:
            Dict mapping node names to activations [batch_size, seq_len, d_model] or [batch_size, d_model]
        """
        collected = {}

        def make_hook_fn(node_name):
            def hook_fn(activation, hook):
                # `blocks.{l}.attn.hook_result` is 4-D
                # [batch, seq, head, d_model]; reduce it to the single head
                # this node refers to so every stored activation is uniformly
                # [batch, seq, d_model]. MLP hooks are already 3-D.
                act = activation.detach()
                if act.dim() == 4:
                    head_idx = self._node_head_index(node_name)
                    if head_idx is not None and head_idx < act.shape[2]:
                        act = act[:, :, head_idx, :]
                    else:
                        # Fallback: collapse the head dimension.
                        act = act.mean(dim=2)
                collected[node_name] = act.cpu()

            return hook_fn

        # Install hooks
        hooks_to_remove = []
        for node_name in self.high_score_nodes:
            hook_point = self._get_hook_point_from_node(node_name)
            if hook_point is not None:
                hook_fn = make_hook_fn(node_name)
                # TransformerLens uses add_hook(name, fn, dir) rather than
                # the stock nn.Module.register_forward_hook(fn). Returns
                # nothing here, so we re-find the handle via the hook list.
                self.model.add_hook(hook_point, hook_fn, dir="fwd")
                handle = (hook_point, hook_fn)
                hooks_to_remove.append(handle)

        # Forward pass
        with torch.no_grad():
            _ = self.model(batch_input_ids)

        # Remove hooks (TL: reset_hooks clears all that were added).
        self.model.reset_hooks()

        return collected

    def steer(
        self,
        inputs: Union[str, torch.Tensor, Dict],
        steering_vectors: Optional[Dict[str, torch.Tensor]] = None,
        coefficient: float = 1.0,
        layer_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with activation steering applied.

        Modifies activations at circuit nodes by adding scaled steering vectors.

        Args:
            inputs: Input to the model. Can be:
                   - str: text to tokenize
                   - torch.Tensor: input_ids
                   - dict: with 'text' or 'input_ids' key
            steering_vectors: Dict of steering vectors to apply. If None, uses
                             self.steering_vectors (must be computed first).
            coefficient: Scaling factor for steering (0.0 = no steering, 1.0 = full).
                        Can be a float applied to all nodes, or a dict mapping
                        node names to individual coefficients.
            layer_weights: Optional dict mapping node names to weights for combining
                          multiple steering vectors. If None, all vectors weighted equally.

        Returns:
            Dict with:
                - 'output': Model output logits
                - 'output_probs': Softmax probabilities
                - 'steered_nodes': List of nodes where steering was applied

        Examples:
            >>> vectors = steering.compute_steering_vector(source_examples, target_examples)
            >>> output = steering.steer("test input", coefficient=1.0)
            >>> # With variable steering strength
            >>> output = steering.steer("test input", coefficient=0.5)
        """
        if steering_vectors is None:
            steering_vectors = self.steering_vectors
            if not steering_vectors:
                raise ValueError(
                    "No steering vectors available. Call compute_steering_vector first."
                )

        # Prepare input
        if isinstance(inputs, str):
            text = inputs
            if hasattr(self.model, "tokenizer"):
                input_ids = self.model.tokenizer.encode(text, return_tensors="pt").to(
                    self.model.cfg.device
                )
            else:
                input_ids = torch.tensor([[ord(c) for c in text[:256]]]).to(self.model.cfg.device)
        elif isinstance(inputs, torch.Tensor):
            input_ids = inputs.to(self.model.cfg.device)
        elif isinstance(inputs, dict):
            if "input_ids" in inputs:
                input_ids = inputs["input_ids"].to(self.model.cfg.device)
            elif "text" in inputs:
                return self.steer(inputs["text"], steering_vectors, coefficient, layer_weights)
            else:
                raise ValueError("Dict input must have 'text' or 'input_ids' key")
        else:
            raise TypeError(f"inputs must be str, Tensor, or dict, got {type(inputs)}")

        # Ensure input_ids has batch dimension
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        device = self.model.cfg.device

        # Setup steering hooks
        steered_nodes = []
        hooks_to_remove = []

        def make_steering_hook(node_name, steering_vector):
            head_idx = self._node_head_index(node_name)

            def hook_fn(activation, hook):
                # Apply steering: activation += coefficient * steering_vector
                coeff = coefficient
                if isinstance(coefficient, dict):
                    coeff = coefficient.get(node_name, 1.0)

                steering_vector_device = steering_vector.to(device)

                # Attention `hook_result` is 4-D [batch, seq, head, d_model];
                # steer only the head this node refers to. MLP `hook_mlp_out`
                # is 3-D [batch, seq, d_model] and is steered as a whole.
                if (
                    activation.dim() == 4
                    and head_idx is not None
                    and head_idx < activation.shape[2]
                ):
                    target = activation[:, :, head_idx, :]
                else:
                    target = activation

                # Broadcast a per-d_model or per-(seq,d_model) vector by
                # prepending leading dims until ranks match.
                while len(steering_vector_device.shape) < len(target.shape):
                    steering_vector_device = steering_vector_device.unsqueeze(0)

                target.add_(steering_vector_device * coeff)
                return activation

            return hook_fn

        # Install steering hooks
        for node_name, steering_vector in steering_vectors.items():
            hook_point = self._get_hook_point_from_node(node_name)
            if hook_point is not None:
                hook_fn = make_steering_hook(node_name, steering_vector)
                # TransformerLens uses add_hook(name, fn, dir) rather than
                # the stock nn.Module.register_forward_hook(fn). Returns
                # nothing here, so we re-find the handle via the hook list.
                self.model.add_hook(hook_point, hook_fn, dir="fwd")
                handle = (hook_point, hook_fn)
                hooks_to_remove.append(handle)
                steered_nodes.append(node_name)

        # Forward pass with steering
        try:
            with torch.no_grad():
                logits = self.model(input_ids)

            # Compute probabilities
            probs = torch.softmax(logits, dim=-1)

            result = {
                "output": logits,
                "output_probs": probs,
                "steered_nodes": steered_nodes,
                "coefficient": coefficient,
            }
        finally:
            # Always remove hooks (TL: reset_hooks).
            self.model.reset_hooks()

        return result

    def get_top_steered_outputs(
        self,
        inputs: Union[str, torch.Tensor],
        steering_vectors: Optional[Dict[str, torch.Tensor]] = None,
        coefficient: float = 1.0,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get top-k steered outputs with probabilities.

        Convenience method to get the most likely tokens after steering.

        Args:
            inputs: Input text or token IDs
            steering_vectors: Steering vectors (uses self.steering_vectors if None)
            coefficient: Steering strength
            top_k: Number of top predictions to return

        Returns:
            Dict with:
                - 'top_tokens': List of top-k token IDs
                - 'top_probs': List of top-k probabilities
                - 'top_logits': List of top-k logits
        """
        result = self.steer(inputs, steering_vectors, coefficient)
        logits = result["output"]
        probs = result["output_probs"]

        # Get top-k for the last position
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]  # [batch, vocab_size]
            probs = probs[:, -1, :]

        top_probs, top_indices = torch.topk(probs.squeeze(0), top_k)
        top_logits = logits.squeeze(0)[top_indices]

        return {
            "top_tokens": top_indices.tolist(),
            "top_probs": top_probs.tolist(),
            "top_logits": top_logits.tolist(),
            "steered_nodes": result["steered_nodes"],
        }

    def measure_steering_effect(
        self,
        inputs: Union[str, torch.Tensor],
        metric_fn: Callable,
        coefficients: List[float] = None,
        steering_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Measure how steering strength affects model output.

        Useful for understanding the effect of different steering coefficients.

        Args:
            inputs: Input text or token IDs
            metric_fn: Function that takes model output and returns a scalar metric
            coefficients: List of coefficient values to test (default: [0, 0.25, 0.5, 0.75, 1.0])
            steering_vectors: Steering vectors (uses self.steering_vectors if None)

        Returns:
            Dict mapping coefficients to metric values:
            {
                0.0: metric_value,
                0.25: metric_value,
                ...
            }
        """
        if coefficients is None:
            coefficients = [0.0, 0.25, 0.5, 0.75, 1.0]

        results = {}
        for coeff in coefficients:
            output = self.steer(inputs, steering_vectors, coefficient=coeff)
            metric_value = metric_fn(output["output"])
            results[coeff] = metric_value

        return results

    def get_steering_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about computed steering vectors.

        Returns:
            Dict mapping node names to their steering statistics:
            {
                "A0.0": {
                    'norm': float,
                    'shape': tuple,
                    'source_norm': float,
                    'target_norm': float,
                },
                ...
            }
        """
        stats = {}
        for node_name, metadata in self.steering_metadata.items():
            stats[node_name] = {
                "steering_norm": metadata.get("norm", 0.0),
                "shape": metadata["shape"],
                "source_norm": metadata["source_mean"].norm().item(),
                "target_norm": metadata["target_mean"].norm().item(),
            }
        return stats

    def find_steering_coefficient(
        self,
        inputs: Union[str, torch.Tensor, Dict],
        target_metric: Callable,
        steering_vectors: Optional[Dict[str, torch.Tensor]] = None,
        coef_range: Tuple[float, float] = (0.0, 2.0),
        num_steps: int = 11,
    ) -> float:
        """
        Find optimal steering coefficient to maximize target metric via grid search.

        Args:
            inputs: Input to the model
            target_metric: Function that takes model output logits and returns a scalar metric to maximize
            steering_vectors: Steering vectors (uses self.steering_vectors if None)
            coef_range: (min_coeff, max_coeff) range for search
            num_steps: Number of coefficients to test in the range

        Returns:
            Optimal coefficient value
        """
        if steering_vectors is None:
            steering_vectors = self.steering_vectors
            if not steering_vectors:
                raise ValueError(
                    "No steering vectors available. Call compute_steering_vector first."
                )

        coefficients = torch.linspace(coef_range[0], coef_range[1], num_steps).tolist()
        best_coeff = coef_range[0]
        best_metric = float("-inf")

        logger.info(f"\nSearching for optimal steering coefficient in range {coef_range}...")
        for coeff in coefficients:
            output = self.steer(inputs, steering_vectors, coefficient=coeff)
            metric_value = target_metric(output["output"])

            if metric_value > best_metric:
                best_metric = metric_value
                best_coeff = coeff

            logger.info(f"  coeff={coeff:.3f}: metric={metric_value:.4f}")

        logger.info(f"Optimal coefficient: {best_coeff:.3f} (metric={best_metric:.4f})")
        return best_coeff

    def steer_with_multi_vectors(
        self,
        inputs: Union[str, torch.Tensor, Dict],
        steering_vectors_list: List[Dict[str, torch.Tensor]],
        coefficients: Optional[List[float]] = None,
        aggregate: str = "sum",
    ) -> Dict[str, Any]:
        """
        Apply multiple steering vectors with optional aggregation.

        Supports combining steering effects from different sources (e.g., multiple concept pairs).

        Args:
            inputs: Input to the model
            steering_vectors_list: List of steering vector dicts
            coefficients: Coefficients for each vector (default: equal weights)
            aggregate: How to combine vectors: "sum", "mean", "weighted_sum"

        Returns:
            Dict with model output and steering metadata
        """
        if not steering_vectors_list:
            raise ValueError("steering_vectors_list cannot be empty")

        if coefficients is None:
            coefficients = [1.0 / len(steering_vectors_list)] * len(steering_vectors_list)
        elif len(coefficients) != len(steering_vectors_list):
            raise ValueError(
                f"coefficients length ({len(coefficients)}) must match "
                f"steering_vectors_list length ({len(steering_vectors_list)})"
            )

        # Aggregate steering vectors
        combined_vectors = {}

        if aggregate == "sum":
            for vec_dict, coeff in zip(steering_vectors_list, coefficients):
                for node_name, vector in vec_dict.items():
                    if node_name not in combined_vectors:
                        combined_vectors[node_name] = torch.zeros_like(vector)
                    combined_vectors[node_name] += vector * coeff

        elif aggregate == "mean":
            for vec_dict in steering_vectors_list:
                for node_name, vector in vec_dict.items():
                    if node_name not in combined_vectors:
                        combined_vectors[node_name] = torch.zeros_like(vector)
                    combined_vectors[node_name] += vector
            for node_name in combined_vectors:
                combined_vectors[node_name] /= len(steering_vectors_list)

        elif aggregate == "weighted_sum":
            for vec_dict, coeff in zip(steering_vectors_list, coefficients):
                for node_name, vector in vec_dict.items():
                    if node_name not in combined_vectors:
                        combined_vectors[node_name] = torch.zeros_like(vector)
                    combined_vectors[node_name] += vector * coeff

        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")

        # Apply combined vectors
        return self.steer(inputs, combined_vectors, coefficient=1.0)

    def get_random_baseline_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Generate random baseline steering vectors with same statistics as actual vectors.

        Useful for comparing steering effect against random noise.

        Returns:
            Dict mapping node names to random vectors
        """
        random_vectors = {}
        for node_name, steering_vector in self.steering_vectors.items():
            # Match shape and norm of actual vector
            random_vector = torch.randn_like(steering_vector)
            # Normalize to match original norm
            if steering_vector.norm() > 0:
                random_vector = random_vector / random_vector.norm() * steering_vector.norm()
            random_vectors[node_name] = random_vector

        return random_vectors

    def get_semantic_baseline_vectors(
        self, baseline_type: str = "opposite"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate semantic baseline vectors for comparison.

        Args:
            baseline_type: Type of semantic baseline
                - "opposite": Negate the steering vectors
                - "zero": All-zero vectors
                - "half": Scale vectors to 0.5x

        Returns:
            Dict mapping node names to baseline vectors
        """
        baseline_vectors = {}

        if baseline_type == "opposite":
            for node_name, steering_vector in self.steering_vectors.items():
                baseline_vectors[node_name] = -steering_vector
        elif baseline_type == "zero":
            for node_name, steering_vector in self.steering_vectors.items():
                baseline_vectors[node_name] = torch.zeros_like(steering_vector)
        elif baseline_type == "half":
            for node_name, steering_vector in self.steering_vectors.items():
                baseline_vectors[node_name] = steering_vector * 0.5
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        return baseline_vectors

    def analyze_steering_importance(
        self,
        inputs: Union[str, torch.Tensor, Dict],
        target_metric: Callable,
        steering_vectors: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Analyze importance of each node to steering effect via ablation.

        Measures how much removing each node's steering hurts the metric.

        Args:
            inputs: Input to the model
            target_metric: Function that takes logits and returns a scalar
            steering_vectors: Full set of steering vectors

        Returns:
            Dict mapping node names to importance scores
        """
        if steering_vectors is None:
            steering_vectors = self.steering_vectors
            if not steering_vectors:
                raise ValueError(
                    "No steering vectors available. Call compute_steering_vector first."
                )

        # Get full steering metric
        full_output = self.steer(inputs, steering_vectors, coefficient=1.0)
        full_metric = target_metric(full_output["output"])

        # Ablate each node
        importances = {}
        logger.info("\nAnalyzing steering node importance...")

        for ablate_node in steering_vectors:
            # Create ablated steering vectors (remove one node)
            ablated_vectors = {
                name: vec for name, vec in steering_vectors.items() if name != ablate_node
            }

            if ablated_vectors:
                ablated_output = self.steer(inputs, ablated_vectors, coefficient=1.0)
                ablated_metric = target_metric(ablated_output["output"])
            else:
                ablated_metric = target_metric(self.steer(inputs, {}, coefficient=0.0)["output"])

            importance = full_metric - ablated_metric
            importances[ablate_node] = importance
            logger.info(f"  {ablate_node}: importance={importance:.4f}")

        return importances
