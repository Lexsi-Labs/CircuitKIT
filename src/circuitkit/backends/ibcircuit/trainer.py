"""
IBCircuit training loop for circuit discovery via Information Bottleneck.

Implements the IB-based circuit discovery algorithm from Bian, Niu, Yuan et al.,
"IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck" (ICML 2025).
Trains example-specific importance weights on a fixed batch for num_epochs,
then averages them to produce task-general component scores.

Reference: https://github.com/ivanniu/IBCircuit
"""

import logging
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer

try:
    from .model_wrapper import IBHookedTransformer
except ImportError:
    from model_wrapper import IBHookedTransformer
try:

    from .ib_utils import (
        compute_baseline_reference,
        compute_task_loss,
        extract_logits_at_positions,
        validate_batch_data,
    )
except ImportError:
    from ib_utils import (
        compute_baseline_reference,
        compute_task_loss,
        extract_logits_at_positions,
        validate_batch_data,
    )

logger = logging.getLogger(__name__)


# Default hyperparameters (from paper)
DEFAULT_CONFIG = {
    "num_epochs": 1000,
    "learning_rate": 0.05,
    "alpha": 1.0,  # Task preservation weight
    "beta": 1.0,  # IB regularization weight (sparsity control)
    "alpha_loss": "kl",  # Loss mode: 'kl' or 'ce'
    "scope": "heads",
    "mask_type": "sigmoid",
    "log_interval": 100,  # Log every N epochs
    "level": "node",
    "mlp_hook": "mlp_out",
}


def run_ib_discovery(  # noqa: C901 - complex function, refactor out of scope for lint pass
    model: HookedTransformer, dataloader, config: Dict[str, Any], device: str
) -> Tuple[Dict, "IBHookedTransformer"]:
    """
    Run IBCircuit training to discover important components.

    This is the main entry point for IBCircuit discovery. It:
    1. Extracts a fixed batch from the dataloader
    2. Computes baseline performance with frozen model
    3. Trains IB weights for num_epochs
    4. Extracts and returns node importance scores

    Args:
        model: Pre-trained HookedTransformer model (will remain frozen)
        dataloader: Iterator yielding batches with keys:
                   - 'tokens': Input token IDs [batch, seq_len]
                   - 'labels': Answer token IDs [batch]
                   - 'answer_positions': Positions to extract logits [batch]
        config: Training configuration dictionary. Keys:
               - num_epochs: Number of training epochs (default: 1000)
               - learning_rate: Adam learning rate (default: 0.01)
               - alpha: Task loss weight (default: 1.0)
               - beta: IB regularization weight (default: 0.001)
               - alpha_loss: Loss mode 'kl' or 'ce' (default: 'kl')
               - mask_type: IB mask type (default: 'sigmoid')
               - log_interval: Logging frequency (default: 100)
               - batch_size: Optional cap on the fixed training batch. The
                 batch from the dataloader is truncated to this many rows;
                 the main OOM lever for large models. If absent or >= the
                 dataloader batch, the whole batch is kept.
        device: Device to run training on ('cuda' or 'cpu')

    Returns:
        tuple:
        - scores (dict): Component name → importance score(s).
            node   level: str → float  (e.g. {"A0.0": 0.95, "MLP 2": 0.31}).
            neuron level: str → Tensor (e.g. {"A0.0": Tensor[d_head]}).
        - ib_model (IBHookedTransformer): Trained wrapper with final IB weights.

    Raises:
        ValueError: If dataloader is empty or batch validation fails.
        KeyError: If required batch keys are missing.
        RuntimeError: If score extraction fails after training completes.

    Example:
        >>> from circuitkit.backends.ibcircuit.trainer import run_ib_discovery
        >>>
        >>> # Load model and data
        >>> model = HookedTransformer.from_pretrained('gpt2')
        >>> dataloader = build_ioi_dataloader(batch_size=100)
        >>>
        >>> # Configure training
        >>> config = {
        ...     'num_epochs': 1000,
        ...     'beta': 0.001,  # Controls sparsity
        ...     'alpha_loss': 'kl'
        ... }
        >>>
        >>> # Run discovery
        >>> scores = run_ib_discovery(model, dataloader, config, 'cuda')
        >>>
        >>> # Extract important nodes (threshold at 0.5)
        >>> important = {k: v for k, v in scores.items() if v > 0.5}
        >>> print(f"Found {len(important)} important nodes")
    """
    # Merge with defaults
    cfg = {**DEFAULT_CONFIG, **config}

    logger.info("=" * 60)
    logger.info("IBCircuit Discovery Starting")
    logger.info("=" * 60)
    logger.debug("Configuration:")
    for key, value in cfg.items():
        logger.info(f"  {key}: {value}")

    # IBCircuit hooks only need attn.hook_z + hook_mlp_out, both of which
    # are always present on a HookedTransformer regardless of the
    # `use_*` flags. The EAP-targeted flags (use_attn_result,
    # use_split_qkv_input, use_hook_mlp_in) materialise per-head [batch,
    # pos, n_heads, d_model] tensors and blow up memory by n_heads× on
    # long-sequence MCQ tasks. Disable them via try/finally so they are
    # restored on every exit path (success, OOM, propagation).
    # Pre-flight memory check: the dominant cost is NOT the model weights
    # (they are frozen — no gradients, no optimizer state) but the per-epoch
    # forward+backward activations on the single fixed batch. The trainable
    # IB masks are tiny. Estimate weights at the model's actual dtype and add
    # a modest activation headroom factor; on a 3B bf16 model this is ~6 GB of
    # weights, comfortably inside an H200, so we only fail when the device is
    # genuinely too small rather than on a 4×-fp32 over-estimate.
    if torch.cuda.is_available() and device.startswith("cuda"):
        n_params = sum(p.numel() for p in model.parameters())
        param_dtype = next(model.parameters()).dtype
        bytes_per_param = torch.finfo(param_dtype).bits // 8  # 2 for bf16/fp16, 4 for fp32
        weights_gb = (n_params * bytes_per_param) / (1024**3)
        # Frozen model: weights + a ~1.5x headroom for the fixed-batch
        # activations and the autograd graph kept for the IB-mask backward.
        estimated_gb = weights_gb * 2.5
        free_gb = (
            torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
        ) / (1024**3)
        if estimated_gb > free_gb * 0.95:
            raise MemoryError(
                f"IBCircuit training would require ~{estimated_gb:.1f} GB but only "
                f"{free_gb:.1f} GB is free on the device. "
                f"Lower `batch_size` (the IBCircuit fixed-batch size) or use "
                f"EAP-IG / AtP*, which require only inference memory."
            )

    _ib_flags = ("use_attn_result", "use_split_qkv_input", "use_hook_mlp_in")
    _saved_flags = {f: getattr(model.cfg, f) for f in _ib_flags if hasattr(model.cfg, f)}
    for _f in _saved_flags:
        setattr(model.cfg, _f, False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═════════════════════════════════════════════════════════════════
    # PHASE 1: SETUP - Extract fixed batch and prepare for training
    # ═════════════════════════════════════════════════════════════════

    logger.info("\nPhase 1: Loading and validating data...")

    # Extract first batch (will be used for all training epochs)
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        raise ValueError("Dataloader is empty - cannot extract training batch")

    # Validate batch structure
    validate_batch_data(batch)

    # Move to device
    input_ids = batch["tokens"].to(device)
    answer_tokens = batch["labels"].to(device)
    answer_positions = batch["answer_positions"].to(device)

    # OOM lever: cap the fixed training batch. IBCircuit holds ONE batch on
    # the device and runs forward+backward on it every epoch, so peak VRAM
    # scales with batch_size × seq_len. Task dataloaders build the fixed
    # batch from the full example pool (often 100-128 rows) — fine for GPT-2
    # IOI, but it OOMs a 3B/4B model on long-sequence tasks. `batch_size`
    # truncates to the first N rows so the per-epoch activation footprint
    # stays bounded regardless of how many examples the task generated.
    # Absent / non-positive → keep the whole batch (legacy behaviour).
    requested_bs = cfg.get("batch_size")
    if isinstance(requested_bs, int) and 0 < requested_bs < len(input_ids):
        logger.info(
            f"  Capping IBCircuit fixed batch {len(input_ids)} → {requested_bs} "
            f"(config['batch_size']) to bound per-epoch activation memory"
        )
        input_ids = input_ids[:requested_bs].contiguous()
        answer_tokens = answer_tokens[:requested_bs].contiguous()
        answer_positions = answer_positions[:requested_bs].contiguous()

    batch_size = len(input_ids)
    seq_len = input_ids.shape[1]

    logger.debug(f"  Batch size: {batch_size}")
    logger.debug(f"  Sequence length: {seq_len}")
    logger.debug(
        f"  Answer positions: min={answer_positions.min()}, " f"max={answer_positions.max()}"
    )
    logger.debug(f"  Fixed batch will be used for all {cfg['num_epochs']} epochs")

    # ═════════════════════════════════════════════════════════════════
    # PHASE 2: BASELINE COMPUTATION - Establish reference performance
    # ═════════════════════════════════════════════════════════════════

    logger.info("\nPhase 2: Computing baseline performance...")

    model.eval()  # Ensure model is in eval mode

    # Compute baseline log probabilities (for KL mode)
    baseline_logprobs = compute_baseline_reference(
        model=model, input_ids=input_ids, answer_positions=answer_positions, device=device
    )

    # Optionally compute baseline CE loss (for CE mode)
    baseline_ce_loss = None
    if cfg["alpha_loss"] == "ce":
        with torch.no_grad():
            _raw = model(input_ids)
            _logits = _raw if isinstance(_raw, torch.Tensor) else _raw.logits
            baseline_logits_at_ans = extract_logits_at_positions(_logits, answer_positions)
            baseline_ce_loss = F.cross_entropy(baseline_logits_at_ans, answer_tokens).item()
        logger.debug(f"  Baseline CE loss: {baseline_ce_loss:.4f}")

    logger.debug(f"  Baseline computed (shape: {baseline_logprobs.shape})")
    with torch.no_grad():
        top_probs = torch.exp(baseline_logprobs[0]).topk(5)
        logger.debug(f"  Baseline top-5 predictions (example 0, pos {answer_positions[0].item()}):")
        for prob, idx in zip(top_probs.values, top_probs.indices):
            marker = " ◀ answer" if idx.item() == answer_tokens[0].item() else ""
            logger.info(f"    {model.to_string(idx.unsqueeze(0))!r:>15}: {prob:.4f}{marker}")

    # ═════════════════════════════════════════════════════════════════
    # PHASE 3: IB MODEL INITIALIZATION
    # ═════════════════════════════════════════════════════════════════

    logger.info("\nPhase 3: Initializing IB wrapper...")

    # Create IB wrapper around frozen model
    ib_model = IBHookedTransformer(
        model=model,
        batch_size=batch_size,
        device=device,
        scope=cfg["scope"],
        mask_type=cfg["mask_type"],
        level=cfg["level"],
        mlp_hook=cfg["mlp_hook"],
    )

    logger.debug(f"  Model: {model.cfg.model_name}")
    logger.debug(f"  Scope: {ib_model.scope}")
    logger.debug(f"  Level: {ib_model.level}")
    logger.debug(f"  MLP Hook: {ib_model.mlp_hook}")
    logger.debug(f"  Layers: {ib_model.n_layers}")
    logger.debug(f"  Heads per layer: {ib_model.n_heads}")
    logger.debug(
        f"  Total trainable parameters: "
        f"{sum(p.numel() for p in ib_model.get_trainable_parameters())}"
    )

    # Setup optimizer (only IB weights are trainable)
    optimizer = torch.optim.Adam(
        ib_model.get_trainable_parameters(),
        lr=cfg["learning_rate"],
    )

    logger.info(f"  Optimizer: Adam (lr={cfg['learning_rate']})")
    logger.info(f"  Level: {ib_model.level}")
    logger.info(f"  Scope: {ib_model.scope}")

    # ═════════════════════════════════════════════════════════════════
    # PHASE 4: TRAINING LOOP
    # ═════════════════════════════════════════════════════════════════

    logger.info("\nPhase 4: Training IB weights...")
    logger.info(f"  Total epochs: {cfg['num_epochs']}")
    logger.info(f"  Logging every {cfg['log_interval']} epochs")

    # Empty the CUDA cache before entering the high-memory training loop.
    # Phase 2 (baseline computation) builds up memory that isn't always
    # immediately returned to the PyTorch allocator. This ensures we start
    # the training loop with a clean slate, which is critical for long
    # sequence tasks like MMLU.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress_bar = tqdm(range(cfg["num_epochs"]), desc="IB Training")

    # Track NaN steps so we can bail only on persistent instability rather
    # than the first one. Canonical IBCircuit (ivanniu/IBCircuit) has no
    # NaN check; we keep one but make it tolerant + add gradient clipping.
    nan_steps = 0
    nan_step_max = max(20, cfg["num_epochs"] // 20)
    grad_clip_norm = float(cfg.get("grad_clip_norm", 1.0))

    # Training loop over fixed batch
    try:
        for epoch in progress_bar:

            # ─────────────────────────────────────────────────────────────
            # Step 1: Forward pass through IB model
            # ─────────────────────────────────────────────────────────────
            outputs, _, ib_kl_loss = ib_model.forward_with_ib(input_ids)

            # outputs: HookedTransformer output with .logits
            # ib_kl_loss: Scalar regularization loss from IB noise

            # ─────────────────────────────────────────────────────────────
            # Step 2: Extract logits at answer positions
            # ─────────────────────────────────────────────────────────────
            logits_at_answers = extract_logits_at_positions(outputs.logits, answer_positions)

            # CRITICAL MEMORY OPTIMIZATION: Explicitly delete the outputs object.
            # The outputs.logits tensor is massive ([batch_size, seq_len, vocab_size]),
            # especially for tasks with long sequences like MMLU. By deleting the root
            # reference immediately after slicing out the answer positions, PyTorch's
            # autograd engine can release the unused sequence positions from the computation
            # graph before the backward pass, drastically reducing peak VRAM usage.
            del outputs

            # Shape: [batch_size, vocab_size]

            # ─────────────────────────────────────────────────────────────
            # Step 3: Compute task preservation loss
            # ─────────────────────────────────────────────────────────────
            task_loss = compute_task_loss(
                ib_logits=logits_at_answers,
                answer_tokens=answer_tokens,
                baseline_logprobs=baseline_logprobs,
                loss_mode=cfg["alpha_loss"],
                baseline_ce_loss=baseline_ce_loss,
            )

            # ─────────────────────────────────────────────────────────────
            # Step 4: Combine losses
            # ─────────────────────────────────────────────────────────────
            # Total loss = α × task_loss + β × IB_loss
            #
            # α (alpha): Weight for task preservation
            #           Higher = more emphasis on maintaining performance
            #           Typical: 1.0
            #
            # β (beta): Weight for information bottleneck
            #          Higher = more sparsity pressure
            #          Typical range: 0.0001 (dense) to 0.01 (very sparse)
            #          Paper uses: 0.001

            total_loss = cfg["alpha"] * task_loss + cfg["beta"] * ib_kl_loss

            # ─────────────────────────────────────────────────────────────
            # Step 5: Backward pass and optimization
            # ─────────────────────────────────────────────────────────────
            optimizer.zero_grad()

            # Tolerant NaN handling: skip the step but keep training. Canonical
            # IBCircuit (ivanniu/IBCircuit) has no NaN check at all and just
            # trains through. We skip-with-warning here, then bail only if
            # NaNs persist across `nan_step_max` consecutive epochs.
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_steps += 1
                if nan_steps == 1:
                    logger.warning(
                        f"IB training hit NaN/Inf at epoch {epoch}; skipping "
                        f"step. Will bail after {nan_step_max} consecutive."
                    )
                if nan_steps >= nan_step_max:
                    raise ValueError(
                        f"IB training unstable: {nan_steps} consecutive "
                        f"NaN/Inf losses. Try a smaller learning rate, fewer "
                        f"epochs, or a task where the model has a real "
                        f"answer-position signal."
                    )
                continue
            nan_steps = 0

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ib_model.get_trainable_parameters(),
                max_norm=grad_clip_norm,
            )
            optimizer.step()

            # ─────────────────────────────────────────────────────────────
            # Step 6: Logging
            # ─────────────────────────────────────────────────────────────
            if epoch % cfg["log_interval"] == 0 or epoch == 0:
                stats = ib_model.get_statistics()
                log_parts = [
                    f"Epoch {epoch:4d}",
                    f"Task Loss: {task_loss.item():7.4f}",
                    f"IB Loss: {ib_kl_loss.item():7.4f}",
                    f"Total: {total_loss.item():7.4f}",
                ]

                attn_label = "Attn Neurons" if cfg["level"] == "neuron" else "Attn Heads"
                mlp_label = "MLP Neurons" if cfg["level"] == "neuron" else "MLPs"
                if "overall_avg_attn_lambda" in stats:
                    log_parts.append(f"Avg Attn λ: {stats['overall_avg_attn_lambda']:.3f}")
                    log_parts.append(
                        f"{attn_label}: {stats['n_attn_heads_important']}/{stats['total_attn_heads']}"
                    )
                if "overall_avg_mlp_lambda" in stats:
                    log_parts.append(f"Avg MLP λ: {stats['overall_avg_mlp_lambda']:.3f}")
                    log_parts.append(
                        f"{mlp_label}: {stats['n_mlps_important']}/{stats['total_mlps']}"
                    )
                logger.debug(" | ".join(log_parts))

    finally:
        progress_bar.close()
        # CRITICAL: Flush streams to ensure logs appear
        sys.stdout.flush()
        sys.stderr.flush()
        # If we leave the training loop via exception, restore flags now
        # (the success-path restore at the end of Phase 5 won't fire).
        if sys.exc_info()[0] is not None:
            for _f, _v in _saved_flags.items():
                setattr(model.cfg, _f, _v)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("\nPhase 5: Extracting final importance scores...")

    # ═════════════════════════════════════════════════════════════════
    # PHASE 5: EXTRACT FINAL SCORES
    # ═════════════════════════════════════════════════════════════════

    try:
        if cfg["level"] == "neuron":
            # extract_neuron_scores returns Dict[str, Tensor]
            # where each tensor is [d_head] for attn or [d_model] for MLP
            scores = ib_model.extract_neuron_scores()

            if scores is None:
                raise RuntimeError("extract_neuron_scores returned None")

            all_vals = torch.cat([s for s in scores.values()])
            logger.debug(
                f"  Extracted {all_vals.numel()} total neuron scores across {len(scores)} components"
            )
            logger.debug(
                f"  Score range: [{all_vals.min():.4f}, {all_vals.max():.4f}] | Average: {all_vals.mean():.4f}"
            )

            # 1. Flatten and find Top 10 individual neurons overall
            all_neurons = []
            for name, t in scores.items():
                for idx, val in enumerate(t):
                    all_neurons.append((name, idx, val.item()))
            all_neurons.sort(key=lambda x: x[2], reverse=True)

            logger.info("\n  Top 10 Important Neurons Overall:")
            for name, idx, val in all_neurons[:10]:
                logger.info(f"    {name:<6} | Neuron {idx:<4d} : {val:.4f}")

            # Find Top 5 components by number of surviving neurons (score > 0.5)
            logger.warning(
                "\n  [WARNING] Displaying components with neurons > 0.5. "
                "Note: For complex, highly-distributed tasks, maximum IB scores may naturally "
                "fall below 0.5. A strict 0.5 threshold might under-report the active circuit."
            )

            component_stats = [
                (name, t.max().item(), (t > 0.5).sum().item()) for name, t in scores.items()
            ]
            component_stats.sort(key=lambda x: x[2], reverse=True)  # Sort by active count

            logger.info("  Top 5 Components by Surviving Neurons (>0.5):")
            has_active = any(c[2] > 0 for c in component_stats)
            if not has_active:
                logger.info("    No components had neurons scoring > 0.5.")
                # Show the highest scoring component as a fallback
                top_comp = max(component_stats, key=lambda x: x[1])
                logger.info(
                    f"    Highest scoring component was {top_comp[0]} with a max score of {top_comp[1]:.4f}"
                )
            else:
                for name, max_val, active_count in component_stats[:5]:
                    if active_count > 0:
                        logger.info(
                            f"    {name:<6} : {active_count:3d} neurons active (Max score: {max_val:.4f})"
                        )

        else:
            # extract_node_scores returns Dict[str, float]
            scores = ib_model.extract_node_scores(threshold=None)

            if scores is None:
                raise RuntimeError("extract_node_scores returned None")

            score_values = list(scores.values())
            logger.debug(f"  Extracted {len(scores)} node scores")
            logger.debug(
                f"  Score range: [{min(score_values):.4f}, {max(score_values):.4f}] | Average: {sum(score_values)/len(score_values):.4f}"
            )

            # Sort by score descending and get Top 10 Nodes instead of printing all of them
            score_items = list(scores.items())
            score_items.sort(key=lambda x: x[1], reverse=True)

            logger.info("\n  Top 10 Important Nodes:")
            for name, score in score_items[:10]:
                logger.info(f"    {name:<6}: {score:.4f}")

        logger.info(
            "\nIBCircuit Discovery Complete. Full scores have been saved to the artifact file."
        )
        for _f, _v in _saved_flags.items():
            setattr(model.cfg, _f, _v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return scores, ib_model

    except Exception as e:
        for _f, _v in _saved_flags.items():
            setattr(model.cfg, _f, _v)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"Failed to extract scores: {e}")
        import traceback

        traceback.print_exc()
        raise RuntimeError("Failed to extract scores during Phase 5") from e


def train_ib_epoch(
    ib_model: IBHookedTransformer,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    answer_tokens: torch.Tensor,
    answer_positions: torch.Tensor,
    baseline_logprobs: torch.Tensor,
    config: Dict[str, Any],
    baseline_ce_loss: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Perform a single IB training step (forward + backward + optimiser update).

    This is a standalone helper for custom training loops or unit testing.
    It is not called internally by run_ib_discovery, which manages its own loop.

    Args:
        ib_model (IBHookedTransformer): Wrapper with trainable IB weights.
        optimizer (torch.optim.Optimizer): Optimiser for IB weights.
        input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
        answer_tokens (torch.Tensor): Ground-truth answer token IDs [batch_size].
        answer_positions (torch.Tensor): Positions to extract logits [batch_size].
        baseline_logprobs (torch.Tensor): Baseline log-probabilities [batch_size, vocab_size].
        config (dict): Training config with keys 'alpha', 'beta', 'alpha_loss'.
        baseline_ce_loss (float | None): Baseline CE scalar. Required for 'ce' mode.

    Returns:
        tuple[float, float, float]: (task_loss, ib_kl_loss, total_loss) as Python floats.
    """
    # Forward pass
    outputs, _, ib_kl_loss = ib_model.forward_with_ib(input_ids)

    # Extract logits at answer positions
    logits_at_answers = extract_logits_at_positions(outputs.logits, answer_positions)

    # CRITICAL MEMORY OPTIMIZATION: Free the full logits tensor early
    # to prevent pinning the massive [batch_size, seq_len, vocab_size]
    # tensor in VRAM during the subsequent backward pass.
    del outputs

    # Compute task loss
    task_loss = compute_task_loss(
        ib_logits=logits_at_answers,
        answer_tokens=answer_tokens,
        baseline_logprobs=baseline_logprobs,
        loss_mode=config["alpha_loss"],
        baseline_ce_loss=baseline_ce_loss,
    )

    # Combined loss
    total_loss = config["alpha"] * task_loss + config["beta"] * ib_kl_loss

    # Backward and update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return task_loss.item(), ib_kl_loss.item(), total_loss.item()
