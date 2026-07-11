# FILE: circuitkit/applications/editing/rome_wrapper.py
"""
ROME (Rank-One Model Editing) Wrapper for CircuitKit.

Implements ROME, a lightweight method for editing knowledge in transformers
by applying rank-one perturbations to weight matrices. ROME works by:

1. Computing a rank-one update to a target weight matrix
2. Deriving the update from prompt-based activations and gradients
3. Applying the update with minimal interference

Key insight: Facts are typically stored in MLP layers. A low-rank edit
to the first weights in the MLP can modify fact storage without catastrophic
forgetting.

Component type is fixed by the method: ROME edits MLP weight matrices by
design, so there is no ``scope`` knob (unlike pruning / quantization).

Reference:
- Meng et al., "Mass-Editing Memory in a Language Model" (ROME/MEMIT)
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from circuitkit.applications.common_utils._tokenization import (  # noqa: E402 - import after intentional pre-import setup
    ScoringError,
    SubjectLocationError,
    build_teacher_forced,
    locate_subject_last_token,
    sample_random_prefixes,
    score_target,
    tokenize_prompt,
)

try:
    from transformer_lens import HookedTransformer
    from transformer_lens.utils import get_act_name
except ImportError:
    HookedTransformer = None
    get_act_name = None

from .knowledge_editing import EditResult  # noqa: E402 - import after intentional pre-import setup


@dataclass
class RomeEditVectors:
    """Vectors used in a ROME edit."""

    rank_one_matrix: torch.Tensor  # Shape: [d_mlp, d_model] — matches W_out
    update_vector: torch.Tensor  # Shape: [d_model] or [d_model, d_hidden]
    original_weight: torch.Tensor  # Original weight matrix before edit
    edited_weight: Optional[torch.Tensor]  # Weight matrix after edit


class RomeHandler:
    """
    Wrapper for ROME (Rank-One Model Editing).

    ROME edits knowledge by applying rank-one updates to MLP weight matrices.
    The key idea is that facts are encoded in MLP layers, and we can modify
    them with minimal side effects using low-rank perturbations.

    Attributes:
        model: HookedTransformer model
        device: Compute device (CPU or CUDA)
        hessian_cache: Cache of Hessian matrices for efficiency
    """

    def __init__(self, model: HookedTransformer):
        """
        Initialize RomeHandler.

        Args:
            model: HookedTransformer model to edit

        Examples:
            >>> from transformer_lens import HookedTransformer
            >>> model = HookedTransformer.from_pretrained("gpt2-small")
            >>> rome = RomeHandler(model)
        """
        self.model = model
        self.device = model.cfg.device
        from ..common_utils._device import warn_if_mps_editing

        warn_if_mps_editing(self.device)
        self.hessian_cache = {}
        self.edit_vectors_cache = {}

    def edit_single_fact(
        self,
        prompt: str,
        subject: str,
        target: str,
        target_layer: int,
        fact_type: str = "factual",
        hessian_sample_size: int = None,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> "EditResult":
        """
        Apply a single ROME edit to a specific layer.

        Strategy:
        1. Tokenize prompt and identify key positions
        2. Compute gradients of target logits w.r.t. MLP weights
        3. Compute or retrieve Hessian approximation
        4. Compute rank-one update vector
        5. Apply update to weight matrix

        Args:
            prompt: Prompt containing fact (e.g., "The capital of France is")
            subject: Subject entity (e.g., "France")
            target: Target fact (e.g., "Paris")
            target_layer: MLP layer to edit
            fact_type: Type of fact being edited
            hessian_sample_size: Number of samples for Hessian estimation.
                                 If None, uses model default.
            use_corpus_C: If True (default, paper-faithful), compute u as
                          (C + λI)^{-1} k / |...| where C ≜ E[k k^T] over a
                          text corpus. If False, fall back to u = k/|k|
                          (pre-1.3 "loud-facts" behaviour, ablation only).
            cov_n_samples: Samples used to estimate C. Paper uses ~100k;
                           default 1000 is a fast approximation.
            cov_texts: Optional iterable of texts used to estimate C. If None
                       and use_corpus_C is True, falls back to a small in-tree
                       corpus (smoke-test grade only).
            corpus_id: Cache key suffix distinguishing different corpora at
                       the same n_samples. Use a stable id (e.g. "wiki100k")
                       when supplying your own cov_texts so the cache doesn't
                       collide with previous runs on a different corpus.
            n_prefixes: Number of random prefixes (in addition to the bare
                        prompt) used to average the v-vector NLL optimisation
                        loss. KL stays anchored to the bare prompt. Paper
                        canonical is 20; default 5 is a fast approximation.
                        Set to 0 to disable (pre-1.4 behaviour).
            prefix_seed: Local RNG seed for reproducible prefix sampling.
                         Does not touch the global random state.

        Returns:
            EditResult with success metrics

        Examples:
            >>> result = rome.edit_single_fact(
            ...     prompt="The capital of France is",
            ...     subject="France",
            ...     target="Lyon",
            ...     target_layer=6
            ... )
            >>> print(f"Edit successful: {result.success}")
        """

        # 1. Pre-flight Guard: Check for empty string inputs
        if not prompt or not subject or not target:
            return EditResult(
                success=False,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=target_layer,
                confidence_before=0.0,
                confidence_after=0.0,
                edit_magnitude=0.0,
                interference_ratio=1.0,
                error_message="Prompt, subject, and target must be non-empty strings.",
            )

        # 2. Token-level Guard: Ensure tokenization didn't result in an empty sequence
        tokens = self.model.to_tokens(prompt)
        if tokens.shape[1] == 0:
            return EditResult(
                success=False,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=target_layer,
                confidence_before=0.0,
                confidence_after=0.0,
                edit_magnitude=0.0,
                interference_ratio=1.0,
                error_message="Tokenization resulted in an empty sequence.",
            )

        try:
            # Validate layer index
            if target_layer < 0 or target_layer >= self.model.cfg.n_layers:
                return EditResult(
                    success=False,
                    fact_prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=target_layer,
                    confidence_before=0.0,
                    confidence_after=0.0,
                    edit_magnitude=0.0,
                    interference_ratio=1.0,
                    error_message=(
                        f"Invalid layer: target layer {target_layer} is out of bounds for model "
                        f"with {self.model.cfg.n_layers} layers."
                    ),
                )

            # Get baseline confidence before editing
            conf_before = self._get_target_confidence(prompt, target)

            # Compute rank-one edit vector
            edit_vectors = self._compute_edit_vectors(
                prompt=prompt,
                subject=subject,
                target=target,
                target_layer=target_layer,
                use_corpus_C=use_corpus_C,
                cov_n_samples=cov_n_samples,
                cov_texts=cov_texts,
                corpus_id=corpus_id,
                n_prefixes=n_prefixes,
                prefix_seed=prefix_seed,
            )

            if edit_vectors is None:
                return EditResult(
                    success=False,
                    fact_prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=target_layer,
                    confidence_before=conf_before,
                    confidence_after=0.0,
                    edit_magnitude=0.0,
                    interference_ratio=1.0,
                    error_message="Failed to compute edit vectors",
                )

            # Apply edit to model weights
            edit_magnitude = self._apply_rome_edit(
                target_layer=target_layer,
                edit_vectors=edit_vectors,
            )

            # Get confidence after editing
            conf_after = self._get_target_confidence(prompt, target)

            # Compute success metrics
            confidence_increase = conf_after - conf_before
            success = edit_magnitude > 0  # Threshold for successful edit

            return EditResult(
                success=success,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=target_layer,
                confidence_before=conf_before,
                confidence_after=conf_after,
                edit_magnitude=edit_magnitude,
                interference_ratio=0.0,  # Will be computed in main editor
                metadata={
                    "method": "rome",
                    "confidence_increase": confidence_increase,
                    "hessian_cached": target_layer in self.hessian_cache,
                },
            )

        except ValueError:
            # Let our explicit API validation errors bubble up
            raise

        except Exception as e:
            return EditResult(
                success=False,
                fact_prompt=prompt,
                subject=subject,
                target=target,
                target_layer=target_layer,
                confidence_before=0.0,
                confidence_after=0.0,
                edit_magnitude=0.0,
                interference_ratio=1.0,
                error_message=f"ROME edit failed: {str(e)}",
            )

    def _compute_edit_vectors(
        self,
        prompt: str,
        subject: str,
        target: str,
        target_layer: int,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> Optional["RomeEditVectors"]:
        """
        Compute ROME rank-one edit vectors.

        Orchestrates _compute_u (left vector) and _optimize_v (right vector),
        then assembles the [d_mlp, d_model] update matrix that is added to W_out.
        """
        try:
            mlp_weight = self._get_mlp_weight(target_layer)
            if mlp_weight is None:
                return None

            # Step 1: Left vector u — activation at hook_post for subject token
            left_vector = self._compute_u(
                prompt,
                subject,
                target_layer,
                use_corpus_C=use_corpus_C,
                cov_n_samples=cov_n_samples,
                cov_texts=cov_texts,
                corpus_id=corpus_id,
            )
            if left_vector is None:
                return None

            # Step 2: Right vector v — via constrained Adam optimization
            right_vector = self._optimize_v(
                prompt,
                subject,
                target,
                target_layer,
                left_vector,
                n_prefixes=n_prefixes,
                prefix_seed=prefix_seed,
            )
            if right_vector is None:
                return None

            # Step 3: Rank-one update matrix [d_mlp, d_model] — matches W_out shape
            rank_one = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)

            return RomeEditVectors(
                rank_one_matrix=rank_one,
                update_vector=right_vector,
                original_weight=mlp_weight.detach().clone(),
                edited_weight=None,
            )

        except Exception as e:
            warnings.warn(f"Error computing edit vectors: {e}")
            return None

    def _compute_u(
        self,
        prompt: str,
        subject: str,
        target_layer: int,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
    ) -> Optional[torch.Tensor]:
        """
        Compute the left vector (u) for the ROME rank-one update.

        Paper-faithful (Eqn. 14): u = (C + λI)^{-1} k / |...|, where
        C ≜ E[k k^T] over a corpus and k is the activation at
        `blocks.{layer}.mlp.hook_post` at the subject's last token. Using
        C^{-1} makes the update minimum-norm in the corpus metric — without
        it, the edit bleeds into unrelated directions ("loud facts" failure
        mode, Bug #6).

        Falls back to u = k/|k| when use_corpus_C=False or when C estimation
        fails. The fallback is the pre-1.3 behaviour, kept for ablation.

        Args:
            prompt:        Prompt text containing the subject.
            subject:       Subject string (used to locate the last-token index).
            target_layer:  MLP layer being edited.
            use_corpus_C:  If True (default, paper-faithful), use C^{-1}.
            cov_n_samples: Samples used when estimating C.
            cov_texts:     Optional iterable of corpus texts; defaults to
                        in-tree fallback corpus.
            corpus_id:     Cache key suffix; pass a stable id like 'wiki1k'
                        when supplying your own `cov_texts`.
        """
        try:
            prompt_tokens = tokenize_prompt(self.model, prompt)
            lookup_idx = locate_subject_last_token(self.model, prompt, subject)

            captured = {}

            def capture_hook_post(activation, hook):
                captured["post"] = activation.detach().clone()

            hook_point = f"blocks.{target_layer}.mlp.hook_post"
            with torch.no_grad():
                self.model.run_with_hooks(
                    prompt_tokens,
                    fwd_hooks=[(hook_point, capture_hook_post)],
                )

            if "post" not in captured:
                return None

            # k shape: [d_mlp]. Variable kept named `k` here for clarity vs the
            # paper; the rank-one matrix is still assembled as `u v^T` downstream.
            k = captured["post"][0, lookup_idx, :].clone()

            if use_corpus_C:
                from circuitkit.applications.common_utils._covariance import get_covariance, solve_with_C

                C = get_covariance(
                    self.model,
                    layer=target_layer,
                    hook_name="mlp.hook_post",
                    texts=cov_texts,
                    n_samples=cov_n_samples,
                    corpus_id=corpus_id,
                )
                if C is not None:
                    v = solve_with_C(C, k, lam=1e-2)  # solves (C + λI) v = k
                    return v / (v.norm() + 1e-8)
                warnings.warn(
                    "Corpus C unavailable; falling back to u = k/|k| " "(loud-facts regime)."
                )

            return k / (k.norm() + 1e-8)

        except Exception as e:
            warnings.warn(f"Error computing u vector: {e}")
            return None

    def _optimize_v(
        self,
        prompt: str,
        subject: str,
        target: str,
        target_layer: int,
        left_vector: torch.Tensor,
        v_num_grad_steps: int = 20,
        v_lr: float = 0.5,
        kl_factor: float = 0.0625,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 3.0,
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> Optional[torch.Tensor]:
        """
        Compute the right vector (v) via constrained Adam optimization.

        Mirrors compute_v.py: initializes a delta vector in the residual stream,
        injects it at hook_mlp_out at the subject's last token position, and
        optimizes with NLL + KL + weight-decay loss for v_num_grad_steps steps.

        Random-prefix averaging (Step 1.4): NLL is averaged across the bare
        prompt and `n_prefixes` short random prefixes prepended to it. The
        injected δ is shared across all variants — only the subject token
        index and the target-prediction position shift per variant.

        KL anchor stays on the BARE prompt only (paper convention): KL measures
        drift in the unedited model's distribution at the subject position; it
        is a specificity penalty, not a generalisation signal.

        After optimization, solves the linear system:
            right_vector = (target_init + delta - cur_output)
                        / dot(cur_input, left_vector)
        on the BARE prompt to yield the v such that W_out + u v^T produces
        the desired output.
        """
        try:
            # ── Bare-prompt teacher-forced sequence (KL anchor + linear solve) ──
            seq_bare = build_teacher_forced(self.model, prompt, target)
            prompt_tokens_bare = seq_bare.full_ids[:, : seq_bare.prompt_len]
            target_first_id = int(seq_bare.target_ids[0].item())
            lookup_idx_bare = locate_subject_last_token(self.model, prompt, subject)

            d_model = self.model.cfg.d_model
            hook_mlp_out = f"blocks.{target_layer}.hook_mlp_out"
            # Match the model's parameter dtype so the optimized delta / loss
            # tensors stay consistent with bf16/fp16 models (otherwise the
            # default-float32 `delta` triggers a dtype-mismatch matmul).
            model_dtype = self.model.cfg.dtype

            # Capture the baseline distribution BEFORE any delta for KL reference,
            # using the bare prompt only.
            with torch.no_grad():
                base_logits = self.model(prompt_tokens_bare)
                kl_distr_init = (
                    torch.nn.functional.log_softmax(base_logits[0, lookup_idx_bare, :], dim=-1)
                    .detach()
                    .clone()
                )

            # ── Sample random prefixes and build per-variant inputs ──
            prefixes = (
                sample_random_prefixes(
                    self.model,
                    seed=prefix_seed,
                    n_short=max(0, n_prefixes // 2),
                    n_med=max(0, n_prefixes - n_prefixes // 2),
                )
                if n_prefixes > 0
                else []
            )

            # Each variant: (prompt_only_token_tensor, subject_idx, last_pos_idx)
            # last_pos_idx is the position whose logits predict the next token
            # (target's first token); for prompt-only forwards that's prompt_len - 1.
            variants: List[Tuple[torch.Tensor, int, int]] = []
            variants.append((prompt_tokens_bare, lookup_idx_bare, prompt_tokens_bare.shape[1] - 1))
            n_skipped = 0
            for pfx in prefixes:
                pref_prompt = pfx + " " + prompt
                try:
                    seq_v = build_teacher_forced(self.model, pref_prompt, target)
                    subj_v = locate_subject_last_token(self.model, pref_prompt, subject)
                except (ScoringError, SubjectLocationError):
                    n_skipped += 1
                    continue
                pt_v = seq_v.full_ids[:, : seq_v.prompt_len]
                variants.append((pt_v, subj_v, pt_v.shape[1] - 1))

            logger.info(
                f"compute_v[L={target_layer}] subject={subject!r} target={target!r}: "
                f"{len(variants)} variant(s) (1 bare + {len(variants) - 1} prefixed, "
                f"{n_skipped} skipped)"
            )

            # ── Optimisation: shared δ across variants ──
            delta = torch.zeros(
                (d_model,), requires_grad=True, device=self.device, dtype=model_dtype
            )
            target_init: Optional[torch.Tensor] = None
            current_subj: int = lookup_idx_bare  # closed-over by inject_delta

            def inject_delta(activation, hook):
                nonlocal target_init
                activation = activation.clone()
                if target_init is None:
                    # Record the clean MLP output at the BARE-prompt subject token.
                    # Only set on the very first call (which is the bare-prompt
                    # forward — see loop ordering below).
                    target_init = activation[0, current_subj, :].detach().clone()
                activation[0, current_subj, :] = activation[0, current_subj, :] + delta
                return activation

            # Freeze model weights; only delta is optimized
            for p in self.model.parameters():
                p.requires_grad_(False)

            opt = torch.optim.Adam([delta], lr=v_lr)

            for it in range(v_num_grad_steps):
                opt.zero_grad()

                # ── NLL: averaged over all variants. Process bare first so
                #    target_init is captured at the correct (bare) location. ──
                nll_total = torch.zeros((), device=self.device, dtype=model_dtype)
                kl_loss = torch.zeros((), device=self.device, dtype=model_dtype)

                for v_idx, (pt_v, subj_v, last_pos) in enumerate(variants):
                    current_subj = subj_v
                    logits = self.model.run_with_hooks(
                        pt_v,
                        fwd_hooks=[(hook_mlp_out, inject_delta)],
                    )
                    log_probs = torch.log_softmax(logits[0, last_pos, :], dim=-1)
                    nll_total = nll_total + (-log_probs[target_first_id])

                    # KL only on the bare prompt (variant 0), at the subject token.
                    if v_idx == 0:
                        kl_log_probs = torch.nn.functional.log_softmax(logits[0, subj_v, :], dim=-1)
                        kl_loss = kl_factor * torch.nn.functional.kl_div(
                            kl_distr_init,
                            kl_log_probs,
                            log_target=True,
                            reduction="sum",
                        )

                nll_loss = nll_total / len(variants)

                # Weight decay: keep delta small relative to the original output
                weight_decay = v_weight_decay * (
                    torch.norm(delta) / (torch.norm(target_init) ** 2 + 1e-8)
                )

                loss = nll_loss + kl_loss + weight_decay

                if it % 5 == 0:
                    logger.debug(
                        f"  v-opt step {it}: loss={loss.item():.4f} "
                        f"(nll={nll_loss.item():.4f}, kl={kl_loss.item():.4f}, "
                        f"wd={weight_decay.item():.4f}, n_var={len(variants)})"
                    )

                if loss.item() < 5e-2:
                    break
                if it == v_num_grad_steps - 1:
                    break

                loss.backward()
                opt.step()

                # Project delta back within L2 ball (clamp_norm_factor * ||target_init||)
                max_norm = clamp_norm_factor * target_init.norm()
                if delta.norm() > max_norm:
                    with torch.no_grad():
                        delta[...] = delta * max_norm / delta.norm()

            # Re-enable model gradients
            for p in self.model.parameters():
                p.requires_grad_(True)

            target_vec = (target_init + delta).detach()

            # ── Linear solve on the BARE prompt only ──
            # Capture cur_input (hook_post) and cur_output (hook_mlp_out) at the
            # bare-prompt subject token; these define the system v* solves.
            post_cap, out_cap = {}, {}

            def cap_post(act, hook):
                post_cap["v"] = act.detach().clone()

            def cap_out(act, hook):
                out_cap["v"] = act.detach().clone()

            with torch.no_grad():
                self.model.run_with_hooks(
                    prompt_tokens_bare,
                    fwd_hooks=[
                        (f"blocks.{target_layer}.mlp.hook_post", cap_post),
                        (hook_mlp_out, cap_out),
                    ],
                )

            cur_input = post_cap["v"][0, lookup_idx_bare, :].clone()  # [d_mlp]
            cur_output = out_cap["v"][0, lookup_idx_bare, :].clone()  # [d_model]

            # Linear solve: v* = (target - cur_output) / <cur_input, u>
            denom = torch.dot(cur_input, left_vector)
            right_vector = (target_vec - cur_output) / (denom + 1e-8)

            logger.info(
                f"compute_v[L={target_layer}] denom={denom.item():.4f} "
                f"|right|={right_vector.norm().item():.4f} "
                f"|delta|={delta.norm().item():.4f}"
            )

            return right_vector

        except Exception as e:
            warnings.warn(f"Error optimizing v vector: {e}")
            for p in self.model.parameters():
                p.requires_grad_(True)
            return None

    def _apply_rome_edit(
        self,
        target_layer: int,
        edit_vectors: RomeEditVectors,
    ) -> float:
        """
        Apply ROME edit to W_out.

        Adds the rank-one matrix (u v^T, shape [d_mlp, d_model]) directly to W_out.
        No manual scaling — the linear system in _optimize_v already encodes the
        correct magnitude.
        """
        try:
            mlp_weight = self._get_mlp_weight(target_layer)
            if mlp_weight is None:
                return 0.0

            upd_matrix = edit_vectors.rank_one_matrix

            # Handle GPT-2 style transposed weights if necessary
            if upd_matrix.shape != mlp_weight.shape:
                if upd_matrix.T.shape == mlp_weight.shape:
                    upd_matrix = upd_matrix.T
                else:
                    warnings.warn(
                        f"Shape mismatch: update {upd_matrix.shape} vs W_out {mlp_weight.shape}. "
                        "Skipping edit."
                    )
                    return 0.0

            with torch.no_grad():
                mlp_weight.data += upd_matrix

            edit_magnitude = torch.norm(upd_matrix).item()
            self._cache_edit_vectors(target_layer, edit_vectors)
            return edit_magnitude

        except Exception as e:
            warnings.warn(f"Error applying ROME edit: {e}")
            return 0.0

    def _get_mlp_weight(self, layer: int) -> Optional[nn.Parameter]:
        """Get MLP output-projection weight at given layer (MEMIT edits W_out)."""
        if layer < 0 or layer >= self.model.cfg.n_layers:
            raise ValueError(
                f"Target layer {layer} is out of bounds for model with {self.model.cfg.n_layers} layers."
            )

        mlp_module = self.model.blocks[layer].mlp

        if hasattr(mlp_module, "W_out"):
            return mlp_module.W_out
        elif hasattr(mlp_module, "w_out"):
            return mlp_module.w_out
        elif hasattr(mlp_module, "c_proj"):
            return mlp_module.c_proj.weight
        else:
            raise AttributeError(
                f"Could not locate an output projection matrix (W_out/c_proj) on MLP at layer {layer}."
            )

    def _get_target_confidence(self, prompt: str, target: str) -> float:
        """First-token probability of `target` given `prompt`. Delegates
        to the unified scorer in `_scoring`, which handles BPE / SentencePiece
        / WordPiece tokeniser conventions, multi-token targets, and BOS
        settings consistently across all TransformerLens-supported models.
        """
        try:
            return score_target(self.model, prompt, target).first_token_prob
        except ScoringError as exc:
            import warnings

            warnings.warn(f"Could not score target {target!r}: {exc}")
            return 0.0

    def _cache_edit_vectors(self, layer: int, edit_vectors: RomeEditVectors):
        """Cache edit vectors for a layer."""
        self.edit_vectors_cache[layer] = edit_vectors

    def get_cached_edit_vectors(self, layer: int) -> Optional[RomeEditVectors]:
        """Retrieve cached edit vectors for a layer."""
        return self.edit_vectors_cache.get(layer)

    def clear_cache(self):
        """Clear all caches."""
        self.hessian_cache.clear()
        self.edit_vectors_cache.clear()


class RomeWrapper:
    """Thin convenience shim over RomeHandler for per-layer editing.

    Mirrors the call style used in application script 25:
        wrapper = RomeWrapper(model, target_layer=layer)
        success = wrapper.edit(subject=..., target=..., prompt=...)
    """

    def __init__(self, model: "HookedTransformer", target_layer: int) -> None:
        self._handler = RomeHandler(model)
        self.target_layer = target_layer

    def edit(self, subject: str, target: str, prompt: str) -> bool:
        result = self._handler.edit_single_fact(
            prompt=prompt,
            subject=subject,
            target=target,
            target_layer=self.target_layer,
        )
        return result.success
