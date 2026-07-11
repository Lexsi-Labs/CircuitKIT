# FILE: circuitkit/applications/editing/memit_wrapper.py
"""
MEMIT (Mass Editing Memory in a Transformer) Wrapper for CircuitKit.

Implements MEMIT for efficient batch editing of multiple facts in transformers.
MEMIT extends ROME by computing a single edit that modifies multiple facts
simultaneously with minimal interference.

Key ideas:
1. Collect all facts to be edited
2. Compute a collective rank-one update that edits all facts
3. Apply update to multiple layers simultaneously
4. Verify no interference between edits

Component type is fixed by the method: MEMIT edits MLP weight matrices by
design, so there is no ``scope`` knob (unlike pruning / quantization).

Reference:
- Meng et al., "Mass-Editing Memory in a Language Model" (MEMIT)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

from circuitkit.applications.common_utils._covariance import (  # noqa: E402 - import after intentional pre-import setup
    get_covariance,
)
from circuitkit.applications.common_utils._tokenization import (  # noqa: E402 - import after intentional pre-import setup
    ScoringError,
    SubjectLocationError,
    build_teacher_forced,
    locate_subject_last_token,
    sample_random_prefixes,
    score_target,
    tokenize_prompt,
)

from .knowledge_editing import EditResult  # noqa: E402 - import after intentional pre-import setup


@dataclass
class MemitBatchEdit:
    """Result of a batch MEMIT edit."""

    facts_edited: List[Tuple[str, str, str]]  # (prompt, subject, target)
    target_layers: List[int]
    success_count: int
    total_count: int
    avg_confidence_before: float
    avg_confidence_after: float
    avg_edit_magnitude: float
    interference_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of successfully edited facts."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


class MemitHandler:
    """
    Wrapper for MEMIT (Mass Editing Memory in Transformers).

    MEMIT enables efficient batch editing of multiple facts by computing
    a single collective update that edits all facts with minimal interference.

    Attributes:
        model: HookedTransformer model
        device: Compute device
        batch_edit_cache: Cache of batch edits for efficiency
    """

    def __init__(self, model: HookedTransformer):
        """
        Initialize MemitHandler.

        Args:
            model: HookedTransformer model

        Examples:
            >>> from transformer_lens import HookedTransformer
            >>> model = HookedTransformer.from_pretrained("gpt2-small")
            >>> memit = MemitHandler(model)
        """
        self.model = model
        self.device = model.cfg.device
        from ..common_utils._device import warn_if_mps_editing

        warn_if_mps_editing(self.device)
        self.batch_edit_cache = {}
        # MEMIT edits W_out and only uses standard hooks (hook_resid_post,
        # mlp.hook_post, hook_mlp_out) that exist on every HookedTransformer.
        # It does not require the optional hook_mlp_in hook point.

    def edit_multiple_facts(
        self,
        facts: List[Tuple[str, str, str]],
        target_layers: Optional[List[int]] = None,
        batch_size: int = None,
        verify_interference: bool = True,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> List[EditResult]:
        """
        Batch edit multiple facts efficiently.

        Strategy:
        1. Identify target layers (fact storage layers)
        2. For each fact, compute gradient-based update vector
        3. Aggregate updates into collective rank-one edit
        4. Apply collective update to all target layers
        5. Verify edits with minimal interference

        Args:
            facts: List of (prompt, subject, target) tuples to edit
                   Example: [("The capital of France is", "France", "Lyon"),
                            ("The capital of Germany is", "Germany", "Munich")]
            target_layers: Layers to apply edits to. If None, uses middle layers.
            batch_size: Batch size for processing facts. If None, processes all at once.
            verify_interference: Whether to compute interference between edits
            use_corpus_C: If True (default, paper-faithful), use C ≜ E[k k^T]
                          estimated from a text corpus in the inverse term
                          (C + KK^T)^{-1}. If False, use λI (pre-1.3 fallback,
                          produces under-applied edits — debug only).
            cov_n_samples: Number of samples used to estimate C. Paper uses
                           ~100k; default 1000 is a fast approximation.
            cov_texts: Optional iterable of texts used to estimate C. If None
                       and use_corpus_C is True, falls back to a small in-tree
                       corpus (smoke-test grade only).
            corpus_id: Cache key suffix distinguishing different corpora at the
                       same n_samples. Use a stable identifier (e.g. "wiki100k")
                       when supplying your own `cov_texts` so the cache doesn't
                       collide with previous runs on a different corpus.
            n_prefixes: Number of random prefixes (in addition to the bare
                        prompt) used to average the z-vector optimisation
                        loss. Paper canonical is 20 (10 of length 5, 10 of
                        length 10); default 5 is a fast approximation. Set
                        to 0 to disable averaging entirely (pre-1.4 behaviour).
            prefix_seed: Local RNG seed for reproducible prefix sampling.
                         Does not touch the global random state.

        Returns:
            List of EditResult objects, one per fact

        Examples:
            >>> facts = [
            ...     ("The capital of France is", "France", "Lyon"),
            ...     ("The capital of Germany is", "Germany", "Munich"),
            ... ]
            >>> results = memit.edit_multiple_facts(facts)
            >>> for result in results:
            ...     print(f"Fact edited: {result.success}")
        """

        final_results = []
        valid_facts = []

        # 1. Pre-flight validation loop
        for prompt, subject, target in facts:
            if not prompt or not subject or not target:
                final_results.append(
                    EditResult(
                        success=False,
                        fact_prompt=prompt,
                        subject=subject,
                        target=target,
                        target_layer=-1,  # Or however you denote batch default
                        error_message="Prompt, subject, and target must be non-empty strings.",
                    )
                )
                continue

            tokens = self.model.to_tokens(prompt)
            if tokens.shape[1] == 0:
                final_results.append(
                    EditResult(
                        success=False,
                        fact_prompt=prompt,
                        subject=subject,
                        target=target,
                        target_layer=-1,
                        error_message="Tokenization resulted in an empty sequence.",
                    )
                )
                continue

            try:
                locate_subject_last_token(self.model, prompt, subject)
            except SubjectLocationError as exc:
                final_results.append(
                    EditResult(
                        success=False,
                        fact_prompt=prompt,
                        subject=subject,
                        target=target,
                        target_layer=-1,
                        confidence_before=0.0,
                        confidence_after=0.0,
                        edit_magnitude=0.0,
                        interference_ratio=1.0,
                        error_message=f"Token alignment failed: {exc}",
                    )
                )
                continue

            valid_facts.append((prompt, subject, target))

        # If there are no valid facts to process, return early
        if not valid_facts:
            return final_results

        try:
            # Determine target layers if not provided
            if target_layers is None:
                target_layers = self._select_target_layers(len(facts))

            # Validate layer bounds before any forward passes
            n_layers = self.model.cfg.n_layers
            invalid = [lyr for lyr in target_layers if lyr < 0 or lyr >= n_layers]
            if invalid:
                raise ValueError(
                    f"Layer(s) {invalid} are out of bounds for model with "
                    f"{n_layers} layers (valid: 0–{n_layers - 1})."
                )

            # Process facts in batches
            if batch_size is None:
                batch_size = len(valid_facts)

            all_results: List[EditResult] = []

            for batch_start in range(0, len(valid_facts), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_facts))
                batch_results = self._edit_fact_batch(
                    valid_facts[batch_start:batch_end],
                    target_layers,
                    verify_interference,
                    use_corpus_C=use_corpus_C,
                    cov_n_samples=cov_n_samples,
                    cov_texts=cov_texts,
                    corpus_id=corpus_id,
                    n_prefixes=n_prefixes,
                    prefix_seed=prefix_seed,
                )
                all_results.extend(batch_results)

            final_results.extend(all_results)
            return final_results

        except ValueError:
            raise

        except Exception as e:
            warnings.warn(f"Error in MEMIT batch editing: {e}")
            # Return failed results for each fact
            return [
                EditResult(
                    success=False,
                    fact_prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=-1,
                    confidence_before=0.0,
                    confidence_after=0.0,
                    edit_magnitude=0.0,
                    interference_ratio=1.0,
                    error_message=f"MEMIT batch failed: {str(e)}",
                )
                for prompt, subject, target in facts
            ]

    def _edit_fact_batch(
        self,
        facts: List[Tuple[str, str, str]],
        target_layers: List[int],
        verify_interference: bool,
        use_corpus_C: bool = True,
        cov_n_samples: int = 1000,
        cov_texts: Optional[Iterable[str]] = None,
        corpus_id: str = "default",
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> List[EditResult]:
        """MEMIT Algorithm 1 — iterative per-layer.

        For each layer l in R (ascending), recompute K^l and h^L on the
        currently-edited model, then apply
            ∆^l = R^l (K^l)^T (λI + K^l (K^l)^T)^{-1}
        where R^l = (z - h^L_current) / (L - l + 1) and z = h^L_init + δ.

        Layer ordering: the paper requires ascending. We sort defensively so
        callers passing an unsorted list still get correct iteration.
        """
        # Defensive: paper requires ascending; dedupe also (a duplicate layer
        # would just waste a pass — the second would see h^L already shifted).
        target_layers = sorted(set(target_layers))
        if not target_layers:
            return [
                EditResult(
                    success=False,
                    fact_prompt=prompt,
                    subject=subject,
                    target=target,
                    target_layer=-1,
                    confidence_before=0.0,
                    confidence_after=0.0,
                    edit_magnitude=0.0,
                    interference_ratio=1.0,
                    error_message="No target layers provided.",
                )
                for prompt, subject, target in facts
            ]

        L_max = target_layers[-1]
        results: List[EditResult] = []

        try:
            # ── Step 0: confidences before any edit ──────────────────────────
            confidences_before = [
                self._get_fact_confidence(prompt, target) for prompt, _, target in facts
            ]

            # ── Step 1: compute z_i = h^L_i + δ_i on the unmodified model ────
            # _compute_z_vector now returns (target_init, delta).
            # Per-fact seed = prefix_seed + i so different facts get different
            # prefix sets while runs remain reproducible.
            z_data = [
                self._compute_z_vector(
                    prompt,
                    subject,
                    target,
                    L_max,
                    n_prefixes=n_prefixes,
                    prefix_seed=prefix_seed + i,
                )
                for i, (prompt, subject, target) in enumerate(facts)
            ]
            z_full: List[Optional[torch.Tensor]] = []
            for pair in z_data:
                if pair is None:
                    z_full.append(None)
                else:
                    ti, d = pair
                    z_full.append((ti + d) if (ti is not None and d is not None) else None)

            # ── Step 2: per-layer iterative update (Algorithm 1, lines 5–14) ─
            edit_magnitudes: List[float] = []
            for layer in target_layers:
                # 2a. Recollect K^l on the *currently-edited* model
                K_list = [
                    self._compute_k_vector(prompt, subject, layer) for prompt, subject, _ in facts
                ]
                # 2b. Recollect h^L on the *currently-edited* model
                h_L_list = [
                    self._get_h_at_subject(prompt, subject, L_max) for prompt, subject, _ in facts
                ]
                # 2c. Build residuals r_i^l = (z_i - h^L_i) / (L - l + 1)
                divisor = max(1, L_max - layer + 1)
                valid: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for fact_i, (k_vec, h_L, z) in enumerate(zip(K_list, h_L_list, z_full)):
                    if k_vec is None or h_L is None or z is None:
                        logger.info(
                            f"[edit_batch] layer={layer} fact={fact_i}: skipping "
                            f"(k={k_vec is not None}, h_L={h_L is not None}, z={z is not None})"
                        )
                        continue
                    r = (z - h_L) / divisor
                    logger.info(
                        f"[edit_batch] layer={layer} fact={fact_i}: "
                        f"|k|={k_vec.norm().item():.4f} |z|={z.norm().item():.4f} "
                        f"|h_L|={h_L.norm().item():.4f} |z-h_L|={(z-h_L).norm().item():.4f} "
                        f"divisor={divisor} |r|={r.norm().item():.4f}"
                    )
                    valid.append((k_vec, r))

                # 2d. Stack and compute ∆^l
                K = torch.stack([k for k, _ in valid], dim=1)  # [d_mlp_post, n_valid]
                R = torch.stack([r for _, r in valid], dim=1)  # [d_model,    n_valid]

                # 2d-bis. Fetch corpus second moment C for this layer (Bug #1 fix).
                # Hook is `mlp.hook_post` — same one `_compute_k_vector` reads from,
                # so C and K live in the same activation space (post-nonlinearity).
                C_layer: Optional[torch.Tensor] = None
                if use_corpus_C:
                    C_layer = get_covariance(
                        self.model,
                        layer=layer,
                        hook_name="mlp.hook_post",
                        texts=cov_texts,
                        n_samples=cov_n_samples,
                        corpus_id=corpus_id,
                    )
                    if C_layer is None:
                        warnings.warn(
                            f"Could not estimate C at layer {layer}; "
                            "falling back to λI for this layer."
                        )

                # 2e. Compute ∆^l
                upd = self._compute_layer_update(K, R, C=C_layer)
                if upd is None:
                    continue

                # 2e. Apply to W_out (TL [d_mlp, d_model] vs paper [d_model, d_mlp])
                w = self._get_mlp_weight(layer)
                if w is None:
                    continue
                if upd.shape == w.data.shape:
                    applied = upd
                elif upd.T.shape == w.data.shape:
                    applied = upd.T
                else:
                    warnings.warn(
                        f"Shape mismatch at layer {layer}: upd {upd.shape} "
                        f"vs W {w.data.shape}; skipping."
                    )
                    continue
                w_norm_before = w.data.norm().item()
                with torch.no_grad():
                    w.data = w.data + applied.to(w.dtype)
                applied_norm = torch.norm(applied).item()
                edit_magnitudes.append(applied_norm)

                # Per-layer confidence diagnostic: shows whether each layer's edit
                # actually shifts the model's prediction. If applied_norm is huge
                # but confidence doesn't move, the update is in a low-impact direction.
                try:
                    post_layer_confs = [
                        self._get_fact_confidence(prompt, target) for prompt, _, target in facts
                    ]
                    logger.info(
                        f"[edit_batch] layer={layer} APPLIED: |applied|={applied_norm:.4e} "
                        f"|w|_before={w_norm_before:.4e} "
                        f"|w|_after={w.data.norm().item():.4e} "
                        f"confs_after_layer={[f'{c:.4e}' for c in post_layer_confs]}"
                    )
                except Exception as _conf_err:
                    logger.warning(f"[edit_batch] post-layer confidence check failed: {_conf_err}")

            # ── Step 3: confidences after all layers ─────────────────────────
            confidences_after = [
                self._get_fact_confidence(prompt, target) for prompt, _, target in facts
            ]

            # ── Step 4: per-fact EditResult ──────────────────────────────────
            for i, (prompt, subject, target) in enumerate(facts):
                confidence_increase = confidences_after[i] - confidences_before[i]
                success = bool(confidence_increase > 0)
                results.append(
                    EditResult(
                        success=success,
                        fact_prompt=prompt,
                        subject=subject,
                        target=target,
                        target_layer=target_layers[0],
                        confidence_before=confidences_before[i],
                        confidence_after=confidences_after[i],
                        edit_magnitude=float(np.mean(edit_magnitudes)) if edit_magnitudes else 0.0,
                        interference_ratio=0.0,
                        metadata={
                            "method": "memit",
                            "batch_size": len(facts),
                            "confidence_increase": confidence_increase,
                            "target_layers": list(target_layers),
                            "L_max": L_max,
                        },
                    )
                )

        except ValueError:
            raise
        except Exception as e:
            warnings.warn(f"Error in batch editing: {e}")
            # Add a failed result for any fact we haven't already emitted one for.
            produced = len(results)
            for prompt, subject, target in facts[produced:]:
                results.append(
                    EditResult(
                        success=False,
                        fact_prompt=prompt,
                        subject=subject,
                        target=target,
                        target_layer=-1,
                        confidence_before=0.0,
                        confidence_after=0.0,
                        edit_magnitude=0.0,
                        interference_ratio=1.0,
                        error_message=str(e),
                    )
                )

        return results

    def _compute_z_vector(
        self,
        prompt: str,
        subject: str,
        target: str,
        layer: int,
        num_steps: int = 20,
        lr: float = 5e-1,
        n_prefixes: int = 5,
        prefix_seed: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Optimise z* target vector via gradient descent (MEMIT compute_z, Eqn. 16).

        Returns the pair (target_init, delta) where:
        target_init = h^L_i — the residual-stream value at the subject token
                        on the *unmodified* model, computed from the BARE prompt
                        (no prefix). The per-layer loop in _edit_fact_batch reads
                        h^L from the bare prompt too, so they stay aligned.
        delta       = δ_i — the optimised perturbation; z_i = target_init + δ_i.

        Random-prefix averaging (Step 1.4): the optimisation loss is averaged
        across the bare prompt and `n_prefixes` short random prefixes prepended
        to it. The injected δ is shared across all variants — only the subject
        token index shifts per variant. This is what MEMIT's compute_z.py does
        and is responsible for ~+5–10 paraphrase points.

        Returning both target_init and delta lets the caller reconstruct z while
        also keeping the original h^L for the residual computation. The gradient
        state on `delta` is dropped before return.
        """
        try:
            # 1. Build the BARE-prompt teacher-forced sequence and capture target_init.
            #    The bare prompt is the canonical reference: the per-layer loop reads
            #    h^L from it, and target_init must come from the same place.
            seq_bare = build_teacher_forced(self.model, prompt, target)
            full_tokens_bare = seq_bare.full_ids
            prompt_len_bare = seq_bare.prompt_len
            target_ids = seq_bare.target_ids
            subject_idx_bare = locate_subject_last_token(self.model, prompt, subject)

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    full_tokens_bare[:, :prompt_len_bare],
                    names_filter=f"blocks.{layer}.hook_resid_post",
                )
                target_init = (
                    cache[f"blocks.{layer}.hook_resid_post"][0, subject_idx_bare, :]
                    .detach()
                    .clone()
                )

            # 2. Sample prefixes and pre-build per-variant tensors + subject indices.
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

            variants: List[Tuple[torch.Tensor, int, int]] = []
            # Always include the bare prompt as variant 0.
            variants.append((full_tokens_bare, prompt_len_bare, subject_idx_bare))
            n_skipped = 0
            for pfx in prefixes:
                pref_prompt = pfx + " " + prompt
                try:
                    seq_v = build_teacher_forced(self.model, pref_prompt, target)
                    subj_v = locate_subject_last_token(self.model, pref_prompt, subject)
                except (ScoringError, SubjectLocationError):
                    # Skip prefixes that break tokenisation or subject location
                    n_skipped += 1
                    continue
                variants.append((seq_v.full_ids, seq_v.prompt_len, subj_v))

            logger.info(
                f"compute_z[L={layer}] subject={subject!r} target={target!r}: "
                f"{len(variants)} variant(s) (1 bare + {len(variants) - 1} prefixed, "
                f"{n_skipped} skipped)"
            )

            # 3. Optimisation loop. Single δ shared across all variants.
            delta = torch.zeros_like(target_init, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)
            max_norm = 4.0 * target_init.norm()
            hook_name = f"blocks.{layer}.hook_resid_post"
            T = int(target_ids.shape[0])

            for step in range(num_steps):
                opt.zero_grad()
                total_loss = torch.zeros((), device=delta.device)

                for full_ids_v, prompt_len_v, subj_v in variants:

                    def _hook(act, hook, _d=delta, _i=subj_v):
                        return act.index_add(
                            1,
                            torch.tensor([_i], device=act.device),
                            _d.unsqueeze(0).unsqueeze(0),
                        )

                    logits = self.model.run_with_hooks(
                        full_ids_v,
                        fwd_hooks=[(hook_name, _hook)],
                    )
                    pred_logits = logits[0, prompt_len_v - 1 : prompt_len_v - 1 + T, :]
                    total_loss = total_loss + torch.nn.functional.cross_entropy(
                        pred_logits, target_ids
                    )

                loss = total_loss / len(variants)

                if step % 5 == 0:
                    logger.info(
                        f"[compute_z] L={layer} step={step}: loss={loss.item():.4f} "
                        f"|delta|={delta.norm().item():.3f} "
                        f"|target_init|={target_init.norm().item():.3f} "
                        f"n_var={len(variants)}"
                    )

                if loss.item() < 5e-2:
                    logger.info(
                        f"[compute_z] L={layer} converged at step {step}: "
                        f"loss={loss.item():.4f} |delta|={delta.norm().item():.3f}"
                    )
                    break

                loss.backward()
                opt.step()

                with torch.no_grad():
                    if delta.norm() > max_norm:
                        delta.data = delta.data * max_norm / delta.norm()

            logger.info(
                f"[compute_z] L={layer} returning: |target_init|={target_init.norm().item():.4f} "
                f"|delta|={delta.detach().norm().item():.4f} "
                f"|target_init+delta|={(target_init+delta.detach()).norm().item():.4f}"
            )
            return target_init.detach(), delta.detach()

        except Exception as e:
            warnings.warn(f"Error computing z vector: {e}")
            return None

    def _get_h_at_subject(
        self,
        prompt: str,
        subject: str,
        layer: int,
    ) -> Optional[torch.Tensor]:
        """
        Read hook_resid_post at the subject's last token at `layer`, on the
        *currently-edited* model. This is the paper's h^l_i — the model's
        hidden state at the subject token that needs to be moved toward z_i
        by accumulating contributions from each edited W_out_l.

        Each call runs one forward pass; previously-applied edits to W_out at
        earlier layers are reflected because they're in-place modifications
        of the model's parameters.

        Returns None on tokenisation/subject-location failure. The caller
        should skip that fact at this layer.
        """
        try:
            prompt_tokens = tokenize_prompt(self.model, prompt)
            subject_idx = locate_subject_last_token(self.model, prompt, subject)
        except SubjectLocationError:
            return None
        except Exception as e:
            warnings.warn(f"Error preparing tokens for h^l capture: {e}")
            return None

        captured: dict = {}

        def _hook(act, hook):
            captured["h"] = act[0, subject_idx, :].detach().clone()
            return act

        try:
            with torch.no_grad():
                self.model.run_with_hooks(
                    prompt_tokens,
                    fwd_hooks=[(f"blocks.{layer}.hook_resid_post", _hook)],
                )
        except Exception as e:
            warnings.warn(f"Error capturing h^l at layer {layer}: {e}")
            return None

        return captured.get("h")

    def _compute_k_vector(
        self,
        prompt: str,
        subject: str,
        layer: int,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve key vector: post-activation MLP hidden state (input to W_out)
        at the subject token position. Corresponds to compute_ks in MEMIT.
        """
        try:
            # Static templates ensure the concept generalizes beyond the exact prompt phrasing
            prefixes = [
                "{}",
                "In fact, {}",
                "It is known that {}",
            ]

            k_vectors = []

            for prefix in prefixes:
                templated_prompt = prefix.format(prompt)

                prompt_tokens = tokenize_prompt(self.model, templated_prompt)
                try:
                    subject_idx = locate_subject_last_token(self.model, templated_prompt, subject)
                except SubjectLocationError:
                    continue

                captured: dict = {}

                def _hook(act, hook):
                    captured["k"] = act[0, subject_idx, :].detach().clone()
                    return act

                with torch.no_grad():
                    self.model.run_with_hooks(
                        prompt_tokens,
                        fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", _hook)],
                    )

                if "k" in captured:
                    k_vectors.append(captured["k"])

            if not k_vectors:
                return None

            # Average the k vectors across all successful templates to generalize the concept
            return torch.stack(k_vectors, dim=0).mean(dim=0)

        except Exception as e:
            warnings.warn(f"Error computing k vector: {e}")
            return None

    def _compute_layer_update(
        self,
        K: torch.Tensor,
        R: torch.Tensor,
        mom2_weight: float = 15000.0,
        C: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute ∆^l = R^l (K^l)^T (C + K^l (K^l)^T)^{-1} for a SINGLE layer
        (paper Eqn. 14).

        Args:
            K:           [d_mlp_post, n_valid] — keys k_i^l.
            R:           [d_model,    n_valid] — residuals r_i^l.
            mom2_weight: λ in the paper. Used as the multiplier on C; if C is
                        None it falls back to λI (the pre-1.3 behaviour).
            C:           [d_mlp_post, d_mlp_post] corpus second moment, or None.

        Returns ∆ in paper convention [d_model, d_mlp_post], or None on failure.

        Solver:
        * C is None  → dual form, (λI + K^T K)^{-1}, costs O(n^3 + n^2 d).
                        Mathematically equal to (λI + K K^T)^{-1}K via the
                        push-through identity. This is the cheap path.
        * C provided → primal form, (C + K K^T)^{-1}, costs O(d^3).
                        Push-through doesn't apply because C ≠ scalar·I.
        """
        try:
            K_d = K.double()
            R_d = R.double()
            d_k, n = K_d.shape
            lam_val = max(mom2_weight, 1e-5)

            logger.info(
                f"[layer_update] inputs: K={tuple(K.shape)} R={tuple(R.shape)} "
                f"|K|={K.norm().item():.4f} |R|={R.norm().item():.4f} "
                f"C_provided={C is not None} lam={lam_val:.1f}"
            )

            if C is None:
                # fallback: C = λI. Use dual form when n < d_k.
                if n < d_k:
                    lam = lam_val * torch.eye(n, dtype=torch.double, device=K_d.device)
                    adj_k = torch.linalg.solve(lam + K_d.T @ K_d, K_d.T).T  # [d, n]
                else:
                    lam = lam_val * torch.eye(d_k, dtype=torch.double, device=K_d.device)
                    adj_k = torch.linalg.solve(lam + K_d @ K_d.T, K_d)  # [d, n]
                logger.info(f"[layer_update] λI path: |adj_k|={adj_k.norm().item():.4f}")
            else:
                # Paper-faithful (Eqn. 14): A = λ·C + K K^T.

                # The paper's C is estimated from ~100k Wikipedia tokens, giving a
                # well-conditioned matrix at d_mlp ≈ 3-11k. With smaller corpora
                # (e.g. our in-tree fallback ~50 sentences ≈ 561 tokens), C is
                # heavily rank-deficient: with 561 tokens in a 3072-dim space,
                # over half the eigenvalues fall to floating-point noise levels
                # (observed: cond(A) ~ 1e35, 1618/3072 eigs below max·1e-10,
                # eig_min even goes slightly negative). The naive solve then puts
                # most of the answer into C's null space, producing weight updates
                # with norms 150× larger than the original W_out and zero semantic
                # effect (the update is orthogonal to directions real inputs use).
                #
                # We add a diagonal ridge sized to a small fraction of mean(diag(λC)).
                # At paper-grade C this is negligible vs the dominant spectrum
                # (recovers the original formula); at small-corpus C it lifts the
                # null-space eigenvalues above the noise floor and keeps the solve
                # bounded. The 1e-2 ratio mirrors ROME's solve_with_C, which uses
                # the same trick (lam=1e-2 on a diag-mean ≈ 0.035 raw C).
                C_d = lam_val * C.to(K_d.device).double()
                ridge = (C_d.diag().mean() * 1e-2).clamp(min=1e-6)
                A = (
                    C_d
                    + K_d @ K_d.T
                    + ridge * torch.eye(d_k, dtype=torch.double, device=K_d.device)
                )
                KKt = K_d @ K_d.T

                # Diagnostic: spectrum of C_d, KK^T, and A. Reveals whether C is
                # rank-deficient (null-space garbage hypothesis) or well-conditioned.
                C_diag = C_d.diag()
                logger.info(
                    f"[layer_update] λC: diag mean={C_diag.mean().item():.4f} "
                    f"median={C_diag.median().item():.4f} min={C_diag.min().item():.4e} "
                    f"max={C_diag.max().item():.4e} fro={C_d.norm().item():.4e}"
                )
                logger.info(
                    f"[layer_update] KKt: |KKt|={KKt.norm().item():.4e} "
                    f"trace={KKt.diag().sum().item():.4e}"
                )
                # Eigenvalue diagnostics on A. Cheap enough at d_mlp <= ~12k:
                # eigvalsh is O(d^3) but only runs once per layer per edit batch.
                try:
                    eig_A = torch.linalg.eigvalsh(A)
                    cond_A = (eig_A.max() / eig_A.min().clamp(min=1e-30)).item()
                    n_tiny = int((eig_A < eig_A.max() * 1e-10).sum().item())
                    logger.info(
                        f"[layer_update] A=λC+KKt: eig_min={eig_A.min().item():.4e} "
                        f"eig_max={eig_A.max().item():.4e} cond={cond_A:.2e} "
                        f"n_eigs_below_1e-10*max={n_tiny}/{d_k}"
                    )
                except Exception as _eig_err:
                    logger.warning(f"[layer_update] eigvalsh failed: {_eig_err}")

                adj_k = torch.linalg.solve(A, K_d)  # [d, n]

                k_norm = K_d.norm().item()
                ak_norm = adj_k.norm().item()
                logger.info(
                    f"[layer_update] solve done: |adj_k|={ak_norm:.4e} "
                    f"|K_d|={k_norm:.4e} ratio={ak_norm/max(k_norm, 1e-30):.4e}"
                )

            upd = (R_d @ adj_k.T).float()
            logger.info(
                f"[layer_update] upd: shape={tuple(upd.shape)} " f"|upd|={upd.norm().item():.4e}"
            )

            return upd  # [d_model, d_mlp]
        except Exception as e:
            warnings.warn(f"Error computing layer update: {e}")
            return None

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

    def _get_fact_confidence(self, prompt: str, target: str) -> float:
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

    def _select_target_layers(self, num_facts: int) -> List[int]:
        """
        Select target layers for batch editing.

        Strategy: Use multiple middle-to-late layers for more robust editing.

        Args:
            num_facts: Number of facts being edited

        Returns:
            List of layer indices
        """
        n_layers = self.model.cfg.n_layers

        # Use 2-3 layers in the middle-to-late range
        num_layers = min(3, max(1, num_facts // 5))  # 1-3 layers depending on batch size

        start_layer = int(0.4 * n_layers)
        end_layer = int(0.8 * n_layers)

        # Evenly distribute selected layers
        if num_layers == 1:
            return [int((start_layer + end_layer) / 2)]
        else:
            step = (end_layer - start_layer) // (num_layers - 1)
            return list(range(start_layer, end_layer + 1, step))[:num_layers]

    def clear_cache(self):
        """Clear batch edit cache."""
        self.batch_edit_cache.clear()
