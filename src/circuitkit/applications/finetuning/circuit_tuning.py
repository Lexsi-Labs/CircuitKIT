"""Circuit-Tuning: LoRA fine-tuning restricted to circuit nodes.

Circuit-Tuning trains lightweight LoRA adapters ONLY on the weight matrices
that correspond to high-scoring circuit nodes.  All other weights are frozen.
This achieves targeted knowledge injection / correction without catastrophic
forgetting.

The key difference from vanilla LoRA:
- Adapter rank and which modules get adapters are chosen by node_scores.
- A KL-divergence retain loss is computed on random prefixes to preserve
  out-of-circuit behaviour.

Usage::

    from circuitkit.applications.finetuning.circuit_tuning import CircuitTuner, CircuitTunerConfig

    cfg = CircuitTunerConfig(lora_rank=8, lr=1e-4, n_steps=200,
                             kl_retain_weight=0.5)
    tuner = CircuitTuner(model, node_scores=scores, config=cfg)
    result = tuner.fit(prompts=["The CEO of Apple is"],
                       targets=["Bob Smith"])
    result = tuner.fit_qa(qa_pairs=[("The CEO of Apple is", "Bob Smith")])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CircuitTunerConfig:
    """Configuration for Circuit-Tuning."""

    lora_rank: int = 8
    lora_alpha: float = 16.0
    lr: float = 1e-4
    n_steps: int = 200
    batch_size: int = 4
    kl_retain_weight: float = 0.3
    n_retain_prefixes: int = 32
    top_k_layers: int = 4
    score_threshold: float = 0.0
    max_new_tokens: int = 1
    device: Optional[str] = None
    #: Set True when the finetuning prompts are already chat-templated. A chat
    #: template renders its own beginning-of-text token into the string, so
    #: tokenizing it with prepend_bos=True would inject a *second* BOS and shift
    #: every position by one. Defaults to False — raw prompts keep the legacy
    #: prepend_bos=True behavior, byte-identical for base models / "off" tasks.
    templated: bool = False


@dataclass
class CircuitTunerResult:
    """Result of a Circuit-Tuning run."""

    n_steps: int
    final_loss: float
    final_ce_loss: float
    final_kl_loss: float
    adapted_layers: List[int]
    n_lora_params: int
    wall_s: float
    loss_history: List[float] = field(default_factory=list)
    error: Optional[str] = None


class _LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around a fixed TransformerLens MLP weight matrix.

    TransformerLens MLP weights (``W_in`` / ``W_gate``) have shape
    ``(d_model, d_mlp)`` and act as ``pre = x @ W``.  The LoRA contribution is
    therefore ``x @ A @ B`` with ``A`` of shape ``(d_model, rank)`` and ``B`` of
    shape ``(rank, d_mlp)``.  ``forward`` returns this contribution so it can be
    added to the ``hook_pre`` activation, keeping the adapter parameters in the
    autograd graph (a ``.data`` weight-patch would sever gradient flow).
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float):
        super().__init__()
        d_in, d_out = weight.shape  # TL layout: (d_model, d_mlp)
        self.weight = weight  # frozen reference
        self.scaling = alpha / rank
        # Match the wrapped weight's dtype (bf16 / fp16 on reduced-precision
        # models); a default-fp32 LoRA matrix raises a dtype mismatch when the
        # contribution is added to the bf16 hook_pre activation.
        self.lora_A = nn.Parameter(
            torch.empty(d_in, rank, device=weight.device, dtype=weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, d_out, device=weight.device, dtype=weight.dtype)
        )
        nn.init.normal_(self.lora_A, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA contribution for input ``x`` of shape ``(..., d_model)``."""
        return (x @ self.lora_A @ self.lora_B) * self.scaling

    def delta(self) -> torch.Tensor:
        """Return the LoRA weight delta (d_model x d_mlp)."""
        return (self.lora_A @ self.lora_B) * self.scaling


def _top_mlp_layers(
    node_scores: Dict[str, float],
    n_layers: int,
    top_k: int,
) -> List[int]:
    layer_scores: Dict[int, float] = {}
    for name, score in node_scores.items():
        if name.startswith("MLP "):
            try:
                lyr = int(name.split()[-1])
                layer_scores[lyr] = max(layer_scores.get(lyr, 0.0), float(score))
            except (ValueError, IndexError):
                pass
    if not layer_scores:
        mid = n_layers // 2
        return list(range(max(0, mid - 1), min(n_layers, mid + top_k - 1)))
    return sorted(layer_scores, key=lambda lyr: layer_scores[lyr], reverse=True)[:top_k]


class CircuitTuner:
    """Fine-tune a model via LoRA restricted to high-scoring circuit MLP layers.

    Args:
        model: HookedTransformer in eval mode.
        node_scores: Node importance scores from circuit discovery.
        config: CircuitTunerConfig.
    """

    def __init__(
        self,
        model,
        node_scores: Optional[Dict[str, float]] = None,
        config: Optional[CircuitTunerConfig] = None,
    ):
        self.model = model
        self.node_scores = node_scores or {}
        self.cfg = config or CircuitTunerConfig()
        self.device = torch.device(
            self.cfg.device if self.cfg.device else next(model.parameters()).device
        )
        self._adapters: Dict[int, Dict[str, _LoRALinear]] = {}
        self._hooks: List[Tuple[str, callable]] = []

    def _build_adapters(self, layers: List[int]) -> int:
        """Attach LoRA adapters to W_in (and W_gate if gated MLP) for each layer."""
        # The adapter hooks read ``hook_mlp_in``; this hook point only exists
        # when use_hook_mlp_in is enabled on the model config.
        self.model.cfg.use_hook_mlp_in = True
        total_params = 0
        for lyr in layers:
            block = self.model.blocks[lyr]
            mlp = block.mlp
            adapters: Dict[str, _LoRALinear] = {}

            if hasattr(mlp, "W_in"):
                adapters["W_in"] = _LoRALinear(
                    mlp.W_in.detach(), self.cfg.lora_rank, self.cfg.lora_alpha
                ).to(self.device)
                total_params += sum(p.numel() for p in adapters["W_in"].parameters())

            if hasattr(mlp, "W_gate"):
                adapters["W_gate"] = _LoRALinear(
                    mlp.W_gate.detach(), self.cfg.lora_rank, self.cfg.lora_alpha
                ).to(self.device)
                total_params += sum(p.numel() for p in adapters["W_gate"].parameters())

            self._adapters[lyr] = adapters
        return total_params

    def _adapter_hooks(self) -> List[Tuple[str, callable]]:
        """Return TransformerLens forward hooks that add the LoRA contribution.

        The contribution is computed from the ``hook_mlp_in`` activation and
        added to ``hook_pre`` (the W_in pre-activation).  Doing this via hooks —
        rather than patching ``W_in.data`` — keeps the adapter parameters in the
        autograd graph so gradients actually flow to them during ``backward``.
        """
        cache: Dict[str, torch.Tensor] = {}
        hooks: List[Tuple[str, callable]] = []

        for lyr, adapters in self._adapters.items():
            in_name = f"blocks.{lyr}.hook_mlp_in"
            pre_name = f"blocks.{lyr}.mlp.hook_pre"

            def _store(act, hook):
                cache[hook.name] = act
                return act

            hooks.append((in_name, _store))

            def _add_lora(pre_act, hook, _l=lyr):
                mlp_in = cache[f"blocks.{_l}.hook_mlp_in"]
                adapters_l = self._adapters[_l]
                contrib = 0.0
                if "W_in" in adapters_l:
                    contrib = contrib + adapters_l["W_in"](mlp_in)
                if "W_gate" in adapters_l:
                    contrib = contrib + adapters_l["W_gate"](mlp_in)
                return pre_act + contrib

            hooks.append((pre_name, _add_lora))

        return hooks

    def _ce_loss(self, prompt: str, target: str) -> torch.Tensor:
        """Cross-entropy loss for target token given prompt (adapters applied).

        When ``self.cfg.templated`` is True the prompt is chat-templated and
        already carries the model's BOS, so it is tokenized with
        ``prepend_bos=False`` to avoid a double BOS shifting every position.
        Raw prompts keep ``prepend_bos=True`` (legacy, byte-identical behavior).
        """
        tok = self.model.tokenizer
        in_ids = self.model.to_tokens(prompt, prepend_bos=not self.cfg.templated).to(self.device)
        tgt_ids = tok(target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        tgt_id = torch.tensor([tgt_ids[0]], device=self.device)
        with self.model.hooks(fwd_hooks=self._hooks):
            logits = self.model(in_ids)[0, -1]
        return F.cross_entropy(logits.unsqueeze(0), tgt_id)

    def _kl_retain_loss(self, retain_ids_list: List[torch.Tensor]) -> torch.Tensor:
        """KL divergence between adapted and reference logits on retain samples.

        Reference logits come from the *un-adapted* model (no hooks); current
        logits are computed with the LoRA adapter hooks applied.
        """
        if not retain_ids_list:
            return torch.tensor(0.0, device=self.device)
        losses = []
        with torch.no_grad():
            ref_logits_list = [self.model(ids)[0, -1].detach() for ids in retain_ids_list]
        for ids, ref_logits in zip(retain_ids_list, ref_logits_list):
            with self.model.hooks(fwd_hooks=self._hooks):
                cur_logits = self.model(ids)[0, -1]
            kl = F.kl_div(
                F.log_softmax(cur_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="sum",
            )
            losses.append(kl)
        return torch.stack(losses).mean()

    def _sample_retain_ids(self, n: int) -> List[torch.Tensor]:
        """Sample n random token sequences for the retain loss."""
        vocab_size = self.model.cfg.d_vocab
        seq_len = 16
        out = []
        for _ in range(n):
            ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
            out.append(ids)
        return out

    def fit(
        self,
        prompts: List[str],
        targets: List[str],
    ) -> CircuitTunerResult:
        """Fit LoRA adapters to make `model(prompt)` predict `target`.

        Args:
            prompts: List of prompt strings.
            targets: Corresponding target strings (first token is the label).

        Returns:
            CircuitTunerResult with training metrics.
        """
        t0 = time.time()
        n_layers = self.model.cfg.n_layers
        layers = _top_mlp_layers(self.node_scores, n_layers, self.cfg.top_k_layers)
        n_params = self._build_adapters(layers)

        all_ada_params = [
            p
            for adapters in self._adapters.values()
            for a in adapters.values()
            for p in a.parameters()
        ]
        opt = torch.optim.AdamW(all_ada_params, lr=self.cfg.lr)

        retain_ids = self._sample_retain_ids(self.cfg.n_retain_prefixes)
        # Build the LoRA adapter hooks once; they reference self._adapters so
        # parameter updates are reflected on every subsequent forward pass.
        self._hooks = self._adapter_hooks()

        # Freeze the base model; only adapter parameters are trainable.
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in all_ada_params:
            p.requires_grad_(True)

        loss_history: List[float] = []
        final_ce = final_kl = 0.0
        error: Optional[str] = None

        n = len(prompts)
        try:
            for step in range(self.cfg.n_steps):
                opt.zero_grad()
                idx = step % n
                ce = self._ce_loss(prompts[idx], targets[idx])
                kl = (
                    self._kl_retain_loss(retain_ids[: self.cfg.batch_size])
                    if self.cfg.kl_retain_weight > 0
                    else torch.tensor(0.0)
                )
                loss = ce + self.cfg.kl_retain_weight * kl
                loss.backward()
                opt.step()

                loss_val = float(loss.detach())
                loss_history.append(loss_val)
                if step == self.cfg.n_steps - 1:
                    final_ce = float(ce.detach())
                    final_kl = float(kl.detach())
                if step % 50 == 0:
                    logger.debug(f"CircuitTuning step {step}: loss={loss_val:.4f}")
        except Exception as exc:
            error = str(exc)
            logger.warning(f"CircuitTuning failed: {exc}")

        return CircuitTunerResult(
            n_steps=self.cfg.n_steps,
            final_loss=loss_history[-1] if loss_history else 0.0,
            final_ce_loss=final_ce,
            final_kl_loss=final_kl,
            adapted_layers=layers,
            n_lora_params=n_params,
            wall_s=round(time.time() - t0, 2),
            loss_history=loss_history,
            error=error,
        )

    def fit_qa(
        self,
        qa_pairs: List[Tuple[str, str]],
    ) -> CircuitTunerResult:
        """Convenience wrapper: accept (question, answer) pairs."""
        prompts = [q for q, _ in qa_pairs]
        targets = [a for _, a in qa_pairs]
        return self.fit(prompts, targets)

    def bake(self) -> None:
        """Fold the trained LoRA adapters into the base MLP weights in-place.

        After this the model produces the adapted behaviour without any hooks.
        Only the circuit's MLP layers (``W_in`` / ``W_gate``) are modified.
        """
        with torch.no_grad():
            for lyr, adapters in self._adapters.items():
                mlp = self.model.blocks[lyr].mlp
                if "W_in" in adapters:
                    mlp.W_in.data = mlp.W_in.data + adapters["W_in"].delta()
                if "W_gate" in adapters:
                    mlp.W_gate.data = mlp.W_gate.data + adapters["W_gate"].delta()
        self._hooks = []

    def restore(self) -> None:
        """Drop all adapter hooks and adapters, leaving the base model untouched.

        The hook-based adapters never modify base weights, so simply clearing
        them fully restores the original model (unless ``bake`` was called)."""
        self._hooks = []
        self._adapters.clear()
