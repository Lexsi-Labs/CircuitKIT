"""
Linear Probe for Hallucination Detection

Simple linear probes trained on circuit activations to detect hallucination signals
during generation. Used as the basis for HallucinationDetector.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from ...utils.device import get_device

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """
    Simple linear probe mapping hidden states to hallucination probability.

    Maps from hidden dimension to [0, 1] probability via linear layer + sigmoid.

    Args:
        input_dim: Dimension of input activations
        dropout: Dropout probability (default 0.0)
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        """Initialize linear probe."""
        super().__init__()
        self.input_dim = input_dim

        # Linear layer: hidden_dim -> 1
        self.linear = nn.Linear(input_dim, 1)

        # Optional dropout for regularization
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: activations -> hallucination probability.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Probability tensor [..., 1] in [0, 1]
        """
        if self.dropout is not None:
            x = self.dropout(x)

        logits = self.linear(x)  # [..., 1]
        prob = torch.sigmoid(logits)  # [..., 1] in [0, 1]

        return prob

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (before sigmoid) for loss computation."""
        if self.dropout is not None:
            # Note: dropout disabled during eval, enabled during training
            x = self.dropout(x)
        return self.linear(x)


class ProbeTrainer:
    """
    Trainer for linear probes on hallucination detection.

    Handles:
    - Data loading
    - Training loop with validation
    - Metrics computation (accuracy, AUROC, loss)
    - Probe checkpointing
    """

    def __init__(
        self,
        probe: LinearProbe,
        device: Optional[str] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize probe trainer.

        Args:
            probe: LinearProbe to train
            device: Device for training ("cuda" or "cpu")
            learning_rate: Adam learning rate
            weight_decay: L2 regularization
        """
        device = device if device is not None else get_device()
        self.probe = probe.to(device)
        self.device = device
        self.optimizer = Adam(
            probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.best_model_state = None
        self.best_val_auroc = -1.0

    def train_epoch(
        self,
        train_loader,
        val_loader,
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch and validate.

        Args:
            train_loader: Training dataloader yielding (activations, labels) tuples
            val_loader: Validation dataloader

        Returns:
            (train_loss, train_auroc, val_loss, val_auroc)
        """
        # Training phase
        self.probe.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch_idx, batch in enumerate(train_loader):
            activations, labels = batch

            # Handle different batch formats
            if isinstance(activations, dict):
                if not activations:
                    continue
                key = list(activations.keys())[0]
                activations = activations[key]

            activations = activations.to(self.device)
            labels = labels.to(self.device).float()

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.probe.get_logits(activations)  # [..., 1]

            # Binary cross entropy loss
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                labels.squeeze(-1) if labels.dim() > 1 else labels,
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.probe.parameters(), 1.0)
            self.optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        # Average training loss
        train_loss /= len(train_loader)

        # Compute training AUROC
        try:
            from sklearn.metrics import roc_auc_score

            train_auroc = roc_auc_score(train_labels, train_preds)
        except Exception as e:
            logger.debug(f"Could not compute train AUROC: {e}")
            train_auroc = 0.0

        # Validation phase
        self.probe.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                activations, labels = batch

                if isinstance(activations, dict):
                    if not activations:
                        continue
                    key = list(activations.keys())[0]
                    activations = activations[key]

                activations = activations.to(self.device)
                labels = labels.to(self.device).float()

                logits = self.probe.get_logits(activations)
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),
                    labels.squeeze(-1) if labels.dim() > 1 else labels,
                )

                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).squeeze(-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= max(len(val_loader), 1)

        try:
            from sklearn.metrics import roc_auc_score

            val_auroc = roc_auc_score(val_labels, val_preds)
        except Exception as e:
            logger.debug(f"Could not compute val AUROC: {e}")
            val_auroc = 0.0

        # Save best model
        if val_auroc > self.best_val_auroc:
            self.best_val_auroc = val_auroc
            self.best_model_state = self.probe.state_dict().copy()

        return train_loss, train_auroc, val_loss, val_auroc

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 10,
        patience: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Train probe until convergence or max epochs.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            epochs: Maximum number of epochs
            patience: Early stopping patience (epochs without improvement)
            verbose: Print progress

        Returns:
            Dictionary with training history and metrics
        """
        best_val_auroc = -1.0
        patience_counter = 0
        history = {
            "train_loss": [],
            "train_auroc": [],
            "val_loss": [],
            "val_auroc": [],
        }

        for epoch in range(epochs):
            train_loss, train_auroc, val_loss, val_auroc = self.train_epoch(
                train_loader, val_loader
            )

            history["train_loss"].append(train_loss)
            history["train_auroc"].append(train_auroc)
            history["val_loss"].append(val_loss)
            history["val_auroc"].append(val_auroc)

            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"train_loss={train_loss:.4f}, train_auroc={train_auroc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_auroc={val_auroc:.4f}"
                )

            # Early stopping
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.probe.load_state_dict(self.best_model_state)
            logger.info(f"Restored best model with val_auroc={self.best_val_auroc:.4f}")

        return history

    def get_probe(self) -> LinearProbe:
        """Get trained probe."""
        return self.probe

    def get_metrics(self) -> dict:
        """Get current training metrics."""
        return {
            "best_val_auroc": self.best_val_auroc,
        }
