"""Phase 3 configuration: meta-policy training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Phase3MetaCfg:
    """Meta-policy training configuration."""
    # Network.
    text_dim: int = 384
    state_dim: int = 12
    hidden_dims: tuple[int, int] = (256, 128)

    # Training.
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    param_loss_weight: float = 0.5

    # Data augmentation.
    augment_factor: int = 10

    # Checkpoint.
    checkpoint_dir: str = "checkpoints/phase3_meta"
