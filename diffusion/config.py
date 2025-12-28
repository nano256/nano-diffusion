from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class NanoDiffusionTrainerConfig:
    model: object = MISSING
    noise_scheduler: object = MISSING
    loss_fn: str = "mse_loss"
    validation_interval: int = 10
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 10
    keep_n_checkpoints: int = 3


@dataclass
class ModelConfig:
    patch_size: int = 2
    hidden_dim: int = 384
    num_attention_heads: int = 6
    num_dit_blocks: int = 12
    num_context_classes: int = 10
    in_channels: int = 16
    device: str = "cpu"
    num_timesteps: int = 1000
    num_attn_heads: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        # Validation logic
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
