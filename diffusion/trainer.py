from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from diffusion.noise_schedulers import AbstractNoiseScheduler
from diffusion.samplers import DDIMSampler
from diffusion.utils import decode_latents


class NanoDiffusionTrainer:
    """Trainer for diffusion models with MLflow logging and checkpointing.

    Handles the complete training loop including loss computation, validation,
    model checkpointing, and optional image generation for monitoring. Integrates
    with MLflow for experiment tracking.

    Args:
        model: The diffusion model to train
        noise_scheduler: Noise scheduler for the forward diffusion process
        checkpoint_dir: Directory to save model checkpoints
        num_sampling_steps: Number of steps for DDIM sampling during validation
        loss_fn: Loss function name from torch.nn.functional
        validation_interval: Run validation every N epochs
        save_every_n_epochs: Save checkpoint every N epochs
        keep_n_checkpoints: Number of recent checkpoints to keep
        vae: Optional VAE for decoding latents to images during validation
        validation_context: Optional list of class labels to generate during validation

    """

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: AbstractNoiseScheduler,
        checkpoint_dir: str | Path,
        num_sampling_steps: int,
        loss_fn: str = "mse_loss",
        validation_interval: int = 10,
        save_every_n_epochs: int = 10,
        keep_n_checkpoints: int = 3,
        vae=None,
        validation_context: list[int] | None = None,
        fixed_validation_noise: bool = False,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.num_sampling_steps = num_sampling_steps
        self.loss_fn = getattr(F, loss_fn)
        self.validation_interval = validation_interval
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_n_checkpoints = keep_n_checkpoints
        self.vae = vae
        # Context is provided in classes, hence the direct conversion
        self.validation_context = (
            None if validation_context is None else torch.tensor(validation_context)
        )
        self.fixed_validation_noise = fixed_validation_noise

        self.best_val_loss = float("inf")
        self.saved_checkpoints = []  # Track saved checkpoint paths
        self.best_checkpoint = None

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(self, batch: tuple[Tensor]):
        """Compute diffusion training loss for a batch.

        Args:
            batch: Tuple of (latents, context), shapes: (B, C, H, W) and (B,)

        Returns:
            MSE loss between predicted and actual noise
        """
        latents, context = batch
        latents = latents.to(dtype=torch.float32)
        device = latents.device
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_timesteps + 1,
            (latents.shape[0],),
            device=device,
        )
        noise = torch.randn_like(latents, device=device)
        noised_latents = self.noise_scheduler(
            latents, timesteps, noise, return_noise=False
        )
        pred_noise = self.model(noised_latents, timesteps, context)
        loss = self.loss_fn(pred_noise, noise)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        if torch.isinf(loss):
            raise ValueError("Loss is inf")

        return loss

    def save_checkpoint(
        self,
        epoch: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        loss: Tensor,
        is_best: bool = False,
    ):
        """Save model checkpoint with optional cleanup of old checkpoints.

        Args:
            epoch: Current epoch number
            optimizer: Optimizer state to save
            lr_scheduler: LR scheduler state to save
            loss: Current loss value
            is_best: If True, save as best model (doesn't count toward keep_n_checkpoints)

        Returns:
            Path to the saved checkpoint file
        """
        checkpoint = {
            "epoch": epoch,
            "model_config": self.model.config,
            "model_state_dict": self.model.state_dict(),
            "noise_scheduler_config": self.noise_scheduler.config,
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "loss": loss,
            "best_val_loss": self.best_val_loss,
        }

        if is_best:
            checkpoint_path = self.checkpoint_dir / f"best_model_epoch_{epoch:04d}.pt"
            torch.save(checkpoint, checkpoint_path)
            if self.best_checkpoint is not None and self.best_checkpoint.exists():
                self.best_checkpoint.unlink()
            self.best_checkpoint = checkpoint_path
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.saved_checkpoints.append(checkpoint_path)
            # Remove old checkpoints if we exceed the limit
            if len(self.saved_checkpoints) > self.keep_n_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        optimizer: Optimizer | None = None,
        lr_scheduler: LRScheduler | None = None,
    ):
        """Load training checkpoint from file.

        Args:
            checkpoint_path: Directory path to checkpoint file
            optimizer: Optimizer that will be loaded
            lr_scheduler: LR scheduler that will be loaded

        Returns:
            checkpoint epoch, checkpoint loss
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return checkpoint["epoch"], checkpoint["loss"]

    def train(
        self,
        epochs: int,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
    ):
        """Execute the training loop.

        Runs training for the specified number of epochs, logging training and
        validation metrics in MLflow, checkpointing, and optional image generation
        for validation purposes.

        Args:
            epochs: Number of epochs to train
            optimizer: PyTorch optimizer
            lr_scheduler: Learning rate scheduler
            train_dataloader: DataLoader for training data, providing (latents, context) tuples
            val_dataloader: Optional DataLoader for validation data
        """
        latent_shape = None
        validation_noise = None

        for epoch in range(epochs):
            num_prev_batches = epoch * len(train_dataloader)
            for batch_idx, batch in enumerate(train_dataloader):
                # Grab the latent shape for image generation sanity checks later
                if latent_shape is None:
                    latents, _ = batch
                    latent_shape = latents.shape[1:]
                step = num_prev_batches + batch_idx
                optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                optimizer.step()

                # Log everything. Assumes mlflow run is already started, if not it
                # just starts an unlabeled run.
                # Detach loss tensor first to suppress scalar conversion warning.
                mlflow.log_metrics(
                    {
                        "loss_train": loss.detach().item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step,
                )

            # Validation and checkpointing
            avg_val_loss = None
            if val_dataloader is not None and epoch % self.validation_interval == 0:
                with torch.no_grad():
                    val_losses = []
                    for batch in val_dataloader:
                        val_losses.append(self.compute_loss(batch).item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    mlflow.log_metric("loss_val", avg_val_loss, step)

                    # Save best model if validation loss improved
                    if avg_val_loss < self.best_val_loss and epoch != 0:
                        self.best_val_loss = avg_val_loss
                        self.save_checkpoint(
                            epoch,
                            optimizer,
                            lr_scheduler,
                            avg_val_loss,
                            is_best=True,
                        )

            # Log some image generations as sanity check
            if (
                self.validation_context is not None
                and self.vae is not None
                and epoch % self.validation_interval == 0
            ):
                model_device = next(self.model.parameters()).device
                vae_device = next(self.vae.parameters()).device
                self.model.eval()
                with torch.no_grad():
                    sampler = DDIMSampler(
                        self.model, self.noise_scheduler, 1000, self.num_sampling_steps
                    )
                    # Generates new initial noise every generation if
                    # fixed_validation_noise is False, otherwise keeps the
                    # same for every generation throughout the training run.
                    if validation_noise is None or self.fixed_validation_noise is False:
                        validation_noise = torch.randn(
                            self.validation_context.shape[0], *latent_shape
                        ).to(model_device)

                    latents = sampler.sample(
                        validation_noise, self.validation_context.to(model_device)
                    ).to(vae_device)

                    images = decode_latents(latents, self.vae)
                    for image, context in zip(images, self.validation_context):
                        mlflow.log_image(
                            image,
                            key=f"{context.item()}",
                            step=step,
                        )
                self.model.train()

            # Save regular checkpoint every N epochs
            if epoch % self.save_every_n_epochs == 0 or epoch == epochs - 1:
                current_loss = (
                    avg_val_loss if avg_val_loss is not None else loss.detach().item()
                )
                self.save_checkpoint(epoch, optimizer, lr_scheduler, current_loss)

            lr_scheduler.step()
