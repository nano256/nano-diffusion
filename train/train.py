"""Training script for Nano-Diffusion models.

This script handles the complete training pipeline:
1. Loads pre-encoded CIFAR-10 latents
2. Initializes model, optimizer, and noise scheduler
3. Sets up MLflow experiment tracking
4. Runs training via NanoDiffusionTrainer
5. Saves checkpoints and logs metrics

Usage:
    python train/train.py                   # Use config/config.yaml
    python train/train.py debug=true        # Override debug flag

Configuration is managed via Hydra from config/config.yaml. More details are
provided in the README of the repository.
"""

import sys
from pathlib import Path

import hydra
import mlflow
import torch
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent))

from diffusion.model import NanoDiffusionModel
from diffusion.noise_schedulers import (
    CosineNoiseScheduler,
    LinearNoiseScheduler,
    SigmoidNoiseScheduler,
)
from diffusion.trainer import NanoDiffusionTrainer
from diffusion.utils import get_available_device, get_kwargs, slugify

SCHEDULER_REGISTRY = {
    "linear": LinearNoiseScheduler,
    "cosine": CosineNoiseScheduler,
    "sigmoid": SigmoidNoiseScheduler,
}


def load_cifar10_latents(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"CIFAR-10 latents not found at {data_path}. "
            f"Run 'python preprocessing/encode_cifar10.py' first."
        )

    data = torch.load(data_path, map_location="cpu")

    train_latents = torch.stack(data["train"]["latents"])
    train_labels = torch.stack(data["train"]["labels"])
    test_latents = torch.stack(data["test"]["latents"])
    test_labels = torch.stack(data["test"]["labels"])

    return train_latents, train_labels, test_latents, test_labels


def create_dataloaders(
    train_latents, train_labels, test_latents, test_labels, batch_size=32
):
    train_dataset = TensorDataset(train_latents, train_labels)
    test_dataset = TensorDataset(test_latents, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../config", config_name="config")
def train(cfg):
    # This resolves any variable-based config and avoids later problems when copying the config
    OmegaConf.resolve(cfg)
    device = (
        get_available_device()
        if cfg.model.device is None
        else torch.device(cfg.model.device)
    )
    print(f"Using device: {device}")

    # Without this dataloaders with several workers throw an error.
    torch.multiprocessing.set_start_method("spawn")

    try:
        data_path = (
            "./data/cifar10_latents_debug/cifar10_latents.pt"
            if cfg.debug is True
            else "./data/cifar10_latents/cifar10_latents.pt"
        )
        train_latents, train_labels, test_latents, test_labels = load_cifar10_latents(
            data_path
        )
        print(
            f"Loaded {len(train_latents)} training samples, {len(test_latents)} test samples"
        )
        print(f"Latent shape: {train_latents[0].shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Move data to device
    train_latents = train_latents.to(device)
    train_labels = train_labels.to(device)
    test_latents = test_latents.to(device)
    test_labels = test_labels.to(device)

    train_loader, val_loader = create_dataloaders(
        train_latents,
        train_labels,
        test_latents,
        test_labels,
        batch_size=cfg.batch_size,
    )

    model = NanoDiffusionModel(cfg.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    noise_scheduler_type = SCHEDULER_REGISTRY[cfg.noise_scheduler.type]
    noise_scheduler = noise_scheduler_type(cfg.noise_scheduler)
    # Pop the optimizer type so it doesn't interfere with its own kwargs afterwards
    optimizer = getattr(torch.optim, cfg.optimizer.type)
    optimizer = optimizer(model.parameters(), **get_kwargs(cfg.optimizer))
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.type)
    lr_scheduler = lr_scheduler(optimizer, **get_kwargs(cfg.lr_scheduler))

    # Only load VAE if we do sanity check image generation throughout the run
    if cfg.trainer.validation_context is not None:
        # Load VAE to CPU to not clog up the GPU during training
        vae = AutoencoderKL.from_pretrained(cfg.model.vae_name).to("cpu").eval()

    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        run = mlflow.active_run()
        checkpoint_dir = Path(
            f"./models/{slugify(cfg.experiment_name)}/{run.info.run_name}"
        ).resolve()

        mlflow.log_params(
            {
                "epochs": cfg.epochs,
                "device": device,
                "debug": cfg.debug,
                "checkpoint_dir": str(checkpoint_dir),
            }
        )

        trainer = NanoDiffusionTrainer(
            model, noise_scheduler, checkpoint_dir, vae=vae, **cfg.trainer
        )

        print("Starting training...")
        trainer.train(
            epochs=cfg.epochs,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

    print("Training completed!")


if __name__ == "__main__":
    train()
