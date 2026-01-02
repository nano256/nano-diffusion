import sys
from pathlib import Path

import hydra
import mlflow
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent))

from diffusion.model import NanoDiffusionModel
from diffusion.trainer import NanoDiffusionTrainer
from diffusion.utils import CosineNoiseScheduler, slugify


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
    if cfg.model.device is None:
        if torch.cuda.is_available():
            cfg.model.device = "cuda"
        elif torch.backends.mps.is_available():
            cfg.model.device = "mps"
        else:
            cfg.model.device = "cpu"

    print(f"Using device: {cfg.model.device}")
    device = torch.device(cfg.model.device)

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

    noise_scheduler = CosineNoiseScheduler(num_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        run = mlflow.active_run()
        checkpoint_dir = Path(
            f"./checkpoints/{slugify(cfg.experiment_name)}/{run.info.run_name}"
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
            model, noise_scheduler, checkpoint_dir, **cfg.trainer
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
