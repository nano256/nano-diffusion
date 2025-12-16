import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent))

import typer

from diffusion.model import ModelConfig, NanoDiffusionModel
from diffusion.trainer import NanoDiffusionTrainer
from diffusion.utils import CosineNoiseScheduler


def load_cifar10_latents(
    data_path="./data/cifar10_latents_debug/cifar10_latents.pt",
    # data_path="./data/cifar10_latents/cifar10_latents.pt"
):
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


def train(experiment_name=None, device: str = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_cifar10"

    print(f"Using device: {device}")
    device = torch.device(device)

    # Without this dataloaders with several workers throw an error.
    torch.multiprocessing.set_start_method("spawn")

    try:
        train_latents, train_labels, test_latents, test_labels = load_cifar10_latents()
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
        train_latents, train_labels, test_latents, test_labels, batch_size=32
    )

    config = ModelConfig(device=device)
    model = NanoDiffusionModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    noise_scheduler = CosineNoiseScheduler(num_timesteps=1000)

    trainer = NanoDiffusionTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        validation_interval=5,
        save_every_n_epochs=10,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    print("Starting training...")
    trainer.train(
        epochs=2,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        experiment_name="cifar10_minimal",
    )

    print("Training completed!")


if __name__ == "__main__":
    typer.run(train)
