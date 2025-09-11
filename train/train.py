#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from diffusion.model import NanoDiffusionModel
from diffusion.trainer import NanoDiffusionTrainer
from diffusion.utils import CosineNoiseScheduler


class ModelConfig:
    def __init__(self):
        self.patch_size = 2
        self.hidden_dim = 384
        self.num_attention_heads = 6
        self.num_dit_blocks = 12
        self.num_context_classes = 10
        self.mlp_ratio = 4.0


def load_cifar10_latents(data_path="./data/cifar10_latents/cifar10_latents.pt"):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    config = ModelConfig()
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
        epochs=100,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        experiment_name="cifar10_minimal",
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
