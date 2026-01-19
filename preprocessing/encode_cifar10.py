from pathlib import Path

import hydra
import torch
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from diffusion.utils import (
    TensorCifarNormalizer,
    TensorDeviceConvertor,
    encode_images,
    get_available_device,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def encode_and_save_cifar10_latents(cfg):
    """
    Encode images to latents and save as .pt files for torch dataset loading
    """
    device = (
        get_available_device()
        if cfg.model.device is None
        else torch.device(cfg.model.device)
    )
    print(f"Using device: {device}")

    # Without this dataloaders with several workers throw an error.
    torch.multiprocessing.set_start_method("spawn")
    # From config files, we can only pass primitives without additional
    # config, hence we solve the dtype like this.
    dtype = getattr(torch, cfg.data_dtype)
    # CPUs are much slower in float16 than in float32,
    # therefore we convert the tensors at the end.
    if device == torch.device("cpu"):
        final_dtype = cfg.dtype
        dtype = torch.float32

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    cifar10_dir = data_dir / "cifar10"
    output_dir_name = "cifar10_latents"

    # Load VAE
    vae = (
        AutoencoderKL.from_pretrained(cfg.model.vae_name, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    # Image transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            TensorDeviceConvertor(device, dtype),
            TensorCifarNormalizer(),
        ]
    )
    print("Download train dataset...")
    trainset = CIFAR10(root=cifar10_dir, train=True, download=True, transform=transform)
    print("Download test dataset...")
    testset = CIFAR10(root=cifar10_dir, train=False, download=True, transform=transform)

    if cfg.debug is True:
        trainset = Subset(trainset, torch.arange(10))
        testset = Subset(testset, torch.arange(10))
        output_dir_name = output_dir_name + "_debug"

    output_dir = data_dir / output_dir_name
    output_dir.mkdir(exist_ok=True)

    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=False, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=2
    )

    print("Transform train dataset...")
    train_latents, train_labels = encode_images(trainloader, vae)
    print("Transform test dataset...")
    test_latents, test_labels = encode_images(testloader, vae)

    # Transform the tensors to the correct dtype before saving
    if device == torch.device("cpu"):
        train_latents = list(map(lambda x: x.to(dtype=final_dtype), train_latents))
        test_latents = list(map(lambda x: x.to(dtype=final_dtype), test_latents))

    data = {
        "train": {
            "latents": train_latents,
            "labels": train_labels,
        },
        "test": {"latents": test_latents, "labels": test_labels},
    }

    torch.save(data, output_dir / "cifar10_latents.pt")


if __name__ == "__main__":
    encode_and_save_cifar10_latents()
