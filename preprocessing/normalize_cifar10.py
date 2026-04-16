"""Transform the CIFAR-10 dataset into a normalized form.

This preprocessing step normalizes CIFAR-10 images.

Usage:
    python preprocessing/normalize_cifar10.py             # Full dataset
    python preprocessing/normalize_cifar10.py debug=True  # Small subset for testing

Output:
    - data/cifar10_normalized/cifar10_normalized.pt (full dataset)
    - data/cifar10_normalized_debug/cifar10_normalized.pt (debug subset)

The output file contains:
    {
        'train': {'data': Tensor, 'labels': Tensor},
        'test': {'data': Tensor, 'labels': Tensor}
    }
"""

from pathlib import Path

import hydra
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from diffusion.utils import (
    TensorCifarNormalizer,
    TensorDtypeConvertor,
    get_available_device,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def normalize_and_save_cifar10(cfg):
    """
    Normalize images and save as .pt files for torch dataset loading
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
    dtype = getattr(torch, cfg.data.dtype)

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    cifar10_dir = data_dir / "cifar10"
    output_dir_name = "cifar10_normalized"

    # Image transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            TensorCifarNormalizer(),
            TensorDtypeConvertor(dtype),
        ]
    )

    print("Download train dataset...")
    trainset = CIFAR10(root=cifar10_dir, train=True, download=True, transform=transform)
    print("Download test dataset...")
    testset = CIFAR10(root=cifar10_dir, train=False, download=True, transform=transform)

    if cfg.data.augment is True:
        output_dir_name = output_dir_name + "_augmented"

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

    train_images = []
    train_labels = []
    print("Transform train dataset...")
    for images, labels in trainloader:
        train_images.append(images)
        train_labels.append(labels)
        if cfg.data.augment is True:
            train_images.append(images.flip(-1))
            train_labels.append(labels)

    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    test_images = []
    test_labels = []
    print("Transform test dataset...")
    for images, labels in testloader:
        test_images.append(images)
        test_labels.append(labels)
        if cfg.data.augment is True:
            test_images.append(images.flip(-1))
            test_labels.append(labels)

    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)

    data = {
        "train": {
            "data": train_images,
            "labels": train_labels,
        },
        "test": {"data": test_images, "labels": test_labels},
    }

    torch.save(data, output_dir / "data.pt")
    print("CIFAR10 preprocessing completed!")


if __name__ == "__main__":
    normalize_and_save_cifar10()
