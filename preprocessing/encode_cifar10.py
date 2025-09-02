import torch
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import pil_to_tensor
from pathlib import Path


def encode_images(dataloader, vae):
    latent_list = []
    class_list = []
    # Convert the
    for images, classes in dataloader:
        # Encode
        with torch.no_grad():
            latents = (
                vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            )
        latent_list.append(torch.unbind(latents))
        class_list.append(torch.unbind(classes))

    return latent_list, class_list


def encode_and_save_cifar10_latents(
    vae_model_id, batch_size=128, device="cuda", dtype=torch.float16
):
    """
    Encode images to latents and save as .pt files for torch dataset loading
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    output_dir = data_dir / "cifar10_latents"
    output_dir.mkdir(exist_ok=True)
    cifar10_dir = data_dir / "cifar10"

    # Load VAE
    vae = (
        AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype).to(device).eval()
    )

    # Image transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),  # [0,1] -> [-1,1]
        ]
    )

    trainset = CIFAR10(root=cifar10_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = CIFAR10(root=cifar10_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    train_latents, train_labels = encode_images(trainloader, vae)
    test_latents, test_labels = encode_images(testloader, vae)

    data = {
        "train": {
            "latents": train_latents,
            "labels": train_labels,
        },
        "test": {"latents": test_latents, "labels": test_labels},
    }

    torch.save(data, output_dir / "cifar10_latents.pt")


if __name__ == "__main__":
    encode_and_save_cifar10_latents(vae_model_id="ostris/vae-kl-f8-d16")
