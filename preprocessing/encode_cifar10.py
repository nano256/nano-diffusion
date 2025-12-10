import torch
from torch.utils.data import DataLoader, Subset
from diffusers.models import AutoencoderKL
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from pathlib import Path
import typer
from tqdm import tqdm


def encode_images(dataloader, vae):
    latent_list = []
    class_list = []
    # Convert the
    for images, classes in tqdm(dataloader):
        # Encode
        with torch.no_grad():
            latents = (
                vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            )
        latent_list.extend(torch.unbind(latents))
        class_list.extend(torch.unbind(classes))

    return latent_list, class_list

# Due to problems with the DataLoaders on Windows and MacOS, we can't use 
# lambda functions in the dataset transform, hence the custom classes.
# https://stackoverflow.com/questions/70608810/pytorch-cant-pickle-lambda
class TensorDeviceConvertor:
    def __init__(self, device=None, dtype=None):
        self.device = device
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(device=self.device, dtype=self.dtype)

# The CIFAR images already are already normalized to [0,1], so we only do 
# the transform to [-1,1].
class TensorNormalizer:
    def __call__(self, tensor):
        return 2.0 * tensor - 1.0  # [0,1] -> [-1,1]

def encode_and_save_cifar10_latents(
    vae_model_id="ostris/vae-kl-f8-d16",
    batch_size: int = 128,
    device="cuda",
    dtype="float16",
    debug: bool = False,
):
    """
    Encode images to latents and save as .pt files for torch dataset loading
    """
    # typer can only pass primitives without additional
    # config, hence we solve the dtype like this.
    dtype = getattr(torch, dtype)
    # CPUs are  much slower in float16 than in float32,
    # therefore we convert the tensors at the end.
    if device == "cpu":
        final_dtype = dtype
        dtype = torch.float32

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
            TensorDeviceConvertor(device, dtype),
            TensorNormalizer(),
        ]
    )
    print("Download train dataset...")
    trainset = CIFAR10(root=cifar10_dir, train=True, download=True, transform=transform)
    print("Download test dataset...")
    testset = CIFAR10(root=cifar10_dir, train=False, download=True, transform=transform)

    if debug is True:
        trainset = Subset(trainset, torch.arange(10))
        testset = Subset(testset, torch.arange(10))

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print("Transform train dataset...")
    train_latents, train_labels = encode_images(trainloader, vae)
    print("Transform test dataset...")
    test_latents, test_labels = encode_images(testloader, vae)

    # Transform the tensors to the correct dtype before saving
    if device == "cpu":
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
    typer.run(encode_and_save_cifar10_latents)
