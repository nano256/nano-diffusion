import re
import unicodedata

import torch
import tqdm
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader


def create_mlp(
    layer_dims: list[int],
    activation: nn.Module = nn.SiLU,
    final_activation: nn.Module | None = None,
    device: torch.device | str | None = None,
):
    """
    Create MLP from list of layer dimensions

    Args:
        layer_dims: List of dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation class (not instance) for hidden layers
        final_activation: Optional activation for final layer

    Returns:
        MLP as PyTorch module
    """
    layers = []

    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        # Add activation except for final layer (unless specified)
        if i < len(layer_dims) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())

    return nn.Sequential(*layers).to(device=device)


def slugify(value: str, allow_unicode: bool = False):
    """Slugify a string.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    Args:
        value: String to be converted
        allow_unicode: I True, preserves Unicode characters, otherwise convert to ASCII-only

    Returns:
        Slugified string
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def get_available_device():
    """Return available device for PyTorch computation.

    Returns optimal device to be used for optimal PyTorch computation.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def encode_images(dataloader: DataLoader, vae):
    """Encode PIL images to VAE-encoded latents.

    Args:
        dataloader: Dataloader containing tuples of PIL images and corresponding classes
        vae: HF-wrapped image variational autoencoder

    Returns:
        list of latents, list of corresponding classes
    """
    latent_list = []
    class_list = []

    for images, classes in tqdm(dataloader):
        with torch.no_grad():
            latents = (
                vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            )
        latent_list.extend(torch.unbind(latents))
        class_list.extend(torch.unbind(classes))

    return latent_list, class_list


def decode_latents(latents: Tensor, vae):
    """Decodes latents back to PIL images.

    Args:
        latents: Image latents, shape: (batch_size, channels, height, width)
        vae: HF-wrapped image variational autoencoder

    Returns:
        list of PIL images
    """
    vae.eval()
    with torch.no_grad():
        # Unscale before decoding
        decoded = vae.decode(latents / vae.config.scaling_factor).sample

    # # [-1, 1] -> [0, 1], clamp since values aren't guaranteed to be within [-1, 1]
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    # Pytorch convention [B, C, H, W] -> [B, H, W, C] to PIL convention
    decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()

    return [
        Image.fromarray((decoded[i] * 255).astype("uint8"))
        for i in range(decoded.shape[0])
    ]


class TensorDeviceConvertor:
    """Convert tensors to a specific device and dtype.

    Utility class for PyTorch dataset transforms. Used to move tensors
    to the correct device (CPU/GPU) and convert to the desired dtype.

    Required because lambda functions can't be pickled for DataLoader
    multiprocessing on Windows and macOS:
    https://stackoverflow.com/questions/70608810/pytorch-cant-pickle-lambda

    Args:
        device: Target device
        dtype: Target data type
    """

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.device = device
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(device=self.device, dtype=self.dtype)


class TensorCifarNormalizer:
    """Normalize CIFAR-10 images from [0, 1] to [-1, 1].

    CIFAR-10 images are normalized to [0, 1] by default. This transform
    converts them to [-1, 1] range, which is standard for diffusion models
    and matches the VAE's expected input range.

    Cannot use lambda functions due to DataLoader multiprocessing
    limitations on Windows and macOS (lambdas can't be pickled).
    """

    def __call__(self, tensor: Tensor):
        return 2.0 * tensor - 1.0  # [0,1] -> [-1,1]
