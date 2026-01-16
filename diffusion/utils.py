import re
import unicodedata

import torch
from PIL import Image


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def decode_latents(latents, vae):
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
class TensorCifarNormalizer:
    def __call__(self, tensor):
        return 2.0 * tensor - 1.0  # [0,1] -> [-1,1]
