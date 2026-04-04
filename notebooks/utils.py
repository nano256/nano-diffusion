from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


def convert_image_to_normalized_tensor(img):
    t = to_tensor(img)
    mean = t.mean()
    std = t.std()
    return (t - mean) / std, mean, std


def convert_normalized_tensors_to_images(t, mean, std):
    img = torch.clip((t * std + mean), 0, 1)

    if len(img.shape) <= 3:
        return [to_pil_image(img)]

    img_list = []
    for i in range(img.shape[0]):
        img_list.append(to_pil_image(img[i]))
    return img_list


def get_picsum_image(id, resolution=256):
    image_url = f"https://picsum.photos/id/{id}/{resolution}/{resolution}"
    return Image.open(BytesIO(requests.get(image_url).content))
