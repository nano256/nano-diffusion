import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Convert 2D images into sequences of patches for vision transformers.

    Divides input images into non-overlapping patches and flattens them
    into sequences, commonly used in Vision Transformer architectures.

    Args:
        patch_size (int): Patch size parameter.
        hidden_dim (int): Embedding dimension for the patches.
        in_channels (int): Number of channels of the input images

    Shape:
        - Input: (batch_size, channels, height, width)
        - Output: (batch_size, num_patches, patch_dim)
    """

    def __init__(self, patch_size, hidden_dim, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.lin_projection = nn.Linear(in_channels * patch_size**2, hidden_dim)

    def patchify(self, x):
        """Convert 2D images into sequences of patches.

        Divides input images into non-overlapping patches.
        Returns patches in left-to-right, top-to-bottom order.

        Args:
            x (torch.tensor): Spacial input of shape (batch_size, channels, height, width).

        Shape:
            - Input: (batch_size, channels, height, width)
            - Output: (batch_size, num_patches, patch_dim)
        """
        # Check if the latent size is evenly divisable by the patch size.
        # No padding supported.
        batch_size, channels, height, width = x.shape

        if height % self.patch_size != 0:
            raise ValueError(
                f"Dimension mismatch: expected height ({height}) being evenly divisible by patch size ({self.patch_size})."
            )

        if width % self.patch_size != 0:
            raise ValueError(
                f"Dimension mismatch: expected width ({width}) being evenly divisible by patch size ({self.patch_size})."
            )

        # Extract patches: (B, C*patch_size*patch_size, num_patches)
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        # Transpose them so they are in the right shape for further processing
        x = x.transpose(-1, -2)
        return x

    def forward(self, x):
        x = self.patchify(x)
        x = self.lin_projection(x)
        return x
