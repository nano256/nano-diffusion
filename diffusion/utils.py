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
        - Output: (batch_size, seq_len, hidden_dim)
    """

    def __init__(self, patch_size, hidden_dim, in_channels):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.lin_projection = nn.Linear(in_channels * patch_size**2, hidden_dim)
        # Number of dimensions of the 2D sinusoidal encoding
        self.pos_encoding_dim = hidden_dim // 4
        # Each dimension of the sinuoidal encoding contains 4 elements, hence the repeating indices
        pos_encoding_idx = torch.repeat_interleave(
            torch.arange(self.pos_encoding_dim), 4
        )
        # The angular frequencies of the sine-cosine positional encoding
        self.omega = 1 / 10 ** (4 * pos_encoding_idx / self.pos_encoding_dim)

    def patchify(self, x):
        """Convert 2D images into sequences of patches.

        Divides input images into non-overlapping patches.
        Returns patches in left-to-right, top-to-bottom order.

        Args:
            x (torch.tensor): Spacial input of shape (batch_size, channels, height, width).

        Shape:
            - Input: (batch_size, channels, height, width)
            - Output: (batch_size, height // patch_size, width // patch_size, patch_dim)
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
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        # Transpose them so they are in the right shape for further processing
        # (B,  num_patches, C*patch_size*patch_size)
        patches = patches.transpose(-1, -2)
        # Bring them in a form so we can add the 2D positional encoding later
        patches = x.reshape(
            batch_size, self.patch_size // height, self.patch_size // width, -1
        )
        return patches

    def add_positional_encodings(self, x):
        # y_dim traverses the patches along the height and x_dim along the width
        batch_size, y_dim, x_dim = x.shape

    def forward(self, x):
        x = self.patchify(x)
        x = self.lin_projection(x)
        return x


class AdaLNSingle(nn.Module):
    """Parameter-efficient scaling and shifting conditioned on the current timestep

    AdaLN-single implementation from PixArt-Alpha (http://arxiv.org/abs/2310.00426).
    Infers scale and shift parameters for all DiT blocks of the model in one
    forward pass.

        Args:
            hidden_dim (int): Embedding size.
            num_layers (int): Number of DiT blocks.

        Shape:
            - Input: (batch_size, hidden_dim)
            - Output: (batch_size, 6 * hidden_dim)
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        # TODO: Implement AdaLN zero initialization
        self.time_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim, 6 * hidden_dim
            ),  # Global [beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2]
            nn.SiLU(),
        )
        # Layer-specific learnable embeddings
        self.layer_embeddings = nn.Parameter(torch.zeros(num_layers, 6 * hidden_dim))

    def forward(self, time_emb):
        """Single forward pass - compute global parameters"""
        global_params = self.time_mlp(time_emb)  # [batch, 6*hidden_dim]
        return global_params

    def get_layer_params(self, global_params, layer_idx):
        """Get parameters for specific layer"""
        layer_emb = self.layer_embeddings[layer_idx]  # [6*hidden_dim]
        layer_params = global_params + layer_emb  # [batch, 6*hidden_dim]
        # Split them into beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2
        return torch.chunk(layer_params, 6, dim=-1)


def create_mlp(layer_dims, activation=nn.SiLU, final_activation=None):
    """
    Create MLP from list of layer dimensions

    Args:
        layer_dims: List of dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation class (not instance) for hidden layers
        final_activation: Optional activation for final layer
    """
    layers = []

    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        # Add activation except for final layer (unless specified)
        if i < len(layer_dims) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation())

    return nn.Sequential(*layers)


def create_dit_block_config():
    """Config creator for DiTBlock"""
    return {
        "hidden_dim": 256,
        "num_attn_heads": 8,
        "attn_dim": 32,
        "dropout": None,
    }


class DiTBlock(nn.Module):
    """Diffusion transformer (DiT) block

    DiT block implementation from Pixart-Alpha (http://arxiv.org/abs/2310.00426).
    Denoises image tokens conditioned on timestep and user inputs.

        Args:
            layer_idx (int): Layer index
            adaln_single (torch.nn.Module): Globally shared AdaLN module
            hidden_dim (int): Embedding size.
            attn_dim (int): Dimension of the the q, k embeddings used in the attention mechanism
            dropout (float): Dropout rate

        Shape:
            - Input: (batch_size, seq_len, hidden_dim)
            - Output: (batch_size, seq_len, hidden_dim)
    """

    def __init__(
        self, layer_idx, adaln_single, hidden_dim, num_attn_heads, attn_dim, dropout
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.adaln_single = adaln_single
        self.dropout = dropout
        self.num_attn_heads = num_attn_heads

        self.multi_head_attn = nn.MultiheadAttention(
            attn_dim, num_attn_heads, dropout, batch_first=True
        )
        self.feedforward = create_mlp([hidden_dim, hidden_dim * 4, hidden_dim], nn.SiLU)

    def forward(self, x, global_adaln_params):
        # Get layer-specific scale and shift params
        beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2 = (
            self.adaln_single.get_layer_params(global_adaln_params, self.layer_idx)
        )

        # Normalize the embeddings before scaling and shifting
        x1 = F.layer_norm(x, x.shape[-1])
        x1 = gamma_1 * x1 + beta_1

        attn_output = self.multi_head_attn(x1, x1, x1)
        attn_output = alpha_1 * attn_output
        x_post_attn = attn_output + x

        x2 = F.layer_norm(x_post_attn, x_post_attn.shape[-1])
        x2 = gamma_2 * x2 + beta_2
        x2 = self.feedforward(x2)
        x2 = alpha_2 * x2

        return x2 + x_post_attn
