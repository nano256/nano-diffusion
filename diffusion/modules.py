import torch
import torch.nn.functional as F
from torch import Tensor, nn

from diffusion.utils import create_mlp


class PatchEmbedding(nn.Module):
    """Convert 2D images into sequences of patches for vision transformers.

    Divides input images into non-overlapping patches and flattens them
    into sequences, commonly used in Vision Transformer architectures.

    Args:
        patch_size: Patch size parameter.
        hidden_dim: Embedding dimension for the patches.
        in_channels: Number of channels of the input images
    """

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        in_channels: int,
        device: torch.device | str,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.lin_projection = nn.Linear(
            in_channels * patch_size**2, hidden_dim, device=device
        )

    def patchify(self, x: Tensor):
        """Convert 2D images into sequences of patches.

        Divides input images into non-overlapping patches.
        Returns patches in left-to-right, top-to-bottom order.

        Args:
            x: Batched image latents, shape: (batch_size, channels, height, width)

        Returns:
            patches, shape: (batch_size, channels * patch_size**2, patch_dim).
            x-positions, shape: (batch_size, channels * patch_size**2).
            y-positions, shape: (batch_size, channels * patch_size**2).
        """
        # Check if the latent size is evenly divisable by the patch size.
        # No padding supported.
        batch_size, channels, height, width = x.shape
        device = x.device

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
        # Generate the positional indices for the patches so we can
        # later use them for the positional encoding
        x_size = width // self.patch_size
        y_size = height // self.patch_size
        x_pos = torch.arange(x_size, device=device).repeat(y_size)
        y_pos = torch.repeat_interleave(torch.arange(y_size, device=device), x_size)
        return patches, x_pos, y_pos

    def add_positional_encodings(self, x: Tensor, x_pos: Tensor, y_pos: Tensor):
        """Adds 2D sinusoidal positional enocding onto patches.

        Args:
            x: Patch sequence, shape: (batch_size, seq_len, patch_dim)
            x_pos: x-positions of patches, shape: (batch_size, seq_len,)
            y_pos: y-positions of patches, shape: (batch_size, seq_len,)

        Returns:
            Positionally encoded patch sequence, shape: (batch_size, seq_len, patch_dim)
        """
        device = x.device
        num_patches = torch.numel(x_pos)
        # Number of dimensions of the 2D sinusoidal encoding
        pos_encoding_dim = self.hidden_dim // 4
        # Each dimension of the sinuoidal encoding contains 4 elements, but we use
        # omega separately fro the sin and cos indices, hence only 2 repetitions.
        pos_encoding_idx = torch.repeat_interleave(
            torch.arange(pos_encoding_dim, device=device), 2
        )
        # The angular frequencies of the sine-cosine positional encoding
        omega = 1 / 10 ** (4 * pos_encoding_idx / pos_encoding_dim)
        pos = torch.stack((x_pos, y_pos)).reshape(2, -1)
        # Repeat the positions so we got the indices for each dimension and transpose it
        # so that the same indices are in the last dimension and match with omega.
        pos = pos.repeat((pos_encoding_dim, 1)).t()
        # This gives us [sin(p_x * omega_0), sin(p_y * omega_0), sin(p_x * omega_1) ...]
        sin_encodings = torch.sin(pos * omega)
        # This gives us [cos(p_x * omega_0), cos(p_y * omega_0), cos(p_x * omega_1) ...]
        cos_encodings = torch.cos(pos * omega)
        # Now we can interleave the sin and cos encodings
        pos_encodings = (
            torch.stack((sin_encodings, cos_encodings), dim=1)
            .transpose(1, 2)
            .reshape(-1, num_patches, 4 * pos_encoding_dim)
        )
        return x + pos_encodings

    def forward(self, x: Tensor, patch_pos: bool = False):
        """Embeds a latent image into a sequence of tokens and adds positional embeddings.

        Args:
            x: Latent images, shape: (batch_size, channels, height, width).
            patch_pos: If true, it returns the x- & y-positions for each patch.

        Returns:
            patches, shape: (batch_size, channels * patch_size**2, patch_dim).
            x-positions, shape: (batch_size, channels * patch_size**2,).
            y-positions, shape: (batch_size, channels * patch_size**2,).
        """
        x, x_pos, y_pos = self.patchify(x)
        x = self.lin_projection(x)
        x = self.add_positional_encodings(x, x_pos, y_pos)
        if patch_pos is True:
            return x, x_pos, y_pos
        else:
            return x


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models.

    Converts integer timesteps into continuous embeddings using sinusoidal
    positional encoding, similar to the original Transformer paper but for
    scalar timestep values instead of sequence positions.

    Args:
        num_timesteps: Maximum number of diffusion timesteps.
        hidden_dim: Dimension of the embedding (must be even).
        device: Device to place the module on.
    """

    def __init__(
        self, num_timesteps: int, hidden_dim: int, device: torch.device | str, **kwargs
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.hidden_dim = hidden_dim

        half_hidden_dim = hidden_dim // 2
        freq_idx = torch.arange(half_hidden_dim).to(device)
        # Register omega so it moves device with the module
        self.register_buffer("omega", 1 / 10 ** (4 * freq_idx / half_hidden_dim))

    def forward(self, timesteps: Tensor):
        """Encode timesteps into sinusoidal embeddings.

        Args:
            timesteps: Integer timestep values, shape: (batch,)

        Returns:
            Timestep embeddings, shape: (batch, hidden_dim)
        """
        # make sure that when several timesteps are given that they have a shape of (batch, 1)
        batched_timesteps = timesteps.reshape(-1, 1)
        sin_encodings = torch.sin(batched_timesteps * self.omega)
        cos_encodings = torch.cos(batched_timesteps * self.omega)
        return torch.stack((sin_encodings, cos_encodings), dim=-1).reshape(
            -1, self.hidden_dim
        )


class AdaLNSingle(nn.Module):
    """Parameter-efficient scaling and shifting conditioned on the current timestep.

    AdaLN-single implementation from PixArt-Alpha (http://arxiv.org/abs/2310.00426).
    Infers scale and shift parameters for all DiT blocks of the model in one
    forward pass.

        Args:
            hidden_dim: Embedding size.
            num_layers: Number of DiT blocks.
            device: Device to place the module on.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_dit_blocks: int,
        device: torch.device | str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(
                hidden_dim, 6 * hidden_dim
            ),  # Global [beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2]
            nn.SiLU(),
        ).to(device=device)
        # Layer-specific learnable embeddings
        self.layer_embeddings = nn.Parameter(
            torch.zeros(num_dit_blocks, 6 * hidden_dim, device=device)
        )

    def forward(self, time_emb: Tensor):
        """Single forward pass - compute global parameters

        Args:
            time_emb: Time embeddings of current timestep, shape: (batch_size, hidden_dim)
        Returns:
            Global parameters, shape: (batch_size, 6 * hidden_dim)
        """
        global_params = self.time_mlp(time_emb)  # [batch, 6*hidden_dim]
        return global_params

    def get_layer_params(self, global_params: Tensor, layer_idx: int):
        """Get scale & shift parameters for specific layer

        Args:
            global_params: Global parameters, shape: (batch_size, 6 * hidden_dim)

        Returns:
            beta_1: pre-attention bias, shape: (batch_size, hidden_dim)
            beta_2: pre-MLP bias, shape: (batch_size, hidden_dim)
            gamma_1: pre-attention scale factor, shape: (batch_size, hidden_dim)
            gamma_2: pre-MLP scale factor, shape: (batch_size, hidden_dim)
            alpha_1: post-attention scale factor, shape: (batch_size, hidden_dim)
            alpha_2: post-MLP scale factor, shape: (batch_size, hidden_dim)
        """
        layer_emb = self.layer_embeddings[layer_idx]  # [6*hidden_dim]
        layer_params = global_params + layer_emb  # [batch, 6*hidden_dim]
        # Split them into beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2
        return torch.chunk(layer_params, 6, dim=-1)


class DiTBlock(nn.Module):
    """Diffusion transformer (DiT) block

    DiT block implementation from Pixart-Alpha (http://arxiv.org/abs/2310.00426).
    Denoises image tokens conditioned on timestep and user inputs.

        Args:
            layer_idx: Layer index.
            adaln_single: Globally shared AdaLN module.
            hidden_dim: Embedding size.
            num_attention_heads: Number of attentions head used.
            activation: Activation class (not instance) for hidden layers.
            normalization_layer: Applied PyTorch normalization method, e.g. "RMSNorm".
            dropout: Dropout rate.
            device: Device to place the module on.
    """

    def __init__(
        self,
        layer_idx: int,
        adaln_single: nn.Module,
        hidden_dim: int,
        num_attention_heads: int,
        activation: str = "SiLU",
        normalization_layer: str = "LayerNorm",
        dropout: float = 0.0,
        device: torch.device | str | None = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.adaln_single = adaln_single
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads

        self.normalization_layer = getattr(nn, normalization_layer)(
            normalized_shape=[self.hidden_dim]
        ).to(device=device)

        self.multi_head_attn = nn.MultiheadAttention(
            hidden_dim,
            num_attention_heads,
            dropout,
            batch_first=True,
            device=device,
        )
        self.feedforward = create_mlp(
            [hidden_dim, hidden_dim * 4, hidden_dim], activation, device=device
        )

    def scale_and_shift(
        self, x: Tensor, scale_factor: Tensor, bias: Tensor | None = None
    ):
        """Apply adaptive layer normalization with scale and shift.

        Normalizes the input, then applies learned scale and shift parameters
        from the timestep conditioning (AdaLN). This allows the model to modulate
        features based on the current diffusion timestep.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_dim)
            scale_factor: Scale parameter from AdaLN, shape (batch, hidden_dim)
            bias: Optional shift parameter from AdaLN, shape (batch, hidden_dim)

        Returns:
            Scaled and shifted tensor, shape (batch, seq_len, hidden_dim)
        """
        # Normalize the embeddings before scaling and shifting
        x1 = self.normalization_layer(x)
        # Transpose batch and seq since all embeddings in the same
        # batch receive the same scale and shift
        x1 = x1.transpose(0, 1)
        x1 = scale_factor * x1
        if bias is not None:
            x1 = x1 + bias
        # Transpose back so we get back to the original dimensions
        return x1.transpose(0, 1)

    def forward(self, x, global_adaln_params, c=None):
        """Forward pass of the DiT block.

        Args:
            x: Patch sequence, shape: (batch_size, seq_len, hidden_dim)
            global_adaln_params: Global scale & shift parameters, shape: (batch_size, 6*hidden_dim)
            c: Context embedding, shape: (batch_size, hidden_dim)

        Returns:
            Patch sequence, shape: (batch_size, seq_len, hidden_dim)
        """
        # Get layer-specific scale and shift params
        beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2 = (
            self.adaln_single.get_layer_params(global_adaln_params, self.layer_idx)
        )

        x1 = self.scale_and_shift(x, gamma_1, beta_1)

        if c is not None:
            # Add conditioning before the attention mechanism
            x1_conditioned = torch.cat((x1, c), dim=-2)
            attn_output, _ = self.multi_head_attn(
                x1, x1_conditioned, x1_conditioned, need_weights=False
            )
        else:
            attn_output, _ = self.multi_head_attn(x1, x1, x1, need_weights=False)

        x2 = self.scale_and_shift(attn_output, alpha_1)
        x_post_attn = attn_output + x

        x2 = self.scale_and_shift(x_post_attn, gamma_2, beta_2)
        x2 = self.feedforward(x2)
        x2 = self.scale_and_shift(x2, alpha_2)

        return x2 + x_post_attn


class Reshaper(nn.Module):
    """Latent Reshaper Module

    Reshapes a sequence of patch embeddinges and reshapes them into an image latent.

        Args:
            patch_size: Patch size parameter.
            hidden_dim: Embedding dimension for the patches.
            in_channels: Number of channels of the input images.
            normalization_layer: Applied PyTorch normalization method, e.g. "RMSNorm".
            device: Device to place the module on.
    """

    def __init__(
        self,
        patch_size: int,
        hidden_dim: int,
        in_channels: int,
        normalization_layer: str = "LayerNorm",
        device: torch.device | None = None,
        **kwargs,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        self.normalization_layer = getattr(nn, normalization_layer)(
            normalized_shape=[self.hidden_dim]
        ).to(device=device)

        self.lin_projection = nn.Linear(
            hidden_dim, in_channels * patch_size**2, device=device
        )

    def forward(self, x: Tensor, x_pos: Tensor, y_pos: Tensor):
        """Forward pass of the Reshaper.

        Args:
            x: Patch sequence of image latent, shape: (batch_size, seq_len, hidden_dim)
            x_pos: X-axis positions of the patch embeddings, shape: (batch_size, seq_len,)
            y_pos: Y-axis positions of the patch embeddings, shape: (batch_size, seq_len,)

        Returns:
            image latents, shape: (batch, channels, height, width)
        """
        width = (torch.max(x_pos) + 1) * self.patch_size
        height = (torch.max(y_pos) + 1) * self.patch_size
        x1 = self.normalization_layer(x)
        # Get the embeddings back to their original dimensions
        x1 = self.lin_projection(x1)
        # Transpose so that we have the dim order (batch, embedding, seq)
        x1 = x1.transpose(-1, -2)
        return F.fold(
            x1,
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
