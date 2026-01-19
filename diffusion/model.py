from omegaconf import DictConfig
from torch import Tensor, nn

from diffusion.modules import (
    AdaLNSingle,
    DiTBlock,
    PatchEmbedding,
    Reshaper,
    TimeEmbedding,
)


class NanoDiffusionModel(nn.Module):
    """Class-conditioned DiT model, widely implemented after Pixart-Alpha.

    For details about the architecture, please refer to https://arxiv.org/abs/2310.00426.
    This model leaves out some of the optimizations of the original architecture for a
    cleaner and easier-to-understand source code. This model is class-conditioned
    so it can be trained on classic datasets such as CIFAR and ImageNet.

    Directly used config vars are documented below. For config vars used by
    sub-modules, please refer to the corresponding class.

    Args:
        config: Config object

    Config vars (direct use):
        num_context_classes: Number of different classes used for conditioning.
        num_dit_blocks: Number of DiT blocks used in the model.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbedding(**config)
        self.time_embedding = TimeEmbedding(**config)
        self.adaln_single = AdaLNSingle(**config)
        if config.num_context_classes:
            self.context_embedding = nn.Embedding(
                num_embeddings=config.num_context_classes,
                embedding_dim=config.hidden_dim,
                device=config.device,
            )
        else:
            raise ValueError("`num_context_classes` is not defined.")

        self.dit_blocks = nn.ModuleList()
        for idx in range(config.num_dit_blocks):
            self.dit_blocks.append(
                DiTBlock(
                    layer_idx=idx,
                    adaln_single=self.adaln_single,
                    **config,
                )
            )

        self.reshaper = Reshaper(**config)

    def forward(self, x: Tensor, timestep: Tensor, context: Tensor):
        """Predicts the noise contained in the given latent image.

        Args:
            x: Noised latent images. Shape: (batch, channels, height, width)
            timestep: Current diffusion timestep. Shape: (batch,)
            context: Class labels used for conditioning the generation. Shape: (batch,)

        Returns:
            Predicted noise. Shape: (batch, channels, height, width)
        """

        time_embedding = self.time_embedding(timestep)
        global_adaln_params = self.adaln_single(time_embedding)
        context_embedding = self.context_embedding(context).unsqueeze(1)
        x1, x_pos, y_pos = self.patch_embedding(x, patch_pos=True)

        for dit_block in self.dit_blocks:
            x1 = dit_block(x1, global_adaln_params, context_embedding)

        return self.reshaper(x1, x_pos, y_pos)
