from torch import nn

from diffusion.modules import (
    AdaLNSingle,
    DiTBlock,
    PatchEmbedding,
    Reshaper,
    TimeEmbedding,
)


class NanoDiffusionModel(nn.Module):
    def __init__(self, config):
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

    def forward(self, x, timestep, context):
        time_embedding = self.time_embedding(timestep)
        global_adaln_params = self.adaln_single(time_embedding)
        context_embedding = self.context_embedding(context).unsqueeze(1)
        x1, x_pos, y_pos = self.patch_embedding(x, patch_pos=True)

        for dit_block in self.dit_blocks:
            x1 = dit_block(x1, global_adaln_params, context_embedding)

        return self.reshaper(x1, x_pos, y_pos)
