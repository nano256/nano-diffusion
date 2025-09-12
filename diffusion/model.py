from utils import PatchEmbedding, DiTBlock, AdaLNSingle, Reshaper, TimeEmbedding
from torch import nn


class NanoDiffusionModel(nn.Module):

    def __init__(self, model_config):
        self.model_config = model_config
        self.patch_embedding = PatchEmbedding(**model_config)
        self.time_embedding = TimeEmbedding(**model_config)
        self.adaln_single = AdaLNSingle(**model_config)
        if model_config.num_context_classes:
            self.context_embedding = nn.Embedding(
                model_config.num_context_classes,
                hidden_dim=model_config.hidden_dim,
                device=model_config.device,
            )
        else:
            raise ValueError("`num_context_classes` is not defined.")

        self.dit_blocks = []
        for idx in range(model_config.num_dit_blocks):
            self.dit_blocks.append(
                DiTBlock(layer_idx=idx, adaln_single=self.adaln_single, **model_config)
            )

        self.reshaper = Reshaper(**model_config)

    def forward(self, x, timestep, context):
        time_embedding = self.time_embedding(timestep)
        global_adaln_params = self.adaln_single(time_embedding)
        context_embedding = self.context_embedding(context)
        x1, x_pos, y_pos = self.patch_embedding(x, patch_pos=True)

        for dit_block in self.dit_blocks:
            x1 = dit_block(x1, global_adaln_params, context_embedding)

        return self.reshaper(x1, x_pos, y_pos)
