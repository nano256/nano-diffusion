import copy

import torch


class EMAModel:
    def __init__(self, model, decay, warmup_steps):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model.requires_grad_(False)

    def _effective_decay(self):
        # ramps from ~0 to self.decay over warmup_steps
        return min(self.decay, (1 + self.step) / (self.warmup_steps + self.step))

    @torch.no_grad()
    def update(self, model):
        decay = self._effective_decay()
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.lerp_(param.data, 1 - decay)
        self.step += 1

    def state_dict(self):
        return {
            "ema_model": self.model.state_dict(),
            "step": self.step,
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["ema_model"])
        self.step = state_dict["step"]
