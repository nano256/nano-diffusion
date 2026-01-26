from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn


class AbstractNoiseScheduler(nn.Module, ABC):
    """Abstract diffusion noise scheduler

    Abstract class for noise schedules implementing the forward diffusion process.
    The implementations are taken from http://arxiv.org/abs/2301.10972.
    """

    DEFAULT_CONFIG = {
        "clip_min": 1e-9,
        "scale_factor": 1.0,
    }

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = OmegaConf.merge(self.DEFAULT_CONFIG, config)

    @abstractmethod
    def gamma_func(self, timesteps: Tensor):
        """Calculates the noise factors for different timesteps.

        Args:
            timesteps: Diffusion timesteps, shape: (batch,)

        Returns:
            noise factors, shape: (batch,)
        """
        pass

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
        return_noise: bool = True,
    ):
        """Noises image latents according to noise schedule.

        Args:
            x: Image latents, shape: (batch, channels, height, width)
            timesteps: Current sampling timesteps, shape: (batch,)
            noise: Initial noise. Randomly generated if not passed, shape: (batch, channels, height, width)
            return_noise: If true, also returns noise tensor.

        Returns:
            noised latents, shape: (batch, channels, height, width)
        """
        device = x.device
        # Make sure that the timesteps are broadcasted correctly
        # so the element-wise operation on the latents work.
        batched_timesteps = timesteps.reshape(-1, 1, 1, 1).float()
        # Normalize the timesteps
        batched_timesteps = batched_timesteps / self.config.num_timesteps
        gamma = self.gamma_func(batched_timesteps)

        if noise is None:
            noise = torch.randn_like(x, device=device)

        if return_noise is True:
            return (
                torch.sqrt(gamma) * self.config.scale_factor * x
                + torch.sqrt(1 - gamma) * noise,
                noise,
            )
        else:
            return (
                torch.sqrt(gamma) * self.config.scale_factor * x
                + torch.sqrt(1 - gamma) * noise
            )


class LinearNoiseScheduler(AbstractNoiseScheduler):
    """Linear diffusion noise scheduler

    Generates a linear noise schedule for the forward diffusion process.

    Config args:
        num_timesteps: Diffusion timesteps, shape (batch,)
        scale_factor: Scale factor on the image during noising
        clip_min: Minimal return value, for numeric stability purposes.
    """

    def gamma_func(self, timesteps):
        # A gamma function that simply is 1-t, timesteps in [0, 1] (normalized)
        return torch.clip(1 - timesteps, self.config.clip_min, 1.0)


class CosineNoiseScheduler(AbstractNoiseScheduler):
    """Cosine diffusion noise scheduler

    Generates a cosine noise schedule for the forward diffusion process.

    Config args:
        num_timesteps: Diffusion timesteps, shape (batch,)
        scale_factor: Scale factor on the image during noising
        clip_min: Minimal return value, for numeric stability purposes
        start: Interpolation start
        end: Interpolation end
        tau: Scale factor
    """

    DEFAULT_CONFIG = {
        "start": 0.2,
        "end": 1.0,
        "tau": 2.0,
        "scale_factor": 1.0,
        "clip_min": 1e-9,
    }

    def gamma_func(self, timesteps: Tensor):
        device = timesteps.device
        # Make sure the parameters are on the same device like the timesteps
        start = torch.tensor(self.config.start, device=device)
        end = torch.tensor(self.config.end, device=device)
        tau = torch.tensor(self.config.tau, device=device)
        # A gamma function based on cosine function, timesteps in [0, 1] (normalized)
        v_start = torch.cos(start * torch.pi / 2) ** (2 * tau)
        v_end = torch.cos(end * torch.pi / 2) ** (2 * tau)
        output = torch.cos((timesteps * (end - start) + start) * torch.pi / 2) ** (
            2 * tau
        )
        output = (v_end - output) / (v_end - v_start)
        return torch.clip(output, self.config.clip_min, 1.0)


class SigmoidNoiseScheduler(AbstractNoiseScheduler):
    """Sigmoid diffusion noise scheduler

    Generates a sigmoid noise schedule for the forward diffusion process.

    Config args:
        num_timesteps: Diffusion timesteps, shape (batch,)
        scale_factor: Scale factor on the image during noising
        clip_min: Minimal return value, for numeric stability purposes
        start: Interpolation start
        end: Interpolation end
        tau: Scale factor
    """

    DEFAULT_CONFIG = {
        "start": 0.0,
        "end": 3.0,
        "tau": 0.7,
        "scale_factor": 1.0,
        "clip_min": 1e-9,
    }

    def gamma_func(self, timesteps: Tensor):
        device = timesteps.device
        # Make sure the parameters are on the same device like the timesteps
        start = torch.tensor(self.config.start, device=device)
        end = torch.tensor(self.config.end, device=device)
        tau = torch.tensor(self.config.tau, device=device)
        # A gamma function based on sigmoid function, timesteps in [0, 1] (normalized)
        v_start = F.sigmoid(start / tau)
        v_end = F.sigmoid(end / tau)
        output = F.sigmoid((timesteps * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start)
        return torch.clip(output, self.config.clip_min, 1.0)
