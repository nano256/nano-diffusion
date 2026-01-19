import torch
from torch import Tensor, nn

from diffusion.noise_schedulers import AbstractNoiseScheduler


class DDIMSampler:
    """DDIM sampler for the forward diffusion process

    Denoising Diffusion Implicit Model sampler described in https://arxiv.org/abs/2010.02502.

    Args:
        model: Diffusion model
        noise_scheduler: Noise scheduler
        num_timesteps: Maximum number of diffusion timesteps
        num_sampling_steps: Number of sampling steps
    """

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: AbstractNoiseScheduler,
        num_timesteps: int,
        num_sampling_steps: int,
        **kwargs,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps

    def step(self, x_t: Tensor, noise_pred: Tensor, gamma_t: Tensor):
        """Predict latent from noised latent and noise prediction.

        Args:
            x_t: noised latent images, shape: (batch_size, channels, height, width)
            noise_pred: Noise predictions, shape: (batch_size, channels, height, width)
            gamma_t: Noise factor, shape: (batch,)

        Returns:
            predicted denoised latent, Shape: (batch_size, channels, height, width)
        """
        # Euler forward step to predict x_0
        return (x_t - torch.sqrt(1 - gamma_t) * noise_pred) / torch.sqrt(gamma_t)

    def sample(
        self,
        x_T: Tensor,
        context: Tensor,
        return_intermediates: bool = False,
    ):
        """Generate image latents from noise and context.

        Denoises and renoises latents over a number of sampling steps conditioned
        on provided context.

        Args:
            x_T: Latent-shaped noise, shape: (batch_size, channels, height, width)
            context: Class labels, shape: (batch,)
            return_intermediates: If true, also returns the intermediate sampling steps.

        Returns:
         Final sample, shape: (batch_size, channels, height, width)
         list of intermediate samples, list[Tensors] each with shape (batch_size, channels, height, width)
        """
        intermediates = []
        # x_T is assumed to be full noise. It also determines the shape of the generated image.
        x_t = x_T
        timesteps = (
            torch.linspace(0, self.num_timesteps, self.num_sampling_steps)
            .flip(0)
            .round()
            .long()
            .to(x_T.device)
        )
        # The gamma function needs normalized timestepp
        gamma_t = self.noise_scheduler.gamma_func(timesteps / self.num_timesteps)

        for idx in range(len(timesteps) - 1):
            # Predict the noise
            noise_pred = self.model(x_t, timesteps[idx], context)
            # Predict x_0 from it
            x_0_pred = self.step(x_t, noise_pred, gamma_t[idx])
            # Add noise again equivalent of the noise step from which we will do the next prediction
            x_t = (
                torch.sqrt(gamma_t[idx + 1]) * x_0_pred
                + torch.sqrt(1 - gamma_t[idx + 1]) * noise_pred
            )
            if return_intermediates:
                intermediates.append(x_t.detach().clone())

        noise_pred = self.model(x_t, timesteps[-1], context)
        x_0 = self.step(x_t, noise_pred, gamma_t[-1])

        if return_intermediates:
            return x_0, intermediates

        return x_0
