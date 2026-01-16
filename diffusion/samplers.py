import torch


class DDIMSampler:
    def __init__(
        self,
        model,
        noise_scheduler,
        num_timesteps,
        num_sampling_steps,
        **kwargs,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps

    def step(self, x_t, noise_pred, gamma_t):
        # Euler forward step to predict x_0
        return (x_t - torch.sqrt(1 - gamma_t) * noise_pred) / torch.sqrt(gamma_t)

    def sample(self, x_T, context, return_intermediates=False, seed=None):
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
