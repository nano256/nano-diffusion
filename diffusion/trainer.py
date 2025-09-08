import torch
import torch.nn.functional as F


class NanoDiffusionTrainer:
    """Trainer for the Nano Diffusion model

    TODO:
    Add all administrative stuff in the Trainer class, e.g. logging, saving model
    checkpoints etc., and everything concerning the training run in the train function.
    """

    def __init__(
        self,
        model,
        optimizer,
        noise_scheduler,
        loss_fn="mse_loss",
        validation_interval=10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.loss_fn = getattr(F, loss_fn)
        self.validation_interval = validation_interval

    def compute_loss(self, batch):
        latents, context = batch
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, latents.shape[0]
        )
        noise = torch.randn(latents.shape[1:])
        noised_latents = self.noise_scheduler(latents, timesteps, noise)
        pred_noise = self.model(noised_latents, timesteps, context)
        return self.loss_fn(pred_noise, noise)

    def train(
        self,
        epochs,
        optimizer,
        lr_scheduler,
        train_dataloader,
    ):
        for epoch in range(epochs):
            for batch in train_dataloader:

                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
