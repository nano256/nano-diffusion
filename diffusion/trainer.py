import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime


class NanoDiffusionTrainer:
    """Trainer for the Nano Diffusion model

    TODO:
    Add all administrative stuff in the Trainer class, e.g. logging, saving model
    checkpoints etc., and everything concerning the training run in the train function.
    """

    def __init__(
        self,
        model,
        noise_scheduler,
        loss_fn="mse_loss",
        validation_interval=10,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.loss_fn = getattr(F, loss_fn)
        self.validation_interval = validation_interval

    def compute_loss(self, batch):
        latents, context = batch
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, (latents.shape[0],)
        )
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler(latents, timesteps, noise, return_noise=False)
        pred_noise = self.model(noised_latents, timesteps, context)
        return self.loss_fn(pred_noise, noise)

    def train(
        self,
        epochs,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader=None,
        experiment_name=None,
        log_dir="./runs",
    ):

        log_dir = Path(log_dir)
        if experiment_name is None:
            experiment_name = (
                f"nano-diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        writer = SummaryWriter(log_dir / experiment_name)
        for epoch in range(epochs):
            num_prev_batches = epoch * len(train_dataloader)
            for batch_idx, batch in enumerate(train_dataloader):

                optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                optimizer.step()

                # Log everything
                writer.add_scalar(
                    "loss_train",
                    loss.item(),
                    num_prev_batches + batch_idx,
                )
                writer.add_scalar(
                    "learning_rate",
                    optimizer.param_groups[0]["lr"],
                    num_prev_batches + batch_idx,
                )

            if (
                val_dataloader is not None
                and epoch % self.validation_interval == 0
                and epoch != 0
            ):
                with torch.no_grad():
                    val_losses = []
                    for batch in val_dataloader:
                        val_losses.append(self.compute_loss(batch).item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    writer.add_scalar(
                        "loss_val",
                        avg_val_loss,
                        num_prev_batches + len(train_dataloader),
                    )
            lr_scheduler.step()
        writer.close()
