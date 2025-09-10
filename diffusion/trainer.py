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
        checkpoint_dir="./checkpoints",
        save_every_n_epochs=10,
        keep_n_checkpoints=3,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.loss_fn = getattr(F, loss_fn)
        self.validation_interval = validation_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_n_checkpoints = keep_n_checkpoints
        self.best_val_loss = float('inf')
        self.saved_checkpoints = []  # Track saved checkpoint paths
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(self, batch):
        latents, context = batch
        timesteps = torch.randint(
            0, self.noise_scheduler.num_timesteps, (latents.shape[0],)
        )
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler(latents, timesteps, noise, return_noise=False)
        pred_noise = self.model(noised_latents, timesteps, context)
        return self.loss_fn(pred_noise, noise)

    def save_checkpoint(self, epoch, optimizer, lr_scheduler, loss, is_best=False):
        """Save model checkpoint with optional cleanup of old checkpoints"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            'best_val_loss': self.best_val_loss,
        }
        
        # Create checkpoint filename
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Track regular checkpoints (not best model)
        if not is_best:
            self.saved_checkpoints.append(checkpoint_path)
            
            # Remove old checkpoints if we exceed the limit
            if len(self.saved_checkpoints) > self.keep_n_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, optimizer=None, lr_scheduler=None):
        """Load model checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
        # Restore training state
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch'], checkpoint['loss']

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

            # Validation and checkpointing
            avg_val_loss = None
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
                    
                    # Save best model if validation loss improved
                    if avg_val_loss < self.best_val_loss:
                        self.best_val_loss = avg_val_loss
                        self.save_checkpoint(epoch, optimizer, lr_scheduler, avg_val_loss, is_best=True)
            
            # Save regular checkpoint every N epochs
            if epoch % self.save_every_n_epochs == 0 or epoch == epochs - 1:
                current_loss = avg_val_loss if avg_val_loss is not None else loss.item()
                self.save_checkpoint(epoch, optimizer, lr_scheduler, current_loss)
                
            lr_scheduler.step()
        writer.close()
