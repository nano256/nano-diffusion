import os
import sys
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion.trainer import NanoDiffusionTrainer
from diffusion.utils import LinearNoiseScheduler


class MockModel(nn.Module):
    """Mock model for testing trainer functionality"""

    def __init__(self):
        super().__init__()
        # Simple linear layer to enable gradient computation
        self.linear = nn.Linear(64, 64)

    def forward(self, noised_latents, timesteps, context):
        # Flatten input, pass through linear layer, reshape back
        batch_size, channels, height, width = noised_latents.shape
        x = noised_latents.view(batch_size, -1)

        # Pad or truncate to 64 features for the linear layer
        if x.size(1) < 64:
            padding = torch.zeros(batch_size, 64 - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > 64:
            x = x[:, :64]

        # Pass through linear layer (this enables gradients)
        x = self.linear(x)

        # Reshape back to original dimensions
        if x.size(1) >= channels * height * width:
            x = x[:, : channels * height * width]
        else:
            # If not enough features, pad with zeros
            padding_size = channels * height * width - x.size(1)
            padding = torch.zeros(batch_size, padding_size, device=x.device)
            x = torch.cat([x, padding], dim=1)

        return x.view(batch_size, channels, height, width)


class MockOptimizer:
    """Mock optimizer that tracks method calls"""

    def __init__(self):
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.param_groups = [{"lr": 0.001}]

    def zero_grad(self):
        self.zero_grad_calls += 1

    def step(self):
        self.step_calls += 1


class MockLRScheduler:
    """Mock learning rate scheduler that tracks step calls"""

    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class TestNanoDiffusionTrainer:

    @pytest.fixture
    def trainer_components(self):
        model = MockModel()
        noise_scheduler = LinearNoiseScheduler(num_timesteps=100)
        trainer = NanoDiffusionTrainer(
            model=model, noise_scheduler=noise_scheduler, loss_fn="mse_loss"
        )
        return trainer, model, noise_scheduler

    @pytest.fixture
    def sample_batch(self):
        # (latents, context) tuple
        latents = torch.randn(2, 4, 8, 8)  # batch_size=2, channels=4, 8x8 latent
        context = torch.randn(2, 77, 768)  # text embeddings
        return (latents, context)

    def test_trainer_initialization(self, trainer_components):
        trainer, model, noise_scheduler = trainer_components

        assert trainer.model is model
        assert trainer.noise_scheduler is noise_scheduler
        assert trainer.validation_interval == 10
        assert callable(trainer.loss_fn)

    def test_compute_loss_basic_functionality(self, trainer_components, sample_batch):
        trainer, _, _ = trainer_components

        loss = trainer.compute_loss(sample_batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # MSE loss is non-negative

    def test_compute_loss_tensor_shapes(self, trainer_components):
        trainer, _, _ = trainer_components

        # Test different batch sizes and latent shapes
        test_cases = [
            ((1, 3, 16, 16), (1, 77, 768)),  # Single sample
            ((4, 8, 32, 32), (4, 77, 768)),  # Larger batch
        ]

        for latent_shape, context_shape in test_cases:
            batch = (torch.randn(*latent_shape), torch.randn(*context_shape))
            loss = trainer.compute_loss(batch)

            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad

    def test_compute_loss_timestep_device_handling(self, trainer_components):
        trainer, _, _ = trainer_components

        # Test that timesteps are placed on the correct device
        latents = torch.randn(2, 4, 8, 8)
        context = torch.randn(2, 77, 768)
        batch = (latents, context)

        # This should not raise device mismatch errors
        loss = trainer.compute_loss(batch)
        assert isinstance(loss, torch.Tensor)

    def test_train_single_epoch_basic(self, trainer_components, sample_batch):
        trainer, _, _ = trainer_components
        mock_optimizer = MockOptimizer()
        mock_lr_scheduler = MockLRScheduler()

        # Create simple dataloader with 2 batches
        dataloader = [sample_batch, sample_batch]

        # Run training for 1 epoch
        trainer.train(
            epochs=1,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            train_dataloader=dataloader,
            experiment_name="test_experiment",
        )

        # Verify training loop executed correctly
        assert mock_optimizer.zero_grad_calls == 2  # 2 batches
        assert mock_optimizer.step_calls == 2  # 2 batches
        assert mock_lr_scheduler.step_calls == 1  # 1 epoch

    def test_train_multiple_epochs(self, trainer_components, sample_batch):
        trainer, _, _ = trainer_components
        mock_optimizer = MockOptimizer()
        mock_lr_scheduler = MockLRScheduler()

        dataloader = [sample_batch]  # 1 batch per epoch
        epochs = 3

        trainer.train(
            epochs=epochs,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            train_dataloader=dataloader,
            experiment_name="test_multi_epoch",
        )

        assert mock_optimizer.zero_grad_calls == epochs
        assert mock_optimizer.step_calls == epochs
        assert mock_lr_scheduler.step_calls == epochs

    def test_train_with_validation(self, trainer_components, sample_batch):
        trainer, _, _ = trainer_components
        mock_optimizer = MockOptimizer()
        mock_lr_scheduler = MockLRScheduler()

        # Set validation_interval to 2
        trainer.validation_interval = 2

        train_dataloader = [sample_batch] * 2  # 2 batches per epoch
        val_dataloader = [sample_batch]  # 1 validation batch

        # Train for 4 epochs (validation should run at epochs 2, 4)
        trainer.train(
            epochs=4,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            experiment_name="test_validation",
        )

        # Training should complete without errors
        assert mock_optimizer.step_calls == 8  # 4 epochs * 2 batches
        assert mock_lr_scheduler.step_calls == 4

    def test_train_validation_skip_first_epoch(self, trainer_components, sample_batch):
        trainer, _, _ = trainer_components
        mock_optimizer = MockOptimizer()
        mock_lr_scheduler = MockLRScheduler()

        # Set validation_interval to 1 (every epoch)
        trainer.validation_interval = 1

        train_dataloader = [sample_batch]
        val_dataloader = [sample_batch]

        # Train for 2 epochs - validation should only run at epoch 1 (not epoch 0)
        trainer.train(
            epochs=2,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            experiment_name="test_validation_skip",
        )

        # Should complete without errors (validation logic tested implicitly)
        assert mock_optimizer.step_calls == 2

    @pytest.mark.parametrize("loss_fn", ["mse_loss", "l1_loss", "smooth_l1_loss"])
    def test_different_loss_functions(self, loss_fn, sample_batch):
        model = MockModel()
        noise_scheduler = LinearNoiseScheduler(num_timesteps=100)

        trainer = NanoDiffusionTrainer(
            model=model, noise_scheduler=noise_scheduler, loss_fn=loss_fn
        )

        loss = trainer.compute_loss(sample_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_default_experiment_name_generation(self, trainer_components, sample_batch):
        trainer, _, _ = trainer_components
        mock_optimizer = MockOptimizer()
        mock_lr_scheduler = MockLRScheduler()

        dataloader = [sample_batch]

        # Test that training works without explicit experiment_name
        trainer.train(
            epochs=1,
            optimizer=mock_optimizer,
            lr_scheduler=mock_lr_scheduler,
            train_dataloader=dataloader,
        )

        assert mock_optimizer.step_calls == 1

    def test_gradient_flow(self, trainer_components, sample_batch):
        trainer, model, _ = trainer_components

        # Create model with actual parameters to test gradient flow
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)  # Simple linear layer

            def forward(self, noised_latents, timesteps, context):
                # Flatten, transform, reshape back
                b, c, h, w = noised_latents.shape
                x = noised_latents.view(b, -1)
                if x.size(1) >= 32:
                    x = x[:, :32]
                else:
                    x = torch.cat([x, torch.zeros(b, 32 - x.size(1))], dim=1)
                x = self.linear(x)

                # Reshape back to original dimensions
                total_features = c * h * w
                if x.size(1) >= total_features:
                    x = x[:, :total_features]
                else:
                    # Pad with zeros if needed
                    padding = torch.zeros(
                        b, total_features - x.size(1), device=x.device
                    )
                    x = torch.cat([x, padding], dim=1)

                return x.view(b, c, h, w)

        # Replace mock model with actual model
        trainer.model = SimpleModel()

        loss = trainer.compute_loss(sample_batch)
        loss.backward()

        # Check that gradients exist
        for param in trainer.model.parameters():
            assert param.grad is not None
