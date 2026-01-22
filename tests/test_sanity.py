"""Sanity tests for nano-diffusion.

Lightweight tests to ensure core functionality works when making architecture changes.
These are not exhaustive unit tests, just practical checks that things don't break.
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.model import NanoDiffusionModel
from diffusion.noise_schedulers import (
    CosineNoiseScheduler,
    LinearNoiseScheduler,
    SigmoidNoiseScheduler,
)
from diffusion.samplers import DDIMSampler
from diffusion.trainer import NanoDiffusionTrainer


@pytest.fixture
def device():
    """Get CPU device for testing (avoids GPU requirements)"""
    return torch.device("cpu")


@pytest.fixture
def model_config():
    """Minimal model config for testing"""
    return OmegaConf.create(
        {
            "patch_size": 2,
            "hidden_dim": 64,
            "num_attention_heads": 4,
            "num_dit_blocks": 2,
            "activation": "SiLU",
            "normalization_layer": "LayerNorm",
            "num_context_classes": 10,
            "num_timesteps": 100,
            "vae_name": "ostris/vae-kl-f8-d16",
            "in_channels": 16,
            "dropout": 0.0,
            "device": "cpu",
        }
    )


@pytest.fixture
def noise_scheduler_config():
    """Minimal noise scheduler config"""
    return OmegaConf.create(
        {
            "num_timesteps": 100,
            "clip_min": 1e-9,
        }
    )


class TestModelSanity:
    """Test that the model can forward and backward without errors"""

    def test_model_forward_pass(self, model_config, device):
        """Test basic forward pass through the model"""
        model = NanoDiffusionModel(model_config).to(device)

        # Create dummy inputs
        batch_size = 2
        latents = torch.randn(batch_size, 16, 4, 4, device=device)
        timesteps = torch.randint(0, 100, (batch_size,), device=device)
        context = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass
        output = model(latents, timesteps, context)

        # Check output shape matches input
        assert output.shape == latents.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_backward_pass(self, model_config, device):
        """Test that gradients flow through the model"""
        model = NanoDiffusionModel(model_config).to(device)

        batch_size = 2
        latents = torch.randn(batch_size, 16, 4, 4, device=device)
        timesteps = torch.randint(0, 100, (batch_size,), device=device)
        context = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass
        output = model(latents, timesteps, context)

        # Compute loss and backward
        loss = output.mean()
        loss.backward()

        # Check that gradients exist and are not NaN
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_model_different_batch_sizes(self, model_config, device):
        """Test model works with different batch sizes"""
        model = NanoDiffusionModel(model_config).to(device)

        for batch_size in [1, 4, 8]:
            latents = torch.randn(batch_size, 16, 4, 4, device=device)
            timesteps = torch.randint(0, 100, (batch_size,), device=device)
            context = torch.randint(0, 10, (batch_size,), device=device)

            output = model(latents, timesteps, context)
            assert output.shape == latents.shape


class TestNoiseSchedulersSanity:
    """Test that noise schedulers work correctly"""

    @pytest.mark.parametrize(
        "scheduler_cls",
        [
            LinearNoiseScheduler,
            CosineNoiseScheduler,
            SigmoidNoiseScheduler,
        ],
    )
    def test_scheduler_forward_pass(
        self, scheduler_cls, noise_scheduler_config, device
    ):
        """Test scheduler can add noise correctly"""
        scheduler = scheduler_cls(noise_scheduler_config)

        batch_size = 2
        x = torch.randn(batch_size, 16, 4, 4, device=device)
        timesteps = torch.randint(0, 100, (batch_size,), device=device)

        # Add noise
        noisy_x, noise = scheduler(x, timesteps, return_noise=True)

        # Check shapes
        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape

        # Check noisy_x is different from original
        assert not torch.allclose(noisy_x, x)

        # Check no NaNs or Infs
        assert not torch.isnan(noisy_x).any()
        assert not torch.isinf(noisy_x).any()

    @pytest.mark.parametrize(
        "scheduler_cls",
        [
            LinearNoiseScheduler,
            CosineNoiseScheduler,
            SigmoidNoiseScheduler,
        ],
    )
    def test_scheduler_gamma_properties(self, scheduler_cls, noise_scheduler_config):
        """Test gamma function produces valid values"""
        scheduler = scheduler_cls(noise_scheduler_config)

        # Test at different timesteps
        timesteps = torch.linspace(0, 1, 50)
        gamma = scheduler.gamma_func(timesteps)

        # Gamma should be between 0 and 1
        assert torch.all(gamma >= 0)
        assert torch.all(gamma <= 1)

        # Should be monotonically decreasing (or at least non-increasing)
        assert torch.all(gamma[:-1] >= gamma[1:] - 1e-6)


class TestDDIMSamplerSanity:
    """Test that the DDIM sampler can generate samples"""

    def test_sampler_basic_generation(
        self, model_config, noise_scheduler_config, device
    ):
        """Test basic sample generation"""
        model = NanoDiffusionModel(model_config).to(device).eval()
        scheduler = CosineNoiseScheduler(noise_scheduler_config)
        sampler = DDIMSampler(
            model=model,
            noise_scheduler=scheduler,
            num_timesteps=100,
            num_sampling_steps=10,
        )

        # Generate from noise
        batch_size = 2
        noise = torch.randn(batch_size, 16, 4, 4, device=device)
        context = torch.randint(0, 10, (batch_size,), device=device)

        with torch.no_grad():
            samples = sampler.sample(noise, context)

        # Check output shape
        assert samples.shape == noise.shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

    def test_sampler_return_intermediates(
        self, model_config, noise_scheduler_config, device
    ):
        """Test sampler can return intermediate denoising steps"""
        model = NanoDiffusionModel(model_config).to(device).eval()
        scheduler = CosineNoiseScheduler(noise_scheduler_config)
        sampler = DDIMSampler(
            model=model,
            noise_scheduler=scheduler,
            num_timesteps=100,
            num_sampling_steps=10,
        )

        noise = torch.randn(1, 16, 4, 4, device=device)
        context = torch.tensor([0], device=device)

        with torch.no_grad():
            final_sample, intermediates = sampler.sample(
                noise, context, return_intermediates=True
            )

        # Check we got intermediates
        assert isinstance(intermediates, list)
        assert len(intermediates) > 0

        # Check shapes
        assert final_sample.shape == noise.shape
        for intermediate in intermediates:
            assert intermediate.shape == noise.shape


class TestTrainerSanity:
    """Test that the trainer can run a basic training loop"""

    def test_trainer_single_step(self, model_config, noise_scheduler_config, device):
        """Test trainer can do a single training step"""
        model = NanoDiffusionModel(model_config).to(device)
        scheduler = CosineNoiseScheduler(noise_scheduler_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NanoDiffusionTrainer(
                model=model,
                noise_scheduler=scheduler,
                checkpoint_dir=tmpdir,
                num_sampling_steps=5,
                loss_fn="mse_loss",
                validation_interval=10,
                save_every_n_epochs=10,
                keep_n_checkpoints=1,
            )

            # Create dummy batch
            batch_size = 2
            latents = torch.randn(batch_size, 16, 4, 4, device=device)
            context = torch.randint(0, 10, (batch_size,), device=device)
            batch = (latents, context)

            # Compute loss
            loss = trainer.compute_loss(batch)

            # Check loss is valid
            assert isinstance(loss, torch.Tensor)
            assert loss.ndim == 0  # Scalar
            assert loss.item() >= 0  # MSE is non-negative
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

    def test_trainer_mini_training_loop(
        self, model_config, noise_scheduler_config, device
    ):
        """Test trainer can run a few epochs without errors"""
        model = NanoDiffusionModel(model_config).to(device)
        scheduler = CosineNoiseScheduler(noise_scheduler_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Create tiny dataset
        dataset_size = 4
        latents = torch.randn(dataset_size, 16, 4, 4, device=device)
        context = torch.randint(0, 10, (dataset_size,), device=device)
        dataloader = [
            (latents[i : i + 2], context[i : i + 2]) for i in range(0, dataset_size, 2)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NanoDiffusionTrainer(
                model=model,
                noise_scheduler=scheduler,
                checkpoint_dir=tmpdir,
                num_sampling_steps=5,
                validation_interval=2,
                save_every_n_epochs=2,
            )

            # Run 3 epochs
            trainer.train(
                epochs=3,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=dataloader,
            )

            # Check that model parameters changed (training happened)
            # This is a smoke test - just ensure no errors occurred

    def test_checkpoint_save_load(self, model_config, noise_scheduler_config, device):
        """Test saving and loading checkpoints"""
        model = NanoDiffusionModel(model_config).to(device)
        scheduler = CosineNoiseScheduler(noise_scheduler_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NanoDiffusionTrainer(
                model=model,
                noise_scheduler=scheduler,
                checkpoint_dir=tmpdir,
                num_sampling_steps=5,
            )

            # Save checkpoint
            checkpoint_path = trainer.save_checkpoint(
                epoch=5,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                loss=0.5,
                is_best=False,
            )

            # Check file exists
            assert checkpoint_path.exists()

            # Check that we can load state dicts (don't test full restore in sanity tests)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert checkpoint["epoch"] == 5
            assert checkpoint["loss"] == 0.5
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint


class TestEndToEndIntegration:
    """Test complete pipeline from data to generation"""

    def test_full_pipeline_smoke_test(
        self, model_config, noise_scheduler_config, device
    ):
        """Smoke test: data → train → generate"""
        # Create model and components
        model = NanoDiffusionModel(model_config).to(device)
        scheduler = CosineNoiseScheduler(noise_scheduler_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Tiny training data
        latents = torch.randn(4, 16, 4, 4, device=device)
        context = torch.randint(0, 10, (4,), device=device)
        dataloader = [(latents[i : i + 2], context[i : i + 2]) for i in range(0, 4, 2)]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for 1 epoch
            trainer = NanoDiffusionTrainer(
                model=model,
                noise_scheduler=scheduler,
                checkpoint_dir=tmpdir,
                num_sampling_steps=5,
            )

            trainer.train(
                epochs=1,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=dataloader,
            )

            # Generate samples
            sampler = DDIMSampler(
                model=model,
                noise_scheduler=scheduler,
                num_timesteps=100,
                num_sampling_steps=5,
            )

            model.eval()
            with torch.no_grad():
                noise = torch.randn(2, 16, 4, 4, device=device)
                gen_context = torch.tensor([0, 1], device=device)
                samples = sampler.sample(noise, gen_context)

            # Check we got valid samples
            assert samples.shape == (2, 16, 4, 4)
            assert not torch.isnan(samples).any()
            assert not torch.isinf(samples).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
