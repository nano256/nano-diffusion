import os
import sys

import pytest
import torch
import torch.nn as nn

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion.noise_schedulers import (
    CosineNoiseScheduler,
    LinearNoiseScheduler,
    SigmoidNoiseScheduler,
)
from diffusion.samplers import DDIMSampler


class TestLinearNoiseScheduler:
    @pytest.fixture
    def scheduler(self):
        return LinearNoiseScheduler(num_timesteps=1000)

    def test_init_parameters(self, scheduler):
        assert scheduler.num_timesteps == 1000
        assert scheduler.clip_min == 1e-9

    @pytest.mark.parametrize(
        "timesteps,expected_approx",
        [
            (torch.tensor([0.0]), 1.0),  # t=0 should give gamma≈1
            (torch.tensor([1.0]), 0.0),  # t=1 should give gamma≈0
            (torch.tensor([0.5]), 0.5),  # t=0.5 should give gamma≈0.5
        ],
    )
    def test_gamma_func_boundary_values(self, scheduler, timesteps, expected_approx):
        gamma = scheduler.gamma_func(timesteps)
        assert torch.allclose(gamma, torch.tensor([expected_approx]), atol=1e-8)

    def test_gamma_func_monotonic_decreasing(self, scheduler):
        timesteps = torch.linspace(0, 1, 100)
        gamma_values = scheduler.gamma_func(timesteps)

        # Should be monotonically decreasing
        diffs = gamma_values[1:] - gamma_values[:-1]
        assert torch.all(diffs <= 0), "Gamma should be monotonically decreasing"

    def test_gamma_func_bounds(self, scheduler):
        timesteps = torch.linspace(0, 1, 50)
        gamma = scheduler.gamma_func(timesteps)

        assert torch.all(gamma >= scheduler.clip_min)
        assert torch.all(gamma <= 1.0)

    def test_forward_pass_shapes(self, scheduler):
        batch_size, channels, height, width = 2, 3, 32, 32
        x = torch.randn(batch_size, channels, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))

        noisy_x, noise = scheduler.forward(x, timesteps)

        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape
        assert not torch.allclose(noisy_x, x)  # Should be different due to noise

    def test_forward_pass_without_return_noise(self, scheduler):
        x = torch.randn(2, 3, 16, 16)
        timesteps = torch.randint(0, 1000, (2,))

        noisy_x = scheduler.forward(x, timesteps, return_noise=False)
        assert noisy_x.shape == x.shape

    def test_custom_noise_input(self, scheduler):
        x = torch.randn(1, 1, 8, 8)
        timesteps = torch.tensor([500])
        custom_noise = torch.ones_like(x)

        noisy_x, returned_noise = scheduler.forward(x, timesteps, noise=custom_noise)

        assert torch.allclose(returned_noise, custom_noise)
        assert noisy_x.shape == x.shape


class TestCosineNoiseScheduler:
    @pytest.fixture
    def scheduler(self):
        return CosineNoiseScheduler(num_timesteps=1000, start=0.2, end=1.0, tau=2.0)

    def test_init_parameters(self, scheduler):
        assert scheduler.num_timesteps == 1000
        assert scheduler.start == 0.2
        assert scheduler.end == 1.0
        assert scheduler.tau == 2.0

    def test_gamma_func_bounds(self, scheduler):
        timesteps = torch.linspace(0, 1, 50)
        gamma = scheduler.gamma_func(timesteps)

        assert torch.all(gamma >= scheduler.clip_min)
        assert torch.all(gamma <= 1.0)

    def test_gamma_func_monotonic_decreasing(self, scheduler):
        timesteps = torch.linspace(0, 1, 100)
        gamma_values = scheduler.gamma_func(timesteps)

        diffs = gamma_values[1:] - gamma_values[:-1]
        assert torch.all(diffs <= 1e-6), "Gamma should be monotonically decreasing"

    @pytest.mark.parametrize(
        "start,end,tau",
        [
            (0.0, 1.0, 1.0),
            (0.1, 0.9, 2.0),
            (0.3, 1.0, 0.5),
        ],
    )
    def test_different_parameters(self, start, end, tau):
        scheduler = CosineNoiseScheduler(1000, start=start, end=end, tau=tau)
        timesteps = torch.tensor([0.0, 0.5, 1.0])
        gamma = scheduler.gamma_func(timesteps)

        assert len(gamma) == 3
        assert torch.all(gamma >= scheduler.clip_min)
        assert torch.all(gamma <= 1.0)


class TestSigmoidNoiseScheduler:
    @pytest.fixture
    def scheduler(self):
        return SigmoidNoiseScheduler(num_timesteps=1000, start=0.0, end=3.0, tau=0.7)

    def test_init_parameters(self, scheduler):
        assert scheduler.start == 0.0
        assert scheduler.end == 3.0
        assert scheduler.tau == 0.7

    def test_gamma_func_bounds(self, scheduler):
        timesteps = torch.linspace(0, 1, 50)
        gamma = scheduler.gamma_func(timesteps)

        assert torch.all(gamma >= scheduler.clip_min)
        assert torch.all(gamma <= 1.0)

    def test_gamma_func_monotonic_decreasing(self, scheduler):
        timesteps = torch.linspace(0, 1, 100)
        gamma_values = scheduler.gamma_func(timesteps)

        diffs = gamma_values[1:] - gamma_values[:-1]
        assert torch.all(diffs <= 1e-6), "Gamma should be monotonically decreasing"


class MockModel(nn.Module):
    """Simple mock model for testing DDIMSampler"""

    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x, timestep, context):
        # Return noise-like tensor with same shape as input
        return torch.randn_like(x)


class TestDDIMSampler:
    @pytest.fixture
    def mock_components(self):
        model = MockModel((2, 3, 16, 16))
        scheduler = LinearNoiseScheduler(num_timesteps=1000)
        sampler = DDIMSampler(
            model=model,
            noise_scheduler=scheduler,
            num_timesteps=1000,
            num_sampling_steps=10,
        )
        return sampler, model, scheduler

    def test_init_parameters(self, mock_components):
        sampler, model, scheduler = mock_components

        assert sampler.model is model
        assert sampler.noise_scheduler is scheduler
        assert sampler.num_timesteps == 1000
        assert sampler.num_sampling_steps == 10

    def test_step_method_basic_math(self, mock_components):
        sampler, _, _ = mock_components

        x_t = torch.randn(2, 3, 16, 16)
        noise_pred = torch.randn_like(x_t)
        gamma_t = torch.tensor(0.5)  # Example gamma value

        x_0_pred = sampler.step(x_t, noise_pred, gamma_t)

        assert x_0_pred.shape == x_t.shape
        assert not torch.allclose(x_0_pred, x_t)  # Should be different

    def test_step_method_gamma_extremes(self, mock_components):
        sampler, _, _ = mock_components

        x_t = torch.randn(1, 1, 8, 8)
        noise_pred = torch.randn_like(x_t)

        # Test gamma close to 1 (little noise)
        gamma_high = torch.tensor(0.99)
        x_0_high = sampler.step(x_t, noise_pred, gamma_high)

        # Test gamma close to 0 (lots of noise)
        gamma_low = torch.tensor(0.01)
        x_0_low = sampler.step(x_t, noise_pred, gamma_low)

        assert x_0_high.shape == x_t.shape
        assert x_0_low.shape == x_t.shape

    def test_sample_method_shapes(self, mock_components):
        sampler, _, _ = mock_components

        batch_size, channels, height, width = 2, 3, 16, 16
        x_T = torch.randn(batch_size, channels, height, width)
        context = torch.randint(0, 10, (batch_size,))

        result = sampler.sample(x_T, context)

        assert result.shape == x_T.shape

    def test_sample_timestep_progression(self, mock_components):
        sampler, model, scheduler = mock_components

        # Track timesteps passed to model
        timestep_calls = []

        def mock_forward(x, timestep, context):
            timestep_calls.append(
                timestep.item() if hasattr(timestep, "item") else timestep
            )
            return torch.randn_like(x)

        model.forward = mock_forward

        x_T = torch.randn(1, 3, 8, 8)
        context = torch.tensor([5])

        result = sampler.sample(x_T, context)

        # Should have called model for each sampling step
        assert len(timestep_calls) == sampler.num_sampling_steps

        # Timesteps should be in decreasing order (going backwards in time)
        assert all(
            timestep_calls[i] >= timestep_calls[i + 1]
            for i in range(len(timestep_calls) - 1)
        )

    @pytest.mark.parametrize("num_sampling_steps", [5, 20, 50])
    def test_different_sampling_steps(self, num_sampling_steps):
        model = MockModel((1, 1, 8, 8))
        scheduler = LinearNoiseScheduler(num_timesteps=1000)
        sampler = DDIMSampler(model, scheduler, 1000, num_sampling_steps)

        x_T = torch.randn(1, 1, 8, 8)
        context = torch.tensor([0])

        result = sampler.sample(x_T, context)
        assert result.shape == x_T.shape

    def test_sample_with_seed(self, mock_components):
        sampler, _, _ = mock_components

        x_T = torch.randn(1, 1, 4, 4)
        context = torch.tensor([1])

        # Test that seed parameter doesn't cause errors (implementation detail)
        result = sampler.sample(x_T, context, seed=42)
        assert result.shape == x_T.shape

    def test_gamma_t_computation(self, mock_components):
        sampler, model, scheduler = mock_components

        # Test that gamma_t values are computed correctly from scheduler
        timesteps = (
            torch.linspace(0, sampler.num_timesteps, sampler.num_sampling_steps)
            .round()
            .long()
        )
        gamma_t = scheduler.gamma_func(timesteps.float() / sampler.num_timesteps)

        assert len(gamma_t) == sampler.num_sampling_steps
        assert torch.all(gamma_t >= scheduler.clip_min)
        assert torch.all(gamma_t <= 1.0)

        # Should be decreasing sequence
        diffs = gamma_t[1:] - gamma_t[:-1]
        assert torch.all(diffs <= 1e-6)


class TestNoiseSchedulerIntegration:
    """Integration tests between schedulers and other components"""

    @pytest.mark.parametrize(
        "scheduler_class",
        [LinearNoiseScheduler, CosineNoiseScheduler, SigmoidNoiseScheduler],
    )
    def test_scheduler_with_euler_sampler(self, scheduler_class):
        model = MockModel((1, 1, 4, 4))

        if scheduler_class == LinearNoiseScheduler:
            scheduler = scheduler_class(num_timesteps=100)
        else:
            scheduler = scheduler_class(num_timesteps=100)

        sampler = DDIMSampler(model, scheduler, 100, 5)

        x_T = torch.randn(1, 1, 4, 4)
        context = torch.tensor([0])

        result = sampler.sample(x_T, context)
        assert result.shape == x_T.shape

    def test_batch_consistency_across_schedulers(self):
        """Test that all schedulers handle batching consistently"""
        schedulers = [
            LinearNoiseScheduler(1000),
            CosineNoiseScheduler(1000),
            SigmoidNoiseScheduler(1000),
        ]

        batch_size = 3
        x = torch.randn(batch_size, 2, 8, 8)
        timesteps = torch.randint(0, 1000, (batch_size,))

        for scheduler in schedulers:
            noisy_x, noise = scheduler.forward(x, timesteps)
            assert noisy_x.shape == x.shape
            assert noise.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
