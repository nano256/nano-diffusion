import os
import sys
import pytest
import torch
import torch.nn as nn

# Add the project root to Python path so that I can import functions from other folders
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion.utils import (
    PatchEmbedding,
    TimeEmbedding,
    AdaLNSingle,
    create_mlp,
    create_dit_block_config,
    DiTBlock,
    Reshaper,
)


class TestPatchEmbedding:

    def test_init(self):
        patch_embed = PatchEmbedding(patch_size=4, hidden_dim=256, in_channels=3)
        assert patch_embed.patch_size == 4
        assert patch_embed.hidden_dim == 256
        assert patch_embed.in_channels == 3
        assert isinstance(patch_embed.lin_projection, nn.Linear)
        assert patch_embed.lin_projection.in_features == 3 * 16
        assert patch_embed.lin_projection.out_features == 256

    def test_patchify_correct_shapes(self):
        patch_embed = PatchEmbedding(patch_size=4, hidden_dim=256, in_channels=3)
        x = torch.randn(2, 3, 32, 32)

        patches, x_pos, y_pos = patch_embed.patchify(x)

        expected_num_patches = 64  # (32//4) * (32//4)
        assert patches.shape == (2, expected_num_patches, 48)  # 3 * 4^2
        assert len(x_pos) == expected_num_patches
        assert len(y_pos) == expected_num_patches

    def test_patchify_position_ordering(self):
        """Test that positions follow left-to-right, top-to-bottom order"""
        patch_embed = PatchEmbedding(patch_size=2, hidden_dim=64, in_channels=1)
        x = torch.randn(1, 1, 4, 4)  # 2x2 patches

        patches, x_pos, y_pos = patch_embed.patchify(x)

        # Should be: (0,0), (1,0), (0,1), (1,1) in that order
        expected_x = torch.tensor([0, 1, 0, 1])
        expected_y = torch.tensor([0, 0, 1, 1])

        assert torch.equal(x_pos, expected_x)
        assert torch.equal(y_pos, expected_y)

    def test_patchify_different_patch_sizes(self):
        for patch_size in [1, 2, 4, 8]:
            patch_embed = PatchEmbedding(patch_size, hidden_dim=64, in_channels=3)
            size = patch_size * 4  # Ensure divisible
            x = torch.randn(1, 3, size, size)

            patches, x_pos, y_pos = patch_embed.patchify(x)
            expected_patches = (size // patch_size) ** 2

            assert patches.shape[1] == expected_patches
            assert len(x_pos) == expected_patches
            assert max(x_pos) == (size // patch_size) - 1
            assert max(y_pos) == (size // patch_size) - 1

    def test_patchify_invalid_dimensions(self):
        patch_embed = PatchEmbedding(patch_size=4, hidden_dim=256, in_channels=3)

        with pytest.raises(ValueError, match="height.*evenly divisible"):
            patch_embed.patchify(torch.randn(2, 3, 33, 32))

        with pytest.raises(ValueError, match="width.*evenly divisible"):
            patch_embed.patchify(torch.randn(2, 3, 32, 33))

    def test_forward_with_patch_pos(self):
        """Test forward pass returning embeddings and positions"""
        patch_embed = PatchEmbedding(patch_size=4, hidden_dim=256, in_channels=3)
        x = torch.randn(2, 3, 32, 32)

        result, x_pos, y_pos = patch_embed.forward(x, patch_pos=True)

        expected_num_patches = 64
        assert result.shape == (2, expected_num_patches, 256)
        assert len(x_pos) == expected_num_patches
        assert len(y_pos) == expected_num_patches
        assert torch.max(x_pos) == 7  # (32//4) - 1
        assert torch.max(y_pos) == 7


class TestTimeEmbedding:

    def test_init_parameters(self):
        time_embed = TimeEmbedding(num_timesteps=1000, hidden_dim=256)
        assert time_embed.num_timesteps == 1000
        assert time_embed.hidden_dim == 256
        assert len(time_embed.omega) == 128

    def test_forward_single_timestep(self):
        time_embed = TimeEmbedding(num_timesteps=1000, hidden_dim=128)
        timestep = torch.tensor([500.0])

        result = time_embed.forward(timestep)

        assert result.shape == (1, 128)
        assert not torch.all(result == 0)  # Should produce non-zero embeddings

    def test_forward_batch_timesteps(self):
        time_embed = TimeEmbedding(num_timesteps=1000, hidden_dim=64)
        timesteps = torch.tensor([0.0, 250.0, 500.0, 999.0])

        result = time_embed.forward(timesteps)

        assert result.shape == (4, 64)
        # Different timesteps should produce different embeddings
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(result[i], result[j], atol=1e-6)

    def test_embedding_properties(self):
        """Test mathematical properties of sinusoidal embeddings"""
        time_embed = TimeEmbedding(num_timesteps=1000, hidden_dim=64)

        # Same timestep should always produce same embedding
        t = torch.tensor([100.0])
        emb1 = time_embed.forward(t)
        emb2 = time_embed.forward(t)
        assert torch.allclose(emb1, emb2)

        # Embedding should be bounded between -1 and 1 (approximately)
        t_batch = torch.linspace(0, 999, 50)
        embeddings = time_embed.forward(t_batch)
        assert torch.all(embeddings >= -1.1)  # Small tolerance for numerical precision
        assert torch.all(embeddings <= 1.1)

    def test_different_hidden_dims(self):
        """Test various hidden dimensions"""
        for hidden_dim in [32, 64, 128, 256, 512]:
            time_embed = TimeEmbedding(1000, hidden_dim)
            t = torch.tensor([123.0])
            result = time_embed.forward(t)
            assert result.shape == (1, hidden_dim)


class TestAdaLNSingle:

    def test_init_structure(self):
        adaln = AdaLNSingle(hidden_dim=256, num_layers=12)

        assert isinstance(adaln.time_mlp, nn.Sequential)
        assert len(adaln.time_mlp) == 2  # Linear + SiLU
        assert adaln.layer_embeddings.shape == (12, 6 * 256)

        # Check first layer dimensions
        first_layer = adaln.time_mlp[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == 256
        assert first_layer.out_features == 6 * 256

    def test_forward_output_shape(self):
        adaln = AdaLNSingle(hidden_dim=128, num_layers=6)
        batch_sizes = [1, 4, 16]

        for batch_size in batch_sizes:
            time_emb = torch.randn(batch_size, 128)
            global_params = adaln.forward(time_emb)
            assert global_params.shape == (batch_size, 6 * 128)

    def test_get_layer_params_structure(self):
        adaln = AdaLNSingle(hidden_dim=256, num_layers=8)
        time_emb = torch.randn(3, 256)
        global_params = adaln.forward(time_emb)

        for layer_idx in [0, 3, 7]:
            params = adaln.get_layer_params(global_params, layer_idx)
            assert (
                len(params) == 6
            )  # beta_1, beta_2, gamma_1, gamma_2, alpha_1, alpha_2
            for param in params:
                assert param.shape == (3, 256)

    def test_parameter_consistency(self):
        """Test that same inputs produce same outputs"""
        adaln = AdaLNSingle(hidden_dim=128, num_layers=3)
        time_emb = torch.randn(2, 128)

        global_1 = adaln.forward(time_emb)
        global_2 = adaln.forward(time_emb)

        assert torch.allclose(global_1, global_2)


class TestCreateMLP:

    def test_basic_structure(self):
        mlp = create_mlp([64, 128, 256, 32])

        layers = list(mlp.children())
        assert len(layers) == 5  # 3 Linear + 2 SiLU

        # Check layer types and dimensions
        assert isinstance(layers[0], nn.Linear) and layers[0].in_features == 64
        assert isinstance(layers[1], nn.SiLU)
        assert isinstance(layers[2], nn.Linear) and layers[2].in_features == 128
        assert isinstance(layers[3], nn.SiLU)
        assert isinstance(layers[4], nn.Linear) and layers[4].out_features == 32

    def test_custom_activations(self):
        mlp = create_mlp([10, 20, 5], activation=nn.ReLU, final_activation=nn.Tanh)

        layers = list(mlp.children())
        assert isinstance(layers[1], nn.ReLU)
        assert isinstance(layers[3], nn.Tanh)

    def test_single_layer(self):
        mlp = create_mlp([128, 64])
        layers = list(mlp.children())
        assert len(layers) == 1
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 128
        assert layers[0].out_features == 64

    def test_functional_correctness(self):
        """Test that MLP actually processes data correctly"""
        mlp = create_mlp([10, 50, 20, 1])
        x = torch.randn(5, 10)

        output = mlp(x)
        assert output.shape == (5, 1)
        assert not torch.all(output == 0)  # Should produce non-zero outputs

    def test_gradient_flow(self):
        """Test that gradients flow through the MLP"""
        mlp = create_mlp([3, 10, 1])
        x = torch.randn(2, 3, requires_grad=True)

        output = mlp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestCreateDiTBlockConfig:

    def test_config_immutability(self):
        """Test that returned config is not shared between calls"""
        config1 = create_dit_block_config()
        config2 = create_dit_block_config()

        config1["hidden_dim"] = 512
        assert config2["hidden_dim"] == 256  # Should remain unchanged


class TestDiTBlock:

    @pytest.fixture
    def dit_components(self):
        adaln = AdaLNSingle(hidden_dim=256, num_layers=1)
        dit_block = DiTBlock(
            layer_idx=0,
            adaln_single=adaln,
            hidden_dim=256,
            num_attn_heads=8,
            dropout=0.1,
        )
        return dit_block, adaln

    def test_initialization(self, dit_components):
        dit_block, adaln = dit_components

        assert dit_block.hidden_dim == 256
        assert dit_block.layer_idx == 0
        assert dit_block.adaln_single is adaln
        assert dit_block.num_attn_heads == 8
        assert isinstance(dit_block.multi_head_attn, nn.MultiheadAttention)
        assert isinstance(dit_block.feedforward, nn.Sequential)

        # Check that MultiheadAttention has correct embed_dim (should be hidden_dim)
        assert dit_block.multi_head_attn.embed_dim == 256
        assert dit_block.multi_head_attn.num_heads == 8

    def test_forward_without_conditioning(self, dit_components):
        """Test forward pass without conditioning input"""
        dit_block, adaln = dit_components

        batch_size, seq_len, hidden_dim = 2, 16, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Get AdaLN parameters
        time_emb = torch.randn(batch_size, hidden_dim)
        global_adaln_params = adaln.forward(time_emb)

        # Forward pass
        output = dit_block.forward(x, global_adaln_params, c=None)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        # Output should be different from input due to processing
        assert not torch.allclose(output, x, atol=1e-6)

    def test_forward_with_conditioning(self, dit_components):
        """Test forward pass with conditioning input"""
        dit_block, adaln = dit_components

        batch_size, seq_len, hidden_dim = 2, 16, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)
        c = torch.randn(batch_size, 8, hidden_dim)  # Conditioning tokens

        # Get AdaLN parameters
        time_emb = torch.randn(batch_size, hidden_dim)
        global_adaln_params = adaln.forward(time_emb)

        # Forward pass with conditioning
        output_with_c = dit_block.forward(x, global_adaln_params, c)

        # Forward pass without conditioning for comparison
        output_without_c = dit_block.forward(x, global_adaln_params, c=None)

        assert output_with_c.shape == (batch_size, seq_len, hidden_dim)
        # Conditioning should produce different results
        assert not torch.allclose(output_with_c, output_without_c, atol=1e-6)

    def test_adaln_parameter_effects(self, dit_components):
        """Test that different AdaLN parameters affect the output"""
        dit_block, adaln = dit_components

        batch_size, seq_len, hidden_dim = 1, 8, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)

        # Different time embeddings should produce different global parameters
        time_emb_1 = torch.randn(batch_size, hidden_dim)
        time_emb_2 = torch.randn(batch_size, hidden_dim)

        global_params_1 = adaln.forward(time_emb_1)
        global_params_2 = adaln.forward(time_emb_2)

        output_1 = dit_block.forward(x, global_params_1)
        output_2 = dit_block.forward(x, global_params_2)

        # Different AdaLN parameters should produce different outputs
        assert not torch.allclose(output_1, output_2, atol=1e-6)

    def test_residual_connections(self, dit_components):
        """Test that residual connections are working"""
        dit_block, adaln = dit_components

        batch_size, seq_len, hidden_dim = 1, 4, 256
        x = torch.randn(batch_size, seq_len, hidden_dim)

        time_emb = torch.randn(batch_size, hidden_dim)
        global_adaln_params = adaln.forward(time_emb)

        output = dit_block.forward(x, global_adaln_params)

        # The magnitude of output should be related to input due to residual connections
        # This is a rough test - in practice, residuals help with gradient flow
        input_norm = torch.norm(x)
        output_norm = torch.norm(output)

        # Output shouldn't be drastically different in magnitude (very rough heuristic)
        assert output_norm > 0.1 * input_norm
        assert output_norm < 10 * input_norm

    def test_different_layer_indices(self):
        """Test that different layer indices produce different behaviors"""
        adaln = AdaLNSingle(hidden_dim=128, num_layers=3)

        dit_block_0 = DiTBlock(0, adaln, 128, 4, 0.0)
        dit_block_2 = DiTBlock(2, adaln, 128, 4, 0.0)

        batch_size, seq_len = 1, 8
        x = torch.randn(batch_size, seq_len, 128)
        time_emb = torch.randn(batch_size, 128)
        global_params = adaln.forward(time_emb)

        output_0 = dit_block_0.forward(x, global_params)
        output_2 = dit_block_2.forward(x, global_params)

        # Different layer indices should produce different outputs
        assert not torch.allclose(output_0, output_2, atol=1e-6)


class TestReshaper:

    def test_initialization(self):
        reshaper = Reshaper(patch_size=4, hidden_dim=256, in_channels=3)

        assert reshaper.patch_size == 4
        assert reshaper.hidden_dim == 256
        assert isinstance(reshaper.lin_projection, nn.Linear)
        assert reshaper.lin_projection.in_features == 256
        assert reshaper.lin_projection.out_features == 48  # 3 * 4^2

    def test_linear_projection_correctness(self):
        """Test that linear projection has correct dimensions for different configs"""
        configs = [
            (2, 64, 1),  # patch_size=2, hidden_dim=64, in_channels=1
            (4, 128, 3),  # patch_size=4, hidden_dim=128, in_channels=3
            (8, 512, 16),  # patch_size=8, hidden_dim=512, in_channels=16
        ]

        for patch_size, hidden_dim, in_channels in configs:
            reshaper = Reshaper(patch_size, hidden_dim, in_channels)

            expected_output_dim = in_channels * (patch_size**2)
            assert reshaper.lin_projection.out_features == expected_output_dim

    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality"""
        # Create 2x2 patches of size 2x2 each
        x_pos = torch.tensor([0, 1, 0, 1])
        y_pos = torch.tensor([0, 0, 1, 1])

        reshaper = Reshaper(patch_size=2, hidden_dim=64, in_channels=1)

        batch_size, num_patches = 2, 4
        x = torch.randn(batch_size, num_patches, 64)

        output = reshaper.forward(x, x_pos, y_pos)

        # Should reconstruct to 4x4 image (2 patches * 2 patch_size in each dim)
        assert output.shape == (batch_size, 1, 4, 4)

    def test_forward_different_patch_configurations(self):
        """Test forward pass with various patch configurations"""
        test_configs = [
            # (patch_size, grid_size, hidden_dim, in_channels)
            (2, 2, 32, 1),  # 2x2 patches, 2x2 grid -> 4x4 image
            (4, 2, 128, 3),  # 4x4 patches, 2x2 grid -> 8x8 image
            (2, 4, 64, 2),  # 2x2 patches, 4x4 grid -> 8x8 image
        ]

        for patch_size, grid_size, hidden_dim, in_channels in test_configs:
            # Create position tensors for grid_size x grid_size grid
            x_pos = torch.arange(grid_size).repeat(grid_size)
            y_pos = torch.repeat_interleave(torch.arange(grid_size), grid_size)

            reshaper = Reshaper(patch_size, hidden_dim, in_channels)

            num_patches = grid_size * grid_size
            x = torch.randn(1, num_patches, hidden_dim)

            output = reshaper.forward(x, x_pos, y_pos)

            expected_height = grid_size * patch_size
            expected_width = grid_size * patch_size
            assert output.shape == (1, in_channels, expected_height, expected_width)

    def test_forward_batch_processing(self):
        """Test that batch processing works correctly"""
        x_pos = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])  # 3x3 grid
        y_pos = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

        reshaper = Reshaper(patch_size=2, hidden_dim=32, in_channels=2)

        batch_sizes = [1, 3, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 9, 32)  # 9 patches
            output = reshaper.forward(x, x_pos, y_pos)

            # 3x3 patches of size 2x2 -> 6x6 image
            assert output.shape == (batch_size, 2, 6, 6)

    def test_layer_norm_application(self):
        """Test that layer normalization is applied before projection"""
        x_pos = torch.tensor([0, 1, 0, 1])
        y_pos = torch.tensor([0, 0, 1, 1])

        reshaper = Reshaper(2, 16, 1)

        # Create input with known statistics
        x = torch.randn(1, 4, 16) * 10 + 5  # Non-normalized input

        # The forward pass should still work (layer norm handles the scaling)
        output = reshaper.forward(x, x_pos, y_pos)
        assert output.shape == (1, 1, 4, 4)
        assert not torch.all(output == 0)

    def test_gradient_flow(self):
        """Test that gradients flow through the reshaper"""
        x_pos = torch.tensor([0, 1, 0, 1])
        y_pos = torch.tensor([0, 0, 1, 1])

        reshaper = Reshaper(2, 32, 1)

        x = torch.randn(1, 4, 32, requires_grad=True)
        output = reshaper.forward(x, x_pos, y_pos)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_position_tensor_validation(self):
        """Test behavior with different position tensor shapes"""
        reshaper = Reshaper(2, 16, 1)

        # Test with different position tensors
        x_pos_variants = [
            torch.tensor([0, 1]),
            torch.tensor([0, 2, 1, 3]),  # Non-contiguous positions
            torch.tensor([1, 0, 3, 2]),  # Different ordering
        ]

        for x_pos in x_pos_variants:
            y_pos = torch.zeros_like(x_pos)  # Keep y_pos simple
            num_patches = len(x_pos)
            x = torch.randn(1, num_patches, 16)

            # Should not crash and should produce reasonable output
            output = reshaper.forward(x, x_pos, y_pos)
            expected_width = (torch.max(x_pos) + 1) * 2  # patch_size = 2
            expected_height = (torch.max(y_pos) + 1) * 2

            assert output.shape == (1, 1, expected_height, expected_width)

    def test_reconstruction_properties(self):
        """Test mathematical properties of the reconstruction"""
        x_pos = torch.tensor([0, 1, 0, 1])
        y_pos = torch.tensor([0, 0, 1, 1])

        reshaper = Reshaper(2, 32, 2)

        # Test with deterministic input
        x = torch.ones(1, 4, 32)  # Constant input
        output = reshaper.forward(x, x_pos, y_pos)

        assert output.shape == (1, 2, 4, 4)

        # Test with zero input
        x_zero = torch.zeros(1, 4, 32)
        output_zero = reshaper.forward(x_zero, x_pos, y_pos)

        # After layer norm and linear projection, zero input might not give zero output
        # but should be consistent
        output_zero_2 = reshaper.forward(x_zero, x_pos, y_pos)
        assert torch.allclose(output_zero, output_zero_2)

    def test_different_input_channels(self):
        """Test reshaper with different numbers of input channels"""
        x_pos = torch.tensor([0, 1, 0, 1])
        y_pos = torch.tensor([0, 0, 1, 1])

        channel_configs = [1, 3, 4, 16, 64]

        for in_channels in channel_configs:
            reshaper = Reshaper(2, 64, in_channels)
            x = torch.randn(2, 4, 64)

            output = reshaper.forward(x, x_pos, y_pos)
            assert output.shape == (2, in_channels, 4, 4)

            # Check that linear projection has correct output size
            assert reshaper.lin_projection.out_features == in_channels * 4  # 2^2


class TestIntegration:
    """Integration tests for component interactions"""

    def test_patch_embed_with_reshaper_dimensions(self):
        """Test that PatchEmbedding and Reshaper have compatible dimensions"""
        patch_size, hidden_dim, in_channels = 4, 256, 3

        patch_embed = PatchEmbedding(patch_size, hidden_dim, in_channels)

        reshaper = Reshaper(patch_size, hidden_dim, in_channels)

        # Check dimension compatibility
        assert (
            patch_embed.lin_projection.out_features
            == reshaper.lin_projection.in_features
        )
        assert (
            patch_embed.lin_projection.in_features
            == reshaper.lin_projection.out_features
        )

    def test_time_embedding_with_adaln_compatibility(self):
        """Test TimeEmbedding output matches AdaLNSingle input requirements"""
        hidden_dim = 128

        time_embed = TimeEmbedding(num_timesteps=1000, hidden_dim=hidden_dim)
        adaln = AdaLNSingle(hidden_dim=hidden_dim, num_layers=6)

        # Test that TimeEmbedding output works with AdaLNSingle
        timesteps = torch.tensor([100.0, 200.0])
        time_emb = time_embed.forward(timesteps)

        assert time_emb.shape[1] == hidden_dim

        # This should work without dimension errors
        global_params = adaln.forward(time_emb)
        assert global_params.shape == (2, 6 * hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
