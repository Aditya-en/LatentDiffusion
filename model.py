import torch
import torch.nn as nn
from typing import Tuple
from ResnetBlock import ResNetBlock
from DownSample import DownSample
from UpSample import UpSample
from SelfAttentionBlock import SelfAttentionBlock

class LatentDiffusionModel(nn.Module):
    """
    Latent Diffusion Model that integrates downsampling, self-attention, and upsampling 
    for processing latent representations.
    """
    def __init__(
        self, 
        input_channels: int, 
        latent_dim: int, 
        image_size: Tuple[int, int], 
        num_heads: int
    ) -> None:
        """
        Initializes the Latent Diffusion Model.

        Args:
            input_channels (int): Number of channels in the input image.
            latent_dim (int): Dimensionality of the latent space.
            image_size (Tuple[int, int]): Tuple (height, width) of the input image.
            num_heads (int): Number of attention heads for the self-attention block.
        """
        super().__init__()
        height, width = image_size
        # ResNet block
        self.resnet1 = ResNetBlock(input_channels, 64)
        self.resnet2 = ResNetBlock(256, 512)
        self.resnet3 = ResNetBlock(512, 512)
        self.resnet4 = ResNetBlock(512, 256)

        # Downsampling
        self.downsample1 = DownSample(height, width, 64, 128, latent_dim)
        self.downsample2 = DownSample(height // 2, width // 2, 128, 256, latent_dim)
        self.downsample3 = DownSample(height // 4, width // 4, 256, 256, latent_dim)

        # Bottleneck with self-attention
        self.self_attention1 = SelfAttentionBlock(num_heads, 128, height // 2, width // 2)
        self.self_attention2 = SelfAttentionBlock(num_heads, 256, height // 4, width // 4)
        self.self_attention3 = SelfAttentionBlock(num_heads, 256, height // 8, width // 8)

        self.self_attention4 = SelfAttentionBlock(num_heads, 128, height // 4, width // 4)
        self.self_attention5 = SelfAttentionBlock(num_heads, 64, height // 2, width // 2)
        self.self_attention6 = SelfAttentionBlock(num_heads, 64, height, width)

        # Upsampling
        self.upsample1 = UpSample(height // 8, width // 8, 256, 128, latent_dim)
        self.upsample2 = UpSample(height // 4, width // 4, 128, 64, latent_dim)
        self.upsample3 = UpSample(height // 2, width // 2, 64, 64, latent_dim)

        self.final_conv = nn.Conv2d(64, input_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Latent Diffusion Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            latent (torch.Tensor): Latent embedding of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        x = self.resnet1(x)
        residual1 = x
        x = self.downsample1(x, latent)
        x = self.self_attention1(x)
        residual2 = x
        x = self.downsample2(x, latent)
        x = self.self_attention2(x)
        residual3 = x
        x = self.downsample3(x, latent)
        x = self.self_attention3(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        x = self.resnet4(x)
        print(x.shape, residual3.shape, latent.shape)
        x = self.upsample1(x, residual3, latent)
        x = self.self_attention4(x)
        x = self.upsample2(x, residual2, latent)
        x = self.self_attention5(x)
        x = self.upsample3(x, residual1, latent)
        x = self.self_attention6(x)
        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    # Testing the Latent Diffusion Model
    batch_size = 8
    input_channels = 3
    latent_dim = 512
    image_size = (64, 64)
    num_heads = 4

    # Input tensor and latent embedding
    x = torch.randn(batch_size, input_channels, *image_size)
    latent = torch.randn(batch_size, latent_dim)

    # Initialize and test the model
    model = LatentDiffusionModel(input_channels, latent_dim, image_size, num_heads)
    output = model(x, latent)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
