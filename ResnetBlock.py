import torch
import torch.nn as nn
from torch import Tensor

class ResNetBlock(nn.Module):
    """
    A residual block for use in the latent diffusion model. This block applies two convolutional layers, 
    each followed by group normalization and a GELU activation, and adds a residual connection.

    Attributes:
        in_channels (int): Number of input channels for the block.
        out_channels (int): Number of output channels for the block.
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        group_norm1 (nn.GroupNorm): Group normalization layer applied after the first convolution.
        group_norm2 (nn.GroupNorm): Group normalization layer applied after the second convolution.
        gelu (nn.GELU): GELU activation function.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes the ResNetBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.conv1: nn.Conv2d = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.group_norm1: nn.GroupNorm = nn.GroupNorm(1, self.out_channels)
        self.group_norm2: nn.GroupNorm = nn.GroupNorm(1, self.out_channels)
        self.gelu: nn.GELU = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass through the ResNet block.
        Args:
            x (Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output tensor of the same shape as the input tensor.
        """
        x = self.conv1(x)
        x = self.group_norm1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.group_norm2(x)
        return x

if __name__ == "__main__":
    x: Tensor = torch.randn(5, 64, 64, 64)  # Input tensor with batch_size=5, channels=64, height=64, width=64
    model: ResNetBlock = ResNetBlock(64, 64)
    out: Tensor = model(x)
    print(out.shape)
