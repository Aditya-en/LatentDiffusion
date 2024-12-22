import torch
import torch.nn as nn
from torch import Tensor
from ResnetBlock import ResNetBlock

class DownSample(nn.Module):
    """
    A DownSample module that applies max pooling, residual blocks, and combines embeddings to downsample.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool_kernel_size (int): Size of the max pooling kernel. Default is 2.
        pool (nn.MaxPool2d): Max pooling layer.
        res_block1 (ResNetBlock): First ResNet block for processing the downsampled input.
        res_block2 (ResNetBlock): Second ResNet block for further processing.
        silu (nn.SiLU): SiLU activation function applied to embeddings.
        linear (nn.Linear): Linear layer to map embeddings to match the spatial dimensions of the input.
    """
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        pool_kernel_size: int = 2
    ) -> None:
        """
        Initializes the DownSample module.

        Args:
            height (int): Height of the input feature map.
            width (int): Width of the input feature map.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            embedding_dim (int): Dimension of the embedding vector.
            pool_kernel_size (int, optional): Size of the max pooling kernel. Defaults to 2.
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.pool_kernel_size: int = pool_kernel_size
        self.pool: nn.MaxPool2d = nn.MaxPool2d(self.pool_kernel_size)
        self.res_block1: ResNetBlock = ResNetBlock(in_channels, out_channels)
        self.res_block2: ResNetBlock = ResNetBlock(out_channels, out_channels)
        self.silu: nn.SiLU = nn.SiLU()
        self.linear: nn.Linear = nn.Linear(embedding_dim, out_channels * (height // 2) * (width // 2))

    def forward(self, x: Tensor, embedd: Tensor) -> Tensor:
        """
        Performs a forward pass through the DownSample module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            embedd (Tensor): Embedding tensor of shape (batch_size, embedding_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, height // 2, width // 2).
        """
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        embedd = self.silu(embedd)
        y = self.linear(embedd).reshape(*x.shape)
        return x + y

if __name__ == "__main__":
    images = torch.randn(10, 64, 64, 64)  # Input tensor with batch_size=10, channels=64, height=64, width=64
    latents = torch.randn(10, 512)  # Embedding tensor with batch_size=10 and embedding_dim=512
    model = DownSample(64, 64, 64, 128, 512)
    out = model(images, latents)
    print(out.shape)
