import torch
import torch.nn as nn
from torch import Tensor
from ResnetBlock import ResNetBlock

class UpSample(nn.Module):
    """
    An UpSample module that upsamples feature maps, incorporates residual connections, and combines embeddings 
    to enhance the output feature maps.

    Attributes:
        upsample (nn.Upsample): Upsampling layer to increase spatial resolution.
        resnet1 (ResNetBlock): First ResNet block for processing the upsampled input.
        resnet2 (ResNetBlock): Second ResNet block for further processing.
        linear (nn.Linear): Linear layer to map embeddings to match the spatial dimensions of the input.
        activation (nn.SiLU): SiLU activation function applied to embeddings.
    """
    def __init__(
        self,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 512
    ) -> None:
        """
        Initializes the UpSample module.

        Args:
            height (int): Height of the output feature map.
            width (int): Width of the output feature map.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            embed_dim (int, optional): Dimension of the embedding vector. Defaults to 512.
        """
        super(UpSample, self).__init__()
        self.upsample: nn.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.resnet1: ResNetBlock = ResNetBlock(in_channels, out_channels)
        self.resnet2: ResNetBlock = ResNetBlock(out_channels, out_channels)
        self.linear: nn.Linear = nn.Linear(embed_dim, out_channels * height * 2 * width * 2)
        self.activation: nn.SiLU = nn.SiLU()

    def forward(self, x: Tensor, residual: Tensor, third_input: Tensor) -> Tensor:
        """
        Performs a forward pass through the UpSample module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height // 2, width // 2).
            residual (Tensor): Residual tensor of shape (batch_size, out_channels, height, width).
            third_input (Tensor): Embedding tensor of shape (batch_size, embed_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        x = self.upsample(x)  # Upsample the input to double its spatial dimensions.
        x = x + residual  # Add the residual connection.
        x = self.resnet1(x)  # Pass through the first ResNet block.
        x = self.resnet2(x)  # Pass through the second ResNet block.
        batch, channels, height, width = x.shape
        third_input = self.activation(third_input)  # Apply SiLU activation to the embedding.
        third_input = self.linear(third_input).reshape(batch, channels, height, width)  # Reshape the embedding to match x.
        x += third_input  # Add the processed embedding to the feature map.
        return x

if __name__ == "__main__":
    upsample: UpSample = UpSample(8, 8, 256, 128)
    x: Tensor = torch.randn(8, 256, 8, 8)  # Input tensor with batch_size=5, channels=64, height=32, width=32
    residual: Tensor = torch.randn(8, 256, 16, 16)  # Residual tensor with matching output dimensions
    third_input: Tensor = torch.randn(8, 512)  # Embedding tensor with batch_size=5 and embedding_dim=512
    print(upsample(x, residual, third_input).shape)  # Expected output shape: (5, 64, 64, 64)
