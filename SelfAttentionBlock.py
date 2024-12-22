import torch
import torch.nn as nn
from torch import Tensor

class SelfAttentionBlock(nn.Module):
    """
    A Self-Attention Block that applies multi-head self-attention followed by 
    a feedforward network with normalization and skip connections.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization before the self-attention layer.
        mha (nn.MultiheadAttention): Multi-head self-attention mechanism.
        layer_norm2 (nn.LayerNorm): Layer normalization after the self-attention layer.
        linear (nn.Linear): First linear transformation in the feedforward network.
        gelu (nn.GELU): GELU activation function for non-linearity.
        linear2 (nn.Linear): Second linear transformation in the feedforward network.
    """
    def __init__(
        self, 
        num_heads: int, 
        in_channels: int, 
        height: int, 
        width: int
    ) -> None:
        """
        Initializes the Self-Attention Block.

        Args:
            num_heads (int): Number of attention heads.
            in_channels (int): Number of input channels (embedding dimension).
            height (int): Height of the input tensor.
            width (int): Width of the input tensor.
        """
        super().__init__()
        self.layer_norm: nn.LayerNorm = nn.LayerNorm([height * width, in_channels])
        self.mha: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm2: nn.LayerNorm = nn.LayerNorm([height * width, in_channels])
        self.linear: nn.Linear = nn.Linear(in_channels, in_channels)
        self.gelu: nn.GELU = nn.GELU()
        self.linear2: nn.Linear = nn.Linear(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass through the Self-Attention Block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor of the same shape as the input.
        """
        shape = x.shape  # Store the original shape
        x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten spatial dimensions
        x = x.permute(0, 2, 1)  # Rearrange dimensions for attention: (batch_size, seq_len, channels)
        skip1 = x  # First skip connection
        x = self.layer_norm(x)  # Apply layer normalization
        x, _ = self.mha(x, x, x)  # Apply multi-head self-attention
        skip2 = x  # Second skip connection
        skip2 += skip1  # Add the first skip connection
        skip2 = self.layer_norm2(skip2)  # Apply second layer normalization
        skip2 = self.linear(skip2)  # First linear transformation
        skip2 = self.gelu(skip2)  # Apply GELU activation
        skip2 = self.linear2(skip2)  # Second linear transformation
        x = x + skip2.reshape(x.shape)  # Add residual connection
        x = x.permute(0, 2, 1)  # Revert to original channel order
        x = x.reshape(shape)  # Reshape to the original shape
        return x

if __name__ == '__main__':
    model: SelfAttentionBlock = SelfAttentionBlock(4, 128, 32, 32)
    x: Tensor = torch.randn(2, 128, 32, 32)  # Input tensor with batch_size=2, channels=128, height=32, width=32
    x = model(x)  # Forward pass
    print(x.shape)  # Expected output shape: (2, 128, 32, 32)
