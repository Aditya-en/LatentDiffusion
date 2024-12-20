import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)
        self.group_norm1 = nn.GroupNorm(1, self.out_channels)
        self.group_norm2 = nn.GroupNorm(1, self.out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.group_norm1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.group_norm2(x)
        return x + residual
    
if __name__ == "__main__":
    x = torch.randn(5, 64, 64, 64)
    model = ResNetBlock(64, 64)
    out = model(x)
    print(out.shape)