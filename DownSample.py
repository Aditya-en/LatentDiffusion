import torch
import torch.nn as nn
from ResnetBlock import ResNetBlock

class DownSample(nn.Module):
    def __init__(self,height, width, in_channels, out_channels, embedding_dim, pool_kernel_size = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_kernel_size = pool_kernel_size
        self.pool = nn.MaxPool2d(self.pool_kernel_size)
        self.res_block1 = ResNetBlock(in_channels, out_channels)
        self.res_block2 = ResNetBlock(out_channels, out_channels)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, out_channels*(height//2)*(width//2))

    def forward(self, x, embedd):
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        embedd = self.silu(embedd)
        y = self.linear(embedd).reshape(*x.shape)
        return x+y

if __name__ == "__main__":
    images = torch.randn(10, 64, 64, 64)
    latents = torch.randn(10, 512)
    model = DownSample(64, 64, 64, 128, 512)
    out = model(images, latents)
    print(out.shape)