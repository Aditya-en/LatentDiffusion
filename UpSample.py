import torch
import torch.nn as nn
from ResnetBlock import ResNetBlock

class UpSample(nn.Module):
    def __init__(self,height, width, in_channels, out_channels, embed_dim=512):
        super(UpSample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.resnet1 = ResNetBlock(in_channels, out_channels)
        self.resnet2 = ResNetBlock(out_channels, out_channels)
        self.linear = nn.Linear(embed_dim, out_channels*height*width)
        self.activation = nn.SiLU()

    def forward(self, x, residual, third_input):
        x = self.upsample(x)
        x = x + residual
        x = self.resnet1(x)
        x = self.resnet2(x)
        third_input = self.activation(third_input)
        third_input = self.linear(third_input).reshape(*x.shape)
        x += third_input
        return x
    
if __name__ == "__main__":
    upsample = UpSample(64, 64, 64, 64)
    x = torch.randn(5, 64, 32, 32)
    residual = torch.randn(5, 64, 64, 64)
    third_input = torch.randn(5, 512)
    print(upsample(x, residual, third_input).shape)