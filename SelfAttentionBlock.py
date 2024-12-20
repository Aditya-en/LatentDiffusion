import torch
import torch.nn as nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, num_heads, in_channels, height, width):
        super().__init__()
        self.layer_norm = nn.LayerNorm([height*width, in_channels])
        self.mha = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm([height*width, in_channels])
        self.linear = nn.Linear(in_channels, in_channels)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        skip1 = x
        x = self.layer_norm(x)
        x, _ = self.mha(x, x, x)
        skip2 = x
        skip2 += skip1
        skip2 = self.layer_norm2(skip2)
        skip2 = self.linear(skip2)
        skip2 = self.gelu(skip2)
        skip2 = self.linear2(skip2)
        x = x + skip2.reshape(x.shape)
        x = x.permute(0, 2, 1)
        x = x.reshape(shape)
        return x

if __name__ == '__main__':
    model = SelfAttentionBlock( 4, 128, 32, 32)
    x = torch.randn(2, 128, 32, 32)
    x = model(x)
    print(x.shape)
