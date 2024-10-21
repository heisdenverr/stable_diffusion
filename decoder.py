import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

"""
the goal here is to check the relationship between pixels and features -> [batch_size, pixels, out_channels]
"""
class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super.__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tesor) -> torch.Tensor:
        # x: (batch_size, features, height, width)

        residue = x

        n, c, h, w = x.shape
        
        # (batch_size, features, height, weight) -> (batch_size, features, pixels)
        x = x.view(n, c, h*w)

        # (batch_size, pixels(height * width), features)
        x = x.transpose(-1, -2)

        x = self.attention(x)
        
        # reverted back to original
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))
        x += residue

        return x





class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding1)

        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels = out_channels:
            self.residual_layer = nn.Identify()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residue = x

        x = self.group_norm1(x)

        x = F.SiLU(x)

        x = self.conv_1(x)

        x = self.group_norm2(x)

        x = F.SiLU(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)