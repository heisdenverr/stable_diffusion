import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_encoder(nn.Sequential):

    """
    Takes an image, reduces image and increases it features
    Then we sample from the joint distribution Z(mean, variance) -> Z(mean, stdev * Noise)
    
    """

    def __init__(self):
        super.__init__(
            # (Batch_size, Channel, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (batch_size, 128, height , width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # (batch_size, 128, height , width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (Global attention) the goal here is to check the relationship between pixels and features -> [batch_size, pixels, out_channels]
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512).
            nn.SiLU(),

            nn.Conv2d(512. 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding_left, padding_right, padding_left, padding_right)
                x = F.pad(0, 1, 0, 1)
            x = module(x)

        # Parameters of a multivariate Gaussian distribution.
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # clamp variance
        log_variance = torch.clamp(log_variance, -30, 20)

        # Take the exponent to remove the log
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # N(0, 1) -> N(mean, variance)
        # X = mean + stdev * Z

        x = mean + stdev*noise

        # Scale output by a constant
        x *= 0.18215

    return x