import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_encoder(nn.Sequential):

    def __init__(self):
        super.__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1)
        )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.__version__)