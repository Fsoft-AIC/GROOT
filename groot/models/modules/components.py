import torch
import torch.nn as nn
from torch import Tensor


def get_activation(type: str, dim: int) -> nn.Module:
    if type == "relu":
        return nn.ReLU()
    elif type == "leaky_relu":
        return nn.LeakyReLU()
    elif type == "elu":
        return nn.ELU()
    elif type == "prelu":
        return nn.PReLU(dim)
    else:
        raise ValueError(f"Activation {type} is not supported.")


class LatentEncoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int):
        super(LatentEncoder, self).__init__()
        self.linear_mu = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim * 2, latent_dim))
        self.linear_logsigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )

    def forward(self, hidden: Tensor):
        mu = self.linear_mu(hidden)
        logsigma = self.linear_logsigma(hidden)
        eps = torch.rand_like(mu)
        z = mu + eps * torch.exp(logsigma * 0.5)
        return z, mu, logsigma


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.

          from:
          https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch
        """
        super(ConvBlock, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm1d(out_channels))
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False), nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):

        identity = x

        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = nn.functional.relu(out)

        return out
