import torch
import torch.nn as nn
from torch import Tensor
from .components import ConvBlock


class CNNDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        input_dim: int,
        seq_len: int,
    ):
        super(CNNDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        dec_layers = [
            nn.Linear(self.latent_dim, self.seq_len * (self.hidden_dim // 2)),
            ConvBlock(self.hidden_dim // 2, self.hidden_dim),
            nn.Conv1d(self.hidden_dim,
                      self.input_dim,
                      kernel_size=1,
                      padding=0),
        ]
        self.dec_conv_module = nn.ModuleList(dec_layers)

    def forward(self, z_rep: Tensor):
        h_rep = z_rep  # B x 1 X L
        for indx, layer in enumerate(self.dec_conv_module):
            if indx == 1:
                h_rep = h_rep.reshape(-1, self.hidden_dim // 2, self.seq_len)
            h_rep = layer(h_rep)
        return h_rep

    @torch.inference_mode()
    def generate_from_latent(self, latent: Tensor) -> Tensor:
        logits = self.forward(latent.unsqueeze(1))
        log_probs = nn.functional.log_softmax(logits, dim=1)
        pred_tokens = log_probs.argmax(1)
        return pred_tokens
