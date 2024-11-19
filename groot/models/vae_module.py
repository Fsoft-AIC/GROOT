import torch
from torch import Tensor
from typing import List, Union
from .base import BaseVAE
from ..common.constants import convert_ids2seqs, convert_seqs2ids, VOCAB
from .modules import CNNDecoder


class CNNVAE(BaseVAE):

    def __init__(
        self,
        expected_kl: float,
        pretrained_encoder_path: str = "facebook/esm2_t12_35M_UR50D",
        num_unfreeze_layers: int = 0,
        latent_dim: int = 64,
        dec_hidden_dim: int = 512,
        pred_hidden_dim: int = 128,
        pred_dropout: float = 0.2,
        max_len: int = 500,
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        kl_weight: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        Kp: float = 0.01,
        Ki: float = 0.0001,
        lr: float = 0.001,
        device: Union[torch.device, str] = "cuda",
        reduction: str = "sum",
        interp_size: int = 16,
        interp_weight: float = 0.001,
        use_interp_sampling: bool = False,
        neg_focus: bool = False,
        neg_floor: float = -2.0,
        neg_size: int = 16,
        neg_norm: float = 4.0,
        neg_weight: float = 0.8,
        use_neg_sampling: bool = False,
        regularize_latent: bool = False,
        latent_weight: float = 0.001,
    ):
        super(CNNVAE, self).__init__(
            expected_kl, pretrained_encoder_path, num_unfreeze_layers, latent_dim, pred_hidden_dim,
            pred_dropout, nll_weight, mse_weight, kl_weight, beta_min, beta_max, Kp, Ki, lr,
            reduction, interp_size, interp_weight, use_interp_sampling, neg_focus, neg_floor,
            neg_size, neg_norm, neg_weight, use_neg_sampling, regularize_latent, latent_weight
        )

        # Decoder
        vocab_size = len(VOCAB)
        self.max_len = max_len

        self.decoder = CNNDecoder(dec_hidden_dim, latent_dim, vocab_size, max_len).to(device)

    def decode(self, latent: Tensor, gt_seq: Tensor = None) -> Tensor:
        """ Decoder workflow in VAE model

        Args:
            latent (Tensor): latent vector of shape `[batch, latent_dim]`
            gt_seq (Tensor): target sequence ids of shape `[batch, seq_len]`

        Returns:
            logits (Tensor): logits of vocabs of shape `[batch, vocab, seq_len]`
        """
        logits = self.decoder(latent)
        return logits

    def forward(self, seqs: List[str]):
        # Encoder
        latent, mu, logvar = self.encode(seqs)

        # Decoder
        input_ids = torch.tensor(
            convert_seqs2ids(seqs, add_sos=False, add_eos=False, max_length=self.max_len),
            dtype=torch.long,
            device=self.device,
        )   # [B, max_len]

        # Interpolative sampling
        if self.training and self.hparams.use_interp_sampling:
            z_i_rep = self.interpolation_sampling(latent)
            interp_z_rep = torch.cat((latent, z_i_rep), 0)
            logits = self.decode(interp_z_rep)
        else:
            logits = self.decode(latent)

        return logits, mu, logvar, input_ids, latent

    def generate_from_latent(self, latents: List[Tensor] | Tensor) -> List[str]:

        def main_process(latent: Tensor):
            pred_tokens = self.decoder.generate_from_latent(latent).tolist()
            seq = convert_ids2seqs(pred_tokens)
            return seq

        with torch.inference_mode():
            if isinstance(latents, Tensor):
                return main_process(latents)
            else:
                seqs = []
                for latent in latents:
                    seqs.extend(main_process(latent))
                return seqs
