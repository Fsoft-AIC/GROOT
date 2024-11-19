import random
import torch
import torch.nn as nn
from torch import Tensor
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics import MeanMetric, MinMetric
from typing import List
from .modules import (
    ESM2Encoder,
    LatentEncoder,
    DropoutPredictor,
    PositionalPIController,
)
from .train_helper import KLDivergence
from ..common.constants import get_token2id


class BaseVAE(LightningModule):

    def __init__(
        self,
        expected_kl: float,
        pretrained_encoder_path: str = "facebook/esm2_t12_35M_UR50D",
        num_unfreeze_layers: int = 0,
        latent_dim: int = 64,
        pred_hidden_dim: int = 128,
        pred_dropout: float = 0.2,
        nll_weight: float = 1.0,
        mse_weight: float = 1.0,
        kl_weight: float = 1.0,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        Kp: float = 0.01,
        Ki: float = 0.0001,
        lr: float = 0.001,
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
        pred_num_layers: int = 1,
    ):
        super(BaseVAE, self).__init__()

        self.save_hyperparameters(ignore=["device"])

        # Interpolation
        self.interp_ids = None

        # Encoder
        self.encoder = ESM2Encoder(pretrained_encoder_path)
        enc_dim = self.encoder.hidden_dim

        # Latent encoder
        self.latent_encoder = LatentEncoder(latent_dim, enc_dim)
        self.glob_attn_module = nn.Sequential(nn.Linear(enc_dim, 1), nn.Softmax(1))

        # Predictor
        self.predictor = DropoutPredictor(latent_dim, pred_hidden_dim, pred_dropout)
        # self.predictor = LDEPredictor(latent_dim, pred_num_layers, pred_hidden_dim)

        self.pi_controller = PositionalPIController(expected_kl, kl_weight,
                                                    beta_min, beta_max, Kp, Ki)

        # Loss functions
        token2id = get_token2id()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.recon_loss = nn.CrossEntropyLoss(ignore_index=token2id["<pad>"],
                                              reduction=reduction)
        self.kl_div = KLDivergence(reduction)
        self.neg_loss = nn.MSELoss(reduction=reduction)

        # Metrics
        self.train_total_loss = MeanMetric()
        self.train_kl_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_mse_loss = MeanMetric()
        self.train_interp_loss = MeanMetric()
        self.train_neg_loss = MeanMetric()
        self.train_latent_loss = MeanMetric()

        self.valid_total_loss = MeanMetric()
        self.valid_kl_loss = MeanMetric()
        self.valid_recon_loss = MeanMetric()
        self.valid_mse_loss = MeanMetric()
        self.valid_latent_loss = MeanMetric()

        self.valid_best_loss = MinMetric()

    def freeze_encoder(self) -> None:
        last_freeze_layer = self.encoder.num_hidden_layers - self.hparams.num_unfreeze_layers - 1
        for name, param in self.encoder.model.named_parameters():
            if "encoder.layer" in name:
                if int(name.split(".")[2]) <= last_freeze_layer:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True

    def encode(self, x: List[str]):
        """ Encoder workflow in VAE model

        Args:
            x (List[str]): A list of protein sequence strings

        Returns:
            latent (Tensor): Latent vector of shape `[batch, latent_dim]`
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
        """
        enc_inp = self.encoder.tokenize(x).to(self.device)
        enc_out = self.encoder(enc_inp)  # [B, L, D]

        global_enc_out = self.glob_attn_module(enc_out)
        z_rep = torch.bmm(global_enc_out.transpose(1, 2), enc_out).squeeze(1)
        z, mu, logsigma = self.latent_encoder(z_rep)
        return z, mu, logsigma

    def predict(self, mu: Tensor) -> Tensor:
        """ Property prediction in VAE model

        Args:
            mu (Tensor): mu vector of shape `[batch, latent_dim]`

        Returns:
            prop (Tensor): Property value of shape `[batch, 1]`
        """
        prop = self.predictor(mu)
        return prop

    def forward(self, seqs: List[str]):
        """Forward workflow of VAE

        Args:
            seqs: List[str]: list of protein sequences

        Returns:
            dec_probs (Tensor): log probability of vocabs of shape [batch, vocab, seq_len]
            pred_property (Tensor): Property value of shape [batch, 1]
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
            seqs_ids (Tensor): decoder input ids of shape `[batch, maxlen]`
        """
        raise NotImplementedError

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return {"optimizer": optimizer}

    def loss_function(self, pred_seq: Tensor, target_seq: Tensor,
                      mu: Tensor, logvar: Tensor, latent: Tensor):
        """Measure loss and control kl weight

        Args:
            pred_seq (Tensor): prob, output of decoder of shape `[batch, vocab, seq_len]`
            target_seq (Tensor): target sequence to reconstruct of shape `[batch, seq_len]`
            pred_prop (Tensor): predicted property values of shape `[batch, 1]`
            target_prob (Tensor): target property values of shape `[batch, 1]`
            mu (Tensor): mean of latent z of shape `[batch, latent_dim]`
            logvar (Tensor): log-variance of latent z of shape `[batch, latent_dim]`

        Returns:
            total_loss (Tensor): total loss, scalar
            kl_loss (Tensor): Kullback-Leibler divergence, scalar
            recon_loss (Tensor): reconstruction loss, scalar
            mse_loss (Tensor): MSE loss, scalar
        """
        # KL divergence
        kl_loss = self.kl_div(mu, logvar)

        # Reconstruction loss
        if self.training and self.hparams.use_interp_sampling:
            hyp_seq = pred_seq[:-self.hparams.interp_size]
        else:
            hyp_seq = pred_seq
        recon_loss = self.recon_loss(hyp_seq, target_seq)

        # MSE (predictor) loss
        # if self.training and self.hparams.use_neg_sampling:
        #     hyp_prop = pred_prop[:-self.hparams.neg_size]
        #     extend_prop = pred_prop[-self.hparams.neg_size:]
        # else:
        #     hyp_prop = pred_prop
        # mse_loss = self.mse_loss(hyp_prop, target_prob)

        # Interpolation loss
        bs = mu.size(0)
        if self.training and self.hparams.use_interp_sampling:
            seq_preds = nn.functional.gumbel_softmax(pred_seq, tau=1, dim=1, hard=True)
            seq_preds = seq_preds.transpose(1, 2).flatten(1, 2)
            seq_dist_mat = torch.cdist(seq_preds, seq_preds, p=1)

            ext_ids = torch.arange(bs, bs + self.hparams.interp_size)
            tr_dists = seq_dist_mat[self.interp_ids[:, 0], self.interp_ids[:, 1]]
            inter_dist1 = seq_dist_mat[ext_ids, self.interp_ids[:, 0]]
            inter_dist2 = seq_dist_mat[ext_ids, self.interp_ids[:, 1]]

            interp_loss = 0.5 * (inter_dist1 + inter_dist2) - 0.5 * tr_dists
            interp_loss = interp_loss.mean() \
                if self.hparams.reduction == "mean" else interp_loss.sum()
            interp_loss = max(0, interp_loss) * self.hparams.interp_weight
        else:
            interp_loss = 0.0

        # Negative sampling loss
        if self.training and self.hparams.use_neg_sampling:
            neg_targets = torch.ones(
                (self.hparams.neg_size), device=self.device) * self.hparams.neg_floor
            neg_loss = self.neg_loss(extend_prop.flatten(), neg_targets.flatten())
        else:
            neg_loss = 0.0

        # Latent regularization
        if self.hparams.regularize_latent:
            latent_loss = 0.5 * torch.linalg.vector_norm(latent, 2, dim=1)**2
            latent_loss = latent_loss.mean() \
                if self.hparams.reduction == "mean" else latent_loss.sum()
        else:
            latent_loss = 0.0

        total_loss = self.hparams.kl_weight * kl_loss \
            + self.hparams.nll_weight * recon_loss \
            + self.hparams.interp_weight * interp_loss \
            + self.hparams.neg_weight * neg_loss \
            + self.hparams.latent_weight * latent_loss

        if self.training:
            # Control kl weight
            self.hparams.kl_weight = self.pi_controller(kl_loss.detach())

        return total_loss, kl_loss, recon_loss, interp_loss, neg_loss, latent_loss

    def predict_property_from_latent(self, mu: Tensor) -> float:
        return self.predictor(mu)

    def model_step(self, batch):
        x, y = batch["sequences"], batch["fitness"]
        y = y.unsqueeze(1)
        dec_probs, mu, logvar, seqs_ids, latent = self.forward(x)

        loss, kl_loss, recon_loss, interp_loss, neg_loss, latent_loss = \
            self.loss_function(dec_probs, seqs_ids, mu, logvar, latent)

        return loss, kl_loss, recon_loss, interp_loss, neg_loss, latent_loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, kl_loss, recon_loss, interp_loss, neg_loss, latent_loss = self.model_step(batch)

        # update and log metrics
        self.train_total_loss(loss)
        self.train_kl_loss(kl_loss)
        self.train_recon_loss(recon_loss)
        self.train_interp_loss(interp_loss)
        self.train_neg_loss(neg_loss)
        self.train_latent_loss(latent_loss)

        self.log("train_loss", self.train_total_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_kldiv", self.train_kl_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_recon_loss", self.train_recon_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_interp_loss", self.train_interp_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_neg_loss", self.train_neg_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_latent_loss", self.train_latent_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_kl_weight", self.hparams.kl_weight, on_step=True,
                 on_epoch=False, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, kl_loss, recon_loss, _, _, latent_loss = self.model_step(batch)

        # update and log metrics
        self.valid_total_loss(loss)
        self.valid_kl_loss(kl_loss)
        self.valid_recon_loss(recon_loss)
        self.valid_latent_loss(latent_loss)

        self.log("valid_loss", self.valid_total_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_kldiv", self.valid_kl_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_recon_loss", self.valid_recon_loss, on_step=False,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("valid_latent_loss", self.valid_latent_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_start(self) -> None:
        self.valid_total_loss.reset()
        self.valid_kl_loss.reset()
        self.valid_mse_loss.reset()
        self.valid_recon_loss.reset()
        self.valid_latent_loss.reset()
        self.valid_best_loss.reset()

    def on_validation_epoch_end(self) -> None:
        cur_valid_loss = self.valid_total_loss.compute()
        self.valid_best_loss(cur_valid_loss)
        self.log("valid_best_loss", self.valid_best_loss.compute(), sync_dist=True, prog_bar=True)

    def generate_from_latent(latent: Tensor):
        raise NotImplementedError

    def predict_fitness_with_scale(self,
                                   seqs: List[str],
                                   scale: float,
                                   factor: float,
                                   i: int):
        with torch.inference_mode():
            latent, *_ = self.encode_with_scale(seqs, scale, factor, i)
            scores = self.predict_property_from_latent(latent)
            return scores.squeeze().cpu().tolist()

    def reconstruct_from_wt(self,
                            wt_seq: List[str],
                            scale: float,
                            factor: float,
                            i: int) -> List[str]:
        """Reconstruct from wild-type sequence(s)"""
        with torch.inference_mode():
            latent, *_ = self.encode_with_scale(wt_seq, scale, factor, i)
            seqs = self.generate_from_latent(latent)
            return seqs

    def reconstruct_from_wt_glob(self,
                                 wt_seq: List[str],
                                 scale: float,
                                 factor: float,
                                 i: int,
                                 batch_size: int) -> List[str]:
        wt_seq_chunk = [wt_seq[i:i + batch_size] for i in range(0, len(wt_seq), batch_size)]
        new_seqs = []
        for wt_seqs in wt_seq_chunk:
            seqs = self.reconstruct_from_wt(wt_seqs, scale, factor, i)
            new_seqs.extend(seqs)
        return new_seqs

    def encode_with_scale(self, x: List[str], scale: float, factor: float, i: int):
        """ Encoder workflow in VAE model

        Args:
            x (List[str]): A list of protein sequence strings

        Returns:
            latent (Tensor): Latent vector of shape `[batch, latent_dim]`
            mu (Tensor): mean vector of shape `[batch, latent_dim]`
            logvar (Tensor): logvar vector of shape `[batch, latent_dim]`
        """
        with torch.inference_mode():
            enc_inp = self.encoder.tokenize(x).to(self.device)
            enc_out = self.encoder(enc_inp)  # [B, L, D]

            global_enc_out = self.glob_attn_module(enc_out)
            z_rep = torch.bmm(global_enc_out.transpose(1, 2), enc_out).squeeze(1)
            z, mu, logsigma = self.latent_encoder(z_rep)

            if random.random() > 0.5:
                num_random = int(0.2 * z.shape[0])
                eps = torch.randn_like(mu[-num_random:])
                z[-num_random:] = z[-num_random:] + (scale - factor * i) * eps
                # z[:num_random] = (1 - scale) * z[:num_random] + scale * eps

            return z, mu, logsigma

    def interpolation_sampling(self, z_rep: Tensor):
        """Get interpolations between z_reps in batch"""
        z_dist_mat = self.pairwise_l2(z_rep)
        k_val = min(len(z_rep), 2)
        _, z_nn_ids = z_dist_mat.topk(k_val, largest=False)
        z_nn_ids = z_nn_ids[:, 1]

        z_nn = z_rep[:, 1].unsqueeze(1)
        z_interp = (z_rep + z_nn) / 2

        subset_ids = torch.randperm(len(z_rep), device=self.device)[:self.hparams.interp_size]
        sub_z_interp = z_interp[subset_ids]
        sub_nn_ids = z_nn_ids[subset_ids]

        self.interp_ids = torch.cat((subset_ids.unsqueeze(1), sub_nn_ids.unsqueeze(1)), dim=1)
        return sub_z_interp

    def add_negative_samples(self, z_rep):
        max2norm = torch.norm(z_rep, p=2, dim=1).max()
        rand_ids = torch.randperm(len(z_rep))
        if self.hparams.neg_focus:
            neg_z = 0.5 * torch.randn_like(z_rep)[:self.hparams.neg_size] \
                + z_rep[rand_ids][:self.hparams.neg_size]
            neg_z = neg_z / torch.linalg.vector_norm(neg_z, 2, dim=1).reshape(-1, 1)
            neg_z = neg_z * (max2norm * self.hparams.neg_norm)
        else:
            center = z_rep.mean(0, keepdims=True)
            dist_set = z_rep - center

            # gets maximally distant rep from center
            dist_sort = torch.norm(dist_set, 2, dim=1).reshape(-1, 1).sort().indices[-1]
            max_dist = dist_set[dist_sort]
            adj_dist = self.hparams.neg_norm * max_dist.repeat(len(z_rep), 1) - dist_set
            neg_z = z_rep + adj_dist
            neg_z = neg_z[rand_ids][:self.hparams.neg_size]

        return neg_z

    def pairwise_l2(self, x):
        bs = x.size(0)
        z1 = x.unsqueeze(0).expand(bs, -1, -1)
        z2 = x.unsqueeze(1).expand(-1, bs, -1)
        dist = torch.pow(z2 - z1, 2).sum(2)
        return dist
