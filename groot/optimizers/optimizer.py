import torch
from tqdm import tqdm
from omegaconf import DictConfig
from .smooth import SmoothInterface


class BaseOptimizer:
    def __init__(
        self,
        vae: torch.nn.Module,
        f: torch.nn.Module,
        config: DictConfig
    ):
        self.vae = vae
        self.f = f
        self.config = config

    def smooth_func(
        self,
        df,
        batch_size: int,
        device: torch.device,
    ):
        print("\nSmoothing f")
        self.f.train()
        smoother = SmoothInterface(self.f, self.config.smooth_config, device)
        existing_seqs = df["sequence"].tolist()
        y_train = torch.from_numpy(df["target"].to_numpy()).float()
        z_node = self.get_all_embeddings(existing_seqs, batch_size)
        self.f = smoother(z_node, y_train)
        self.f.eval()
        print("Done smoothing f\n")

    def get_all_embeddings(self, seqs, batch_size):
        all_z = []
        loader = torch.utils.data.DataLoader(seqs, batch_size=batch_size)
        for batch in loader:
            with torch.no_grad():
                z = self.encode_into_latent(batch)
            all_z.append(z)
        z = torch.cat(all_z, dim=0)
        return z

    def encode_seqs(self, pool_seqs):
        batch_size = 128
        all_z = []
        loader = torch.utils.data.DataLoader(pool_seqs, batch_size=batch_size)
        for batch in loader:
            with torch.no_grad():
                z = self.encode_into_latent(batch)
            all_z.append(z)
        z = torch.cat(all_z, dim=0)
        return z

    def optimize(self, pool_seqs):
        raise NotImplementedError

    def encode_into_latent(self, pool_seqs) -> torch.Tensor:
        z = self.vae.encode(pool_seqs)[0]
        return z

    def decode_into_sequences(self, z):
        return self.vae.generate_from_latent(z)


class GradientAscent(BaseOptimizer):

    def optimize(self, pool_seqs):
        z = self.encode_seqs(pool_seqs)
        z.requires_grad = True
        optimizer = torch.optim.Adam([z], lr=self.config.lr)
        self.f = self.f.to(z.device)
        for _ in tqdm(range(self.config.num_iteration), desc="Gradient Ascent"):
            optimizer.zero_grad()
            out = (-1) * self.f(z).sum()
            out.backward()
            optimizer.step()
        scores = self.f(z).view(-1)
        k = scores.size(0) if scores.size(0) < self.config.topK else self.config.topK
        top_scores, indices = torch.topk(scores, k)
        z = z[indices, :]

        if self.config.return_latent:
            return z.detach(), top_scores
        return self.decode_into_sequences(z), top_scores


class BFGS(BaseOptimizer):

from ..utils.graph_utils import build_knn_graph
            z = z.double()
            z.requires_grad = True
            optimizer = torch.optim.LBFGS([z], max_iter=self.config.num_iteration)
            optimizer.step(closure)
            all_z.append(z.detach())

        self.vae = self.vae.to(device).double()
        z = torch.cat(all_z, dim=0).to(device)
        scores = self.f(z).view(-1)
        k = scores.size(0) if scores.size(0) < self.config.topK else self.config.topK
        topk_scores, indices = torch.topk(scores, k)
        z = z[indices, :]
        if self.config.return_latent:
            return z.detach(), topk_scores
        return self.decode_into_sequences(z), topk_scores


class OptimizerInterface:

    def __init__(
        self,
        vae: torch.nn.Module,
        f: torch.nn.Module,
        config: DictConfig
    ):
        if config.algo_name == "gradient_ascent":
            self.optimizer = GradientAscent(vae, f, config)
        elif config.algo_name == 'LBFGS':
            self.optimizer = BFGS(vae, f, config)
        else:
            raise ValueError(f"{config.algo_name} is not supported.")

    def smooth_func(
        self,
        df,
        batch_size: int,
        device: torch.device,
    ):
        self.optimizer.smooth_func(df, batch_size, device)

    def optimize(self, pool_seqs):
        return self.optimizer.optimize(pool_seqs)
