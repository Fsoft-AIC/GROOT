import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import tqdm
from .label_prop import LabelPropagation
from ..utils.graph_utils import build_knn_graph


class BaseSmooth:
    def __init__(self, f, config, device):
        self.f = f
        self.config = config
        self.device = device
        self.smooth_range = config.range

    def propagate_label(self, z_node, y_train, edge_index, edge_weight, train_index):
        raise NotImplementedError

    def train_surrogate_function(self, z_node, y_soft):
        self.f = self.f.to(z_node.device)
        self.f.train()
        optimizer = torch.optim.Adam(self.f.parameters(), lr=self.config.lr)
        z_node = z_node.to(dtype=torch.float32)
        for _ in tqdm(range(self.config.num_epoch)):
            optimizer.zero_grad()
            out = self.f(z_node)
            loss = F.mse_loss(out.view(-1), y_soft.view(-1))
            loss.backward()
            optimizer.step()
        print("Done training")
        return self.f

    def __call__(self, z, y):
        z_node, edge_index, edge_weight, train_index, test_index = build_knn_graph(
            z, self.config.num_node,
            self.config.num_neighbor,
            self.config.dist_metric,
            self.device,
            self.smooth_range,
        )
        z_node = z_node.to(dtype=torch.float32)
        print("Num graph nodes: ", z_node.shape[0])
        self.model = self.model.to(z_node.device)
        self.f = self.f.to(z_node.device)
        y = y.to(z_node.device).unsqueeze(-1)

        y_soft = self.propagate_label(z_node, y, edge_index, edge_weight, train_index)

        self.f = self.train_surrogate_function(z_node, y_soft)
        self.f.eval()
        with torch.no_grad():
            print(
                "Final loss: ",
                float(
                    F.mse_loss(self.f(z_node[train_index]),
                               y_soft[train_index])))
        return self.f


class LabelPropagationSmooth(BaseSmooth):

    def __init__(self, f, config, device):
        super(LabelPropagationSmooth, self).__init__(f, config, device)
        self.model = LabelPropagation(num_layers=config.num_layers,
                                      alpha=config.alpha)

    def propagate_label(self, z_node, y_train, edge_index, edge_weight, train_index):
        z_node = z_node.to(dtype=torch.float32)
        y_soft = self.f(z_node)

        # init nodes
        # y = torch.zeros_like(y_soft)
        y = torch.fill(torch.zeros_like(y_soft), torch.mean(y_train))
        y[train_index] = y_train.to(dtype=y.dtype)
        N = z_node.shape[0]
        adj_t = SparseTensor(row=edge_index[0],
                             col=edge_index[1],
                             sparse_sizes=(N, N)).t()

        # infer labels to new nodes
        y_soft = self.model(y, adj_t, mask=train_index, edge_weight=edge_weight)
        if not self.config.smooth_all:
            y_soft[train_index] = y_train.to(dtype=y.dtype)

        return y_soft


class SmoothInterface:

    def __init__(self, f, config, device):
        if config.algo == "lp":
            self.method = LabelPropagationSmooth(f, config, device)
        else:
            raise ValueError(f"{config.algo} is not supported.")

    def __call__(self, z, y):
        f = self.method(z, y)
        return f
