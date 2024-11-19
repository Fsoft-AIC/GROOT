import numpy as np
import torch
import torch_geometric.utils as pyg_utils
import torch.nn.functional as F
from torch import Tensor


def cdist(x: Tensor, metric: str = "euclidean"):
    if metric == "euclidean":
        dist = torch.cdist(x, x, p=2)
    elif metric == "cosine":
        dist = 1 - F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
    elif metric == "correlation":
        dist = 1 - torch.corrcoef(x)
    else:
        raise ValueError("Supported metrics are 'euclidean', 'cosine', and 'correlation'.")

    return dist


def interpolate_new_nodes(nodes: Tensor, N: int, range: list):
    ids = torch.randint(0, nodes.size(0), size=(N,))
    selected_nodes = nodes[ids]
    beta = torch.distributions.Uniform(range[0], range[1]).sample((N, 1)).to(selected_nodes.device)
    new_nodes = beta * selected_nodes + (1 - beta) * torch.randn_like(selected_nodes)
    return new_nodes


def get_nearest_neighbors_from_dist(
    dist: Tensor,
    num_neighbor: int,
    return_weight: bool,
    use_euclidean: bool,
    device: torch.device
):
    dist = torch.from_numpy(dist).to(device) if isinstance(dist, np.ndarray) else dist
    rank = torch.argsort(dist, dim=1)[:, :num_neighbor]
    dist = torch.gather(dist, dim=1, index=rank)

    edge_weight = None
    if return_weight:
        edge_weight = dist.view(-1, 1)
        if use_euclidean:
            edge_weight = 1 / (edge_weight + 1e-6)
        else:
            edge_weight = 1 - edge_weight

    rows = []
    cols = []
    for i in range(rank.size(0)):
        rows.extend(rank[i].tolist())
        cols.extend([i for _ in range(num_neighbor)])
    edge_index = torch.stack([torch.tensor(rows, device=device),
                              torch.tensor(cols, device=device)], dim=0)

    return edge_index, edge_weight


def build_knn_graph(z, N, num_neighbor, metric, device, range):
    n = z.size(0)
    z_new = interpolate_new_nodes(z, N - n, range)
    nodes = torch.cat([z, z_new], dim=0).to(device)

    use_euclidean = True if metric == "euclidean" else False
    dist = cdist(nodes, metric=metric)
    dist = dist + torch.eye(nodes.size(0), dtype=dist.dtype, device=dist.device) * 1e6
    edge_index, edge_weight = get_nearest_neighbors_from_dist(
        dist, num_neighbor, True, use_euclidean, device
    )

    if edge_weight is None:
        edge_index = pyg_utils.to_undirected(edge_index)
    else:
        edge_index, edge_weight = pyg_utils.to_undirected(edge_index, edge_weight, reduce="mean")
    train_index = torch.arange(0, n)
    test_index = torch.arange(n, N)
    return nodes, edge_index, edge_weight, train_index, test_index
