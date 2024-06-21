import dataclasses
import torch
import torchinfo
import numpy as np
import warnings
import scipy.sparse as sp

from dataclasses import dataclass
from torch import nn
from datetime import datetime
from pathlib import Path
from typing import Optional
from accelerate import Accelerator
from torch_geometric.datasets import Planetoid
from sklearn.metrics import roc_auc_score

@dataclass
class TrainConfig:
    dataset: str
    hidden_size: int
    num_layers: int
    pair_norm_scale: float
    dropedge_prob: float
    lr: float
    seed: int
    results_path: str
    epochs: int


def print_model_summary(model, *, node_shape, edge_shape, depth=3, dataset_name="Cora"):
    """
    Args:
        - model: the model to summarize
        - batch_size: the batch size to use for the summary
        - shape: the shape of the input tensor
        - depth: the depth of the summary
        - batch_size_torchinfo: the batch size to use for torchinfo
    """
    summary = torchinfo.summary(
        model,
        input_size=[(node_shape), (edge_shape)],
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # no text output
    )
    log(summary)
    output_bytes_large = summary.total_output_bytes 
    total_bytes = summary.total_input + output_bytes_large + summary.total_param_bytes
    log(
        f"\n--- With dataset {dataset_name} ---\n"
        f"Forward/backward pass size: {output_bytes_large / 1e9:0.2f} GB\n"
        f"Estimated Total Size: {total_bytes / 1e9:0.2f} GB\n"
        + "=" * len(str(summary).splitlines()[-1])
        + "\n"
    )

def drop_edge(edge_index, edge_attr=None, drop_prob=0.1):
    if drop_prob < 0.01:
        return edge_index, edge_attr

    edge_mask = torch.rand(edge_index.size(1)) > drop_prob
    edge_index = edge_index[:, edge_mask]

    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]

    return edge_index, edge_attr


def cycle(dl):
    # We don't use itertools.cycle because it caches the entire iterator.
    while True:
        for data in dl:
            yield data

def split_val(data, val_ratio: float = 0.2):
    """
    Splits the dataset into training and validation sets.

    Args:
        - data: the dataset
        - val_ratio: the ratio of validation nodes

    Returns:
        - the dataset with the train_mask and val_mask attributes set
    """
    num_nodes = data.y.size(0)
    val_size = int(num_nodes * val_ratio)  
    indices = torch.randperm(num_nodes)

    train_mask = data.train_mask.clone()
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # ensure that the validation set is not empty and disjoint from the test set
    train_indices = indices[~data.test_mask][val_size:]
    val_indices = indices[~data.test_mask][:val_size]

    train_mask.fill_(False)
    train_mask[train_indices] = True
    val_mask[val_indices] = True

    # check that the masks are disjoint
    assert not val_mask[data.test_mask].any()

    data.train_mask = train_mask
    data.val_mask = val_mask
    return data

def make_cora(cora_path: str):
    return Planetoid(root=cora_path, name="Cora")[0]

def make_citeseer(citeseer_path: str):
    return Planetoid(root=citeseer_path, name="CiteSeer")[0]

def create_edge_split(data, val_ratio=0.2, seed=123):
    """
    Splits the edges of the graph into training and validation sets.

    Args:
        - data: the dataset
        - val_ratio: the ratio of validation edges
    """
    edges = data.edge_index.t().numpy()
    total_edges = edges.shape[0]

    np.random.seed(seed)
    np.random.shuffle(edges)

    val_edge_num = int(total_edges * val_ratio)
    val_edges = edges[:val_edge_num]
    train_edges = edges[val_edge_num:]

    # 创建负样本
    adj_matrix = sp.coo_matrix((np.ones(total_edges), (data.edge_index[0], data.edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    adj_matrix = adj_matrix.tolil()
    adj_matrix[train_edges[:, 0], train_edges[:, 1]] = 0
    adj_matrix[train_edges[:, 1], train_edges[:, 0]] = 0
    negative_edges = np.row_stack(np.where(adj_matrix.toarray() == 0))
    np.random.shuffle(negative_edges)
    negative_edges = negative_edges[:val_edge_num].T

    return train_edges, val_edges, negative_edges
    
def compute_auc(model, data, val_edges, negative_edges):
    """
    Compute the AUC of the model on the validation set.

    Args:
        - model: the model
        - data: the dataset
        - val_edges: the validation edges
        - negative_edges: the negative edges
    """
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)  # 获取节点嵌入

        pos_scores = torch.sigmoid((z[val_edges[:, 0]] * z[val_edges[:, 1]]).sum(dim=1)) # positive edges
        neg_scores = torch.sigmoid((z[negative_edges[0]] * z[negative_edges[1]]).sum(dim=1)) # negative edges

        # AUC
        labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
        scores = torch.cat([pos_scores, neg_scores])
        auc_score = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
        return auc_score

def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path)
    log(f"Results will be saved to '{results_path}'")
    return results_path

@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module
    

def init_config_from_args(cls, args):
    """
    Initialize a dataclass from a Namespace.
    """
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})

_accelerator: Optional[Accelerator] = None


def init_logger(accelerator: Accelerator):
    global _accelerator
    if _accelerator is not None:
        raise ValueError("Accelerator already set")
    _accelerator = accelerator


def log(message):
    global _accelerator
    if _accelerator is None:
        warnings.warn("Accelerator not set, using print instead.")
        print_fn = print
    else:
        print_fn = _accelerator.print
    print_fn(message)
