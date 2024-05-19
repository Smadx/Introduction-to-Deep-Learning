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
    hidden_sizec: int
    hidden_sizel: int
    num_layers: int
    pair_norm_scale: float
    dropedge_prob: float
    lr: float
    seed: int
    results_path: str
    epochc: int
    epochl: int
    loop: bool


def print_model_summary(model, *, node_shape, edge, depth=3, dataset_name="Cora", device="cuda"):
    """
    Print a summary of the model, including the size of the forward/backward pass and the estimated total size.

    Args:
        - model: the model to summarize
        - node_shape: the shape of the node features
        - edge: the edge index
        - depth: the depth of the summary
        - dataset_name: the name of the dataset
    """
    summary = torchinfo.summary(
        model,
        input_data=(torch.randn(node_shape), edge),
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # no text output
        device=device,
    )
    log(summary)

def drop_edge(edge_index, edge_attr=None, drop_prob=0.1):
    """
    Drops edges from the edge index and edge attribute tensor.

    Args:
        - edge_index: the edge index tensor
        - edge_attr: the edge attribute tensor
        - drop_prob: the probability of dropping an edge
    """
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
    Splits the training set into new training and validation sets, ensuring no overlap with the test set.

    Args:
        data: the dataset, with pre-defined 'train_mask' and 'test_mask'.
        val_ratio: the fraction of the original training set to be used as the validation set.

    Returns:
        Modified dataset with updated 'train_mask' and new 'val_mask'.
    """
    train_indices = data.train_mask.nonzero(as_tuple=False).squeeze()
    shuffled_train_indices = train_indices[torch.randperm(train_indices.size(0))]

    val_size = int(len(shuffled_train_indices) * val_ratio)

    # Create new validation and training masks
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask = data.train_mask.clone()  

    val_mask[shuffled_train_indices[:val_size]] = True
    train_mask[shuffled_train_indices[:val_size]] = False

    data.val_mask = val_mask
    data.train_mask = train_mask

    return data

def make_cora(cora_path: str):
    return Planetoid(root=cora_path, name="Cora")[0]

def make_citeseer(citeseer_path: str):
    return Planetoid(root=citeseer_path, name="CiteSeer")[0]

def create_edge_split(data, val_ratio=0.2):
    """
    Splits the edges of the graph into training and validation sets, ensuring that only edges from the training set are considered.

    Args:
        - data: the dataset
        - val_ratio: the ratio of validation edges
    """
    node_mask = data.train_mask
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]

    # 应用边的训练掩码
    train_edges = data.edge_index[:, edge_mask].t().numpy()
    
    np.random.shuffle(train_edges)
    
    val_edge_num = int(len(train_edges) * val_ratio)
    val_edges = train_edges[:val_edge_num]
    train_edges = train_edges[val_edge_num:]

    # 创建负样本
    total_edges = data.edge_index.size(1)
    adj_matrix = sp.coo_matrix((np.ones(total_edges), (data.edge_index[0], data.edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    adj_matrix = adj_matrix.tolil()
    adj_matrix[val_edges[:, 0], val_edges[:, 1]] = 0
    adj_matrix[val_edges[:, 1], val_edges[:, 0]] = 0
    negative_edges = np.row_stack(np.where(adj_matrix.toarray() == 0))
    np.random.shuffle(negative_edges)
    negative_edges = negative_edges[:len(train_edges)].T

    return train_edges, val_edges, negative_edges

def test_edge_split(data):
    """
    Splits the edges of the graph into training and validation sets, ensuring that only edges from the training set are considered.

    Args:
        - data: the dataset
    """
    node_mask = data.train_mask
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]

    # 应用边的训练掩码
    train_edges = data.edge_index[:, edge_mask].t().numpy()
    
    node_test = data.test_mask
    edge_mask_test = node_test[data.edge_index[0]] & node_test[data.edge_index[1]]
    test_edges = data.edge_index[:, edge_mask_test].t().numpy()


    # 创建负样本
    total_edges = data.edge_index.size(1)
    adj_matrix = sp.coo_matrix((np.ones(total_edges), (data.edge_index[0], data.edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    adj_matrix = adj_matrix.tolil()
    adj_matrix[test_edges[:, 0], test_edges[:, 1]] = 0
    adj_matrix[test_edges[:, 1], test_edges[:, 0]] = 0
    negative_edges = np.row_stack(np.where(adj_matrix.toarray() == 0))
    np.random.shuffle(negative_edges)
    negative_edges = negative_edges[:len(train_edges)].T

    return train_edges, test_edges, negative_edges
    
def compute_auc(pos_scores, neg_scores):
    """
    Compute the AUC of the model on the validation set.

    Args:
        - pos_scores: the scores of the positive edges
        - neg_scores: the scores of the negative edges
    """
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
