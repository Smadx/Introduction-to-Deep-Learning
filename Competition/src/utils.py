import dataclasses
import torch
import numpy as np
import pandas as pd
import warnings

from dataclasses import dataclass
from torch import nn, LongTensor
from torch.nn import functional as F
from torch_sparse import SparseTensor
from datetime import datetime
from pathlib import Path
from typing import Optional
from accelerate import Accelerator

@dataclass
class TrainConfig:
    data_dir: str
    embedding_dim: int
    num_layers: int
    results_path: str
    seed: int
    epochs: int
    check_step: int
    batch_size: int
    lr: float
    K: int
    lambda_: float

class BPRLoss(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0):
        reg_loss = self.lambda_ * (users_emb_0.norm(2).pow(2) + pos_items_emb_0.norm(2).pow(2) + neg_items_emb_0.norm(2).pow(2)) # L2 loss

        pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
        pos_scores = torch.sum(pos_scores, dim=-1)
        neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
        neg_scores = torch.sum(neg_scores, dim=-1)
        
        loss = -F.logsigmoid(pos_scores - neg_scores).sum() + reg_loss
        
        return loss

def collate_fn(batch):
    edge_index = torch.stack(batch, dim=1) if isinstance(batch[0], torch.Tensor) else torch.tensor(batch)
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index should have shape [2, num_messages]")
    return edge_index

def make_dataset(data_dir: str) -> tuple[LongTensor, LongTensor, SparseTensor, SparseTensor, LongTensor]:
    train_df = pd.read_csv(data_dir + "train_data.csv")
    val_df = pd.read_csv(data_dir + "val_data.csv")

    train_edge_index = LongTensor(train_df[["user_id", "item_id"]].values.T)
    val_edge_index = LongTensor(val_df[["user_id", "item_id"]].values.T)

    edge_index = torch.cat((train_edge_index, val_edge_index), 1)

    train_sparse_tensor = SparseTensor(row=train_edge_index[0], col=train_edge_index[1] + 53424, sparse_sizes=(53424 + 10000, 53424 + 10000))
    val_sparse_tensor = SparseTensor(row=val_edge_index[0], col=val_edge_index[1] + 53424, sparse_sizes=(53424 + 10000, 53424 + 10000))

    return train_edge_index, val_edge_index, train_sparse_tensor, val_sparse_tensor, edge_index

def RecallPrecision_at_K(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])
    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    
    return recall.item(), precision.item()

def NDCG_at_K(groundTruth, r, k):
    assert len(r) == len(groundTruth)
    test_matrix = torch.zeros((len(r), k))
    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)
    dcg = r * (1. / torch.log2(torch.arange(2, k + 2)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    
    return torch.mean(ndcg).item()

def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
        
    return user_pos_items

def get_metrics(model, edge_index, sparse_edge_index, exclude_edge_index, train_sparse_edge_index, k):
    
    user_embedding, _, item_embedding, _  = model.forward(train_sparse_edge_index)
    
    user_embedding = np.array(user_embedding.cpu().detach().numpy())
    item_embedding = np.array(item_embedding.cpu().detach().numpy())
                  
    rating = torch.tensor(np.matmul(user_embedding, item_embedding.T))
    
    user_pos_items = get_user_positive_items(exclude_edge_index)
    exclude_users = []
    exclude_items = []
    for user, items in user_pos_items.items():
        exclude_users.extend([user] * len(items))
        exclude_items.extend(items)
    rating[exclude_users, exclude_items] = -(1 << 10)
    
    _, top_K_items = torch.topk(rating, k=k)
    
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]
    
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))
    
    recall, precision = RecallPrecision_at_K(test_user_pos_items_list, r, k)
    ndcg = NDCG_at_K(test_user_pos_items_list, r, k)

    return recall, precision, ndcg

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