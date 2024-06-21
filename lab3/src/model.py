import torch
from torch import nn, mean, pow, sqrt
import torch.nn.functional as F
import torch.sparse as sparse
import numpy as np


# GCN模型
class GCN(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_size, cfg, classifier=True,loop=False):
        super(GCN, self).__init__()
        self.conv_in = GCNConv(in_channels, hidden_size, loop=loop)
        self.convs = nn.ModuleList([])
        for i in range(cfg.num_layers - 1):
            ch_in = int(hidden_size / (2 ** i))
            ch_out = int(hidden_size / (2 ** (i + 1)))
            self.convs.append(GCNConv(ch_in, ch_out, loop=loop))
        self.out = GCNConv(int(hidden_size / (2 ** (cfg.num_layers - 1))), num_classes, loop=loop)
        if cfg.pair_norm_scale is not None:
            self.norm = PairNorm(cfg.pair_norm_scale)
        self.classifier = classifier

    def forward(self, x, edge_index):
        x = F.relu(self.conv_in(x, edge_index))
        if hasattr(self, "norm"):
            x = self.norm(x)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.out(x, edge_index)
        if self.classifier:
            x = F.log_softmax(x, dim=1)
        return x

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, loop: bool):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        self.loop = loop

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):

        num_nodes = x.size(0)

        if self.loop:
            loop_index = torch.arange(0, num_nodes, device=edge_index.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        values = torch.ones(edge_index.size(1), device="cuda")
        adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes), dtype=torch.float)

        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)

        return output

class PairNorm(torch.nn.Module):
    """
    The pair normalization layer

    Args:
        - scale: the scale parameter
    """
    def __init__(self, scale=1):
        super(PairNorm, self).__init__()
        self.scale = scale

    def forward(self, x):
        mean_x = x.mean(dim=0, keepdim=True)
        x = x - mean_x
        std_x = x.pow(2).mean().sqrt()
        x = self.scale * x / std_x
        return x