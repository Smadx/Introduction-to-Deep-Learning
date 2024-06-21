import torch
from torch import nn, mean, pow, sqrt
import torch.nn.functional as F
import numpy as np


# GCN模型
class GCN(nn.Module):
    def __init__(self, in_channels, num_classes, cfg, classifier=True):
        super(GCN, self).__init__()
        self.conv_in = GCNConv(in_channels, cfg.hidden_size)
        self.convs = nn.ModuleList([])
        for i in range(cfg.num_layers - 1):
            ch_in = cfg.hidden_size * (2 ** i)
            ch_out = cfg.hidden_size * (2 ** (i + 1))
            self.convs.append(GCNConv(ch_in, ch_out))
        self.fc = nn.Linear(cfg.hidden_size * (2 ** (cfg.num_layers - 1)), num_classes)
        if cfg.pair_norm_scale is not None:
            self.norm = PairNorm(cfg.pair_norm_scale)
        self.classifier = classifier

    def forward(self, x, adj):
        x = F.relu(self.conv_in(x, adj))
        if hasattr(self, "norm"):
            x = self.norm(x)
        for conv in self.convs:
            x = F.relu(conv(x, adj))
        if self.classifier:
            x = F.log_softmax(self.fc(x), dim=1)
        return x

class GCNConv(nn.Module):
    """
    The GCN layer

    Args:
        - in_features: the number of input features
        - out_features: the number of output features
    """
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
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