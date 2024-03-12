import torch.nn as nn

from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1 = nn.Linear(cfg.input_size, cfg.hidden_size_1)
        self.l2 = nn.Linear(cfg.hidden_size_1, cfg.hidden_size_2)
        self.l3 = nn.Linear(cfg.hidden_size_2, cfg.input_size)
        if cfg.act_fn == "sigmoid":
            self.act_fn = F.sigmoid
        elif cfg.act_fn == "relu":
            self.act_fn = F.relu
        elif cfg.act_fn == "tanh":
            self.act_fn = F.tanh
        elif cfg.act_fn == "silu":
            self.act_fn = F.silu
        else:
            raise ValueError("Not supported activation function")
        
    def forward(self, x):
        x = self.l1(x)
        x = self.act_fn(x)
        x = self.l2(x)
        x = self.act_fn(x)
        x = self.l3(x)
        return x
