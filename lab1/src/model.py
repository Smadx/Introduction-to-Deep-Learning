import torch.nn as nn

from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        # Activation function
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
        
        # Layers    
        self.layer_in = nn.Linear(1, cfg.hidden_size)
        self.feature_size = [cfg.hidden_size * (2 ** i) for i in range(cfg.n_muti_layers + 1)]

        self.layers = nn.ModuleList([])

        for i in range(cfg.n_muti_layers):
            self.layers.add_module(f"diff_layer_{i}", nn.Linear(self.feature_size[i], self.feature_size[i+1]))

        self.layers.add_module("mid_layer", nn.Linear(self.feature_size[-1], self.feature_size[-1]))

        self.feature_size = list(reversed(self.feature_size))
        for i in range(cfg.n_muti_layers):
            self.layers.add_module(f"coll_layer_{i}", nn.Linear(self.feature_size[i], self.feature_size[i+1]))

        self.layer_out = nn.Linear(cfg.hidden_size, 1)
        
        
    def forward(self, x):
        x = self.act_fn(self.layer_in(x))
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x))
        x = self.layer_out(x)
        return x
