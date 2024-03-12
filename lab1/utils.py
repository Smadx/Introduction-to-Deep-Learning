import dataclasses
import torch
import torchinfo
import numpy as np
import warnings

from dataclasses import dataclass
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from datetime import datetime
from pathlib import Path
from typing import Optional
from accelerate import Accelerator

@dataclass
class TrainConfig:
    input_size: int
    hidden_size_1: int
    hidden_size_2: int
    batch_size: int
    lr: float
    seed: int
    weight_decay: float
    clip_grad_norm: bool
    evaluate: bool
    results_path: str
    data_path_x: str
    data_path_y: str
    epochs: int

def func(x):
    return np.log2(x) + np.cos(np.pi * x * 0.5)

class FunctionDataset(Dataset):
    def __init__(self, N: int):
        self.x = np.linspace(1, 16, N)
        self.y = func(self.x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def print_model_summary(model, *, batch_size, shape, depth=3, batch_size_torchinfo=1):
    # 打印模型概览
    summary = torchinfo.summary(
        model,
        [(batch_size_torchinfo, *shape)],  # 模型输入尺寸
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # 不显示额外信息
    )
    log(summary)
    if batch_size is None or batch_size == batch_size_torchinfo:
        return
    output_bytes_large = summary.total_output_bytes / batch_size_torchinfo * batch_size
    total_bytes = summary.total_input + output_bytes_large + summary.total_param_bytes
    log(
        f"\n--- With batch size {batch_size} ---\n"
        f"Forward/backward pass size: {output_bytes_large / 1e9:0.2f} GB\n"
        f"Estimated Total Size: {total_bytes / 1e9:0.2f} GB\n"
        + "=" * len(str(summary).splitlines()[-1])
        + "\n"
    )

def make_dataloader(N: int, batch_size: int)-> DataLoader:
    """
    从path中加载数据集,按照8:1:1比例划分训练集、验证集和测试集,并返回DataLoader
    """
    dataset = FunctionDataset(N)
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = n - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

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
    从args中初始化配置
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
