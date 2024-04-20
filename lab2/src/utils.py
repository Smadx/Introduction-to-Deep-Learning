import dataclasses
import torch
import torchinfo
import torchvision
import numpy as np
import warnings

from dataclasses import dataclass
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from datetime import datetime
from pathlib import Path
from typing import Optional
from accelerate import Accelerator

@dataclass
class TrainConfig:
    in_channels: int
    norm_groups: int
    dropout_prob: float
    n_resnet_blocks: int
    batch_size: int
    lr: float
    seed: int
    results_path: str
    epochs: int


def print_model_summary(model, *, batch_size, shape, depth=4, batch_size_torchinfo=1):
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
        [(batch_size_torchinfo, *shape)],  # Input shape
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # no text output
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

def make_dataloader(set: Dataset, batch_size: int, k: float)-> DataLoader:
    """
    Splits the dataset into a training and validation set, and returns the corresponding dataloaders.

    Args:
        - set: the dataset to split
        - batch_size: the batch size
        - k: the fraction of the dataset to use for training

    Returns:
        - train_loader: the training dataloader
        - val_loader: the validation dataloader
    """
    N = len(set)
    train_size = int(k * N)
    val_size = N - train_size
    train_set, val_set = random_split(set, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def cycle(dl):
    # We don't use itertools.cycle because it caches the entire iterator.
    while True:
        for data in dl:
            yield data

def maybe_unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch
    else:
        return batch, None

def make_cifar(*, train, download):
    return CIFAR10(
        root="data",
        download=download,
        train=train,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
    )

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
