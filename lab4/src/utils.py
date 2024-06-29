import dataclasses
import torch
import numpy as np
import warnings
import json

from datasets import Dataset
from dataclasses import dataclass
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from typing import Optional
from accelerate import Accelerator

@dataclass
class TrainConfig:
    model_dir: str
    tokenizer_dir: str
    results_path: str
    task_type: str
    task: str
    num_classes: int
    r: int
    lora_alpha: int
    lora_dropout: float
    load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: torch.dtype
    prompt: str
    batch_size: int
    seed: int
    lr: float
    epochs: int

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)
    
dtype_mapping = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def make_dataset(data: Dataset, tokenizer, batch_size: int, prompt: str = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    def tokenize_function(examples):
        sentences = [f"{prompt} Sentence 1: {s1} Sentence 2: {s2}" for s1, s2 in zip(examples["sentence1"], examples["sentence2"])]
        return tokenizer(sentences, truncation=True, max_length=None)
    
    tokenized_data = data.map(tokenize_function, batched=True, remove_columns=["idx", "sentence1", "sentence2"])
    tokenized_data = tokenized_data.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    
    # 创建DataLoader
    train_loader = DataLoader(tokenized_data["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    val_loader = DataLoader(tokenized_data["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
    test_loader = DataLoader(tokenized_data["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
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