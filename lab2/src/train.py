import argparse

import torch
import yaml
import os
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from utils import (
    TrainConfig,
    make_dataloader,
    make_cifar,
    cycle,
    maybe_unpack_batch,
    init_logger,
    init_config_from_args,
    log,
    print_model_summary,
    handle_results_path,
)

from model import CNN

def main():
    parser = argparse.ArgumentParser()

    # Architecture
    parser.add_argument("--in-channels", type=int, default=128)
    parser.add_argument("--norm-groups", type=int, default=32)
    parser.add_argument("--dropout-prob", type=float, default=0.5)
    parser.add_argument("--n-resnet-blocks", type=int, default=3)

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)

    init_logger(accelerator)
    cfg = init_config_from_args(TrainConfig, args)

    model = CNN(cfg)
    print_model_summary(model, batch_size=cfg.batch_size,shape=(3, 32, 32))
    with accelerator.local_main_process_first():
        train_loader, val_loader = make_dataloader(make_cifar(train=True, download=True), cfg.batch_size, 0.8)
    log(f"Train on {accelerator.device}")
    Trainer(
        model,
        train_loader,
        accelerator,
        make_opt=lambda params: torch.optim.Adam(params, lr=cfg.lr),
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
    ).train()
    Evaluator(
        model,
        val_loader,
        accelerator,
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
    ).evaluate()

class Trainer:
    """
    Trainer of the model

    Args:
        - model: the model to train
        - train_loader: the training data loader
        - accelerator: the accelerator
        - make_opt: a function that takes the model parameters and returns an optimizer
        - config: the configuration
        - results_path: the path to save the results
        - var_loader: optional, the validation data loader
    """
    def __init__(self, model, train_loader, accelerator, make_opt, config, results_path):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.train_loader = cycle(train_loader)
        self.accelerator = accelerator
        self.opt = accelerator.prepare(make_opt(self.model.parameters()))
        self.critierion = torch.nn.CrossEntropyLoss()
        self.cfg = config
        self.step = 0
        self.train_num_steps = config.epochs * len(train_loader)
        self.results_path = results_path
        self.device = self.accelerator.device
        print('Train on', self.device)
        self.model.to(self.device)
        self.checkpoint_path = self.results_path / f"model.pt"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                image, label = maybe_unpack_batch(next(self.train_loader))
                image, label = image.to(self.device), label.to(self.device)
                self.opt.zero_grad()
                y = self.model(image)
                loss = self.critierion(y, label)
                self.accelerator.backward(loss)
                self.opt.step()
                pbar.set_description(f'Loss: {loss.item():.4f}')
                loss_list.append(loss.item())
                self.step += 1
                pbar.update()

        
        with open(self.results_path / 'loss_list.txt', 'w') as f:
            print(loss_list, file=f)
        plt.plot(loss_list)
        plt.savefig(self.results_path / 'loss_list.png')
        
        self.save()

    def save(self):
        """
        Save model to checkpoint_path
        """
        self.model.eval()
        checkpoint_path = Path(self.checkpoint_path)
        checkpoint_dir = checkpoint_path.parent

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint = {
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        log(f"Saved model to {checkpoint_path}")
    
class Evaluator:
    """
    Evaluate the model on the validation set

    Args:
        - model: the model to evaluate
        - val_loader: the validation data loader
        - accelerator: the accelerator
        - config: the configuration
        - results_path: the path to save the results
    """
    def __init__(self, model, val_loader, accelerator, config, results_path):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.cfg = config
        self.device = self.accelerator.device
        self.model.to(self.device)
        self.results_path = results_path
        self.checkpoint_path = results_path / "model.pt"

    def evaluate(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        y_pred = []
        label_list = []
        loss_list = []
        for image, label in self.val_loader:
            image, label = image.to(self.device), label.to(self.device)
            with torch.no_grad():
                y = self.model(image)
            loss = torch.nn.CrossEntropyLoss()(y, label)
            y = torch.argmax(y, dim=1)
            y = y.cpu().numpy()
            y_pred.append(y)
            label = label.cpu().numpy()
            label_list.append(label)
            del image, label, y
            torch.cuda.empty_cache()
            loss_list.append(loss.item())
        y_pred = np.concatenate(y_pred)
        label_list = np.concatenate(label_list)
        acc = np.mean(y_pred == label_list)
        log(f"Accuracy: {acc:.4f}")
        log(f"Loss: {np.mean(loss_list):.4f}")

if __name__ == "__main__":
    main()