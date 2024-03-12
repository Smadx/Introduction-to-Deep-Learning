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

from tqdm import tqdm

from utils import (
    TrainConfig,
    get_date_str,
    make_dataloader,
    init_logger,
    init_config_from_args,
    log,
    print_model_summary,
    handle_results_path,
)

from model import MLP

def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--N", type=int, default=200)

    # Architecture
    parser.add_argument("--input-size", type=int, default=1)
    parser.add_argument("--hidden-size-1", type=int, default=64)
    parser.add_argument("--hidden-size-2", type=int, default=32)
    parser.add_argument("--act-fn", type=str, default="relu")

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--clip-grad-norm", type=bool, default=True)

    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1_000)

    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)

    init_logger(accelerator)
    cfg = init_config_from_args(TrainConfig, args)

    model = MLP(cfg)
    print_model_summary(model, batch_size=cfg.batch_size,shape=(1, 1), batch_size_torchinfo=cfg.batch_size)
    with accelerator.local_main_process_first():
        train_loader, val_loader, test_loader = make_dataloader(cfg.N)


class Trainer:
    def __init__(self, model, train_loader, accelerator, make_opt, config, results_path, is_before: bool):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.train_loader = train_loader
        self.accelerator = accelerator
        self.opt = accelerator.prepare(make_opt(self.model.parameters()))
        self.critierion = torch.nn.MSELoss()
        self.cfg = config
        self.results_path = results_path
        self.is_before = is_before
        self.device = self.accelerator.device
        print('Train on', self.device)
        self.model.to(self.device)
        self.checkpoint_path = self.results_path / f"model_{'before' if is_before else 'after'}.pt"

        self.step = 0

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.cfg.epochs,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            iter_data = iter(self.train_loader)
            while self.step < self.cfg.epochs:
                try:
                    inputs, targets = next(iter_data)
                except StopIteration:
                    iter_data = iter(self.train_loader)
                    inputs, targets = next(iter_data)
                inputs = inputs.unsqueeze(1)
                targets = targets.unsqueeze(1)
                """if self.is_before:
                    # 划分出前6000列
                    inputs = inputs[:, :, 6000:]
                    targets = targets[:, :, 6000:]
                else:
                    # 划分出6000列之后的数据
                    inputs = inputs[:, :, 6000:]
                    targets = targets[:, :, 6000:]"""
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.opt.zero_grad()
                outputs = self.model(inputs)
                loss = self.critierion(outputs, targets)
                self.accelerator.backward(loss)
                if self.cfg.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                loss_list.append(loss.item())
                min_position = loss_list.index(min(loss_list)) + 1
                pbar.set_description(f"Loss: {loss.item():.18f}; min_position: {min_position}")
                self.step += 1
                self.accelerator.wait_for_everyone()
                pbar.update()
        
        with open('loss_list.txt', 'w') as f:
            print(loss_list, file=f)
        plt.plot(loss_list)
        plt.savefig('loss_list.png')
        

        self.save()

    def save(self):
        """
        把模型保存到指定路径
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
    
