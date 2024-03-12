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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=12345)

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
        checkpoint_path=handle_results_path(cfg.results_path) / "model.pt",
    ).evaluate()

class Trainer:
    def __init__(self, model, train_loader, accelerator, make_opt, config, results_path):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.train_loader = train_loader
        self.accelerator = accelerator
        self.opt = accelerator.prepare(make_opt(self.model.parameters()))
        self.critierion = torch.nn.MSELoss()
        self.cfg = config
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
        for epoch in tqdm(range(self.cfg.epochs)):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                y_pred = self.model(x)
                loss = self.critierion(y_pred, y)
                self.accelerator.backward(loss)
                self.opt.step()
                loss_list.append(loss.item())
            print(f'Epoch {epoch+1}/{self.cfg.epochs}, Loss: {loss.item()}')
        
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
    
class Evaluator:
    def __init__(self, model, val_loader, accelerator, config, checkpoint_path):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.cfg = config
        self.device = self.accelerator.device
        self.model.to(self.device)
        self.checkpoint_path = checkpoint_path

    def evaluate(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        loss_list = []
        y_pred_list = []
        y_true_list = []
        x_list = []
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                y_pred = self.model(x)
            loss = torch.nn.MSELoss()(y_pred, y)
            loss_list.append(loss.item())
            y_pred = y_pred.cpu().numpy()
            y_true = y.cpu().numpy()
            x = x.cpu().numpy()
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
            x_list.append(x)
            del x, y, y_pred, y_true
            torch.cuda.empty_cache()
        
        print(f'Loss: {np.mean(loss_list)}')
        y_pred_list = np.concatenate(y_pred_list)
        y_true_list = np.concatenate(y_true_list)
        x_list = np.concatenate(x_list)
        # 画出预测结果
        plt.scatter(x_list, y_true_list, label='True')
        plt.scatter(x_list, y_pred_list, label='Pred')
        plt.legend()
        plt.savefig('eval.png')