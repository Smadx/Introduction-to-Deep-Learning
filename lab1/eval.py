import argparse

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path

from tqdm import tqdm

from utils import (
    TrainConfig,
    handle_results_path,
    get_date_str,
    make_dataloader,
    log,
    print_model_summary,
)

from train import Trainer

from model import MLP

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results-path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    accelerator = Accelerator(split_batches=True)

    model = MLP(cfg)
    print_model_summary(model, batch_size=cfg.batch_size,shape=(1, 1), batch_size_torchinfo=cfg.batch_size)
    train_loader, val_loader, test_loader = make_dataloader(cfg.N, cfg.batch_size)
    log(f"Train on {accelerator.device}")
    checkpoint_path =  Trainer(
        model,
        train_loader,
        accelerator,
        make_opt=lambda params: torch.optim.Adam(params, lr=cfg.lr),
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
        var_loader=val_loader,
    ).train()
    Evaluator(
        model=model,
        test_loader=test_loader,
        accelerator=accelerator,
        config=cfg,
        results_path=checkpoint_path.parent,
    ).evaluate()


class Evaluator:
    """
    Evaluate the model on the test set.

    Args:
        - model: the model to evaluate
        - test_loader: the test dataloader
        - accelerator: the accelerator
        - config: the config
        - results_path: the path to save the results
    """
    def __init__(self, model, test_loader, accelerator, config, results_path):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.test_loader = test_loader
        self.accelerator = accelerator
        self.cfg = config
        self.device = self.accelerator.device
        self.model.to(self.device)
        self.results_path = results_path
        self.checkpoint_path = results_path / "model.pt"
        self.eval_path = results_path / f"eval_{get_date_str()}"
        self.eval_path.mkdir()

    def evaluate(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        loss_list = []
        y_pred_list = []
        y_true_list = []
        x_list = []
        for x, y in tqdm(self.test_loader):
            x, y = x.float(), y.float()
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
        
        print(f'Loss_mean: {np.mean(loss_list)}')
        y_pred_list = np.concatenate(y_pred_list)
        y_true_list = np.concatenate(y_true_list)
        x_list = np.concatenate(x_list)

        plt.figure()
        plt.scatter(x_list, y_true_list, label='True')
        plt.scatter(x_list, y_pred_list, label='Pred')
        plt.legend()
        plt.savefig(self.eval_path / 'eval.png')

if __name__ == "__main__":
    main()