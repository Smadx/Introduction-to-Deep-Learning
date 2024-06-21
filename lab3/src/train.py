import argparse

import torch
import yaml
import os
import dataclasses
import matplotlib.pyplot as plt

from torch.nn.functional import binary_cross_entropy
from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path

from tqdm import tqdm

from utils import (
    TrainConfig,
    make_cora,
    make_citeseer,
    split_val,
    create_edge_split,
    drop_edge,
    init_logger,
    init_config_from_args,
    log,
    compute_auc,
    print_model_summary,
    handle_results_path,
)

from model import GCN

def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--dataset", type=str, default="cora")

    # Architecture
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--pair-norm-scale", type=float, default=None)

    # Training
    parser.add_argument("--dropedge-prob", type=float, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)

    init_logger(accelerator)
    cfg = init_config_from_args(TrainConfig, args)

    with accelerator.local_main_process_first():
        if cfg.dataset == "cora":
            data = make_cora("datasets/cora")
        elif cfg.dataset == "citeseer":
            data = make_citeseer("datasets/citeseer")
        else:
            raise ValueError(f"Unknown dataset {cfg.dataset}")
        data = split_val(data)
    classifer = GCN(data.num_features, data.num_classes, cfg, classifier=True)
    print_model_summary(classifer, 
                        node_shape=(data.x.shape[0], data.x.shape[1]), 
                        edge_shape=(data.edge_index.shape[0], data.edge_index.shape[1]),
                        depth=3, dataset_name=cfg.dataset)
    link_predictor = GCN(data.num_features, data.num_classes, cfg, classifier=False)
    print_model_summary(link_predictor,
                        node_shape=(data.x.shape[0], data.x.shape[1]),
                        edge_shape=(data.edge_index.shape[0], data.edge_index.shape[1]),
                        depth=3, dataset_name=cfg.dataset)
    log(f"Train on {accelerator.device}")
    Trainer(
        classifer,
        data,
        accelerator,
        make_opt=lambda params: torch.optim.Adam(params, lr=cfg.lr),
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
        classifier=True,
    ).train()
    Trainer(
        link_predictor,
        data,
        accelerator,
        make_opt=lambda params: torch.optim.Adam(params, lr=cfg.lr),
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
        classifier=False,
    ).train()
    Evaluator(
        classifer,
        data,
        accelerator,
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
        classifier=True,
    ).evaluate()
    Evaluator(
        link_predictor,
        data,
        accelerator,
        config=cfg,
        results_path=handle_results_path(cfg.results_path),
        classifier=False,
    ).evaluate()

class Trainer:
    """
    Trainer of the model

    Args:
        - model: the model to train
        - data: the dataset of graph
        - accelerator: the accelerator
        - make_opt: a function that takes the model parameters and returns an optimizer
        - config: the configuration
        - results_path: the path to save the results
        - classifier: whether the model is a classifier or a link predictor
    """
    def __init__(self, model, data, accelerator, make_opt, config, results_path, classifier=True):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.data= data
        self.accelerator = accelerator
        self.opt = accelerator.prepare(make_opt(self.model.parameters()))
        self.critierion = torch.nn.NLLLoss()
        self.cfg = config
        self.step = 0
        self.epoch = self.cfg.epochs
        self.results_path = results_path
        self.classifier = classifier
        if self.classifier == False:
            self.train_edges, self.val_edges, self.neg_val_edges = create_edge_split(data)
        self.device = self.accelerator.device
        print('Train on', self.device)
        self.model.to(self.device)
        self.checkpoint_path = self.results_path / f"classifier.pt" if self.classifier else self.results_path / f"link_predictor.pt"

        self.results_path.mkdir(parents=True, exist_ok=True)
        with open(self.results_path / "config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f)

    def train(self):
        loss_list = []
        with tqdm(
            initial=self.step,
            total=self.epoch,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.epoch:
                self.opt.zero_grad()
                edge_index = self.data.edge_index
                if self.cfg.dropedge_prob is not None:
                    edge_index, _ = drop_edge(self.data.edge_index, drop_prob=self.cfg.dropedge_prob)
                output = self.model(self.data.x, edge_index)
                if self.classifier:
                    loss = self.critierion(output[self.data.train_mask], self.data.y[self.data.train_mask])
                else:
                    pos_out = torch.sigmoid((output[self.train_edges[:, 0]] * output[self.train_edges[:, 1]]).sum(dim=1))
                    neg_out = torch.sigmoid((output[self.neg_val_edges[:, 0]] * output[self.neg_val_edges[:, 1]]).sum(dim=1))
                    loss = binary_cross_entropy(torch.cat([pos_out, neg_out]), torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]))
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
        - data: the dataset of graph
        - accelerator: the accelerator
        - config: the configuration
        - results_path: the path to save the results
        - classifier: whether the model is a classifier or a link predictor
    """
    def __init__(self, model, data, accelerator, config, results_path, classifier=True):
        super().__init__()
        self.model = accelerator.prepare(model)
        self.data = data
        self.accelerator = accelerator
        self.cfg = config
        self.device = self.accelerator.device
        self.model.to(self.device)
        self.results_path = results_path
        self.classifier = classifier
        if self.classifier == False:
            self.train_edges, self.val_edges, self.neg_val_edges = create_edge_split(data)
        self.checkpoint_path = results_path / f"classifier.pt" if self.classifier else results_path / f"link_predictor.pt"

    def evaluate(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        with torch.no_grad():
            if self.classifier:
                preds = self.model(self.data.x, self.data.edge_index).argmax(dim=1)
                val_correct = (preds[self.data.val_mask] == self.data.y[self.data.val_mask]).sum()
                class_accuracy = int(val_correct) / int(self.data.val_mask.sum())
                log(f"Validation accuracy: {class_accuracy:.4f}")
            else:
                output = self.model(self.data.x, self.data.edge_index)
                pos_out = torch.sigmoid((output[self.val_edges[:, 0]] * output[self.val_edges[:, 1]]).sum(dim=1))
                neg_out = torch.sigmoid((output[self.neg_val_edges[:, 0]] * output[self.neg_val_edges[:, 1]]).sum(dim=1))
                auc = compute_auc(pos_out, neg_out)
                log(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()