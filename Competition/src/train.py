import argparse
import os
import json
import dataclasses
from tqdm import tqdm

import utils
import torch

from torch import nn, optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader

from Competition.src.model import LightGCN

def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_dir", type=str, default="../dataset/")

    # Architecture
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--lambda_", type=float, default=1e-5)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--check_step", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2**16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--results_path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)

    utils.init_logger(accelerator)
    cfg = utils.init_config_from_args(utils.TrainConfig, args)

    results_path= utils.handle_results_path(cfg.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    with open(args.results_path + '/config.json', 'w') as json_file:
        json.dump(dataclasses.asdict(cfg), json_file, indent=4)

    # Load data
    train_edge_index, val_edge_index, train_sparse_tensor, val_sparse_tensor, edge_index = utils.make_dataset(cfg.data_dir)

    train_loader = DataLoader(train_edge_index.T, batch_size=cfg.batch_size, shuffle=True, collate_fn=utils.collate_fn)

    model = LightGCN(num_users=53424, num_items=10000, embedding_dim=cfg.embedding_dim, K=cfg.K)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    criterion = utils.BPRLoss(cfg.lambda_)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    edge_index = edge_index.to(accelerator.device)
    train_sparse_tensor = train_sparse_tensor.to(accelerator.device)

    max_score = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for _, batch_pos_edges in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            user_emb_final, user_emb_0, item_emb_final, item_emb_0 = model(train_sparse_tensor)
            batch_pos_edges = batch_pos_edges.T
            batch_neg_edges = negative_sampling(edge_index, num_nodes=[53424, 10000], num_neg_samples=batch_pos_edges.shape[1])

            user_indices, pos_item_indices, neg_item_indices = batch_pos_edges[0], batch_pos_edges[1], batch_neg_edges[1]

            user_emb_final, user_emb_0 = user_emb_final[user_indices], user_emb_0[user_indices]

            pos_items_emb_final, pos_items_emb_0 = item_emb_final[pos_item_indices], item_emb_0[pos_item_indices]
            neg_items_emb_final, neg_items_emb_0 = item_emb_final[neg_item_indices], item_emb_0[neg_item_indices]

            loss = criterion(user_emb_final, user_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0)
            total_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()

        if (epoch + 1) % cfg.check_step == 0:
            model.eval()
            total_loss /= len(train_loader)
            recall, precision, ndcg = utils.get_metrics(model, val_edge_index, val_sparse_tensor, train_edge_index, train_sparse_tensor, cfg.K)
            score = 0.75 * recall + 0.25 * ndcg

            if score > max_score:
                max_score = score
                torch.save(model.state_dict(), os.path.join(cfg.results_path, 'LightGCN_best_score.pt'))

            utils.log(f'[{epoch+1:02d}/{cfg.epochs}] | loss: {total_loss:.6f} | recall@{cfg.K}: {recall:.6f} | '
                f'precision@{cfg.K}: {precision:.6f} | ndcg@{cfg.K}: {ndcg:.6f} | score: {score:.6f}')
            
if __name__ == "__main__":
    main()