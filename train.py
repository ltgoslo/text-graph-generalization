from dataset import TextGraphDataset
from device import device
from model import TextRGCN
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse
import jsonlines
import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn
from utils import set_seed
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Compositional reasoning over KGs and text using LM+GNNs")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use during training.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="The weight decay to use during training.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="The pretrained model to use")
    parser.add_argument("--scheduler", type=str, default="constant", help="The lr scheduler protocol to use")
    parser.add_argument("--train_path", type=str, default=None, help="The question dataset file for train")
    parser.add_argument("--val_path", type=str, default=None, help="The question dataset file for val")
    parser.add_argument("--output_path", type=str, default=None, help="The output predictions path")
    parser.add_argument("--epochs", type=int, default=3, help="The number of epochs.")
    parser.add_argument("--gnn_layers", type=int, default=5, help="The number of layers in the GNN model.")
    parser.add_argument("--gnn_hidden", type=int, default=1024, help="The hidden size of the GNN model.")
    parser.add_argument("--static", action='store_true', help="Use static entity embeddings")
    parser.add_argument("--unidirectional", action='store_true', help="Use deep unidirectional mode")
    parser.add_argument("--bidirectional", action='store_true', help="Use deep unidirectional mode")
    parser.add_argument("--debug", action='store_true', help="Trigger debug mode")
    parser.add_argument("--GNN_only", action='store_true', help="Trigger GNN only ablation")
    parser.add_argument("--LM_only", action='store_true', help="Trigger LM only ablation")
    parser.add_argument("--lr", type=float, default=3e-5, help="The learning rate).")
    parser.add_argument("--gnn_lr", type=float, default=3e-5, help="The learning rate).")
    parser.add_argument("--llm_lr", type=float, default=1e-6, help="The learning rate).")
    parser.add_argument("--head_lr", type=float, default=1e-5, help="The learning rate).")
    #parser.add_argument("--seed", type=int, default=42, help="The rng seed")
    parser.add_argument('--seeds', nargs='+', help='Set of seeds to run with', required=True)
    parser.add_argument("--device_max_steps", default=200000, type=int, help="Total number of training steps to perform per device")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout value")
    args = parser.parse_args()
    return args


def train(model, loader):
    total_loss = 0.0
    correct = 0.0
    n = 0.0
    for uids, input_ids, attention_mask, graph_data, head_tail_indicies, special_node_index, special_token_index, label in tqdm(loader, leave=False):
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        out = model(input_ids, attention_mask, graph_data, head_tail_indicies, special_node_index, special_token_index)
        loss = criterion(out.squeeze(-1), label)
        total_loss += loss.item()
        pred = (nn.functional.sigmoid(out) > 0.5).float().squeeze()
        correct += (pred == label).sum()
        n += len(label)
        loss.backward()
        optimizer.step()
        scheduler.step()
    acc = correct / n
    loss = total_loss / len(loader)
    return acc, loss


def val(model, loader):
    val_loss = 0.0
    val_preds = []
    val_raw_preds = []
    val_uids = []
    correct = 0.0
    n = 0.0
    with torch.no_grad():
        for uids, input_ids, attention_mask, graph_data, head_tail_indicies, special_node_index, special_token_index, label in tqdm(val_loader, leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            out = model(input_ids, attention_mask, graph_data, head_tail_indicies, special_node_index, special_token_index)
            loss = criterion(out.squeeze(-1), label)
            val_loss += loss.item()
            pred = (nn.functional.sigmoid(out) > 0.5).float().squeeze()
            correct += (pred == label).sum()
            n += len(label)
            val_uids += uids.tolist()
            val_raw_preds += out.cpu().tolist()
            val_preds += pred.tolist()
    val_acc = correct / n
    val_loss = val_loss / len(val_loader)
    return val_acc, val_loss, val_uids, val_raw_preds, val_preds


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(0)
    with jsonlines.open(args.train_path, 'r') as f:
        train_questions = [o for o in f]
    with jsonlines.open(args.val_path, 'r') as f:
        val_questions = [o for o in f]

    train_dataset = TextGraphDataset(train_questions, args=args)
    val_dataset = TextGraphDataset(val_questions,
                                   args=args,
                                   head_emb=train_dataset.head_emb,
                                   tail_emb=train_dataset.tail_emb,
                                   tokenizer=train_dataset.tokenizer)

    logging.info(f'{train_dataset.decode(0)}')
    logging.info('Starting training...')
    for seed in args.seeds:
        logging.info(f'Setting seed {seed}...')
        set_seed(int(seed))
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                drop_last=False)
        config = {
            'lr': args.lr,
            'gnn_lr': args.gnn_lr,
            'lm_lr': args.llm_lr,
            'head_lr': args.head_lr,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
            'dropout': args.dropout,
            'gnn_layers': args.gnn_layers,
            'scheduler': args.scheduler,
            'LM_ONLY': args.LM_only,
            'GNN_ONLY': args.GNN_only,
            'unidirectional': args.unidirectional,
            'bidirectional': args.bidirectional,
            'disjoint': args.static,
        }
        wandb.init(project="results", config=config, entity="sondrewo", name=f"{args.output_path}_seed_{seed}")
        criterion = torch.nn.BCEWithLogitsLoss()
        model = TextRGCN(args).to(device)
        if args.bidirectional:
            model.text_encoder.resize_token_embeddings(len(train_dataset.tokenizer))

        optimizer = torch.optim.AdamW(
                [
                    {"params": model.gcn.parameters(), "lr": args.gnn_lr},
                    {"params": model.text_encoder.parameters(), "lr": args.llm_lr},
                    {"params": model.output.parameters(), "lr": args.head_lr},
                    {"params": model.output2.parameters(), "lr": args.head_lr},
                ],
                lr=args.lr,
                betas=(0.9, 0.98),
                eps=1e-6,
            )

        args.device_max_steps = args.epochs * len(train_loader)
        if args.scheduler == "cosine":
            def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
                def lr_lambda(current_step):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                    lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
                    return lr
                return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            scheduler = cosine_schedule_with_warmup(optimizer, int(args.device_max_steps * args.warmup_proportion), args.device_max_steps, 0.1)

        elif args.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                torch.optim.lr_scheduler.LinearLR(optimizer,
                                                start_factor=1e-9,
                                                end_factor=1.0,
                                                total_iters=int(args.device_max_steps * args.warmup_proportion)),
                torch.optim.lr_scheduler.LinearLR(optimizer,
                                                start_factor=1.0,
                                                end_factor=1e-9,
                                                total_iters=args.device_max_steps)
            ])

        elif args.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

        for epoch in range(args.epochs):
            model.train()
            train_acc, train_loss = train(model, train_loader)
            wandb.log({'training_accuracy': train_acc, 'training_loss': train_loss})
            logging.info(f'Train accuracy: {train_acc}, train.loss: {train_loss}')

            model.eval()
            val_acc, val_loss, val_uids, val_raw_preds, val_preds = val(model, val_loader)
            wandb.log({'val_accuracy': val_acc, 'val_loss': val_loss})
            logging.info(f'val accuracy: {val_acc}, val.loss: {val_loss}')

        logging.info('Writing final predictions...')
        final_preds = []
        for uid, raw, pred in zip(val_uids, val_raw_preds, val_preds):
            obj = {
                'uid': uid,
                'question': val_dataset.questions[uid],
                'relations': val_dataset.question_paths[uid],
                'head_tail': val_dataset.paths[uid],
                'label': val_dataset.labels[uid],
                'raw_pred': raw,
                'pred': pred
            }
            final_preds.append(obj)

        with jsonlines.open(f'./results/{args.output_path}_seed_{seed}.jsonl', 'w') as f:
            f.write_all(final_preds)
            print('Saved predictions...')
        wandb.finish()
