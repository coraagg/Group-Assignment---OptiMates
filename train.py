import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_data_loader
from models import LinearModel, MLP, BasicCNN, OptimizedCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model, device, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, pred = output.max(1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def build_experiment_name(args):
    name = f"{args.model}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}"

    if args.model == 'mlp':
        name += (
            f"_hs{args.hidden_size}"
            f"_layers{args.num_layers}"
            f"_act{args.activation}"
            f"_drop{args.dropout}"
            f"_wd{args.weight_decay}"
            f"_bn{int(args.use_batchnorm)}"
        )

    if args.augment:
        name += "_aug"

    return name


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['linear', 'mlp', 'basic_cnn', 'optimized_cnn']
    )

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=42)

    # MLP-specific arguments
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'tanh'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--use_batchnorm', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loader(
        batch_size=args.batch_size,
        augment=args.augment
    )

    # Build model
    if args.model == 'linear':
        model = LinearModel()

    elif args.model == 'mlp':
        hidden_sizes = [args.hidden_size] * args.num_layers
        model = MLP(
            hidden_sizes=hidden_sizes,
            activation=args.activation,
            dropout=args.dropout,
            use_batchnorm=args.use_batchnorm
        )

    elif args.model == 'basic_cnn':
        model = BasicCNN()

    else:
        model = OptimizedCNN()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # AdamW 更适合 weight_decay 实验
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器：训练更稳一点
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    os.makedirs(args.log_dir, exist_ok=True)

    exp_name = build_experiment_name(args)
    log_file = os.path.join(args.log_dir, f"{exp_name}.csv")
    best_model_path = os.path.join(args.log_dir, f"{exp_name}_best.pth")
    result_file = os.path.join(args.log_dir, "test_results.txt")

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch',
            'train_loss',
            'train_acc',
            'val_loss',
            'val_acc',
            'lr'
        ])

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    # Load best model for final test
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(
            f"{exp_name} | "
            f"Best Val Acc={best_val_acc:.2f}% | "
            f"Test Acc={test_acc:.2f}%\n"
        )


if __name__ == '__main__':
    main()