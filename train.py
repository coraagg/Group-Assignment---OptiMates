import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv

from utils import get_data_loader
from models import LinearModel, MLP, BasicCNN, OptimizedCNN   # 各成员将实现

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
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
    return total_loss / len(loader), 100. * correct / total

def evaluate(model, device, loader, criterion):
    model.eval()
    total_loss = 0
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
    return total_loss / len(loader), 100. * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['linear', 'mlp', 'basic_cnn', 'optimized_cnn'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--loss', type=str, default='crossentropy', choices=['crossentropy', 'nll'])
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_loader(args.batch_size, augment=args.augment)

    # 模型选择
    if args.model == 'linear':
        model = LinearModel()
    elif args.model == 'mlp':
        model = MLP()   # 可增加参数传递
    elif args.model == 'basic_cnn':
        model = BasicCNN()
    elif args.model == 'optimized_cnn':
        model = OptimizedCNN()
    model = model.to(device)

    # 损失函数
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:  # nll
        criterion = nn.NLLLoss()
        # 注意：NLLLoss 需要模型输出 log_softmax，各模型需在 forward 中返回 F.log_softmax(x, dim=1)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 日志
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{args.model}_bs{args.batch_size}_lr{args.lr}_loss{args.loss}.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(f'Epoch {epoch:2d}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

    # 最终测试
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    with open(os.path.join(args.log_dir, 'test_results.txt'), 'a') as f:
        f.write(f'{args.model}_bs{args.batch_size}_lr{args.lr}_loss{args.loss}: Test Acc={test_acc:.2f}%\n')

if __name__ == '__main__':
    main()