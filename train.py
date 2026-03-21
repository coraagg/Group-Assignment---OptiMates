import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
from utils import get_data_loader
from models import LinearModel, MLP, BasicCNN, OptimizedCNN

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
    parser.add_argument('--model', type=str, required=True,
                        choices=['linear', 'mlp', 'basic_cnn', 'optimized_cnn'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--loss', type=str, default='crossentropy',
                        choices=['crossentropy', 'nll'])
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_data_loader(
        batch_size=args.batch_size, augment=args.augment
    )

    # 选择模型
    if args.model == 'linear':
        model = LinearModel()
    elif args.model == 'mlp':
        model = MLP(hidden_sizes=[512, 512], dropout=0.3)
    elif args.model == 'basic_cnn':
        model = BasicCNN()
    else:  # optimized_cnn
        model = OptimizedCNN()
    model = model.to(device)

    # 损失函数
    if args.loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:  # nll
        # NLLLoss 要求模型输出 log_softmax，此处简单处理：不修改模型，仅提示
        criterion = nn.NLLLoss()
        print("Warning: NLLLoss requires log_softmax output. Model not adapted.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{args.model}_bs{args.batch_size}_lr{args.lr}_loss{args.loss}.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, device, val_loader, criterion)
        print(f'Epoch {epoch:2d}/{args.epochs}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model.pth'))

    # 最终测试
    model.load_state_dict(torch.load(os.path.join(args.log_dir, 'best_model.pth')))
    test_loss, test_acc = evaluate(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    with open(os.path.join(args.log_dir, 'test_results.txt'), 'a') as f:
        f.write(f'{args.model}_bs{args.batch_size}_lr{args.lr}_loss{args.loss}: Test Acc={test_acc:.2f}%\n')

if __name__ == '__main__':
    main()
