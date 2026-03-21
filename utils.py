import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_data_loader(batch_size=64, augment=True):
    """
    返回 CIFAR-100 的训练、验证、测试 DataLoader
    """
    # CIFAR-100 的均值和标准差
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # 基础转换：ToTensor + Normalize
    base_transform = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    # 训练时是否使用数据增强
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            *base_transform
        ])
    else:
        train_transform = transforms.Compose(base_transform)

    # 验证和测试集不需要增强
    val_test_transform = transforms.Compose(base_transform)

    # 下载训练集（train=True）
    full_train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    # 注意：random_split 会继承数据集的 transform，但此处 train_dataset 的 transform 已经是 train_transform
    # 为了让验证集使用 val_test_transform，需要手动设置
    val_dataset.dataset.transform = val_test_transform

    # 测试集
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
