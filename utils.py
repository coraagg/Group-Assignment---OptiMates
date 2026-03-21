import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loader(batch_size=64, augment=True):
    """
    返回 CIFAR-100 的训练、验证、测试 DataLoader

    成员1任务：
    1. 下载 CIFAR-100
    2. 划分训练集 / 验证集 / 测试集
    3. 实现标准化
    4. 实现数据增强（仅训练集使用）
    """

    # CIFAR-100 的均值和标准差（题目要求）
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # 基础变换：转 Tensor + 标准化
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 训练集增强
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = base_transform

    # 验证集 / 测试集不做增强
    val_test_transform = base_transform

    # -----------------------------
    # 关键点：
    # 为了避免 random_split 后 train/val 共用同一个 transform，
    # 这里分别建立两个训练集副本：
    # 一个用于 train（带增强）
    # 一个用于 val（不带增强）
    # -----------------------------
    full_train_dataset_for_train = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    full_train_dataset_for_val = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=val_test_transform
    )

    # 划分训练集和验证集：80% / 20%
    total_size = len(full_train_dataset_for_train)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # 为了保证 train / val 划分一致，使用同一个随机种子
    generator = torch.Generator().manual_seed(42)

    train_dataset, _ = random_split(
        full_train_dataset_for_train,
        [train_size, val_size],
        generator=generator
    )

    # 重新设置同样的随机种子，确保 val 使用的是同一批划分索引
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        full_train_dataset_for_val,
        [train_size, val_size],
        generator=generator
    )

    # 测试集
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_test_transform
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader, test_loader