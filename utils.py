import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_data_loader(batch_size=64, augment=False):
    transform_list = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if augment:
        transform_list.insert(0, transforms.RandomRotation(10))
        transform_list.insert(0, transforms.RandomAffine(0, translate=(0.1, 0.1)))
    transform = transforms.Compose(transform_list)

    # 下载训练集
    full_train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    # 划分训练集和验证集 (80% / 20%)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader