import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loader(batch_size=64, augment=True):
    """
    Return train, validation, and test DataLoaders for CIFAR-100.

    Member 1 tasks:
    1. Download CIFAR-100
    2. Split training set into train / validation sets
    3. Implement normalization
    4. Implement data augmentation (only for training set)
    """

    # CIFAR-100 mean and std (as required by the assignment)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Base transform: ToTensor + Normalize
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Training set augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = base_transform

    # Validation / test sets use only base transform (no augmentation)
    val_test_transform = base_transform

    # ------------------------------------------------------------------
    # To avoid sharing the same transform between train and validation sets,
    # we create two separate copies of the training dataset:
    # - one with augmentation for training
    # - one without augmentation for validation
    # ------------------------------------------------------------------
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

    # Split: 80% training, 20% validation
    total_size = len(full_train_dataset_for_train)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # Use the same random seed to ensure consistent split
    generator = torch.Generator().manual_seed(42)

    train_dataset, _ = random_split(
        full_train_dataset_for_train,
        [train_size, val_size],
        generator=generator
    )

    # Re‑seed to get the same split for the validation dataset
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        full_train_dataset_for_val,
        [train_size, val_size],
        generator=generator
    )

    # Test set
    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_test_transform
    )

    # DataLoaders
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
