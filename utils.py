import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loader(batch_size=64, augment=True, resize_to_224=False):
    """
    Return train, validation, and test DataLoaders for CIFAR-100.

    Args:
        batch_size: batch size
        augment: whether to apply data augmentation on training set
        resize_to_224: whether to resize images to 224x224 (required for pretrained ResNet)
    """
    # CIFAR-100 mean and std
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Base transform: resize (if needed) -> ToTensor -> Normalize
    base_transforms = []
    if resize_to_224:
        base_transforms.append(transforms.Resize(224))
    base_transforms.append(transforms.ToTensor())
    base_transforms.append(transforms.Normalize(mean, std))
    base_transform = transforms.Compose(base_transforms)

    # Training transform: may include augmentation
    if augment:
        train_transforms = []
        if resize_to_224:
            # For 224 input, use random crop of size 224 with padding
            train_transforms.append(transforms.RandomCrop(224, padding=4))
        else:
            # Original 32x32 random crop
            train_transforms.append(transforms.RandomCrop(32, padding=4))
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(transforms.Normalize(mean, std))
        train_transform = transforms.Compose(train_transforms)
    else:
        train_transform = base_transform

    # Validation and test sets use only base transform (no augmentation)
    val_test_transform = base_transform

    # Create two copies of the training dataset to keep different transforms
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

    # Split: 80% train, 20% validation
    total_size = len(full_train_dataset_for_train)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, _ = random_split(
        full_train_dataset_for_train,
        [train_size, val_size],
        generator=generator
    )

    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(
        full_train_dataset_for_val,
        [train_size, val_size],
        generator=generator
    )

    test_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=val_test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
