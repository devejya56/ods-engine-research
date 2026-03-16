import torch
from torchvision import datasets, transforms
import os

# Project root directory (two levels up from this file)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, '..', 'datasets', 'cifar10')

def get_cifar10_dataloaders(batch_size=64, subset_size=None, data_dir=None):
    """
    Loads CIFAR-10. Optionally subsets the training data.
    Uses absolute path to locate data regardless of CWD.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data already exists locally
    needs_download = not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py'))
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=needs_download, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=needs_download, transform=transform_test)
    
    if subset_size is not None and subset_size < len(train_dataset):
        # We just take the first N samples for simplicity
        indices = list(range(subset_size))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using {subset_size} training samples.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
