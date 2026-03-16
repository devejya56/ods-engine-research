import torch
from torchvision import datasets, transforms
import os

# Project root directory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_DATA_ROOT = os.path.join(_PROJECT_ROOT, '..', 'datasets')

def get_dataloader(dataset_name='cifar10', batch_size=64, subset_size=None, data_dir=None):
    """
    Universal dataset loader for research experiments.
    Supported: 'cifar10', 'fashion_mnist', 'stl10', 'mnist', 'svhn'
    """
    if data_dir is None:
        data_dir = os.path.join(_DEFAULT_DATA_ROOT, dataset_name)
    
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
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
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    elif dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])
        # SVHN uses 'split' instead of 'train'
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)

    elif dataset_name == 'stl10':
        # STL-10 is larger (96x96). We normalize with standard ImageNet stats or STL-10 specific.
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
        ])
        # STL10 uses 'split' instead of 'train' boolean
        train_dataset = datasets.STL10(root=data_dir, split='train', download=True, transform=transform_train)
        test_dataset = datasets.STL10(root=data_dir, split='test', download=True, transform=transform_test)

    elif dataset_name == 'synthetic':
        from .synthetic import get_synthetic_dataloaders
        # We can pass extra kwargs through if needed, but for now we use defaults
        input_shape = (1, 28, 28) # Default to match Fashion-MNIST shape if not specified
        return get_synthetic_dataloaders(batch_size=batch_size, subset_size=subset_size, input_shape=input_shape)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if subset_size is not None and subset_size < len(train_dataset):
        indices = list(range(subset_size))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"[{dataset_name}] Using {subset_size} training samples.")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
