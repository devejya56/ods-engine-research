import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticOverfittingDataset(Dataset):
    """
    Synthetic dataset for testing ODS.
    - subset_size: number of samples.
    - input_shape: (C, H, W).
    - noisiness: proportion of labels to flip (to force overfitting/memorization).
    """
    def __init__(self, subset_size=1000, input_shape=(3, 32, 32), num_classes=10, noise_ratio=0.5):
        self.data = torch.randn(subset_size, *input_shape)
        # Create "learnable" patterns: put a small signal in the mean related to the class
        self.targets = torch.randint(0, num_classes, (subset_size,))
        
        for i in range(subset_size):
            label = self.targets[i]
            # Add a weak class-dependent signal
            self.data[i] += 0.5 * label / num_classes
        
        # Overfitting phase: flip some labels after initial training
        if noise_ratio > 0:
            n_noise = int(subset_size * noise_ratio)
            indices = torch.randperm(subset_size)[:n_noise]
            self.targets[indices] = torch.randint(0, num_classes, (n_noise,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_synthetic_dataloaders(batch_size=64, subset_size=1000, input_shape=(3, 32, 32), noise_ratio=0.5):
    train_dataset = SyntheticOverfittingDataset(subset_size, input_shape, noise_ratio=noise_ratio)
    test_dataset = SyntheticOverfittingDataset(200, input_shape, noise_ratio=0) # Clean test set
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
