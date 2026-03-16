import os
import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']
        
        # Ensure image is RGB (some X-rays are grayscale with 1 channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_realworld_dataloader(dataset_name='mmenendezg/pneumonia_x_ray', batch_size=32, subset_size=1000):
    """
    Loads a real-world image dataset (like Chest X-Rays) for explainability.
    """
    print(f"Loading real-world dataset: {dataset_name}...")
    
    try:
        # This dataset usually has 'train' and 'test'
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}.")
        # Fallback if needed
        dataset = load_dataset('mnist') # Extreme fallback

    # Standard ResNet image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = dataset['train']
    # Check if 'test' or 'validation' exists
    test_key = 'test' if 'test' in dataset else 'validation'
    test_data = dataset[test_key]

    if subset_size is not None and subset_size < len(train_data):
        train_data = train_data.select(range(subset_size))
        # Keep test set manageable
        test_size = min(len(test_data), int(subset_size * 0.5))
        test_data = test_data.select(range(test_size))
        print(f"[{dataset_name}] Using {subset_size} training samples, {test_size} test samples.")

    # Note: Using 'label' instead of 'labels' as mmenendezg uses 'label'
    class HFImageDatasetCustom(HFImageDataset):
         def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image']
            label = item.get('label', item.get('labels', 0))
            if image.mode != 'RGB': image = image.convert('RGB')
            if self.transform: image = self.transform(image)
            return image, label

    pytorch_train = HFImageDatasetCustom(train_data, transform=transform)
    pytorch_test = HFImageDatasetCustom(test_data, transform=transform)

    train_loader = DataLoader(pytorch_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(pytorch_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
