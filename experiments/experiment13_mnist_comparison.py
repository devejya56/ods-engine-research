import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from ods_engine.wrappers import ODSConnector
from training.train import train_model
from training.early_stopping_standard import ConventionalEarlyStopping

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_mnist_dataloaders(batch_size=64, subset_size=None):
    print("Loading MNIST dataset...")
    from torchvision import datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if subset_size:
        indices = list(range(subset_size))
        train_set = torch.utils.data.Subset(train_set, indices)
        test_set = torch.utils.data.Subset(test_set, list(range(min(len(test_set), subset_size // 5))))
        
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def run_experiment(subset_size=None, epochs=20):
    print(f"\n{'='*70}")
    print(f"  Experiment 13: ODS vs Conventional Early Stopping (MNIST)")
    print(f"{'='*70}")

    train_loader, test_loader = get_mnist_dataloaders(subset_size=subset_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Run with Conventional Early Stopping (monitoring Test Loss)
    print("\n>>> Run 1: Conventional Early Stopping (monitoring Test Loss)")
    model_es = SimpleCNN().to(device)
    optimizer_es = optim.Adam(model_es.parameters(), lr=1e-3)
    es_tracker = ConventionalEarlyStopping(patience=3)
    
    results_es = train_model(
        model=model_es, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_es, 
        val_early_stopping=es_tracker 
    )

    # 2. Run with ODS Early Stopping
    print("\n>>> Run 2: ODS Early Stopping")
    model_ods = SimpleCNN().to(device)
    optimizer_ods = optim.Adam(model_ods.parameters(), lr=1e-3)
    ods_connector = ODSConnector(
        model=model_ods,
        patience=2, warmup=5, threshold=3.5, adaptive=True, adaptive_k=1.2
    )
    
    results_ods = train_model(
        model=model_ods, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_ods, ods_connector=ods_connector
    )

    # 3. Output Table
    print("\n" + "="*60)
    print(f"{'Method':<25} | {'Stopped':<8} | {'Final Acc %':<12} | {'Peak Acc %':<12}")
    print("-" * 60)
    
    def print_row(res, name):
        final_acc = res['test_accuracy'][-1]
        peak_acc = max(res['test_accuracy'])
        stopped = res.get('stopped_epoch', len(res['test_accuracy']))
        print(f"{name:<25} | {stopped:<8} | {final_acc:<12.2f} | {peak_acc:<12.2f}")
        return {'method': name, 'stopped': stopped, 'final_acc': final_acc, 'peak_acc': peak_acc}

    stats_es = print_row(results_es, "Conventional (Val Loss)")
    stats_ods = print_row(results_ods, "ODS (Adaptive Signals)")
    print("="*60)

    # Save and Plot
    save_dir = os.path.join(_RESULTS_DIR, 'mnist_comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'exp13_data.json'), 'w') as f:
        json.dump({'es': results_es, 'ods': results_ods, 'stats': [stats_es, stats_ods]}, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        run_experiment(subset_size=1000, epochs=10)
    else:
        run_experiment(subset_size=10000, epochs=20)
