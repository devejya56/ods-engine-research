import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from data_loaders.real_world_loader import get_realworld_dataloader
from ods_engine.wrappers import ODSConnector
from training.train import train_model
from training.early_stopping_standard import ConventionalEarlyStopping

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 128), # Adjusted for pneumonia image size (likely 224x224 -> /8 = 28)
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def run_experiment(subset_size=None, epochs=20):
    print(f"\n{'='*70}")
    print(f"  Experiment 12: ODS vs Conventional Early Stopping (Pneumonia)")
    print(f"{'='*70}")

    train_loader, test_loader = get_realworld_dataloader(
        dataset_name='mmenendezg/pneumonia_x_ray', 
        batch_size=32, subset_size=subset_size
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Run with Conventional Early Stopping (monitoring test/val loss)
    print("\n>>> Run 1: Conventional Early Stopping (monitoring Test Loss)")
    model_es = SimpleCNN(num_classes=5).to(device)
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
        patience=2, warmup=5, threshold=3.0, adaptive=True, adaptive_k=1.2
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
    save_dir = os.path.join(_RESULTS_DIR, 'cnn_comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'exp12_data.json'), 'w') as f:
        json.dump({'es': results_es, 'ods': results_ods, 'stats': [stats_es, stats_ods]}, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        run_experiment(subset_size=500, epochs=5)
    else:
        run_experiment(subset_size=5000, epochs=20)
