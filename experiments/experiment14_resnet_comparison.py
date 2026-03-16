import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from data_loaders.real_world_loader import get_realworld_dataloader
from ods_engine.wrappers import ODSConnector
from training.train import train_model
from training.early_stopping_standard import ConventionalEarlyStopping

def get_resnet_model(num_classes=5):
    # Load pretrained ResNet-18
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze initial layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last block and fc layer
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    # Adjust for 5 classes (mmenendezg/pneumonia_x_ray)
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def run_experiment(subset_size=None, epochs=15):
    print(f"\n{'='*70}")
    print(f"  Experiment 14: ODS vs Conventional Early Stopping (ResNet-18)")
    print(f"{'='*70}")

    train_loader, test_loader = get_realworld_dataloader(
        dataset_name='mmenendezg/pneumonia_x_ray', 
        subset_size=subset_size,
        batch_size=32
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Run with Conventional Early Stopping
    print("\n>>> Run 1: Conventional Early Stopping (monitoring Test Loss)")
    model_es = get_resnet_model().to(device)
    optimizer_es = optim.Adam(model_es.parameters(), lr=1e-4) # Lower LR for fine-tuning
    es_tracker = ConventionalEarlyStopping(patience=3)
    
    results_es = train_model(
        model=model_es, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_es, 
        val_early_stopping=es_tracker 
    )

    # 2. Run with ODS Early Stopping
    print("\n>>> Run 2: ODS Early Stopping")
    model_ods = get_resnet_model().to(device)
    optimizer_ods = optim.Adam(model_ods.parameters(), lr=1e-4)
    ods_connector = ODSConnector(
        model=model_ods,
        patience=2, warmup=0, threshold=3.5, adaptive=True, adaptive_k=1.2
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

    # Save results
    save_dir = os.path.join(_RESULTS_DIR, 'resnet_comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'exp14_data.json'), 'w') as f:
        json.dump({'es': results_es, 'ods': results_ods, 'stats': [stats_es, stats_ods]}, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        run_experiment(subset_size=500, epochs=10)
    else:
        run_experiment(subset_size=1000, epochs=20)
