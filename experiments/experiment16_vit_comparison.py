import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from data_loaders.real_world_loader import get_realworld_dataloader
from ods_engine.wrappers import ODSConnector
from training.train import train_model
from training.early_stopping_standard import ConventionalEarlyStopping

def get_vit_model(num_classes=5):
    # Load pretrained ViT-B-16
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    
    # Freeze initial layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the encoder last layer and heads
    for param in model.encoder.layers[-1].parameters():
        param.requires_grad = True
    for param in model.heads.parameters():
        param.requires_grad = True
        
    num_ftrs = model.heads.head.in_features
    # Adjust for 5 classes (mmenendezg/pneumonia_x_ray)
    model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    return model

def run_experiment(subset_size=None, epochs=10):
    print(f"\n{'='*70}")
    print(f"  Experiment 16: ODS vs Conventional Early Stopping (Vision Transformer)")
    print(f"{'='*70}")

    # ViT requires 224x224 input
    train_loader, test_loader = get_realworld_dataloader(
        dataset_name='mmenendezg/pneumonia_x_ray', 
        subset_size=subset_size,
        batch_size=16 # ViT is memory intensive
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Run with Conventional Early Stopping
    print("\n>>> Run 1: Conventional Early Stopping (monitoring Test Loss)")
    model_es = get_vit_model().to(device)
    optimizer_es = optim.Adam(model_es.parameters(), lr=1e-5) # Lower LR for ViT
    es_tracker = ConventionalEarlyStopping(patience=3)
    
    results_es = train_model(
        model=model_es, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_es, 
        val_early_stopping=es_tracker 
    )

    # 2. Run with ODS (Zero Warmup)
    print("\n>>> Run 2: ODS Early Stopping (Zero Warmup)")
    model_ods0 = get_vit_model().to(device)
    optimizer_ods0 = optim.Adam(model_ods0.parameters(), lr=1e-5)
    ods0_connector = ODSConnector(
        model=model_ods0,
        patience=2, warmup=0, threshold=3.5, adaptive=True, adaptive_k=1.2
    )
    
    results_ods0 = train_model(
        model=model_ods0, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_ods0, ods_connector=ods0_connector
    )

    # 3. Run with ODS (3-Epoch Warmup)
    print("\n>>> Run 3: ODS Early Stopping (3-Epoch Warmup)")
    model_ods3 = get_vit_model().to(device)
    optimizer_ods3 = optim.Adam(model_ods3.parameters(), lr=1e-5)
    ods3_connector = ODSConnector(
        model=model_ods3,
        patience=2, warmup=3, threshold=3.5, adaptive=True, adaptive_k=1.2
    )
    
    results_ods3 = train_model(
        model=model_ods3, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_ods3, ods_connector=ods3_connector
    )

    # Output Table
    print("\n" + "="*80)
    print(f"{'Method':<30} | {'Stopped':<8} | {'Final Acc %':<12} | {'Peak Acc %':<12}")
    print("-" * 80)
    
    def print_row(res, name):
        final_acc = res['test_accuracy'][-1]
        peak_acc = max(res['test_accuracy'])
        stopped = res.get('stopped_epoch', len(res['test_accuracy']))
        print(f"{name:<30} | {stopped:<8} | {final_acc:<12.2f} | {peak_acc:<12.2f}")
        return {'method': name, 'stopped': stopped, 'final_acc': final_acc, 'peak_acc': peak_acc}

    stats_es = print_row(results_es, "Conventional (Val Loss)")
    stats_ods0 = print_row(results_ods0, "ODS (Zero Warmup)")
    stats_ods3 = print_row(results_ods3, "ODS (3-Epoch Warmup)")
    print("="*80)

    # Save results
    save_dir = os.path.join(_RESULTS_DIR, 'vit_comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'exp16_data.json'), 'w') as f:
        json.dump({
            'es': results_es, 
            'ods0': results_ods0, 
            'ods3': results_ods3, 
            'stats': [stats_es, stats_ods0, stats_ods3]
        }, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    if args.quick:
        # Mini subset for verification
        run_experiment(subset_size=100, epochs=5)
    else:
        # Real validation
        run_experiment(subset_size=300, epochs=10)
