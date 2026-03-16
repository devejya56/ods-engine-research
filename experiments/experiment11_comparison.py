import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from data_loaders.real_world_loader import get_realworld_dataloader
from ods_engine.wrappers import ODSConnector
from training.train import train_model

def get_pretrained_resnet_for_xray(num_classes=5):
    """Loads a pretrained ResNet-18 and replaces the head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer 4 and fc
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def run_comparison(subset_size=500, epochs=15):
    print(f"\n{'='*70}")
    print(f"  Experiment 11: ODS vs No-ODS (Comparative Analysis)")
    print(f"{'='*70}")

    train_loader, test_loader = get_realworld_dataloader(
        dataset_name='mmenendezg/pneumonia_x_ray', 
        batch_size=32, subset_size=subset_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ─── 1. Run WITH ODS ────────────────────────────────────────────────
    print("\n>>> Running with ODS Early Stopping...")
    model_ods = get_pretrained_resnet_for_xray(num_classes=5).to(device)
    optimizer_ods = optim.Adam(model_ods.layer4.parameters(), lr=1e-4)
    optimizer_ods.add_param_group({'params': model_ods.fc.parameters(), 'lr': 1e-3})
    
    ods_connector = ODSConnector(
        model=model_ods,
        alpha=1.0, beta=10.0, gamma=1.0,
        threshold=3.0, patience=2, warmup=2,
        adaptive=True, adaptive_k=1.5,
        dataset_size=subset_size
    )
    
    results_ods = train_model(
        model=model_ods, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_ods, ods_connector=ods_connector
    )

    # ─── 2. Run WITHOUT ODS ─────────────────────────────────────────────
    print("\n>>> Running WITHOUT ODS (Fixed Epochs)...")
    model_plain = get_pretrained_resnet_for_xray(num_classes=5).to(device)
    optimizer_plain = optim.Adam(model_plain.layer4.parameters(), lr=1e-4)
    optimizer_plain.add_param_group({'params': model_plain.fc.parameters(), 'lr': 1e-3})
    
    results_plain = train_model(
        model=model_plain, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer_plain, ods_connector=None # No ODS
    )

    # ─── 3. Comparison Table ────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"{'Method':<15} | {'Stopped':<8} | {'Final Acc %':<12} | {'Peak Acc %':<12}")
    print("-" * 50)
    
    def get_stats(res, method_name):
        final_acc = res['test_accuracy'][-1]
        peak_acc = max(res['test_accuracy'])
        stopped = res.get('stopped_epoch', len(res['test_accuracy']))
        print(f"{method_name:<15} | {stopped:<8} | {final_acc:<12.2f} | {peak_acc:<12.2f}")
        return {
            'method': method_name,
            'stopped': stopped,
            'final_acc': final_acc,
            'peak_acc': peak_acc
        }

    stats_ods = get_stats(results_ods, "ODS (Adaptive)")
    stats_plain = get_stats(results_plain, "Standard (Fix)")
    print("="*50)

    # Save results
    save_dir = os.path.join(_RESULTS_DIR, 'comparison')
    os.makedirs(save_dir, exist_ok=True)
    
    comparison_data = {
        'stats': [stats_ods, stats_plain],
        'ods_curves': results_ods,
        'plain_curves': results_plain
    }
    
    with open(os.path.join(save_dir, 'exp11_comparison.json'), 'w') as f:
        json.dump(comparison_data, f, indent=4)
        
    print(f"Comparison data saved to {save_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run small subset')
    args = parser.parse_args()

    if args.quick:
        run_comparison(subset_size=100, epochs=5)
    else:
        run_comparison(subset_size=1000, epochs=15)
