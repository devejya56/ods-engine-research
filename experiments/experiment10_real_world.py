import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from data_loaders.real_world_loader import get_realworld_dataloader
from ods_engine.wrappers import ODSConnector
from training.train import train_model

def get_pretrained_resnet_for_xray(num_classes=2):
    """Loads a pretrained ResNet-18 and replaces the head for binary classification."""
    print("Loading Pretrained ResNet-18...")
    # Standard torchvision usage for pretrained models
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers to simulate rapid transfer learning/fine-tuning
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze the last layer (layer4) and fc for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    # Replace the classifying head for 2 classes (Normal vs Pneumonia)
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def plot_explainability_dashboard(results, save_dir, exp_name):
    """
    Creates a layman-friendly, highly intuitive dashboard.
    Instead of complex 4-panel technical views, it focuses on the "Why we stopped" narrative.
    """
    epochs = range(1, len(results['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
    
    # Top Panel: The classic Overfitting story (Loss & Accuracy)
    ax1.plot(epochs, results['train_loss'], 'b-', marker='o', label='Training Loss (Memorization)')
    ax1.set_ylabel('Training Error (Lower is better)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Phase 13: X-Ray Prediction - Preventing Medical Overfitting', fontsize=14, pad=20)
    
    ax1_acc = ax1.twinx()
    ax1_acc.plot(epochs, results['test_accuracy'], 'g-', marker='s', label='Test Accuracy (Real World Performance)')
    ax1_acc.set_ylabel('Accuracy %', color='g')
    ax1_acc.tick_params(axis='y', labelcolor='g')
    
    # Highlight the Stop Point
    stop_epoch = results.get('stopped_epoch', len(epochs))
    if stop_epoch < len(epochs) or results.get('stop_method') == 'ODS':
        ax1.axvline(x=stop_epoch, color='r', linestyle='--', linewidth=2, label='ODS Early Stop')
        ax1.annotate('ODS stopped training here!\n(Before it memorizes noise)', 
                     xy=(stop_epoch, results['train_loss'][stop_epoch-1]), 
                     xytext=(stop_epoch-3, max(results['train_loss'])*0.8),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="r", lw=1))
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_acc.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom Panel: The ODS Engine Signal
    ax2.plot(epochs, results['ods_score'], 'purple', marker='^', linewidth=2, label='ODS Danger Signal')
    
    # Draw adaptive threshold line
    thresholds = results.get('ods_threshold', [4.0] * len(epochs))
    if isinstance(thresholds, list) and len(thresholds) == len(epochs):
        ax2.plot(epochs, thresholds, 'r:', label='Danger Threshold')
    else:
        # Fallback for scalar or mismatched length
        single_thresh = thresholds[-1] if isinstance(thresholds, list) else thresholds
        ax2.axhline(y=single_thresh, color='red', linestyle=':', label='Danger Threshold')
    
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Overfitting Risk Score', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.set_title('Under the Hood: The Overfitting Detection Score (ODS)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_explanation.png"), dpi=300)
    plt.close()
    print(f"Explainability dashboard saved to {os.path.join(save_dir, exp_name + '_explanation.png')}")


def run_experiment(subset_size=500, epochs=15):
    print(f"\n{'='*70}")
    print(f"  Experiment 10: Real World Application (Chest X-Ray Pneumonia)")
    print(f"{'='*70}")

    train_loader, test_loader = get_realworld_dataloader(
        dataset_name='mmenendezg/pneumonia_x_ray', 
        batch_size=32, subset_size=subset_size
    )

    model = get_pretrained_resnet_for_xray(num_classes=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ODS config tuned for quick fine-tuning of pretrained models
    ods_connector = ODSConnector(
        model=model,
        alpha=1.0, beta=10.0, gamma=1.0,
        threshold=3.0, patience=2, warmup=2, 
        adaptive=True, adaptive_k=1.5, 
        dataset_size=subset_size,
        track_sota_benchmarks=False 
    )

    optimizer = optim.Adam(model.layer4.parameters(), lr=1e-4) # Only update layer4 and fc
    optimizer.add_param_group({'params': model.fc.parameters(), 'lr': 1e-3})

    results = train_model(
        model=model, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, optimizer=optimizer, ods_connector=ods_connector
    )

    exp_name = "Exp10_RealWorld"
    if subset_size <= 200:
        exp_name += "_Quick"
    
    save_dir = os.path.join(_RESULTS_DIR, 'logs')
    graph_dir = os.path.join(_RESULTS_DIR, 'graphs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    with open(os.path.join(save_dir, f'{exp_name}_data.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # Generate the custom layman-friendly dashboard
    plot_explainability_dashboard(results, save_dir=graph_dir, exp_name=exp_name)

    print(f"  Finished: Stopped at {results['stopped_epoch']}, Peak: {max(results['test_accuracy']):.2f}%")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run a very small training subset')
    args = parser.parse_args()

    if args.quick:
        run_experiment(subset_size=100, epochs=5)
    else:
        run_experiment(subset_size=1000, epochs=15)
