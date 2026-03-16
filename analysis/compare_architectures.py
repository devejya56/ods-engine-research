import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Style setup
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': '#FFFFFF',
    'savefig.dpi': 200,
})

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results', 'logs')
_GRAPH_DIR = os.path.join(_PROJECT_ROOT, '..', 'results', 'graphs')

def load_data(filename):
    path = os.path.join(_RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def generate_comparison():
    print("Generating Architecture Comparison (CNN vs Transformer)...")
    
    # Using ResNet as the CNN baseline
    cnn_data = load_data('Exp5_ResNet_1000_Improved_data.json')
    # Using ViT as the Transformer
    vit_data = load_data('Exp8_ViT_ODS_VerySmall_data.json')

    if not cnn_data or not vit_data:
        print(" [!] Missing data files for comparison.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle('Architecture Robustness Analysis: CNN (ResNet) vs. Transformer (ViT)', fontsize=18, fontweight='bold')

    # 1. Accuracy curves
    axs[0, 0].plot(cnn_data['test_accuracy'], label='CNN (ResNet-18)', color='#1E88E5', linewidth=2)
    axs[0, 0].plot(vit_data['test_accuracy'], label='Transformer (ViT-Tiny)', color='#8E24AA', linewidth=2)
    axs[0, 0].set_title('Test Accuracy Comparison', fontweight='bold')
    axs[0, 0].set_ylabel('Accuracy (%)')
    axs[0, 0].legend()

    # 2. ODS Scores
    axs[0, 1].plot(cnn_data['ods_score'], label='CNN ODS', color='#1E88E5', alpha=0.6)
    axs[0, 1].plot(vit_data['ods_score'], label='Transformer ODS', color='#8E24AA', linewidth=2)
    axs[0, 1].set_title('ODS Scoring Dynamics', fontweight='bold')
    axs[0, 1].set_ylabel('ODS Score')
    axs[0, 1].legend()

    # 3. Normalized Gradients
    axs[1, 0].plot(cnn_data['grad_norm'], label='CNN Gradient Norm', color='#1E88E5', alpha=0.5)
    axs[1, 0].plot(vit_data['grad_norm'], label='Transformer Gradient Norm', color='#8E24AA', alpha=0.7)
    axs[1, 0].set_title('Gradient Norm Comparison (Raw)', fontweight='bold')
    axs[1, 0].legend()

    # 4. Final Performance Bar Chart
    labels = ['CNN (ResNet)', 'Transformer (ViT)']
    peak_accs = [max(cnn_data['test_accuracy']), max(vit_data['test_accuracy'])]
    stop_epochs = [cnn_data['stopped_epoch'], vit_data['stopped_epoch']]

    x = np.arange(len(labels))
    width = 0.35
    axs[1, 1].bar(x, peak_accs, width, color=['#1E88E5', '#8E24AA'])
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(labels)
    axs[1, 1].set_title('Peak Accuracy achieved with ODS Stopping', fontweight='bold')
    axs[1, 1].set_ylabel('Accuracy (%)')

    for i, v in enumerate(peak_accs):
        axs[1, 1].text(i, v + 0.5, f'{v:.1f}%\n(Stop @ {stop_epochs[i]})', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(_GRAPH_DIR, exist_ok=True)
    save_path = os.path.join(_GRAPH_DIR, 'arch_comparison_cnn_vs_vit.png')
    plt.savefig(save_path)
    plt.close()
    print(f"SUCCESS: Comparison Plot saved to: {save_path}")

if __name__ == '__main__':
    generate_comparison()
