import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

def generate_dashboard():
    print("Generating Final Research Dashboard...")
    configs = [
        ('CNN_1000', 'Exp5_CNN_1000_Original_data.json', 'Exp5_CNN_1000_Improved_data.json'),
        ('CNN_5000', 'Exp5_CNN_5000_Original_data.json', 'Exp5_CNN_5000_Improved_data.json'),
        ('ResNet_1000', 'Exp5_ResNet_1000_Original_data.json', 'Exp5_ResNet_1000_Improved_data.json'),
    ]

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

    peak_accs = {'Original': [], 'Improved': []}
    labels = []

    for i, (label, orig_file, imp_file) in enumerate(configs):
        orig_data = load_data(orig_file)
        imp_data = load_data(imp_file)

        if not orig_data or not imp_data:
            print(f"  [!] Skipping {label} - files missing (Original: {orig_file}, Improved: {imp_file})")
            continue

        labels.append(label)
        peak_accs['Original'].append(max(orig_data['test_accuracy']))
        peak_accs['Improved'].append(max(imp_data['test_accuracy']))

        ax = fig.add_subplot(gs[0, i])
        
        # Original
        ax.plot(orig_data['test_accuracy'], label='Original ODS', color='#FF9800', linewidth=2, alpha=0.8)
        # Improved
        ax.plot(imp_data['test_accuracy'], label='Improved ODS', color='#2196F3', linewidth=2.5)

        # Mark stop points
        if orig_data['stopped_epoch'] < len(orig_data['test_accuracy']):
            ax.plot(orig_data['stopped_epoch'], orig_data['test_accuracy'][orig_data['stopped_epoch']-1], 
                    'X', color='#E53935', markersize=10, label='Stopped (Orig)')
        
        if imp_data['stopped_epoch'] < len(imp_data['test_accuracy']):
             ax.plot(imp_data['stopped_epoch'], imp_data['test_accuracy'][imp_data['stopped_epoch']-1], 
                    'X', color='#43A047', markersize=10, label='Stopped (Imp)')

        ax.set_title(f'{label} Accuracy Comparison', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.legend(fontsize=8)

    if not labels:
        print("Error: No data loaded. Dashboard could not be generated.")
        return

    # Summary Bar Chart
    ax_bar = fig.add_subplot(gs[1, :])
    x = np.arange(len(labels))
    width = 0.35

    ax_bar.bar(x - width/2, peak_accs['Original'], width, label='Original ODS', color='#FFB74D')
    ax_bar.bar(x + width/2, peak_accs['Improved'], width, label='Improved ODS', color='#64B5F6')

    ax_bar.set_title('Peak Accuracy: Original vs. Improved ODS', fontweight='bold', fontsize=14)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel('Peak Test Accuracy (%)')
    ax_bar.legend()

    # Add numeric labels on bars
    for i, v in enumerate(peak_accs['Original']):
        ax_bar.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    for i, v in enumerate(peak_accs['Improved']):
        ax_bar.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

    plt.suptitle('Research Dashboard: Validation-Free Early Overfitting Detection (ODS)', fontsize=18, fontweight='bold', y=0.98)
    
    os.makedirs(_GRAPH_DIR, exist_ok=True)
    save_path = os.path.join(_GRAPH_DIR, 'final_exp5_dashboard.png')
    plt.savefig(save_path)
    plt.close()
    print(f"SUCCESS: Dashboard saved to: {save_path}")

if __name__ == '__main__':
    generate_dashboard()
