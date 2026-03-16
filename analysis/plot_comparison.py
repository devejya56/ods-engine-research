import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ─── Global Style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

ODS_COLOR = '#2196F3'       # Blue
VAL_COLOR = '#FF9800'       # Orange
ACCENT_RED = '#E53935'
ACCENT_GREEN = '#43A047'


def plot_comparison(all_results, save_dir, exp_name):
    """
    Generates a polished comparison dashboard: ODS vs Validation stopping.
    
    Layout:
    ┌────────────────────┬────────────────────┐
    │  Stopped Epoch     │  Peak Accuracy     │
    ├────────────────────┴────────────────────┤
    │         Summary Table                    │
    └──────────────────────────────────────────┘
    """
    os.makedirs(save_dir, exist_ok=True)
    
    labels = list(all_results.keys())
    stopped_epochs = [r['stopped_epoch'] for r in all_results.values()]
    final_accs = [r['test_accuracy'][-1] for r in all_results.values()]
    peak_accs = [max(r['test_accuracy']) for r in all_results.values()]
    
    colors = [ODS_COLOR if 'ODS' in l else VAL_COLOR for l in labels]
    short_labels = [l.replace('Exp3_', '').replace('_', '\n') for l in labels]
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], hspace=0.4, wspace=0.3)

    # ── 1. Stopped Epoch ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(short_labels, stopped_epochs, color=colors, alpha=0.85, 
                   edgecolor='white', linewidth=1.5, width=0.6)
    
    for bar, val in zip(bars, stopped_epochs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_title('When Did Training Stop?')
    ax1.set_ylabel('Epoch')
    ax1.set_ylim(0, max(stopped_epochs) * 1.15)

    # ── 2. Peak Accuracy ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(short_labels, peak_accs, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, width=0.6)
    
    for bar, val in zip(bars, peak_accs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_title('How Good Was the Model?')
    ax2.set_ylabel('Peak Test Accuracy (%)')
    min_acc = min(peak_accs)
    ax2.set_ylim(max(0, min_acc - 10), max(peak_accs) * 1.05)

    # ── 3. Summary Table ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    col_headers = ['Experiment', 'Method', 'Stopped Epoch', 'Final Acc (%)', 
                   'Peak Acc (%)', 'Epochs Saved vs Full']
    
    table_data = []
    for label, results in all_results.items():
        method = 'ODS' if 'ODS' in label else 'Validation'
        stopped = results['stopped_epoch']
        total = len(results['train_loss'])
        saved = total - stopped if results.get('stop_method', 'none') != 'none' else 0
        table_data.append([
            label.replace('Exp3_', ''),
            method,
            str(stopped),
            f"{results['test_accuracy'][-1]:.1f}",
            f"{max(results['test_accuracy']):.1f}",
            f"{saved} ({100*saved/total:.0f}%)" if saved > 0 else "—"
        ])
    
    table = ax3.table(cellText=table_data, colLabels=col_headers,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header
    for j in range(len(col_headers)):
        table[(0, j)].set_facecolor('#37474F')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Color-code method column
    for i in range(1, len(table_data) + 1):
        method_text = table_data[i-1][1]
        bg_color = '#E3F2FD' if method_text == 'ODS' else '#FFF3E0'
        for j in range(len(col_headers)):
            table[(i, j)].set_facecolor(bg_color if j == 1 else 
                                        ('#F5F5F5' if i % 2 == 0 else 'white'))
        table[(i, 1)].set_facecolor(bg_color)
    
    ax3.set_title('Detailed Comparison', pad=20, fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ODS_COLOR, alpha=0.85, label='ODS (No Validation Set)'),
                       Patch(facecolor=VAL_COLOR, alpha=0.85, label='Validation-Based')]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 0.98))
    
    fig.suptitle(f'Comparative Study — {exp_name}', fontsize=16, fontweight='bold', y=1.02)
    
    save_path = os.path.join(save_dir, f'{exp_name}_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Comparison saved: {save_path}")


def plot_accuracy_curves_overlay(all_results, save_dir, exp_name):
    """
    Overlays test accuracy curves with ODS/Val color coding and stop markers.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for label, results in all_results.items():
        epochs = range(1, len(results['test_accuracy']) + 1)
        stopped = results['stopped_epoch']
        is_ods = 'ODS' in label
        color = ODS_COLOR if is_ods else VAL_COLOR
        style = '-' if is_ods else '--'
        short = label.replace('Exp3_', '')
        
        ax.plot(epochs, results['test_accuracy'], color=color, linestyle=style,
                linewidth=2, label=short, zorder=3)
        
        # Mark the stop point
        stop_method = results.get('stop_method', 'none')
        if stop_method != 'none' and stopped <= len(results['test_accuracy']):
            stop_acc = results['test_accuracy'][stopped - 1]
            marker = 'D' if is_ods else 's'
            ax.plot(stopped, stop_acc, marker=marker, color=color, markersize=10,
                    zorder=5, markeredgecolor='white', markeredgewidth=1.5)
            ax.annotate(f'{stop_acc:.1f}%', xy=(stopped, stop_acc),
                        xytext=(5, 8), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')
    
    ax.set_title(f'Test Accuracy Comparison — {exp_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.legend(loc='lower right', ncol=2)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{exp_name}_accuracy_overlay.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  📈 Accuracy overlay saved: {save_path}")
