import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ─── Global Style Configuration ──────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'lines.linewidth': 1.8,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# Color palette
COLORS = {
    'train_loss': '#E53935',      # Red
    'test_loss': '#FF8A65',       # Light orange
    'test_acc': '#43A047',        # Green
    'grad_norm': '#1E88E5',       # Blue
    'confidence': '#8E24AA',      # Purple
    'ods': '#00ACC1',             # Teal
    'stop_line': '#D32F2F',       # Dark red
    'phase_learn': '#E8F5E9',     # Light green bg
    'phase_transition': '#FFF9C4', # Light yellow bg
    'phase_memorize': '#FFEBEE',  # Light red bg
}


def _add_stop_marker(ax, stopped_epoch, stop_method, max_epoch):
    """Add vertical dashed line + annotation for early stopping."""
    if stop_method != 'none' and stopped_epoch < max_epoch:
        ax.axvline(x=stopped_epoch, color=COLORS['stop_line'], 
                   linestyle='--', linewidth=1.5, alpha=0.8, zorder=5)
        ymin, ymax = ax.get_ylim()
        ax.annotate(f'Stop ({stop_method})\nEpoch {stopped_epoch}',
                    xy=(stopped_epoch, ymax), fontsize=8,
                    ha='center', va='top', color=COLORS['stop_line'],
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              edgecolor=COLORS['stop_line'], alpha=0.85))


def _add_phase_regions(ax, epochs, train_losses, test_accs):
    """
    Auto-detect and shade the three training phases:
    1. Learning Phase: loss dropping + accuracy rising
    2. Transition Phase: loss still dropping but accuracy plateaus
    3. Memorization Phase: loss near zero, accuracy stagnant/dropping
    """
    if len(train_losses) < 10:
        return  # Not enough data to detect phases
    
    n = len(train_losses)
    
    # Detect transition: where test accuracy first plateaus
    # Use a rolling window to smooth
    window = max(5, n // 20)
    acc_smooth = np.convolve(test_accs, np.ones(window)/window, mode='valid')
    
    # Find where accuracy growth rate drops below threshold
    if len(acc_smooth) > 5:
        acc_diff = np.diff(acc_smooth)
        # Transition starts where growth rate first drops below 10% of max growth
        max_growth = np.max(np.abs(acc_diff[:len(acc_diff)//2])) + 1e-8
        transition_idx = None
        for i in range(len(acc_diff)):
            if acc_diff[i] / max_growth < 0.1 and i > n * 0.1:
                transition_idx = i + window // 2
                break
        
        if transition_idx is None:
            transition_idx = n // 3
        
        # Memorization starts where loss is < 10% of initial loss
        initial_loss = train_losses[0]
        memo_idx = n  # default: no memorization detected
        for i in range(transition_idx, n):
            if train_losses[i] < initial_loss * 0.05:
                memo_idx = i
                break
        
        epochs_list = list(epochs)
        ymin, ymax = ax.get_ylim()
        
        # Learning phase
        if transition_idx > 0:
            ax.axvspan(epochs_list[0], epochs_list[min(transition_idx, n-1)], 
                       alpha=0.08, color='green', zorder=0)
        # Transition phase
        if memo_idx > transition_idx:
            ax.axvspan(epochs_list[min(transition_idx, n-1)], epochs_list[min(memo_idx, n-1)],
                       alpha=0.08, color='orange', zorder=0)
        # Memorization phase
        if memo_idx < n:
            ax.axvspan(epochs_list[min(memo_idx, n-1)], epochs_list[-1],
                       alpha=0.08, color='red', zorder=0)


def plot_training_curves(results, save_dir, exp_name):
    """
    Creates a comprehensive training dynamics visualization with 5 panels:
    1. Loss curves (train + test) with phase regions
    2. Test accuracy with peak annotation
    3. Gradient norm evolution
    4. Prediction confidence
    5. ODS score with threshold
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(results['train_loss']) + 1))
    stopped_epoch = results.get('stopped_epoch', len(results['train_loss']))
    stop_method = results.get('stop_method', 'none')
    has_ods = 'ods_score' in results and len(results['ods_score']) > 0
    has_test_loss = 'test_loss' in results and len(results['test_loss']) > 0
    
    num_plots = 5 if has_ods else 4
    fig = plt.figure(figsize=(14, 3.5 * num_plots))
    gs = gridspec.GridSpec(num_plots, 1, hspace=0.35)

    # ── 1. Loss Curves ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, results['train_loss'], color=COLORS['train_loss'], 
             label='Train Loss', zorder=3)
    if has_test_loss:
        ax1.plot(epochs, results['test_loss'], color=COLORS['test_loss'], 
                 linestyle='--', label='Test Loss', zorder=3)
    
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.legend(loc='upper right')
    
    # Shade phases on the loss panel
    _add_phase_regions(ax1, epochs, results['train_loss'], results['test_accuracy'])
    _add_stop_marker(ax1, stopped_epoch, stop_method, max(epochs))

    # ── 2. Test Accuracy ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, results['test_accuracy'], color=COLORS['test_acc'], zorder=3)
    ax2.fill_between(epochs, results['test_accuracy'], alpha=0.1, 
                      color=COLORS['test_acc'], zorder=2)
    
    # Annotate peak accuracy
    if results['test_accuracy']:
        peak_acc = max(results['test_accuracy'])
        peak_epoch = results['test_accuracy'].index(peak_acc) + 1
        ax2.annotate(f'Peak: {peak_acc:.1f}%\n(Epoch {peak_epoch})',
                     xy=(peak_epoch, peak_acc),
                     xytext=(peak_epoch + len(epochs)*0.05, peak_acc - 3),
                     fontsize=9, color=COLORS['test_acc'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['test_acc'], lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                               edgecolor=COLORS['test_acc'], alpha=0.9))
    
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    _add_stop_marker(ax2, stopped_epoch, stop_method, max(epochs))

    # ── 3. Gradient Norm ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(epochs, results['grad_norm'], color=COLORS['grad_norm'], 
             alpha=0.4, linewidth=1, zorder=2)
    # Smoothed line
    if len(results['grad_norm']) > 10:
        window = max(5, len(results['grad_norm']) // 20)
        smoothed = np.convolve(results['grad_norm'], np.ones(window)/window, mode='valid')
        smooth_epochs = epochs[window//2:window//2+len(smoothed)]
        ax3.plot(smooth_epochs, smoothed, color=COLORS['grad_norm'], 
                 linewidth=2, label='Smoothed', zorder=3)
        ax3.legend(loc='upper right')
    
    ax3.set_title('Gradient Norm')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('||∇θ L||')
    _add_stop_marker(ax3, stopped_epoch, stop_method, max(epochs))

    # ── 4. Prediction Confidence ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(epochs, results['confidence'], color=COLORS['confidence'], zorder=3)
    ax4.fill_between(epochs, results['confidence'], alpha=0.1,
                      color=COLORS['confidence'], zorder=2)
    ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Max (1.0)')
    
    ax4.set_title('Prediction Confidence')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('max(softmax)')
    ax4.legend(loc='lower right')
    _add_stop_marker(ax4, stopped_epoch, stop_method, max(epochs))

    # ── 5. ODS Score ─────────────────────────────────────────────────────
    if has_ods:
        ax5 = fig.add_subplot(gs[4])
        ax5.plot(epochs, results['ods_score'], color=COLORS['ods'], 
                 linewidth=2, zorder=3)
        ax5.fill_between(epochs, results['ods_score'], alpha=0.1,
                          color=COLORS['ods'], zorder=2)
        
        # Show threshold if ODS params or history available
        thresholds = results.get('ods_threshold', None)
        if thresholds and isinstance(thresholds, list) and len(thresholds) == len(epochs):
             ax5.plot(epochs, thresholds, color=COLORS['stop_line'], linestyle=':',
                        linewidth=1.2, alpha=0.7, label='Adaptive Threshold')
             ax5.legend(loc='upper left')
        else:
            threshold = None
            if 'ods_params' in results:
                threshold = results['ods_params'].get('threshold', results['ods_params'].get('adaptive_threshold'))
            if threshold is not None:
                ax5.axhline(y=threshold, color=COLORS['stop_line'], linestyle=':',
                            linewidth=1.5, alpha=0.7, label=f'Threshold = {threshold}')
                ax5.legend(loc='upper left')
        
        ax5.set_title('Overfitting Detection Score (ODS)')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('ODS')
        _add_stop_marker(ax5, stopped_epoch, stop_method, max(epochs))

    # ── Title & Save ─────────────────────────────────────────────────────
    fig.suptitle(f'Training Dynamics — {exp_name}', fontsize=16, fontweight='bold', y=1.01)
    
    save_path = os.path.join(save_dir, f'{exp_name}_dynamics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Plot saved: {save_path}")


def plot_summary_dashboard(results, save_dir, exp_name):
    """
    Creates a compact 2x2 summary dashboard with the most important
    metrics overlaid for a quick overview.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(results['train_loss']) + 1))
    stopped_epoch = results.get('stopped_epoch', len(results['train_loss']))
    stop_method = results.get('stop_method', 'none')
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Summary Dashboard — {exp_name}', fontsize=16, fontweight='bold')

    # ── Top-left: Loss & Accuracy (dual axis) ─────────────────────────
    ax_loss = axs[0, 0]
    ax_acc = ax_loss.twinx()
    
    l1 = ax_loss.plot(epochs, results['train_loss'], color=COLORS['train_loss'],
                      label='Train Loss', linewidth=2)
    l2 = ax_acc.plot(epochs, results['test_accuracy'], color=COLORS['test_acc'],
                     label='Test Accuracy', linewidth=2)
    
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss', color=COLORS['train_loss'])
    ax_acc.set_ylabel('Accuracy (%)', color=COLORS['test_acc'])
    ax_loss.set_title('Performance Overview')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc='center right')

    # ── Top-right: Signal evolution (normalized) ──────────────────────
    ax_sig = axs[0, 1]
    
    # Normalize all signals to [0, 1] for comparison
    def normalize(arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)
    
    ax_sig.plot(epochs, normalize(results['grad_norm']), color=COLORS['grad_norm'],
                label='Gradient Norm', alpha=0.7)
    ax_sig.plot(epochs, normalize(results['confidence']), color=COLORS['confidence'],
                label='Confidence', alpha=0.7)
    if 'ods_score' in results and len(results['ods_score']) > 0:
        ax_sig.plot(epochs, normalize(results['ods_score']), color=COLORS['ods'],
                    label='ODS Score', linewidth=2.5)
    
    ax_sig.set_title('Normalized Signal Comparison')
    ax_sig.set_xlabel('Epoch')
    ax_sig.set_ylabel('Normalized [0, 1]')
    ax_sig.legend()

    # ── Bottom-left: Train-Test gap (generalization gap) ──────────────
    ax_gap = axs[1, 0]
    if 'test_loss' in results and len(results['test_loss']) > 0:
        gap = [t - tr for t, tr in zip(results['test_loss'], results['train_loss'])]
        ax_gap.plot(epochs, gap, color='#FF6F00', linewidth=2)
        ax_gap.fill_between(epochs, gap, alpha=0.15, color='#FF6F00')
        ax_gap.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax_gap.set_title('Generalization Gap (Test Loss − Train Loss)')
        ax_gap.set_ylabel('Loss Difference')
    else:
        ax_gap.text(0.5, 0.5, 'Test loss not tracked', ha='center', va='center',
                    transform=ax_gap.transAxes, fontsize=12, color='gray')
        ax_gap.set_title('Generalization Gap')
    ax_gap.set_xlabel('Epoch')

    # ── Bottom-right: Key metrics table ───────────────────────────────
    ax_table = axs[1, 1]
    ax_table.axis('off')
    
    peak_acc = max(results['test_accuracy']) if results['test_accuracy'] else 0.0
    peak_epoch = results['test_accuracy'].index(peak_acc) + 1 if results['test_accuracy'] else 0
    final_acc = results['test_accuracy'][-1] if results['test_accuracy'] else 0.0
    
    table_data = [
        ['Total Epochs', f"{len(epochs)}"],
        ['Stopped At', f"Epoch {stopped_epoch}" if stop_method != 'none' else 'N/A'],
        ['Stop Method', stop_method.upper() if stop_method != 'none' else 'None'],
        ['Final Accuracy', f"{final_acc:.2f}%"],
        ['Peak Accuracy', f"{peak_acc:.2f}% (Epoch {peak_epoch})"],
        ['Final Train Loss', f"{results['train_loss'][-1]:.4f}" if results['train_loss'] else "N/A"],
        ['Final Confidence', f"{results['confidence'][-1]:.4f}" if results['confidence'] else "N/A"],
    ]
    
    if 'ods_score' in results and len(results['ods_score']) > 0:
        table_data.append(['Final ODS', f"{results['ods_score'][-1]:.4f}"])
    
    threshold_val = 'N/A'
    if 'ods_threshold' in results and results['ods_threshold']:
        threshold_val = f"{results['ods_threshold'][-1]:.4f}"
    elif 'ods_params' in results:
        p = results['ods_params']
        threshold_val = p.get('threshold', p.get('adaptive_threshold', 'N/A'))
    
    table_data.append(['ODS Threshold', f"{threshold_val}"])

    if 'ods_params' in results:
        p = results['ods_params']
        table_data.append(['ODS Params', f"α={p.get('alpha','-')} β={p.get('beta','-')} γ={p.get('gamma','-')}"])
    
    table = ax_table.table(cellText=table_data, colLabels=['Metric', 'Value'],
                           loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    
    # Style header
    for j in range(2):
        table[(0, j)].set_facecolor('#37474F')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            table[(i, j)].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')
    
    ax_table.set_title('Key Metrics', pad=20)
    
    # Add stop markers to applicable plots
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        _add_stop_marker(ax, stopped_epoch, stop_method, max(epochs))

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{exp_name}_dashboard.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  📋 Dashboard saved: {save_path}")
