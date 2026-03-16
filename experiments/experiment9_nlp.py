import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, '..', 'results')

from data_loaders.nlp_loader import get_nlp_dataloader
from models.nlp_model import NLPModelWrapper
from ods_engine.wrappers import ODSConnector
from analysis.plot_training_curves import plot_training_curves, plot_summary_dashboard

def train_nlp_model(model, train_loader, test_loader, epochs=10, lr=2e-5, ods_connector=None):
    """
    NLP specific training loop.
    Re-implements parts of train.py because HuggingFace data_loaders format batches directly as dicts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Fine-tuning transformers usually requires a much smaller learning rate
    optimizer = optim.AdamW(model.parameters(), lr=lr) 
    
    results = {
        'train_loss': [], 'test_accuracy': [], 'test_loss': [],
        'ods_score': [], 'stopped_epoch': epochs, 'stop_method': 'none',
        'ods_threshold': [], 'grad_norm': [], 'confidence': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        last_batch_outputs = None
        
        for batch in train_loader:
            # HuggingFace data_loaders standard: input_ids, attention_mask, labels
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # returns logits inside our wrapper
            loss = criterion(outputs, targets)
            loss.backward()
            
            last_batch_outputs = outputs
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} Train Loss: {epoch_loss:.4f}")
        
        # Evaluate
        model.eval()
        correct, total, eval_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device)
                }
                targets = batch['label'].to(device)
                outputs = model(inputs)
                eval_loss += criterion(outputs, targets).item() * targets.size(0)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        
        acc = 100. * correct / total
        test_loss_avg = eval_loss / len(test_loader.dataset)
        
        results['train_loss'].append(epoch_loss)
        results['test_accuracy'].append(acc)
        results['test_loss'].append(test_loss_avg)
        
        ods_score = 0.0
        should_stop = False
        
        if ods_connector:
            val_batch = ({'input_ids': next(iter(test_loader))['input_ids'].to(device),
                          'attention_mask': next(iter(test_loader))['attention_mask'].to(device)}, 
                          next(iter(test_loader))['label'].to(device))
            
            should_stop = ods_connector.on_epoch_end(epoch_loss, last_batch_outputs, 
                                                     valuation_data=val_batch, 
                                                     criterion=criterion)
            m = ods_connector.get_metrics()
            ods_score = m['ods_score']
            results['ods_threshold'].append(m['ods_threshold'])
            
            if 'grad_norm' in m: results['grad_norm'].append(m['grad_norm'])
            if 'confidence' in m: results['confidence'].append(m['confidence'])
        else:
            results['ods_threshold'].append(0.0)

        results['ods_score'].append(ods_score)

        if should_stop:
            print(f">>> ODS Stop at {epoch} (Score: {ods_score:.4f})")
            results['stopped_epoch'], results['stop_method'] = epoch, 'ODS'
            break

        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f} | Acc: {acc:.2f}% | ODS: {ods_score:.4f}")

    return results

def run_experiment(subset_size=1000, epochs=15):
    print(f"\n{'='*70}")
    print(f"  Experiment 9: NLP Fine-Tuning Robustness (IMDB | DistilBERT)")
    print(f"{'='*70}")

    train_loader, test_loader = get_nlp_dataloader(
        dataset_name='imdb', model_name='distilbert-base-uncased', 
        batch_size=16, subset_size=subset_size
    )

    model = NLPModelWrapper(model_name='distilbert-base-uncased', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Note: Transformers have massive gradient norms due to parameter count. 
    # We use aggressive thresholding or rely more on curvature and confidence.
    ods_connector = ODSConnector(
        model=model,
        alpha=1.0, beta=10.0, gamma=1.5,
        threshold=4.0, patience=2, warmup=3, # Faster warmup for NLP
        adaptive=True, adaptive_k=3.0,
        dataset_size=subset_size,
        track_sota_benchmarks=False # Skip SOTA metrics to save memory on heavy models
    )

    results = train_nlp_model(
        model=model, train_loader=train_loader, test_loader=test_loader,
        epochs=epochs, lr=2e-5, ods_connector=ods_connector
    )

    exp_name = "Exp9_NLP"
    if subset_size <= 200:
        exp_name += "_Quick"
    
    save_dir = os.path.join(_RESULTS_DIR, 'logs')
    graph_dir = os.path.join(_RESULTS_DIR, 'graphs')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    results['ods_params'] = {
        'adaptive_threshold': ods_connector.engine.adaptive_threshold,
        'effective_patience': ods_connector.engine.patience,
        'dataset': 'imdb',
        'model': 'distilbert-base-uncased'
    }

    with open(os.path.join(save_dir, f'{exp_name}_data.json'), 'w') as f:
        json.dump(results, f, indent=4)

    try:
        plot_training_curves(results, save_dir=graph_dir, exp_name=exp_name)
    except Exception as e:
        print(f"Error plotting curves: {e}")
        
    try:
        plot_summary_dashboard(results, save_dir=graph_dir, exp_name=exp_name)
    except Exception as e:
        print(f"Error plotting dashboard: {e}")

    print(f"  Finished: Stopped at {results['stopped_epoch']}, Peak: {max(results['test_accuracy']):.2f}%")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run a very small training subset')
    args = parser.parse_args()

    if args.quick:
        run_experiment(subset_size=100, epochs=3)
    else:
        run_experiment(subset_size=1000, epochs=15)
