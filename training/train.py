import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, test_loader, epochs=200, lr=0.001, 
                optimizer=None,
                loss_tracker=None, grad_tracker=None, conf_tracker=None,
                early_stopping_ods=None, val_early_stopping=None,
                ods_connector=None):
    """
    Main training loop with support for:
    - Signal tracking (loss curvature, gradient norm, confidence)
    - ODS-based early stopping (classic and connector-based)
    - Validation-based early stopping
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    results = {
        'train_loss': [], 'test_accuracy': [], 'test_loss': [],
        'grad_norm': [], 'grad_norm_normalized': [], 'confidence': [],
        'ods_score': [], 'stopped_epoch': epochs, 'stop_method': 'none',
        'ods_threshold': []
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_confidences = []
        batch_grad_norms = []
        last_batch_outputs = None
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # --- Signal Capture ---
            if grad_tracker:
                batch_grad_norms.append(grad_tracker.calculate_norm(model))
            if conf_tracker:
                batch_confidences.append(conf_tracker.calculate_batch_confidence(outputs))
            
            last_batch_outputs = outputs
            # -----------------------

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- Update Trackers/Connector ---
        epoch_grad_norm = sum(batch_grad_norms)/len(batch_grad_norms) if batch_grad_norms else 0.0
        epoch_grad_norm_normalized = 0.0
        if grad_tracker:
            epoch_grad_norm_normalized = grad_tracker.calculate_normalized_norm(model)
            grad_tracker.log(epoch_grad_norm, normalized_norm=epoch_grad_norm_normalized)
        
        epoch_conf = sum(batch_confidences)/len(batch_confidences) if batch_confidences else 0.0
        if conf_tracker: conf_tracker.log(epoch_conf)
        if loss_tracker: loss_tracker.log(epoch_loss)

        # Evaluate
        acc, test_loss_avg = evaluate(model, test_loader, criterion, device)
        
        # Log to results
        results['train_loss'].append(epoch_loss)
        results['test_accuracy'].append(acc)
        results['test_loss'].append(test_loss_avg)
        
        # Capture metrics for results dict
        if ods_connector:
            m = ods_connector.get_metrics()
            # If trackers aren't provided, pull from connector for logging
            if not grad_tracker:
                results['grad_norm'].append(m.get('grad_norm', 0.0))
                results['grad_norm_normalized'].append(m.get('grad_norm_normalized', 0.0))
            else:
                results['grad_norm'].append(epoch_grad_norm)
                results['grad_norm_normalized'].append(epoch_grad_norm_normalized)
            
            if not conf_tracker:
                results['confidence'].append(m.get('confidence', 0.0))
            else:
                results['confidence'].append(epoch_conf)
        else:
            results['grad_norm'].append(epoch_grad_norm)
            results['grad_norm_normalized'].append(epoch_grad_norm_normalized)
            results['confidence'].append(epoch_conf)
        
        # --- ODS Logic ---
        ods_score = 0.0
        should_stop = False
        
        if ods_connector:
            # For SOTA Benchmarking: Pass a small sample of test data for Flatness calculation
            val_batch = None
            if getattr(ods_connector, 'track_sota', False):
                val_inputs, val_targets = next(iter(test_loader))
                val_batch = (val_inputs.to(device), val_targets.to(device))
            
            should_stop = ods_connector.on_epoch_end(epoch_loss, last_batch_outputs, 
                                                     valuation_data=val_batch, 
                                                     criterion=criterion)
            m = ods_connector.get_metrics()
            ods_score = m['ods_score']
            results['ods_threshold'].append(m['ods_threshold'])
            
            # Log SOTA metrics if available
            if 'weight_norm' in m:
                if 'weight_norm' not in results: results['weight_norm'] = []
                results['weight_norm'].append(m['weight_norm'])
            if 'sharpness' in m:
                if 'sharpness' not in results: results['sharpness'] = []
                results['sharpness'].append(m['sharpness'])
        elif early_stopping_ods:
            ods_score, should_stop = early_stopping_ods.calculate(loss_tracker, grad_tracker, conf_tracker)
            if hasattr(early_stopping_ods, 'get_active_threshold'):
                results['ods_threshold'].append(early_stopping_ods.get_active_threshold())
            else:
                results['ods_threshold'].append(0.0)

        results['ods_score'].append(ods_score)
        # Ensure ods_threshold is appended if not already done
        if not ods_connector and not early_stopping_ods:
             results['ods_threshold'].append(0.0)

        if should_stop:
            print(f">>> ODS Stop at {epoch} (Score: {ods_score:.4f})")
            results['stopped_epoch'], results['stop_method'] = epoch, 'ODS'
            break
        
        # --- Val Logic ---
        if val_early_stopping:
            # Support both a direct callable and a step-based object
            if hasattr(val_early_stopping, 'step'):
                should_stop_val = val_early_stopping.step(test_loss_avg)
            else:
                should_stop_val = val_early_stopping(test_loss_avg)
                
            if should_stop_val:
                print(f">>> Val Stop at {epoch}")
                results['stopped_epoch'], results['stop_method'] = epoch, 'validation'
                break

        print(f"Epoch [{epoch}/{epochs}] Loss: {epoch_loss:.4f} | Acc: {acc:.2f}% | ODS: {ods_score:.4f}")

    return results

def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, targets).item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return 100. * correct / total, loss / len(loader.dataset)
