from .engine import ODSCore
from .trackers import LossTracker, GradientTracker, ConfidenceTracker, WeightNormTracker, FlatnessTracker
import torch

class ODSConnector:
    """
    Standard PyTorch training loop connector.
    Usage:
        ods = ODSConnector(model, ...)
        for epoch in epochs:
            # Training loop...
            ods.on_epoch_end(loss, outputs)
            if ods.should_stop(): break
    """
    def __init__(self, model, track_sota_benchmarks=False, **ods_kwargs):
        self.model = model
        self.engine = ODSCore(**ods_kwargs)
        self.loss_tracker = LossTracker()
        self.grad_tracker = GradientTracker()
        self.conf_tracker = ConfidenceTracker()
        
        # SOTA Benchmarks
        self.track_sota = track_sota_benchmarks
        self.weight_tracker = WeightNormTracker() if track_sota_benchmarks else None
        self.flatness_tracker = FlatnessTracker() if track_sota_benchmarks else None
        
        self.last_ods = 0.0
        self.active_threshold = 3.0
        self._should_stop = False

    def on_epoch_end(self, epoch_loss, epoch_outputs, valuation_data=None, criterion=None):
        """Captures epoch signals and updates ODS."""
        self.loss_tracker.add(epoch_loss)
        self.grad_tracker.add(self.model)
        self.conf_tracker.add(epoch_outputs)
        
        if self.track_sota:
            self.weight_tracker.add(self.model)
            if valuation_data and criterion:
                x, y = valuation_data
                self.flatness_tracker.add(self.model, x, y, criterion)
        
        # Calculate scores
        curv = self.loss_tracker.calculate_smoothed_curvature()
        grad = self.grad_tracker.normalized_norms[-1]
        conf = self.conf_tracker.confidences[-1]
        
        self.last_ods = self.engine.compute_score(curv, grad, conf)
        self._should_stop, self.active_threshold = self.engine.update(self.last_ods)
        
        return self._should_stop

    def should_stop(self):
        return getattr(self, '_should_stop', False)

    def get_metrics(self):
        metrics = {
            'ods_score': self.last_ods,
            'ods_threshold': self.active_threshold,
            'is_stopping': self.should_stop(),
            'grad_norm': self.grad_tracker.norms[-1] if self.grad_tracker.norms else 0.0,
            'grad_norm_normalized': self.grad_tracker.normalized_norms[-1] if self.grad_tracker.normalized_norms else 0.0,
            'confidence': self.conf_tracker.confidences[-1] if self.conf_tracker.confidences else 0.0
        }
        if self.track_sota:
            metrics['weight_norm'] = self.weight_tracker.norms[-1] if self.weight_tracker.norms else 0.0
            metrics['sharpness'] = self.flatness_tracker.sharpness_history[-1] if self.flatness_tracker.sharpness_history else 0.0
        return metrics

# Optional Lightning Integration
try:
    import pytorch_lightning as pl
    class ODSLightningCallback(pl.Callback):
        """
        PyTorch Lightning Callback for ODS Early Stopping.
        """
        def __init__(self, **ods_kwargs):
            super().__init__()
            self.ods_kwargs = ods_kwargs
            self.connector = None

        def on_train_start(self, trainer, pl_module):
            self.connector = ODSConnector(pl_module, **self.ods_kwargs)

        def on_train_epoch_end(self, trainer, pl_module):
            # Extract metrics from lightning trainer
            # Note: Lightning metrics naming varies by user preference
            avg_loss = trainer.callback_metrics.get("train_loss")
            
            # For confidence, we need model outputs. 
            # In Lightning, this usually requires custom logging.
            # This is a basic implementation placeholder.
            if avg_loss is not None:
                # In a real scenario, we'd pull a batch from val_dataloader for flatness
                # For now, we update with training loss and placeholder outputs
                stop = self.connector.on_epoch_end(avg_loss, torch.zeros(1, 10)) 
                self.log("ods_score", self.connector.last_ods)
                if stop:
                    trainer.should_stop = True

except ImportError:
    class ODSLightningCallback:
        def __init__(self, *args, **kwargs):
            raise ImportError("pytorch_lightning is not installed.")
