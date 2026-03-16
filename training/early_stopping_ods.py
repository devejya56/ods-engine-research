from ods_engine import ODSCore

class EarlyStoppingODS:
    """
    Backward compatibility wrapper for EarlyStoppingODS.
    Now uses the unified ods_engine.ODSCore.
    """
    def __init__(self, **kwargs):
        # Translate old parameter names if any (currently they match)
        self.core = ODSCore(**kwargs)

    def calculate(self, loss_tracker, grad_tracker, conf_tracker):
        """
        Adapts old tracker interfaces to the new ODSCore.
        """
        if hasattr(loss_tracker, 'calculate_smoothed_curvature'):
             curvature = loss_tracker.calculate_smoothed_curvature()
        else:
             curvature = loss_tracker.calculate_curvature()
             
        if hasattr(grad_tracker, 'normalized_norms') and grad_tracker.normalized_norms:
            grad_norm = grad_tracker.normalized_norms[-1]
        elif hasattr(grad_tracker, 'norms') and grad_tracker.norms:
            grad_norm = grad_tracker.norms[-1]
        else:
            grad_norm = 0.0
            
        confidence = conf_tracker.confidences[-1] if conf_tracker.confidences else 0.0
        
        ods_score = self.core.compute_score(curvature, grad_norm, confidence)
        should_stop, active_threshold = self.core.update(ods_score)
        
        return ods_score, should_stop

    def get_active_threshold(self):
        return self.core.adaptive_threshold if self.core.adaptive_threshold is not None else self.core.base_threshold

    @property
    def patience(self):
        return self.core.patience

    @property
    def warmup(self):
        return self.core.warmup
