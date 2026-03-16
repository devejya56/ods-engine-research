from ods_engine import GradientTracker as EngineGradientTracker

class GradientTracker:
    """Compatibility wrapper for GradientTracker."""
    def __init__(self):
        self.tracker = EngineGradientTracker()

    def log(self, norm, normalized_norm=0.0):
        self.tracker.norms.append(norm)
        self.tracker.normalized_norms.append(normalized_norm)

    def calculate_norm(self, model):
        """Temporary call to capture current norm."""
        # This is slightly redundant with the new add() method but kept for train.py compatibility
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def calculate_normalized_norm(self, model):
        # Again, kept for train.py compatibility
        norm_sum = 0.0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                norm_sum += p.grad.data.norm(2).item() / p.numel()
                param_count += 1
        return norm_sum / max(1, param_count)

    @property
    def norms(self):
        return self.tracker.norms

    @property
    def normalized_norms(self):
        return self.tracker.normalized_norms
