import torch
import numpy as np

class LossTracker:
    def __init__(self, smoothing_window=5):
        self.losses = []
        self._curvature_history = []
        self.smoothing_window = smoothing_window

    def add(self, loss):
        self.losses.append(float(loss))

    def calculate_curvature(self):
        """Second-order finite difference: Rt = Lt+1 - 2Lt + Lt-1"""
        if len(self.losses) < 3:
            return 0.0
        Lt_plus_1 = self.losses[-1]
        Lt = self.losses[-2]
        Lt_minus_1 = self.losses[-3]
        curvature = Lt_plus_1 - 2 * Lt + Lt_minus_1
        self._curvature_history.append(curvature)
        return max(0.0, curvature)

    def calculate_smoothed_curvature(self, window=None):
        """Smoothed curvature via windowed average of recent curvature values."""
        if len(self.losses) < 3:
            return 0.0
        
        window = window or self.smoothing_window
        curvature = self.calculate_curvature()
        
        recent = self._curvature_history[-window:]
        return sum(recent) / len(recent)

    def calculate_curvature_trend(self, window=None):
        """
        Rate of change of curvature (is overfitting accelerating?).
        Uses slope of last `window` curvature values via simple linear regression.
        """
        window = window or self.smoothing_window
        if len(self._curvature_history) < window:
            return 0.0

        recent = self._curvature_history[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0
        return numerator / denominator

class GradientTracker:
    def __init__(self):
        self.norms = []
        self.normalized_norms = []

    def add(self, model):
        """Captures global norm and per-parameter normalized norm."""
        total_norm = 0.0
        norm_sum = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # Normalize by sqrt(numel) for more balanced signal across layers
                norm_sum += param_norm / (p.numel()**0.5)
                param_count += 1
        
        self.norms.append(total_norm ** 0.5)
        self.normalized_norms.append(norm_sum / max(1, param_count))

class ConfidenceTracker:
    def __init__(self):
        self.confidences = []

    def add(self, outputs):
        """Average maximum probability across the batch."""
        probs = torch.softmax(outputs, dim=1)
        avg_max_prob = probs.max(dim=1)[0].mean().item()
        self.confidences.append(avg_max_prob)

class WeightNormTracker:
    """Tracks L2 norm of model parameters - a common indicator of overfitting/memorization."""
    def __init__(self):
        self.norms = []

    def add(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.requires_grad:
                total_norm += p.data.norm(2).item() ** 2
        self.norms.append(total_norm ** 0.5)

class FlatnessTracker:
    """
    Estimates Loss Landscape Flatness (inverse of Sharpness).
    Calculates the change in loss under small random perturbations of weights.
    Higher values = Sharper landscape = Likely overfitting.
    """
    def __init__(self, perturbation_epsilon=0.01):
        self.epsilon = perturbation_epsilon
        self.sharpness_history = []

    def add(self, model, x, y, criterion):
        """
        Calculates sharpness: L(w + eps) - L(w).
        Note: This is a stochastic estimate using a single random perturbation.
        """
        model.eval()
        with torch.no_grad():
            original_loss = criterion(model(x), y).item()
            
            # Perturb
            original_params = [p.data.clone() for p in model.parameters()]
            for p in model.parameters():
                if p.requires_grad:
                    # Scale perturbation by parameter norm for relative flatness
                    noise = torch.randn_like(p) * self.epsilon * (p.data.norm(2) + 1e-6)
                    p.data.add_(noise)
            
            perturbed_loss = criterion(model(x), y).item()
            
            # Restore
            for p, original in zip(model.parameters(), original_params):
                p.data.copy_(original)
                
            sharpness = max(0.0, (perturbed_loss - original_loss) / (original_loss + 1e-6))
            self.sharpness_history.append(sharpness)
        model.train()
