from ods_engine import ConfidenceTracker as EngineConfidenceTracker
import torch

class ConfidenceTracker:
    """Compatibility wrapper for ConfidenceTracker."""
    def __init__(self):
        self.tracker = EngineConfidenceTracker()

    def log(self, confidence):
        self.tracker.confidences.append(confidence)

    def calculate_batch_confidence(self, outputs):
        """Temporary call to capture current batch confidence."""
        probs = torch.softmax(outputs, dim=1)
        return probs.max(dim=1)[0].mean().item()

    @property
    def confidences(self):
        return self.tracker.confidences
