from ods_engine import LossTracker as EngineLossTracker

class LossTracker:
    """Compatibility wrapper for LossTracker."""
    def __init__(self, smoothing_window=5):
        self.tracker = EngineLossTracker(smoothing_window=smoothing_window)

    def log(self, loss):
        self.tracker.add(loss)

    def calculate_curvature(self):
        return self.tracker.calculate_curvature()

    def calculate_smoothed_curvature(self, window=5):
        return self.tracker.calculate_smoothed_curvature(window=window)

    def calculate_curvature_trend(self, window=5):
        return self.tracker.calculate_curvature_trend(window=window)

    @property
    def losses(self):
        return self.tracker.losses

    @property
    def _curvature_history(self):
        return self.tracker._curvature_history
