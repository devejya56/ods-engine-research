import math

class ODSCore:
    """
    Core engine for Overfitting Detection Score calculations.
    """
    def __init__(self, alpha=150.0, beta=100.0, gamma=1.0,
                 threshold=2.0, patience=3, warmup=10,
                 adaptive=True, adaptive_k=1.2,
                 min_threshold=1.0, ema_alpha=0.4,
                 dataset_size=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_threshold = threshold
        self.patience = patience
        self.warmup = warmup
        self.adaptive = adaptive
        self.adaptive_k = adaptive_k
        self.min_threshold = min_threshold
        self.ema_alpha = ema_alpha

        # Auto-adjust patience if dataset size is provided
        if dataset_size is not None and dataset_size > 0:
            self.patience = max(3, dataset_size // 1000)

        self.ods_history = []
        self.current_smoothed_ods = None
        self.counter = 0
        self.stable_min_ods = None # Tracks the 'best' (lowest) score seen so far
        self.adaptive_threshold = None

    def compute_score(self, curvature, grad_norm, confidence):
        """Calculates the raw ODS score with EMA smoothing."""
        curv = max(0.0, curvature)
        raw_score = (self.alpha * curv) + (self.beta * grad_norm) + (self.gamma * confidence)
        
        if self.current_smoothed_ods is None:
            self.current_smoothed_ods = raw_score
        else:
            prev = self.current_smoothed_ods
            self.current_smoothed_ods = (self.ema_alpha * raw_score) + ((1.0 - self.ema_alpha) * prev)
            
        return self.current_smoothed_ods

    def _update_adaptive_threshold(self, current_ods):
        """
        Updates the threshold based on the minimum stable score seen.
        This relative approach is more sensitive than mean-based statistics.
        """
        if self.stable_min_ods is None or current_ods < self.stable_min_ods:
            self.stable_min_ods = current_ods
        
        # Threshold is a multiplier of the best observed score
        return max(self.min_threshold, self.stable_min_ods * self.adaptive_k)

    def update(self, ods_score):
        """
        Updates internal state and checks for early stopping.
        Returns (should_stop, current_threshold)
        """
        # Track best score seen so far even during warmup for better adaptive baseline
        if self.stable_min_ods is None or ods_score < self.stable_min_ods:
            self.stable_min_ods = ods_score

        self.ods_history.append(ods_score)
        
        current_epoch = len(self.ods_history)
        if current_epoch <= self.warmup:
            return False, self.base_threshold

        if self.adaptive:
            # Threshold is a multiplier of the best observed score
            self.adaptive_threshold = max(self.min_threshold, self.stable_min_ods * self.adaptive_k)

        active_threshold = self.adaptive_threshold if self.adaptive else self.base_threshold

        if ods_score > active_threshold:
            self.counter += 1
            if self.counter >= self.patience:
                return True, active_threshold
        else:
            self.counter = 0

        return False, active_threshold
