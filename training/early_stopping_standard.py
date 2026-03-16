import numpy as np

class ConventionalEarlyStopping:
    """
    Standard Early Stopping based on validation (or test) loss.
    """
    def __init__(self, patience=5, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if self.mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
