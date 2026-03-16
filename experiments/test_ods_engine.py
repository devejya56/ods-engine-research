import torch
import torch.nn as nn
from ods_engine import ODSCore, ODSConnector
from ods_engine.trackers import LossTracker, GradientTracker, ConfidenceTracker
import unittest

class TestODSEngine(unittest.TestCase):
    def setUp(self):
        # Sample model for tracker tests
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        # Dummy inputs and targets
        self.inputs = torch.randn(8, 10)
        self.targets = torch.randint(0, 2, (8,))
        self.criterion = nn.CrossEntropyLoss()

    def test_loss_tracker(self):
        tracker = LossTracker(smoothing_window=3)
        losses = [0.5, 0.4, 0.35, 0.3, 0.28]
        # To populate curvature history correctly, we call calculate_curvature after each add
        # (This is how it's used in practice)
        for l in losses:
            tracker.add(l)
            tracker.calculate_curvature()
        
        # Last three curvatures in history should be:
        # (0.35 - 2*0.4 + 0.5) = 0.05
        # (0.3 - 2*0.35 + 0.4) = 0.0
        # (0.28 - 2*0.3 + 0.35) = 0.03
        # _curvature_history = [0.0, 0.0, 0.05, 0.0, 0.03] (first two are 0.0 because len < 3)
        
        curv = tracker._curvature_history[-1]
        self.assertAlmostEqual(curv, 0.03, places=5)
        
        # calculate_smoothed_curvature will call calculate_curvature again!
        # _curvature_history will become [..., 0.03, 0.03]
        smoothed_curv = tracker.calculate_smoothed_curvature(window=2)
        # window 2 of [..., 0.03, 0.03] is 0.03
        self.assertAlmostEqual(smoothed_curv, 0.03, places=5)

    def test_gradient_tracker(self):
        tracker = GradientTracker()
        
        # Mock gradients
        for p in self.model.parameters():
            p.grad = torch.ones_like(p.data) * 0.1
            
        tracker.add(self.model)
        self.assertTrue(len(tracker.norms) == 1)
        self.assertTrue(tracker.norms[0] > 0)
        self.assertTrue(len(tracker.normalized_norms) == 1)

    def test_confidence_tracker(self):
        tracker = ConfidenceTracker()
        outputs = torch.tensor([[2.0, -1.0], [0.5, 0.5]]) # Big gap vs no gap
        tracker.add(outputs)
        self.assertTrue(len(tracker.confidences) == 1)
        self.assertTrue(0.5 <= tracker.confidences[0] <= 1.0)

    def test_ods_core_adaptive(self):
        core = ODSCore(warmup=2, adaptive=True, adaptive_k=1.1, patience=2, min_threshold=0.0)
        
        # Warmup phase
        stop, thresh = core.update(1.0) # Epoch 1
        self.assertFalse(stop)
        stop, thresh = core.update(0.8) # Epoch 2
        self.assertFalse(stop)
        
        # Post-warmup
        # stable_min = 0.8, thresh = 0.8 * 1.1 = 0.88
        stop, thresh = core.update(1.2) # Epoch 3: > 0.88 (counter=1)
        self.assertFalse(stop)
        self.assertAlmostEqual(thresh, 0.88)
        
        stop, thresh = core.update(1.3) # Epoch 4: > 0.88 (counter=2)
        self.assertTrue(stop)

    def test_ods_connector(self):
        connector = ODSConnector(self.model, warmup=0)
        outputs = torch.randn(8, 2)
        
        # Mock a training step
        loss = 0.5
        for p in self.model.parameters():
            p.grad = torch.randn_like(p.data)
            
        should_stop = connector.on_epoch_end(loss, outputs)
        metrics = connector.get_metrics()
        
        self.assertIn('ods_score', metrics)
        self.assertIn('grad_norm', metrics)
        self.assertIn('confidence', metrics)
        self.assertFalse(should_stop)

if __name__ == '__main__':
    unittest.main()
