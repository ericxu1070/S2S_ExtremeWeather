"""Tests for OcampoWeighting.

The original port had two silent defects this file pins down:
  1. its update schedule ((epoch - start) % every == 0) never intersected the trainer's
     validation epochs (epoch ≡ 4 mod 5 for val_every_n_epochs=5), so weights stayed
     uniform forever;
  2. its "cap" was min(w, w*cap) — an unconditional halving, not a ceiling.
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.variable_weighting import OcampoWeighting


def _val_epochs(n_epochs, val_every=5):
    """The epochs on which Trainer calls variable_weighting.update()."""
    return [e for e in range(n_epochs) if (e + 1) % val_every == 0]


class TestSchedule:
    def test_inactive_before_start_epoch(self):
        w = OcampoWeighting(n_vars=5, start_epoch=20, update_every=10,
                            logsp_idx=3, logtp_idx=4)
        w.update(4, torch.full((5,), 0.01))
        assert torch.equal(w.weights, torch.ones(5))
        assert w.get_weights(4, torch.device("cpu")) is None

    def test_fires_on_trainer_val_cadence(self):
        """With val epochs 4,9,14,19,24,29,34,... and start=20/every=10, updates must
        land at 24 and 34 — the default schedule (fires only at 20,30,40) never ran."""
        w = OcampoWeighting(n_vars=5, start_epoch=20, update_every=10,
                            logsp_idx=3, logtp_idx=4)
        fired = []
        for epoch in _val_epochs(40):
            before = w.last_update_epoch
            w.update(epoch, torch.full((5,), 0.01))
            if w.last_update_epoch != before:
                fired.append(epoch)
        assert fired == [24, 34]
        assert w.get_weights(24, torch.device("cpu")) is not None

    def test_weights_change_after_update(self):
        w = OcampoWeighting(n_vars=5, start_epoch=0, update_every=1,
                            logsp_idx=3, logtp_idx=4)
        w.update(0, torch.tensor([0.01, 0.02, 0.04, 0.01, 0.01]))
        assert not torch.allclose(w.weights, torch.ones(5))


class TestWeights:
    def test_cap_is_a_ceiling(self):
        """A tiny log_sp val loss must not hand it the whole gradient: its RAW weight is
        capped at logsp_cap, so post-normalization it sits BELOW equal-loss channels."""
        w = OcampoWeighting(n_vars=5, start_epoch=0, update_every=1,
                            logsp_idx=3, logtp_idx=4, base_weight=0.005,
                            logsp_cap=0.5, logtp_cap=0.5)
        # equal losses except log_sp, whose loss is ~0 (raw weight would be 5000x others)
        w.update(0, torch.tensor([0.005, 0.005, 0.005, 1e-6, 0.005]))
        ratio = (w.weights[3] / w.weights[0]).item()
        assert abs(ratio - 0.5) < 1e-5   # capped raw 0.5 vs raw 1.0 for the others

    def test_normalized_to_mean_one(self):
        """The total loss scale must not jump when the weighting switches on — only the
        relative weights carry information (0.005/val_loss is otherwise a 10-100x rescale)."""
        w = OcampoWeighting(n_vars=5, start_epoch=0, update_every=1,
                            logsp_idx=3, logtp_idx=4)
        w.update(0, torch.tensor([0.5, 0.05, 0.005, 0.01, 0.02]))
        assert abs(w.weights.mean().item() - 1.0) < 1e-6

    def test_state_dict_roundtrip(self):
        """Weights are training state: without checkpointing them, every resume would
        silently reset to uniform until the next update epoch."""
        w = OcampoWeighting(n_vars=5, start_epoch=0, update_every=10,
                            logsp_idx=3, logtp_idx=4)
        w.update(0, torch.tensor([0.01, 0.02, 0.04, 0.01, 0.03]))
        w2 = OcampoWeighting(n_vars=5, start_epoch=0, update_every=10,
                             logsp_idx=3, logtp_idx=4)
        w2.load_state_dict(w.state_dict())
        assert torch.equal(w2.weights, w.weights)
        assert w2.last_update_epoch == w.last_update_epoch
        # rate limit survives the round trip: epoch 5 is < 10 epochs after epoch 0
        w2.update(5, torch.full((5,), 0.01))
        assert w2.last_update_epoch == 0
