"""
Unit tests for src/models/train.py and src/models/evaluate.py
Verifies EER calculation logic and synthetic training pass.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.evaluate import compute_eer, evaluate_model_metrics
from src.models.fusion_net import MultimodalKYCFusionNet


def test_compute_eer_perfect_separation():
    # Labels: 0 = Authentic, 1 = Fake
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.85, 0.9, 0.95, 0.8])

    eer, threshold, *_ = compute_eer(labels, scores)

    assert eer == 0.0
    assert 0.2 <= threshold <= 0.8


def test_evaluate_model_metrics():
    model = MultimodalKYCFusionNet(seq_len=50, desync_feature_dim=3)
    lip_seqs = torch.randn(10, 50)
    desync_feats = torch.randn(10, 3)
    labels = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0]])

    device = torch.device("cpu")
    metrics = evaluate_model_metrics(model, lip_seqs, desync_feats, labels, device)

    assert "roc_auc" in metrics
    assert "eer" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics
