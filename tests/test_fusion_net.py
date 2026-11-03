"""
Unit tests for src/models/fusion_net.py
Verifies MultimodalKYCFusionNet architecture, 131-dim fusion head (128 Bi-LSTM + 3 desync_features),
forward pass shapes, and ablation models.
"""

import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fusion_net import MultimodalKYCFusionNet, VisionOnlyNet, DesyncOnlyNet, ConstrainedLateFusion


def test_fusion_net_forward_pass():
    model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3)
    batch_size = 4
    lip_seq = torch.randn(batch_size, 100)
    desync_feats = torch.randn(batch_size, 3)

    out = model(lip_seq, desync_feats)

    assert out.shape == (batch_size, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_fusion_net_3d_input_shape():
    model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3)
    batch_size = 2
    lip_seq_3d = torch.randn(batch_size, 1, 100)
    desync_feats = torch.randn(batch_size, 3)

    out = model(lip_seq_3d, desync_feats)
    assert out.shape == (batch_size, 1)


def test_fusion_head_dimension_alignment():
    """Asserts that Bi-LSTM 128-dim embedding + 3-dim desync_vec yields 131-dim fusion linear layer."""
    model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3, lstm_hidden_dim=64)
    first_fusion_linear = model.fusion_fc[0]

    assert isinstance(first_fusion_linear, torch.nn.Linear)
    assert first_fusion_linear.in_features == 131  # 64*2 (128) + 3 = 131


def test_ablation_models():
    v_net = VisionOnlyNet(seq_len=100)
    d_net = DesyncOnlyNet(desync_feature_dim=3)

    batch_size = 2
    lip_seq = torch.randn(batch_size, 100)
    desync_feats = torch.randn(batch_size, 3)

    out_v = v_net(lip_seq)
    out_d = d_net(desync_features=desync_feats)

    assert out_v.shape == (batch_size, 1)
    assert out_d.shape == (batch_size, 1)


def test_constrained_late_fusion():
    v_net = VisionOnlyNet(seq_len=100)
    d_net = DesyncOnlyNet(desync_feature_dim=3)
    fusion_model = ConstrainedLateFusion(
        vision_net=v_net,
        desync_net=d_net,
        w_vis=5.3688,
        w_desync=4.9111,
        intercept=-6.0411,
    )

    batch_size = 4
    lip_seq = torch.randn(batch_size, 100)
    desync_feats = torch.randn(batch_size, 3)

    out = fusion_model(lip_seq, desync_feats)
    assert out.shape == (batch_size, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

