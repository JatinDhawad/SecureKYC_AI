"""
SecureKYC AI - MultimodalKYCFusionNet Verification Script
End-to-end forward pass verification script testing the 131-dim fusion head
(128-dim Bi-LSTM + 3-dim desync_vec) on Apple Silicon MPS hardware.
"""

import sys
from pathlib import Path
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_device, DEVICE
from src.models.fusion_net import MultimodalKYCFusionNet


def verify_fusion_net_forward_pass():
    """
    Instantiates MultimodalKYCFusionNet, passes a random batch on torch.device('mps'),
    and asserts correct output shape (B, 1) with zero dimension errors.
    """
    print("=" * 70)
    print(" MultimodalKYCFusionNet End-to-End Forward Pass Verification ")
    print("=" * 70)

    device = get_device()
    print(f"[*] Compute Backend Target: {device}")

    # 1. Instantiate MultimodalKYCFusionNet
    model = MultimodalKYCFusionNet(
        seq_len=100,
        desync_feature_dim=3,
        conv_out_channels=32,
        lstm_hidden_dim=64,  # Bi-LSTM output = 64 * 2 = 128 dims
        lstm_num_layers=2,
        dropout_rate=0.3,
    ).to(device)

    model.eval()

    # 2. Generate Random Test Batch
    batch_size = 8
    seq_len = 100

    lip_sequence = torch.randn((batch_size, seq_len), device=device, dtype=torch.float32)

    # 3-dim desync_features: [mean_penalty (S_desync), max_penalty, failed_ratio]
    desync_features = torch.tensor([
        [0.00, 0.00, 0.00],  # Authentic aligned stream sample
        [0.12, 0.25, 0.60],  # Desynced deepfake sample
        [0.05, 0.10, 0.20],
        [0.22, 0.40, 0.80],
        [0.01, 0.02, 0.00],
        [0.15, 0.30, 0.50],
        [0.00, 0.00, 0.00],
        [0.35, 0.55, 1.00],
    ], device=device, dtype=torch.float32)

    print(f"[*] Input Tensor Shapes:")
    print(f"    - lip_sequence:     {list(lip_sequence.shape)} on {lip_sequence.device}")
    print(f"    - desync_features:  {list(desync_features.shape)} on {desync_features.device}")
    print(f"      (Dim 0: Mean S_desync | Dim 1: Max Penalty | Dim 2: Failed Ratio)")

    # 3. Execute Model Forward Pass
    with torch.no_grad():
        output = model(lip_sequence, desync_features)

    print(f"\n[✓] Forward Pass Completed Successfully!")
    print(f"    - Output Tensor Shape:  {list(output.shape)}")
    print(f"    - Output Device:        {output.device}")
    print(f"    - Deepfake Probabilities (first 4 batch items):")
    for b_idx in range(4):
        print(f"      • Sample {b_idx + 1}: P(Deepfake) = {output[b_idx].item():.4f}")

    # Strict Assertions
    assert output.shape == (batch_size, 1), f"Expected output shape ({batch_size}, 1), got {list(output.shape)}"
    assert (output >= 0.0).all() and (output <= 1.0).all(), "Output probabilities must be in range [0.0, 1.0]"

    print("=" * 70)
    print(" SUCCESS: MultimodalKYCFusionNet 131-dim fusion forward pass verified.")
    print("=" * 70)


if __name__ == "__main__":
    verify_fusion_net_forward_pass()
