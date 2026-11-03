"""
Checkpoint Re-evaluation: confusion matrices on train and test splits.
Uses model(...) output directly — no extra sigmoid call — matching train.py convention.
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR, get_device
from src.models.fusion_net import MultimodalKYCFusionNet
from src.models.train import split_dataset_by_identity


def confusion_block(y_true, y_pred, y_score, split_name):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    acc = (tn + tp) / max(1, len(y_true))

    print(f"\n{'='*55}")
    print(f"  {split_name}  (N={len(y_true)}, {int((y_true==0).sum())} real, {int((y_true==1).sum())} fake)")
    print(f"{'='*55}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"              Pred:Real   Pred:Fake")
    print(f"  Actual:Real    {tn:4d}       {fp:4d}     (real total={tn+fp})")
    print(f"  Actual:Fake    {fn:4d}       {tp:4d}     (fake total={fn+tp})")
    print(f"  Accuracy  : {acc*100:.1f}%")

    real_scores = y_score[y_true == 0]
    fake_scores = y_score[y_true == 1]
    print(f"\n  P(Fake) distribution — real clips : min={real_scores.min():.4f}  mean={real_scores.mean():.4f}  max={real_scores.max():.4f}")
    print(f"  P(Fake) distribution — fake clips : min={fake_scores.min():.4f}  mean={fake_scores.mean():.4f}  max={fake_scores.max():.4f}")
    print(f"  All P(Fake) values:")
    for i, (s, l) in enumerate(zip(y_score, y_true)):
        label = "real" if l == 0 else "fake"
        flag  = " ← WRONG" if (l == 0 and s >= 0.5) or (l == 1 and s < 0.5) else ""
        print(f"    [{i:3d}] {label:4s}  P(Fake)={s:.4f}{flag}")


device = get_device()

# Load dataset and split
data = torch.load(PROCESSED_DATA_DIR / "dataset.pt", map_location="cpu")
train_dict, val_dict, test_dict = split_dataset_by_identity(data)

# Load checkpoint
ckpt_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"
ckpt = torch.load(ckpt_path, map_location=device)

model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

print(f"\nCheckpoint epoch: {ckpt['epoch']}  (val_loss recorded={ckpt['val_loss']:.4f})")

for split_name, split in [("TRAIN SPLIT", train_dict), ("TEST SPLIT", test_dict)]:
    lips   = split["lip_sequences"].to(device)
    desync = split["desync_features"].to(device)
    labels = split["labels"].numpy().flatten()

    with torch.no_grad():
        scores = model(lips, desync).cpu().numpy().flatten()

    preds = (scores >= 0.5).astype(int)
    confusion_block(labels, preds, scores, split_name)
