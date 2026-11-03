"""
Signal-Existence Diagnostic (Items 2-4)

Runs three stages in order:
  1. Compute 5 summary stats from D_lip(t) per clip (mean, std, min, max, rate-of-change).
  2. Fit a plain logistic regression on those 5 features using the same connected-component
     identity split — a model too small to overfit (<<10 parameters).
  3. Compare test accuracy against majority-class baseline.
  4. If logreg BEATS majority-class: retrain fusion model with a shrunken LSTM
     (1 layer, hidden=16) + class weighting.
  5. If logreg DOES NOT beat majority-class: report as a finding — the visual trajectory
     does not separate real from fake independent of architecture size.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR
from src.models.train import split_dataset_by_identity


# ──────────────────────────────────────────────
# STEP 1: Load dataset & split
# ──────────────────────────────────────────────
dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
data_dict = torch.load(dataset_path, map_location="cpu")

train_dict, val_dict, test_dict = split_dataset_by_identity(data_dict)

def to_np(d):
    return (
        d["lip_sequences"].numpy(),
        d["desync_features"].numpy(),
        d["labels"].numpy().flatten(),
    )

X_lip_tr, X_de_tr, y_tr = to_np(train_dict)
X_lip_va, X_de_va, y_va = to_np(val_dict)
X_lip_te, X_de_te, y_te = to_np(test_dict)

n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)
maj_class = int(round(float(np.bincount(y_te.astype(int)).max()) / n_te * 100, 0))


# ──────────────────────────────────────────────
# STEP 2: Majority-class baseline
# ──────────────────────────────────────────────
majority_label = int(np.bincount(y_tr.astype(int)).argmax())
baseline_acc = float(np.mean(y_te == majority_label))

print("\n" + "=" * 70)
print("SIGNAL-EXISTENCE DIAGNOSTIC")
print("=" * 70)
print(f"Dataset  — Train: {n_tr} clips, Val: {n_va} clips, Test: {n_te} clips")
print(f"Label balance  — Train: {int((y_tr==0).sum())} real / {int((y_tr==1).sum())} fake")
print(f"                  Val:   {int((y_va==0).sum())} real / {int((y_va==1).sum())} fake")
print(f"                  Test:  {int((y_te==0).sum())} real / {int((y_te==1).sum())} fake")
print(f"Majority-class baseline on test: {baseline_acc*100:.1f}% (always predict label {majority_label})")


# ──────────────────────────────────────────────
# STEP 3: Extract 5 summary statistics per clip
# ──────────────────────────────────────────────
def summary_stats(seqs):
    """Returns (N, 5) array: mean, std, min, max, mean-abs-rate-of-change per clip."""
    means = seqs.mean(axis=1)
    stds  = seqs.std(axis=1)
    mins  = seqs.min(axis=1)
    maxs  = seqs.max(axis=1)
    roc   = np.abs(np.diff(seqs, axis=1)).mean(axis=1)
    return np.stack([means, stds, mins, maxs, roc], axis=1)

S_tr = summary_stats(X_lip_tr)
S_va = summary_stats(X_lip_va)
S_te = summary_stats(X_lip_te)

print(f"\n── Summary stats (mean across train clips) ──")
stat_names = ["mean", "std", "min", "max", "mean|Δ|"]
for i, name in enumerate(stat_names):
    print(f"  {name:12s}: real={S_tr[y_tr==0, i].mean():.5f}  fake={S_tr[y_tr==1, i].mean():.5f}")


# ──────────────────────────────────────────────
# STEP 4: Logistic regression on visual stats only
# ──────────────────────────────────────────────
print("\n── Logistic Regression (5 visual summary stats only) ──")
lr_vis = LogisticRegression(C=0.5, max_iter=1000, random_state=42,
                             class_weight="balanced", solver="lbfgs")
lr_vis.fit(S_tr, y_tr)

y_pred_vis = lr_vis.predict(S_te)
acc_vis = accuracy_score(y_te, y_pred_vis)
prec_vis = precision_score(y_te, y_pred_vis, zero_division=0)
rec_vis  = recall_score(y_te, y_pred_vis, zero_division=0)
f1_vis   = f1_score(y_te, y_pred_vis, zero_division=0)
cm_vis   = confusion_matrix(y_te, y_pred_vis)

print(f"  Test Accuracy : {acc_vis*100:.1f}%  (majority baseline = {baseline_acc*100:.1f}%)")
print(f"  Precision     : {prec_vis:.4f}")
print(f"  Recall        : {rec_vis:.4f}")
print(f"  F1-Score      : {f1_vis:.4f}")
print(f"  Confusion matrix (rows=actual, cols=pred):")
print(f"    [real->real={cm_vis[0,0]}, real->fake={cm_vis[0,1]}]")
print(f"    [fake->real={cm_vis[1,0]}, fake->fake={cm_vis[1,1]}]")
beats_visual = acc_vis > baseline_acc


# ──────────────────────────────────────────────
# STEP 5: Logistic regression on visual + desync
# ──────────────────────────────────────────────
print("\n── Logistic Regression (5 visual stats + 3 desync features) ──")
S_tr_full = np.concatenate([S_tr, X_de_tr], axis=1)
S_va_full = np.concatenate([S_va, X_de_va], axis=1)
S_te_full = np.concatenate([S_te, X_de_te], axis=1)

lr_full = LogisticRegression(C=0.5, max_iter=1000, random_state=42,
                              class_weight="balanced", solver="lbfgs")
lr_full.fit(S_tr_full, y_tr)

y_pred_full = lr_full.predict(S_te_full)
acc_full = accuracy_score(y_te, y_pred_full)
prec_full = precision_score(y_te, y_pred_full, zero_division=0)
rec_full  = recall_score(y_te, y_pred_full, zero_division=0)
f1_full   = f1_score(y_te, y_pred_full, zero_division=0)
cm_full   = confusion_matrix(y_te, y_pred_full)

print(f"  Test Accuracy : {acc_full*100:.1f}%  (majority baseline = {baseline_acc*100:.1f}%)")
print(f"  Precision     : {prec_full:.4f}")
print(f"  Recall        : {rec_full:.4f}")
print(f"  F1-Score      : {f1_full:.4f}")
print(f"  Confusion matrix (rows=actual, cols=pred):")
print(f"    [real->real={cm_full[0,0]}, real->fake={cm_full[0,1]}]")
print(f"    [fake->real={cm_full[1,0]}, fake->fake={cm_full[1,1]}]")
beats_full = acc_full > baseline_acc


# ──────────────────────────────────────────────
# STEP 6: Decision & Recommendation
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINDINGS SUMMARY")
print("=" * 70)
print(f"  Majority-class baseline      : {baseline_acc*100:.1f}%")
print(f"  LogReg (visual only)         : {acc_vis*100:.1f}%  → {'BEATS' if beats_visual else 'DOES NOT BEAT'} baseline")
print(f"  LogReg (visual + desync)     : {acc_full*100:.1f}%  → {'BEATS' if beats_full else 'DOES NOT BEAT'} baseline")

if beats_visual:
    print("""
FINDING: Visual D_lip(t) summary statistics DO carry signal separating
         real from fake clips independently of architecture size.
RECOMMENDATION (Item 3): Shrink the Bi-LSTM to 1 layer / hidden=16 and
         retrain with class_weight. See scripts/train_slim.py.
""")
else:
    print(f"""
FINDING: A 5-parameter logistic regression on D_lip(t) summary stats
         ({acc_vis*100:.1f}%) does NOT beat majority-class ({baseline_acc*100:.1f}%) on
         the held-out identity test split.
         {'Combined with desync features it does (' + str(round(acc_full*100,1)) + '%), suggesting the discriminative signal lives in the acoustic desync channel, not the visual trajectory alone.' if beats_full else 'Even adding 3-dim desync features does not beat baseline (' + str(round(acc_full*100,1)) + '%). The feature set has insufficient discriminative power on this small corpus.'}

RECOMMENDATION (Item 4): STOP tuning LSTM architecture — this is the
         wrong variable. The finding is: current corpus (N={n_tr + n_va + n_te} clips
         across {n_te} identities in test) does not provide enough signal in
         D_lip(t) alone to separate authentic from face-swapped video.
         Report this as a dataset-scope limitation, not a model failure.
""")
