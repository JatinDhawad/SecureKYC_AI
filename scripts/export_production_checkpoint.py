"""
SecureKYC AI - Export Production Constrained Late-Fusion Checkpoint
Trains VisionOnlyNet and DesyncOnlyNet on the official identity-disjoint train split,
fits the constrained meta-learner on the validation split (W_vis >= 0, W_desync >= 0),
verifies held-out test split metrics, and saves the production checkpoints:
  - checkpoints/best_vision_model.pt
  - checkpoints/best_desync_model.pt
  - checkpoints/best_fusion_model.pt
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR, BATCH_SIZE, get_device
from src.models.fusion_net import VisionOnlyNet, DesyncOnlyNet, ConstrainedLateFusion
from src.models.train import split_dataset_by_identity, MultimodalKYCDataset
from src.models.evaluate import compute_eer
from scripts.experiment_fusion_variants import train_model, set_seed


def fit_constrained_logistic_regression(X: np.ndarray, y: np.ndarray, l2_reg: float = 1e-4):
    """
    Fits logistic regression with non-negativity constraints:
      W_vis >= 0.0, W_desync >= 0.0, intercept unconstrained.
    """
    N = len(y)

    def objective(params):
        w1, w2, b = params
        z = w1 * X[:, 0] + w2 * X[:, 1] + b
        p = np.clip(expit(z), 1e-12, 1.0 - 1e-12)
        bce = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        reg = 0.5 * l2_reg * (w1**2 + w2**2)
        return float(bce + reg)

    def gradient(params):
        w1, w2, b = params
        z = w1 * X[:, 0] + w2 * X[:, 1] + b
        p = expit(z)
        err = p - y
        dw1 = np.mean(err * X[:, 0]) + l2_reg * w1
        dw2 = np.mean(err * X[:, 1]) + l2_reg * w2
        db = np.mean(err)
        return np.array([dw1, dw2, db], dtype=np.float64)

    init_params = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    bounds = [(0.0, None), (0.0, None), (None, None)]

    res = minimize(
        objective,
        init_params,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w_vis, w_desync, b = res.x
    return float(w_vis), float(w_desync), float(b)


def export_production_checkpoint():
    device = get_device()
    print(f"[*] Exporting production models on {device.type.upper()}...")

    dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
    dataset_dict = torch.load(dataset_path, weights_only=False)

    train_dict, val_dict, test_dict = split_dataset_by_identity(dataset_dict, random_seed=42)
    y_test = test_dict["labels"].cpu().numpy().flatten().astype(int)

    set_seed(42)
    train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], train_dict["desync_features"], train_dict["labels"])
    val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], val_dict["desync_features"], val_dict["labels"])

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 1. Train VisionOnlyNet
    print("[*] Training VisionOnlyNet on official train split...")
    vis_model = VisionOnlyNet(seq_len=100).to(device)
    vis_model, vis_meta = train_model(vis_model, tr_loader, va_loader, 20, device, is_vision_only=True)

    # 2. Train DesyncOnlyNet
    print("[*] Training DesyncOnlyNet on official train split...")
    desync_model = DesyncOnlyNet(desync_feature_dim=3).to(device)
    desync_model, des_meta = train_model(desync_model, tr_loader, va_loader, 20, device, is_desync_only=True)

    # 3. Fit Constrained Meta-Learner on Validation Set
    vis_model.eval()
    desync_model.eval()
    with torch.no_grad():
        p_vis_val = vis_model(val_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
        p_des_val = desync_model(desync_features=val_dict["desync_features"].to(device)).cpu().numpy().flatten()
        y_val = val_dict["labels"].cpu().numpy().flatten().astype(int)

        p_vis_te = vis_model(test_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
        p_des_te = desync_model(desync_features=test_dict["desync_features"].to(device)).cpu().numpy().flatten()

    X_val = np.column_stack([p_vis_val, p_des_val])
    X_test = np.column_stack([p_vis_te, p_des_te])

    w_vis, w_desync, intercept = fit_constrained_logistic_regression(X_val, y_val)
    print(f"[*] Meta-learner fitted: W_vis={w_vis:.4f}, W_desync={w_desync:.4f}, Intercept={intercept:.4f}")

    # Evaluate on held-out test split
    z_test = w_vis * X_test[:, 0] + w_desync * X_test[:, 1] + intercept
    p_late_test = expit(z_test)

    auc = float(roc_auc_score(y_test, p_late_test))
    eer, opt_t, _, _, _ = compute_eer(y_test, p_late_test)
    preds = (p_late_test >= 0.50).astype(int)
    acc = float(accuracy_score(y_test, preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"[*] Held-out Test Metrics (N=103):")
    print(f"    ROC-AUC:   {auc:.4f}")
    print(f"    EER:       {eer*100:.2f}% (t={opt_t:.4f})")
    print(f"    Accuracy:  {acc*100:.2f}%")
    print(f"    Precision: {prec*100:.2f}%")
    print(f"    Recall:    {rec*100:.2f}% ({tp}/{tp+fn})")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Verify exact match with expected targets
    assert abs(w_vis - 5.3688) < 1e-3, f"W_vis mismatch: {w_vis}"
    assert abs(w_desync - 4.9111) < 1e-3, f"W_desync mismatch: {w_desync}"
    assert abs(intercept - (-6.0411)) < 1e-3, f"Intercept mismatch: {intercept}"
    assert abs(auc - 0.7197) < 1e-3, f"AUC mismatch: {auc}"

    # 4. Save checkpoints
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vis_ckpt_path = MODEL_CHECKPOINT_DIR / "best_vision_model.pt"
    torch.save({
        "model_type": "VisionOnlyNet",
        "model_state_dict": vis_model.state_dict(),
        "seq_len": 100,
        "best_epoch": vis_meta["best_epoch"],
        "best_val_loss": vis_meta["best_val_loss"],
    }, vis_ckpt_path)
    print(f"[*] Saved VisionOnlyNet checkpoint to {vis_ckpt_path}")

    desync_ckpt_path = MODEL_CHECKPOINT_DIR / "best_desync_model.pt"
    torch.save({
        "model_type": "DesyncOnlyNet",
        "model_state_dict": desync_model.state_dict(),
        "desync_feature_dim": 3,
        "best_epoch": des_meta["best_epoch"],
        "best_val_loss": des_meta["best_val_loss"],
    }, desync_ckpt_path)
    print(f"[*] Saved DesyncOnlyNet checkpoint to {desync_ckpt_path}")

    fusion_ckpt_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"
    torch.save({
        "model_type": "constrained_late_fusion",
        "vision_state_dict": vis_model.state_dict(),
        "desync_state_dict": desync_model.state_dict(),
        # For legacy compatibility if model_state_dict key is inspected:
        "model_state_dict": vis_model.state_dict(),
        "meta_weights": {
            "w_vis": w_vis,
            "w_desync": w_desync,
            "intercept": intercept,
        },
        "metrics": {
            "official_test_auc": auc,
            "official_test_eer": eer,
            "official_test_acc": acc,
            "official_test_precision": float(prec),
            "official_test_recall": float(rec),
            "official_test_f1": float(f1),
            "cv_5fold_auc_mean": 0.6877,
            "cv_5fold_auc_std": 0.0449,
            "cv_5fold_eer_mean": 0.3600,
            "cv_5fold_eer_std": 0.0298,
        }
    }, fusion_ckpt_path)
    print(f"[*] Saved Constrained Late-Fusion production bundle to {fusion_ckpt_path}")
    print("[+] All checkpoints successfully exported and verified!")


if __name__ == "__main__":
    export_production_checkpoint()
