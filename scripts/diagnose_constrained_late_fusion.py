"""
SecureKYC AI - Constrained Late-Fusion Meta-Learner Benchmark
Fits meta-learner with non-negativity bounds: W_vis >= 0, W_desync >= 0.
Evaluates 5-fold CV stability across pooled train+val identities (with test strictly held out),
and refits on the official train/val split.
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, get_device
from src.models.fusion_net import VisionOnlyNet, DesyncOnlyNet
from src.models.train import split_dataset_by_identity, MultimodalKYCDataset
from src.models.evaluate import compute_eer
from scripts.experiment_fusion_variants import train_model, set_seed


def fit_constrained_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    l2_reg: float = 1e-4,
) -> Tuple[float, float, float]:
    """
    Fits logistic regression with non-negativity constraints:
      W_vis >= 0.0, W_desync >= 0.0, intercept unconstrained.
    Uses L-BFGS-B with exact analytical BCE gradient and L2 regularization.
    """
    # X shape: (N, 2), y shape: (N,)
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


def predict_constrained_lr(X: np.ndarray, w_vis: float, w_desync: float, b: float) -> np.ndarray:
    z = w_vis * X[:, 0] + w_desync * X[:, 1] + b
    return expit(z)


def main():
    device = get_device()
    print("=" * 85)
    print(f" CONSTRAINED LATE-FUSION BENCHMARK (W_vis >= 0, W_desync >= 0) on {device.type.upper()} ")
    print("=" * 85)

    dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
    dataset_dict = torch.load(dataset_path, weights_only=False)

    # 1. Official Dataset Partition (25 train / 9 val / 9 test identities)
    train_dict, val_dict, test_dict = split_dataset_by_identity(dataset_dict, random_seed=42)
    test_ids = set(test_dict["identities"])
    y_test = test_dict["labels"].cpu().numpy().flatten().astype(int)

    all_lip = dataset_dict["lip_sequences"]
    all_desync = dataset_dict["desync_features"]
    all_labels = dataset_dict["labels"]
    all_ids = dataset_dict["identities"]

    # 2. Pooled Train+Val identities for 5-Fold Cross-Validation (34 identities, 588 clips)
    train_val_mask = np.array([spk not in test_ids for spk in all_ids])
    pool_lip = all_lip[train_val_mask]
    pool_desync = all_desync[train_val_mask]
    pool_labels = all_labels[train_val_mask]
    pool_ids = [spk for spk, m in zip(all_ids, train_val_mask) if m]
    unique_pool_ids = sorted(list(set(pool_ids)))

    print(f"[*] Pooled train+val: {len(unique_pool_ids)} identities, {len(pool_labels)} clips.")
    print(f"[*] Strictly held-out test split: {len(y_test)} clips (57 Real, 46 Fake).")

    # =========================================================================
    # PART 1: 5-Fold Cross-Validation with Non-Negativity Constraints
    # =========================================================================
    print("\n" + "=" * 85)
    print(" PART 1: 5-FOLD CROSS-VALIDATION WITH NON-NEGATIVITY CONSTRAINTS (TEST HELD OUT) ")
    print("=" * 85)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    print("Fold | Val IDs | Val N |  W_vis  | W_desync | Intercept | Spatial AUC | Late AUC | Delta vs. Spatial |  EER (%)")
    print("-" * 105)

    for fold, (train_id_idx, val_id_idx) in enumerate(kf.split(unique_pool_ids), 1):
        set_seed(42 + fold)
        tr_id_set = set([unique_pool_ids[i] for i in train_id_idx])
        va_id_set = set([unique_pool_ids[i] for i in val_id_idx])

        tr_mask = np.array([spk in tr_id_set for spk in pool_ids])
        va_mask = np.array([spk in va_id_set for spk in pool_ids])

        tr_y = pool_labels[tr_mask].view(-1, 1).float()
        va_y = pool_labels[va_mask].view(-1, 1).float()

        tr_ds = MultimodalKYCDataset(pool_lip[tr_mask], pool_desync[tr_mask], tr_y)
        va_ds = MultimodalKYCDataset(pool_lip[va_mask], pool_desync[va_mask], va_y)

        tr_loader_f = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader_f = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

        vis_f = VisionOnlyNet(seq_len=100).to(device)
        vis_f, _ = train_model(vis_f, tr_loader_f, va_loader_f, 20, device, is_vision_only=True)

        des_f = DesyncOnlyNet(desync_feature_dim=3).to(device)
        des_f, _ = train_model(des_f, tr_loader_f, va_loader_f, 20, device, is_desync_only=True)

        vis_f.eval()
        des_f.eval()
        with torch.no_grad():
            p_vis_v = vis_f(pool_lip[va_mask].to(device)).cpu().numpy().flatten()
            p_des_v = des_f(desync_features=pool_desync[va_mask].to(device)).cpu().numpy().flatten()
            y_v = pool_labels[va_mask].cpu().numpy().flatten().astype(int)

            p_vis_t = vis_f(test_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
            p_des_t = des_f(desync_features=test_dict["desync_features"].to(device)).cpu().numpy().flatten()

        X_v = np.column_stack([p_vis_v, p_des_v])
        X_t = np.column_stack([p_vis_t, p_des_t])

        # Constrained fit: W_vis >= 0, W_desync >= 0
        w_v, w_d, b_f = fit_constrained_logistic_regression(X_v, y_v)
        p_late_t = predict_constrained_lr(X_t, w_v, w_d, b_f)

        auc_spat = float(roc_auc_score(y_test, p_vis_t))
        auc_late = float(roc_auc_score(y_test, p_late_t))
        eer_f, _, _, _, _ = compute_eer(y_test, p_late_t)

        fold_results.append({
            "fold": fold,
            "val_ids": len(va_id_set),
            "val_n": len(y_v),
            "w_vis": w_v,
            "w_desync": w_d,
            "intercept": b_f,
            "spatial_auc": auc_spat,
            "late_auc": auc_late,
            "delta_auc": auc_late - auc_spat,
            "eer": eer_f,
        })

        print(
            f"  {fold:2d} |   {len(va_id_set):2d}    |  {len(y_v):3d}  | {w_v:7.4f} | {w_d:8.4f} |  {b_f:8.4f} |   {auc_spat:.4f}    |  {auc_late:.4f}  |   {auc_late - auc_spat:+7.4f}   | {eer_f*100:6.2f}%"
        )

    w_v_all = [r["w_vis"] for r in fold_results]
    w_d_all = [r["w_desync"] for r in fold_results]
    b_all = [r["intercept"] for r in fold_results]
    auc_l_all = [r["late_auc"] for r in fold_results]
    auc_s_all = [r["spatial_auc"] for r in fold_results]
    eer_all = [r["eer"] for r in fold_results]

    print("-" * 105)
    print(
        f"Mean |         |       | {np.mean(w_v_all):7.4f} | {np.mean(w_d_all):8.4f} |  {np.mean(b_all):8.4f} |   {np.mean(auc_s_all):.4f}    |  {np.mean(auc_l_all):.4f}  |   {np.mean([r['delta_auc'] for r in fold_results]):+7.4f}   | {np.mean(eer_all)*100:6.2f}%"
    )
    print(
        f"Std  |         |       | {np.std(w_v_all):7.4f} | {np.std(w_d_all):8.4f} |  {np.std(b_all):8.4f} |   {np.std(auc_s_all):.4f}    |  {np.std(auc_l_all):.4f}  |   {np.std([r['delta_auc'] for r in fold_results]):+7.4f}   | {np.std(eer_all)*100:6.2f}%"
    )

    # =========================================================================
    # PART 2: Refit Final Meta-Learner on Official Train/Val Split (with constraint)
    # =========================================================================
    print("\n" + "=" * 85)
    print(" PART 2: CONSTRAINED META-LEARNER FIT ON OFFICIAL TRAIN+VAL SPLIT ")
    print("=" * 85)

    set_seed(42)
    train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], train_dict["desync_features"], train_dict["labels"])
    val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], val_dict["desync_features"], val_dict["labels"])

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    vis_model = VisionOnlyNet(seq_len=100).to(device)
    vis_model, _ = train_model(vis_model, tr_loader, va_loader, 20, device, is_vision_only=True)

    desync_model = DesyncOnlyNet(desync_feature_dim=3).to(device)
    desync_model, _ = train_model(desync_model, tr_loader, va_loader, 20, device, is_desync_only=True)

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

    # Fit with constraint
    w_vis_final, w_des_final, b_final = fit_constrained_logistic_regression(X_val, y_val)
    p_late_final = predict_constrained_lr(X_test, w_vis_final, w_des_final, b_final)

    auc_final = float(roc_auc_score(y_test, p_late_final))
    eer_final, opt_t_final, _, _, _ = compute_eer(y_test, p_late_final)
    bin_preds = (p_late_final >= 0.50).astype(int)
    acc_final = float(accuracy_score(y_test, bin_preds))
    prec_final, rec_final, f1_final, _ = precision_recall_fscore_support(y_test, bin_preds, average="binary", zero_division=0)
    cm_final = confusion_matrix(y_test, bin_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_final.ravel()

    print(f"• Constrained Meta-Learner Parameters (Official Val Split N=104):")
    print(f"    W_vis:     {w_vis_final:7.4f}")
    print(f"    W_desync:  {w_des_final:7.4f}")
    print(f"    Intercept: {b_final:7.4f}")
    print(f"\n• Test Set Metrics (N=103):")
    print(f"    ROC-AUC:   {auc_final:.4f}")
    print(f"    EER:       {eer_final*100:.2f}% (at threshold t = {opt_t_final:.4f})")
    print(f"    Accuracy:  {acc_final*100:.2f}% (at threshold t = 0.50)")
    print(f"    Precision: {prec_final*100:.2f}%")
    print(f"    Recall:    {rec_final*100:.2f}% ({tp}/{tp+fn} fakes)")
    print(f"    F1-Score:  {f1_final:.4f}")
    print(f"    Confusion Matrix at t=0.50: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # =========================================================================
    # PART 3: Full Comparison Table
    # =========================================================================
    print("\n" + "=" * 95)
    print(f"{'FULL COMPARISON: CONSTRAINED LATE FUSION vs. BASELINES':^95}")
    print("=" * 95)
    print(f"{'Model Configuration':<42} | {'ROC-AUC':^9} | {'EER (%)':^9} | {'Accuracy':^10} | {'Recall':^10} | {'F1-Score':^9}")
    print("-" * 95)
    print(f"{'Current Baseline: Standalone Spatial':<42} | {'0.7105':^9} | {'34.94%':^9} | {'64.08%':^10} | {'28.26%':^10} | {'0.4127':^9}")
    print(f"{'Current Baseline: Fused (Raw Concat)':<42} | {'0.6423':^9} | {'43.67%':^9} | {'65.05%':^10} | {'41.30%':^10} | {'0.5135':^9}")
    print("-" * 95)
    print(f"{'Candidate 2: Unconstrained Late Fusion':<42} | {'0.7159':^9} | {'34.94%':^9} | {'67.96%':^10} | {'47.83%':^10} | {'0.5714':^9}")
    print(f"{'Candidate 2: Constrained Late Fusion (Official)':<42} | {auc_final:^9.4f} | {eer_final*100:^8.2f}% | {acc_final*100:^9.2f}% | {rec_final*100:^9.2f}% | {f1_final:^9.4f}")
    print(f"{'Candidate 2: Constrained Late Fusion (5-Fold Mean)':<42} | {np.mean(auc_l_all):^9.4f} | {np.mean(eer_all)*100:^8.2f}% | {'—':^10} | {'—':^10} | {'—':^9}")
    print("=" * 95)


if __name__ == "__main__":
    main()
