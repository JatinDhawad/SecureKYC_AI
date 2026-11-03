"""
Diagnostic Script for Candidate 2 (Late Fusion) Verification
Item 1: EER Threshold & Operating Point Analysis (confirming distinct decision boundary)
Item 2: 5-Fold Cross-Validation Stability of Meta-Learner Weights and Test AUC
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
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


def main():
    device = get_device()
    print("=" * 85)
    print(f" CANDIDATE 2 DIAGNOSTICS: EER OPERATING POINT & 5-FOLD CV STABILITY ({device.type.upper()}) ")
    print("=" * 85)

    dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
    dataset_dict = torch.load(dataset_path, weights_only=False)

    # Deterministic split: 25 train / 9 val / 9 test identities
    train_dict, val_dict, test_dict = split_dataset_by_identity(dataset_dict, random_seed=42)

    # Datasets
    train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], train_dict["desync_features"], train_dict["labels"])
    val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], val_dict["desync_features"], val_dict["labels"])

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 1. Train primary base models on the official train split
    set_seed(42)
    print("\n[*] Training primary VisionOnlyNet (Bi-LSTM) and DesyncOnlyNet on official train split...")
    vis_model = VisionOnlyNet(seq_len=100).to(device)
    vis_model, _ = train_model(vis_model, tr_loader, va_loader, 20, device, is_vision_only=True)

    desync_model = DesyncOnlyNet(desync_feature_dim=3).to(device)
    desync_model, _ = train_model(desync_model, tr_loader, va_loader, 20, device, is_desync_only=True)

    # Predict on validation and test sets
    vis_model.eval()
    desync_model.eval()
    with torch.no_grad():
        p_vis_val = vis_model(val_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
        p_des_val = desync_model(desync_features=val_dict["desync_features"].to(device)).cpu().numpy().flatten()
        y_val = val_dict["labels"].cpu().numpy().flatten().astype(int)

        p_vis_te = vis_model(test_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
        p_des_te = desync_model(desync_features=test_dict["desync_features"].to(device)).cpu().numpy().flatten()
        y_te = test_dict["labels"].cpu().numpy().flatten().astype(int)

    # Official Candidate 2 Meta-Classifier fit on official validation split
    lr_official = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    lr_official.fit(np.column_stack([p_vis_val, p_des_val]), y_val)
    p_late_te = lr_official.predict_proba(np.column_stack([p_vis_te, p_des_te]))[:, 1]

    # =========================================================================
    # ITEM 1: EER Threshold and Curve Analysis
    # =========================================================================
    print("\n" + "=" * 85)
    print(" ITEM 1: EER THRESHOLD & OPERATING POINT ANALYSIS (TEST SET N=103: 57 Real, 46 Fake) ")
    print("=" * 85)

    eer_vis, opt_t_vis, th_vis, far_vis, frr_vis = compute_eer(y_te, p_vis_te)
    eer_late, opt_t_late, th_late, far_late, frr_late = compute_eer(y_te, p_late_te)

    print(f"• Standalone Spatial: EER = {eer_vis*100:.2f}% at optimal threshold t = {opt_t_vis:.4f}")
    print(f"• Candidate 2 Late:   EER = {eer_late*100:.2f}% at optimal threshold t = {opt_t_late:.4f}")

    # Inspect the exact curve near the optimal threshold
    idx_v = np.argmin(np.abs(far_vis - frr_vis))
    idx_l = np.argmin(np.abs(far_late - frr_late))

    print("\n--- Raw FAR / FRR Curve Near EER Operating Point (Standalone Spatial) ---")
    print("   Threshold  |  FAR (%)  (False Alarms)  |  FRR (%)  (Missed Fakes)  |  |FAR - FRR|")
    print("   " + "-" * 72)
    for off in range(-3, 4):
        i = idx_v + off
        t = th_vis[i]
        fa = int(round(far_vis[i] * 57))
        fr = int(round(frr_vis[i] * 46))
        marker = " <-- Optimal EER" if off == 0 else ""
        print(f"   t = {t:.4f}  |  {far_vis[i]*100:6.2f}% ({fa:2d}/57 real)     |  {frr_vis[i]*100:6.2f}% ({fr:2d}/46 fake)    |    {abs(far_vis[i]-frr_vis[i])*100:5.2f}%{marker}")

    print("\n--- Raw FAR / FRR Curve Near EER Operating Point (Late Fusion) ---")
    print("   Threshold  |  FAR (%)  (False Alarms)  |  FRR (%)  (Missed Fakes)  |  |FAR - FRR|")
    print("   " + "-" * 72)
    for off in range(-3, 4):
        i = idx_l + off
        t = th_late[i]
        fa = int(round(far_late[i] * 57))
        fr = int(round(frr_late[i] * 46))
        marker = " <-- Optimal EER" if off == 0 else ""
        print(f"   t = {t:.4f}  |  {far_late[i]*100:6.2f}% ({fa:2d}/57 real)     |  {frr_late[i]*100:6.2f}% ({fr:2d}/46 fake)    |    {abs(far_late[i]-frr_late[i])*100:5.2f}%{marker}")

    # Decision boundary analysis
    preds_vis_eer = (p_vis_te >= opt_t_vis).astype(int)
    preds_late_eer = (p_late_te >= opt_t_late).astype(int)
    diff_mask_eer = (preds_vis_eer != preds_late_eer)
    diff_count_eer = int(np.sum(diff_mask_eer))

    print(f"\n• Sample-Level Classification Differences at EER Thresholds:")
    print(f"  Out of 103 test clips, {diff_count_eer} clips ({diff_count_eer/103*100:.1f}%) are classified differently:")
    for idx in np.where(diff_mask_eer)[0]:
        actual = "Fake (1)" if y_te[idx] == 1 else "Real (0)"
        p_v = p_vis_te[idx]
        p_l = p_late_te[idx]
        p_d = p_des_te[idx]
        print(f"    - Clip #{idx:2d} (Actual: {actual}): Spatial P={p_v:.4f} (pred {preds_vis_eer[idx]}), Desync P={p_d:.4f} -> Late P={p_l:.4f} (pred {preds_late_eer[idx]})")

    # At default decision threshold 0.50
    preds_vis_50 = (p_vis_te >= 0.50).astype(int)
    preds_late_50 = (p_late_te >= 0.50).astype(int)
    diff_mask_50 = (preds_vis_50 != preds_late_50)
    diff_count_50 = int(np.sum(diff_mask_50))

    print(f"\n• Sample-Level Classification Differences at Default Decision Threshold (t = 0.50):")
    print(f"  Out of 103 test clips, {diff_count_50} clips ({diff_count_50/103*100:.1f}%) are classified differently.")
    print(f"  Spatial at t=0.50: TN={int((preds_vis_50==0)[y_te==0].sum())}, FP={int((preds_vis_50==1)[y_te==0].sum())}, FN={int((preds_vis_50==0)[y_te==1].sum())}, TP={int((preds_vis_50==1)[y_te==1].sum())} -> Recall: {(preds_vis_50==1)[y_te==1].mean()*100:.1f}%")
    print(f"  Late    at t=0.50: TN={int((preds_late_50==0)[y_te==0].sum())}, FP={int((preds_late_50==1)[y_te==0].sum())}, FN={int((preds_late_50==0)[y_te==1].sum())}, TP={int((preds_late_50==1)[y_te==1].sum())} -> Recall: {(preds_late_50==1)[y_te==1].mean()*100:.1f}%")

    # =========================================================================
    # ITEM 2: 5-Fold Cross-Validation on Pooled Train+Val Identities
    # =========================================================================
    print("\n" + "=" * 85)
    print(" ITEM 2: 5-FOLD CROSS-VALIDATION STABILITY OF META-LEARNER (TEST SET HELD OUT) ")
    print("=" * 85)

    test_ids = set(test_dict["identities"])
    all_lip = dataset_dict["lip_sequences"]
    all_desync = dataset_dict["desync_features"]
    all_labels = dataset_dict["labels"]
    all_ids = dataset_dict["identities"]

    # Pool train + val identities (34 unique identities)
    train_val_mask = np.array([spk not in test_ids for spk in all_ids])
    pool_lip = all_lip[train_val_mask]
    pool_desync = all_desync[train_val_mask]
    pool_labels = all_labels[train_val_mask]
    pool_ids = [spk for spk, m in zip(all_ids, train_val_mask) if m]
    unique_pool_ids = sorted(list(set(pool_ids)))

    print(f"[*] Pooled train+val corpus: {len(unique_pool_ids)} identities, {len(pool_labels)} clips.")
    print(f"[*] Strictly held-out test split: {len(test_dict['labels'])} clips across 9 identities.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    print("\nExecuting 5-Fold Cross-Validation (training base models on 4/5, fitting meta-learner on 1/5 out-of-fold val, evaluating on fixed test):")
    print("Fold | Val IDs | Val N |  W_vis  | W_desync | Intercept | Spatial AUC | Late AUC |  Delta AUC  | EER (%)")
    print("-" * 105)

    for fold, (train_id_idx, val_id_idx) in enumerate(kf.split(unique_pool_ids), 1):
        set_seed(42 + fold)
        tr_id_set = set([unique_pool_ids[i] for i in train_id_idx])
        va_id_set = set([unique_pool_ids[i] for i in val_id_idx])

        tr_mask = np.array([spk in tr_id_set for spk in pool_ids])
        va_mask = np.array([spk in va_id_set for spk in pool_ids])

        # Convert labels to 2D float tensor (N, 1) to match BCELoss
        tr_y = pool_labels[tr_mask].view(-1, 1).float()
        va_y = pool_labels[va_mask].view(-1, 1).float()

        tr_ds = MultimodalKYCDataset(pool_lip[tr_mask], pool_desync[tr_mask], tr_y)
        va_ds = MultimodalKYCDataset(pool_lip[va_mask], pool_desync[va_mask], va_y)

        tr_loader_f = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_loader_f = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Train fold base models
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

        # Fit meta-learner on out-of-fold validation set
        X_v = np.column_stack([p_vis_v, p_des_v])
        X_t = np.column_stack([p_vis_t, p_des_t])

        lr_f = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
        lr_f.fit(X_v, y_v)

        p_late_t = lr_f.predict_proba(X_t)[:, 1]

        auc_spat = float(roc_auc_score(y_te, p_vis_t))
        auc_late = float(roc_auc_score(y_te, p_late_t))
        eer_f, _, _, _, _ = compute_eer(y_te, p_late_t)
        w_v, w_d = lr_f.coef_[0]
        b_f = lr_f.intercept_[0]

        fold_results.append({
            "fold": fold,
            "val_ids": len(va_id_set),
            "val_n": len(y_v),
            "w_vis": float(w_v),
            "w_desync": float(w_d),
            "intercept": float(b_f),
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
        f"Mean |         |       | {np.mean(w_v_all):7.4f} | {np.mean(w_d_all):8.4f} |  {np.mean(b_all):8.4f} |   {np.mean(auc_s_all):.4f}    |  {np.mean(auc_l_all):.4f}  |   {np.mean(auc_l_all) - np.mean(auc_s_all):+7.4f}   | {np.mean(eer_all)*100:6.2f}%"
    )
    print(
        f"Std  |         |       | {np.std(w_v_all):7.4f} | {np.std(w_d_all):8.4f} |  {np.std(b_all):8.4f} |   {np.std(auc_s_all):.4f}    |  {np.std(auc_l_all):.4f}  |   {np.std([r['delta_auc'] for r in fold_results]):+7.4f}   | {np.std(eer_all)*100:6.2f}%"
    )


if __name__ == "__main__":
    main()
