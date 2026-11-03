"""
SecureKYC AI - Fusion Architecture Experimentation Engine
Tests Candidate 1 (Standardized Early Fusion) and Candidate 2 (Late Fusion)
against the current baselines (Fused AUC=0.6423, Standalone-Spatial AUC=0.7105)
on the exact same identity-disjoint test split (N=103).
"""

import sys
import random
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    get_device,
)
from src.models.fusion_net import MultimodalKYCFusionNet, VisionOnlyNet, DesyncOnlyNet
from src.models.train import split_dataset_by_identity, MultimodalKYCDataset
from src.models.evaluate import compute_eer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_cm(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    acc = (tn + tp) / max(1, total)
    lines = [
        f"\n  CONFUSION MATRIX: {name} (N={total})",
        f"  {'=' * 58}",
        f"                   Pred Real (0)   Pred Fake (1)   Total",
        f"  Actual Real (0)      {tn:5d}           {fp:5d}        {tn+fp:5d}",
        f"  Actual Fake (1)      {fn:5d}           {tp:5d}        {fn+tp:5d}",
        f"  Total                {tn+fn:5d}           {fp+tp:5d}        {total:5d}",
        f"  {'-' * 58}",
        f"  TN: {tn:3d} | FP: {fp:3d} | FN: {fn:3d} | TP: {tp:3d} | Accuracy: {acc*100:.2f}%",
    ]
    return "\n".join(lines)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    is_vision_only: bool = False,
    is_desync_only: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains a model with validation checkpointing on val_loss."""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        for lip_seq, desync_vec, label in train_loader:
            lip_seq = lip_seq.to(device)
            desync_vec = desync_vec.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            if is_vision_only:
                preds = model(lip_seq)
            elif is_desync_only:
                preds = model(desync_features=desync_vec)
            else:
                preds = model(lip_seq, desync_vec)

            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

        # Validation pass
        model.eval()
        total_val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for lip_seq, desync_vec, label in val_loader:
                lip_seq = lip_seq.to(device)
                desync_vec = desync_vec.to(device)
                label = label.to(device)

                if is_vision_only:
                    preds = model(lip_seq)
                elif is_desync_only:
                    preds = model(desync_features=desync_vec)
                else:
                    preds = model(lip_seq, desync_vec)

                loss = criterion(preds, label)
                total_val_loss += loss.item() * len(label)
                val_total += len(label)

        val_loss = total_val_loss / max(1, val_total)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return model, {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    binary_preds = (scores >= threshold).astype(int)
    acc = float(accuracy_score(y_true, binary_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, binary_preds, average="binary", zero_division=0)
    roc_auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5
    eer, opt_thresh, _, _, _ = compute_eer(y_true, scores)
    cm = confusion_matrix(y_true, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "eer": eer,
        "opt_thresh": opt_thresh,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cm": cm,
        "scores": scores,
    }


def run_candidate_1_standardized_fusion(
    train_dict: Dict[str, torch.Tensor],
    val_dict: Dict[str, torch.Tensor],
    test_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Candidate 1: Standardized Early Fusion
    z-scores S_desync and max_penalty using train-split statistics before entering fusion_fc.
    leaves failed_ratio as-is on [0, 1].
    Retrains MultimodalKYCFusionNet from scratch.
    """
    set_seed(RANDOM_SEED)

    # 1. Compute train-split statistics
    tr_desync = train_dict["desync_features"].clone()
    s_mean = float(tr_desync[:, 0].mean())
    s_std = float(tr_desync[:, 0].std()) + 1e-8
    m_mean = float(tr_desync[:, 1].mean())
    m_std = float(tr_desync[:, 1].std()) + 1e-8

    print(f"\n--- Candidate 1: Feature Standardization Stats (Train Split) ---")
    print(f"  S_desync:    mean = {s_mean:.6f}, std = {s_std:.6f}")
    print(f"  max_penalty: mean = {m_mean:.6f}, std = {m_std:.6f}")
    print(f"  failed_ratio: unmodified [0, 1] range")

    def standardize(desync_t: torch.Tensor) -> torch.Tensor:
        out = desync_t.clone()
        out[:, 0] = (out[:, 0] - s_mean) / s_std
        out[:, 1] = (out[:, 1] - m_mean) / m_std
        return out

    tr_desync_std = standardize(train_dict["desync_features"])
    va_desync_std = standardize(val_dict["desync_features"])
    te_desync_std = standardize(test_dict["desync_features"])

    train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], tr_desync_std, train_dict["labels"])
    val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], va_desync_std, val_dict["labels"])
    test_ds = MultimodalKYCDataset(test_dict["lip_sequences"], te_desync_std, test_dict["labels"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3).to(device)
    model, info = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Evaluate on test split
    model.eval()
    with torch.no_grad():
        preds = model(test_dict["lip_sequences"].to(device), te_desync_std.to(device))
        scores = preds.cpu().numpy().flatten()

    y_test = test_dict["labels"].cpu().numpy().flatten().astype(int)
    metrics = compute_metrics(y_test, scores)
    metrics["best_epoch"] = info["best_epoch"]
    metrics["best_val_loss"] = info["best_val_loss"]
    return metrics


def run_candidate_2_late_fusion(
    train_dict: Dict[str, torch.Tensor],
    val_dict: Dict[str, torch.Tensor],
    test_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Candidate 2: Late Fusion
    Trains visual branch (VisionOnlyNet Bi-LSTM) and desync branch (DesyncOnlyNet) independently.
    Combines output probabilities via Logistic Regression fit on the validation split.
    """
    set_seed(RANDOM_SEED)

    train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], train_dict["desync_features"], train_dict["labels"])
    val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], val_dict["desync_features"], val_dict["labels"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n--- Candidate 2: Training Independent Branches ---")

    # 1. Train Visual Branch (Bi-LSTM)
    print("  [2a] Training Visual Branch (VisionOnlyNet)...")
    set_seed(RANDOM_SEED)
    vis_model = VisionOnlyNet(seq_len=100).to(device)
    vis_model, vis_info = train_model(
        model=vis_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        is_vision_only=True,
    )

    # 2. Train Desync Branch (DesyncOnlyNet)
    print("  [2b] Training Desync Branch (DesyncOnlyNet)...")
    set_seed(RANDOM_SEED)
    desync_model = DesyncOnlyNet(desync_feature_dim=3).to(device)
    desync_model, desync_info = train_model(
        model=desync_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        is_desync_only=True,
    )

    # 3. Extract predicted probabilities on Val Split
    vis_model.eval()
    desync_model.eval()
    with torch.no_grad():
        p_vis_val = vis_model(val_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
        p_desync_val = desync_model(desync_features=val_dict["desync_features"].to(device)).cpu().numpy().flatten()
        y_val = val_dict["labels"].cpu().numpy().flatten().astype(int)

        p_vis_test = vis_model(test_dict["lip_sequences"].to(device)).cpu().numpy().flatten()
        p_desync_test = desync_model(desync_features=test_dict["desync_features"].to(device)).cpu().numpy().flatten()
        y_test = test_dict["labels"].cpu().numpy().flatten().astype(int)

    # Evaluate individual branch performance on test split for direct reference
    vis_metrics = compute_metrics(y_test, p_vis_test)
    desync_metrics = compute_metrics(y_test, p_desync_test)

    # 4. Fit Logistic Regression meta-classifier on validation split
    X_val = np.column_stack([p_vis_val, p_desync_val])
    X_test = np.column_stack([p_vis_test, p_desync_test])

    lr_meta = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
    lr_meta.fit(X_val, y_val)

    p_late_test = lr_meta.predict_proba(X_test)[:, 1]
    late_metrics = compute_metrics(y_test, p_late_test)
    late_metrics["lr_weights"] = lr_meta.coef_[0].tolist()
    late_metrics["lr_intercept"] = float(lr_meta.intercept_[0])

    print(f"  [2c] Late Fusion Logistic Regression fit on Val (N={len(y_val)}):")
    print(f"       Weights: [W_vis={late_metrics['lr_weights'][0]:.4f}, W_desync={late_metrics['lr_weights'][1]:.4f}], Intercept={late_metrics['lr_intercept']:.4f}")

    return late_metrics, vis_metrics, desync_metrics


def main():
    device = get_device()
    print("=" * 80)
    print(f" FUSION ARCHITECTURE EXPERIMENTATION — BENCHMARK ({device.type.upper()}) ")
    print("=" * 80)

    dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
    dataset_dict = torch.load(dataset_path, weights_only=False)
    train_dict, val_dict, test_dict = split_dataset_by_identity(dataset_dict, random_seed=42)

    y_test = test_dict["labels"].cpu().numpy().flatten().astype(int)
    print(f"Test Split: N={len(y_test)} (Real: {(y_test==0).sum()}, Fake: {(y_test==1).sum()})")

    # Baseline numbers from results.md / N=691 benchmark
    baseline_fused = {
        "accuracy": 0.6505,
        "precision": 0.6786,
        "recall": 0.4130,
        "f1": 0.5135,
        "roc_auc": 0.6423,
        "eer": 0.4367,
        "tn": 48,
        "fp": 9,
        "fn": 27,
        "tp": 19,
    }
    baseline_spatial = {
        "accuracy": 0.6408,
        "precision": 0.7647,
        "recall": 0.2826,
        "f1": 0.4127,
        "roc_auc": 0.7105,
        "eer": 0.3494,
        "tn": 53,
        "fp": 4,
        "fn": 33,
        "tp": 13,
    }

    # Run Candidate 1
    c1_metrics = run_candidate_1_standardized_fusion(train_dict, val_dict, test_dict, device)

    # Run Candidate 2
    c2_metrics, vis_metrics, desync_metrics = run_candidate_2_late_fusion(train_dict, val_dict, test_dict, device)

    # Print Detailed Confusion Matrices
    print(format_cm(y_test, (c1_metrics["scores"] >= 0.5).astype(int), "Candidate 1: Standardized Early Fusion"))
    print(format_cm(y_test, (c2_metrics["scores"] >= 0.5).astype(int), "Candidate 2: Late Fusion (Logistic Regression)"))

    # Print Side-by-Side Comparison Table
    print("\n" + "=" * 105)
    print(f"{'FUSION ARCHITECTURE CANDIDATES vs. BASELINE (HELD-OUT TEST SPLIT, N=103)':^105}")
    print("=" * 105)
    header = f"{'Architecture Variant':<40} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^9} | {'ROC-AUC':^9} | {'EER (%)':^9}"
    print(header)
    print("-" * 105)

    def print_row(name: str, m: Dict[str, Any]):
        print(
            f"{name:<40} | "
            f"{m['accuracy'] * 100:^9.2f}% | "
            f"{m['precision'] * 100:^9.2f}% | "
            f"{m['recall'] * 100:^9.2f}% | "
            f"{m['f1']:^9.4f} | "
            f"{m['roc_auc']:^9.4f} | "
            f"{m['eer'] * 100:^9.2f}%"
        )

    print_row("Current Baseline: Standalone Spatial", baseline_spatial)
    print_row("Current Baseline: Fused (Raw Concat)", baseline_fused)
    print("-" * 105)
    print_row("Candidate 1: Standardized Early Fusion", c1_metrics)
    print_row("Candidate 2: Late Fusion (Logistic Reg)", c2_metrics)
    print("-" * 105)
    print_row("Ref: Independent Visual Branch", vis_metrics)
    print_row("Ref: Independent Desync Branch", desync_metrics)
    print("=" * 105)

    # Comparison verdict
    print("\n--- Summary & Benchmark Against Standalone-Spatial (0.7105 AUC) ---")
    for name, m in [("Candidate 1 (Standardized Early)", c1_metrics), ("Candidate 2 (Late Fusion)", c2_metrics)]:
        diff = m["roc_auc"] - baseline_spatial["roc_auc"]
        beats = diff > 0
        status = "BEATS" if beats else "UNDERPERFORMS"
        print(f"  • {name:<36}: AUC = {m['roc_auc']:.4f} (Delta vs. Standalone Spatial: {diff:+.4f}) -> {status}")


if __name__ == "__main__":
    main()
