"""
SecureKYC AI - Complete Retraining, Empirical Diagnostics & 3-Way Ablation Benchmark
Retrains checkpoints/best_fusion_model.pt from scratch on data/processed/dataset.pt (N=691)
using Apple Silicon MPS acceleration and the verified 25 train / 9 val / 9 test identity split.
"""

import sys
import csv
import random
from pathlib import Path
from typing import Dict, Tuple, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA_DIR,
    MODEL_CHECKPOINT_DIR,
    PAPER_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    get_device,
)
from src.models.fusion_net import MultimodalKYCFusionNet, VisionOnlyNet, DesyncOnlyNet
from src.models.train import split_dataset_by_identity, MultimodalKYCDataset
from src.models.evaluate import compute_eer, save_evaluation_plots


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    acc = (tn + tp) / max(1, total)

    lines = [
        f"\n{'=' * 65}",
        f"  CONFUSION MATRIX — {split_name} (N={total})",
        f"{'=' * 65}",
        f"                   Predicted Real (0)    Predicted Fake (1)    Total",
        f"  Actual Real (0)        {tn:5d}                 {fp:5d}          {tn+fp:5d}",
        f"  Actual Fake (1)        {fn:5d}                 {tp:5d}          {fn+tp:5d}",
        f"  Total                  {tn+fn:5d}                 {fp+tp:5d}          {total:5d}",
        f"{'-' * 65}",
        f"  True Negatives (TN) : {tn:4d}  |  False Positives (FP): {fp:4d}",
        f"  False Negatives (FN): {fn:4d}  |  True Positives (TP) : {tp:4d}",
        f"  Accuracy            : {acc * 100:.2f}%",
    ]
    return "\n".join(lines)


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: Path = None,
    log_csv_path: Path = None,
    is_vision_only: bool = False,
    is_desync_only: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains a model with validation checkpointing and metric logging."""
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    if log_csv_path:
        with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc"])

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

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

            bs = len(label)
            total_train_loss += loss.item() * bs
            bin_preds = (preds >= 0.5).float()
            correct_train += (bin_preds == label).sum().item()
            total_train += bs

        train_loss = total_train_loss / max(1, total_train)
        train_acc = correct_train / max(1, total_train)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds_list = []
        val_labels_list = []

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
                val_preds_list.extend(preds.cpu().numpy().flatten())
                val_labels_list.extend(label.cpu().numpy().flatten())

        val_total = len(val_labels_list)
        val_loss = total_val_loss / max(1, val_total)
        val_scores = np.array(val_preds_list)
        val_labels = np.array(val_labels_list)
        val_acc = float(np.mean((val_scores >= 0.5).astype(int) == val_labels))
        val_auc = float(roc_auc_score(val_labels, val_scores)) if len(np.unique(val_labels)) > 1 else 0.5

        if log_csv_path:
            with open(log_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, round(train_loss, 4), round(train_acc, 4), round(val_loss, 4), round(val_acc, 4), round(val_auc, 4)])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"    Epoch [{epoch:02d}/{num_epochs:02d}] -> Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, AUC: {val_auc:.4f}")

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    if save_path and best_state is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "val_loss": best_val_loss,
            "num_epochs": num_epochs,
        }, save_path)
        print(f"    [✓] Best model checkpoint saved to: {save_path} (epoch {best_epoch}, val_loss={best_val_loss:.4f})")

    return model, {"best_epoch": best_epoch, "best_val_loss": best_val_loss}


def evaluate_split_performance(
    model: nn.Module,
    split_dict: Dict[str, Any],
    device: torch.device,
    split_name: str,
    is_vision_only: bool = False,
    is_desync_only: bool = False,
) -> Dict[str, Any]:
    """Computes full classification metrics for a given split."""
    model.eval()
    lip_seqs = split_dict["lip_sequences"].to(device)
    desync_feats = split_dict["desync_features"].to(device)
    labels = split_dict["labels"].cpu().numpy().flatten().astype(int)

    with torch.no_grad():
        if is_vision_only:
            preds = model(lip_seqs)
        elif is_desync_only:
            preds = model(desync_features=desync_feats)
        else:
            preds = model(lip_seqs, desync_feats)
        scores = preds.cpu().numpy().flatten()

    binary_preds = (scores >= 0.5).astype(int)
    acc = float(accuracy_score(labels, binary_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(labels, binary_preds, average="binary", zero_division=0)
    roc_auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.5
    eer, opt_thresh, _, far_arr, frr_arr = compute_eer(labels, scores)

    cm = confusion_matrix(labels, binary_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "split_name": split_name,
        "n_samples": len(labels),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "eer": eer,
        "opt_thresh": opt_thresh,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "scores": scores,
        "labels": labels,
    }


def analyze_decision_boundary(eval_res: Dict[str, Any], split_name: str):
    """Analyzes whether predictions collapse to majority-class or form a real boundary."""
    labels = eval_res["labels"]
    scores = eval_res["scores"]
    preds = (scores >= 0.5).astype(int)

    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    pred_real_count = int((preds == 0).sum())
    pred_fake_count = int((preds == 1).sum())
    maj_label = int(eval_res["n_fake"] > eval_res["n_real"])
    maj_rate = max(eval_res["n_real"], eval_res["n_fake"]) / len(labels)

    collapsed = (pred_real_count == 0) or (pred_fake_count == 0)

    print(f"\n{'=' * 65}")
    print(f"  DECISION BOUNDARY & COLLAPSE ANALYSIS — {split_name}")
    print(f"{'=' * 65}")
    print(f"  Ground Truth: Real={eval_res['n_real']} ({eval_res['n_real']/len(labels)*100:.1f}%), Fake={eval_res['n_fake']} ({eval_res['n_fake']/len(labels)*100:.1f}%)")
    print(f"  Predictions : Real={pred_real_count} ({pred_real_count/len(labels)*100:.1f}%), Fake={pred_fake_count} ({pred_fake_count/len(labels)*100:.1f}%)")
    print(f"  Majority-Class Baseline: {maj_rate*100:.1f}% (Always predict {'Fake (1)' if maj_label==1 else 'Real (0)'})")
    print(f"  Model Accuracy         : {eval_res['accuracy']*100:.2f}%")
    print(f"  Collapse Status        : {'**COLLAPSED TO MAJORITY CLASS**' if collapsed else '✓ NON-COLLAPSED (Learns real two-sided decision boundary)'}")
    print(f"\n  Prediction Distribution Details:")
    print(f"    Real Clips P(Fake) -> Min: {real_scores.min():.4f}, Mean: {real_scores.mean():.4f}, Median: {np.median(real_scores):.4f}, Max: {real_scores.max():.4f}")
    print(f"    Fake Clips P(Fake) -> Min: {fake_scores.min():.4f}, Mean: {fake_scores.mean():.4f}, Median: {np.median(fake_scores):.4f}, Max: {fake_scores.max():.4f}")
    print(f"    Separation (Mean Fake - Mean Real): {fake_scores.mean() - real_scores.mean():+.4f}")


def run_branch_use_sanity_check(
    model: nn.Module,
    test_dict: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Sweeps synthetic desync magnitude while holding visual sequence constant,
    testing whether the fusion model's output responds to the desync branch.
    """
    model.eval()
    lip_seqs = test_dict["lip_sequences"].to(device)
    desync_feats = test_dict["desync_features"].to(device)
    labels = test_dict["labels"].cpu().numpy().flatten().astype(int)

    # 1. Select a genuine authentic test sample (real label, baseline desync=0)
    real_indices = np.where(labels == 0)[0]
    sample_idx = real_indices[0]
    for idx in real_indices:
        if desync_feats[idx, 0].item() == 0.0:
            sample_idx = idx
            break

    fixed_lip = lip_seqs[sample_idx : sample_idx + 1]  # (1, 100)
    orig_desync = desync_feats[sample_idx : sample_idx + 1]

    with torch.no_grad():
        p_orig = model(fixed_lip, orig_desync).item()

    # --- TEST (a): Sweep S_desync while holding failed_ratio fixed at realistic non-zero value (0.15) ---
    s_vals = np.linspace(0.000, 0.050, 11)
    sweep_s_results = []

    print(f"\n{'=' * 75}")
    print("  BRANCH-USE SANITY CHECK — TEST (a): S_desync Sweep (failed_ratio fixed = 0.15)")
    print(f"{'=' * 75}")
    print(f"  Selected Test Sample Index: {sample_idx} (Identity: {test_dict['identities'][sample_idx]}, True Label: Real)")
    print(f"  Holding failed_ratio fixed at 0.15 (1 failed closure in ~6-7 events):")
    print(f"  {'S_desync':>10} | {'Max Penalty':>12} | {'Failed Ratio':>13} | {'P(Fake)':>10} | {'Delta P vs S=0':>16}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*16}")

    with torch.no_grad():
        p_base_s = model(fixed_lip, torch.tensor([[0.000, 0.000, 0.15]], dtype=torch.float32, device=device)).item()

    for s in s_vals:
        max_p = s * 1.5 if s > 0 else 0.0
        synth_vec = torch.tensor([[s, max_p, 0.15]], dtype=torch.float32, device=device)
        with torch.no_grad():
            p_fake = model(fixed_lip, synth_vec).item()
        delta = p_fake - p_base_s
        sweep_s_results.append({"s_desync": float(s), "p_fake": p_fake, "delta": delta})
        print(f"  {s:>10.4f} | {max_p:>12.4f} | {0.15:>13.2f} | {p_fake:>10.4f} | {delta:>+16.4f}")

    # --- TEST (b): Sweep failed_ratio while holding S_desync fixed at small value (0.010) ---
    fr_vals = [0.00, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.67, 0.80, 1.00]
    sweep_fr_results = []

    print(f"\n{'=' * 75}")
    print("  BRANCH-USE SANITY CHECK — TEST (b): failed_ratio Sweep (S_desync fixed = 0.010)")
    print(f"{'=' * 75}")
    print(f"  Holding S_desync fixed at 0.010 (typical fake event penalty) and Max Penalty at 0.015:")
    print(f"  {'Failed Ratio':>13} | {'S_desync':>10} | {'Max Penalty':>12} | {'P(Fake)':>10} | {'Delta P vs FR=0':>16}")
    print(f"  {'-'*13}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*16}")

    with torch.no_grad():
        p_base_fr = model(fixed_lip, torch.tensor([[0.010, 0.015, 0.00]], dtype=torch.float32, device=device)).item()

    for fr in fr_vals:
        synth_vec = torch.tensor([[0.010, 0.015, fr]], dtype=torch.float32, device=device)
        with torch.no_grad():
            p_fake = model(fixed_lip, synth_vec).item()
        delta = p_fake - p_base_fr
        sweep_fr_results.append({"failed_ratio": float(fr), "p_fake": p_fake, "delta": delta})
        print(f"  {fr:>13.2f} | {0.010:>10.4f} | {0.015:>12.4f} | {p_fake:>10.4f} | {delta:>+16.4f}")

    # 3. Population-level branch ablation across all 103 test samples
    with torch.no_grad():
        p_full = model(lip_seqs, desync_feats).cpu().numpy().flatten()
        zero_desync = torch.zeros_like(desync_feats)
        p_no_desync = model(lip_seqs, zero_desync).cpu().numpy().flatten()
        zero_lip = torch.zeros_like(lip_seqs)
        p_no_lip = model(zero_lip, desync_feats).cpu().numpy().flatten()

    mean_diff_desync = float(np.mean(np.abs(p_full - p_no_desync)))
    max_diff_desync = float(np.max(np.abs(p_full - p_no_desync)))
    mean_diff_lip = float(np.mean(np.abs(p_full - p_no_lip)))
    max_diff_lip = float(np.max(np.abs(p_full - p_no_lip)))

    delta_s_range = abs(sweep_s_results[-1]["p_fake"] - sweep_s_results[0]["p_fake"])
    delta_fr_range = abs(sweep_fr_results[-1]["p_fake"] - sweep_fr_results[0]["p_fake"])

    print(f"\n  Independent Sensitivity Comparison:")
    print(f"    • S_desync Magnitude Range [0.00 -> 0.05]   : Delta P = {sweep_s_results[-1]['p_fake'] - sweep_s_results[0]['p_fake']:+.4f} (range: {delta_s_range:.4f})")
    print(f"    • failed_ratio Proportion  [0.00 -> 1.00]   : Delta P = {sweep_fr_results[-1]['p_fake'] - sweep_fr_results[0]['p_fake']:+.4f} (range: {delta_fr_range:.4f})")
    print(f"    • Population Mean |P_full - P_zero_desync|  : {mean_diff_desync:.4f} (Max: {max_diff_desync:.4f})")
    print(f"    • Population Mean |P_full - P_zero_lip|     : {mean_diff_lip:.4f} (Max: {max_diff_lip:.4f})")

    if delta_s_range < 0.03 and delta_fr_range > 0.30:
        finding = (
            "CRITICAL FINDING: The desync branch response is driven almost entirely by the discrete "
            "failed_ratio (event failure frequency), NOT by continuous S_desync aperture magnitude. "
            "The model acts as an event failure rate accumulator, not a magnitude-sensitive discriminator."
        )
    else:
        finding = "Both S_desync magnitude and failed_ratio drive prediction responses."
    print(f"    • Interpretation: {finding}")

    return {
        "sweep_s": sweep_s_results,
        "sweep_fr": sweep_fr_results,
        "mean_diff_desync": mean_diff_desync,
        "max_diff_desync": max_diff_desync,
        "mean_diff_lip": mean_diff_lip,
        "max_diff_lip": max_diff_lip,
        "delta_s_range": delta_s_range,
        "delta_fr_range": delta_fr_range,
    }


def main():
    set_seed(RANDOM_SEED)
    device = get_device()
    print("=" * 80)
    print(f" SECUREKYC AI — MODEL RETRAINING & COMPREHENSIVE BENCHMARK ({device.type.upper()}) ")
    print("=" * 80)

    # 1. Load data & partition by verified connected-component split
    dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing {dataset_path}")

    print(f"[*] Loading dataset from {dataset_path}...")
    dataset_dict = torch.load(dataset_path)

    train_dict, val_dict, test_dict = split_dataset_by_identity(dataset_dict)

    train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], train_dict["desync_features"], train_dict["labels"])
    val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], val_dict["desync_features"], val_dict["labels"])
    test_ds = MultimodalKYCDataset(test_dict["lip_sequences"], test_dict["desync_features"], test_dict["labels"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Retrain MultimodalKYCFusionNet from scratch (DO NOT RESUME)
    print("\n[1/4] Retraining MultimodalKYCFusionNet from scratch on MPS...")
    fusion_model_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"
    curves_csv_path = PROCESSED_DATA_DIR / "training_curves.csv"

    # Ensure clean fresh initialization
    fusion_model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3).to(device)

    fusion_model, fusion_info = train_single_model(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        save_path=fusion_model_path,
        log_csv_path=curves_csv_path,
    )

    # 3. Comprehensive Performance Evaluation (Train & Test)
    print("\n[2/4] Evaluating Full Performance & Decision Boundary...")
    train_eval = evaluate_split_performance(fusion_model, train_dict, device, "TRAIN SPLIT")
    test_eval = evaluate_split_performance(fusion_model, test_dict, device, "TEST SPLIT")

    # Print Confusion Matrices
    print(format_confusion_matrix(train_eval["labels"], (train_eval["scores"] >= 0.5).astype(int), "TRAIN SPLIT"))
    print(f"  Train Precision: {train_eval['precision']:.4f}")
    print(f"  Train Recall   : {train_eval['recall']:.4f}")
    print(f"  Train F1-Score : {train_eval['f1']:.4f}")
    print(f"  Train ROC-AUC  : {train_eval['roc_auc']:.4f}")

    print(format_confusion_matrix(test_eval["labels"], (test_eval["scores"] >= 0.5).astype(int), "HELD-OUT TEST SPLIT"))
    print(f"  Test Precision : {test_eval['precision']:.4f}")
    print(f"  Test Recall    : {test_eval['recall']:.4f}")
    print(f"  Test F1-Score  : {test_eval['f1']:.4f}")
    print(f"  Test ROC-AUC   : {test_eval['roc_auc']:.4f}")
    print(f"  Test EER       : {test_eval['eer']*100:.2f}% (at threshold={test_eval['opt_thresh']:.4f})")

    # Analyze Decision Boundary / Collapse
    analyze_decision_boundary(test_eval, "HELD-OUT TEST SPLIT")

    # 4. Branch-use Sanity Check
    print("\n[3/4] Re-running Branch-Use Sanity Check on New Checkpoint...")
    sanity_results = run_branch_use_sanity_check(fusion_model, test_dict, device)

    # 5. Three-Way Ablation Benchmark
    print("\n[4/4] Executing Fresh Three-Way Ablation on N=691...")

    # (A) Spatial-Only: Zero out desync_vec on the fused model + Train standalone VisionOnlyNet
    print("  -> Training Standalone VisionOnlyNet (Spatial-Only)...")
    vision_model = VisionOnlyNet(seq_len=100).to(device)
    vision_model, _ = train_single_model(
        model=vision_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        is_vision_only=True,
    )
    spatial_standalone_eval = evaluate_split_performance(vision_model, test_dict, device, "TEST (Spatial Standalone)", is_vision_only=True)

    # Spatial-Only via Zero-Out in Fused model (paper convention)
    zero_desync_test = dict(test_dict)
    zero_desync_test["desync_features"] = torch.zeros_like(test_dict["desync_features"])
    spatial_zero_eval = evaluate_split_performance(fusion_model, zero_desync_test, device, "TEST (Spatial Zero-Out)")

    # (B) Passive-Only: Zero out lip_seq on the fused model + Train standalone DesyncOnlyNet
    print("  -> Training Standalone DesyncOnlyNet (Passive-Only)...")
    desync_model = DesyncOnlyNet(desync_feature_dim=3).to(device)
    desync_model, _ = train_single_model(
        model=desync_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        device=device,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        is_desync_only=True,
    )
    passive_standalone_eval = evaluate_split_performance(desync_model, test_dict, device, "TEST (Passive Standalone)", is_desync_only=True)

    # Passive-Only via Zero-Out in Fused model
    zero_lip_test = dict(test_dict)
    zero_lip_test["lip_sequences"] = torch.zeros_like(test_dict["lip_sequences"])
    passive_zero_eval = evaluate_split_performance(fusion_model, zero_lip_test, device, "TEST (Passive Zero-Out)")

    # Save fresh paper curves
    figures_dir = PAPER_DIR / "figures"
    fpr_f, tpr_f, _ = roc_curve(test_eval["labels"], test_eval["scores"])
    fpr_s, tpr_s, _ = roc_curve(spatial_zero_eval["labels"], spatial_zero_eval["scores"])
    fpr_p, tpr_p, _ = roc_curve(passive_zero_eval["labels"], passive_zero_eval["scores"])

    fused_for_plot = {
        "fpr": fpr_f, "tpr": tpr_f, "roc_auc": test_eval["roc_auc"],
        "eer": test_eval["eer"], "threshold": test_eval["opt_thresh"],
        "thresholds": np.linspace(0, 1, 1000),
        "far_arr": compute_eer(test_eval["labels"], test_eval["scores"])[3],
        "frr_arr": compute_eer(test_eval["labels"], test_eval["scores"])[4],
    }
    spatial_for_plot = {"fpr": fpr_s, "tpr": tpr_s, "roc_auc": spatial_zero_eval["roc_auc"]}
    passive_for_plot = {"fpr": fpr_p, "tpr": tpr_p, "roc_auc": passive_zero_eval["roc_auc"]}

    save_evaluation_plots(fused_for_plot, spatial_for_plot, passive_for_plot, figures_dir)

    # Summary Ablation Table
    print("\n" + "=" * 105)
    print(f"{'SECUREKYC AI — THREE-WAY ABLATION BENCHMARK (N=691, HELD-OUT TEST N=103)':^105}")
    print("=" * 105)
    print(f"{'Configuration':<38} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1':^8} | {'ROC-AUC':^9} | {'EER':^9}")
    print("-" * 105)

    def print_ablation_row(name: str, m: Dict[str, Any]):
        print(
            f"{name:<38} | "
            f"{m['accuracy'] * 100:^9.2f}% | "
            f"{m['precision'] * 100:^9.2f}% | "
            f"{m['recall'] * 100:^9.2f}% | "
            f"{m['f1']:^8.4f} | "
            f"{m['roc_auc']:^9.4f} | "
            f"{m['eer'] * 100:^8.2f}%"
        )

    print_ablation_row("Spatial-Only (Zero Desync)", spatial_zero_eval)
    print_ablation_row("Spatial-Only (Standalone Bi-LSTM)", spatial_standalone_eval)
    print_ablation_row("Passive-Only (Zero Lip)", passive_zero_eval)
    print_ablation_row("Passive-Only (Standalone Linear)", passive_standalone_eval)
    print("-" * 105)
    print_ablation_row("MultimodalKYCFusionNet (Fused)", test_eval)
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
