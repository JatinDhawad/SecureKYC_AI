"""
SecureKYC AI - Model Evaluation, EER Calculation & Ablation Study Engine
Evaluates MultimodalKYCFusionNet on the real held-out test split (split BY IDENTITY).

Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Equal Error Rate (EER).
Runs three ablation configurations:
  1. Fused (Full Model): Both lip_sequence and desync_vec.
  2. Spatial-Only: Zero out desync_vec (torch.zeros_like(desync_features)).
  3. Passive-Only: Zero out lip_seq (torch.zeros_like(lip_sequences)).

Saves ROC curve (paper/figures/roc_curves.png) and EER plot (paper/figures/eer_curves.png) for the research paper.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PROCESSED_DATA_DIR,
    MODEL_CHECKPOINT_DIR,
    PAPER_DIR,
    get_device,
)
from src.models.fusion_net import MultimodalKYCFusionNet
from src.models.train import split_dataset_by_identity


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) where False Acceptance Rate (FAR) == False Rejection Rate (FRR).

    Args:
        labels: 1D numpy array of ground truth binary labels (0 = authentic, 1 = fake).
        scores: 1D numpy array of predicted deepfake confidence scores in [0, 1].

    Returns:
        Tuple of (eer_value, optimal_threshold, thresholds, far_arr, frr_arr).
    """
    thresholds = np.linspace(0.0, 1.0, 1000)
    far_list = []
    frr_list = []

    # Authentic class = 0 (Real), Fake class = 1 (Deepfake)
    authentic_mask = (labels == 0)
    fake_mask = (labels == 1)

    n_authentic = np.sum(authentic_mask)
    n_fake = np.sum(fake_mask)

    if n_authentic == 0 or n_fake == 0:
        return 0.0, 0.5, thresholds, np.zeros(1000), np.zeros(1000)

    for t in thresholds:
        # False Acceptance: Authentic classified as Fake (score >= t)
        fa = np.sum((scores >= t) & authentic_mask)
        # False Rejection: Fake classified as Authentic (score < t)
        fr = np.sum((scores < t) & fake_mask)

        far = fa / n_authentic
        frr = fr / n_fake

        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    diff = np.abs(far_arr - frr_arr)
    min_idx = np.argmin(diff)

    eer = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)
    opt_threshold = float(thresholds[min_idx])

    return eer, opt_threshold, thresholds, far_arr, frr_arr


def evaluate_model_metrics(
    model: nn.Module,
    lip_seqs: torch.Tensor,
    desync_feats: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, and EER for a PyTorch model configuration."""
    model.eval()
    model.to(device)

    with torch.no_grad():
        preds = model(lip_seqs.to(device), desync_feats.to(device))
        scores = preds.cpu().numpy().flatten()
        y_true = labels.cpu().numpy().flatten()

    unique_classes = len(np.unique(y_true))
    roc_auc = float(roc_auc_score(y_true, scores)) if unique_classes > 1 else 0.5
    eer, opt_thresh, thresholds, far_arr, frr_arr = compute_eer(y_true, scores)

    binary_preds = (scores >= opt_thresh).astype(int)
    acc = float(accuracy_score(y_true, binary_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, binary_preds, average="binary", zero_division=0
    )

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores) if unique_classes > 1 else (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))

    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "eer": eer,
        "threshold": opt_thresh,
        "y_true": y_true,
        "scores": scores,
        "fpr": fpr,
        "tpr": tpr,
        "far_arr": far_arr,
        "frr_arr": frr_arr,
        "thresholds": thresholds,
    }


def save_evaluation_plots(
    fused_res: Dict[str, Any],
    spatial_res: Dict[str, Any],
    passive_res: Dict[str, Any],
    figures_dir: Path,
):
    """Generates and saves research paper figure artifacts (roc_curves.png and eer_curves.png)."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. ROC Curves Plot ---
    plt.figure(figsize=(7, 6))
    plt.plot(fused_res["fpr"], fused_res["tpr"], label=f"Fused Model (AUC = {fused_res['roc_auc']:.4f})", color="#0055FF", linewidth=2.5)
    plt.plot(spatial_res["fpr"], spatial_res["tpr"], label=f"Spatial-Only (AUC = {spatial_res['roc_auc']:.4f})", color="#FF5500", linestyle="--", linewidth=2.0)
    plt.plot(passive_res["fpr"], passive_res["tpr"], label=f"Passive-Only (AUC = {passive_res['roc_auc']:.4f})", color="#22AA55", linestyle="-.", linewidth=2.0)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Chance")

    plt.xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold")
    plt.title("SecureKYC AI - ROC Curves Across Model Ablations", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_path = figures_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[✓] Saved ROC curves plot to {roc_path}")

    # --- 2. EER / DET Curves Plot ---
    plt.figure(figsize=(7, 6))
    plt.plot(fused_res["thresholds"], fused_res["far_arr"], label="FAR (False Acceptance Rate)", color="#EE2200", linewidth=2.2)
    plt.plot(fused_res["thresholds"], fused_res["frr_arr"], label="FRR (False Rejection Rate)", color="#0088FF", linewidth=2.2)

    # Plot EER Intersection Point
    eer_val = fused_res["eer"]
    opt_t = fused_res["threshold"]
    plt.plot(opt_t, eer_val, "ko", markersize=8, label=f"EER Point = {eer_val*100:.2f}% (t={opt_t:.3f})")

    plt.xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    plt.ylabel("Error Rate", fontsize=11, fontweight="bold")
    plt.title("SecureKYC AI - Equal Error Rate (EER) Threshold Intersection", fontsize=12, fontweight="bold")
    plt.legend(loc="upper center", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    eer_path = figures_dir / "eer_curves.png"
    plt.savefig(eer_path, dpi=300)
    plt.close()
    print(f"[✓] Saved EER curves plot to {eer_path}")


def run_full_evaluation(
    data_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes model evaluation on held-out test split across 3 ablation configurations:
      1. Fused (Full Model): Both lip_sequence and desync_features.
      2. Spatial-Only: Zero out desync_vec (torch.zeros_like(desync_features)).
      3. Passive-Only: Zero out lip_seq (torch.zeros_like(lip_sequences)).
    """
    device = get_device()
    if data_path is None:
        data_path = PROCESSED_DATA_DIR / "dataset.pt"

    if checkpoint_path is None:
        checkpoint_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"

    if not data_path.exists():
        raise FileNotFoundError(f"[!] Extracted dataset missing at {data_path}. Execute 'python3 src/models/dataset_extractor.py'.")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"[!] Trained model checkpoint missing at {checkpoint_path}. Execute 'python3 src/models/train.py'.")

    print("=" * 80)
    print(f" Executing MultimodalKYCFusionNet Evaluation on {device.type.upper()} ({device}) ")
    print("=" * 80)

    # 1. Load dataset and perform identity-based held-out test split
    data_dict = torch.load(data_path)
    _, _, test_dict = split_dataset_by_identity(data_dict)

    lip_seqs = test_dict["lip_sequences"]
    desync_feats = test_dict["desync_features"]
    labels = test_dict["labels"]

    print(f"[*] Evaluation Target: Held-out Test Split ({len(labels)} clips across unseen identities)")

    # 2. Load trained model checkpoint
    model = MultimodalKYCFusionNet(seq_len=lip_seqs.shape[1], desync_feature_dim=desync_feats.shape[1])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[✓] Loaded trained model checkpoint from {checkpoint_path} (Best Epoch: {checkpoint['epoch']})")

    # 3. Compute metrics for 3 Ablation Configurations

    # Config 1: Fused (Full Model)
    fused_metrics = evaluate_model_metrics(model, lip_seqs, desync_feats, labels, device)

    # Config 2: Spatial-Only (Zero out desync_vec)
    zero_desync = torch.zeros_like(desync_feats)
    spatial_metrics = evaluate_model_metrics(model, lip_seqs, zero_desync, labels, device)

    # Config 3: Passive-Only (Zero out lip_seq)
    zero_lip = torch.zeros_like(lip_seqs)
    passive_metrics = evaluate_model_metrics(model, zero_lip, desync_feats, labels, device)

    # 4. Generate & Save Research Figure Artifacts
    figures_dir = PAPER_DIR / "figures"
    save_evaluation_plots(fused_metrics, spatial_metrics, passive_metrics, figures_dir)

    # 5. Print Single Comparison Summary Table
    print("\n" + "=" * 98)
    print(f"{'SECUREKYC AI RESEARCH MODEL ABLATION COMPARISON TABLE':^98}")
    print("=" * 98)
    print(
        f"{'Model Variant / Feature Configuration':<38} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^9} | {'ROC-AUC':^9} | {'EER (%)':^9}"
    )
    print("-" * 98)

    def print_row(name: str, m: Dict[str, Any]):
        print(
            f"{name:<38} | "
            f"{m['accuracy'] * 100:^9.2f}% | "
            f"{m['precision'] * 100:^9.2f}% | "
            f"{m['recall'] * 100:^9.2f}% | "
            f"{m['f1']:^9.4f} | "
            f"{m['roc_auc']:^9.4f} | "
            f"{m['eer'] * 100:^9.2f}%"
        )

    print_row("Spatial-Only (Zero out desync_vec)", spatial_metrics)
    print_row("Passive-Only (Zero out lip_seq)", passive_metrics)
    print("-" * 98)
    print_row("Proposed: MultimodalKYCFusionNet (Fused)", fused_metrics)
    print("=" * 98 + "\n")

    return {
        "fused": fused_metrics,
        "spatial_only": spatial_metrics,
        "passive_only": passive_metrics,
    }


if __name__ == "__main__":
    run_full_evaluation()
