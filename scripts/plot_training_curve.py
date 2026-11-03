#!/usr/bin/env python3
"""
Script: scripts/plot_training_curve.py
Plots training and validation dynamics directly from the actual N=691 retrain log:
data/processed/training_curves.csv
Outputs high-definition publication figure to paper/figures/training_curve.png.
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "training_curves.csv"
OUT_PATH = PROJECT_ROOT / "paper" / "figures" / "training_curve.png"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Training curves CSV not found at: {CSV_PATH}")

epochs = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []
val_auc = []

with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        train_acc.append(float(row["train_acc"]) * 100.0)
        val_loss.append(float(row["val_loss"]))
        val_acc.append(float(row["val_acc"]) * 100.0)
        val_auc.append(float(row["val_auc"]))

print(f"Loaded {len(epochs)} epochs from {CSV_PATH}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Panel 1: Loss Dynamics
ax1 = axes[0]
ax1.plot(epochs, train_loss, marker="o", markersize=4, linewidth=2.0, color="#2563EB", label="Train Loss (BCE)")
ax1.plot(epochs, val_loss, marker="s", markersize=4, linewidth=2.0, color="#DC2626", linestyle="--", label="Val Loss (BCE)")
ax1.set_title("(A) Cross-Entropy Loss Convergence (N=691 Retrain)", fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Training Epoch", fontsize=10)
ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
ax1.set_xticks(epochs[::2] if len(epochs) > 10 else epochs)
ax1.grid(True, alpha=0.3, linestyle=":")
ax1.legend(loc="upper right", fontsize=9.5)

# Panel 2: Accuracy & Ranking Dynamics
ax2 = axes[1]
ax2.plot(epochs, train_acc, marker="o", markersize=4, linewidth=2.0, color="#059669", label="Train Accuracy (%)")
ax2.plot(epochs, val_acc, marker="s", markersize=4, linewidth=2.0, color="#D97706", linestyle="--", label="Val Accuracy (%)")
ax2.plot(epochs, [x * 100.0 for x in val_auc], marker="^", markersize=4, linewidth=2.0, color="#7C3AED", linestyle="-.", label="Val ROC-AUC (×100)")
ax2.set_title("(B) Classification Accuracy & Ranking Discrimination", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Training Epoch", fontsize=10)
ax2.set_ylabel("Metric Score (%)", fontsize=10)
ax2.set_xticks(epochs[::2] if len(epochs) > 10 else epochs)
ax2.set_ylim(50, 100)
ax2.grid(True, alpha=0.3, linestyle=":")
ax2.legend(loc="lower right", fontsize=9.5)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"[✓] Successfully plotted real training dynamics to: {OUT_PATH}")
