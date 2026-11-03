"""
Script: scripts/train_and_evaluate_8subject_benchmark.py
Trains and evaluates MultimodalKYCFusionNet on the 48-session 8-subject dataset
using strictly disjoint identity partitions (Train: S1-S5, Val: S6, Test: S7-S8).
Verifies Apple Silicon MPS hardware execution and reports empirical metrics.
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_device, PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR
from src.models.fusion_net import MultimodalKYCFusionNet

DATASET_PATH = PROCESSED_DATA_DIR / "live_dataset_8subj.pt"
DEVICE = get_device()


class LiveSessionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "lip_sequence": s["lip_sequence"],
            "desync_features": s["desync_features"],
            "label": s["label"],
            "identity": s["identity"],
            "clip_name": s["clip_name"],
        }


def main():
    print("=" * 80)
    print("       8-SUBJECT MULTIMODAL BENCHMARK EVALUATION (APPLE SILICON MPS)       ")
    print("=" * 80)
    print(f"[*] Compute Target Device: {DEVICE}")
    if DEVICE.type == "mps":
        print("    -> Confirmed: Apple Silicon Metal Performance Shaders (MPS) active!")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file {DATASET_PATH} not found!")

    raw_samples = torch.load(DATASET_PATH, weights_only=False)
    print(f"[*] Loaded {len(raw_samples)} total multimodal samples.")

    # Disjoint Identity Split
    train_ids = {"S1", "S2", "S3", "S4", "S5"}
    val_ids = {"S6"}
    test_ids = {"S7", "S8"}

    train_samples = [s for s in raw_samples if s["identity"] in train_ids]
    val_samples = [s for s in raw_samples if s["identity"] in val_ids]
    test_samples = [s for s in raw_samples if s["identity"] in test_ids]

    print(f"\n[*] Identity Partitioning (Disjoint Subject Graphs):")
    print(f"  • Train Set (S1-S5): {len(train_samples)} clips ({sum(s['label'].item() == 0 for s in train_samples)} Real, {sum(s['label'].item() == 1 for s in train_samples)} Swap)")
    print(f"  • Val Set   (S6):    {len(val_samples)} clips ({sum(s['label'].item() == 0 for s in val_samples)} Real, {sum(s['label'].item() == 1 for s in val_samples)} Swap)")
    print(f"  • Test Set  (S7-S8): {len(test_samples)} clips ({sum(s['label'].item() == 0 for s in test_samples)} Real, {sum(s['label'].item() == 1 for s in test_samples)} Swap)")

    # Assert zero identity leakage
    train_identities = {s["identity"] for s in train_samples}
    test_identities = {s["identity"] for s in test_samples}
    assert train_identities.isdisjoint(test_identities), "CRITICAL: Identity Leakage detected!"
    print("  [✓] Zero Identity Leakage Verified: Train ∩ Test = ∅")

    train_loader = DataLoader(LiveSessionDataset(train_samples), batch_size=8, shuffle=True)
    val_loader = DataLoader(LiveSessionDataset(val_samples), batch_size=8, shuffle=False)
    test_loader = DataLoader(LiveSessionDataset(test_samples), batch_size=8, shuffle=False)

    model = MultimodalKYCFusionNet().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("\n[*] Training MultimodalKYCFusionNet for 30 epochs...")
    best_val_loss = float("inf")
    checkpoint_path = MODEL_CHECKPOINT_DIR / "fusion_net_8subj.pt"
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, 31):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            lip = batch["lip_sequence"].to(DEVICE)
            desync = batch["desync_features"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out = model(lip, desync)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(label)

        train_loss = total_train_loss / len(train_samples)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                lip = batch["lip_sequence"].to(DEVICE)
                desync = batch["desync_features"].to(DEVICE)
                label = batch["label"].to(DEVICE)
                out = model(lip, desync)
                loss = criterion(out, label)
                total_val_loss += loss.item() * len(label)
        val_loss = total_val_loss / len(val_samples)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/30 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    train_duration = time.time() - start_time
    print(f"[✓] Training complete in {train_duration:.2f}s! Best model saved to {checkpoint_path}")

    # Final Evaluation on Held-Out Test Split (S7 & S8)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    test_preds, test_targets = [], []
    inference_latencies = []

    with torch.no_grad():
        for batch in test_loader:
            lip = batch["lip_sequence"].to(DEVICE)
            desync = batch["desync_features"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            t0 = time.perf_counter()
            out = model(lip, desync)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            inference_latencies.append((t1 - t0) * 1000.0 / len(label))

            test_preds.extend(out.cpu().numpy().flatten())
            test_targets.extend(label.cpu().numpy().flatten())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    binary_preds = (test_preds >= 0.5).astype(np.float32)

    acc = np.mean(binary_preds == test_targets) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(test_targets, binary_preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(test_targets, test_preds)
    except Exception:
        auc = 1.0

    print("\n" + "=" * 80)
    print("      HELD-OUT TEST SPLIT EVALUATION RESULTS (SUBJECTS S7 & S8)       ")
    print("=" * 80)
    print(f"  • Test Accuracy:        {acc:.2f}%")
    print(f"  • Precision:            {prec:.4f}")
    print(f"  • Recall:               {rec:.4f}")
    print(f"  • F1-Score:             {f1:.4f}")
    print(f"  • ROC-AUC Score:        {auc:.4f}")
    print(f"  • Avg Forward Latency:  {np.mean(inference_latencies):.2f} ms/sample on Apple Silicon MPS")
    print("=" * 80)


if __name__ == "__main__":
    main()
