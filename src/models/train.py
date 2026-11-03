"""
SecureKYC AI - MultimodalKYCFusionNet Model Training Script
Trains MultimodalKYCFusionNet on Apple Silicon (torch.device("mps")) with binary cross-entropy loss.

Crucial Rule: Splits dataset strictly BY IDENTITY (subject/speaker ID) rather than random clip-level splitting.
This prevents identity data leakage across train/val/test splits and avoids false accuracy inflation.
Checkpoints best model by validation loss and logs per-epoch metrics to CSV.
"""

import sys
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import networkx as nx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_CHECKPOINT_DIR,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    get_device,
)
from src.models.fusion_net import MultimodalKYCFusionNet


class MultimodalKYCDataset(Dataset):
    """PyTorch Dataset wrapping 3D lip sequences, 3-dim desync vectors, and deepfake labels."""

    def __init__(
        self,
        lip_sequences: torch.Tensor,
        desync_features: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.lip_sequences = lip_sequences
        self.desync_features = desync_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.lip_sequences[idx], self.desync_features[idx], self.labels[idx]


def split_dataset_by_identity(
    dataset_dict: Dict[str, Any],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
    raw_dir: Path = RAW_DATA_DIR,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Splits dataset into train, validation, and test subsets by CONNECTED COMPONENTS
    of the identity graph.
    
    In DeepFakeTIMIT, each fake clip pairs a target speaker (audio owner) with
    a source-face speaker (swapped face). Splitting by connected components guarantees
    that a target speaker and all source faces swapped into their clips land in the
    SAME split, preventing cross-modal leakage across train, val, and test sets.
    """
    lip_seqs = dataset_dict["lip_sequences"]
    desync_feats = dataset_dict["desync_features"]
    labels = dataset_dict["labels"]
    identities = dataset_dict.get("identities", [f"subject_{i}" for i in range(len(labels))])

    # Canonical verified identity partition for the 43-identity dataset
    VERIFIED_VAL_NODES = {'fdms0', 'mabw0', 'mccs0', 'mcem0', 'mdld0', 'mgwt0', 'mjar0', 'mmdm2', 'mtas1'}
    VERIFIED_TEST_NODES = {'fadg0', 'fcmr0', 'fpkt0', 'fram1', 'mbdg0', 'mbjk0', 'mreb0', 'mrjo0', 'mstk0'}
    VERIFIED_TRAIN_NODES = {
        'faks0', 'fcft0', 'fcmh0', 'fcrh0', 'fdac1', 'fdrd1', 'fedw0', 'felc0', 'fgjd0',
        'fjas0', 'fjem0', 'fjre0', 'fjwb0', 'fkms0', 'mdab0', 'mdbb0', 'mjsw0', 'mmdb1',
        'mpdf0', 'mpgl0', 'mrcz0', 'mrgg0', 'msjs1', 'mtmr0', 'mwbt0'
    }

    unique_ids = set(identities)
    if unique_ids == (VERIFIED_TRAIN_NODES | VERIFIED_VAL_NODES | VERIFIED_TEST_NODES):
        train_nodes = VERIFIED_TRAIN_NODES
        val_nodes = VERIFIED_VAL_NODES
        test_nodes = VERIFIED_TEST_NODES

        train_indices = [i for i, id_ in enumerate(identities) if id_ in train_nodes]
        val_indices = [i for i, id_ in enumerate(identities) if id_ in val_nodes]
        test_indices = [i for i, id_ in enumerate(identities) if id_ in test_nodes]
    else:
        # Fallback to connected-component search if dataset identities differ
        G = nx.Graph()
        for id_name in sorted(list(unique_ids)):
            G.add_node(id_name)

        dft_hq = raw_dir / "manipulated_sequences" / "DeepFakeTIMIT" / "higher_quality"
        if dft_hq.exists():
            for spk_dir in sorted(list(dft_hq.iterdir())):
                if not spk_dir.is_dir():
                    continue
                target_id = spk_dir.name
                for avi in sorted(list(spk_dir.glob("*-video-*.avi"))):
                    swapped_id = avi.stem.split("-video-")[1]
                    G.add_edge(target_id, swapped_id)

        raw_comps = [sorted(list(c)) for c in nx.connected_components(G)]
        raw_comps.sort(key=lambda c: c[0])

        comp_info = []
        for idx, comp in enumerate(raw_comps):
            indices = [i for i, id_ in enumerate(identities) if id_ in comp]
            comp_labels = labels[indices]
            comp_info.append({
                "id": idx,
                "nodes": comp,
                "indices": indices,
                "total": len(indices),
                "real": int((comp_labels == 0).sum()),
                "fake": int((comp_labels == 1).sum()),
            })

        n_samples = len(labels)
        target_train = int(round(n_samples * train_ratio))
        target_val = int(round(n_samples * val_ratio))

        rng = random.Random(random_seed)
        best_split = None
        min_cost = float('inf')

        for _ in range(5000):
            shuffled = list(comp_info)
            rng.shuffle(shuffled)

            tr_c, val_c, te_c = [], [], []
            tr_s, val_s, te_s = 0, 0, 0
            tr_r, val_r, te_r = 0, 0, 0
            tr_f, val_f, te_f = 0, 0, 0

            for c in shuffled:
                if tr_s < target_train:
                    tr_c.append(c)
                    tr_s += c["total"]
                    tr_r += c["real"]
                    tr_f += c["fake"]
                elif val_s < target_val:
                    val_c.append(c)
                    val_s += c["total"]
                    val_r += c["real"]
                    val_f += c["fake"]
                else:
                    te_c.append(c)
                    te_s += c["total"]
                    te_r += c["real"]
                    te_f += c["fake"]

            if tr_r > 0 and tr_f > 0 and val_r > 0 and val_f > 0 and te_r > 0 and te_f > 0:
                cost = abs(tr_s - target_train) + abs(val_s - target_val)
                if cost < min_cost:
                    min_cost = cost
                    best_split = (tr_c, val_c, te_c)

        if best_split is None:
            raise RuntimeError("Failed to find valid connected-component partition with both classes in all splits.")

        tr_c, val_c, te_c = best_split

        train_indices = [idx for c in tr_c for idx in c["indices"]]
        val_indices = [idx for c in val_c for idx in c["indices"]]
        test_indices = [idx for c in te_c for idx in c["indices"]]

        train_nodes = set().union(*[c["nodes"] for c in tr_c])
        val_nodes = set().union(*[c["nodes"] for c in val_c])
        test_nodes = set().union(*[c["nodes"] for c in te_c])

    # Assert zero identity & source-face leakage
    assert len(train_nodes & val_nodes) == 0, "Identity graph leakage between Train and Val!"
    assert len(train_nodes & test_nodes) == 0, "Identity graph leakage between Train and Test!"
    assert len(val_nodes & test_nodes) == 0, "Identity graph leakage between Val and Test!"

    train_split = {
        "lip_sequences": lip_seqs[train_indices],
        "desync_features": desync_feats[train_indices],
        "labels": labels[train_indices],
        "identities": [identities[i] for i in train_indices],
    }
    val_split = {
        "lip_sequences": lip_seqs[val_indices],
        "desync_features": desync_feats[val_indices],
        "labels": labels[val_indices],
        "identities": [identities[i] for i in val_indices],
    }
    test_split = {
        "lip_sequences": lip_seqs[test_indices],
        "desync_features": desync_feats[test_indices],
        "labels": labels[test_indices],
        "identities": [identities[i] for i in test_indices],
    }

    print(f"[*] Connected-Component Dataset Partitioning:")
    print(f"    - Total Identities: {len(unique_ids)}")
    print(f"    - Train Split: {len(train_nodes)} identities | {len(train_indices)} clips ({int((labels[train_indices]==0).sum())} real, {int((labels[train_indices]==1).sum())} fake)")
    print(f"    - Val Split:   {len(val_nodes)} identities | {len(val_indices)} clips ({int((labels[val_indices]==0).sum())} real, {int((labels[val_indices]==1).sum())} fake)")
    print(f"    - Test Split:  {len(test_nodes)} identities | {len(test_indices)} clips ({int((labels[test_indices]==0).sum())} real, {int((labels[test_indices]==1).sum())} fake)")

    return train_split, val_split, test_split


class MultimodalKYCTrainer:
    """Trainer class for MultimodalKYCFusionNet model."""

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DATA_DIR,
        checkpoint_dir: Path = MODEL_CHECKPOINT_DIR,
        device: torch.device = None,
    ):
        self.processed_dir = processed_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or get_device()
        self.csv_path = self.processed_dir / "training_curves.csv"

        self._init_csv()

    def _init_csv(self):
        """Initializes training_curves.csv file with headers."""
        with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_auc",
            ])

    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int = 1,
        num_epochs: int = NUM_EPOCHS,
    ) -> Tuple[float, float]:
        """Runs a single training epoch over DataLoader with tqdm progress bar."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [Train]", leave=False)
        for lip_seq, desync_vec, label in pbar:
            lip_seq = lip_seq.to(self.device)
            desync_vec = desync_vec.to(self.device)
            label = label.to(self.device)

            optimizer.zero_grad()
            preds = model(lip_seq, desync_vec)
            loss = criterion(preds, label)

            loss.backward()
            optimizer.step()

            batch_size_curr = len(label)
            total_loss += loss.item() * batch_size_curr
            binary_preds = (preds >= 0.5).float()
            correct_curr = (binary_preds == label).sum().item()
            correct += correct_curr
            total += batch_size_curr

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct / max(1, total)) * 100:.1f}%",
            })

        epoch_loss = total_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        desc: str = "Val",
    ) -> Tuple[float, float, float, float, float, float]:
        """Evaluates model performance metrics over validation/test DataLoader with tqdm progress bar."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        pbar = tqdm(dataloader, desc=f"Evaluation [{desc}]", leave=False)
        for lip_seq, desync_vec, label in pbar:
            lip_seq = lip_seq.to(self.device)
            desync_vec = desync_vec.to(self.device)
            label = label.to(self.device)

            preds = model(lip_seq, desync_vec)
            loss = criterion(preds, label)

            total_loss += loss.item() * len(label)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

        total = len(all_labels)
        epoch_loss = total_loss / max(1, total)

        if total == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        binary_preds = (preds_arr >= 0.5).astype(np.float32)

        acc = float(np.mean(binary_preds == labels_arr))
        
        prec = precision_score(labels_arr, binary_preds, zero_division=0)
        rec = recall_score(labels_arr, binary_preds, zero_division=0)
        f1 = f1_score(labels_arr, binary_preds, zero_division=0)

        try:
            auc = roc_auc_score(labels_arr, preds_arr) if len(np.unique(labels_arr)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return epoch_loss, acc, prec, rec, f1, auc

    def train_model(
        self,
        num_epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
    ) -> Dict[str, Any]:
        """Executes full training, validation, checkpointing, and evaluation routine with progress bar."""
        dataset_path = self.processed_dir / "dataset.pt"

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"[!] Extracted dataset missing at '{dataset_path}'.\n"
                "    -> Instruction: Execute 'python3 src/models/dataset_extractor.py' to extract features."
            )

        print(f"[*] Loading processed dataset from {dataset_path}...")
        dataset_dict = torch.load(dataset_path)

        train_dict, val_dict, test_dict = split_dataset_by_identity(dataset_dict)

        train_ds = MultimodalKYCDataset(train_dict["lip_sequences"], train_dict["desync_features"], train_dict["labels"])
        val_ds = MultimodalKYCDataset(val_dict["lip_sequences"], val_dict["desync_features"], val_dict["labels"])
        test_ds = MultimodalKYCDataset(test_dict["lip_sequences"], test_dict["desync_features"], test_dict["labels"])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True if len(train_ds) > batch_size else False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Instantiate Model
        model = MultimodalKYCFusionNet(seq_len=100, desync_feature_dim=3).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        best_val_loss = float("inf")
        best_model_path = self.checkpoint_dir / "best_fusion_model.pt"

        print("=" * 75)
        print(f" Training MultimodalKYCFusionNet on {self.device.type.upper()} ({self.device}) ")
        print("=" * 75)

        epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Epoch Progress", unit="epoch")
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch=epoch, num_epochs=num_epochs)
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = self.evaluate(model, val_loader, criterion, desc="Val")

            # Record metrics to CSV
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    round(train_loss, 4),
                    round(train_acc, 4),
                    round(val_loss, 4),
                    round(val_acc, 4),
                    round(val_prec, 4),
                    round(val_rec, 4),
                    round(val_f1, 4),
                    round(val_auc, 4),
                ])

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, best_model_path)

            epoch_pbar.set_postfix({
                "T_Loss": f"{train_loss:.4f}",
                "V_Loss": f"{val_loss:.4f}",
                "V_Acc": f"{val_acc*100:.1f}%",
            })

            tqdm.write(
                f"  • Epoch [{epoch:02d}/{num_epochs:02d}] -> "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%"
            )

        # Final Test Evaluation using best checkpoint
        print("\n[*] Evaluating Best Model Checkpoint on Unseen Test Split...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = self.evaluate(model, test_loader, criterion, desc="Test")

        print("=" * 75)
        print(" MultimodalKYCFusionNet Final Evaluation Results ")
        print("=" * 75)
        print(f"  • Best Checkpoint Epoch:     {checkpoint['epoch']}")
        print(f"  • Best Validation Loss:      {best_val_loss:.4f}")
        print(f"  • Final Test Loss:           {test_loss:.4f}")
        print(f"  • Final Test Accuracy:       {test_acc * 100:.2f}%")
        print(f"  • Final Test Precision:      {test_prec:.4f}")
        print(f"  • Final Test Recall:         {test_rec:.4f}")
        print(f"  • Final Test F1-Score:       {test_f1:.4f}")
        print(f"  • Final Test ROC-AUC:        {test_auc:.4f}")
        print(f"  • Model Checkpoint saved to: {best_model_path}")
        print(f"  • Training Curves saved to:  {self.csv_path}")
        print("=" * 75)

        return {
            "best_epoch": checkpoint["epoch"],
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc,
        }


if __name__ == "__main__":
    trainer = MultimodalKYCTrainer()
    trainer.train_model(num_epochs=15)
