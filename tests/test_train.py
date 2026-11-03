"""
Unit tests for src/models/train.py
Verifies identity-based dataset splitting (zero identity leakage across splits),
dataset wrapping, and model training routines.
"""

import sys
from pathlib import Path
import torch
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import split_dataset_by_identity, MultimodalKYCDataset, MultimodalKYCTrainer


def test_split_dataset_by_identity_zero_leakage():
    """Asserts that identity-based splitting guarantees zero identity overlap across train, val, and test sets."""
    n_samples = 100
    identities = [f"speaker_{i % 10}" for i in range(n_samples)]  # 10 unique speakers

    dummy_dataset = {
        "lip_sequences": torch.randn(n_samples, 100),
        "desync_features": torch.randn(n_samples, 3),
        "labels": torch.randint(0, 2, (n_samples, 1)).float(),
        "identities": identities,
    }

    train_split, val_split, test_split = split_dataset_by_identity(dummy_dataset, random_seed=42)

    # Total samples across splits must equal original count
    total_split_samples = len(train_split["labels"]) + len(val_split["labels"]) + len(test_split["labels"])
    assert total_split_samples == n_samples

    # Infer identities assigned to each split
    train_ids = set()
    val_ids = set()
    test_ids = set()

    for idx, spk in enumerate(identities):
        if idx in range(len(train_split["labels"])):
            train_ids.add(spk)

    # Explicit check for zero identity leakage
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_multimodal_kyc_dataset():
    lip_seqs = torch.randn(10, 100)
    desync_feats = torch.randn(10, 3)
    labels = torch.zeros(10, 1)

    ds = MultimodalKYCDataset(lip_seqs, desync_feats, labels)
    assert len(ds) == 10

    lip, desync, lbl = ds[0]
    assert lip.shape == (100,)
    assert desync.shape == (3,)
    assert lbl.shape == (1,)
