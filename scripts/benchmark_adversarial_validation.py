"""
Adversarial Live Attack Validation Benchmarking Script
(Step B5: Real OBS/DeepFaceLive session clips only)

This script evaluates the trained model against REAL adversarial clips captured
during OBS Virtual Camera + DeepFaceLive live attack sessions (Step B5).

NOTE: generate_adversarial_attack_samples() was deliberately removed. That function
used np.random.uniform to fabricate lip sequences and desync features, which violates
this project's zero-synthetic-data rule (see .agents/AGENTS.md Rule 1). Any metrics
derived from it (the prior "81.82% Accuracy / 0.8952 ROC-AUC" figures) are retracted.

HOW TO USE THIS SCRIPT:
  1. Run a live OBS + DeepFaceLive session following docs/adversarial_testing_setup.md.
  2. Record the attack clips as .mp4 files via OBS (or the WebSocket session recorder).
  3. Run src/models/dataset_extractor.py over those clips with label=1 to extract
     (lip_seq, desync_vec) tensors, saving them to data/processed/adversarial_attacks.pt.
  4. Run this script — it will load that .pt file and pass it through evaluate.py.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, MODEL_CHECKPOINT_DIR, get_device
from src.models.fusion_net import MultimodalKYCFusionNet
from src.models.evaluate import evaluate_model_metrics
from src.models.train import split_dataset_by_identity


def evaluate_real_world_validation_set(
    adv_path: Path = PROCESSED_DATA_DIR / "adversarial_attacks.pt",
) -> Dict[str, Any]:
    """
    Evaluates the trained model on the held-out genuine test split combined with
    REAL live-attack clips captured via OBS + DeepFaceLive (Step B5).

    Requires data/processed/adversarial_attacks.pt to have been produced by
    running dataset_extractor.py over actual recorded face-swap session clips.
    Raises FileNotFoundError if the file is missing rather than generating
    synthetic replacements.
    """
    device = get_device()
    dataset_path = PROCESSED_DATA_DIR / "dataset.pt"
    ckpt_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"

    if not adv_path.exists():
        raise FileNotFoundError(
            f"\n[!] {adv_path} not found.\n"
            "    Real adversarial validation is pending until Step B5's OBS/DeepFaceLive\n"
            "    sessions are recorded and fed through dataset_extractor.py.\n"
            "    Do NOT run generate_adversarial_attack_samples() — that function was\n"
            "    removed because it violated the zero-synthetic-data rule."
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"[!] Model checkpoint not found at {ckpt_path}.")

    print("\n" + "=" * 92)
    print(" Real-World Validation Set Evaluation (Genuine + Live OBS/DeepFaceLive Attacks) ")
    print("=" * 92)

    gen_data = torch.load(dataset_path, map_location="cpu")
    adv_data = torch.load(adv_path, map_location="cpu")

    _, _, test_dict = split_dataset_by_identity(gen_data)

    comb_lip    = torch.cat([test_dict["lip_sequences"],  adv_data["lip_sequences"]],  dim=0)
    comb_desync = torch.cat([test_dict["desync_features"], adv_data["desync_features"]], dim=0)
    comb_labels = torch.cat([test_dict["labels"],          adv_data["labels"]],          dim=0)

    n_gen = len(test_dict["labels"])
    n_adv = len(adv_data["labels"])
    print(f"[*] {n_gen} genuine test clips + {n_adv} real adversarial attack clips = {len(comb_labels)} total")

    model = MultimodalKYCFusionNet(seq_len=comb_lip.shape[1], desync_feature_dim=comb_desync.shape[1])
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[✓] Loaded checkpoint from {ckpt_path}")

    metrics = evaluate_model_metrics(model, comb_lip, comb_desync, comb_labels, device)

    print(f"\n{'Dataset / Split':<38} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1':^8} | {'ROC-AUC':^9} | {'EER (%)':^9}")
    print("-" * 100)
    print(
        f"{'Real-World (Genuine + OBS Attack clips)':<38} | "
        f"{metrics['accuracy']*100:^9.2f}% | "
        f"{metrics['precision']*100:^9.2f}% | "
        f"{metrics['recall']*100:^9.2f}% | "
        f"{metrics['f1']:^8.4f} | "
        f"{metrics['roc_auc']:^9.4f} | "
        f"{metrics['eer']*100:^9.2f}%"
    )
    print("=" * 100 + "\n")

    return metrics


if __name__ == "__main__":
    evaluate_real_world_validation_set()
