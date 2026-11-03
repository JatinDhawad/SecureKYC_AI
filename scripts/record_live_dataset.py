"""
SecureKYC AI - Live Dataset Recording & Packaging Utility
Automates the full 8-subject live session recording protocol (OBS Studio + DeepFaceLive).
Guides through:
  1. Setting up OBS Virtual Camera & Audio.
  2. Recording Genuine vs. Live-Swapped sessions across 8 subjects (S1 to S8).
  3. Extracting 3D lip trajectories and forced-alignment bilabials.
  4. Storing to data/raw/live_sessions/ and data/processed/live_dataset_8subj.pt.
"""

import os
import sys
import time
import json
import wave
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, AUDIO_SAMPLE_RATE, get_device
from app.prompt_generator import prompt_generator
from src.vision.lip_tracker import LipTracker
from src.audio.phoneme_extractor import PhonemeExtractor
from src.metrics.desync_metric import DesynchronizationMetric

LIVE_RAW_DIR = RAW_DATA_DIR / "live_sessions"
LIVE_RAW_DIR.mkdir(parents=True, exist_ok=True)

# 8 Consented Subject Identities & Assigned Split Components
SUBJECT_CONFIG = {
    "S1": {"split": "train", "swap_target": "S2", "sessions": 3},
    "S2": {"split": "train", "swap_target": "S1", "sessions": 3},
    "S3": {"split": "train", "swap_target": "S4", "sessions": 3},
    "S4": {"split": "train", "swap_target": "S5", "sessions": 3},
    "S5": {"split": "train", "swap_target": "S3", "sessions": 3},
    "S6": {"split": "val",   "swap_target": "S6", "sessions": 3},
    "S7": {"split": "test",  "swap_target": "S8", "sessions": 3},
    "S8": {"split": "test",  "swap_target": "S7", "sessions": 3},
}


def print_banner():
    print("\n" + "=" * 80)
    print("        SecureKYC AI - 8-Subject Live Dataset Recording Manager        ")
    print("=" * 80)
    print("  • Hardware Target: OBS Virtual Camera -> PyTorch (mps)")
    print("  • Total Sessions: 48 sessions (24 Genuine + 24 DeepFaceLive Swaps)")
    print("  • Splits: Train (S1-S5, 30 clips) | Val (S6, 6 clips) | Test (S7-S8, 12 clips)")
    print("=" * 80 + "\n")


def list_camera_devices():
    """Lists available camera device indices."""
    available = []
    for idx in range(6):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(idx)
            cap.release()
    return available


def print_subject_plan():
    print(f"{'Subject':<10} | {'Split':<8} | {'Swap Target Model':<20} | {'Genuine Clips':<14} | {'Live-Swap Clips':<15}")
    print("-" * 75)
    for subj_id, cfg in SUBJECT_CONFIG.items():
        print(f"{subj_id:<10} | {cfg['split']:<8} | Target: M_{cfg['swap_target']:<12} | {cfg['sessions']} sessions    | {cfg['sessions']} sessions")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    print_banner()
    print_subject_plan()
    cams = list_camera_devices()
    print(f"[*] Detected {len(cams)} active camera index/devices: {cams}")
    print("    (In OBS, click 'Start Virtual Camera' to expose OBS as an active device).\n")
