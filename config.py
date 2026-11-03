"""
SecureKYC AI - Global Configuration Module
Dynamic Bilabial-Targeted Active Prompting for Real-Time Deepfake Injection Defense
"""

import os
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PAPER_DIR = BASE_DIR / "paper"
FAKEAVCELEB_RAW_DIR = RAW_DATA_DIR / "FakeAVCeleb_v1.2"
LIVE_SESSIONS_RAW_DIR = RAW_DATA_DIR / "live_sessions"



def get_device():
    """
    Select optimal compute backend for PyTorch.
    Prioritizes Apple Silicon Metal Acceleration ('mps'), then CUDA, then CPU.
    """
    if not HAS_TORCH:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# Hardware & Latency Settings
TARGET_FPS = 30
MAX_FRAME_LATENCY_MS = 33.3  # Target real-time streaming budget per frame
AUDIO_SAMPLE_RATE = 16000     # Required by Wav2Vec2 backbone
FIXED_SEQ_LEN = 100           # Fixed length for input lip aperture sequence tensors

# MediaPipe 3D FaceMesh Landmark Indices
# Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LANDMARK_UPPER_LIP_INNER = 13
LANDMARK_LOWER_LIP_INNER = 14
LANDMARK_UPPER_LIP_OUTER = 0
LANDMARK_LOWER_LIP_OUTER = 17

# Normalization Scale Anchors (Forehead to Chin, Left to Right Cheek)
LANDMARK_FACE_TOP = 10
LANDMARK_FACE_BOTTOM = 152
LANDMARK_FACE_LEFT = 234
LANDMARK_FACE_RIGHT = 454

# Lip Aperture Thresholds
# Empirically calibrated from 95th percentile of genuine VidTIMIT bilabial closure apertures (N=320 across 15 speakers)
# Formula: Penalty(t) = max(0, min(D_lip over [t-2, t+2] frames) - THRESHOLD)
LIP_CLOSURE_THRESHOLD_NORMALIZED = 0.0242  # Calibrated threshold (95th percentile)

# Wav2Vec2 & Audio Phoneme Settings
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
BILABIAL_PHONEMES = {"M", "B", "P"}  # Plosive & nasal bilabials forcing lip closure

# Active Bilabial Challenge Prompts
ACTIVE_PROMPT_POOL = [
    "My Bank Passcode",
    "Bumble Bee Password",
    "Mobile Payment Protocol",
    "Prime Member Pass",
    "Big Blue Map",
]

# Desynchronization Metric Parameters
DESYNC_TIME_WINDOW_MS = 100  # Analysis window surrounding detected bilabial timestamp
DESYNC_PENALTY_WEIGHT = 1.5

# Model Training Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

