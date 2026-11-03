"""
Unit tests verifying explicit error handling and failure logging across:
- src/vision/lip_tracker.py (MediaPipe initialization failure)
- src/audio/phoneme_extractor.py (MMS_FA load failure & audio extraction failure)
- app/main.py (checkpoint load failure & malformed frame decode failure)
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.lip_tracker import LipTracker
from src.audio.phoneme_extractor import PhonemeExtractor
from app.main import decode_b64_image, load_model_checkpoints
import app.main as app_main


def test_lip_tracker_init_failure_raises_runtime_error():
    """Verifies that LipTracker raises RuntimeError when MediaPipe fails to initialize."""
    with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", side_effect=Exception("MediaPipe tasks init error")):
        with pytest.raises(RuntimeError, match="MediaPipe initialization failed"):
            LipTracker()


def test_lip_tracker_process_frame_uninitialized_raises():
    """Verifies that LipTracker.process_frame raises RuntimeError if face_mesh is None."""
    tracker = LipTracker()
    tracker.face_mesh = None
    with pytest.raises(RuntimeError, match="not initialized or has been closed"):
        tracker.process_frame(np.zeros((100, 100, 3), dtype=np.uint8))


def test_phoneme_extractor_mms_fa_load_failure_raises():
    """Verifies that PhonemeExtractor raises RuntimeError when MMS_FA bundle fails to load."""
    extractor = PhonemeExtractor(lazy_load=True)
    with patch("torchaudio.pipelines.MMS_FA.get_model", side_effect=Exception("Bundle download error")):
        with pytest.raises(RuntimeError, match="Failed to load torchaudio MMS_FA"):
            extractor._load_forced_alignment_model()


def test_phoneme_extractor_extract_audio_corrupt_file_raises(tmp_path):
    """Verifies that extract_audio_from_video raises RuntimeError on corrupt audio files."""
    extractor = PhonemeExtractor(lazy_load=True)
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"NOT_A_VALID_MP4_OR_AV_CONTAINER_DATA")

    with pytest.raises(RuntimeError, match="Audio extraction failed"):
        extractor.extract_audio_from_video(corrupt_file)


def test_app_checkpoint_missing_raises_fatal_startup(tmp_path):
    """Verifies that app.main raises fatal startup error if checkpoint does not exist."""
    with patch("app.main.MODEL_CHECKPOINT_DIR", tmp_path), \
         patch("app.main.fusion_net", None):
        with pytest.raises(FileNotFoundError, match="Fatal startup failure"):
            load_model_checkpoints()


def test_app_checkpoint_corrupt_raises_fatal_startup(tmp_path):
    """Verifies that app.main raises fatal startup error if checkpoint file is corrupt."""
    bad_ckpt = tmp_path / "best_fusion_model.pt"
    bad_ckpt.write_bytes(b"CORRUPTED_TORCH_CHECKPOINT")

    with patch("app.main.MODEL_CHECKPOINT_DIR", tmp_path), \
         patch("app.main.fusion_net", None):
        with pytest.raises(RuntimeError, match="Fatal startup failure"):
            load_model_checkpoints()


def test_decode_b64_image_malformed_raises_value_error():
    """Verifies that decode_b64_image raises ValueError on malformed or undecodable base64 string."""
    with pytest.raises(ValueError, match="Malformed frame decode failure"):
        decode_b64_image("invalid_base64_!@#$%^&*()")

    with pytest.raises(ValueError, match="Malformed frame decode failure"):
        # Valid base64 encoding of random non-image bytes
        import base64
        garbage_b64 = base64.b64encode(b"this is not an image").decode("utf-8")
        decode_b64_image(garbage_b64)
