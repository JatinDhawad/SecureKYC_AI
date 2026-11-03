"""
Unit tests for dataset directory validation and loud placeholder error reporting.
Verifies VidTIMIT and Celeb-DF presence and checks loud failure on empty FaceForensics++ placeholder.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.models.dataset_extractor import VidTIMITDatasetExtractor


def test_real_dataset_directory_validation():
    """Verifies that actual data/raw directory validates VidTIMIT and Celeb-DF v2."""
    extractor = VidTIMITDatasetExtractor(raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)
    status = extractor.validate_dataset_directories(require_faceforensics=False)

    assert status["vidtimit_present"] is True, "VidTIMIT dataset should be detected in data/raw."
    assert status["celeb_df_present"] is True, "Celeb-DF v2 dataset should be detected in data/raw."


def test_faceforensics_empty_placeholder_raises_loud_error():
    """Asserts that require_faceforensics=True raises an instructional FileNotFoundError if empty."""
    extractor = VidTIMITDatasetExtractor(raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR)
    with pytest.raises(FileNotFoundError) as exc_info:
        extractor.validate_dataset_directories(require_faceforensics=True)

    err_msg = str(exc_info.value)
    assert "FaceForensics++ dataset placeholder empty" in err_msg
    assert "github.com/ondyari/FaceForensics" in err_msg


def test_missing_vidtimit_raises_instructional_error(tmp_path):
    """Asserts missing VidTIMIT raises instructional error pointing to download script."""
    extractor = VidTIMITDatasetExtractor(raw_dir=tmp_path, processed_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        extractor.validate_dataset_directories()

    err_msg = str(exc_info.value)
    assert "VidTIMIT dataset missing" in err_msg
    assert "download_vidtimit.py" in err_msg
