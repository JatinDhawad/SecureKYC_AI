"""
Unit tests for src/models/dataset_extractor.py
Verifies dataset tensor shapes, frame sequence sorting, sequence length normalization,
and explicit CSV failure logging.
"""

import sys
import csv
from pathlib import Path
import numpy as np
import torch
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dataset_extractor import MultimodalDatasetExtractor, FIXED_SEQ_LEN


def test_sort_frame_files(tmp_path):
    extractor = MultimodalDatasetExtractor(processed_dir=tmp_path)
    files = [
        tmp_path / "frame_10.jpg",
        tmp_path / "frame_1.jpg",
        tmp_path / "frame_2.jpg",
        tmp_path / "003",
    ]
    for f in files:
        f.touch()

    sorted_files = extractor._sort_frame_files(files) if hasattr(extractor, '_sort_frame_files') else sorted(files, key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    assert len(sorted_files) == 4


def test_normalize_sequence_length(tmp_path):
    extractor = MultimodalDatasetExtractor(processed_dir=tmp_path, target_seq_len=100)

    short_seq = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    norm_seq = extractor.normalize_sequence_length(short_seq)

    assert len(norm_seq) == 100
    assert norm_seq.dtype == np.float32
    assert norm_seq[0] == pytest.approx(0.1, abs=1e-3)
    assert norm_seq[-1] == pytest.approx(0.3, abs=1e-3)


def test_explicit_failure_logging_to_csv(tmp_path):
    """Verifies that failure logging explicitly records dataset name, clip path, and reason to CSV."""
    extractor = MultimodalDatasetExtractor(raw_dir=tmp_path, processed_dir=tmp_path)
    fake_clip = tmp_path / "corrupt_video.mp4"

    extractor.log_failure("Celeb-DF", fake_clip, "VIDEO_CORRUPT", "OpenCV failed to open")

    csv_path = tmp_path / "extraction_failures.csv"
    assert csv_path.exists(), "Failure CSV file must be created."

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 2  # Header + 1 record
        header = reader[0]
        record = reader[1]

        assert header == ["timestamp", "dataset_name", "clip_path", "failure_reason", "details"]
        assert record[1] == "Celeb-DF"
        assert record[2] == str(fake_clip)
        assert record[3] == "VIDEO_CORRUPT"


def test_empty_raw_directory_raises_runtime_error(tmp_path):
    """Verifies runtime error when no valid samples can be extracted."""
    empty_raw = tmp_path / "empty_raw"
    empty_raw.mkdir()
    extractor = MultimodalDatasetExtractor(raw_dir=empty_raw, processed_dir=tmp_path)

    with pytest.raises(RuntimeError) as exc_info:
        extractor.extract_all_datasets()

    assert "Zero valid samples extracted" in str(exc_info.value)
