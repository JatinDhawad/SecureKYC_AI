"""
Unit tests for src/metrics/desync_metric.py
Verifies cross-modal desynchronization penalty logic using the [t-2, t+2] frame window
and empirical threshold under real vs fake audio-visual streams.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.lip_tracker import LipFrameResult
from src.audio.phoneme_extractor import PhonemeInterval, AudioPhonemeResult
from src.metrics.desync_metric import DesynchronizationMetric, DesyncMetricResult
from config import LIP_CLOSURE_THRESHOLD_NORMALIZED


def test_desync_metric_empty_inputs():
    metric = DesynchronizationMetric()
    result = metric.compute_desync_score(
        lip_trajectories=[],
        audio_result=AudioPhonemeResult(audio_duration_ms=1000.0),
    )

    assert isinstance(result, DesyncMetricResult)
    assert result.overall_desync_score == 0.0
    assert result.num_bilabial_events == 0
    assert result.failed_closure_count == 0


def test_desync_metric_aligned_authentic_stream():
    """
    Verifies that when lips close within [t-2, t+2] window of acoustic bilabial event,
    Penalty(t) = max(0, min(D_lip) - THRESHOLD) evaluates to 0.0 (no penalty).
    """
    threshold = LIP_CLOSURE_THRESHOLD_NORMALIZED
    metric = DesynchronizationMetric(closure_threshold=threshold)

    # Audio bilabial event /M/ at 200ms
    audio_res = AudioPhonemeResult(
        audio_duration_ms=1000.0,
        bilabial_intervals=[
            PhonemeInterval(phoneme="M", start_time_ms=200.0, end_time_ms=280.0, is_bilabial=True),
        ],
    )

    # Visual stream with frames at 40ms intervals (25 FPS). Frame at t=240ms has aperture 0.015 < threshold (0.0242)
    lip_frames = [
        LipFrameResult(frame_timestamp_ms=160.0, face_detected=True, lip_aperture=0.18, is_closed=False, raw_aperture_px=18.0, face_height_px=100.0, processing_latency_ms=2.0),
        LipFrameResult(frame_timestamp_ms=200.0, face_detected=True, lip_aperture=0.08, is_closed=False, raw_aperture_px=8.0, face_height_px=100.0, processing_latency_ms=2.0),
        LipFrameResult(frame_timestamp_ms=240.0, face_detected=True, lip_aperture=0.015, is_closed=True, raw_aperture_px=1.5, face_height_px=100.0, processing_latency_ms=2.0),  # Closest frame
        LipFrameResult(frame_timestamp_ms=280.0, face_detected=True, lip_aperture=0.09, is_closed=False, raw_aperture_px=9.0, face_height_px=100.0, processing_latency_ms=2.0),
        LipFrameResult(frame_timestamp_ms=320.0, face_detected=True, lip_aperture=0.20, is_closed=False, raw_aperture_px=20.0, face_height_px=100.0, processing_latency_ms=2.0),
    ]

    result = metric.compute_desync_score(lip_frames, audio_res)

    assert result.num_bilabial_events == 1
    assert result.failed_closure_count == 0
    assert result.overall_desync_score == 0.0  # min(D_lip) = 0.015 <= 0.0242 -> penalty = 0.0


def test_desync_metric_desynced_deepfake_stream():
    """
    Verifies that when lips fail to close in [t-2, t+2] window of bilabial event,
    Penalty(t) = max(0, min(D_lip) - THRESHOLD) evaluates to > 0.0.
    """
    threshold = LIP_CLOSURE_THRESHOLD_NORMALIZED
    metric = DesynchronizationMetric(closure_threshold=threshold)

    # Audio bilabial event /B/ at 200ms
    audio_res = AudioPhonemeResult(
        audio_duration_ms=1000.0,
        bilabial_intervals=[
            PhonemeInterval(phoneme="B", start_time_ms=200.0, end_time_ms=280.0, is_bilabial=True),
        ],
    )

    # Deepfake video stream where minimum lip aperture in [t-2, t+2] window is 0.25 > threshold
    lip_frames = [
        LipFrameResult(frame_timestamp_ms=160.0, face_detected=True, lip_aperture=0.30, is_closed=False, raw_aperture_px=30.0, face_height_px=100.0, processing_latency_ms=2.0),
        LipFrameResult(frame_timestamp_ms=200.0, face_detected=True, lip_aperture=0.28, is_closed=False, raw_aperture_px=28.0, face_height_px=100.0, processing_latency_ms=2.0),
        LipFrameResult(frame_timestamp_ms=240.0, face_detected=True, lip_aperture=0.25, is_closed=False, raw_aperture_px=25.0, face_height_px=100.0, processing_latency_ms=2.0),  # Min aperture in window
        LipFrameResult(frame_timestamp_ms=280.0, face_detected=True, lip_aperture=0.29, is_closed=False, raw_aperture_px=29.0, face_height_px=100.0, processing_latency_ms=2.0),
        LipFrameResult(frame_timestamp_ms=320.0, face_detected=True, lip_aperture=0.32, is_closed=False, raw_aperture_px=32.0, face_height_px=100.0, processing_latency_ms=2.0),
    ]

    result = metric.compute_desync_score(lip_frames, audio_res)

    expected_penalty = max(0.0, 0.25 - threshold)

    assert result.num_bilabial_events == 1
    assert result.failed_closure_count == 1
    assert result.overall_desync_score == pytest.approx(expected_penalty, abs=1e-4)
    assert result.overall_desync_score > 0.0
