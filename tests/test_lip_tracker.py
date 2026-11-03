"""
SecureKYC AI - Unit and Empirical Integration Tests for 3D Lip Aperture Tracker
Tests LipTracker against real video clips from data/raw/ to verify D_lip(t) trajectory,
mouth-open vs mouth-closed tracking accuracy, and short-gap-only face interpolation constraints.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, LIP_CLOSURE_THRESHOLD_NORMALIZED
from src.vision.lip_tracker import LipTracker, LipFrameResult


def get_real_sample_clips():
    """Returns 2-3 real MP4 video clips from local raw dataset."""
    orig_dir = RAW_DATA_DIR / "original_sequences"
    mp4_files = list(orig_dir.rglob("*.mp4"))
    assert len(mp4_files) >= 2, f"Expected at least 2 real MP4 sample clips in '{orig_dir}'"
    return mp4_files[:3]


def test_lip_tracker_initialization():
    """Verifies proper initialization of LipTracker."""
    tracker = LipTracker()
    assert tracker.face_mesh is not None, "LipTracker face_mesh failed to initialize."
    tracker.close()


def test_lip_tracker_empty_frame():
    """Verifies robust handling of empty/None image frames."""
    tracker = LipTracker()
    res = tracker.process_frame(None, timestamp_ms=0.0)

    assert res.face_detected is False
    assert res.lip_aperture == 0.0
    assert res.is_closed is True
    tracker.close()


def test_3d_distance_formula():
    """Verifies exact Euclidean 3D distance calculation."""
    tracker = LipTracker()
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    dist = tracker.calculate_3d_distance(p1, p2)
    assert dist == pytest.approx(5.0, abs=1e-5)
    tracker.close()


def test_real_clips_mouth_open_closed_tracking():
    """
    Empirical test running LipTracker against real sample clips.
    Verifies that D_lip(t) correctly captures mouth-open and mouth-closed dynamics.
    """
    sample_clips = get_real_sample_clips()
    tracker = LipTracker(max_num_faces=1)

    print("\n" + "=" * 70)
    print(" LipTracker Empirical Verification against Real Dataset Video Clips ")
    print("=" * 70)

    for clip_idx, clip_path in enumerate(sample_clips):
        cap = cv2.VideoCapture(str(clip_path))
        assert cap.isOpened(), f"Could not open test clip: {clip_path}"

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration_ms = (1.0 / fps) * 1000.0

        results = []
        frame_count = 0

        while cap.isOpened() and frame_count < 90:  # Process first 3 seconds
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp_ms = frame_count * frame_duration_ms
            res = tracker.process_frame(frame, timestamp_ms=timestamp_ms)
            if res.face_detected:
                results.append(res)
            frame_count += 1

        cap.release()

        assert len(results) > 10, f"Clip {clip_path.name} failed face detection on majority of frames."

        apertures = [r.lip_aperture for r in results]
        min_ap = min(apertures)
        max_ap = max(apertures)
        mean_ap = np.mean(apertures)

        print(f"[*] Clip {clip_idx + 1}/{len(sample_clips)}: {clip_path.name}")
        print(f"    - Frames Analyzed:  {len(results)}")
        print(f"    - Min D_lip (Closed): {min_ap:.4f}")
        print(f"    - Max D_lip (Open):   {max_ap:.4f}")
        print(f"    - Mean D_lip:         {mean_ap:.4f}")

        # Assert D_lip dynamic range: mouth-open aperture > mouth-closed aperture
        assert max_ap > min_ap + 0.01, f"Clip {clip_path.name} showed no variation between open/closed mouth."
        assert min_ap >= 0.0, "Lip aperture cannot be negative."

    tracker.close()
    print("=" * 70)


def test_gap_constrained_interpolation():
    """
    Verifies that missing face drop gaps <= 3 frames are interpolated,
    whereas gaps > 3 frames flag the clip as unreliable_face_track = True.
    """
    tracker = LipTracker(max_interpolation_gap=3)

    # 1. Short Gap (2 missing frames): Should interpolate smoothly
    short_gap_results = [
        LipFrameResult(frame_timestamp_ms=0, face_detected=True, lip_aperture=0.10, is_closed=False, raw_aperture_px=10, face_height_px=100, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=33, face_detected=False, lip_aperture=0.0, is_closed=True, raw_aperture_px=0, face_height_px=1, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=66, face_detected=False, lip_aperture=0.0, is_closed=True, raw_aperture_px=0, face_height_px=1, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=99, face_detected=True, lip_aperture=0.04, is_closed=True, raw_aperture_px=4, face_height_px=100, processing_latency_ms=1.0),
    ]

    interp_short = tracker.interpolate_missing_apertures(short_gap_results)
    assert interp_short[1].lip_aperture > 0.0, "Short gap frame 1 should be interpolated."
    assert interp_short[2].lip_aperture > 0.0, "Short gap frame 2 should be interpolated."
    assert interp_short[0].unreliable_face_track is False, "Short gap clip should be reliable."

    # 2. Long Gap (4 missing frames): Should NOT interpolate indefinitely and MUST flag clip
    long_gap_results = [
        LipFrameResult(frame_timestamp_ms=0, face_detected=True, lip_aperture=0.10, is_closed=False, raw_aperture_px=10, face_height_px=100, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=33, face_detected=False, lip_aperture=0.0, is_closed=True, raw_aperture_px=0, face_height_px=1, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=66, face_detected=False, lip_aperture=0.0, is_closed=True, raw_aperture_px=0, face_height_px=1, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=99, face_detected=False, lip_aperture=0.0, is_closed=True, raw_aperture_px=0, face_height_px=1, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=132, face_detected=False, lip_aperture=0.0, is_closed=True, raw_aperture_px=0, face_height_px=1, processing_latency_ms=1.0),
        LipFrameResult(frame_timestamp_ms=165, face_detected=True, lip_aperture=0.04, is_closed=True, raw_aperture_px=4, face_height_px=100, processing_latency_ms=1.0),
    ]

    interp_long = tracker.interpolate_missing_apertures(long_gap_results)
    assert all(r.unreliable_face_track for r in interp_long), "Long gap (>3 frames) must flag clip as unreliable_face_track."

    tracker.close()
