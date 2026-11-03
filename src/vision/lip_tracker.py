"""
SecureKYC AI - 3D Lip Aperture Tracker Module
Extracts 3D facial landmarks via MediaPipe FaceMesh / FaceLandmarker and computes
normalized 3D lip aperture distance D_lip(t):

    D_lip(t) = ||P13 - P14||_2 / (||P1 - P152||_2 + 1e-6)

Where:
- P13:  Upper inner lip center landmark (#13)
- P14:  Lower inner lip center landmark (#14)
- P1:   Nose tip landmark (#1)
- P152: Chin landmark (#152)

Includes gap-constrained linear interpolation for dropped face detections (gaps <= 3 frames).
Clips with face drop gaps > 3 consecutive frames are flagged with unreliable_face_track = True.
"""

import sys
import time
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LANDMARK_UPPER_LIP_INNER,  # 13
    LANDMARK_LOWER_LIP_INNER,  # 14
    LIP_CLOSURE_THRESHOLD_NORMALIZED,
    MODEL_CHECKPOINT_DIR,
)

# Landmark Indices for 3D Lip Aperture and Normalization Anchors
LANDMARK_NOSE_TIP = 1
LANDMARK_CHIN = 152


@dataclass
class LipFrameResult:
    """Dataclass holding extracted 3D lip aperture metrics for a single frame."""
    frame_timestamp_ms: float
    face_detected: bool
    lip_aperture: float              # D_lip(t) normalized distance
    is_closed: bool                  # True if lip aperture < threshold
    raw_aperture_px: float           # ||P13 - P14||_2
    face_height_px: float            # ||P1 - P152||_2
    processing_latency_ms: float      # Latency in milliseconds
    unreliable_face_track: bool = False  # Set to True if missing gap > 3 frames
    landmarks_3d: Optional[np.ndarray] = None


class LipTracker:
    """
    Real-time 3D facial landmark extractor and normalized lip aperture calculator.
    Processes frames in real time and enforces short-gap-only face drop interpolation.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        closure_threshold: float = LIP_CLOSURE_THRESHOLD_NORMALIZED,
        max_interpolation_gap: int = 3,
    ):
        self.closure_threshold = closure_threshold
        self.max_interpolation_gap = max_interpolation_gap
        self.api_type = None

        # 1. Attempt legacy mp.solutions API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=static_image_mode,
                    max_num_faces=max_num_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.api_type = "solutions"
            except Exception as e:
                logger.exception("Failed to initialize legacy MediaPipe solutions.face_mesh: %s", e)
                self.face_mesh = None

        # 2. Attempt modern mediapipe.tasks API (MediaPipe 1.0+)
        if self.api_type is None:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                task_model_path = MODEL_CHECKPOINT_DIR / "face_landmarker.task"
                MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

                if not task_model_path.exists():
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    print(f"[*] Downloading face_landmarker.task to {task_model_path}...")
                    import ssl
                    ctx = ssl._create_unverified_context()
                    with urllib.request.urlopen(url, context=ctx) as response, open(task_model_path, "wb") as out_file:
                        out_file.write(response.read())
                    print("[✓] Downloaded face_landmarker.task successfully.")

                base_options = python.BaseOptions(model_asset_path=str(task_model_path))
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence,
                )
                self.face_mesh = vision.FaceLandmarker.create_from_options(options)
                self.api_type = "tasks"
            except Exception as err:
                logger.exception("Failed to initialize modern MediaPipe tasks FaceLandmarker: %s", err)
                self.face_mesh = None
                self.api_type = None

        if self.face_mesh is None:
            raise RuntimeError(
                "MediaPipe initialization failed: neither solutions.face_mesh nor tasks.FaceLandmarker could be initialized."
            )

    def calculate_3d_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Computes Euclidean 3D distance between two landmark points: ||p1 - p2||_2."""
        return float(np.linalg.norm(p1 - p2))

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float = 0.0,
    ) -> LipFrameResult:
        """
        Processes an BGR/RGB frame to extract 3D lip aperture D_lip(t).

        Returns:
            LipFrameResult containing D_lip(t), raw distance, scale anchor, and closure state.
        """
        start_time = time.perf_counter()

        if self.face_mesh is None:
            raise RuntimeError("LipTracker cannot process frame: MediaPipe face_mesh is not initialized or has been closed.")

        if frame is None or frame.size == 0:
            latency = (time.perf_counter() - start_time) * 1000.0
            return LipFrameResult(
                frame_timestamp_ms=timestamp_ms,
                face_detected=False,
                lip_aperture=0.0,
                is_closed=True,
                raw_aperture_px=0.0,
                face_height_px=1.0,
                processing_latency_ms=latency,
            )

        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        h, w, _ = frame.shape
        face_landmarks = None

        if self.api_type == "solutions":
            results = self.face_mesh.process(rgb_frame)
            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
        elif self.api_type == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.face_mesh.detect(mp_image)
            if results and results.face_landmarks:
                face_landmarks = results.face_landmarks[0]

        if not face_landmarks:
            latency = (time.perf_counter() - start_time) * 1000.0
            return LipFrameResult(
                frame_timestamp_ms=timestamp_ms,
                face_detected=False,
                lip_aperture=0.0,
                is_closed=True,
                raw_aperture_px=0.0,
                face_height_px=1.0,
                processing_latency_ms=latency,
            )

        # Convert to 3D coordinate numpy array (x * w, y * h, z * w)
        coords = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks]
        )

        p13_upper = coords[LANDMARK_UPPER_LIP_INNER]  # Upper inner lip (#13)
        p14_lower = coords[LANDMARK_LOWER_LIP_INNER]  # Lower inner lip (#14)
        p1_nose = coords[LANDMARK_NOSE_TIP]            # Nose tip (#1)
        p152_chin = coords[LANDMARK_CHIN]              # Chin (#152)

        raw_aperture = self.calculate_3d_distance(p13_upper, p14_lower)
        face_height = self.calculate_3d_distance(p1_nose, p152_chin)

        # D_lip(t) = ||P13 - P14||_2 / (||P1 - P152||_2 + 1e-6)
        normalized_aperture = raw_aperture / (face_height + 1e-6)
        is_closed = normalized_aperture < self.closure_threshold
        latency = (time.perf_counter() - start_time) * 1000.0

        return LipFrameResult(
            frame_timestamp_ms=timestamp_ms,
            face_detected=True,
            lip_aperture=normalized_aperture,
            is_closed=is_closed,
            raw_aperture_px=raw_aperture,
            face_height_px=face_height,
            processing_latency_ms=latency,
            landmarks_3d=coords,
        )

    def interpolate_missing_apertures(
        self, lip_results: List[LipFrameResult]
    ) -> List[LipFrameResult]:
        """
        Linear interpolation for missing face detection frames.
        Only interpolates short gaps (<= max_interpolation_gap consecutive frames).
        For longer gaps (> max_interpolation_gap), flags unreliable_face_track = True.
        """
        if not lip_results:
            return lip_results

        n = len(lip_results)
        i = 0
        has_long_gap = False

        # Identify missing frame runs
        while i < n:
            if not lip_results[i].face_detected:
                gap_start = i
                while i < n and not lip_results[i].face_detected:
                    i += 1
                gap_end = i  # Exclusive
                gap_length = gap_end - gap_start

                if gap_length <= self.max_interpolation_gap:
                    # Short gap: interpolate if bounded by valid detections
                    prev_idx = gap_start - 1 if gap_start > 0 else None
                    next_idx = gap_end if gap_end < n else None

                    if prev_idx is not None and lip_results[prev_idx].face_detected and next_idx is not None and lip_results[next_idx].face_detected:
                        val_start = lip_results[prev_idx].lip_aperture
                        val_end = lip_results[next_idx].lip_aperture
                        for step_i, g_idx in enumerate(range(gap_start, gap_end)):
                            alpha = (step_i + 1) / (gap_length + 1)
                            interp_val = val_start + alpha * (val_end - val_start)
                            lip_results[g_idx].lip_aperture = interp_val
                            lip_results[g_idx].is_closed = interp_val < self.closure_threshold
                    elif prev_idx is not None and lip_results[prev_idx].face_detected:
                        val = lip_results[prev_idx].lip_aperture
                        for g_idx in range(gap_start, gap_end):
                            lip_results[g_idx].lip_aperture = val
                            lip_results[g_idx].is_closed = val < self.closure_threshold
                    elif next_idx is not None and lip_results[next_idx].face_detected:
                        val = lip_results[next_idx].lip_aperture
                        for g_idx in range(gap_start, gap_end):
                            lip_results[g_idx].lip_aperture = val
                            lip_results[g_idx].is_closed = val < self.closure_threshold
                else:
                    # Long gap (> max_interpolation_gap frames): flag clip as unreliable
                    has_long_gap = True
            else:
                i += 1

        if has_long_gap:
            for r in lip_results:
                r.unreliable_face_track = True

        return lip_results

    def close(self):
        """Release MediaPipe resources."""
        if self.face_mesh is not None and hasattr(self.face_mesh, "close"):
            self.face_mesh.close()
