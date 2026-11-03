"""
SecureKYC AI - Cross-Modal Phoneme-Viseme Desynchronization Metric
Calculates millisecond lip-aperture distance D_lip(t) desynchronization penalties
during acoustic bilabial events (/b/, /m/, /p/).

Formula:
    Penalty(t) = max(0, min(D_lip over [t-2, t+2] frames) - THRESHOLD)
    S_desync   = mean(Penalty(t) for t in T_bilabial)

Pure physical alignment calculation without artificial synthetic offsets.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

from config import (
    LIP_CLOSURE_THRESHOLD_NORMALIZED,
    DESYNC_PENALTY_WEIGHT,
    DESYNC_TIME_WINDOW_MS,
)
from src.vision.lip_tracker import LipFrameResult
from src.audio.phoneme_extractor import PhonemeInterval, AudioPhonemeResult


@dataclass
class DesyncMetricResult:
    """Evaluation result for cross-modal audio-visual desynchronization."""
    overall_desync_score: float        # S_desync: mean penalty across all bilabial events
    max_penalty_during_bilabials: float# Maximum penalty across T_bilabial
    num_bilabial_events: int           # Total bilabial phonemes evaluated
    failed_closure_count: int          # Count of bilabial events where lips failed to close
    mean_aperture_during_bilabials: float # Average minimum lip aperture during bilabials
    event_scores: List[Dict[str, Any]] # Per-event detailed score breakdowns


class DesynchronizationMetric:
    """
    Cross-Modal Phoneme-Viseme Desynchronization Score Calculator.
    Correlates acoustic bilabial timestamps with temporal 3D lip aperture trajectories
    in [t-2, t+2] frame windows around bilabial events.
    """

    def __init__(
        self,
        closure_threshold: float = LIP_CLOSURE_THRESHOLD_NORMALIZED,
        penalty_weight: float = DESYNC_PENALTY_WEIGHT,
        time_window_ms: float = DESYNC_TIME_WINDOW_MS,
    ):
        self.closure_threshold = closure_threshold
        self.penalty_weight = penalty_weight
        self.time_window_ms = time_window_ms

    def compute_desync_score(
        self,
        lip_trajectories: List[LipFrameResult],
        audio_result: AudioPhonemeResult,
    ) -> DesyncMetricResult:
        """
        Computes cross-modal desynchronization score S_desync between visual lip frames
        and acoustic bilabial phoneme intervals.

        For each acoustic bilabial timestamp t:
            - Inspects D_lip in the window [t-2, t+2] frames.
            - Penalty(t) = max(0, min(D_lip over window) - THRESHOLD)
            - S_desync = mean(Penalty(t) for t in T_bilabial)
        """
        bilabials = audio_result.bilabial_intervals

        if not bilabials or not lip_trajectories:
            return DesyncMetricResult(
                overall_desync_score=0.0,
                max_penalty_during_bilabials=0.0,
                num_bilabial_events=0,
                failed_closure_count=0,
                mean_aperture_during_bilabials=0.0,
                event_scores=[],
            )

        event_scores = []
        failed_closures = 0
        aperture_sum = 0.0

        # Get frame indices with valid face detections
        detected_frame_indices = [i for i, f in enumerate(lip_trajectories) if f.face_detected]

        for interval in bilabials:
            mid_t = (interval.start_time_ms + interval.end_time_ms) / 2.0

            if not detected_frame_indices:
                # If no face detected anywhere in trajectory
                event_scores.append({
                    "phoneme": interval.phoneme,
                    "start_ms": interval.start_time_ms,
                    "end_ms": interval.end_time_ms,
                    "min_aperture": 1.0,
                    "is_closed": False,
                    "penalty": 1.0,
                })
                failed_closures += 1
                continue

            # Find closest frame to middle of bilabial event
            closest_idx = min(detected_frame_indices, key=lambda i: abs(lip_trajectories[i].frame_timestamp_ms - mid_t))

            # Window: [t-2, t+2] frames
            w_start = max(0, closest_idx - 2)
            w_end = min(len(lip_trajectories), closest_idx + 3)

            window_frames = lip_trajectories[w_start:w_end]
            valid_apertures = [f.lip_aperture for f in window_frames if f.face_detected and f.lip_aperture > 0.0]

            if not valid_apertures:
                min_aperture = 1.0
                mean_aperture = 1.0
            else:
                min_aperture = float(np.min(valid_apertures))
                mean_aperture = float(np.mean(valid_apertures))

            aperture_sum += min_aperture
            is_closed = min_aperture <= self.closure_threshold

            # Penalty(t) = max(0, min(D_lip over window) - THRESHOLD)
            penalty = float(max(0.0, min_aperture - self.closure_threshold))

            if not is_closed:
                failed_closures += 1

            event_scores.append({
                "phoneme": interval.phoneme,
                "start_ms": interval.start_time_ms,
                "end_ms": interval.end_time_ms,
                "min_aperture": min_aperture,
                "mean_aperture": mean_aperture,
                "is_closed": is_closed,
                "penalty": penalty,
            })

        total_events = len(bilabials)
        s_desync = float(np.mean([ev["penalty"] for ev in event_scores])) if event_scores else 0.0
        max_penalty = float(np.max([ev["penalty"] for ev in event_scores])) if event_scores else 0.0
        mean_aperture_all = aperture_sum / total_events if total_events > 0 else 0.0

        return DesyncMetricResult(
            overall_desync_score=s_desync,
            max_penalty_during_bilabials=max_penalty,
            num_bilabial_events=total_events,
            failed_closure_count=failed_closures,
            mean_aperture_during_bilabials=mean_aperture_all,
            event_scores=event_scores,
        )


def calibrate_closure_threshold(sample_apertures: List[float] = None, percentile: float = 95.0) -> float:
    """
    Calibrates the physical lip closure threshold D_lip from empirical speech closure distributions.
    Computes the calibrated threshold from authentic bilabial closure aperture samples.
    """
    if not sample_apertures:
        return LIP_CLOSURE_THRESHOLD_NORMALIZED

    arr = np.array(sample_apertures, dtype=np.float32)
    calibrated_val = float(np.percentile(arr, percentile))
    return round(calibrated_val, 4)
