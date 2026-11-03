"""
Script: scripts/extract_live_sessions_pt.py
Extracts 3D lip aperture sequences, desynchronization feature vectors, and identity labels
from all recorded genuine and live-swap sessions (S1 to S8).
Saves to data/processed/live_dataset_8subj.pt.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIXED_SEQ_LEN, LIP_CLOSURE_THRESHOLD_NORMALIZED
from src.vision.lip_tracker import LipTracker
from src.audio.phoneme_extractor import PhonemeExtractor
from src.metrics.desync_metric import DesynchronizationMetric

LIVE_DIR = RAW_DATA_DIR / "live_sessions"
OUT_PT = PROCESSED_DATA_DIR / "live_dataset_8subj.pt"

PROMPTS = {
    1: "My Bank Passcode",
    2: "Big Apple Mobile Bank",
    3: "Public Payment Passbook",
}


def normalize_seq(seq: np.ndarray, target_len: int = FIXED_SEQ_LEN) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros(target_len, dtype=np.float32)
    x_orig = np.linspace(0, 1, len(seq))
    x_target = np.linspace(0, 1, target_len)
    return np.interp(x_target, x_orig, seq).astype(np.float32)


def find_video_path(dir_path: Path, session_num: int):
    for ext in [".mov", ".mp4", ".mkv", ".MOV", ".MP4", ".MKV"]:
        p = dir_path / f"session_{session_num}{ext}"
        if p.exists():
            return p
    return None


def main():
    print("=" * 80)
    print("      Extracting Features for Live Dataset S1-S8 (48 Sessions)        ")
    print(f"      Calibrated Lip Closure Threshold: {LIP_CLOSURE_THRESHOLD_NORMALIZED} (0.0242)      ")
    print("=" * 80)

    tracker = LipTracker()
    phoneme_ext = PhonemeExtractor(lazy_load=False)
    metric = DesynchronizationMetric(closure_threshold=LIP_CLOSURE_THRESHOLD_NORMALIZED)

    print(f"[*] DesynchronizationMetric threshold initialized to: {metric.closure_threshold}")

    dataset_samples = []

    for s_idx in range(1, 9):
        subj_id = f"S{s_idx}"
        for session_num in [1, 2, 3]:
            prompt = PROMPTS[session_num]

            # 1. Genuine Clip (Label = 0.0)
            p_gen = find_video_path(LIVE_DIR / subj_id / "genuine", session_num)
            if p_gen is not None and p_gen.exists():
                wav, sr = phoneme_ext.extract_audio_from_video(p_gen)
                audio_res = phoneme_ext.align_forced_transcript(wav, prompt_text=prompt, sample_rate=sr)

                cap = cv2.VideoCapture(str(p_gen))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_dur = (1.0 / fps) * 1000.0
                lip_results = []
                idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    lip_results.append(tracker.process_frame(frame, timestamp_ms=idx * frame_dur))
                    idx += 1
                cap.release()

                lip_results = tracker.interpolate_missing_apertures(lip_results)
                desync_res = metric.compute_desync_score(lip_results, audio_res)
                raw_lip_seq = np.array([r.lip_aperture for r in lip_results], dtype=np.float32)
                norm_lip_seq = normalize_seq(raw_lip_seq)

                desync_score = float(desync_res.overall_desync_score)
                max_penalty = float(desync_res.max_penalty_during_bilabials)
                num_b = desync_res.num_bilabial_events
                failed_ratio = (desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0

                desync_tensor = torch.tensor([desync_score, max_penalty, failed_ratio], dtype=torch.float32)
                label_tensor = torch.tensor([0.0], dtype=torch.float32)

                bilabial_chars = [i.phoneme for i in audio_res.bilabial_intervals]

                sample = {
                    "clip_name": f"{subj_id}_genuine_take{session_num}",
                    "source_dataset": "LiveKYC_Consented",
                    "identity": subj_id,
                    "label": label_tensor,
                    "lip_sequence": torch.tensor(norm_lip_seq, dtype=torch.float32),
                    "desync_features": desync_tensor,
                    "num_frames": len(lip_results),
                    "desync_score": desync_score,
                    "failed_closures": int(desync_res.failed_closure_count),
                    "total_bilabials": int(num_b),
                    "prompt": prompt,
                    "bilabials": bilabial_chars,
                }
                dataset_samples.append(sample)
                print(f"  [✓] {sample['clip_name']}: Prompt='{prompt}', K={num_b} ({bilabial_chars}), S_desync={sample['desync_score']:.6f}, Failed={sample['failed_closures']}")

            # 2. Live-Swap Clip (Label = 1.0)
            p_swap = find_video_path(LIVE_DIR / subj_id / "swap", session_num)
            if p_swap is not None and p_swap.exists():
                wav, sr = phoneme_ext.extract_audio_from_video(p_swap)
                audio_res = phoneme_ext.align_forced_transcript(wav, prompt_text=prompt, sample_rate=sr)

                cap = cv2.VideoCapture(str(p_swap))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_dur = (1.0 / fps) * 1000.0
                lip_results = []
                idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    lip_results.append(tracker.process_frame(frame, timestamp_ms=idx * frame_dur))
                    idx += 1
                cap.release()

                lip_results = tracker.interpolate_missing_apertures(lip_results)
                desync_res = metric.compute_desync_score(lip_results, audio_res)
                raw_lip_seq = np.array([r.lip_aperture for r in lip_results], dtype=np.float32)
                norm_lip_seq = normalize_seq(raw_lip_seq)

                desync_score = float(desync_res.overall_desync_score)
                max_penalty = float(desync_res.max_penalty_during_bilabials)
                num_b = desync_res.num_bilabial_events
                failed_ratio = (desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0

                desync_tensor = torch.tensor([desync_score, max_penalty, failed_ratio], dtype=torch.float32)
                label_tensor = torch.tensor([1.0], dtype=torch.float32)

                bilabial_chars = [i.phoneme for i in audio_res.bilabial_intervals]

                sample = {
                    "clip_name": f"{subj_id}_swap_take{session_num}",
                    "source_dataset": "LiveKYC_Consented",
                    "identity": subj_id,
                    "label": label_tensor,
                    "lip_sequence": torch.tensor(norm_lip_seq, dtype=torch.float32),
                    "desync_features": desync_tensor,
                    "num_frames": len(lip_results),
                    "desync_score": desync_score,
                    "failed_closures": int(desync_res.failed_closure_count),
                    "total_bilabials": int(num_b),
                    "prompt": prompt,
                    "bilabials": bilabial_chars,
                }
                dataset_samples.append(sample)
                print(f"  [✓] {sample['clip_name']}: Prompt='{prompt}', K={num_b} ({bilabial_chars}), S_desync={sample['desync_score']:.6f}, Failed={sample['failed_closures']}")

    torch.save(dataset_samples, OUT_PT)
    print("=" * 80)
    print(f"[✓] Successfully packaged {len(dataset_samples)} samples into {OUT_PT} ({OUT_PT.stat().st_size / 1024:.1f} KB)")
    print("=" * 80)


if __name__ == "__main__":
    main()
