"""
Script: scripts/evaluate_fakeavceleb_pilot.py
Evaluates Cohen's d effect size and bilabial seal aperture distributions
on a matched sample of RealVideo vs Wav2Lip reenactments from FakeAVCeleb v1.2.
Confirms the presence of neural acoustic-visemic desynchronization before full scaling.
"""

import sys
import csv
from pathlib import Path
import cv2
import numpy as np
from transformers import pipeline
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LIP_CLOSURE_THRESHOLD_NORMALIZED
from src.vision.lip_tracker import LipTracker
from src.audio.phoneme_extractor import PhonemeExtractor
from src.metrics.desync_metric import DesynchronizationMetric

CSV_PATH = Path("/Users/jatindhawad/Downloads/FakeAVCeleb_v1.2/meta_data.csv")
BASE_DIR = Path("/Users/jatindhawad/Downloads/FakeAVCeleb_v1.2")


def get_pilot_pairs(max_pairs: int = 20):
    pairs = []
    # Map by (source_id, clip_file)
    real_map = {}
    wav2lip_map = {}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row.get("method", "").strip()
            t = row.get("type", "").strip()
            p = row.get("path", "").strip()
            race = row.get("race", "").strip()
            gender = row.get("gender", "").strip()
            source_id = row.get("source", "").strip()
            fname = p.split("/")[-1]

            if t == "RealVideo-RealAudio":
                full_p = BASE_DIR / t / race / gender / source_id / fname
                if full_p.exists():
                    real_map[(source_id, fname)] = full_p

            elif t == "FakeVideo-RealAudio" and m == "wav2lip":
                full_p = BASE_DIR / t / race / gender / source_id / fname
                if full_p.exists():
                    # Link to source real clip name
                    real_fname = fname.split("_")[0] + ".mp4"
                    wav2lip_map[(source_id, real_fname)] = full_p

    for key, real_path in real_map.items():
        if key in wav2lip_map:
            pairs.append((real_path, wav2lip_map[key]))
            if len(pairs) >= max_pairs:
                break

    return pairs


def main():
    print("=" * 85)
    print(f"{'FAKEAVCELEB PILOT EVALUATION: REAL VS WAV2LIP REENACTMENTS':^85}")
    print("=" * 85)

    pairs = get_pilot_pairs(max_pairs=20)
    print(f"[*] Discovered {len(pairs)} matched Real-Wav2Lip video pairs.")

    tracker = LipTracker()
    phoneme_ext = PhonemeExtractor(lazy_load=False)
    metric = DesynchronizationMetric()
    asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

    real_scores, wav2lip_scores = [], []
    real_failed, wav2lip_failed = [], []
    real_min_aps, wav2lip_min_aps = [], []

    print(f"{'Pair':<6} | {'Real S_desync':<14} | {'Wav2Lip S_desync':<17} | {'Real Min Ap':<12} | {'W2L Min Ap':<12} | {'W2L Fail Ratio':<14}")
    print("-" * 85)

    for idx, (real_p, w2l_p) in enumerate(pairs):
        wav, sr = phoneme_ext.extract_audio_from_video(real_p)
        if wav is None or len(wav) < sr:
            continue

        # 1. Transcribe audio to get words
        asr_res = asr({"raw": wav, "sampling_rate": sr})
        transcript = asr_res["text"].strip()
        if not transcript:
            continue

        try:
            audio_res = phoneme_ext.align_forced_transcript(wav, prompt_text=transcript, sample_rate=sr)
        except Exception:
            continue

        if not audio_res.bilabial_intervals:
            continue

        # 2. Track Real Video
        cap = cv2.VideoCapture(str(real_p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_dur = (1.0 / fps) * 1000.0
        real_lips = []
        f_idx = 0
        while cap.isOpened() and f_idx < 150:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            real_lips.append(tracker.process_frame(frame, timestamp_ms=f_idx * frame_dur))
            f_idx += 1
        cap.release()

        # 3. Track Wav2Lip Video
        cap = cv2.VideoCapture(str(w2l_p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_dur = (1.0 / fps) * 1000.0
        w2l_lips = []
        f_idx = 0
        while cap.isOpened() and f_idx < 150:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            w2l_lips.append(tracker.process_frame(frame, timestamp_ms=f_idx * frame_dur))
            f_idx += 1
        cap.release()

        if not real_lips or not w2l_lips:
            continue

        res_real = metric.compute_desync_score(real_lips, audio_res)
        res_w2l = metric.compute_desync_score(w2l_lips, audio_res)

        num_b = res_real.num_bilabial_events
        if num_b == 0:
            continue

        real_scores.append(res_real.overall_desync_score)
        wav2lip_scores.append(res_w2l.overall_desync_score)

        real_failed.append(res_real.failed_closure_count / num_b)
        wav2lip_failed.append(res_w2l.failed_closure_count / num_b)

        real_min_aps.append(res_real.mean_aperture_during_bilabials)
        wav2lip_min_aps.append(res_w2l.mean_aperture_during_bilabials)

        print(f"P{idx+1:<5} | {res_real.overall_desync_score:>10.4f}     | {res_w2l.overall_desync_score:>10.4f}        | {res_real.mean_aperture_during_bilabials:>10.4f}  | {res_w2l.mean_aperture_during_bilabials:>10.4f}  | {res_w2l.failed_closure_count}/{num_b} ({res_w2l.failed_closure_count/num_b*100:>4.1f}%)")

    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
        sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        return (np.mean(y) - np.mean(x)) / sp

    d_desync = cohens_d(real_scores, wav2lip_scores)
    d_ap = cohens_d(real_min_aps, wav2lip_min_aps)
    d_fail = cohens_d(real_failed, wav2lip_failed)

    print("=" * 85)
    print(f"EMPIRICAL EFFECT SIZE RESULTS ON FAKEAVCELEB (N={len(real_scores)} Pairs):")
    print(f"  • Real Video S_desync:     {np.mean(real_scores):.4f} ± {np.std(real_scores, ddof=1):.4f}")
    print(f"  • Wav2Lip S_desync:        {np.mean(wav2lip_scores):.4f} ± {np.std(wav2lip_scores, ddof=1):.4f}")
    print(f"  • Cohen's d (S_desync):    {d_desync:.2f} (Standardized Effect Size)")
    print(f"  • Cohen's d (Min Aperture):{d_ap:.2f}")
    print(f"  • Cohen's d (Fail Ratio):  {d_fail:.2f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
