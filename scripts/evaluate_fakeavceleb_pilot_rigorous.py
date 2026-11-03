"""
Script: scripts/evaluate_fakeavceleb_pilot_rigorous.py
Runs full extraction (D_lip trajectory, forced alignment, desync computation) on 25 RealVideo-RealAudio
and 25 FakeVideo-RealAudio (Wav2Lip) clips from FakeAVCeleb v1.2.
Computes Cohen's d using subject-level means as the statistical unit to eliminate pseudoreplication.
Applies the formal decision gate (d >= 0.80 -> proceed; d < 0.50 -> stop and report findings).
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict
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

CSV_PATH = Path("data/raw/FakeAVCeleb_v1.2/meta_data.csv")
BASE_DIR = Path("data/raw/FakeAVCeleb_v1.2")


def get_distinct_pairs(target_count: int = 25):
    """Pulls matched (Real, Wav2Lip) video pairs from unique subjects."""
    real_by_subj = {}
    w2l_by_subj = {}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("type", "").strip()
            m = row.get("method", "").strip()
            race = row.get("race", "").strip()
            gender = row.get("gender", "").strip()
            source_id = row.get("source", "").strip()
            p = row.get("path", "").strip()
            fname = p.split("/")[-1] if "/" in p else p

            if t == "RealVideo-RealAudio" and source_id not in real_by_subj:
                full_p = BASE_DIR / t / race / gender / source_id / fname
                if full_p.exists():
                    real_by_subj[source_id] = (full_p, fname)

            elif t == "FakeVideo-RealAudio" and m == "wav2lip" and source_id not in w2l_by_subj:
                full_p = BASE_DIR / t / race / gender / source_id / fname
                if full_p.exists():
                    w2l_by_subj[source_id] = (full_p, fname)

    pairs = []
    common_subjs = sorted(set(real_by_subj.keys()) & set(w2l_by_subj.keys()))
    for subj in common_subjs[:target_count]:
        pairs.append((subj, real_by_subj[subj][0], w2l_by_subj[subj][0]))

    return pairs


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(y) - np.mean(x)) / sp if sp > 1e-9 else 0.0


def main():
    print("=" * 95)
    print(f"{'RIGOROUS FAKEAVCELEB PILOT: REAL VS WAV2LIP (SUBJECT-LEVEL EXTRACTION)':^95}")
    print("=" * 95)

    pairs = get_distinct_pairs(target_count=25)
    print(f"[*] Pulled {len(pairs)} distinct subject pairs (Total {len(pairs)*2} video clips).")

    tracker = LipTracker()
    phoneme_ext = PhonemeExtractor(lazy_load=False)
    metric = DesynchronizationMetric()
    asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

    # Data structures for subject-level aggregation
    subj_real_desync = {}
    subj_w2l_desync = {}
    subj_real_min_ap = {}
    subj_w2l_min_ap = {}
    subj_real_fail_ratio = {}
    subj_w2l_fail_ratio = {}

    print("\n" + "-" * 95)
    print(f"{'Subj ID':<10} | {'Condition':<9} | {'Frames':<7} | {'Bilabials':<10} | {'S_desync':<10} | {'Min Aperture':<14} | {'Failed Ratio'}")
    print("-" * 95)

    for idx, (subj_id, real_p, w2l_p) in enumerate(pairs):
        # 1. Process Real Video
        wav_r, sr_r = phoneme_ext.extract_audio_from_video(real_p)
        if wav_r is None or len(wav_r) < sr_r:
            print(f"  [!] {subj_id} Real: Audio extraction failed, skipping pair.")
            continue

        asr_r = asr({"raw": wav_r, "sampling_rate": sr_r})
        text_r = asr_r["text"].strip()
        if not text_r:
            print(f"  [!] {subj_id} Real: ASR empty, skipping pair.")
            continue

        try:
            audio_r = phoneme_ext.align_forced_transcript(wav_r, prompt_text=text_r, sample_rate=sr_r)
        except Exception:
            continue

        if not audio_r.bilabial_intervals:
            continue

        cap_r = cv2.VideoCapture(str(real_p))
        fps_r = cap_r.get(cv2.CAP_PROP_FPS) or 30.0
        frame_dur_r = (1.0 / fps_r) * 1000.0
        lips_r = []
        f_idx = 0
        while cap_r.isOpened() and f_idx < 150:
            ret, frame = cap_r.read()
            if not ret or frame is None:
                break
            lips_r.append(tracker.process_frame(frame, timestamp_ms=f_idx * frame_dur_r))
            f_idx += 1
        cap_r.release()

        # 2. Process Wav2Lip Video
        wav_w, sr_w = phoneme_ext.extract_audio_from_video(w2l_p)
        if wav_w is None or len(wav_w) < sr_w:
            continue

        cap_w = cv2.VideoCapture(str(w2l_p))
        fps_w = cap_w.get(cv2.CAP_PROP_FPS) or 30.0
        frame_dur_w = (1.0 / fps_w) * 1000.0
        lips_w = []
        f_idx = 0
        while cap_w.isOpened() and f_idx < 150:
            ret, frame = cap_w.read()
            if not ret or frame is None:
                break
            lips_w.append(tracker.process_frame(frame, timestamp_ms=f_idx * frame_dur_w))
            f_idx += 1
        cap_w.release()

        if not lips_r or not lips_w:
            continue

        desync_r = metric.compute_desync_score(lips_r, audio_r)
        desync_w = metric.compute_desync_score(lips_w, audio_r)

        num_b = desync_r.num_bilabial_events
        if num_b == 0:
            continue

        # Record subject-level values
        r_fail = desync_r.failed_closure_count / num_b
        w_fail = desync_w.failed_closure_count / num_b

        subj_real_desync[subj_id] = desync_r.overall_desync_score
        subj_w2l_desync[subj_id] = desync_w.overall_desync_score
        subj_real_min_ap[subj_id] = desync_r.mean_aperture_during_bilabials
        subj_w2l_min_ap[subj_id] = desync_w.mean_aperture_during_bilabials
        subj_real_fail_ratio[subj_id] = r_fail
        subj_w2l_fail_ratio[subj_id] = w_fail

        print(f"{subj_id:<10} | {'REAL':<9} | {len(lips_r):>3} fr  | {num_b:>2} events  | {desync_r.overall_desync_score:>8.4f}   | {desync_r.mean_aperture_during_bilabials:>10.4f}     | {desync_r.failed_closure_count}/{num_b} ({r_fail*100:>4.1f}%)")
        print(f"{subj_id:<10} | {'WAV2LIP':<9} | {len(lips_w):>3} fr  | {num_b:>2} events  | {desync_w.overall_desync_score:>8.4f}   | {desync_w.mean_aperture_during_bilabials:>10.4f}     | {desync_w.failed_closure_count}/{num_b} ({w_fail*100:>4.1f}%)")
        print("-" * 95)

    valid_subjs = list(subj_real_desync.keys())
    N = len(valid_subjs)

    r_desync_vals = [subj_real_desync[s] for s in valid_subjs]
    w_desync_vals = [subj_w2l_desync[s] for s in valid_subjs]
    r_min_ap_vals = [subj_real_min_ap[s] for s in valid_subjs]
    w_min_ap_vals = [subj_w2l_min_ap[s] for s in valid_subjs]
    r_fail_vals = [subj_real_fail_ratio[s] for s in valid_subjs]
    w_fail_vals = [subj_w2l_fail_ratio[s] for s in valid_subjs]

    d_desync = cohens_d(r_desync_vals, w_desync_vals)
    d_min_ap = cohens_d(r_min_ap_vals, w_min_ap_vals)
    d_fail = cohens_d(r_fail_vals, w_fail_vals)

    print("\n" + "=" * 95)
    print(f"{'SUBJECT-LEVEL STATISTICAL REPORT (N = ' + str(N) + ' UNIQUE SUBJECTS)':^95}")
    print("=" * 95)
    print(f"  • Real Video S_desync:     {np.mean(r_desync_vals):.4f} ± {np.std(r_desync_vals, ddof=1):.4f}")
    print(f"  • Wav2Lip S_desync:        {np.mean(w_desync_vals):.4f} ± {np.std(w_desync_vals, ddof=1):.4f}")
    print(f"  • Cohen's d (S_desync):    {d_desync:+.3f} (Standardized Effect Size)")
    print(f"  • Cohen's d (Min Aperture):{d_min_ap:+.3f}")
    print(f"  • Cohen's d (Fail Ratio):  {d_fail:+.3f}")

    print("\n" + "=" * 95)
    print(f"{'DECISION GATE EVALUATION':^95}")
    print("=" * 95)
    if d_desync >= 0.80:
        print(f"  [✓] DECISION GATE PASSED (d = {d_desync:.3f} >= 0.80):")
        print("      Large effect size confirmed. Proceeding to process the full ~9,700-clip category.")
    elif d_desync < 0.50:
        print(f"  [X] DECISION GATE TRIGGERED: STOP (d = {d_desync:.3f} < 0.50):")
        print("      No strong separation detected between Real and Wav2Lip on unconstrained YouTube video.")
        print("      Reason: Wav2Lip was trained on SyncNet to explicitly minimize mouth aperture during low energy,")
        print("      while unconstrained celebrity interviews contain head turns, smiles, and conversational reductions.")
        print("      Per protocol: STOPPING full extraction on FakeAVCeleb and reporting findings.")
    else:
        print(f"  [~] MODERATE EFFECT SIZE (0.50 <= d = {d_desync:.3f} < 0.80): Further analysis required.")
    print("=" * 95)


if __name__ == "__main__":
    main()
