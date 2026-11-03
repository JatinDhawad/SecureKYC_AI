"""
SecureKYC AI - Empirical Lip Closure Threshold Calibration Script
Processes genuine (non-spoofed) audio-visual speaker clips from VidTIMIT,
extracts real 3D lip aperture D_lip(t) at every forced-alignment bilabial event (/b/, /m/, /p/),
derives the empirical 95th percentile threshold, updates config.py, and saves the distribution plot.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.vision.lip_tracker import LipTracker, LipFrameResult
from src.audio.phoneme_extractor import PhonemeExtractor, AudioPhonemeResult

# VidTIMIT Sentence Transcripts Dictionary
# Loads complete transcript dictionary covering all 15 speakers' full sentence sets (107 unique sentences)
TRANSCRIPTS_FILE = PROCESSED_DATA_DIR / "timit_15spk_transcripts.json"
if TRANSCRIPTS_FILE.exists():
    with open(TRANSCRIPTS_FILE, "r") as f:
        VIDTIMIT_TRANSCRIPTS = json.load(f)
else:
    VIDTIMIT_TRANSCRIPTS = {
        "sa1": "She had your dark suit in greasy wash water all year.",
        "sa2": "Don't ask me to carry an oily rag like that.",
    }


def sort_frame_files(frame_files):
    """Sorts frame image files numerically (001, 002, 003...)."""
    def extract_num(p: Path) -> int:
        match = re.search(r"\d+", p.stem)
        return int(match.group()) if match else 0
    return sorted(frame_files, key=extract_num)


def get_word_positions(prompt_text: str):
    """Maps alphabetic characters to word, position within word, and sentence position."""
    words = re.findall(r"[a-zA-Z']+", prompt_text)
    char_map = []
    for w_idx, word in enumerate(words):
        clean_w = [c.upper() for c in word if c.isalpha()]
        for c_idx, c in enumerate(clean_w):
            if c_idx == 0:
                pos = "initial"
            elif c_idx == len(clean_w) - 1:
                pos = "final"
            else:
                pos = "medial"
            is_sent_final = (w_idx == len(words) - 1)
            char_map.append((c, word, pos, is_sent_final))
    return char_map


def collect_empirical_closure_apertures():
    """
    Processes genuine VidTIMIT speaker audio-video pairs across all 15 speakers,
    correlates forced-alignment bilabial timestamps with 3D lip tracker trajectories,
    and returns all minimum lip apertures and annotated event records.
    """
    orig_dir = RAW_DATA_DIR / "original_sequences"
    speaker_dirs = sorted([d for d in orig_dir.iterdir() if d.is_dir() and d.name not in ["Celeb-real", "YouTube-real", "youtube"]])

    if not speaker_dirs:
        raise FileNotFoundError(f"No VidTIMIT speaker folders found in '{orig_dir}'.")

    print(f"[*] Found {len(speaker_dirs)} VidTIMIT speaker directories for threshold calibration.")

    tracker = LipTracker()
    phoneme_extractor = PhonemeExtractor(lazy_load=False)
    closure_apertures = []
    event_records = []

    speakers_processed = 0

    for speaker_dir in speaker_dirs[:15]:  # Calibrate across 15 real speaker directories
        audio_dir = speaker_dir / "audio"
        video_dir = speaker_dir / "video"

        if not audio_dir.exists() or not video_dir.exists():
            continue

        speakers_processed += 1
        print(f"[*] Calibrating on speaker [{speakers_processed}/15]: {speaker_dir.name}...")

        # Process all sentences for this speaker
        wav_files = sorted(audio_dir.glob("*.wav"))
        for wav_path in wav_files:
            sentence_key = wav_path.stem
            prompt_text = VIDTIMIT_TRANSCRIPTS.get(sentence_key)
            if not prompt_text:
                continue

            sentence_video_dir = video_dir / sentence_key
            if not sentence_video_dir.exists():
                continue

            # 1. Audio Forced Alignment
            sr, audio_int16 = wavfile.read(str(wav_path))
            waveform = audio_int16.astype(np.float32) / 32768.0 if audio_int16.dtype == np.int16 else audio_int16.astype(np.float32)
            audio_res = phoneme_extractor.align_forced_transcript(waveform, prompt_text=prompt_text, sample_rate=sr)

            if not audio_res.bilabial_intervals:
                continue

            # 2. Vision 3D Lip Tracker on frame sequence
            frame_files = sort_frame_files(list(sentence_video_dir.glob("*")))
            if not frame_files:
                continue

            lip_results = []
            frame_duration_ms = 40.0  # 25 fps = 40 ms per frame

            for f_idx, f_path in enumerate(frame_files):
                frame_img = cv2.imread(str(f_path))
                if frame_img is None:
                    continue
                res = tracker.process_frame(frame_img, timestamp_ms=f_idx * frame_duration_ms)
                lip_results.append(res)

            if not lip_results:
                continue

            char_map = get_word_positions(prompt_text)
            bilabial_char_map = [x for x in char_map if x[0] in {"B", "M", "P"}]

            # 3. Inspect [t-2, t+2] window for every acoustic bilabial event
            for b_idx, interval in enumerate(audio_res.bilabial_intervals):
                mid_t = (interval.start_time_ms + interval.end_time_ms) / 2.0
                frame_indices = [i for i, r in enumerate(lip_results) if r.face_detected]

                if not frame_indices:
                    continue

                closest_idx = min(frame_indices, key=lambda i: abs(lip_results[i].frame_timestamp_ms - mid_t))

                w_start = max(0, closest_idx - 2)
                w_end = min(len(lip_results), closest_idx + 3)

                window_frames = lip_results[w_start:w_end]
                valid_apertures = [f.lip_aperture for f in window_frames if f.face_detected and f.lip_aperture > 0.0]

                if valid_apertures:
                    min_ap = float(np.min(valid_apertures))
                    closure_apertures.append(min_ap)

                    w_info = bilabial_char_map[b_idx] if b_idx < len(bilabial_char_map) else (interval.phoneme, "unknown", "unknown", False)
                    event_records.append({
                        "speaker": speaker_dir.name,
                        "sentence": sentence_key,
                        "prompt": prompt_text,
                        "phoneme": interval.phoneme,
                        "word": w_info[1],
                        "position": w_info[2],
                        "is_sent_final": w_info[3],
                        "start_time_ms": round(interval.start_time_ms, 1),
                        "end_time_ms": round(interval.end_time_ms, 1),
                        "confidence": round(interval.confidence, 4),
                        "d_lip": round(min_ap, 4),
                    })

    tracker.close()
    return closure_apertures, event_records


def run_calibration_and_update_config():
    """Calculates empirical closure threshold statistics, updates config.py, and plots distribution."""
    print("=" * 75)
    print(" SecureKYC AI - Empirical Lip Closure Threshold Calibration ")
    print("=" * 75)

    apertures, event_records = collect_empirical_closure_apertures()
    assert len(apertures) > 0, "Failed to collect empirical closure apertures from dataset."

    # Save detailed events breakdown
    events_json_path = PROCESSED_DATA_DIR / "calibration_events_breakdown.json"
    with open(events_json_path, "w") as f:
        json.dump(event_records, f, indent=2)
    print(f"\n[✓] Saved complete N={len(event_records)} event records to {events_json_path}")

    arr = np.array(apertures, dtype=np.float32)

    min_ap = float(np.min(arr))
    max_ap = float(np.max(arr))
    mean_ap = float(np.mean(arr))
    std_ap = float(np.std(arr))
    p10 = float(np.percentile(arr, 10.0))
    p25 = float(np.percentile(arr, 25.0))
    median_ap = float(np.median(arr))
    p75 = float(np.percentile(arr, 75.0))
    p90 = float(np.percentile(arr, 90.0))
    p95 = float(np.percentile(arr, 95.0))
    p99 = float(np.percentile(arr, 99.0))
    calibrated_threshold = round(p95, 4)

    print(f"\n[*] Empirical Bilabial Closure Aperture Distribution Statistics:")
    print(f"    - Sample Size (Bilabial Events N): {len(arr)}")
    print(f"    - Min D_lip:                      {min_ap:.4f}")
    print(f"    - 10th Percentile:                 {p10:.4f}")
    print(f"    - 25th Percentile (Q1):            {p25:.4f}")
    print(f"    - Median D_lip (50th Percentile):  {median_ap:.4f}")
    print(f"    - 75th Percentile (Q3):            {p75:.4f}")
    print(f"    - 90th Percentile:                 {p90:.4f}")
    print(f"    - 95th Percentile (Calibrated):     {calibrated_threshold:.4f}")
    print(f"    - 99th Percentile:                 {p99:.4f}")
    print(f"    - Max D_lip:                      {max_ap:.4f}")
    print(f"    - Mean D_lip:                      {mean_ap:.4f} ± {std_ap:.4f}")
    print(f"    - Interquartile Range (IQR):       {p75 - p25:.4f}")

    # Breakdown by Phoneme
    print(f"\n[*] Phoneme Identity Breakdown:")
    for p_name in ["B", "M", "P"]:
        p_vals = [e["d_lip"] for e in event_records if e["phoneme"] == p_name]
        if p_vals:
            print(f"    - /{p_name}/ (N={len(p_vals):3d}): Mean={np.mean(p_vals):.4f} ± {np.std(p_vals):.4f} | Median={np.median(p_vals):.4f} | 95th={np.percentile(p_vals, 95):.4f} | Range=[{min(p_vals):.4f}, {max(p_vals):.4f}]")

    # Breakdown by Word Position
    print(f"\n[*] Word Position Breakdown:")
    for pos in ["initial", "medial", "final"]:
        pos_vals = [e["d_lip"] for e in event_records if e["position"] == pos]
        if pos_vals:
            print(f"    - {pos.capitalize():8s} (N={len(pos_vals):3d}): Mean={np.mean(pos_vals):.4f} ± {np.std(pos_vals):.4f} | Median={np.median(pos_vals):.4f} | 95th={np.percentile(pos_vals, 95):.4f} | Range=[{min(pos_vals):.4f}, {max(pos_vals):.4f}]")

    # Breakdown by Speaker
    print(f"\n[*] Speaker Distribution Breakdown:")
    speakers = sorted(list(set(e["speaker"] for e in event_records)))
    for spk in speakers:
        s_vals = [e["d_lip"] for e in event_records if e["speaker"] == spk]
        print(f"    - {spk:6s} (N={len(s_vals):2d}): Mean={np.mean(s_vals):.4f} ± {np.std(s_vals):.4f} | Median={np.median(s_vals):.4f} | Max={max(s_vals):.4f}")

    # 1. Update config.py with calibrated threshold
    config_path = PROJECT_ROOT / "config.py"
    config_content = config_path.read_text()

    old_pattern = r"LIP_CLOSURE_THRESHOLD_NORMALIZED\s*=\s*[0-9\.]+"
    new_comment = f"# Empirically calibrated from 95th percentile of genuine VidTIMIT bilabial closure apertures (N={len(arr)} across 15 speakers)"
    comment_pattern = r"# Empirically calibrated from 95th percentile of genuine VidTIMIT bilabial closure apertures.*"

    config_content = re.sub(comment_pattern, new_comment, config_content)
    config_content = re.sub(old_pattern, f"LIP_CLOSURE_THRESHOLD_NORMALIZED = {calibrated_threshold}", config_content)
    config_path.write_text(config_content)
    print(f"\n[✓] Updated {config_path.name} with LIP_CLOSURE_THRESHOLD_NORMALIZED = {calibrated_threshold}")

    # 2. Plot calibration distribution for research paper citation
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PROCESSED_DATA_DIR / "desync_threshold_calibration.png"

    plt.figure(figsize=(9, 5.5))
    plt.hist(arr, bins=30, color="#2b5c8f", edgecolor="black", alpha=0.8, density=True)
    plt.axvline(mean_ap, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Mean D_lip ({mean_ap:.4f})")
    plt.axvline(p95, color="#d62728", linestyle="-", linewidth=2.5, label=f"95th Percentile / Calibrated Threshold ({calibrated_threshold:.4f})")

    plt.title(f"Empirical Distribution of Genuine Bilabial Lip Closure Apertures $D_{{\\text{{lip}}}}$ (N={len(arr)})", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Normalized Lip Aperture $D_{\\text{lip}} = ||P_{13} - P_{14}||_2 / (||P_1 - P_{152}||_2 + 10^{-6})$", fontsize=10)
    plt.ylabel("Probability Density", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    plt.savefig(str(plot_path), dpi=300)
    plt.close()
    print(f"[✓] Saved empirical calibration distribution plot to {plot_path}")
    print("=" * 75)

    return calibrated_threshold


if __name__ == "__main__":
    run_calibration_and_update_config()
