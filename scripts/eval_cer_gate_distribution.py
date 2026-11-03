#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
import soundfile as sf
import numpy as np

from config import ACTIVE_PROMPT_POOL
from src.audio.phoneme_extractor import PhonemeExtractor, compute_cer

def main():
    print("=" * 80)
    print("CER GATE DISTRIBUTION & FULL 335x5 ACTIVE PROMPT MISMATCH EVALUATION")
    print("=" * 80)

    # 1. Load transcripts
    with open("data/processed/timit_15spk_transcripts.json") as f:
        transcripts = json.load(f)

    # 2. Find clips
    wav_files = sorted(Path("data/raw/original_sequences").glob("*/audio/*.wav"))
    matched = []
    for w in wav_files:
        sid = w.stem.lower()
        if sid in transcripts:
            matched.append((w, sid, transcripts[sid]))

    print(f"[*] Found {len(matched)} real VidTIMIT clips with verified transcripts.")

    # 3. Initialize Wav2Vec2 CTC
    extractor = PhonemeExtractor(lazy_load=True)
    extractor.load_asr_model()

    # 4. Decode all matched clips
    cers = []
    detailed_records = []
    clips_above_60 = []

    t0 = time.time()
    for idx, (wpath, sid, ref_text) in enumerate(matched):
        wf, sr = sf.read(str(wpath))
        if wf.ndim > 1:
            wf = wf[:, 0]
        hyp_text = extractor.decode_transcription(wf.astype(np.float32), sample_rate=sr)
        cer = compute_cer(ref_text, hyp_text)
        cers.append(cer)
        record = {
            "path": str(wpath),
            "speaker": wpath.parent.parent.name,
            "sentence_id": sid,
            "reference": ref_text,
            "decoded": hyp_text,
            "cer": float(cer)
        }
        detailed_records.append(record)
        if cer > 0.60:
            clips_above_60.append(record)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(matched):
            print(f"  Processed {idx + 1}/{len(matched)} clips ({(idx+1)/len(matched)*100:.1f}%)")

    t1 = time.time()
    print(f"[*] Decoded {len(matched)} clips in {t1 - t0:.1f}s.")

    # 5. Genuine CER Distribution Statistics & Deciles
    cers_arr = np.array(cers)
    n = len(cers_arr)
    c_min = float(np.min(cers_arr))
    c_med = float(np.median(cers_arr))
    c_mean = float(np.mean(cers_arr))
    c_p95 = float(np.percentile(cers_arr, 95))
    c_max = float(np.max(cers_arr))
    n_above_60 = len(clips_above_60)
    frac_above_60 = float(n_above_60 / n)

    deciles = {}
    for p in range(10, 101, 10):
        deciles[f"D{p//10}_{p}th"] = float(np.percentile(cers_arr, p))

    print("\n" + "=" * 80)
    print("GENUINE CER DECILES (N=335)")
    print("=" * 80)
    for k, v in deciles.items():
        print(f"  {k:10s}: {v:.4f}")
    print("=" * 80)

    # 6. Evaluate all 335 clips against ALL 5 ACTIVE_PROMPT_POOL prompts (335 x 5 = 1,675 mismatches)
    print("\n" + "=" * 80)
    print("EVALUATING 335 CLIPS x 5 PROMPTS (1,675 MISMATCHED PAIRS)")
    print("=" * 80)
    mismatch_cers = []
    mismatches_below_60 = []

    for r in detailed_records:
        dec = r["decoded"]
        for prompt in ACTIVE_PROMPT_POOL:
            # CER of decoded utterance against challenge prompt
            cer_m = compute_cer(prompt, dec)
            mismatch_cers.append(float(cer_m))
            if cer_m < 0.60:
                mismatches_below_60.append({
                    "speaker": r["speaker"],
                    "sentence_id": r["sentence_id"],
                    "decoded": dec,
                    "prompt": prompt,
                    "cer": float(cer_m)
                })

    m_arr = np.array(mismatch_cers)
    m_n = len(m_arr)
    m_min = float(np.min(m_arr))
    m_p5 = float(np.percentile(m_arr, 5))
    m_med = float(np.median(m_arr))
    m_below_60_cnt = len(mismatches_below_60)
    m_far_at_60 = float(m_below_60_cnt / m_n)

    print(f"Total Mismatches (N)          : {m_n}")
    print(f"Min Mismatch CER              : {m_min:.4f}")
    print(f"5th Percentile Mismatch CER   : {m_p5:.4f}")
    print(f"Median Mismatch CER           : {m_med:.4f}")
    print(f"Count Below 0.60 (False Accept): {m_below_60_cnt}")
    print(f"False-Accept Rate at 0.60     : {m_far_at_60:.4f} ({m_far_at_60*100:.2f}%)")
    print("=" * 80)

    # Save complete results
    out_file = Path("data/processed/cer_gate_eval_results.json")
    results = {
        "summary": {
            "n": n,
            "min": c_min,
            "median": c_med,
            "mean": c_mean,
            "p95": c_p95,
            "max": c_max,
            "count_above_60": n_above_60,
            "fraction_above_60": frac_above_60,
            "deciles": deciles
        },
        "mismatch_evaluation_335x5": {
            "n_mismatches": m_n,
            "min_cer": m_min,
            "p5_cer": m_p5,
            "median_cer": m_med,
            "count_below_60": m_below_60_cnt,
            "false_accept_rate_at_60": m_far_at_60,
            "mismatches_below_60": mismatches_below_60
        },
        "all_clips": detailed_records,
        "clips_above_60": clips_above_60
    }
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[✓] Full evaluation saved to {out_file}")

if __name__ == "__main__":
    main()
