#!/usr/bin/env python3
"""
scripts/measure_offline_per_frame.py

Measures per-frame processing latency on the offline video analysis endpoint:
  POST /analyze-offline-video in app/main.py

Measurement Protocol:
1. Calls /analyze-offline-video via FastAPI TestClient on each clip in data/demo_clips/
   using the canonical sentence prompts from scripts/measure_ws_latency.py.
2. The very first call of the whole process is recorded separately as the cold call.
3. For each clip: 1 discarded warm-up call, followed by 5 timed repetitions.
4. For each run, records:
   - Endpoint-reported per-frame time: data["per_frame_latency_ms"]
   - Endpoint-reported total latency: data["latency_ms"]
   - Client wall-clock time divided by frame count: wall_time_ms / frames_evaluated
   - Client wall-clock total time: wall_time_ms
   - Verdict, frames evaluated, CER/WER if present.
5. Flags any clip rejected early by the CER prompt gate (REJECTED_PROMPT_ALIGNMENT_FAILED)
   because it did not run the full path. Excludes those clips from steady-state statistics
   and reports them separately.
6. Saves comprehensive results to data/processed/offline_per_frame_results.json.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_device, PROCESSED_DATA_DIR
from app.main import app

TRANSCRIPTS = {
    "sa2": "Don't ask me to carry an oily rag like that.",
    "sx369": "bulk paper over the living room murals.",
}

DEMO_CLIPS = [
    # 6 Genuine clips
    {
        "filename": "real_01_fadg0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "real_02_fcmh0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "real_03_fdac1_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "real_04_fgjd0_sx369.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sx369",
        "prompt": TRANSCRIPTS["sx369"],
    },
    {
        "filename": "real_05_fjre0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "real_06_fdms0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    # 4 Fake clips
    {
        "filename": "fake_01_fadg0_sa2_fram1.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "fake_02_faks0_sa2_fedw0.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "fake_03_fcft0_sa2_fdrd1.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
    },
    {
        "filename": "fake_04_msjs1_sx369_mrgg0.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sx369",
        "prompt": TRANSCRIPTS["sx369"],
    },
]


def run_endpoint_call(client: TestClient, clip_path: Path, prompt: str) -> Dict[str, Any]:
    with open(clip_path, "rb") as f:
        video_bytes = f.read()

    t_start = time.perf_counter()
    response = client.post(
        "/analyze-offline-video",
        files={"video": (clip_path.name, video_bytes, "video/mp4")},
        data={"expected_transcript": prompt},
    )
    t_end = time.perf_counter()

    wall_ms = round((t_end - t_start) * 1000.0, 3)

    if response.status_code != 200:
        raise RuntimeError(f"Endpoint returned HTTP {response.status_code}: {response.text}")

    resp_json = response.json()
    frames = resp_json.get("frames_evaluated") or 1
    endpoint_per_frame = resp_json.get("per_frame_latency_ms")
    endpoint_total = resp_json.get("latency_ms")
    wall_per_frame = round(wall_ms / max(frames, 1), 3)

    return {
        "verdict_text": resp_json.get("verdict_text"),
        "deepfake_probability": resp_json.get("deepfake_probability"),
        "frames_evaluated": frames,
        "endpoint_latency_ms": endpoint_total,
        "endpoint_per_frame_latency_ms": endpoint_per_frame,
        "wall_latency_ms": wall_ms,
        "wall_per_frame_latency_ms": wall_per_frame,
        "acoustic_cer": resp_json.get("acoustic_cer"),
        "acoustic_wer": resp_json.get("acoustic_wer"),
        "raw_response": resp_json,
    }


def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    return {
        "mean": round(mean_val, 3),
        "std": round(std_val, 3),
        "min": round(min_val, 3),
        "max": round(max_val, 3),
        "n": len(values),
    }


def main():
    device = get_device()
    torch_version = torch.__version__
    clips_dir = PROJECT_ROOT / "data" / "demo_clips"
    output_file = PROCESSED_DATA_DIR / "offline_per_frame_results.json"

    print("=" * 80)
    print("SecureKYC AI: Offline Endpoint Per-Frame Latency Benchmark")
    print("=" * 80)
    print(f"Device: {device} (Apple Silicon MPS: {device.type == 'mps'})")
    print(f"PyTorch Version: {torch_version}")
    print(f"Endpoint: POST /analyze-offline-video (FastAPI TestClient)")
    print(f"Demo Clips Directory: {clips_dir}")
    print(f"Clips count: {len(DEMO_CLIPS)}")
    print("Protocol: 1 cold call (first overall), 1 discarded warm-up per clip, 5 timed runs per clip")
    print("=" * 80)

    client = TestClient(app)

    cold_call_result = None
    clip_results: List[Dict[str, Any]] = []

    for clip_idx, item in enumerate(DEMO_CLIPS):
        clip_path = clips_dir / item["filename"]
        if not clip_path.exists():
            print(f"ERROR: Clip not found: {clip_path}")
            continue

        print(f"\n--- [{clip_idx + 1}/{len(DEMO_CLIPS)}] {item['filename']} ({item['ground_truth']}) ---")
        print(f"Prompt: \"{item['prompt']}\"")

        # 1. Warm-up call (or Cold call if first of the whole process)
        if clip_idx == 0:
            print("  [Run 0: COLD CALL - First call of the process]")
            cold_run = run_endpoint_call(client, clip_path, item["prompt"])
            cold_call_result = {
                "clip": item["filename"],
                "frames_evaluated": cold_run["frames_evaluated"],
                "endpoint_per_frame_latency_ms": cold_run["endpoint_per_frame_latency_ms"],
                "endpoint_latency_ms": cold_run["endpoint_latency_ms"],
                "wall_per_frame_latency_ms": cold_run["wall_per_frame_latency_ms"],
                "wall_latency_ms": cold_run["wall_latency_ms"],
                "verdict_text": cold_run["verdict_text"],
            }
            print(f"    Cold Call Result -> Endpoint Per-Frame: {cold_run['endpoint_per_frame_latency_ms']} ms | Wall Per-Frame: {cold_run['wall_per_frame_latency_ms']} ms | Total Wall: {cold_run['wall_latency_ms']} ms")
            print("  [Run 1: Discarded Warm-up run]")
            _ = run_endpoint_call(client, clip_path, item["prompt"])
        else:
            print("  [Run 0: Discarded Warm-up run]")
            _ = run_endpoint_call(client, clip_path, item["prompt"])

        # 2. Five timed repetitions
        timed_runs: List[Dict[str, Any]] = []
        for rep in range(1, 6):
            res = run_endpoint_call(client, clip_path, item["prompt"])
            timed_runs.append(res)
            print(f"    Timed Run {rep}/5: Endpoint Per-Frame: {res['endpoint_per_frame_latency_ms']} ms | Wall Per-Frame: {res['wall_per_frame_latency_ms']} ms | Total Endpoint: {res['endpoint_latency_ms']} ms | Total Wall: {res['wall_latency_ms']} ms | Verdict: {res['verdict_text']}")

        # Analyze clip results
        endpoint_pfs = [r["endpoint_per_frame_latency_ms"] for r in timed_runs]
        wall_pfs = [r["wall_per_frame_latency_ms"] for r in timed_runs]
        endpoint_totals = [r["endpoint_latency_ms"] for r in timed_runs]
        wall_totals = [r["wall_latency_ms"] for r in timed_runs]
        verdicts = [r["verdict_text"] for r in timed_runs]
        is_cer_rejected = any(v == "REJECTED_PROMPT_ALIGNMENT_FAILED" for v in verdicts)

        clip_summary = {
            "filename": item["filename"],
            "ground_truth": item["ground_truth"],
            "sentence_id": item["sentence_id"],
            "prompt": item["prompt"],
            "frames_evaluated": timed_runs[0]["frames_evaluated"],
            "cer_rejected": is_cer_rejected,
            "verdicts": verdicts,
            "endpoint_per_frame_stats": compute_stats(endpoint_pfs),
            "wall_per_frame_stats": compute_stats(wall_pfs),
            "endpoint_total_latency_stats": compute_stats(endpoint_totals),
            "wall_total_latency_stats": compute_stats(wall_totals),
            "raw_runs": [
                {
                    "rep": i + 1,
                    "endpoint_per_frame_latency_ms": r["endpoint_per_frame_latency_ms"],
                    "endpoint_latency_ms": r["endpoint_latency_ms"],
                    "wall_per_frame_latency_ms": r["wall_per_frame_latency_ms"],
                    "wall_latency_ms": r["wall_latency_ms"],
                    "verdict_text": r["verdict_text"],
                    "acoustic_cer": r["acoustic_cer"],
                }
                for i, r in enumerate(timed_runs)
            ],
        }
        clip_results.append(clip_summary)

    # Separate full-path clips from CER-rejected clips
    steady_state_clips = [c for c in clip_results if not c["cer_rejected"]]
    rejected_clips = [c for c in clip_results if c["cer_rejected"]]

    # Aggregate steady-state runs
    all_steady_endpoint_pfs = []
    all_steady_wall_pfs = []
    all_steady_endpoint_totals = []
    all_steady_wall_totals = []
    for c in steady_state_clips:
        for r in c["raw_runs"]:
            all_steady_endpoint_pfs.append(r["endpoint_per_frame_latency_ms"])
            all_steady_wall_pfs.append(r["wall_per_frame_latency_ms"])
            all_steady_endpoint_totals.append(r["endpoint_latency_ms"])
            all_steady_wall_totals.append(r["wall_latency_ms"])

    steady_endpoint_pf_stats = compute_stats(all_steady_endpoint_pfs)
    steady_wall_pf_stats = compute_stats(all_steady_wall_pfs)
    steady_endpoint_total_stats = compute_stats(all_steady_endpoint_totals)
    steady_wall_total_stats = compute_stats(all_steady_wall_totals)

    # Aggregate rejected runs
    all_rejected_endpoint_pfs = []
    all_rejected_wall_pfs = []
    for c in rejected_clips:
        for r in c["raw_runs"]:
            all_rejected_endpoint_pfs.append(r["endpoint_per_frame_latency_ms"])
            all_rejected_wall_pfs.append(r["wall_per_frame_latency_ms"])
    rejected_endpoint_pf_stats = compute_stats(all_rejected_endpoint_pfs)
    rejected_wall_pf_stats = compute_stats(all_rejected_wall_pfs)

    output_data = {
        "metadata": {
            "device": str(device),
            "is_mps": device.type == "mps",
            "torch_version": torch_version,
            "transport": "FastAPI TestClient (in-process ASGI HTTP)",
            "endpoint": "/analyze-offline-video",
            "total_clips_tested": len(clip_results),
            "steady_state_clips_count": len(steady_state_clips),
            "rejected_clips_count": len(rejected_clips),
            "runs_per_clip": 5,
        },
        "cold_call": cold_call_result,
        "steady_state_summary": {
            "description": "Steady-state statistics across clips completing the full path (excluding CER-rejected clips)",
            "endpoint_per_frame_ms": steady_endpoint_pf_stats,
            "wall_per_frame_ms": steady_wall_pf_stats,
            "endpoint_total_latency_ms": steady_endpoint_total_stats,
            "wall_total_latency_ms": steady_wall_total_stats,
        },
        "cer_rejected_summary": {
            "description": "Statistics for clips rejected early by the CER prompt alignment gate",
            "clips": [c["filename"] for c in rejected_clips],
            "endpoint_per_frame_ms": rejected_endpoint_pf_stats,
            "wall_per_frame_ms": rejected_wall_pf_stats,
        },
        "per_clip_results": clip_results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT: OFFLINE PER-FRAME LATENCY")
    print("=" * 80)
    print(f"Output saved to: {output_file}")
    print(f"Device: {device} | PyTorch: {torch_version}")
    print(f"\n[COLD CALL (First call of entire process on {cold_call_result['clip']})]")
    print(f"  Endpoint Per-Frame: {cold_call_result['endpoint_per_frame_latency_ms']} ms")
    print(f"  Wall Per-Frame:     {cold_call_result['wall_per_frame_latency_ms']} ms")
    print(f"  Total Endpoint:     {cold_call_result['endpoint_latency_ms']} ms")
    print(f"  Total Wall Time:    {cold_call_result['wall_latency_ms']} ms")
    print(f"  Frames Evaluated:   {cold_call_result['frames_evaluated']}")

    print(f"\n[STEADY-STATE BENCHMARK ({len(steady_state_clips)} clips, {steady_endpoint_pf_stats['n']} timed runs)]")
    print(f"  Endpoint Per-Frame Time:")
    print(f"    Mean ± SD: {steady_endpoint_pf_stats['mean']} ± {steady_endpoint_pf_stats['std']} ms")
    print(f"    Min - Max: {steady_endpoint_pf_stats['min']} - {steady_endpoint_pf_stats['max']} ms")
    print(f"  Wall Per-Frame Time (Total Wall / Frames):")
    print(f"    Mean ± SD: {steady_wall_pf_stats['mean']} ± {steady_wall_pf_stats['std']} ms")
    print(f"    Min - Max: {steady_wall_pf_stats['min']} - {steady_wall_pf_stats['max']} ms")
    print(f"  Total Endpoint Latency:")
    print(f"    Mean ± SD: {steady_endpoint_total_stats['mean']} ± {steady_endpoint_total_stats['std']} ms")
    print(f"    Min - Max: {steady_endpoint_total_stats['min']} - {steady_endpoint_total_stats['max']} ms")
    print(f"  Total Wall Time:")
    print(f"    Mean ± SD: {steady_wall_total_stats['mean']} ± {steady_wall_total_stats['std']} ms")
    print(f"    Min - Max: {steady_wall_total_stats['min']} - {steady_wall_total_stats['max']} ms")

    if rejected_clips:
        print(f"\n[CER-REJECTED CLIPS (Excluded from Steady-State Stats: {len(rejected_clips)} clip(s))]")
        for rc in rejected_clips:
            print(f"  * {rc['filename']}: Verdict={rc['verdicts'][0]}, CER={rc['raw_runs'][0]['acoustic_cer']}")
            print(f"    Endpoint Per-Frame: {rc['endpoint_per_frame_stats']['mean']} ± {rc['endpoint_per_frame_stats']['std']} ms (range {rc['endpoint_per_frame_stats']['min']} - {rc['endpoint_per_frame_stats']['max']} ms)")
            print(f"    Wall Per-Frame:     {rc['wall_per_frame_stats']['mean']} ± {rc['wall_per_frame_stats']['std']} ms")

    print(f"\n[PER-CLIP BREAKDOWN]")
    print(f"{'Clip Filename':<32} {'Frames':<7} {'CER?':<6} {'Endpoint PF Mean±SD (ms)':<25} {'Wall PF Mean±SD (ms)':<23}")
    print("-" * 95)
    for c in clip_results:
        cer_flag = "YES" if c["cer_rejected"] else "NO"
        ep_str = f"{c['endpoint_per_frame_stats']['mean']} ± {c['endpoint_per_frame_stats']['std']} ({c['endpoint_per_frame_stats']['min']}-{c['endpoint_per_frame_stats']['max']})"
        wall_str = f"{c['wall_per_frame_stats']['mean']} ± {c['wall_per_frame_stats']['std']}"
        print(f"{c['filename']:<32} {c['frames_evaluated']:<7} {cer_flag:<6} {ep_str:<25} {wall_str:<23}")
    print("=" * 80)


if __name__ == "__main__":
    main()
