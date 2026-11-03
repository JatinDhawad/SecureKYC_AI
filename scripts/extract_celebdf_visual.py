"""
SecureKYC AI - Visual-Only Celeb-DF Feature Extractor

Extracts 3D lip aperture trajectories D_lip(t) across Celeb-DF v2 clips (Celeb-real,
YouTube-real, Celeb-synthesis) using MediaPipe LipTracker. No audio / desync processing.
Outputs normalized visual trajectories (FIXED_SEQ_LEN=100) and metadata.

Supports multiprocessing with macOS 'spawn' method via top-level process_one().
"""

import sys
import os
import re
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import torch
import multiprocessing as mp

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FIXED_SEQ_LEN
from src.vision.lip_tracker import LipTracker, LipFrameResult

_worker_tracker: Optional[LipTracker] = None


def get_worker_tracker() -> LipTracker:
    """Lazily initializes and caches a LipTracker instance per worker process."""
    global _worker_tracker
    if _worker_tracker is None:
        _worker_tracker = LipTracker(static_image_mode=False)
    return _worker_tracker


def normalize_sequence_length(seq: np.ndarray, target_seq_len: int = FIXED_SEQ_LEN) -> np.ndarray:
    """Interpolates lip aperture sequence to target length (100 points)."""
    if len(seq) == 0:
        return np.zeros(target_seq_len, dtype=np.float32)
    x_orig = np.linspace(0, 1, len(seq))
    x_target = np.linspace(0, 1, target_seq_len)
    return np.interp(x_target, x_orig, seq).astype(np.float32)


def process_one(item: Tuple[str, float, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Top-level worker function required for multiprocessing under 'spawn' mode.
    Processes a single MP4 video file and extracts the normalized D_lip(t) trajectory.

    Args:
        item: Tuple of (clip_path_str, label, meta_dict)

    Returns:
        Dict containing success flag, extracted tensor, or failure reason.
    """
    clip_path_str, label, meta = item
    clip_path = Path(clip_path_str)

    if not clip_path.exists():
        return {
            "success": False,
            "path": clip_path_str,
            "label": label,
            "reason": "FILE_NOT_FOUND",
            "meta": meta,
        }

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return {
            "success": False,
            "path": clip_path_str,
            "label": label,
            "reason": "VIDEO_CORRUPT",
            "meta": meta,
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration_ms = (1.0 / fps) * 1000.0

    tracker = get_worker_tracker()
    lip_results: List[LipFrameResult] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        timestamp_ms = frame_idx * frame_duration_ms
        res = tracker.process_frame(frame, timestamp_ms=timestamp_ms)
        lip_results.append(res)
        frame_idx += 1

    cap.release()

    if not lip_results:
        return {
            "success": False,
            "path": clip_path_str,
            "label": label,
            "reason": "ZERO_FRAMES",
            "meta": meta,
        }

    detected = [r for r in lip_results if r.face_detected]
    if not detected:
        return {
            "success": False,
            "path": clip_path_str,
            "label": label,
            "reason": "NO_FACE_DETECTED",
            "meta": meta,
        }

    lip_results = tracker.interpolate_missing_apertures(lip_results)

    if any(r.unreliable_face_track for r in lip_results):
        return {
            "success": False,
            "path": clip_path_str,
            "label": label,
            "reason": "UNRELIABLE_FACE_TRACK",
            "meta": meta,
        }

    aperture_seq = np.array([r.lip_aperture for r in lip_results if r.lip_aperture > 0.0], dtype=np.float32)
    if len(aperture_seq) == 0:
        return {
            "success": False,
            "path": clip_path_str,
            "label": label,
            "reason": "ZERO_VALID_APERTURES",
            "meta": meta,
        }

    norm_seq = normalize_sequence_length(aperture_seq, target_seq_len=FIXED_SEQ_LEN)

    return {
        "success": True,
        "path": clip_path_str,
        "label": label,
        "lip_sequence": norm_seq,
        "raw_frame_count": frame_idx,
        "meta": meta,
    }


def collect_celebdf_manifest(raw_dir: Path = RAW_DATA_DIR) -> Tuple[List[Tuple[str, float, Dict[str, Any]]], Dict[str, int]]:
    """
    Collects all Celeb-DF v2 clips from raw directories:
    - Celeb-real: label=0.0
    - YouTube-real: label=0.0
    - Celeb-synthesis: label=1.0

    Returns:
        manifest: List of items for process_one
        counts: Breakdown dict of real/fake counts
    """
    orig_dir = raw_dir / "original_sequences"
    manip_dir = raw_dir / "manipulated_sequences"

    celeb_real_dir = orig_dir / "Celeb-real"
    yt_real_dir = orig_dir / "YouTube-real"
    celeb_synth_dir = manip_dir / "Celeb-synthesis"

    manifest = []
    counts = {
        "celeb_real": 0,
        "youtube_real": 0,
        "total_real": 0,
        "celeb_synthesis": 0,
        "total_clips": 0,
    }

    # 1. Celeb-real
    if celeb_real_dir.exists():
        for p in sorted(celeb_real_dir.glob("*.mp4")):
            m = re.match(r"^(id\d+)_", p.name)
            target_id = m.group(1) if m else p.stem.split("_")[0]
            meta = {
                "dataset": "Celeb-real",
                "target_id": target_id,
                "source_id": None,
                "identity": target_id,
            }
            manifest.append((str(p), 0.0, meta))
            counts["celeb_real"] += 1

    # 2. YouTube-real
    if yt_real_dir.exists():
        for p in sorted(yt_real_dir.glob("*.mp4")):
            yt_id = f"yt_{p.stem}"
            meta = {
                "dataset": "YouTube-real",
                "target_id": yt_id,
                "source_id": None,
                "identity": yt_id,
            }
            manifest.append((str(p), 0.0, meta))
            counts["youtube_real"] += 1

    counts["total_real"] = counts["celeb_real"] + counts["youtube_real"]

    # 3. Celeb-synthesis
    if celeb_synth_dir.exists():
        for p in sorted(celeb_synth_dir.glob("*.mp4")):
            m = re.match(r"^(id\d+)_(id\d+)_", p.name)
            if m:
                target_id, source_id = m.group(1), m.group(2)
            else:
                parts = p.stem.split("_")
                target_id = parts[0]
                source_id = parts[1] if len(parts) > 1 else None
            meta = {
                "dataset": "Celeb-synthesis",
                "target_id": target_id,
                "source_id": source_id,
                "identity": target_id,
            }
            manifest.append((str(p), 1.0, meta))
            counts["celeb_synthesis"] += 1

    counts["total_clips"] = len(manifest)
    return manifest, counts


def monitor_pool_cpu(pids: List[int], stop_event, interval: float = 0.5) -> List[str]:
    """Polls CPU utilization of worker PIDs using ps."""
    import subprocess
    snapshots = []
    pid_str = ",".join(str(p) for p in pids)
    while not stop_event.is_set():
        try:
            cmd = ["ps", "-p", pid_str, "-o", "pid,%cpu,%mem,time,comm"]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.stdout.strip():
                snapshots.append(res.stdout.strip())
        except Exception:
            pass
        time.sleep(interval)
    return snapshots


def run_benchmark(num_clips: int = 8, num_workers: int = 4):
    """
    Runs small benchmark on Celeb-DF clips to verify worker processes and measure clips/sec throughput.
    Monitors live CPU utilization via ps.
    """
    import threading

    manifest, counts = collect_celebdf_manifest()
    print("=" * 70)
    print("CELEB-DF VISUAL EXTRACTION BENCHMARK")
    print("=" * 70)
    print(f"Manifest counts in data/raw: {counts['celeb_real']} Celeb-real, {counts['youtube_real']} YouTube-real, {counts['celeb_synthesis']} Celeb-synthesis")
    print(f"Total available clips: {counts['total_clips']}")

    # Pick balanced subset: half real, half fake
    reals = [m for m in manifest if m[1] == 0.0][: num_clips // 2]
    fakes = [m for m in manifest if m[1] == 1.0][: num_clips // 2]
    batch = reals + fakes

    print(f"\nBenchmarking {len(batch)} clips across {num_workers} worker processes (macOS spawn)...")

    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()

    with ctx.Pool(processes=num_workers) as pool:
        # Get worker PIDs
        worker_pids = [p.pid for p in pool._pool if p.pid is not None]
        print(f"Active Worker PIDs: {worker_pids}")

        stop_event = threading.Event()
        cpu_snapshots: List[str] = []

        def _monitor():
            snapshots = monitor_pool_cpu(worker_pids, stop_event, interval=0.4)
            cpu_snapshots.extend(snapshots)

        mon_thread = threading.Thread(target=_monitor, daemon=True)
        mon_thread.start()

        # Run extraction
        results = []
        for idx, res in enumerate(pool.imap(process_one, batch)):
            status = "PASS" if res["success"] else f"FAIL ({res.get('reason')})"
            clip_name = Path(res["path"]).name
            print(f"  [{idx+1}/{len(batch)}] {clip_name} ({res['meta']['dataset']}): {status}")
            results.append(res)

        stop_event.set()
        mon_thread.join(timeout=1.0)

    t1 = time.perf_counter()
    elapsed = t1 - t0
    throughput = len(batch) / elapsed
    passes = sum(1 for r in results if r["success"])

    print("\n" + "─" * 70)
    print("BENCHMARK RESULTS")
    print("─" * 70)
    print(f"Processed:   {len(batch)} clips ({passes} valid, {len(batch) - passes} filtered)")
    print(f"Time Taken:  {elapsed:.2f} seconds")
    print(f"Throughput:  {throughput:.2f} clips/second ({throughput * num_workers:.1f} video frames/sec effective)")
    print("─" * 70)

    if cpu_snapshots:
        print("\nLive CPU Utilization Snapshot (via ps during benchmark):")
        print(cpu_snapshots[-1])
        print("─" * 70)

    return throughput, results


def run_full_extraction(
    num_workers: int = 8,
    output_path: Path = PROCESSED_DATA_DIR / "celebdf_visual.pt",
    failure_log_path: Path = PROCESSED_DATA_DIR / "celebdf_extraction_failures.csv",
    chunks_dir: Path = PROCESSED_DATA_DIR / "celebdf_chunks",
    chunk_size: int = 250,
):
    """
    Executes visual extraction over the full Celeb-DF corpus in parallel with incremental checkpointing.
    Saves chunks to chunks_dir every chunk_size items to allow lossless resume if interrupted.
    Persists consolidated trajectories to output_path and logs failures to failure_log_path.
    """
    manifest, counts = collect_celebdf_manifest()
    total_clips = len(manifest)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Scan existing chunks to resume if partially extracted
    already_processed_paths = set()
    existing_chunks = sorted(list(chunks_dir.glob("chunk_*.pt")))
    successful_from_chunks = []

    for c_path in existing_chunks:
        try:
            c_data = torch.load(c_path, map_location="cpu")
            paths = c_data.get("file_paths", [])
            already_processed_paths.update(paths)
            for idx, p in enumerate(paths):
                successful_from_chunks.append({
                    "lip_sequence": c_data["lip_sequences"][idx].numpy(),
                    "label": float(c_data["labels"][idx].item()),
                    "path": p,
                    "meta": {
                        "identity": c_data["identities"][idx],
                        "target_id": c_data["target_ids"][idx],
                        "source_id": c_data["source_ids"][idx],
                        "dataset": c_data["datasets"][idx],
                    },
                })
        except Exception as e:
            print(f"[!] Warning: failed to load existing chunk {c_path}: {e}")

    # Also load already failed paths if failure log exists
    if failure_log_path.exists():
        with open(failure_log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) > 0:
                    already_processed_paths.add(row[0])
    else:
        with open(failure_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label", "dataset", "reason", "target_id", "source_id"])

    remaining_manifest = [m for m in manifest if m[0] not in already_processed_paths]

    print("=" * 70)
    print("STARTING FULL CELEB-DF VISUAL FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Corpus breakdown: {counts['total_real']} Real ({counts['celeb_real']} Celeb-real, {counts['youtube_real']} YouTube-real)")
    print(f"                  {counts['celeb_synthesis']} Fake (Celeb-synthesis)")
    print(f"Total clips:      {total_clips}")
    print(f"Already done:     {len(already_processed_paths)} ({len(successful_from_chunks)} valid in {len(existing_chunks)} chunks)")
    print(f"Remaining:        {len(remaining_manifest)} clips to process")
    print(f"Workers:          {num_workers} parallel worker processes (spawn)")
    print(f"Target file:      {output_path}")
    print("=" * 70)

    if not remaining_manifest:
        print("[✓] All clips already extracted in chunks! Consolidating directly...")
    else:
        ctx = mp.get_context("spawn")
        t0 = time.perf_counter()

        current_chunk_successful = []
        failure_counts = {}
        chunk_idx = len(existing_chunks)
        processed_session = 0

        def save_chunk(items, idx):
            if not items:
                return
            c_file = chunks_dir / f"chunk_{idx:05d}.pt"
            c_lip_seqs = torch.stack([torch.from_numpy(s["lip_sequence"]) for s in items])
            c_labels = torch.tensor([s["label"] for s in items], dtype=torch.float32).unsqueeze(1)
            c_identities = [s["meta"]["identity"] for s in items]
            c_target_ids = [s["meta"]["target_id"] for s in items]
            c_source_ids = [s["meta"]["source_id"] for s in items]
            c_datasets = [s["meta"]["dataset"] for s in items]
            c_file_paths = [s["path"] for s in items]

            torch.save({
                "lip_sequences": c_lip_seqs,
                "labels": c_labels,
                "identities": c_identities,
                "target_ids": c_target_ids,
                "source_ids": c_source_ids,
                "datasets": c_datasets,
                "file_paths": c_file_paths,
            }, c_file)

        with ctx.Pool(processes=num_workers) as pool:
            for idx, res in enumerate(pool.imap_unordered(process_one, remaining_manifest, chunksize=4)):
                processed_session += 1
                if res["success"]:
                    current_chunk_successful.append(res)
                    successful_from_chunks.append(res)
                else:
                    reason = res.get("reason", "UNKNOWN")
                    failure_counts[reason] = failure_counts.get(reason, 0) + 1
                    with open(failure_log_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        meta = res.get("meta", {})
                        writer.writerow([
                            res.get("path"),
                            res.get("label"),
                            meta.get("dataset"),
                            reason,
                            meta.get("target_id"),
                            meta.get("source_id"),
                        ])

                # Save chunk if batch full
                if len(current_chunk_successful) >= chunk_size:
                    save_chunk(current_chunk_successful, chunk_idx)
                    chunk_idx += 1
                    current_chunk_successful = []

                if processed_session % 100 == 0 or processed_session == len(remaining_manifest):
                    elapsed = time.perf_counter() - t0
                    rate = processed_session / elapsed
                    rem = (len(remaining_manifest) - processed_session) / rate if rate > 0 else 0
                    print(
                        f"[{processed_session}/{len(remaining_manifest)}] "
                        f"Total Valid: {len(successful_from_chunks)} | "
                        f"Rate: {rate:.2f} clips/s | "
                        f"ETA: {rem/60:.1f} min"
                    )

            # Flush remaining in current chunk
            if current_chunk_successful:
                save_chunk(current_chunk_successful, chunk_idx)

    # Consolidate all successful items to final PyTorch PT file
    if not successful_from_chunks:
        raise RuntimeError("Extraction failed: zero valid clips extracted!")

    print("\nConsolidating all extracted chunks into final dataset tensor...")
    lip_seqs = torch.stack([torch.from_numpy(s["lip_sequence"]) for s in successful_from_chunks])
    labels = torch.tensor([s["label"] for s in successful_from_chunks], dtype=torch.float32).unsqueeze(1)
    identities = [s["meta"]["identity"] for s in successful_from_chunks]
    target_ids = [s["meta"]["target_id"] for s in successful_from_chunks]
    source_ids = [s["meta"]["source_id"] for s in successful_from_chunks]
    datasets = [s["meta"]["dataset"] for s in successful_from_chunks]
    file_paths = [s["path"] for s in successful_from_chunks]

    save_dict = {
        "lip_sequences": lip_seqs,
        "labels": labels,
        "identities": identities,
        "target_ids": target_ids,
        "source_ids": source_ids,
        "datasets": datasets,
        "file_paths": file_paths,
    }

    torch.save(save_dict, output_path)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successfully extracted:   {len(successful_from_chunks)} / {total_clips} clips ({len(successful_from_chunks)/total_clips*100:.1f}%)")
    print(f"Tensor shape (lip_seqs):  {list(lip_seqs.shape)}")
    print(f"Labels shape:             {list(labels.shape)}")
    print(f"Real / Fake balance:      {int((labels==0).sum())} real / {int((labels==1).sum())} fake")
    print(f"Failures logged to:       {failure_log_path}")
    print(f"Saved dataset file:       {output_path}")
    print("=" * 70)

    return save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract visual-only trajectories from Celeb-DF v2")
    parser.add_argument("--benchmark", action="store_true", help="Run small benchmark batch (8 clips)")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of worker processes")
    parser.add_argument("--num-clips", type=int, default=8, help="Number of clips for benchmark")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(num_clips=args.num_clips, num_workers=min(args.num_workers, args.num_clips))
    else:
        run_full_extraction(num_workers=args.num_workers)

