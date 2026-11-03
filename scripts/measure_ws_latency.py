"""
Script: scripts/measure_ws_latency.py
Measures post-utterance WebSocket verification latency on /ws/verify.

Protocol:
1. Runs full challenge-response session on 5 genuine demo clips (real_01..real_05)
   and 4 fake clips (fake_01..fake_04) from data/demo_clips/.
2. Discards 2 warm-up runs per clip, then measures 5 repeated runs per clip.
3. Records end-of-utterance -> verdict-received timestamp (client side) and
   server-side processing time (server side), in ms.
4. Records the first (cold) session separately.
5. Prints every raw per-run number, then mean ± std, min, max, and n.
6. Writes comprehensive results to data/processed/ws_latency_results.json.
7. Plainly states client transport (in-process vs live socket) and device (mps/cpu).
"""

import sys
import os
import json
import time
import base64
import wave
import io
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import quote

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_device, PROCESSED_DATA_DIR
from app.main import app, load_model_checkpoints


# Canonical transcripts from TIMIT
TRANSCRIPTS = {
    "sa2": "Don't ask me to carry an oily rag like that.",
    "sx369": "bulk paper over the living room murals.",
}

# Curated benchmark demo clips specification
DEFAULT_DEMO_CLIPS = [
    # Genuine clips (5 distinct clips covering real_01..real_04 plus real_05)
    {
        "filename": "real_01_fadg0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fadg0",
    },
    {
        "filename": "real_02_fcmh0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fcmh0",
    },
    {
        "filename": "real_03_fdac1_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fdac1",
    },
    {
        "filename": "real_04_fgjd0_sx369.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sx369",
        "prompt": TRANSCRIPTS["sx369"],
        "speaker": "fgjd0",
    },
    {
        "filename": "real_05_fjre0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fjre0",
    },
    {
        "filename": "real_06_fdms0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fdms0",
    },
    # Deepfake clips (4 distinct clips covering fake_01..fake_04)
    {
        "filename": "fake_01_fadg0_sa2_fram1.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fadg0 (swap fram1)",
    },
    {
        "filename": "fake_02_faks0_sa2_fedw0.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "faks0 (swap fedw0)",
    },
    {
        "filename": "fake_03_fcft0_sa2_fdrd1.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fcft0 (swap fdrd1)",
    },
    {
        "filename": "fake_04_msjs1_sx369_mrgg0.mp4",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sx369",
        "prompt": TRANSCRIPTS["sx369"],
        "speaker": "msjs1 (swap mrgg0)",
    },
]

EXTRA_REAL_CLIPS = [
    {
        "filename": "real_06_fdms0_sa2.mp4",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "prompt": TRANSCRIPTS["sa2"],
        "speaker": "fdms0",
    },
]


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Computes mean, sample standard deviation (ddof=1), min, max, and n."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    return {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "n": len(arr),
    }


def prepare_clip_data(clip_info: Dict[str, Any], demo_dir: Path) -> Dict[str, Any]:
    """Loads video frames and audio for a clip, preparing Base64 payloads in memory."""
    clip_path = demo_dir / clip_info["filename"]
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    # 1. Decode video frames
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_ms = 1000.0 / fps
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {clip_path}")

    # Pre-encode frames to JPEG Base64
    frame_payloads = []
    for idx, frame in enumerate(frames):
        _, buf = cv2.imencode(".jpg", frame)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        frame_payloads.append({
            "type": "video_frame",
            "image": b64_str,
            "timestamp_ms": idx * frame_ms,
        })

    # 2. Extract 16kHz mono WAV audio via ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", str(clip_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    audio_bytes = proc.stdout
    audio_b64 = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "filename": clip_info["filename"],
        "ground_truth": clip_info["ground_truth"],
        "sentence_id": clip_info["sentence_id"],
        "prompt": clip_info["prompt"],
        "speaker": clip_info["speaker"],
        "frame_payloads": frame_payloads,
        "audio_b64": audio_b64,
        "sample_rate": 16000,
        "total_frames": len(frame_payloads),
        "fps": fps,
    }


def execute_session_inprocess(
    client: Any,
    clip_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs challenge-response verification session via FastAPI TestClient (in-process ASGI)."""
    prompt = clip_data["prompt"]
    ws_url = f"/ws/verify?prompt={quote(prompt)}"

    with client.websocket_connect(ws_url) as ws:
        init_msg = ws.receive_json()
        assert init_msg.get("type") == "challenge_init"

        # Stream all video frames
        for p in clip_data["frame_payloads"]:
            ws.send_json(p)
            ack = ws.receive_json()
            assert ack.get("type") == "frame_ack"

        # End of utterance -> measure client latency
        client_start_t = time.perf_counter()
        ws.send_json({
            "type": "utterance_complete",
            "audio_b64": clip_data["audio_b64"],
            "sample_rate": clip_data["sample_rate"],
            "prompt": prompt,
        })
        verdict = ws.receive_json()
        client_end_t = time.perf_counter()

    client_ms = (client_end_t - client_start_t) * 1000.0
    server_ms = float(verdict.get("latency_ms", 0.0))

    return {
        "client_latency_ms": round(client_ms, 2),
        "server_latency_ms": round(server_ms, 2),
        "verdict_text": verdict.get("verdict_text"),
        "is_deepfake": verdict.get("is_deepfake"),
        "deepfake_probability": verdict.get("deepfake_probability"),
        "prompt_verified": verdict.get("prompt_verified"),
        "p_visual": verdict.get("p_visual"),
        "p_desync": verdict.get("p_desync"),
        "desync_score": verdict.get("desync_score"),
        "raw_verdict": verdict,
    }


def execute_session_socket(
    ws_uri: str,
    clip_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Runs challenge-response verification session over real TCP WebSocket."""
    import websockets.sync.client as ws_sync

    prompt = clip_data["prompt"]
    ws_url = f"{ws_uri}?prompt={quote(prompt)}"

    with ws_sync.connect(ws_url) as ws:
        init_str = ws.recv()
        init_msg = json.loads(init_str)
        assert init_msg.get("type") == "challenge_init"

        for p in clip_data["frame_payloads"]:
            ws.send(json.dumps(p))
            ack_str = ws.recv()
            ack = json.loads(ack_str)
            assert ack.get("type") == "frame_ack"

        client_start_t = time.perf_counter()
        ws.send(json.dumps({
            "type": "utterance_complete",
            "audio_b64": clip_data["audio_b64"],
            "sample_rate": clip_data["sample_rate"],
            "prompt": prompt,
        }))
        verdict_str = ws.recv()
        client_end_t = time.perf_counter()
        verdict = json.loads(verdict_str)

    client_ms = (client_end_t - client_start_t) * 1000.0
    server_ms = float(verdict.get("latency_ms", 0.0))

    return {
        "client_latency_ms": round(client_ms, 2),
        "server_latency_ms": round(server_ms, 2),
        "verdict_text": verdict.get("verdict_text"),
        "is_deepfake": verdict.get("is_deepfake"),
        "deepfake_probability": verdict.get("deepfake_probability"),
        "prompt_verified": verdict.get("prompt_verified"),
        "p_visual": verdict.get("p_visual"),
        "p_desync": verdict.get("p_desync"),
        "desync_score": verdict.get("desync_score"),
        "raw_verdict": verdict,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure WebSocket verification latency on SecureKYC AI.")
    parser.add_argument(
        "--transport",
        choices=["in-process", "socket"],
        default="in-process",
        help="WebSocket transport mechanism: in-process (TestClient ASGI) or real socket (websockets TCP).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for socket transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for socket transport (default: 8000).",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
        help="Number of discarded warm-up runs per clip (default: 2).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of measured runs per clip (default: 5).",
    )
    parser.add_argument(
        "--include-all-real",
        action="store_true",
        help="Include all available genuine clips in data/demo_clips/ (e.g., real_06).",
    )
    args = parser.parse_args()

    demo_dir = PROJECT_ROOT / "data" / "demo_clips"
    if not demo_dir.exists():
        print(f"[!] Error: demo directory does not exist: {demo_dir}")
        sys.exit(1)

    # Resolve clips to test
    clips_spec = list(DEFAULT_DEMO_CLIPS)
    if args.include_all_real:
        for extra in EXTRA_REAL_CLIPS:
            if (demo_dir / extra["filename"]).exists() and extra not in clips_spec:
                clips_spec.append(extra)

    # Hardware & Device Verification
    dev = get_device()
    dev_str = getattr(dev, "type", str(dev)).lower()
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    torch_version = torch.__version__

    # Setup client
    server_process = None
    if args.transport == "in-process":
        from fastapi.testclient import TestClient
        load_model_checkpoints()
        client = TestClient(app)
        run_fn = lambda cdata: execute_session_inprocess(client, cdata)
        transport_label = "in-process (FastAPI TestClient ASGI WebSocket)"
    else:
        # Check if live server is reachable
        import urllib.request
        health_url = f"http://{args.host}:{args.port}/health"
        server_ready = False
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    server_ready = True
        except Exception:
            server_ready = False

        if not server_ready:
            print(f"[*] Starting local uvicorn background server on {args.host}:{args.port}...")
            server_process = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn",
                    "app.main:app",
                    "--host", args.host,
                    "--port", str(args.port),
                    "--log-level", "warning",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait for readiness
            for _ in range(30):
                time.sleep(0.5)
                try:
                    with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                        if resp.status == 200:
                            server_ready = True
                            break
                except Exception:
                    pass
            if not server_ready:
                if server_process:
                    server_process.kill()
                print("[!] Error: Could not start or connect to uvicorn server.")
                sys.exit(1)
            print("[+] Live server ready.")

        ws_uri = f"ws://{args.host}:{args.port}/ws/verify"
        run_fn = lambda cdata: execute_session_socket(ws_uri, cdata)
        transport_label = f"real socket ({ws_uri})"

    print("=" * 88)
    print("  SECUREKYC AI - WEBSOCKET POST-UTTERANCE LATENCY BENCHMARK")
    print("=" * 88)
    print("System Configuration:")
    print(f"  • Client Transport Mode   : {transport_label}")
    print(f"  • Execution Device        : {dev_str} (MPS Available: {mps_available})")
    print(f"  • PyTorch Version         : {torch_version}")
    print(f"  • Discarded Warm-up Runs  : {args.warmups} per clip")
    print(f"  • Measurement Runs        : {args.repeats} per clip")
    num_gen = sum(1 for c in clips_spec if c["ground_truth"] == "GENUINE")
    num_fake = sum(1 for c in clips_spec if c["ground_truth"] == "DEEPFAKE")
    print(f"  • Total Test Clips        : {len(clips_spec)} ({num_gen} Genuine, {num_fake} Fake)")
    print("=" * 88)

    # 1. Preload & encode all clip payloads into memory
    print("\n[*] Pre-loading video frames and audio for all demo clips into memory...")
    loaded_clips = []
    for spec in clips_spec:
        cdata = prepare_clip_data(spec, demo_dir)
        loaded_clips.append(cdata)
        print(f"  [+] Loaded {spec['filename']:<32} ({cdata['total_frames']} frames, prompt: '{cdata['prompt'][:30]}...')")

    # 2. First (Cold) Session Execution - Recorded Separately
    print("\n" + "=" * 88)
    print("  INITIAL (COLD) SESSION - RECORDED SEPARATELY")
    print("=" * 88)
    cold_clip = loaded_clips[0]
    print(f"Executing cold initial session on: {cold_clip['filename']} (Ground Truth: {cold_clip['ground_truth']})")
    cold_res = run_fn(cold_clip)
    delta_cold = cold_res["client_latency_ms"] - cold_res["server_latency_ms"]
    print(f"  • Client Latency (End-of-Utterance -> Verdict) : {cold_res['client_latency_ms']:8.2f} ms")
    print(f"  • Server Processing Latency (Internal Timer)   : {cold_res['server_latency_ms']:8.2f} ms")
    print(f"  • Transport & Deserialization Delta            : {delta_cold:8.2f} ms")
    print(f"  • Verdict Returned                             : {cold_res['verdict_text']}")
    print(f"  • Deepfake Probability                         : {cold_res['deepfake_probability']}")
    print("=" * 88)

    cold_session_record = {
        "clip": cold_clip["filename"],
        "ground_truth": cold_clip["ground_truth"],
        "client_latency_ms": cold_res["client_latency_ms"],
        "server_latency_ms": cold_res["server_latency_ms"],
        "transport_delta_ms": round(delta_cold, 2),
        "verdict_text": cold_res["verdict_text"],
        "is_deepfake": cold_res["is_deepfake"],
        "deepfake_probability": cold_res["deepfake_probability"],
    }

    # 3. Warm-up and Measurement Runs
    print("\n" + "=" * 88)
    print("  WARM MEASUREMENT RUNS (Per-Clip Trials after Discarded Warm-ups)")
    print("=" * 88)

    all_raw_runs = []
    per_clip_stats = {}

    for idx, cdata in enumerate(loaded_clips, start=1):
        cname = cdata["filename"]
        gt = cdata["ground_truth"]
        print(f"\n[{idx}/{len(loaded_clips)}] Clip: {cname} (Ground Truth: {gt})")

        # Discarded warm-up runs
        for w_idx in range(1, args.warmups + 1):
            w_res = run_fn(cdata)
            print(f"  [Warmup {w_idx} Discarded] Client: {w_res['client_latency_ms']:6.2f} ms | Server: {w_res['server_latency_ms']:6.2f} ms | Verdict: {w_res['verdict_text']}")

        # Measured repetitions
        clip_client_times = []
        clip_server_times = []

        for r_idx in range(1, args.repeats + 1):
            m_res = run_fn(cdata)
            cli_t = m_res["client_latency_ms"]
            srv_t = m_res["server_latency_ms"]
            clip_client_times.append(cli_t)
            clip_server_times.append(srv_t)

            run_entry = {
                "clip": cname,
                "ground_truth": gt,
                "speaker": cdata["speaker"],
                "run_index": r_idx,
                "client_latency_ms": cli_t,
                "server_latency_ms": srv_t,
                "verdict_text": m_res["verdict_text"],
                "is_deepfake": m_res["is_deepfake"],
                "deepfake_probability": m_res["deepfake_probability"],
                "p_visual": m_res["p_visual"],
                "p_desync": m_res["p_desync"],
                "desync_score": m_res["desync_score"],
            }
            all_raw_runs.append(run_entry)

            print(
                f"  Run {r_idx}: Client = {cli_t:6.2f} ms | Server = {srv_t:6.2f} ms | "
                f"Verdict = {m_res['verdict_text']:<30} | P(fake) = {m_res['deepfake_probability']}"
            )

        # Clip summary stats
        c_cli_stats = compute_stats(clip_client_times)
        c_srv_stats = compute_stats(clip_server_times)
        per_clip_stats[cname] = {
            "ground_truth": gt,
            "speaker": cdata["speaker"],
            "client_ms": c_cli_stats,
            "server_ms": c_srv_stats,
        }
        print(f"  -- Clip Summary (n={len(clip_client_times)}):")
        print(f"     Client : Mean ± Std = {c_cli_stats['mean']} ± {c_cli_stats['std']} ms (min: {c_cli_stats['min']}, max: {c_cli_stats['max']})")
        print(f"     Server : Mean ± Std = {c_srv_stats['mean']} ± {c_srv_stats['std']} ms (min: {c_srv_stats['min']}, max: {c_srv_stats['max']})")

    # 4. Global Summary Statistics
    all_cli_times = [r["client_latency_ms"] for r in all_raw_runs]
    all_srv_times = [r["server_latency_ms"] for r in all_raw_runs]

    genuine_cli_times = [r["client_latency_ms"] for r in all_raw_runs if r["ground_truth"] == "GENUINE"]
    genuine_srv_times = [r["server_latency_ms"] for r in all_raw_runs if r["ground_truth"] == "GENUINE"]

    fake_cli_times = [r["client_latency_ms"] for r in all_raw_runs if r["ground_truth"] == "DEEPFAKE"]
    fake_srv_times = [r["server_latency_ms"] for r in all_raw_runs if r["ground_truth"] == "DEEPFAKE"]

    summary_stats = {
        "all_warm_runs": {
            "client_ms": compute_stats(all_cli_times),
            "server_ms": compute_stats(all_srv_times),
        },
        "genuine_warm_runs": {
            "client_ms": compute_stats(genuine_cli_times),
            "server_ms": compute_stats(genuine_srv_times),
        },
        "fake_warm_runs": {
            "client_ms": compute_stats(fake_cli_times),
            "server_ms": compute_stats(fake_srv_times),
        },
    }

    # Print Final Summary Table
    print("\n" + "=" * 88)
    print("  FINAL LATENCY BENCHMARK SUMMARY & AGGREGATE STATISTICS")
    print("=" * 88)
    print(f"Client Transport Mode  : {transport_label}")
    print(f"Execution Device       : {dev_str} (MPS Available: {mps_available})")
    print(f"Cold Initial Session   : Client = {cold_session_record['client_latency_ms']:.2f} ms | Server = {cold_session_record['server_latency_ms']:.2f} ms")
    print("-" * 88)

    print("Per-Clip Warm Latency Averages (5 runs each after 2 discarded warm-ups):")
    print(f"{'Clip Filename':<32} | {'GT':<8} | {'Client Mean ± Std (ms)':<23} | {'Server Mean ± Std (ms)':<23}")
    print("-" * 88)
    for cname, cinfo in per_clip_stats.items():
        cli_str = f"{cinfo['client_ms']['mean']} ± {cinfo['client_ms']['std']} ms"
        srv_str = f"{cinfo['server_ms']['mean']} ± {cinfo['server_ms']['std']} ms"
        print(f"{cname:<32} | {cinfo['ground_truth']:<8} | {cli_str:<23} | {srv_str:<23}")
    print("-" * 88)

    print("\nAggregate Partition Statistics:")
    print(f"• Genuine Clips (N = {len(genuine_cli_times)} runs):")
    print(f"    Client Latency : Mean ± Std = {summary_stats['genuine_warm_runs']['client_ms']['mean']} ± {summary_stats['genuine_warm_runs']['client_ms']['std']} ms "
          f"[Min: {summary_stats['genuine_warm_runs']['client_ms']['min']}, Max: {summary_stats['genuine_warm_runs']['client_ms']['max']}, n: {summary_stats['genuine_warm_runs']['client_ms']['n']}]")
    print(f"    Server Latency : Mean ± Std = {summary_stats['genuine_warm_runs']['server_ms']['mean']} ± {summary_stats['genuine_warm_runs']['server_ms']['std']} ms "
          f"[Min: {summary_stats['genuine_warm_runs']['server_ms']['min']}, Max: {summary_stats['genuine_warm_runs']['server_ms']['max']}, n: {summary_stats['genuine_warm_runs']['server_ms']['n']}]")

    print(f"• Deepfake Clips (N = {len(fake_cli_times)} runs):")
    print(f"    Client Latency : Mean ± Std = {summary_stats['fake_warm_runs']['client_ms']['mean']} ± {summary_stats['fake_warm_runs']['client_ms']['std']} ms "
          f"[Min: {summary_stats['fake_warm_runs']['client_ms']['min']}, Max: {summary_stats['fake_warm_runs']['client_ms']['max']}, n: {summary_stats['fake_warm_runs']['client_ms']['n']}]")
    print(f"    Server Latency : Mean ± Std = {summary_stats['fake_warm_runs']['server_ms']['mean']} ± {summary_stats['fake_warm_runs']['server_ms']['std']} ms "
          f"[Min: {summary_stats['fake_warm_runs']['server_ms']['min']}, Max: {summary_stats['fake_warm_runs']['server_ms']['max']}, n: {summary_stats['fake_warm_runs']['server_ms']['n']}]")

    print(f"• Overall Combined (All Warm Runs, N = {len(all_cli_times)}):")
    print(f"    Client Latency : Mean ± Std = {summary_stats['all_warm_runs']['client_ms']['mean']} ± {summary_stats['all_warm_runs']['client_ms']['std']} ms "
          f"[Min: {summary_stats['all_warm_runs']['client_ms']['min']}, Max: {summary_stats['all_warm_runs']['client_ms']['max']}, n: {summary_stats['all_warm_runs']['client_ms']['n']}]")
    print(f"    Server Latency : Mean ± Std = {summary_stats['all_warm_runs']['server_ms']['mean']} ± {summary_stats['all_warm_runs']['server_ms']['std']} ms "
          f"[Min: {summary_stats['all_warm_runs']['server_ms']['min']}, Max: {summary_stats['all_warm_runs']['server_ms']['max']}, n: {summary_stats['all_warm_runs']['server_ms']['n']}]")
    print("=" * 88)

    # 5. Output JSON Results
    results_json_path = PROCESSED_DATA_DIR / "ws_latency_results.json"
    results_json_path.parent.mkdir(parents=True, exist_ok=True)

    final_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "client_transport": transport_label,
            "device": dev_str,
            "mps_available": mps_available,
            "torch_version": torch_version,
            "warmup_runs_per_clip": args.warmups,
            "measurement_runs_per_clip": args.repeats,
            "num_clips": len(loaded_clips),
            "num_genuine_clips": num_gen,
            "num_fake_clips": num_fake,
            "total_warm_runs": len(all_raw_runs),
        },
        "cold_session": cold_session_record,
        "per_clip_statistics": per_clip_stats,
        "summary_statistics": summary_stats,
        "raw_runs": all_raw_runs,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[+] Comprehensive benchmark results successfully written to:\n    {results_json_path}")

    # Clean up server process if launched
    if server_process:
        server_process.terminate()
        server_process.wait()


if __name__ == "__main__":
    main()
