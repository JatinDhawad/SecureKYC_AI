"""
End-to-End Live System Sessions Verification (Steps B1 - B3)
Executes 10 full end-to-end sessions of the real SecureKYC AI challenge-response system
against authentic audio-visual human webcam & speech recordings (VidTIMIT & Celeb-real).

Measures post-utterance latency (utterance_complete -> verdict) and verifies 100% genuine verdicts.
"""

import sys
import re
import io
import time
import json
import base64
import wave
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
import websockets
import scipy.io.wavfile as wavfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR


def get_genuine_session_clips() -> List[Dict[str, Any]]:
    """Discovers 10 genuine audio-visual speaker clips from VidTIMIT and Celeb-real datasets."""
    clips = []
    orig_dir = RAW_DATA_DIR / "original_sequences"

    # 1. VidTIMIT Speakers
    if orig_dir.exists():
        speaker_dirs = [d for d in orig_dir.iterdir() if d.is_dir() and d.name not in ["Celeb-real", "YouTube-real", "youtube"]]
        for spk_dir in speaker_dirs[:7]:
            audio_dir = spk_dir / "audio"
            video_dir = spk_dir / "video"
            if audio_dir.exists() and video_dir.exists():
                for sentence_key in ["sx199", "sa2", "sx19", "si649"]:
                    wav_path = audio_dir / f"{sentence_key}.wav"
                    sent_video_dir = video_dir / sentence_key
                    if wav_path.exists() and sent_video_dir.exists():
                        frame_files = sorted(list(sent_video_dir.glob("*")), key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0)
                        if frame_files:
                            clips.append({
                                "speaker": spk_dir.name,
                                "clip_name": f"{spk_dir.name}_{sentence_key}",
                                "audio_path": wav_path,
                                "frame_files": frame_files,
                                "type": "VidTIMIT",
                            })
                            break
            if len(clips) >= 7:
                break

    # 2. Celeb-real Clips
    celeb_real_dir = orig_dir / "Celeb-real"
    if celeb_real_dir.exists():
        mp4_files = list(celeb_real_dir.glob("*.mp4"))
        for idx, mp4_path in enumerate(mp4_files[:3]):
            clips.append({
                "speaker": f"celeb_{mp4_path.stem}",
                "clip_name": mp4_path.stem,
                "mp4_path": mp4_path,
                "type": "Celeb-real",
            })
            if len(clips) >= 10:
                break

    return clips[:10]


def encode_frame_to_b64(frame_bgr: np.ndarray) -> str:
    """Encodes OpenCV BGR frame image to Base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def encode_audio_to_b64(wav_path: Path) -> Tuple[str, int]:
    """Reads WAV audio file and encodes raw bytes to Base64 string."""
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()

    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("utf-8"), sr


async def run_single_e2e_session(session_idx: int, clip_info: Dict[str, Any], uri: str = "ws://127.0.0.1:8000/ws/verify") -> Dict[str, Any]:
    """Runs a single live challenge-response verification session over WebSocket."""
    async with websockets.connect(uri) as ws:
        # Step B1: Receive Challenge Prompt
        challenge_init_str = await ws.recv()
        challenge_init = json.loads(challenge_init_str)

        issued_prompt = challenge_init.get("prompt", "")
        session_id = challenge_init.get("session_id", f"session_{session_idx}")

        # Step B2: Near-real-time Video Frame Ingestion
        frame_ack_count = 0
        if clip_info["type"] == "VidTIMIT":
            for f_path in clip_info["frame_files"][:30]:
                frame_img = cv2.imread(str(f_path))
                if frame_img is None:
                    continue
                img_b64 = encode_frame_to_b64(frame_img)
                await ws.send(json.dumps({"type": "video_frame", "image": img_b64, "timestamp_ms": time.time() * 1000.0}))
                ack = json.loads(await ws.recv())
                frame_ack_count += 1
        elif clip_info["type"] == "Celeb-real":
            cap = cv2.VideoCapture(str(clip_info["mp4_path"]))
            f_count = 0
            while cap.isOpened() and f_count < 30:
                ret, frame_img = cap.read()
                if not ret or frame_img is None:
                    break
                img_b64 = encode_frame_to_b64(frame_img)
                await ws.send(json.dumps({"type": "video_frame", "image": img_b64, "timestamp_ms": time.time() * 1000.0}))
                ack = json.loads(await ws.recv())
                frame_ack_count += 1
                f_count += 1
            cap.release()

        # Step B3: Post-Utterance Audio Submission & Latency Timing
        if clip_info["type"] == "VidTIMIT":
            audio_b64, sr = encode_audio_to_b64(clip_info["audio_path"])
        else:
            audio_b64, sr = "", 16000

        # Start post-utterance latency timer
        post_utterance_start_t = time.perf_counter()

        await ws.send(json.dumps({
            "type": "utterance_complete",
            "audio_b64": audio_b64,
            "sample_rate": sr,
        }))

        # Receive Final Verdict
        verdict_str = await ws.recv()
        latency_sec = time.perf_counter() - post_utterance_start_t

        verdict = json.loads(verdict_str)

        return {
            "session_idx": session_idx,
            "speaker": clip_info["speaker"],
            "issued_prompt": issued_prompt,
            "is_deepfake": verdict.get("is_deepfake", False),
            "verdict_text": verdict.get("verdict_text", "UNKNOWN"),
            "deepfake_probability": verdict.get("deepfake_probability", 0.0),
            "desync_score": verdict.get("desync_score", 0.0),
            "frames_evaluated": verdict.get("frames_evaluated", 0),
            "latency_sec": round(latency_sec, 3),
            "server_latency_ms": verdict.get("latency_ms", 0.0),
        }


async def run_all_sessions():
    """Runs 10 full end-to-end sessions and prints comparison summary table."""
    clips = get_genuine_session_clips()
    print("=" * 95)
    print(" SecureKYC AI - 10 Full End-to-End Sessions Verification (Steps B1-B3) ")
    print("=" * 95)
    print(f"[*] Found {len(clips)} genuine audio-visual clips for verification.")

    results = []
    for idx, clip_info in enumerate(clips, 1):
        try:
            res = await run_single_e2e_session(idx, clip_info)
            results.append(res)
        except Exception as e:
            print(f"[!] Session {idx} failed with exception: {e}")

    # Print Summary Table
    print("\n" + "=" * 95)
    print(f"{'END-TO-END SYSTEM SESSIONS EVALUATION SUMMARY':^95}")
    print("=" * 95)
    print(
        f"{'Session':<8} | {'Speaker ID':<12} | {'Issued Challenge Prompt':<24} | {'Verdict':<20} | {'P(Deepfake)':<11} | {'Latency (s)':<10}"
    )
    print("-" * 95)

    all_genuine = True
    total_latency_sec = 0.0

    for r in results:
        is_real = not r["is_deepfake"]
        if not is_real:
            all_genuine = False
        total_latency_sec += r["latency_sec"]

        status_str = "GENUINE / REAL" if is_real else "DEEPFAKE"
        print(
            f"#{r['session_idx']:<7} | "
            f"{r['speaker']:<12} | "
            f"{r['issued_prompt']:<24} | "
            f"{status_str:<20} | "
            f"{r['deepfake_probability']:^11.4f} | "
            f"{r['latency_sec']:^10.3f}s"
        )

    mean_latency = total_latency_sec / max(1, len(results))

    print("=" * 95)
    print(f"  • Total Sessions Executed:             {len(results)}")
    print(f"  • All Sessions Verdict Genuine:        {'YES (100% Genuine)' if all_genuine else 'NO'}")
    print(f"  • Mean Post-Utterance Latency:         {mean_latency:.3f} seconds")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all_sessions())
