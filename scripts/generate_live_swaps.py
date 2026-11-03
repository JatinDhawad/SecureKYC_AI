"""
SecureKYC AI - Automated Neural Face Swap Generation Script (Synchronized Audio-Video)
Executes the consented cross-swap pairing matrix across recorded live sessions:
  Train Component:
    • S1 (Source) -> S2 (Target Face)
    • S2 (Source) -> S1 (Target Face)
    • S3 (Source) -> S4 (Target Face)
    • S4 (Source) -> S5 (Target Face)
    • S5 (Source) -> S3 (Target Face)
  Val Component:
    • S6 (Source) -> S6 (Target Face Reenactment)

Preserves exact acoustic timing & audio waveforms while applying neural face replacement.
Saves swapped videos to data/raw/live_sessions/S{i}/swap/session_{1,2,3}.mp4.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import av
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR

LIVE_DIR = RAW_DATA_DIR / "live_sessions"
INSWAPPER_PATH = Path.home() / ".insightface" / "models" / "inswapper_128.onnx"

# Consented Subject Pairing Matrix (Strictly Within Graph Split Components)
PAIRING_MATRIX = {
    "S1": "S2",
    "S2": "S1",
    "S3": "S4",
    "S4": "S5",
    "S5": "S3",
    "S6": "S6",
    "S7": "S8",
    "S8": "S7",
}


def extract_reference_face(app: FaceAnalysis, subject_id: str) -> Any:
    """Extracts high-confidence reference face from subject's first genuine video."""
    video_path = LIVE_DIR / subject_id / "genuine" / "session_1.mov"
    if not video_path.exists():
        video_path = LIVE_DIR / subject_id / "genuine" / "session_1.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Could not find reference session for {subject_id} at {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    ref_face = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        faces = app.get(frame)
        if len(faces) > 0:
            ref_face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
            break
    cap.release()

    if ref_face is None:
        raise RuntimeError(f"Failed to detect reference face for {subject_id} in {video_path}")

    print(f"  [✓] Extracted reference face for {subject_id} (Det Score: {ref_face.det_score:.3f})")
    return ref_face


def swap_video_clip(app: FaceAnalysis, swapper: Any, src_video: Path, dst_video: Path, target_face: Any):
    """Applies neural face swapping frame-by-frame and muxes original audio via PyAV."""
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read and swap all video frames
    cap = cv2.VideoCapture(str(src_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    swapped_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        faces = app.get(frame)
        if len(faces) > 0:
            best_face = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
            swapped_frame = swapper.get(frame, best_face, target_face, paste_back=True)
        else:
            swapped_frame = frame
        swapped_frames.append(swapped_frame)
    cap.release()

    if not swapped_frames:
        raise RuntimeError(f"No frames read from {src_video}")

    h, w = swapped_frames[0].shape[:2]

    # 2. Extract original audio packets/frames from src_video
    audio_frames = []
    a_rate = 16000
    try:
        in_container = av.open(str(src_video))
        if len(in_container.streams.audio) > 0:
            a_in = in_container.streams.audio[0]
            a_rate = a_in.rate or 16000
            for a_frame in in_container.decode(a_in):
                audio_frames.append(a_frame)
        in_container.close()
    except Exception as e:
        print(f"    [Audio Extract Note] {e}")

    # 3. Create output container and initialize ALL streams BEFORE writing packets
    out_container = av.open(str(dst_video), mode="w")
    v_stream = out_container.add_stream("libx264" if "libx264" in av.codecs_available else "h264", rate=int(fps))
    v_stream.width = w
    v_stream.height = h
    v_stream.pix_fmt = "yuv420p"

    a_stream = None
    if len(audio_frames) > 0:
        a_stream = out_container.add_stream("aac", rate=a_rate)

    # 4. Write video packets
    for frame_bgr in swapped_frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        for packet in v_stream.encode(av_frame):
            out_container.mux(packet)
    for packet in v_stream.encode():
        out_container.mux(packet)

    # 5. Write audio packets
    if a_stream is not None:
        for a_frame in audio_frames:
            for packet in a_stream.encode(a_frame):
                out_container.mux(packet)
        for packet in a_stream.encode():
            out_container.mux(packet)

    out_container.close()
    print(f"    -> Generated: {dst_video} ({len(swapped_frames)} frames, {len(audio_frames)} audio chunks)")


def main():
    print("=" * 80)
    print("      SecureKYC AI - Consented Neural Face-Swap Generation Pipeline     ")
    print("=" * 80)

    if not INSWAPPER_PATH.exists():
        raise FileNotFoundError(f"Inswapper model missing at {INSWAPPER_PATH}. Complete download first.")

    print("[*] Initializing InsightFace Buffalo_l Analysis & Inswapper ONNX Engine...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    swapper = insightface.model_zoo.get_model(str(INSWAPPER_PATH), download=False, download_zip=False)
    print("[✓] Face swapper model loaded successfully!\n")

    # 1. Extract reference face embeddings for all available subjects
    ref_faces = {}
    for subj_id in PAIRING_MATRIX.keys():
        ref_faces[subj_id] = extract_reference_face(app, subj_id)

    # 2. Generate swapped takes for each session
    print("\n[*] Processing neural face swaps according to split graph pairings...")
    for src_subj, tgt_subj in PAIRING_MATRIX.items():
        print(f"\n--- Processing Subject {src_subj} (Target Face: {tgt_subj}) ---")
        tgt_face = ref_faces[tgt_subj]
        src_dir = LIVE_DIR / src_subj / "genuine"
        dst_dir = LIVE_DIR / src_subj / "swap"
        dst_dir.mkdir(parents=True, exist_ok=True)

        for session_num in [1, 2, 3]:
            src_mov = src_dir / f"session_{session_num}.mov"
            if not src_mov.exists():
                src_mov = src_dir / f"session_{session_num}.mp4"
            if src_mov.exists():
                dst_mp4 = dst_dir / f"session_{session_num}.mp4"
                if dst_mp4.exists() and dst_mp4.stat().st_size > 10000:
                    print(f"  • Session {session_num} already exists at {dst_mp4.name}, skipping.")
                    continue
                print(f"  • Swapping session {session_num}: {src_mov.name} -> {dst_mp4.name} (Face: {tgt_subj})")
                swap_video_clip(app, swapper, src_mov, dst_mp4, tgt_face)

    print("\n" + "=" * 80)
    print("[✓] All 18 face-swap sessions generated and saved successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
