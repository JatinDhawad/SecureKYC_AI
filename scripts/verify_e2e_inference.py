"""
SecureKYC AI - End-to-End Inference Verification with Known Real & Fake Clips
Empirically verifies:
1. Correct verdict direction on real and fake test clips.
2. Rejection of prompt mismatch injection (client prompt != session challenge).
3. Rejection of acoustic alignment failure (audio not matching issued challenge prompt).
"""

import sys
import re
import json
import base64
import wave
import time
from urllib.parse import quote
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, get_device
from app.main import app, load_model_checkpoints


def encode_frame_to_b64(frame_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


def encode_audio_to_b64(wav_path: Path) -> Tuple[str, int]:
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("utf-8"), sr


def run_e2e_session(
    client: TestClient,
    clip_name: str,
    frames: List[np.ndarray],
    audio_path: Path,
    prompt: str,
    expected_label: str,
    submitted_prompt: Optional[str] = None,
    fps: float = 25.0,
) -> dict:
    frame_ms = 1000.0 / fps
    print(f"\n{'='*75}")
    print(f" TESTING: {clip_name} (Expected Ground Truth: {expected_label})")
    print(f"{'='*75}")
    print(f"[*] Total frames: {len(frames)} at {fps} FPS ({frame_ms:.1f}ms/frame), Audio: {audio_path.name}")
    print(f"[*] Issued prompt: '{prompt}'")
    if submitted_prompt and submitted_prompt != prompt:
        print(f"[*] [SECURITY INJECTION TEST] Client submitted different prompt: '{submitted_prompt}'")

    ws_url = f"/ws/verify?prompt={quote(prompt)}"
    with client.websocket_connect(ws_url) as ws:
        init_msg = ws.receive_json()
        assert init_msg.get("type") == "challenge_init"
        assert init_msg.get("prompt") == prompt
        print(f"[*] Connected. Session ID: {init_msg.get('session_id')}, Challenge: '{init_msg.get('prompt')}'")

        # Stream all video frames
        for idx, frame in enumerate(frames):
            img_b64 = encode_frame_to_b64(frame)
            ws.send_json({
                "type": "video_frame",
                "image": img_b64,
                "timestamp_ms": idx * frame_ms,
            })
            ack = ws.receive_json()
            assert ack.get("type") == "frame_ack"

        print(f"[*] Streamed {len(frames)} frames with successful face/lip tracking ACKs.")

        # Complete utterance with audio and prompt
        audio_b64, sr = encode_audio_to_b64(audio_path)
        payload = {
            "type": "utterance_complete",
            "audio_b64": audio_b64,
            "sample_rate": sr,
        }
        if submitted_prompt is not None:
            payload["prompt"] = submitted_prompt

        ws.send_json(payload)
        verdict = ws.receive_json()

    print("\n[*] Full Pipeline Verdict:")
    print(f"    Verdict Text:        {verdict.get('verdict_text')}")
    print(f"    Is Deepfake:         {verdict.get('is_deepfake')}")
    print(f"    Prompt Verified:     {verdict.get('prompt_verified')}")
    print(f"    Deepfake Prob:       {verdict.get('deepfake_probability')}")
    print(f"    P(Visual):           {verdict.get('p_visual')}")
    print(f"    P(Desync):           {verdict.get('p_desync')}")
    print(f"    Desync Score:        {verdict.get('desync_score')}")
    print(f"    Max Penalty:         {verdict.get('max_penalty')}")
    print(f"    Failed Closure Ratio:{verdict.get('failed_closure_ratio')}")
    if "error" in verdict:
        print(f"    Error/Flag Note:     {verdict.get('error')}")
    print(f"    Server Latency:      {verdict.get('latency_ms')} ms")

    if verdict.get("verdict_text") not in ["REJECTED_PROMPT_MISMATCH", "REJECTED_PROMPT_ALIGNMENT_FAILED"]:
        p_vis = verdict["p_visual"]
        p_desync = verdict["p_desync"]
        expected_z = 5.3688 * p_vis + 4.9111 * p_desync - 6.0411
        expected_p = float(1.0 / (1.0 + np.exp(-expected_z)))
        actual_p = verdict["deepfake_probability"]
        formula_diff = abs(actual_p - expected_p)
        print(f"    Formula Check:       sigma(5.3688*{p_vis:.4f} + 4.9111*{p_desync:.4f} - 6.0411) = {expected_p:.4f} (diff: {formula_diff:.6f})")
        assert formula_diff < 1e-3, f"Meta-learner formula evaluation mismatch: diff={formula_diff}"

    return verdict


def main():
    device = get_device()
    dev_str = getattr(device, "type", str(device)).upper()
    print("=" * 75)
    print(f" SECUREKYC AI - E2E INFERENCE & PROMPT VERIFICATION ({dev_str}) ")
    print("=" * 75)

    load_model_checkpoints()
    client = TestClient(app)

    # -------------------------------------------------------------------------
    # Test 1: Genuine VidTIMIT Clip (fadg0/sa2: "Don't ask me to carry an oily rag like that.")
    # -------------------------------------------------------------------------
    real_sa2_dir = RAW_DATA_DIR / "original_sequences" / "fadg0" / "video" / "sa2"
    real_sa2_wav = RAW_DATA_DIR / "original_sequences" / "fadg0" / "audio" / "sa2.wav"
    assert real_sa2_dir.exists() and real_sa2_wav.exists()

    real_sa2_files = sorted(
        [p for p in real_sa2_dir.glob("*") if p.is_file() and not p.name.startswith(".")],
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    real_sa2_frames = [cv2.imread(str(f)) for f in real_sa2_files]
    real_sa2_frames = [f for f in real_sa2_frames if f is not None]

    prompt_sa2 = "Don't ask me to carry an oily rag like that."
    verdict_real_sa2 = run_e2e_session(
        client=client,
        clip_name="Genuine VidTIMIT Speaker: fadg0/sa2",
        frames=real_sa2_frames,
        audio_path=real_sa2_wav,
        prompt=prompt_sa2,
        expected_label="GENUINE (REAL)",
        fps=25.0,
    )
    assert verdict_real_sa2["is_deepfake"] is False
    assert verdict_real_sa2["verdict_text"] == "AUTHENTIC_USER_CONFIRMED"
    assert verdict_real_sa2["prompt_verified"] is True

    # -------------------------------------------------------------------------
    # Test 2: DeepFakeTIMIT Fake Clip with Bilabial Desync (lower_quality/fadg0/sa2-video-fram1.avi)
    # -------------------------------------------------------------------------
    fake_sa2_avi = RAW_DATA_DIR / "manipulated_sequences" / "DeepFakeTIMIT" / "lower_quality" / "fadg0" / "sa2-video-fram1.avi"
    fake_sa2_wav = RAW_DATA_DIR / "manipulated_sequences" / "DeepFakeTIMIT" / "lower_quality" / "fadg0" / "sa2.wav"
    assert fake_sa2_avi.exists() and fake_sa2_wav.exists()

    cap = cv2.VideoCapture(str(fake_sa2_avi))
    fake_sa2_frames = []
    while cap.isOpened():
        ret, f = cap.read()
        if not ret or f is None:
            break
        fake_sa2_frames.append(f)
    cap.release()

    verdict_fake_sa2 = run_e2e_session(
        client=client,
        clip_name="DeepFakeTIMIT Fake: fadg0/sa2-video-fram1 (lower_quality)",
        frames=fake_sa2_frames,
        audio_path=fake_sa2_wav,
        prompt=prompt_sa2,
        expected_label="DEEPFAKE (SYNTHETIC)",
        fps=25.0,
    )
    assert verdict_fake_sa2["is_deepfake"] is True
    assert verdict_fake_sa2["verdict_text"] == "DEEPFAKE_VISUAL_DETECTED"
    assert verdict_fake_sa2["deepfake_probability"] >= 0.50

    # -------------------------------------------------------------------------
    # Test 3: Security Check 1 - Prompt Mismatch Injection
    # -------------------------------------------------------------------------
    verdict_mismatch = run_e2e_session(
        client=client,
        clip_name="Security Test 1: Prompt Mismatch Injection",
        frames=real_sa2_frames[:30],
        audio_path=real_sa2_wav,
        prompt=prompt_sa2,
        submitted_prompt="Different Unrelated Challenge Phrase",
        expected_label="REJECTED (PROMPT MISMATCH)",
        fps=25.0,
    )
    assert verdict_mismatch["is_deepfake"] is None
    assert verdict_mismatch["deepfake_probability"] is None
    assert verdict_mismatch["p_visual"] is None
    assert verdict_mismatch["p_desync"] is None
    assert verdict_mismatch["verdict_text"] == "REJECTED_PROMPT_MISMATCH"
    assert verdict_mismatch["prompt_verified"] is False

    # -------------------------------------------------------------------------
    # Test 4: Security Check 2 - Acoustic Alignment Failure (Audio Doesn't Match Challenge)
    # -------------------------------------------------------------------------
    # Issue a challenge prompt ('Bumble Bee Password') but submit unrelated audio (sa2.wav: "Don't ask me...")
    verdict_align_fail = run_e2e_session(
        client=client,
        clip_name="Security Test 2: Audio/Challenge Phonetic Mismatch",
        frames=real_sa2_frames[:30],
        audio_path=real_sa2_wav,
        prompt="Bumble Bee Password",
        expected_label="REJECTED (ALIGNMENT FAILED)",
        fps=25.0,
    )
    assert verdict_align_fail["is_deepfake"] is None
    assert verdict_align_fail["deepfake_probability"] is None
    assert verdict_align_fail["p_visual"] is None
    assert verdict_align_fail["p_desync"] is None
    assert verdict_align_fail["verdict_text"] == "REJECTED_PROMPT_ALIGNMENT_FAILED"
    assert verdict_align_fail["prompt_verified"] is False
    assert "acoustic_cer" in verdict_align_fail
    assert verdict_align_fail["acoustic_cer"] > 0.60

    def _fmt_prob(val):
        return f"{val:.4f}" if val is not None else "null"

    # Summary Comparison
    print(f"\n{'='*95}")
    print(f"{'END-TO-END INFERENCE & SECURITY VERIFICATION SUMMARY':^95}")
    print(f"{'='*95}")
    print(f"{'Test Description':<40} | {'Expected':<12} | {'Verdict Text':<30} | {'P(Fake)':<8}")
    print("-" * 95)
    print(f"{'fadg0/sa2 (VidTIMIT Genuine)':<40} | {'GENUINE':<12} | {verdict_real_sa2['verdict_text']:<30} | {_fmt_prob(verdict_real_sa2['deepfake_probability']):<8}")
    print(f"{'fadg0/sa2 (DF-TIMIT Fake)':<40} | {'DEEPFAKE':<12} | {verdict_fake_sa2['verdict_text']:<30} | {_fmt_prob(verdict_fake_sa2['deepfake_probability']):<8}")
    print(f"{'Prompt Mismatch Injection':<40} | {'REJECT':<12} | {verdict_mismatch['verdict_text']:<30} | {_fmt_prob(verdict_mismatch['deepfake_probability']):<8}")
    print(f"{'Audio/Challenge Mismatch':<40} | {'REJECT':<12} | {verdict_align_fail['verdict_text']:<30} | {_fmt_prob(verdict_align_fail['deepfake_probability']):<8}")
    print("=" * 95)
    print(" [+] ALL 4 INFERENCE & PROMPT SECURITY CHECKS PASSED EMPIRICALLY!\n")


if __name__ == "__main__":
    main()
