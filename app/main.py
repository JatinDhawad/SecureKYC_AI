"""
SecureKYC AI - Real-Time FastAPI & WebSocket Challenge-Response Backend
Architecture: Dynamic Bilabial Challenge-Response Protocol
  1. Server picks challenge prompt phrase and sends it over WebSocket.
  2. Client streams JPEG frames for ~3-5s response window; server tracks 3D lip aperture D_lip(t) in near-real-time.
  3. Client sends completed utterance audio clip once window ends; server executes forced alignment & desync metric ONCE.
  4. Server passes 131-dim fused features into MultimodalKYCFusionNet on Apple Silicon (mps) and returns verdict over WebSocket.
"""

import os
import re
import sys
import io
import wave
import base64
import time
import random
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import uvicorn
import av
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

logger = logging.getLogger(__name__)

from config import (
    BASE_DIR,
    RAW_DATA_DIR,
    ACTIVE_PROMPT_POOL,
    MODEL_CHECKPOINT_DIR,
    LIP_CLOSURE_THRESHOLD_NORMALIZED,
    FIXED_SEQ_LEN,
    get_device,
)
from src.vision.lip_tracker import LipTracker, LipFrameResult
from src.audio.phoneme_extractor import PhonemeExtractor, AudioPhonemeResult
from src.metrics.desync_metric import DesynchronizationMetric, DesyncMetricResult
from src.models.fusion_net import MultimodalKYCFusionNet, VisionOnlyNet, DesyncOnlyNet, ConstrainedLateFusion
from app.prompt_generator import prompt_generator

app = FastAPI(
    title="SecureKYC AI - Enterprise Challenge-Response Liveness Backend",
    description="Dynamic Bilabial-Targeted Active Prompting & Real-Time Deepfake Injection Defense",
    version="2.0.0",
)

STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global Inference Resources
device = get_device()
lip_tracker: Optional[LipTracker] = None
phoneme_extractor: Optional[PhonemeExtractor] = None
vision_net: Optional[VisionOnlyNet] = None
desync_net: Optional[DesyncOnlyNet] = None
meta_weights: Dict[str, float] = {
    "w_vis": 5.3688,
    "w_desync": 4.9111,
    "intercept": -6.0411,
}
fusion_net: Optional[Any] = None
desync_metric = DesynchronizationMetric()


def load_model_checkpoints():
    """Initializes LipTracker, PhonemeExtractor, and Constrained Late-Fusion production models."""
    global lip_tracker, phoneme_extractor, vision_net, desync_net, meta_weights, fusion_net
    if lip_tracker is None:
        lip_tracker = LipTracker(static_image_mode=False)

    if phoneme_extractor is None:
        phoneme_extractor = PhonemeExtractor(lazy_load=False)
    if phoneme_extractor._secondary_model is None:
        phoneme_extractor.load_asr_model()

    ckpt_path = MODEL_CHECKPOINT_DIR / "best_fusion_model.pt"
    if not ckpt_path.exists():
        err_msg = f"Fatal startup failure: Model checkpoint not found at {ckpt_path}."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    if vision_net is None or desync_net is None or fusion_net is None:
        v_model = VisionOnlyNet(seq_len=100)
        d_model = DesyncOnlyNet(desync_feature_dim=3)
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if "vision_state_dict" in ckpt and "desync_state_dict" in ckpt:
                v_model.load_state_dict(ckpt["vision_state_dict"])
                d_model.load_state_dict(ckpt["desync_state_dict"])
                if "meta_weights" in ckpt:
                    meta_weights = ckpt["meta_weights"]
            elif (MODEL_CHECKPOINT_DIR / "best_vision_model.pt").exists() and (MODEL_CHECKPOINT_DIR / "best_desync_model.pt").exists():
                v_ckpt = torch.load(MODEL_CHECKPOINT_DIR / "best_vision_model.pt", map_location=device)
                d_ckpt = torch.load(MODEL_CHECKPOINT_DIR / "best_desync_model.pt", map_location=device)
                v_model.load_state_dict(v_ckpt["model_state_dict"])
                d_model.load_state_dict(d_ckpt["model_state_dict"])
            else:
                raise ValueError("Checkpoint does not contain vision_state_dict and desync_state_dict")

            v_model.to(device)
            v_model.eval()
            d_model.to(device)
            d_model.eval()

            vision_net = v_model
            desync_net = d_model
            fusion_net = ConstrainedLateFusion(
                vision_net=v_model,
                desync_net=d_model,
                w_vis=meta_weights["w_vis"],
                w_desync=meta_weights["w_desync"],
                intercept=meta_weights["intercept"],
            ).to(device)
            fusion_net.eval()

            logger.info("Loaded Constrained Late-Fusion models (W_vis=%.4f, W_desync=%.4f, intercept=%.4f) from %s",
                        meta_weights["w_vis"], meta_weights["w_desync"], meta_weights["intercept"], ckpt_path)
        except Exception as e:
            logger.exception("Fatal startup failure: Failed to load model checkpoint from %s: %s", ckpt_path, e)
            raise RuntimeError(f"Fatal startup failure: Could not load trained weights from {ckpt_path}: {e}") from e



@app.on_event("startup")
async def startup_event():
    logger.info("Starting SecureKYC AI backend and loading model checkpoints...")
    load_model_checkpoints()


@app.get("/", response_class=FileResponse)
async def get_index():
    """Serves the interactive web verification dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>SecureKYC AI Challenge-Response Backend Active</h1>")


@app.get("/offline", response_class=FileResponse)
async def get_offline_page():
    """Serves the offline video analysis demo page."""
    offline_file = STATIC_DIR / "offline.html"
    if offline_file.exists():
        return FileResponse(str(offline_file))
    return HTMLResponse("<h1>Offline Video Analysis Mode</h1>")


@app.post("/analyze-offline-video")
async def analyze_offline_video(
    video: UploadFile = File(...),
    expected_transcript: Optional[str] = Form(None),
):
    """
    Analyzes an uploaded video clip in offline forensic mode.

    Workflow:
      1. Tracks 3D lip aperture D_lip(t) via LipTracker across all video frames and interpolates missing frames.
      2. Runs VisionOnlyNet on the normalized 100-point aperture trajectory to compute p_visual.
      3. If expected_transcript is provided:
         - Extracts audio from video via PhonemeExtractor.
         - Runs forced alignment with Wav2Vec2 against expected_transcript to identify bilabial boundaries.
         - Computes cross-modal acoustic-viseme desynchronization score S_desync and failed closure ratio.
         - Runs DesyncOnlyNet to compute p_desync.
         - Computes fused deepfake probability via constrained late-fusion meta-learner:
           logit = w_vis * p_vis + w_desync * p_desync + intercept
           deepfake_probability = sigmoid(logit)
      4. If no expected_transcript is provided:
         - Skips desync analysis completely (no guessing or defaulting transcripts).
         - Uses visual-only probability (deepfake_probability = p_visual, p_desync = None, desync_score = None, failed_closure_ratio = None).
         - Explicitly notes that only the visual branch was evaluated.

    Returns JSON:
      {
        "verdict_text": str,
        "deepfake_probability": float,
        "p_visual": float,
        "p_desync": Optional[float],
        "desync_score": Optional[float],
        "failed_closure_ratio": Optional[float],
        ...
      }
    """
    load_model_checkpoints()
    assert lip_tracker is not None, "LipTracker not initialized"
    assert vision_net is not None, "VisionOnlyNet not initialized"
    assert desync_net is not None, "DesyncOnlyNet not initialized"
    assert phoneme_extractor is not None, "PhonemeExtractor not initialized"
    assert desync_metric is not None, "DesyncMetric not initialized"

    start_infer_t = time.perf_counter()
    suffix = Path(video.filename or "uploaded_video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)

    try:
        # Write uploaded file to disk
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # 1. Process visual frames
        cap = cv2.VideoCapture(str(temp_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 0:
            fps = 25.0
        frame_ms = 1000.0 / fps

        lip_frame_results: List[LipFrameResult] = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            res = lip_tracker.process_frame(frame, timestamp_ms=frame_idx * frame_ms)
            lip_frame_results.append(res)
            frame_idx += 1
        cap.release()

        if not lip_frame_results:
            raise HTTPException(status_code=400, detail="Unable to extract video frames or file is empty.")

        # Interpolate missing apertures and resample to FIXED_SEQ_LEN
        processed_frames = lip_tracker.interpolate_missing_apertures(lip_frame_results)
        apertures = np.array([r.lip_aperture for r in processed_frames], dtype=np.float32)

        if len(apertures) > 0:
            x_orig = np.linspace(0, 1, len(apertures))
            x_target = np.linspace(0, 1, FIXED_SEQ_LEN)
            norm_seq = np.interp(x_target, x_orig, apertures).astype(np.float32)
        else:
            norm_seq = np.zeros(FIXED_SEQ_LEN, dtype=np.float32)

        lip_tensor = torch.from_numpy(norm_seq).unsqueeze(0).to(device)
        with torch.no_grad():
            p_vis = float(vision_net(lip_tensor).cpu().numpy()[0, 0])

        transcript_clean = expected_transcript.strip() if expected_transcript else ""

        # Branch 1: Transcript provided -> Multimodal Late-Fusion
        if transcript_clean:
            waveform, sr = phoneme_extractor.extract_audio_from_video(temp_path)
            if waveform is None or len(waveform) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No audio stream found in uploaded video file. Audio is required to align against the provided transcript.",
                )

            # Security Check 1: Free-form acoustic speech decoding & prompt verification (Wav2Vec2ForCTC)
            # Reusing the calibrated threshold max_cer <= 0.60 from /ws/verify
            is_matched, cer, wer, decoded_hyp = phoneme_extractor.verify_prompt_acoustic_match(
                waveform, prompt_text=transcript_clean, sample_rate=sr, max_cer=0.60
            )

            total_latency_ms = round((time.perf_counter() - start_infer_t) * 1000.0, 2)
            per_frame_latency_ms = round(total_latency_ms / max(len(processed_frames), 1), 2)

            if not is_matched:
                logger.warning(
                    "Offline acoustic verification failed: audio does not match transcript '%s' (CER: %.4f, WER: %.4f, decoded: '%s')",
                    transcript_clean, cer, wer, decoded_hyp,
                )
                return {
                    "verdict_text": "REJECTED_PROMPT_ALIGNMENT_FAILED",
                    "transcript_verified": False,
                    "prompt_verified": False,
                    "decoded_transcript": decoded_hyp,
                    "acoustic_cer": round(cer, 4),
                    "acoustic_wer": round(wer, 4),
                    "deepfake_probability": None,
                    "is_deepfake": None,
                    "p_visual": round(p_vis, 4),
                    "p_desync": None,
                    "desync_score": None,
                    "failed_closure_ratio": None,
                    "branch_used": "transcript_rejected",
                    "note": f"Acoustic verification failed: audio does not match the submitted transcript (CER: {cer:.4f} > 0.60; decoded: '{decoded_hyp}'). Desync analysis was aborted to prevent bogus alignment.",
                    "error": f"Transcript acoustic verification failed: audio does not correspond to submitted transcript (CER: {cer:.4f} > 0.60; decoded: '{decoded_hyp}')",
                    "frames_evaluated": len(processed_frames),
                    "bilabial_events_found": 0,
                    "latency_ms": total_latency_ms,
                    "per_frame_latency_ms": per_frame_latency_ms,
                }

            # Security Check 2: Forced Alignment to identify bilabial boundaries
            audio_res = phoneme_extractor.align_forced_transcript(waveform, prompt_text=transcript_clean, sample_rate=sr)

            if not audio_res.bilabial_intervals:
                logger.warning(
                    "Offline verification failed: zero bilabials found for transcript '%s'",
                    transcript_clean,
                )
                return {
                    "verdict_text": "REJECTED_PROMPT_ALIGNMENT_FAILED",
                    "transcript_verified": True,
                    "prompt_verified": True,
                    "decoded_transcript": decoded_hyp,
                    "acoustic_cer": round(cer, 4),
                    "acoustic_wer": round(wer, 4),
                    "deepfake_probability": None,
                    "is_deepfake": None,
                    "p_visual": round(p_vis, 4),
                    "p_desync": None,
                    "desync_score": None,
                    "failed_closure_ratio": None,
                    "branch_used": "zero_bilabials_rejected",
                    "note": f"Transcript contains zero bilabial consonants (/b/, /m/, /p/). Desync metric cannot evaluate lip closure without bilabials.",
                    "error": f"Transcript verification failed: zero bilabial consonant intervals detected for submitted transcript '{transcript_clean}'",
                    "frames_evaluated": len(processed_frames),
                    "bilabial_events_found": 0,
                    "latency_ms": total_latency_ms,
                    "per_frame_latency_ms": per_frame_latency_ms,
                }

            desync_res = desync_metric.compute_desync_score(processed_frames, audio_res)

            s_desync = float(desync_res.overall_desync_score)
            max_penalty = float(desync_res.max_penalty_during_bilabials)
            num_b = desync_res.num_bilabial_events
            failed_ratio = float(desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0

            desync_tensor = torch.tensor([[s_desync, max_penalty, failed_ratio]], dtype=torch.float32).to(device)
            with torch.no_grad():
                p_desync = float(desync_net(desync_features=desync_tensor).cpu().numpy()[0, 0])

            w_vis = float(meta_weights.get("w_vis", 5.3688))
            w_desync = float(meta_weights.get("w_desync", 4.9111))
            intercept = float(meta_weights.get("intercept", -6.0411))
            logit_z = w_vis * p_vis + w_desync * p_desync + intercept
            deepfake_prob = float(1.0 / (1.0 + np.exp(-logit_z)))

            # Option 1 Explicit Rule Composition:
            # Physical closure failure threshold calibrated from 95th percentile of authentic VidTIMIT speakers (0.0242)
            # S_desync > LIP_CLOSURE_THRESHOLD_NORMALIZED represents physical failure of acoustic-viseme closure.
            DESYNC_ALERT_THRESHOLD = LIP_CLOSURE_THRESHOLD_NORMALIZED
            is_desync_detected = s_desync > DESYNC_ALERT_THRESHOLD
            is_neural_detected = deepfake_prob >= 0.5

            if is_desync_detected:
                verdict_text = "DEEPFAKE_DESYNC_DETECTED"
            elif is_neural_detected:
                verdict_text = "DEEPFAKE_VISUAL_DETECTED"
            else:
                verdict_text = "AUTHENTIC_USER_CONFIRMED"

            return {
                "verdict_text": verdict_text,
                "transcript_verified": True,
                "prompt_verified": True,
                "decoded_transcript": decoded_hyp,
                "acoustic_cer": round(cer, 4),
                "acoustic_wer": round(wer, 4),
                "deepfake_probability": round(deepfake_prob, 4),
                "p_visual": round(p_vis, 4),
                "p_desync": round(p_desync, 4),
                "desync_score": round(s_desync, 4),
                "failed_closure_ratio": round(failed_ratio, 4),
                "branch_used": "multimodal_late_fusion",
                "note": "Multimodal evaluation (visual aperture trajectory + forced alignment acoustic-viseme desynchronization)",
                "frames_evaluated": len(processed_frames),
                "bilabial_events_found": num_b,
                "latency_ms": total_latency_ms,
                "per_frame_latency_ms": per_frame_latency_ms,
            }

        # Branch 2: No transcript provided -> Visual Only
        else:
            is_deepfake = p_vis >= 0.5
            verdict_text = "DEEPFAKE_VISUAL_DETECTED" if is_deepfake else "AUTHENTIC_VISUAL_CONFIRMED"

            total_latency_ms = round((time.perf_counter() - start_infer_t) * 1000.0, 2)
            per_frame_latency_ms = round(total_latency_ms / max(len(processed_frames), 1), 2)

            return {
                "verdict_text": verdict_text,
                "deepfake_probability": round(p_vis, 4),
                "p_visual": round(p_vis, 4),
                "p_desync": None,
                "desync_score": None,
                "failed_closure_ratio": None,
                "branch_used": "visual_only",
                "note": "Visual branch only. Cross-modal desync was skipped because no expected transcript was provided.",
                "frames_evaluated": len(processed_frames),
                "latency_ms": total_latency_ms,
                "per_frame_latency_ms": per_frame_latency_ms,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Offline video analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Offline video analysis failed: {e}") from e
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass



@app.get("/health")
async def health_check():
    """Health check endpoint returning system status, device target, and loaded models."""
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": fusion_net is not None,
        "lip_tracker_loaded": lip_tracker is not None,
        "phoneme_extractor_loaded": phoneme_extractor is not None,
    }


@app.get("/api/prompt/random")
async def get_random_prompt():
    """Generates a dynamic bilabial-targeted challenge prompt phrase with session-level tracking."""
    challenge = prompt_generator.generate_challenge(session_id="api_request")
    return challenge


def decode_b64_image(image_b64: str) -> np.ndarray:
    """
    Decodes Base64 image string into OpenCV BGR frame.
    Logs exception with stack trace and raises ValueError on malformed or undecodable image data.
    """
    try:
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("OpenCV cv2.imdecode returned None for base64 image data")
        return frame
    except Exception as e:
        logger.exception("Malformed frame decode failure: %s", e)
        raise ValueError(f"Malformed frame decode failure: {e}") from e


def decode_b64_audio(audio_b64: str) -> Tuple[np.ndarray, int]:
    """Decodes Base64 WAV/PCM audio string into float32 1D numpy array."""
    try:
        header, encoded = audio_b64.split(",", 1) if "," in audio_b64 else ("", audio_b64)
        raw_bytes = base64.b64decode(encoded)

        # Try parsing as WAV file
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)
                audio_int16 = np.frombuffer(frames, dtype=np.int16)
                waveform = audio_int16.astype(np.float32) / 32768.0
                return waveform, sr
        except Exception:
            # Fallback: raw int16 PCM at 16kHz
            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            if len(audio_int16) == 0:
                raise ValueError("Decoded audio buffer is empty")
            waveform = audio_int16.astype(np.float32) / 32768.0
            return waveform, 16000
    except Exception as e:
        logger.exception("Audio decoding error: %s", e)
        raise ValueError(f"Audio decoding error: {e}") from e


@app.websocket("/ws/verify")
async def challenge_response_websocket(websocket: WebSocket):
    """
    Challenge-Response WebSocket Endpoint.
    Workflow:
      1. Server sends challenge prompt: {"type": "challenge_init", "prompt": phrase, "window_sec": 4.0}
      2. Client streams JPEG frames; server runs 3D LipTracker in near-real-time.
      3. Client sends utterance completion payload: {"type": "utterance_complete", "audio_b64": wav_b64}
      4. Server runs forced alignment using exact issued prompt phrase + desync metric ONCE, returning verdict.
    """
    await websocket.accept()
    load_model_checkpoints()
    assert lip_tracker is not None, "LipTracker failed to initialize"
    assert phoneme_extractor is not None, "PhonemeExtractor failed to initialize"
    assert desync_metric is not None, "DesyncMetric failed to initialize"
    assert vision_net is not None, "VisionOnlyNet failed to initialize"
    assert desync_net is not None, "DesyncOnlyNet failed to initialize"

    session_id = f"ws_{id(websocket)}"
    query_prompt = websocket.query_params.get("prompt")
    if query_prompt:
        prompt_phrase = query_prompt
        prompt_generator.issued_sessions[session_id] = {
            "session_id": session_id,
            "prompt": prompt_phrase,
            "bilabial_target_count": prompt_generator.count_bilabials(prompt_phrase),
            "target_phonemes": ["/b/", "/m/", "/p/"],
            "timestamp_ms": time.time() * 1000.0,
        }
    else:
        challenge = prompt_generator.generate_challenge(session_id=session_id)
        prompt_phrase = challenge["prompt"]
    response_window_sec = 4.0

    # Step 1: Send Challenge Prompt
    await websocket.send_json({
        "type": "challenge_init",
        "session_id": session_id,
        "prompt": prompt_phrase,
        "window_sec": response_window_sec,
        "target_phonemes": ["/b/", "/m/", "/p/"],
        "timestamp_ms": time.time() * 1000.0,
    })

    lip_frame_results: List[LipFrameResult] = []
    start_timestamp_ms = time.time() * 1000.0

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            if msg_type == "video_frame":
                # Step 2: Near-Real-Time 3D Lip Tracker on incoming frame stream
                image_b64 = msg.get("image", "")
                frame_ts = msg.get("timestamp_ms", time.time() * 1000.0)

                try:
                    frame = decode_b64_image(image_b64)
                    res = lip_tracker.process_frame(frame, timestamp_ms=frame_ts)
                    lip_frame_results.append(res)
                    await websocket.send_json({
                        "type": "frame_ack",
                        "face_detected": res.face_detected,
                        "lip_aperture": round(res.lip_aperture, 4),
                        "frame_count": len(lip_frame_results),
                    })
                except Exception as err:
                    logger.exception("Error decoding or processing incoming video_frame: %s", err)
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Frame decoding/tracking error: {err}",
                        "face_detected": False,
                        "frame_count": len(lip_frame_results),
                    })

            elif msg_type == "utterance_complete":
                # Step 3 & 4: Post-Utterance Forced Alignment, Desync Computation, and Fused Inference
                start_infer_t = time.perf_counter()
                audio_b64 = msg.get("audio_b64", "")
                sample_rate = msg.get("sample_rate", 16000)

                # Check 1: Verify client-submitted prompt (if provided) matches the session's actual issued prompt
                client_prompt = msg.get("prompt")
                if client_prompt is not None:
                    def _norm(text: str) -> str:
                        return re.sub(r"[^\w\s]", "", text.lower()).strip()

                    if _norm(client_prompt) != _norm(prompt_phrase):
                        logger.warning(
                            "Security rejection in session %s: client submitted prompt '%s' does not match issued prompt '%s'",
                            session_id, client_prompt, prompt_phrase,
                        )
                        await websocket.send_json({
                            "type": "verdict",
                            "session_id": session_id,
                            "prompt": prompt_phrase,
                            "submitted_prompt": client_prompt,
                            "is_deepfake": None,
                            "prompt_verified": False,
                            "deepfake_probability": None,
                            "liveness_confidence_pct": None,
                            "p_visual": None,
                            "p_desync": None,
                            "desync_score": None,
                            "max_penalty": None,
                            "failed_closure_ratio": None,
                            "verdict_text": "REJECTED_PROMPT_MISMATCH",
                            "error": f"Security violation: Submitted prompt '{client_prompt}' does not match issued session prompt '{prompt_phrase}'",
                            "frames_evaluated": len(lip_frame_results),
                            "latency_ms": round((time.perf_counter() - start_infer_t) * 1000.0, 2),
                        })
                        break

                waveform, sr = decode_b64_audio(audio_b64) if audio_b64 else (np.zeros(0, dtype=np.float32), 16000)

                # Check 2: Free-form acoustic speech decoding & prompt verification (Wav2Vec2ForCTC)
                # Calibrated acceptance threshold: max_cer <= 0.60 cleanly separates matching speech (CER <= 0.50)
                # from mismatching challenge phrases (CER >= 0.72) with zero empirical overlap.
                is_matched = False
                cer = 1.0
                wer = 1.0
                decoded_hyp = ""
                if len(waveform) > 0 and phoneme_extractor is not None:
                    is_matched, cer, wer, decoded_hyp = phoneme_extractor.verify_prompt_acoustic_match(
                        waveform, prompt_text=prompt_phrase, sample_rate=sr, max_cer=0.60
                    )

                if not is_matched:
                    logger.warning(
                        "Acoustic verification failed in session %s: audio does not match prompt '%s' (CER: %.4f, WER: %.4f, decoded: '%s')",
                        session_id, prompt_phrase, cer, wer, decoded_hyp,
                    )
                    await websocket.send_json({
                        "type": "verdict",
                        "session_id": session_id,
                        "prompt": prompt_phrase,
                        "decoded_transcript": decoded_hyp,
                        "acoustic_cer": round(cer, 4),
                        "acoustic_wer": round(wer, 4),
                        "is_deepfake": None,
                        "prompt_verified": False,
                        "deepfake_probability": None,
                        "liveness_confidence_pct": None,
                        "p_visual": None,
                        "p_desync": None,
                        "desync_score": None,
                        "max_penalty": None,
                        "failed_closure_ratio": None,
                        "verdict_text": "REJECTED_PROMPT_ALIGNMENT_FAILED",
                        "error": f"Challenge verification failed: audio utterance does not correspond to issued challenge prompt '{prompt_phrase}' (CER: {cer:.4f} > 0.60; decoded: '{decoded_hyp}')",
                        "frames_evaluated": len(lip_frame_results),
                        "latency_ms": round((time.perf_counter() - start_infer_t) * 1000.0, 2),
                    })
                    break

                # 3. Forced Alignment ONCE on verified matching audio utterance strictly against issued session prompt
                if len(waveform) > 0 and phoneme_extractor is not None:
                    audio_res = phoneme_extractor.align_forced_transcript(waveform, prompt_text=prompt_phrase, sample_rate=sr)
                else:
                    audio_res = AudioPhonemeResult(audio_duration_ms=0.0)

                if not audio_res.bilabial_intervals:
                    logger.warning(
                        "Challenge verification failed in session %s: zero bilabial boundaries found for prompt '%s'",
                        session_id, prompt_phrase,
                    )
                    await websocket.send_json({
                        "type": "verdict",
                        "session_id": session_id,
                        "prompt": prompt_phrase,
                        "is_deepfake": None,
                        "prompt_verified": False,
                        "deepfake_probability": None,
                        "liveness_confidence_pct": None,
                        "p_visual": None,
                        "p_desync": None,
                        "desync_score": None,
                        "max_penalty": None,
                        "failed_closure_ratio": None,
                        "verdict_text": "REJECTED_PROMPT_ALIGNMENT_FAILED",
                        "error": f"Challenge verification failed: zero bilabial consonant intervals detected for challenge prompt '{prompt_phrase}'",
                        "frames_evaluated": len(lip_frame_results),
                        "latency_ms": round((time.perf_counter() - start_infer_t) * 1000.0, 2),
                    })
                    break

                # Interpolate missing face detection frames
                processed_lip_results: List[LipFrameResult] = []
                if lip_frame_results:
                    if lip_tracker is not None:
                        processed_lip_results = lip_tracker.interpolate_missing_apertures(lip_frame_results)
                    else:
                        processed_lip_results = lip_frame_results
                    aperture_seq = np.array([r.lip_aperture for r in processed_lip_results], dtype=np.float32)
                else:
                    processed_lip_results = []
                    aperture_seq = np.zeros(100, dtype=np.float32)

                # Interpolate sequence length to 100 points
                if len(aperture_seq) > 0:
                    x_orig = np.linspace(0, 1, len(aperture_seq))
                    x_target = np.linspace(0, 1, FIXED_SEQ_LEN)
                    norm_seq = np.interp(x_target, x_orig, aperture_seq).astype(np.float32)
                else:
                    norm_seq = np.zeros(FIXED_SEQ_LEN, dtype=np.float32)

                # 3b. Compute Desynchronization Metric ONCE on verified bilabial intervals
                if desync_metric is not None:
                    desync_res = desync_metric.compute_desync_score(processed_lip_results, audio_res)
                    s_desync = float(desync_res.overall_desync_score)
                    max_penalty = float(desync_res.max_penalty_during_bilabials)
                    num_b = desync_res.num_bilabial_events
                    failed_ratio = (desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0
                else:
                    s_desync, max_penalty, failed_ratio = 1.0, 1.0, 1.0


                # 4. Constrained Late-Fusion Inference: Run both base models & apply meta-learner formula
                lip_tensor = torch.from_numpy(norm_seq).unsqueeze(0).to(device)
                desync_tensor = torch.tensor([[s_desync, max_penalty, failed_ratio]], dtype=torch.float32).to(device)

                with torch.no_grad():
                    p_vis = float(vision_net(lip_tensor).cpu().numpy()[0, 0])
                    p_desync = float(desync_net(desync_features=desync_tensor).cpu().numpy()[0, 0])
                    w_vis = float(meta_weights.get("w_vis", 5.3688))
                    w_desync = float(meta_weights.get("w_desync", 4.9111))
                    intercept = float(meta_weights.get("intercept", -6.0411))
                    logit_z = w_vis * p_vis + w_desync * p_desync + intercept
                    deepfake_prob = float(1.0 / (1.0 + np.exp(-logit_z)))

                latency_ms = (time.perf_counter() - start_infer_t) * 1000.0


                # Option 1 Explicit Rule Composition:
                # S_desync > LIP_CLOSURE_THRESHOLD_NORMALIZED represents physical failure of acoustic-viseme closure.
                # If physical desync is detected, it overrides a false-negative neural output.
                DESYNC_ALERT_THRESHOLD = LIP_CLOSURE_THRESHOLD_NORMALIZED
                is_desync_detected = s_desync > DESYNC_ALERT_THRESHOLD
                is_neural_detected = deepfake_prob >= 0.5
                is_deepfake = is_neural_detected or is_desync_detected

                if is_desync_detected:
                    verdict_text = "DEEPFAKE_DESYNC_DETECTED"
                elif is_neural_detected:
                    verdict_text = "DEEPFAKE_VISUAL_DETECTED"
                else:
                    verdict_text = "AUTHENTIC_USER_CONFIRMED"

                await websocket.send_json({
                    "type": "verdict",
                    "prompt": prompt_phrase,
                    "prompt_verified": True,
                    "is_deepfake": is_deepfake,
                    "deepfake_probability": round(deepfake_prob, 4),
                    "liveness_confidence_pct": round((1.0 - deepfake_prob) * 100.0, 2),
                    "p_visual": round(p_vis, 4),
                    "p_desync": round(p_desync, 4),
                    "desync_score": round(s_desync, 4),
                    "max_penalty": round(max_penalty, 4),
                    "failed_closure_ratio": round(failed_ratio, 4),
                    "verdict_text": verdict_text,
                    "frames_evaluated": len(lip_frame_results),
                    "latency_ms": round(latency_ms, 2),
                })

                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket verification error: {e}")
    finally:
        prompt_generator.clear_session(session_id)


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """General streaming endpoint for webcam frames."""
    await websocket.accept()
    load_model_checkpoints()
    assert lip_tracker is not None, "LipTracker failed to initialize"

    try:
        while True:
            data = await websocket.receive_json()
            image_b64 = data.get("image", "")
            if not image_b64:
                continue

            try:
                start_frame_t = time.perf_counter()
                frame = decode_b64_image(image_b64)
                res = lip_tracker.process_frame(frame)
                frame_latency_ms = (time.perf_counter() - start_frame_t) * 1000.0
                await websocket.send_json({
                    "face_detected": res.face_detected,
                    "lip_aperture": round(res.lip_aperture, 4),
                    "is_closed": res.is_closed,
                    "latency_ms": round(frame_latency_ms, 1),
                })
            except Exception as err:
                logger.exception("Error decoding/processing frame in /ws/stream: %s", err)
                await websocket.send_json({
                    "error": f"Frame decoding failure: {err}",
                    "face_detected": False,
                })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
