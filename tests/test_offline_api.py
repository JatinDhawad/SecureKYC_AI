"""
Tests for Offline Video Analysis API (/analyze-offline-video & /offline)
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CLIP = PROJECT_ROOT / "data" / "processed" / "fadg0_sa2_genuine.mp4"
TRANSCRIPT_SA2 = "Don't ask me to carry an oily rag like that."


def test_offline_page_served():
    """Verify that GET /offline returns 200 and the offline demo page."""
    resp = client.get("/offline")
    assert resp.status_code == 200
    assert "Offline Video Deepfake Analysis" in resp.text
    assert "36.02%" in resp.text
    assert "0.6877" in resp.text


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="Test clip fadg0_sa2_genuine.mp4 not present")
def test_offline_video_multimodal_analysis():
    """Verify POST /analyze-offline-video with valid video and expected transcript."""
    with open(REAL_CLIP, "rb") as f:
        files = {"video": (REAL_CLIP.name, f, "video/mp4")}
        data = {"expected_transcript": TRANSCRIPT_SA2}
        resp = client.post("/analyze-offline-video", files=files, data=data)

    assert resp.status_code == 200
    json_data = resp.json()
    assert "verdict_text" in json_data
    assert "deepfake_probability" in json_data
    assert "p_visual" in json_data
    assert "p_desync" in json_data
    assert "desync_score" in json_data
    assert "failed_closure_ratio" in json_data
    assert json_data["p_desync"] is not None
    assert json_data["desync_score"] is not None
    assert json_data["branch_used"] == "multimodal_late_fusion"


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="Test clip fadg0_sa2_genuine.mp4 not present")
def test_offline_video_visual_only_analysis():
    """Verify POST /analyze-offline-video without transcript skips desync."""
    with open(REAL_CLIP, "rb") as f:
        files = {"video": (REAL_CLIP.name, f, "video/mp4")}
        resp = client.post("/analyze-offline-video", files=files)

    assert resp.status_code == 200
    json_data = resp.json()
    assert "verdict_text" in json_data
    assert "deepfake_probability" in json_data
    assert "p_visual" in json_data
    assert json_data["p_desync"] is None
    assert json_data["desync_score"] is None
    assert json_data["failed_closure_ratio"] is None
    assert json_data["branch_used"] == "visual_only"
    assert "Visual branch only" in json_data["note"]


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="Test clip fadg0_sa2_genuine.mp4 not present")
def test_offline_video_nonsense_transcript_rejected():
    """Verify POST /analyze-offline-video with nonsense transcript aborts desync and rejects alignment."""
    with open(REAL_CLIP, "rb") as f:
        files = {"video": (REAL_CLIP.name, f, "video/mp4")}
        data = {"expected_transcript": "zxhsdghsdfghsgzdfhs"}
        resp = client.post("/analyze-offline-video", files=files, data=data)

    assert resp.status_code == 200
    json_data = resp.json()
    assert json_data["verdict_text"] == "REJECTED_PROMPT_ALIGNMENT_FAILED"
    assert json_data["transcript_verified"] is False
    assert json_data["deepfake_probability"] is None
    assert json_data["p_desync"] is None
    assert json_data["desync_score"] is None
    assert json_data["failed_closure_ratio"] is None
    assert json_data["branch_used"] == "transcript_rejected"
    assert "acoustic_cer" in json_data
    assert json_data["acoustic_cer"] > 0.60
    assert "error" in json_data


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="Test clip fadg0_sa2_genuine.mp4 not present")
def test_offline_video_bilabial_nonsense_transcript_rejected_via_cer():
    """Verify POST /analyze-offline-video rejects nonsense text containing bilabials via CER (not zero-bilabial guard)."""
    with open(REAL_CLIP, "rb") as f:
        files = {"video": (REAL_CLIP.name, f, "video/mp4")}
        # Contains 'b', 'p', 'm' (bilabials present, so zero-bilabial guard will NOT trigger)
        data = {"expected_transcript": "banana pancake muffin"}
        resp = client.post("/analyze-offline-video", files=files, data=data)

    assert resp.status_code == 200
    json_data = resp.json()
    assert json_data["verdict_text"] == "REJECTED_PROMPT_ALIGNMENT_FAILED"
    assert json_data["transcript_verified"] is False
    assert json_data["branch_used"] == "transcript_rejected"  # Confirms CER check rejected it
    assert json_data["deepfake_probability"] is None
    assert json_data["p_desync"] is None
    assert json_data["desync_score"] is None
    assert json_data["failed_closure_ratio"] is None
    assert "acoustic_cer" in json_data
    assert json_data["acoustic_cer"] == 0.825  # High CER (0.825 > 0.60)
    assert json_data["acoustic_cer"] > 0.60
    assert "error" in json_data
    assert "decoded_transcript" in json_data


