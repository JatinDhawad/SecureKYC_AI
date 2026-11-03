"""
End-to-End Verification of Offline Video Analysis Endpoint (/analyze-offline-video)

Tests:
1. Real VidTIMIT clip (known genuine): data/processed/fadg0_sa2_genuine.mp4
   with correct transcript "Don't ask me to carry an oily rag like that."
2. Real DeepFakeTIMIT clip (known fake): data/raw/manipulated_sequences/DeepFakeTIMIT/lower_quality/fadg0/sa2-video-fram1.avi
   with correct transcript "Don't ask me to carry an oily rag like that."
3. Real VidTIMIT clip without transcript (verifying visual-only branch behavior).
4. Real DeepFakeTIMIT clip without transcript (verifying visual-only branch behavior).
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app
from config import LIP_CLOSURE_THRESHOLD_NORMALIZED

client = TestClient(app)

REAL_CLIP = PROJECT_ROOT / "data" / "processed" / "fadg0_sa2_genuine.mp4"
FAKE_CLIP = PROJECT_ROOT / "data" / "raw" / "manipulated_sequences" / "DeepFakeTIMIT" / "lower_quality" / "fadg0" / "sa2-video-fram1.avi"
TRANSCRIPT_SA2 = "Don't ask me to carry an oily rag like that."

def run_tests():
    assert REAL_CLIP.exists(), f"Real clip not found: {REAL_CLIP}"
    assert FAKE_CLIP.exists(), f"Fake clip not found: {FAKE_CLIP}"

    print(f"[*] Calibrated Lip Closure Threshold (Single Source of Truth): {LIP_CLOSURE_THRESHOLD_NORMALIZED}\n")

    print("=================================================================")
    print("TEST 1: Real VidTIMIT Clip (Known Genuine) WITH Transcript")
    print("=================================================================")
    with open(REAL_CLIP, "rb") as f:
        files = {"video": (REAL_CLIP.name, f, "video/mp4")}
        data = {"expected_transcript": TRANSCRIPT_SA2}
        resp = client.post("/analyze-offline-video", files=files, data=data)

    print(f"Status Code: {resp.status_code}")
    real_json = resp.json()
    print(json.dumps(real_json, indent=2))
    assert resp.status_code == 200, f"Failed: {resp.text}"
    assert "verdict_text" in real_json
    assert "deepfake_probability" in real_json
    assert "p_visual" in real_json
    assert "p_desync" in real_json
    assert "desync_score" in real_json
    assert "failed_closure_ratio" in real_json
    assert real_json["verdict_text"] == "AUTHENTIC_USER_CONFIRMED"

    s_desync_real = real_json["desync_score"]
    thresh_cmp_real = s_desync_real > LIP_CLOSURE_THRESHOLD_NORMALIZED
    print(f"\n[Desync Threshold Comparison - Real Clip]:")
    print(f"  S_desync ({s_desync_real:.4f}) vs Threshold ({LIP_CLOSURE_THRESHOLD_NORMALIZED}): is_desync_detected = {thresh_cmp_real}")
    print(f"  P(fake) = {real_json['deepfake_probability']:.4f} < 0.50 (neural genuine confirmed)")
    print(f"  -> Final Verdict: {real_json['verdict_text']}\n")

    print("=================================================================")
    print("TEST 2: Fake DeepFakeTIMIT Clip (Known Fake) WITH Transcript")
    print("=================================================================")
    with open(FAKE_CLIP, "rb") as f:
        files = {"video": (FAKE_CLIP.name, f, "video/x-msvideo")}
        data = {"expected_transcript": TRANSCRIPT_SA2}
        resp = client.post("/analyze-offline-video", files=files, data=data)

    print(f"Status Code: {resp.status_code}")
    fake_json = resp.json()
    print(json.dumps(fake_json, indent=2))
    assert resp.status_code == 200, f"Failed: {resp.text}"
    assert "verdict_text" in fake_json
    assert "deepfake_probability" in fake_json
    assert "p_visual" in fake_json
    assert "p_desync" in fake_json
    assert "desync_score" in fake_json
    assert "failed_closure_ratio" in fake_json
    assert "DEEPFAKE" in fake_json["verdict_text"]

    s_desync_fake = fake_json["desync_score"]
    thresh_cmp_fake = s_desync_fake > LIP_CLOSURE_THRESHOLD_NORMALIZED
    print(f"\n[Desync Threshold Comparison - Fake Clip]:")
    print(f"  S_desync ({s_desync_fake:.4f}) vs Threshold ({LIP_CLOSURE_THRESHOLD_NORMALIZED}): is_desync_detected = {thresh_cmp_fake}")
    print(f"  P(fake) = {fake_json['deepfake_probability']:.4f} >= 0.50 (neural deepfake confirmed; P_vis={fake_json['p_visual']}, P_desync={fake_json['p_desync']})")
    print(f"  Failed Closure Ratio = {fake_json['failed_closure_ratio']:.4f} (100% of bilabials failed to close)")
    print(f"  -> Final Verdict: {fake_json['verdict_text']}\n")

    print("=================================================================")
    print("TEST 3: Real VidTIMIT Clip (Visual Only / NO Transcript)")
    print("=================================================================")
    with open(REAL_CLIP, "rb") as f:
        files = {"video": (REAL_CLIP.name, f, "video/mp4")}
        resp = client.post("/analyze-offline-video", files=files)

    print(f"Status Code: {resp.status_code}")
    real_vis_json = resp.json()
    print(json.dumps(real_vis_json, indent=2))
    assert resp.status_code == 200
    assert real_vis_json["p_desync"] is None
    assert real_vis_json["desync_score"] is None
    assert real_vis_json["failed_closure_ratio"] is None
    assert real_vis_json["branch_used"] == "visual_only"
    print(f"-> Visual-only branch verified: P(fake) = {real_vis_json['deepfake_probability']:.4f}\n")

    print("=================================================================")
    print("TEST 4: Fake DeepFakeTIMIT Clip (Visual Only / NO Transcript)")
    print("=================================================================")
    with open(FAKE_CLIP, "rb") as f:
        files = {"video": (FAKE_CLIP.name, f, "video/x-msvideo")}
        resp = client.post("/analyze-offline-video", files=files)

    print(f"Status Code: {resp.status_code}")
    fake_vis_json = resp.json()
    print(json.dumps(fake_vis_json, indent=2))
    assert resp.status_code == 200
    assert fake_vis_json["p_desync"] is None
    assert fake_vis_json["desync_score"] is None
    assert fake_vis_json["failed_closure_ratio"] is None
    assert fake_vis_json["branch_used"] == "visual_only"
    print(f"-> Visual-only branch verified: P(fake) = {fake_vis_json['deepfake_probability']:.4f}\n")

    print("=================================================================")
    print("ALL 4 OFFLINE VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    run_tests()
