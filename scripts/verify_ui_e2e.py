"""
End-to-End Browser UI Verification with Playwright
Tests the Offline Video Deepfake Analysis UI in a real headless browser.
"""

import sys
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_CLIP = PROJECT_ROOT / "data" / "demo_clips" / "real_01_fadg0_sa2.mp4"
FAKE_CLIP_VISUAL = PROJECT_ROOT / "data" / "demo_clips" / "fake_01_fadg0_sa2_fram1.mp4"
FAKE_CLIP_DESYNC = PROJECT_ROOT / "data" / "demo_clips" / "fake_02_faks0_sa2_fedw0.mp4"
TRANSCRIPT_SA2 = "Don't ask me to carry an oily rag like that."

ARTIFACT_DIR = Path("/Users/jatindhawad/.gemini/antigravity-ide/brain/6c1a6016-a5f2-4f4e-be9e-eb3a9d6718f0")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def run_browser_ui_tests():
    print("=" * 70)
    print("STARTING PLAYWRIGHT BROWSER UI VERIFICATION")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        # Step 1: Verify Root Page & Latency Pill
        print("\n[Step 1] Loading http://127.0.0.1:8000/ ...")
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")

        latency_text = page.locator("#latencyText").text_content() or ""
        print(f"  Header Latency Text on /: '{latency_text.strip()}'")
        assert "undefined" not in latency_text, f"Bug present: {latency_text}"

        sample_suite_count = page.locator(".sample-suite").count()
        celeb_real_count = page.locator("#btnSampleReal").count()
        celeb_fake_count = page.locator("#btnSampleFake").count()
        print(f"  .sample-suite count: {sample_suite_count}, #btnSampleReal: {celeb_real_count}, #btnSampleFake: {celeb_fake_count}")
        assert sample_suite_count == 0, "sample-suite still present on index.html"
        assert celeb_real_count == 0, "btnSampleReal still present"
        assert celeb_fake_count == 0, "btnSampleFake still present"

        # Step 2: Navigate to /offline
        print("\n[Step 2] Navigating to http://127.0.0.1:8000/offline ...")
        page.goto("http://127.0.0.1:8000/offline", wait_until="networkidle")
        page_title = page.title()
        print(f"  Page Title: '{page_title}'")
        assert "Offline Video Deepfake Analysis" in page_title

        # Verify no sample test buttons on /offline
        assert page.locator(".sample-suite").count() == 0
        assert page.locator("#btnSampleReal").count() == 0
        assert page.locator("#btnSampleFake").count() == 0

        # Step 3: Test Case 1 - Genuine Video Clip
        print("\n[Step 3] Running Test Case 1: Genuine Video Clip (real_01_fadg0_sa2.mp4)...")
        page.locator("#videoFileInput").set_input_files(str(REAL_CLIP))
        page.locator("#transcriptInput").fill(TRANSCRIPT_SA2)

        print("  Clicking 'Run Deepfake Analysis'...")
        page.locator("#btnRunAnalysis").click()

        # Wait for resultsContent to become visible
        page.locator("#resultsContent").wait_for(state="visible", timeout=30000)
        page.wait_for_timeout(500)

        real_verdict = page.locator("#verdictTextBadge").text_content() or ""
        real_prob = page.locator("#probPercentDisplay").text_content() or ""
        real_p_vis = page.locator("#metricPVis").text_content() or ""
        real_p_desync = page.locator("#metricPDesync").text_content() or ""
        real_s_desync = page.locator("#metricSDesync").text_content() or ""
        real_closure = page.locator("#metricClosureRatio").text_content() or ""
        real_latency_header = page.locator("#offlineLatencyText").text_content() or ""
        real_raw_json_str = page.locator("#rawJsonResponse").evaluate("el => el.textContent")
        real_raw_json = json.loads(real_raw_json_str)

        print("\n  [Real Clip UI Output]:")
        print(f"    Verdict Badge:         {real_verdict.strip()}")
        print(f"    Probability Display:   {real_prob.strip()}")
        print(f"    Visual Score (P_vis):  {real_p_vis.strip()}")
        print(f"    Desync Score (P_des):  {real_p_desync.strip()}")
        print(f"    Acoustic Desync (S):   {real_s_desync.strip()}")
        print(f"    Closure Failure Ratio: {real_closure.strip()}")
        print(f"    Header Latency Pill:   {real_latency_header.strip()}")
        print(f"    Server latency_ms:     {real_raw_json.get('latency_ms')} ms")

        screenshot_real = ARTIFACT_DIR / "ui_real_clip_result.png"
        page.screenshot(path=str(screenshot_real), full_page=True)
        print(f"  Saved screenshot to: {screenshot_real}")

        assert real_verdict.strip() == "AUTHENTIC_USER_CONFIRMED"
        assert "undefined" not in real_latency_header

        # Step 4: Test Case 2 - Deepfake Video Clip (Visual Swap)
        print("\n[Step 4] Running Test Case 2: Deepfake Clip (fake_01_fadg0_sa2_fram1.mp4)...")
        page.locator("#videoFileInput").set_input_files(str(FAKE_CLIP_VISUAL))
        page.locator("#transcriptInput").fill(TRANSCRIPT_SA2)

        print("  Clicking 'Run Deepfake Analysis'...")
        page.locator("#btnRunAnalysis").click()

        # Wait for loading to finish
        page.locator("#loadingState").wait_for(state="hidden", timeout=30000)
        page.locator("#resultsContent").wait_for(state="visible", timeout=10000)
        page.wait_for_timeout(500)

        fake_verdict = page.locator("#verdictTextBadge").text_content() or ""
        fake_prob = page.locator("#probPercentDisplay").text_content() or ""
        fake_p_vis = page.locator("#metricPVis").text_content() or ""
        fake_p_desync = page.locator("#metricPDesync").text_content() or ""
        fake_s_desync = page.locator("#metricSDesync").text_content() or ""
        fake_closure = page.locator("#metricClosureRatio").text_content() or ""
        fake_latency_header = page.locator("#offlineLatencyText").text_content() or ""
        fake_raw_json_str = page.locator("#rawJsonResponse").evaluate("el => el.textContent")
        fake_raw_json = json.loads(fake_raw_json_str)

        print("\n  [Fake Clip UI Output]:")
        print(f"    Verdict Badge:         {fake_verdict.strip()}")
        print(f"    Probability Display:   {fake_prob.strip()}")
        print(f"    Visual Score (P_vis):  {fake_p_vis.strip()}")
        print(f"    Desync Score (P_des):  {fake_p_desync.strip()}")
        print(f"    Acoustic Desync (S):   {fake_s_desync.strip()}")
        print(f"    Closure Failure Ratio: {fake_closure.strip()}")
        print(f"    Header Latency Pill:   {fake_latency_header.strip()}")
        print(f"    Server latency_ms:     {fake_raw_json.get('latency_ms')} ms")

        screenshot_fake = ARTIFACT_DIR / "ui_fake_clip_result.png"
        page.screenshot(path=str(screenshot_fake), full_page=True)
        print(f"  Saved screenshot to: {screenshot_fake}")

        assert "DEEPFAKE" in fake_verdict.strip()
        assert "undefined" not in fake_latency_header

        # Step 5: Test Case 3 - Deepfake Video Clip with Acoustic-Viseme Desync
        print("\n[Step 5] Running Test Case 3: Acoustic-Viseme Desync Clip (fake_02_faks0_sa2_fedw0.mp4)...")
        page.locator("#videoFileInput").set_input_files(str(FAKE_CLIP_DESYNC))
        page.locator("#transcriptInput").fill(TRANSCRIPT_SA2)

        page.locator("#btnRunAnalysis").click()
        page.locator("#loadingState").wait_for(state="hidden", timeout=30000)
        page.locator("#resultsContent").wait_for(state="visible", timeout=10000)
        page.wait_for_timeout(500)

        desync_verdict = page.locator("#verdictTextBadge").text_content() or ""
        desync_prob = page.locator("#probPercentDisplay").text_content() or ""
        desync_s = page.locator("#metricSDesync").text_content() or ""
        desync_closure = page.locator("#metricClosureRatio").text_content() or ""

        print("\n  [Desync Clip UI Output]:")
        print(f"    Verdict Badge:         {desync_verdict.strip()}")
        print(f"    Probability Display:   {desync_prob.strip()}")
        print(f"    Acoustic Desync (S):   {desync_s.strip()}")
        print(f"    Closure Failure Ratio: {desync_closure.strip()}")

        assert desync_verdict.strip() == "DEEPFAKE_DESYNC_DETECTED"

        # Step 6: Test Case 4 - Nonsense Transcript Rejection
        print("\n[Step 6] Running Test Case 4: Nonsense Transcript ('zxhsdghsdfghsgzdfhs')...")
        page.locator("#videoFileInput").set_input_files(str(REAL_CLIP))
        page.locator("#transcriptInput").fill("zxhsdghsdfghsgzdfhs")

        page.locator("#btnRunAnalysis").click()
        page.locator("#loadingState").wait_for(state="hidden", timeout=30000)
        page.locator("#resultsContent").wait_for(state="visible", timeout=10000)
        page.wait_for_timeout(500)

        rejected_verdict = page.locator("#verdictTextBadge").text_content() or ""
        rejected_prob = page.locator("#probPercentDisplay").text_content() or ""
        rejected_desync_note = page.locator("#metricDesyncNote").text_content() or ""
        rejected_raw_json_str = page.locator("#rawJsonResponse").evaluate("el => el.textContent")
        rejected_raw_json = json.loads(rejected_raw_json_str)

        print("\n  [Nonsense Transcript UI Output]:")
        print(f"    Verdict Badge:         {rejected_verdict.strip()}")
        print(f"    Probability Display:   {rejected_prob.strip()}")
        print(f"    Desync Note:           {rejected_desync_note.strip()}")
        print(f"    Decoded Transcript:    {rejected_raw_json.get('decoded_transcript')}")
        print(f"    Acoustic CER:          {rejected_raw_json.get('acoustic_cer')}")
        print(f"    P(fake):               {rejected_raw_json.get('deepfake_probability')}")
        print(f"    Desync Score:          {rejected_raw_json.get('desync_score')}")

        screenshot_rejected = ARTIFACT_DIR / "ui_nonsense_transcript_rejected.png"
        page.screenshot(path=str(screenshot_rejected), full_page=True)
        print(f"  Saved screenshot to: {screenshot_rejected}")

        assert rejected_verdict.strip() == "REJECTED_PROMPT_ALIGNMENT_FAILED"
        assert rejected_prob.strip() == "REJECTED"
        assert rejected_raw_json.get("deepfake_probability") is None
        assert rejected_raw_json.get("desync_score") is None
        assert rejected_raw_json.get("acoustic_cer") > 0.60

        browser.close()
        print("\n" + "=" * 70)
        print("ALL BROWSER UI PLAYWRIGHT E2E TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

if __name__ == "__main__":
    run_browser_ui_tests()
