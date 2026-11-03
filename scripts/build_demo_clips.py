"""
Build Curated Demo Clips for Live Offline Video Deepfake Analysis Demo

Curated Set:
1. Real Clip 1: real_01_fadg0_sa2.mp4 (Speaker: fadg0, Sentence: sa2)
2. Real Clip 2: real_02_fcmh0_sa2.mp4 (Speaker: fcmh0, Sentence: sa2)
3. Real Clip 3: real_03_fdac1_sa2.mp4 (Speaker: fdac1, Sentence: sa2)
4. Fake Clip 1: fake_01_fadg0_sa2_fram1.mp4 (Speaker: fadg0 swapped with fram1, Sentence: sa2)
5. Fake Clip 2: fake_02_faks0_sa2_fedw0.mp4 (Speaker: faks0 swapped with fedw0, Sentence: sa2)
6. Fake Clip 3: fake_03_fcft0_sa2_fdrd1.mp4 (Speaker: fcft0 swapped with fdrd1, Sentence: sa2)
"""

import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

DEMO_DIR = PROJECT_ROOT / "data" / "demo_clips"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "data" / "processed" / "timit_15spk_transcripts.json") as f:
    TRANSCRIPTS = json.load(f)

CLIPS_SPEC = [
    # --- Real VidTIMIT Clips ---
    {
        "filename": "real_01_fadg0_sa2.mp4",
        "label": "GENUINE (VidTIMIT fadg0)",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "type": "real",
        "speaker": "fadg0",
        "src_video_frames": PROJECT_ROOT / "data" / "raw" / "original_sequences" / "fadg0" / "video" / "sa2",
        "src_audio": PROJECT_ROOT / "data" / "raw" / "original_sequences" / "fadg0" / "audio" / "sa2.wav",
    },
    {
        "filename": "real_02_fcmh0_sa2.mp4",
        "label": "GENUINE (VidTIMIT fcmh0)",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "type": "real",
        "speaker": "fcmh0",
        "src_video_frames": PROJECT_ROOT / "data" / "raw" / "original_sequences" / "fcmh0" / "video" / "sa2",
        "src_audio": PROJECT_ROOT / "data" / "raw" / "original_sequences" / "fcmh0" / "audio" / "sa2.wav",
    },
    {
        "filename": "real_03_fdac1_sa2.mp4",
        "label": "GENUINE (VidTIMIT fdac1)",
        "ground_truth": "GENUINE",
        "sentence_id": "sa2",
        "type": "real",
        "speaker": "fdac1",
        "src_video_frames": PROJECT_ROOT / "data" / "raw" / "original_sequences" / "fdac1" / "video" / "sa2",
        "src_audio": PROJECT_ROOT / "data" / "raw" / "original_sequences" / "fdac1" / "audio" / "sa2.wav",
    },
    # --- Fake DeepFakeTIMIT Clips ---
    {
        "filename": "fake_01_fadg0_sa2_fram1.mp4",
        "label": "DEEPFAKE (DeepFakeTIMIT fadg0 swapped with fram1)",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "type": "fake",
        "speaker": "fadg0 (swap fram1)",
        "src_avi": PROJECT_ROOT / "data" / "raw" / "manipulated_sequences" / "DeepFakeTIMIT" / "lower_quality" / "fadg0" / "sa2-video-fram1.avi",
    },
    {
        "filename": "fake_02_faks0_sa2_fedw0.mp4",
        "label": "DEEPFAKE (DeepFakeTIMIT faks0 swapped with fedw0)",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "type": "fake",
        "speaker": "faks0 (swap fedw0)",
        "src_avi": PROJECT_ROOT / "data" / "raw" / "manipulated_sequences" / "DeepFakeTIMIT" / "lower_quality" / "faks0" / "sa2-video-fedw0.avi",
    },
    {
        "filename": "fake_03_fcft0_sa2_fdrd1.mp4",
        "label": "DEEPFAKE (DeepFakeTIMIT fcft0 swapped with fdrd1)",
        "ground_truth": "DEEPFAKE",
        "sentence_id": "sa2",
        "type": "fake",
        "speaker": "fcft0 (swap fdrd1)",
        "src_avi": PROJECT_ROOT / "data" / "raw" / "manipulated_sequences" / "DeepFakeTIMIT" / "lower_quality" / "fcft0" / "sa2-video-fdrd1.avi",
    },
]

def build_clips():
    print("=================================================================")
    print("STEP 1: Muxing 6 Curated Clips into data/demo_clips/ as .mp4")
    print("=================================================================")

    for spec in CLIPS_SPEC:
        out_path = DEMO_DIR / spec["filename"]
        if spec["type"] == "real":
            # Mux frame sequence + WAV audio
            cmd = [
                "ffmpeg", "-y", "-framerate", "25",
                "-f", "image2", "-start_number", "1",
                "-i", str(spec["src_video_frames"] / "%03d"),
                "-i", str(spec["src_audio"]),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest",
                str(out_path)
            ]
        else:
            # Mux AVI into web-standard MP4 with H.264 + AAC
            cmd = [
                "ffmpeg", "-y",
                "-i", str(spec["src_avi"]),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path)
            ]

        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"[!] Error muxing {spec['filename']}: {res.stderr}")
            sys.exit(1)
        print(f"  [+] Built: {spec['filename']} ({out_path.stat().st_size / 1024:.1f} KB)")

    print("\n=================================================================")
    print("STEP 2: Executing /analyze-offline-video on all 6 clips")
    print("=================================================================")

    results_report = []

    for spec in CLIPS_SPEC:
        clip_path = DEMO_DIR / spec["filename"]
        transcript = TRANSCRIPTS[spec["sentence_id"]]

        with open(clip_path, "rb") as f:
            resp = client.post(
                "/analyze-offline-video",
                files={"video": (clip_path.name, f, "video/mp4")},
                data={"expected_transcript": transcript},
            )

        assert resp.status_code == 200, f"Analysis failed for {spec['filename']}: {resp.text}"
        data = resp.json()
        spec["result"] = data
        results_report.append(spec)

        print(f"\nClip: {spec['filename']}")
        print(f"  Label:                {spec['label']}")
        print(f"  Expected Transcript:  \"{transcript}\"")
        print(f"  Verdict:              {data['verdict_text']}")
        print(f"  P(fake):              {data['deepfake_probability'] * 100:.2f}% (prob={data['deepfake_probability']})")
        print(f"  P_vis:                {data['p_visual']}")
        print(f"  P_desync:             {data['p_desync']}")
        print(f"  S_desync:             {data['desync_score']}")
        print(f"  Failed Closure Ratio: {data['failed_closure_ratio']}")

    print("\n=================================================================")
    print("STEP 3: Writing data/demo_clips/README.txt Cheat Sheet")
    print("=================================================================")

    readme_path = DEMO_DIR / "README.txt"
    lines = [
        "=" * 78,
        "SECUREKYC AI - OFFLINE VIDEO FORENSIC DEMO CHEAT SHEET",
        "=" * 78,
        "",
        "This directory contains 6 curated test clips for demonstrating the offline",
        "video analysis frontend (accessible at http://127.0.0.1:8000/offline).",
        "",
        "HOW TO DEMO:",
        "1. Open http://127.0.0.1:8000/offline in your browser.",
        "2. Drag-and-drop or select one of the 6 MP4 clips below.",
        "3. Copy-paste the exact Expected Transcript into the transcript input box.",
        "4. Click 'Run Deepfake Analysis'.",
        "5. Observe the continuous probability bar, branch metrics breakdown, and research caveat.",
        "",
        "NOTE ON TRANSCRIPTS:",
        "All 6 clips use standard TIMIT sentence 'sa2'. You only need to copy this single",
        "transcript phrase once to your clipboard to test all 6 clips:",
        "  Don't ask me to carry an oily rag like that.",
        "",
        "-" * 78,
        "CURATED DEMO CLIPS DIRECTORY & PRE-RECORDED LIVE RESULTS",
        "-" * 78,
        "",
    ]

    for idx, spec in enumerate(results_report, start=1):
        res = spec["result"]
        lines.extend([
            f"CLIP {idx}: {spec['filename']}",
            f"  Ground Truth:         {spec['ground_truth']}",
            f"  Dataset & Speaker:    {spec['label']}",
            f"  Exact Transcript:     \"{TRANSCRIPTS[spec['sentence_id']]}\"",
            f"  Expected Verdict:     {res['verdict_text']}",
            f"  Deepfake Probability: {res['deepfake_probability'] * 100:.2f}% (P_fake = {res['deepfake_probability']:.4f})",
            f"  Visual Branch Score:  P_vis = {res['p_visual']:.4f}",
            f"  Desync Branch Score:  P_desync = {res['p_desync']:.4f}",
            f"  Acoustic Desync (S):  S_desync = {res['desync_score']:.4f} (Threshold tau = 0.0242)",
            f"  Failed Closure Ratio: {res['failed_closure_ratio']:.4f}",
            f"  Evaluation Mode:      {res['branch_used']}",
            "",
        ])

    lines.extend([
        "=" * 78,
        "SUMMARY OF DEMO RESULTS ACROSS ALL 6 CLIPS",
        "=" * 78,
        "Genuine Clips (3 distinct speakers):",
        f"  1. real_01_fadg0_sa2.mp4:         P(fake) = {results_report[0]['result']['deepfake_probability']*100:.2f}% -> {results_report[0]['result']['verdict_text']}",
        f"  2. real_02_fcmh0_sa2.mp4:         P(fake) = {results_report[1]['result']['deepfake_probability']*100:.2f}% -> {results_report[1]['result']['verdict_text']}",
        f"  3. real_03_fdac1_sa2.mp4:         P(fake) = {results_report[2]['result']['deepfake_probability']*100:.2f}% -> {results_report[2]['result']['verdict_text']}",
        "",
        "Deepfake Clips (3 distinct speakers):",
        f"  4. fake_01_fadg0_sa2_fram1.mp4:   P(fake) = {results_report[3]['result']['deepfake_probability']*100:.2f}% -> {results_report[3]['result']['verdict_text']}",
        f"  5. fake_02_faks0_sa2_fedw0.mp4:   P(fake) = {results_report[4]['result']['deepfake_probability']*100:.2f}% -> {results_report[4]['result']['verdict_text']}",
        f"  6. fake_03_fcft0_sa2_fdrd1.mp4:   P(fake) = {results_report[5]['result']['deepfake_probability']*100:.2f}% -> {results_report[5]['result']['verdict_text']}",
        "=" * 78,
    ])

    readme_content = "\n".join(lines)
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"  [+] Saved cheat sheet to: {readme_path}")
    print("\n=================================================================")
    print("DEMO CLIPS GENERATION AND PRE-RECORDING COMPLETED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    build_clips()
