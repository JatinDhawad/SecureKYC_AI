#!/usr/bin/env python3
"""
Generate publication-grade academic architectural and conceptual diagrams for SecureKYC AI:
1. threat_model_diagram.png (Fig. 1)
2. speech_aerodynamics_diagram.png (Fig. 2)
3. system_architecture_diagram.png (Fig. 4)

Designed specifically for IEEE 2-column format:
- High contrast, clean vector lines, human-drawn academic engineering aesthetic.
- Zero SaaS card wrappers, zero pill badges, zero drop shadows.
- Column-optimized aspect ratio (480px width) with large, crisp typography (11px-14px)
  so text is immediately readable (~9pt) in the IEEE paper column.
- Renders via Playwright headless Chromium at device_scale_factor=2 (high DPI).
"""

from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
DOCS_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. FIG 1: ACADEMIC THREAT MODEL FLOWCHART
# -----------------------------------------------------------------------------
HTML_THREAT_MODEL = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #FFFFFF;
    width: 480px;
    padding: 6px;
  }
  .flow-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
  }
  .box {
    width: 100%;
    background: #FFFFFF;
    border: 1.4px solid #1E293B;
    border-radius: 4px;
    padding: 7px 10px;
    text-align: left;
  }
  .box-title {
    font-size: 13px;
    font-weight: 700;
    color: #0F172A;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .domain-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: #475569;
  }
  .box-desc {
    font-size: 11.2px;
    color: #334155;
    margin-top: 2px;
    line-height: 1.25;
  }
  .arrow-flow {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2px 0;
  }
  .arrow-flow span {
    font-size: 10px;
    font-weight: 600;
    color: #475569;
    background: #FFFFFF;
    padding: 0 4px;
  }
  .arrow-flow svg {
    width: 13px;
    height: 11px;
    fill: #1E293B;
  }
  .boundary-line {
    width: 100%;
    margin: 5px 0;
    border-top: 1.6px dashed #DC2626;
    position: relative;
    text-align: center;
  }
  .boundary-line span {
    position: relative;
    top: -7.5px;
    background: #FFFFFF;
    padding: 0 6px;
    font-size: 9.5px;
    font-weight: 800;
    color: #DC2626;
    letter-spacing: 0.4px;
    text-transform: uppercase;
  }
  .alert-box {
    background: #FEF2F2;
    border-color: #DC2626;
  }
  .alert-box .box-title {
    color: #991B1B;
  }
  .alert-box .domain-label {
    color: #DC2626;
    font-weight: 800;
  }
</style>
</head>
<body>
<div class="flow-container">
  <!-- Step 1 -->
  <div class="box">
    <div class="box-title">
      <span>1. Attacker &amp; Physical Rig (Puppeteer)</span>
      <span class="domain-label">[Physical Domain]</span>
    </div>
    <div class="box-desc">Live operator sits before webcam, miming challenge prompts &amp; active head movements.</div>
  </div>

  <div class="arrow-flow">
    <span>physical video &amp; audio</span>
    <svg viewBox="0 0 24 24"><path d="M12 18l-6-8h12z"/></svg>
  </div>

  <!-- Step 2 -->
  <div class="box" style="border-color:#2563EB;">
    <div class="box-title" style="color:#1D4ED8;">
      <span>2. Generative Engine (DeepFaceLive / Roop)</span>
      <span class="domain-label" style="color:#2563EB;">[Neural Synthesis]</span>
    </div>
    <div class="box-desc">Real-time model renders victim facial identity onto puppeteer keypoints frame-by-frame.</div>
  </div>

  <div class="arrow-flow">
    <span>rendered synthetic frame buffer</span>
    <svg viewBox="0 0 24 24"><path d="M12 18l-6-8h12z"/></svg>
  </div>

  <!-- Step 3 -->
  <div class="box" style="border-color:#D97706;">
    <div class="box-title" style="color:#B45309;">
      <span>3. Kernel Loopback Driver (OBS VirtualCam / v4l2)</span>
      <span class="domain-label" style="color:#D97706;">[OS Kernel Device]</span>
    </div>
    <div class="box-desc">Virtual device intercepts synthetic stream, exposing it as a valid USB webcam (/dev/video*).</div>
  </div>

  <div class="boundary-line">
    <span>OS Kernel / Sandboxed Browser Boundary (Driver Verification Denied)</span>
  </div>

  <!-- Step 4 -->
  <div class="box" style="border-color:#0284C7;">
    <div class="box-title" style="color:#0369A1;">
      <span>4. Sandboxed Browser: WebRTC Video Session</span>
      <span class="domain-label" style="color:#0284C7;">[Client Sandbox]</span>
    </div>
    <div class="box-desc">Browser ingests pristine synthetic stream devoid of physical optical cues (no glare or moir&eacute;).</div>
  </div>

  <div class="arrow-flow">
    <span>live WebRTC transmission</span>
    <svg viewBox="0 0 24 24"><path d="M12 18l-6-8h12z"/></svg>
  </div>

  <!-- Step 5 -->
  <div class="box alert-box">
    <div class="box-title">
      <span>5. Remote KYC Server: Legacy Passive PAD Fails</span>
      <span class="domain-label">[Vulnerability]</span>
    </div>
    <div class="box-desc">ISO/IEC 30107-3 passive checks pass; puppeteer satisfies liveness &rarr; Spoof admitted.</div>
  </div>
</div>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# 2. FIG 2: ACADEMIC SPEECH AERODYNAMICS SCHEMATIC
# -----------------------------------------------------------------------------
HTML_SPEECH_AERODYNAMICS = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #FFFFFF;
    width: 480px;
    padding: 6px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    width: 100%;
  }
  .panel {
    border: 1.5px solid #1E293B;
    border-radius: 4px;
    padding: 7px;
    background: #FFFFFF;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .panel-human {
    border-color: #166534;
  }
  .panel-fake {
    border-color: #991B1B;
  }
  .panel-head {
    font-size: 11.2px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding-bottom: 4px;
    margin-bottom: 5px;
    border-bottom: 1.2px solid #E2E8F0;
    text-align: center;
  }
  .panel-human .panel-head { color: #166534; border-color: #BBF7D0; }
  .panel-fake .panel-head { color: #991B1B; border-color: #FECDD3; }

  .step-box {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 3px;
    padding: 5px 6px;
    margin-bottom: 3px;
  }
  .step-title {
    font-size: 10.6px;
    font-weight: 700;
    color: #0F172A;
  }
  .step-desc {
    font-size: 9.8px;
    color: #334155;
    line-height: 1.2;
    margin-top: 1px;
  }
  .arrow-d {
    text-align: center;
    font-size: 9px;
    color: #64748B;
    margin: -2px 0 2px 0;
  }
  .math-box {
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 3px;
    padding: 5px 6px;
    text-align: center;
    margin: 4px 0;
  }
  .panel-human .math-box {
    background: #F0FDF4;
    border-color: #86EFAC;
  }
  .panel-fake .math-box {
    background: #FEF2F2;
    border-color: #FECDD3;
  }
  .math-formula {
    font-family: "Cambria Math", "Times New Roman", Georgia, serif;
    font-size: 10.8px;
    font-weight: 700;
    color: #0F172A;
  }
  .math-note {
    font-size: 9.2px;
    color: #475569;
    margin-top: 1px;
  }
  .verdict {
    font-size: 9.6px;
    font-weight: 800;
    text-align: center;
    padding: 3px;
    border-radius: 2px;
    margin-top: 2px;
  }
  .verdict-pass {
    background: #DCFCE7;
    color: #166534;
    border: 1px solid #86EFAC;
  }
  .verdict-fail {
    background: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
  }
</style>
</head>
<body>
<div class="grid">
  <!-- Human Panel -->
  <div class="panel panel-human">
    <div>
      <div class="panel-head">(a) Genuine Human Speaker</div>
      <div class="step-box">
        <div class="step-title">1. Velopharyngeal Seal</div>
        <div class="step-desc">Soft palate isolates nasal cavity during /b/ and /p/ implosion.</div>
      </div>
      <div class="arrow-d">&darr;</div>
      <div class="step-box">
        <div class="step-title">2. Airtight Vermilion Contact</div>
        <div class="step-desc">Lips compress firmly. Aperture drops below &tau; = 0.0242.</div>
      </div>
      <div class="arrow-d">&darr;</div>
      <div class="step-box">
        <div class="step-title">3. Intra-Oral Pressure Buildup</div>
        <div class="step-desc">Airflow builds oral pressure: P_oral &approx; 6 to 10 cm H2O.</div>
      </div>
    </div>
    <div>
      <div class="math-box">
        <div class="math-formula">P_oral - P_amb = (&rho;/2)&middot;(U / A_orifice)&sup2;</div>
        <div class="math-note">A_orifice &rarr; 0 mandates high pressure seal</div>
      </div>
      <div class="verdict verdict-pass">&check; Invariant Satisfied (D_lip &le; &tau;)</div>
    </div>
  </div>

  <!-- DeepFake Panel -->
  <div class="panel panel-fake">
    <div>
      <div class="panel-head">(b) Neural Reenactment Deepfake</div>
      <div class="step-box">
        <div class="step-title">1. Latent Temporal Smoothing</div>
        <div class="step-desc">Autoencoders interpolate expressions; fast 40-80 ms closures blurred.</div>
      </div>
      <div class="arrow-d">&darr;</div>
      <div class="step-box">
        <div class="step-title">2. Missing Biomechanical Loss</div>
        <div class="step-desc">Losses optimize pixel similarity (L1/SSIM); no air-seal constraint.</div>
      </div>
      <div class="arrow-d">&darr;</div>
      <div class="step-box">
        <div class="step-title">3. Persistent Aperture Gap</div>
        <div class="step-desc">Mouth seam blending preserves luminous gap (D_lip &gt; 0.07-0.16).</div>
      </div>
    </div>
    <div>
      <div class="math-box">
        <div class="math-formula" style="color:#991B1B;">P_oral &approx; P_amb &nbsp; (A_orifice &gt; 0)</div>
        <div class="math-note">Acoustic plosive sounds while lips stay open</div>
      </div>
      <div class="verdict verdict-fail">&cross; Violation Flagged (S_desync &gt; &tau;)</div>
    </div>
  </div>
</div>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# 3. FIG 4: ACADEMIC SYSTEM ARCHITECTURE SCHEMATIC
# -----------------------------------------------------------------------------
HTML_SYSTEM_ARCHITECTURE = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #FFFFFF;
    width: 480px;
    padding: 4px;
  }
  .arch-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
  }
  .top-prompt {
    border: 1.3px solid #1E293B;
    background: #F8FAFC;
    border-radius: 3px;
    padding: 4px 7px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .top-prompt .title {
    font-size: 9.6px;
    font-weight: 800;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    white-space: nowrap;
  }
  .top-prompt .prompt {
    font-family: "Consolas", monospace;
    font-size: 9.0px;
    font-weight: 700;
    color: #1D4ED8;
    background: #EFF6FF;
    padding: 1.5px 5px;
    border-radius: 2px;
    border: 1px solid #93C5FD;
    white-space: nowrap;
  }
  .streams-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .stream-box {
    border: 1.3px solid #CBD5E1;
    border-radius: 3px;
    padding: 5px;
    background: #FFFFFF;
  }
  .stream-title {
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: #0F172A;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 2px;
    margin-bottom: 3px;
    display: flex;
    justify-content: space-between;
  }
  .stream-title .badge {
    font-size: 8.5px;
    font-weight: 700;
    padding: 1px 3px;
    border-radius: 2px;
  }
  .badge-vis { background: #EFF6FF; color: #1D4ED8; border: 0.5px solid #BFDBFE; }
  .badge-aud { background: #FEF3C7; color: #92400E; border: 0.5px solid #FDE68A; }

  .sub-block {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 2px;
    padding: 3px 4px;
    margin-bottom: 2px;
  }
  .sub-block-title {
    font-size: 9.6px;
    font-weight: 700;
    color: #1E293B;
  }
  .sub-block-desc {
    font-size: 8.8px;
    color: #334155;
    line-height: 1.15;
    margin-top: 0.5px;
  }
  .arr-down {
    text-align: center;
    font-size: 8px;
    color: #64748B;
    margin: -1px 0 1px 0;
  }
  .model-block {
    background: #EFF6FF;
    border: 1.1px solid #3B82F6;
    border-radius: 2px;
    padding: 3px 4px;
    text-align: center;
  }
  .model-title {
    font-size: 9.5px;
    font-weight: 800;
    color: #1D4ED8;
  }
  .model-out {
    font-size: 8.6px;
    font-weight: 700;
    color: #2563EB;
    margin-top: 0.5px;
  }

  /* Fusion block */
  .fusion-box {
    border: 1.4px solid #2563EB;
    border-radius: 3px;
    padding: 5px 7px;
    background: #F8FAFC;
  }
  .fusion-title {
    font-size: 10px;
    font-weight: 800;
    color: #1E3A8A;
    text-transform: uppercase;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3px;
    border-bottom: 1px solid #BFDBFE;
    padding-bottom: 2px;
  }
  .fusion-title span:last-child {
    font-size: 8.8px;
    font-weight: 700;
    color: #2563EB;
  }
  .fusion-inner {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    margin-bottom: 3px;
  }
  .f-col {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 2px;
    padding: 3px 4px;
    font-size: 8.8px;
    line-height: 1.2;
    color: #334155;
  }
  .f-col strong { color: #0F172A; }
  .decision-row {
    background: #0F172A;
    color: #FFFFFF;
    border-radius: 2px;
    padding: 3.5px 7px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 9.2px;
    font-weight: 700;
  }
  .decision-row .verdicts {
    display: flex;
    gap: 5px;
    align-items: center;
  }
  .decision-row .v-pass { color: #4ADE80; }
  .decision-row .v-fail { color: #F87171; }
  .decision-row .latency {
    font-size: 8.6px;
    color: #94A3B8;
    white-space: nowrap;
    margin-left: 8px;
  }
</style>
</head>
<body>
<div class="arch-container">
  <!-- Active Challenge -->
  <div class="top-prompt">
    <div class="title">Active Challenge-Response Protocol</div>
    <div class="prompt">Phrase: "My Bank Passcode" (/m/, /b/, /p/)</div>
  </div>

  <!-- Dual Streams -->
  <div class="streams-grid">
    <!-- Visual Stream -->
    <div class="stream-box" style="border-color:#93C5FD;">
      <div class="stream-title">
        <span>Visual Pipeline</span>
        <span class="badge badge-vis">30 FPS Stream</span>
      </div>
      <div class="sub-block">
        <div class="sub-block-title">WebRTC Frame Ingestion</div>
        <div class="sub-block-desc">Continuous RGB video decoding at 30 FPS.</div>
      </div>
      <div class="arr-down">&darr;</div>
      <div class="sub-block">
        <div class="sub-block-title">MediaPipe 3D Face Mesh</div>
        <div class="sub-block-desc">Vermilion #13, #14 &amp; commissures #61, #291.</div>
      </div>
      <div class="arr-down">&darr;</div>
      <div class="sub-block">
        <div class="sub-block-title">Normalized Lip Aperture</div>
        <div class="sub-block-desc">D_lip(t) = ||p13 - p14|| / ||p61 - p291||.</div>
      </div>
      <div class="arr-down">&darr;</div>
      <div class="model-block">
        <div class="model-title">VisionOnlyNet (1D-CNN + BiLSTM)</div>
        <div class="model-out">Output: Posterior Probability p_vis</div>
      </div>
    </div>

    <!-- Acoustic Stream -->
    <div class="stream-box" style="border-color:#FCD34D;">
      <div class="stream-title">
        <span>Acoustic Pipeline</span>
        <span class="badge badge-aud">Post-Utterance</span>
      </div>
      <div class="sub-block">
        <div class="sub-block-title">Acoustic Signal Capture</div>
        <div class="sub-block-desc">Single-pass 16 kHz mono waveform at utterance end.</div>
      </div>
      <div class="arr-down">&darr;</div>
      <div class="sub-block">
        <div class="sub-block-title">Meta MMS_FA Acoustic Model</div>
        <div class="sub-block-desc">Calculates frame phonetic emission matrix E.</div>
      </div>
      <div class="arr-down">&darr;</div>
      <div class="sub-block">
        <div class="sub-block-title">Dynamic Trellis Alignment</div>
        <div class="sub-block-desc">Pins bilabial timestamp boundaries T_bilabial.</div>
      </div>
      <div class="arr-down">&darr;</div>
      <div class="model-block" style="border-color:#D97706; background:#FFFBEB;">
        <div class="model-title" style="color:#B45309;">DesyncOnlyNet (MLP Branch)</div>
        <div class="model-out" style="color:#D97706;">Output: Posterior Probability p_des</div>
      </div>
    </div>
  </div>

  <!-- Constrained Late Fusion -->
  <div class="fusion-box">
    <div class="fusion-title">
      <span>Constrained Late-Fusion Meta-Learner</span>
      <span>Bounds: W_vis &ge; 0, W_desync &ge; 0</span>
    </div>
    <div class="fusion-inner">
      <div class="f-col">
        <strong>Desync Feature Vector:</strong><br>
        v_desync = [S_desync, P_max, r_fail]&trade; across W_k = [f_start-2, f_end+2]
      </div>
      <div class="f-col">
        <strong>Constrained Logistic Meta-Learner:</strong><br>
        z = W_vis&middot;p_vis + W_desync&middot;p_des + b &nbsp;&rarr;&nbsp; p_fused = &sigma;(z)
      </div>
    </div>
    <div class="decision-row">
      <div class="verdicts">
        <span>VERDICT:</span>
        <span class="v-pass">&check; GENUINE</span>
        <span>/</span>
        <span class="v-fail">&cross; SPOOF INJECTION</span>
      </div>
      <div class="latency">Total Latency: 224.3 &plusmn; 21.7 ms (Apple Silicon MPS)</div>
    </div>
  </div>
</div>
</body>
</html>
"""

def generate_all_diagrams():
    diagrams = [
        ("threat_model_diagram.png", HTML_THREAT_MODEL, 480, 270),
        ("speech_aerodynamics_diagram.png", HTML_SPEECH_AERODYNAMICS, 480, 260),
        ("system_architecture_diagram.png", HTML_SYSTEM_ARCHITECTURE, 480, 310),
    ]

    print("[*] Launching Playwright to render human-made, publication-grade diagrams...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for fname, html_content, w, h in diagrams:
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            page.set_content(html_content, wait_until="networkidle")
            page.wait_for_timeout(200)

            out_paper = FIGURES_DIR / fname
            out_docs = DOCS_FIGURES_DIR / fname

            page.screenshot(path=str(out_paper), full_page=True)
            shutil.copy(str(out_paper), str(out_docs))
            print(f"[✓] Rendered: {out_paper} ({out_paper.stat().st_size / 1024:.1f} KB)")
            page.close()
        browser.close()
    print("[✓] All 3 diagrams generated successfully in paper/figures and docs/figures!")

if __name__ == "__main__":
    generate_all_diagrams()
