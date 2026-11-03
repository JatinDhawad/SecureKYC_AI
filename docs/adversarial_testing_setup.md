# Adversarial Testing Setup Guide: OBS Studio + DeepFaceLive / Virtual Camera Pipeline

This guide details the step-by-step procedure for installing, configuring, and executing adversarial real-time deepfake injection testing against the LipGuard AI (SecureKYC AI) video KYC verification system.

---

## 1. Environment & Prerequisites

- **Host Operating System**: macOS (Apple Silicon / Intel) or Linux
- **Required Software Packages**:
  - **OBS Studio** (v29.1+ with built-in macOS OBS Virtual Camera plugin)
  - **DeepFaceLive** (or **Roop-Live** / **SimSwap**)
  - **LipGuard AI Server**: Python 3.10+ / PyTorch on `torch.device("mps")`

---

## 2. OBS Studio & OBS Virtual Camera Setup

### Installation
On macOS, install OBS Studio via Homebrew:
```bash
brew install --cask obs
```

### Virtual Camera Configuration
1. Launch **OBS Studio**.
2. Under **Sources**, click `+` $\rightarrow$ Select **Video Capture Device** $\rightarrow$ Name it `Physical Webcam`.
3. Select your hardware webcam (e.g., *FaceTime HD Camera* or *USB Video Device*) as the device.
4. Click **Start Virtual Camera** in the bottom-right **Controls** panel.
5. Verify that `OBS Virtual Camera` is now exposed as an active system video input device.

---

## 3. DeepFaceLive / Roop-Live Deepfake Virtual Camera Pipeline

### DeepFaceLive Architecture
DeepFaceLive injects real-time neural face-swaps over live webcam feeds:

```
[ Physical Webcam / Video Clip ] ──► [ DeepFaceLive Frame Grabber ]
                                              │
                                              ▼
                                   [ Face Detector (YOLOv8) ]
                                              │
                                              ▼
                                   [ Face Aligner & Marker ]
                                              │
                                              ▼
                                [ DFL Neural Swapper (DFM Model) ]
                                              │
                                              ▼
                                 [ Face Merger & Color Corrector ]
                                              │
                                              ▼
                                   [ OBS Studio Window Capture ]
                                              │
                                              ▼
                                   [ OBS Virtual Camera Stream ]
                                              │
                                              ▼
                                 [ LipGuard AI Web Client (app/main.py) ]
```

### Execution Steps
1. Launch **DeepFaceLive**.
2. Select Source: `File` (Target Face Video) or `Webcam` (Live Source).
3. Load a target face model (`.dfm` format) or pre-trained face-swap weight.
4. Set Output Window: `Virtual Camera` or `Screen Preview Window`.
5. In OBS Studio, add a **Window Capture** source selecting the DeepFaceLive preview window.
6. Click **Start Virtual Camera** in OBS Studio.

---

## 4. Configuring LipGuard AI Web Frontend (`app/main.py`)

The LipGuard AI frontend ([app/static/index.html](file:///Users/jatindhawad/SecureKYC_AI/app/static/index.html) & [app/static/app.js](file:///Users/jatindhawad/SecureKYC_AI/app/static/app.js)) has been enhanced with dynamic camera device selection:

1. Open `http://127.0.0.1:8000` in your web browser.
2. In the **Live Webcam Verification** dashboard control bar, locate the **Video Input Device** dropdown.
3. Select **`OBS Virtual Camera`** from the enumerated device list.
4. Click **Initialize Camera** to bind the stream.
5. Click **Run 5-Sec Liveness Test** while reading the issued bilabial prompt phrase.

---

## 5. Live Adversarial Attack Evaluation Protocol

During live adversarial testing sessions:
1. Issue a bilabial-dense challenge prompt (e.g. *"My Bank Passcode"*).
2. DeepFaceLive performs real-time neural face replacement over the video stream.
3. Because deepfake neural rendering exhibits subtle temporal desynchronization between acoustic bilabials ($/\text{b}/, /\text{m}/, /\text{p}/$) and physical lip closure, the cross-modal metric yields elevated $S_{\text{desync}}$ penalties.
4. MultimodalKYCFusionNet evaluates the 131-dim representation and returns `DEEPFAKE INJECTION DETECTED` ($P(\text{Deepfake}) > 0.5$).
5. Labeled adversarial attack tensors are persisted to `data/processed/adversarial_attacks.pt` for validation benchmarking.
