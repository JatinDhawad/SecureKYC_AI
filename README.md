# SecureKYC AI

> **Official Project Title**: Dynamic Bilabial-Targeted Active Prompting for Real-Time Deepfake Injection Defense in Video KYC Protocols  
> **Academic Alignment**: Capstone Project & Scopus-Indexed Research Paper Target  
> **Target Hardware**: Apple Silicon Mac (PyTorch Metal Performance Shaders `torch.device("mps")`)

---

## 1. Executive Summary & Core Mission

Remote identity verification (Know Your Customer / KYC) protocols used by banks, fintechs, and government portals are increasingly exploited by real-time AI synthetic media (live deepfakes). Attackers deploy live face-swapping engines (e.g. DeepFaceLive, Roop-Live) routed through virtual camera drivers (OBS Virtual Camera, ManyCam) to spoof digital onboarding sessions.

### The Research Solution: SecureKYC AI
Legacy passive liveness checks fail against live injection attacks because a human actor physically drives the deepfake in real time. **SecureKYC AI** introduces a dynamic, interactive challenge-response protocol:

1. **Dynamic Phoneme-Targeted Prompting**: Prompts the user with unpredictable phrases packed with bilabial consonants (`/m/`, `/b/`, `/p/`) like *"My Bank Passcode"*. Bilabials force complete physical lip closure prior to acoustic release.
2. **Cross-Modal Phoneme-Viseme Desynchronization Metric**: Computes millisecond lip aperture distance trajectory ($D_{\text{lip}}(t)$) desync penalties during acoustic bilabial sound occurrences.
3. **Multimodal PyTorch Fusion Classifier**: Fuses spatial landmark dynamics, temporal sequence features, and our custom desync score into a unified PyTorch neural network trained locally (`MultimodalKYCFusionNet`).

---

## 2. Research Claims for Publication

1. **Active Bilabial-Targeted Prompting Protocol**: Unlike generic prompts (blinking, head turning), dynamically generates bilabial plosive phrases forcing physical lip closure.
2. **Sub-33ms Causal Desynchronization Metric**: Lightweight real-time metric correlating acoustic phoneme timestamps with 3D lip-aperture distance trajectories within 50ms streaming windows.
3. **Multimodal PyTorch Feature Fusion Architecture**: A custom PyTorch neural network combining Conv1D spatial encoders, Bi-LSTM temporal sequence layers, and our custom cross-modal desync metric.

---

## 3. Directory Structure

```
secure_kyc_ai/
├── README.md                  # Project setup & documentation
├── requirements.txt            # Python dependency manifest
├── config.py                  # Global configurations & thresholds
├── data/
│   ├── raw/                   # FaceForensics++, VidTIMIT raw clips
│   └── processed/             # Extracted feature tensors (.pt / .npy)
├── src/                       # Core Source Code
│   ├── vision/
│   │   └── lip_tracker.py     # MediaPipe 3D lip-tracking
│   ├── audio/
│   │   └── phoneme_extractor.py # Wav2Vec2 bilabial timestamp extractor
│   ├── metrics/
│   │   └── desync_metric.py   # Cross-modal Phoneme-Viseme desync algorithm
│   └── models/
│       ├── fusion_net.py      # PyTorch Multimodal Fusion classifier
│       ├── train.py           # Training loop & validation
│       └── evaluate.py        # ROC-AUC, EER calculation
├── app/                       # Full-Stack Web Streaming Application
│   ├── main.py                # FastAPI server & WebSocket handlers
│   └── static/                # HTML5/JS Dashboard UI
├── paper/                     # IEEE Research Paper Drafts
│   └── research_paper_draft.md
└── tests/                     # Automated unit and integration tests
```

---

## 4. Setup & Installation

### Environment Setup
```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
python3 -m pytest tests/ -v
```
