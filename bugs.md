# SecureKYC AI - System Bugs & Edge-Case Incident Log

This document tracks all software bugs, edge-case failures, and architectural observations discovered during end-to-end system testing (Steps B1–B3).

---

## Logged Issues & Incident Reports

### Issue #1: PyTorch `BatchNorm1d` Failure on Single-Sample Batches
- **Module Affected**: `src/models/train.py`
- **Root Cause**: When the dataset size ($N=33$ training clips) was split into mini-batches of size $16$, the final mini-batch contained exactly $1$ sample (`torch.Size([1, 64])`). `nn.BatchNorm1d` in PyTorch raises `ValueError: Expected more than 1 value per channel when training` during training mode because variance cannot be calculated for a single sample.
- **Resolution**: Updated `DataLoader` in `src/models/train.py` to pass `drop_last=True if len(train_ds) > batch_size else False`.
- **Status**: **RESOLVED**

---

### Issue #2: `torchaudio.functional.TokenSpan` Attribute Name Mismatch
- **Module Affected**: `src/audio/phoneme_extractor.py`
- **Root Cause**: The code referenced `span.token_index` during forced alignment span iteration. In `torchaudio>=2.1.0`, the `TokenSpan` dataclass field is named `span.token`.
- **Resolution**: Refactored `span.token_index` to `span.token` in `align_forced_transcript()`.
- **Status**: **RESOLVED**

---

### Issue #3: Silent Synthesis MP4 Audio Stream Handling
- **Module Affected**: `src/models/dataset_extractor.py`
- **Root Cause**: Deepfake synthesis benchmark clips (e.g., Celeb-DF `Celeb-synthesis`) are silent face-swap video files lacking an audio track. Attempting audio extraction on these clips returned zero audio samples.
- **Resolution**: Per Rule 1 (Zero Synthetic Data / Explicit Failure Logging), rather than zero-filling synthetic audio or silently swallowing errors, `dataset_extractor.py` explicitly records `AUDIO_CORRUPT` to `data/processed/extraction_failures.csv` ($1,000$ failure entries logged).
- **Status**: **RESOLVED & LOGGED FOR PAPER**

---

### Issue #4: Playwright Browser Subagent mac-arm64 CDN Driver Error
- **Module Affected**: Browser subagent integration
- **Root Cause**: Playwright CDN returned HTTP `404 Not Found` when trying to download the driver binary for `playwright-1.57.0-mac-arm64.zip`.
- **Resolution**: Endpoint accessibility, OpenAPI schema (`/openapi.json`), health check (`/health`), and WebSocket challenge-response protocol (`/ws/verify`) were programmatically verified via automated test scripts (`tests/test_end_to_end_sessions.py`).
- **Status**: **NOTED IN DOCUMENTATION**
