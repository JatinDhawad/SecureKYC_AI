# LipGuard AI (SecureKYC AI) - Academic & Defense Panel Brief

This document provides rigorous, technical responses to key panel-defense questions for academic defense and peer review. Every answer is grounded directly in the codebase implementation, citing absolute file paths, function definitions, mathematical formulations, and empirical evaluation metrics.

---

## Question 1: "Isn't this just an API wrapper?"

### Answer:
No. LipGuard AI is an original, multi-stage signal-processing and deep learning engine built ground-up in PyTorch. It does not rely on third-party deepfake detection cloud APIs or black-box web services. Every score, feature vector, and verdict is computed directly on local compute hardware (accelerated via Apple Silicon `torch.device("mps")`).

The proprietary core consists of five decoupled, custom-engineered modules:

1. **Custom 3D Spatial Lip Aperture Tracking**:
   Implemented in [src/vision/lip_tracker.py](file:///Users/jatindhawad/SecureKYC_AI/src/vision/lip_tracker.py#L45-L85), `LipTracker.process_frame()` extracts 3D inner lip landmarks (#13 upper, #14 lower) and normalizes aperture by 3D facial vertical height (nose tip #1 to chin #152):
   $$D_{\text{lip}}(t) = \frac{\|P_{13}(t) - P_{14}(t)\|_2}{\|P_1(t) - P_{152}(t)\|_2 + 10^{-6}}$$
   Short tracking drops ($\le 3$ frames) are dynamically interpolated with gap-constrained bounding in `interpolate_missing_apertures()`.

2. **Acoustic Forced Alignment Phoneme Extraction**:
   Implemented in [src/audio/phoneme_extractor.py](file:///Users/jatindhawad/SecureKYC_AI/src/audio/phoneme_extractor.py#L50-L110), `PhonemeExtractor.align_forced_transcript()` resamples audio to 16 kHz, computes logit emissions via `torchaudio.pipelines.MMS_FA`, and executes dynamic programming trellis forced alignment via `torchaudio.functional.forced_align()`.

3. **Empirically Calibrated Cross-Modal Desynchronization Metric**:
   Implemented in [src/metrics/desync_metric.py](file:///Users/jatindhawad/SecureKYC_AI/src/metrics/desync_metric.py#L40-L90), `DesynchronizationMetric.compute_desync_score()` evaluates lip aperture across a $[t-2, t+2]$ frame temporal window surrounding each acoustic bilabial event:
   $$\text{Penalty}(t) = \max\left(0, \min_{w \in [t-2, t+2]} D_{\text{lip}}(w) - 0.0242\right)$$
   The threshold $\tau = 0.0242$ was calibrated empirically at the 95th percentile across 320 verified authentic bilabial closure events ($N=322$ initial pass $\to$ $N=320$ after canonical transcript verification) in [src/metrics/calibrate_threshold.py](file:///Users/jatindhawad/SecureKYC_AI/src/metrics/calibrate_threshold.py).

4. **PyTorch Multimodal Fusion Neural Network**:
   Implemented in [src/models/fusion_net.py](file:///Users/jatindhawad/SecureKYC_AI/src/models/fusion_net.py#L30-L100), `MultimodalKYCFusionNet` passes normalized 1D lip sequence through 1D Conv layers and a 2-layer Bi-LSTM to extract a 128-dim visual embedding, concatenating it with a 3-dim desync vector $[S_{\text{desync}}, \text{max\_penalty}, \text{failed\_ratio}]$ into a 131-dim fusion classifier.

5. **Real-Time Challenge-Response Engine**:
   Implemented in [app/main.py](file:///Users/jatindhawad/SecureKYC_AI/app/main.py#L165-L285), the WebSocket backend manages session-level prompt history ([app/prompt_generator.py](file:///Users/jatindhawad/SecureKYC_AI/app/prompt_generator.py#L43-L115)) and delivers an average post-utterance inference latency of **224 ms** (mean server latency: $223.7 \pm 21.6$ ms, client end-to-end: $224.3 \pm 21.7$ ms, full-pipeline range: $184.4$–$262.0$ ms, initial cold start: $1875.1$ ms) verified in live WebSocket sessions.

---

## Question 2: "Isn't live detection identical to offline detection?"

### Answer:
No. Live Video KYC liveness verification introduces fundamental operational constraints, security threat models, and latency requirements that render offline video classification algorithms inadequate:

1. **Active Challenge-Response Protocol vs. Static Video Analysis**:
   Offline detectors evaluate pre-recorded, static video files where an attacker can pre-render arbitrary deepfake face-swaps. In contrast, LipGuard AI operates an **Active Prompt Protocol**: the server generates a dynamic, non-repeating bilabial challenge prompt over WebSocket ([app/prompt_generator.py](file:///Users/jatindhawad/SecureKYC_AI/app/prompt_generator.py#L65-L100)) at session initialization. The user must speak that specific phrase within a 4-second window. Pre-rendered video replay attacks cannot anticipate the challenge phrase.

2. **Asynchronous Dual-Budget Architecture**:
   Offline models can spend seconds or minutes processing full video files. Live verification requires a dual-budget design implemented in [app/main.py](file:///Users/jatindhawad/SecureKYC_AI/app/main.py#L198-L285):
   - **Streaming Budget (33ms per frame)**: `LipTracker` processes incoming webcam frames in near-real-time to maintain 30 FPS telemetry.
   - **Post-Utterance Budget ($\le 1.0\text{ s}$)**: Heavy forced-alignment emission matrix computation and PyTorch multimodal forward pass execute **ONCE** after the user finishes speaking, completing in **0.193 seconds** ($193\text{ ms}$).

3. **Zero Identity Leakage Benchmarking**:
   Offline deepfake benchmarks frequently suffer from identity leakage (training and testing on clips of the same subject), producing falsely inflated accuracy. In [src/models/train.py](file:///Users/jatindhawad/SecureKYC_AI/src/models/train.py#L40-L80), `split_dataset_by_identity()` partitions data strictly by unique subject ID across 43 identities (25 train, 9 val, 9 test), guaranteeing zero identity overlap ($\text{Train} \cap \text{Test} = \emptyset$). On the held-out identity test set ($N=103$ clips: 57 genuine, 46 fake), the early-fusion baseline (`MultimodalKYCFusionNet`) achieves **65.05% accuracy, 67.86% precision, 41.30% recall, 0.5135 F1-score, 0.6423 ROC-AUC, and 43.67% EER**. Our deployed **Constrained Late-Fusion meta-learner** eliminates inter-modal gradient interference, advancing performance to **68.93% accuracy, 70.59% precision, 52.17% recall, 0.6000 F1-score, 0.7197 ROC-AUC, and 36.02% EER** (with a representative 5-fold cross-validation generalization estimate of **$0.6877 \pm 0.0449$ ROC-AUC**).

---

## Question 3: "What makes the model novel vs existing literature?"

### Answer:
Existing deepfake audio-visual lip-sync literature relies primarily on **unconstrained, blind ASR decoding** (e.g., raw Wav2Vec2 character logits) to predict phonemes from speech. Blind ASR models struggle on short, 3-to-4 second Video KYC utterances, producing noisy character transcriptions and inaccurate timestamp boundaries.

LipGuard AI introduces three major architectural innovations reflecting the actual as-built implementation:

1. **Active Prompting Exploitation via Forced Alignment**:
   Because the Video KYC system explicitly prompts the user with a known challenge phrase (e.g., *"My Bank Passcode"*), LipGuard AI eliminates unconstrained ASR decoding entirely. Implemented in [src/audio/phoneme_extractor.py](file:///Users/jatindhawad/SecureKYC_AI/src/audio/phoneme_extractor.py#L70-L115), `align_forced_transcript()` leverages `torchaudio.functional.forced_align()` with the known transcript text. This dynamic programming trellis alignment recovers exact, millisecond-accurate timestamp boundaries $T_{\text{bilabial}}$ for all bilabial consonants ($/\text{b}/, /\text{m}/, /\text{p}/$).

2. **Physical Biomechanical Bilabial Constraint ($D_{\text{lip}}$ Window Penalty)**:
   Acoustic bilabial production physically mandates total airflow obstruction through complete lip closure. Neural face-swap and lip-sync generative models frequently fail to enforce total lip closure at exact acoustic emission timestamps. LipGuard AI computes the window penalty:
   $$\text{Penalty}(t) = \max\left(0, \min_{w \in [t-2, t+2]} D_{\text{lip}}(w) - 0.0242\right)$$
   where $\tau = 0.0242$ is derived empirically from authentic VidTIMIT speakers across $N=320$ verified acoustic bilabial events at the 95th percentile ($N=322$ initial pass $\to$ $N=320$ after canonical transcript verification in [src/metrics/calibrate_threshold.py](file:///Users/jatindhawad/SecureKYC_AI/src/metrics/calibrate_threshold.py)).

3. **131-Dimensional Multimodal Feature Vector Schema**:
   In [src/models/fusion_net.py](file:///Users/jatindhawad/SecureKYC_AI/src/models/fusion_net.py#L30-L75), `MultimodalKYCFusionNet` fuses temporal visual dynamics with a 3-dimensional desynchronization feature vector $[S_{\text{desync}}, \text{max\_penalty}, \text{failed\_ratio}]$:
   - **$S_{\text{desync}}$**: Mean closure penalty across all bilabial events.
   - **$\text{max\_penalty}$**: Maximum single closure penalty duration.
   - **$\text{failed\_ratio}$**: Ratio of bilabial events failing the physical aperture threshold.

   **Real adversarial validation (live-generation) is future scope**: Any prior preliminary figures (such as 81.82% Accuracy / 0.8952 ROC-AUC) derived from synthetic uniform distributions have been completely retracted in strict compliance with the zero-synthetic-data rule. Live-generation validation using OBS Virtual Camera + DeepFaceLive was **not completed**; it is explicitly designated future scope. No numeric placeholder or fabricated claim is reported.

---

## Future Work: Live DeepFaceLive Validation (Not Completed)

**Planned but not executed.** A live-generation validation protocol was designed in which a self-trained DeepFaceLive model (.dfm) would route synthetic facial video through an OBS Virtual Camera loopback and through the identical LipGuard AI pipeline. The planned protocol specified 3 challenge phrases per session, paired genuine vs. swap recordings, and a target 600 ms measured audio offset. **This validation was not completed and is explicitly future scope.** All evaluation evidence in this brief is based exclusively on offline benchmark datasets: DeepFakeTIMIT reenactment (primary evaluation), VidTIMIT (genuine baseline), Celeb-DF v2 (visual-only negative result), FakeAVCeleb v1.2 (desync negative result), and the 8-subject offline InsightFace inswapper set. The threat model may describe virtual-camera injection as the motivating threat, but validation used offline reenactment benchmarks as a proxy.

