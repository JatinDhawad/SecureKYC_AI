# LipGuard AI - Live Demo & Defense Rehearsal Cheat Sheet
*A concise, one-page quick-glance cheat sheet for live system demonstration and defense panels.*

---

## 1. Opening Framing (Say This First)
> "LipGuard AI is an active challenge-response verification system that detects video deepfakes by cross-referencing millisecond 3D lip-aperture tracking against Wav2Vec2 acoustic forced alignment. Crucially, this is **not** a generic deepfake detector for arbitrary face swaps—as our Celeb-DF experiments proved, face swaps that preserve original actor audio leave viseme timing intact. Instead, LipGuard specifically targets **facial reenactment, voice-driven synthesis, and injection lip-sync attacks**, where generative models physically fail to close the lips during bilabial consonants (/b/, /m/, /p/)."

---

## 2. Suggested Demo Clip Sequence (`data/demo_clips/`)

| Step | File | Sentence | Transcript to Paste | What to Highlight |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `real_01_fadg0_sa2.mp4` | `sa2` | `Don't ask me to carry an oily rag like that.` | **Authentic Baseline**: $P=17.22\%$, $S=0.0000$, 0/1 failed. Full closure on "me". |
| **2** | `fake_02_faks0_sa2_fedw0.mp4` | `sa2` | `Don't ask me to carry an oily rag like that.` | **Clean Desync Detection**: $P=97.36\%$, $S=0.0463 > \tau$ ($0.0242$), 1/1 failed. |
| **3** | `real_04_fgjd0_sx369.mp4` | `sx369` | `bulk paper over the living room murals.` | **Multi-Bilabial Real**: $P=25.87\%$, all 5 bilabials achieve physical seal. |
| **4** | `fake_04_msjs1_sx369_mrgg0.mp4` | `sx369` | `bulk paper over the living room murals.` | **The Closer**: $P=96.80\%$, $S=0.0485$, 4/5 bilabials fail closure ($80\%$ failure rate). |

---

## 3. Processing One-Liners (To Fill the ~300ms Inference Pause)
- **Running a Real Clip**: *"Aligning 3D aperture trajectory against audio to verify complete physical seal during acoustic bilabials..."*
- **Running a Fake Clip**: *"Running Wav2Vec2 forced alignment against facial landmark velocities to test for synthetic viseme lag..."*

---

## 4. Key Panel Questions (Updated Defense Answers)

### Q1: "Isn't this just an API wrapper?"
**Answer**: No. It is an original PyTorch engine running locally on Apple Silicon Metal (`mps`). It features custom 3D FaceMesh aperture tracking ($D_{\text{lip}}(t)$ normalized by 3D facial height), Wav2Vec2 forced-alignment trellis extraction (`MMS_FA`), an acoustic ASR prompt-verification boundary ($\tau_{\text{CER}} = 0.60$), an empirical biomechanical penalty metric ($\tau = 0.0242$), and a constrained late-fusion meta-learner combining `VisionOnlyNet` ($T=100$) and `DesyncOnlyNet` ($3\text{D}$) with non-negative weights: $\text{logit} = 5.3688 \cdot P_{\text{vis}} + 4.9111 \cdot P_{\text{desync}} - 6.0411$.

### Q2: "Isn't live detection identical to offline detection?"
**Answer**: No. Offline detectors analyze static files vulnerable to pre-rendered replay. Our live protocol issues a dynamic, non-repeating bilabial prompt over WebSocket with a 4s window. Audio is first verified via ASR decoding to prevent phrase-substitution attacks. Architecturally, it enforces a dual-budget constraint: a **33ms streaming frame budget** for continuous 3D landmark telemetry, and a single **$<200\text{ms}$ post-utterance budget** ($193\text{ms}$ observed) for alignment and neural fusion.

### Q3: "What makes this novel vs. existing literature?"
**Answer**: Literature relies on unconstrained, blind ASR which degrades on short 3s KYC clips. By exploiting the active challenge prompt, we replace blind guessing with millisecond-exact trellis forced alignment. Furthermore, our **DeepFakeTIMIT findings** revealed that while expression-preserving swaps (Celeb-DF) preserve viseme sync, reenactment models consistently fail the physical bilabial seal constraint ($42.47\%$ failure rate across 996 opportunities). Non-negative late fusion prevents weight sign inversion collapse, maintaining an AUC floor $\ge 0.6175$ across all identities.

---

## 5. The Error Rate / "How Often Is It Wrong?" Question
> **Panelist**: *"What is your real error rate, and how often does this fail?"*

**Answer (Calm, direct, no hedging)**:
> "On our held-out, identity-disjoint test split ($N=103$ clips across 9 unseen subjects), the model achieves an **Equal Error Rate (EER) of 36.02%** and a **5-fold cross-validation mean ROC-AUC of 0.6877 (± 0.0449)**.
> 
> Our demo clips illustrate the specific physical mechanisms—complete closure in genuine speech versus viseme lag and failed seals in reenactments. They are illustrative examples of physical failure modes, not a claim of 100% field accuracy. In a production biometric KYC pipeline, this model serves as an injection-defense layer: physical desync ($S_{\text{desync}} > 0.0242$) provides an immediate physical override against neural false negatives, but automated decisions require continuous probability scoring alongside secondary face verification."

---

## 6. Future Work: Live DeepFaceLive Validation (Not Completed)

**If asked about live attack testing — be direct:**

> "Live-generation validation using a self-trained DeepFaceLive model through an OBS Virtual Camera was **not completed** and is explicitly future scope. The planned protocol involved 3 challenge phrases, paired genuine vs. swap recordings through the identical pipeline, and a target 600 ms measured audio offset. All demo evidence is based on offline benchmarks: DeepFakeTIMIT reenactment (primary evaluation), VidTIMIT (genuine baseline). The threat model describes virtual-camera injection as the motivating threat; offline reenactment benchmarks serve as a proxy for the adversarial case."
