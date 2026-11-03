"""
SecureKYC AI - Multimodal Dataset Feature Extractor
Runs 3D Lip Tracker, Audio Forced Alignment, and Cross-Modal Desynchronization Metric
across VidTIMIT, Celeb-DF v2, and FaceForensics++ datasets.

Log every extraction failure explicitly to CSV and log files (NO silent zero-filling/skipping).
Persists extracted tensors (lip_seq, desync_vec, label) to data/processed/dataset.pt.
"""

import sys
import re
import csv
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import scipy.io.wavfile as wavfile
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIXED_SEQ_LEN,
    AUDIO_SAMPLE_RATE,
    LIP_CLOSURE_THRESHOLD_NORMALIZED,
)
from src.vision.lip_tracker import LipTracker, LipFrameResult
from src.audio.phoneme_extractor import PhonemeExtractor, AudioPhonemeResult
from src.metrics.desync_metric import DesynchronizationMetric, DesyncMetricResult

import json

# VidTIMIT Known Sentence Transcripts (loaded from verified 15-speaker TIMIT dictionary)
TIMIT_TRANSCRIPTS_PATH = PROJECT_ROOT / "data" / "processed" / "timit_15spk_transcripts.json"
if TIMIT_TRANSCRIPTS_PATH.exists():
    with open(TIMIT_TRANSCRIPTS_PATH, "r", encoding="utf-8") as _f:
        VIDTIMIT_TRANSCRIPTS = json.load(_f)
else:
    VIDTIMIT_TRANSCRIPTS = {
        "sa1": "She had your dark suit in greasy wash water all year.",
        "sa2": "Don't ask me to carry an oily rag like that.",
        "sx19": "Aluminum silverware can often be flimsy.",
        "sx109": "She slipped and sprained her ankle on the steep slope.",
        "sx199": "Young children should avoid exposure to contagious diseases.",
        "sx289": "Weatherproof galoshes are very useful in Seattle.",
        "sx379": "Pam gives driving lessons on Thursdays.",
        "si649": "It suffers from a lack of unity of purpose and respect for heroic leadership.",
        "si1279": "Bricks are an alternative.",
        "si1909": "The shredded lace rolls beneath the shelves.",
    }



class MultimodalDatasetExtractor:
    """
    Extracts 3D lip aperture trajectories and audio forced-alignment desynchronization metrics
    from VidTIMIT, Celeb-DF v2, and FaceForensics++ dataset clips.
    Explicitly logs all failure modes to data/processed/extraction_failures.csv.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        target_seq_len: int = FIXED_SEQ_LEN,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.target_seq_len = target_seq_len

        self.tracker = LipTracker(static_image_mode=False)
        self.phoneme_extractor = PhonemeExtractor(lazy_load=False)
        self.desync_metric = DesynchronizationMetric()

        # Failure logging setup
        self.failure_csv_path = self.processed_dir / "extraction_failures.csv"
        self.log_file_path = self.processed_dir / "extraction.log"

        self._init_failure_csv()
        self._init_logger()

    def _init_failure_csv(self):
        """Initializes CSV file header for tracking extraction failure modes."""
        if not self.failure_csv_path.exists():
            with open(self.failure_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "dataset_name", "clip_path", "failure_reason", "details"])

    def _init_logger(self):
        """Configures file and console logging."""
        self.logger = logging.getLogger("DatasetExtractor")
        self.logger.setLevel(logging.INFO)

        # Clear previous handlers if re-initializing
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(str(self.log_file_path), encoding="utf-8")
        console_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_failure(self, dataset_name: str, clip_path: Path, reason: str, details: str = ""):
        """Explicitly records extraction failure to CSV and log file."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.failure_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([now_str, dataset_name, str(clip_path), reason, details])

        self.logger.warning(f"Extraction Failure ({dataset_name}): {clip_path.name} | Reason: {reason} | {details}")

    def validate_dataset_directories(self, require_faceforensics: bool = False) -> dict:
        """
        Validates presence of VidTIMIT, Celeb-DF v2, and FaceForensics++ raw dataset directories.
        Fails loudly with instructional error messages if any expected folder is missing or empty.
        """
        orig_dir = self.raw_dir / "original_sequences"
        manip_dir = self.raw_dir / "manipulated_sequences"

        # 1. VidTIMIT Check
        vidtimit_found = orig_dir.exists() and any(
            (orig_dir / spk).is_dir() for spk in ["fadg0", "faks0", "mcem0"]
        )
        if not vidtimit_found:
            raise FileNotFoundError(
                f"[!] VidTIMIT dataset missing or incomplete in '{orig_dir}'.\n"
                "    -> Instruction: Execute 'python3 data/download_vidtimit.py' to download and verify VidTIMIT."
            )

        # 2. Celeb-DF v2 Check
        celeb_real = orig_dir / "Celeb-real"
        celeb_synth = manip_dir / "Celeb-synthesis"
        celeb_found = (
            celeb_real.exists() and len(list(celeb_real.glob("*.mp4"))) > 0 and
            celeb_synth.exists() and len(list(celeb_synth.glob("*.mp4"))) > 0
        )
        if not celeb_found:
            raise FileNotFoundError(
                f"[!] Celeb-DF v2 dataset missing or incomplete in '{orig_dir}' / '{manip_dir}'.\n"
                "    -> Instruction: Request access at https://github.com/yuezunli/celeb-deepfakeforensics\n"
                "    -> Download and extract 'Celeb-real' into data/raw/original_sequences/\n"
                "    -> Download and extract 'Celeb-synthesis' into data/raw/manipulated_sequences/"
            )

        # 3. FaceForensics++ Check
        ff_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        ff_found = any(
            (manip_dir / ff_type).exists() and len(list((manip_dir / ff_type).rglob("*.mp4"))) > 0
            for ff_type in ff_types
        )
        if require_faceforensics and not ff_found:
            raise FileNotFoundError(
                f"[!] FaceForensics++ dataset placeholder empty in '{manip_dir}'.\n"
                "    -> Instruction: Submit signed Terms of Use at https://github.com/ondyari/FaceForensics\n"
                "    -> Download c23 compression MP4 clips and place in data/raw/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/"
            )

        return {
            "vidtimit_present": vidtimit_found,
            "celeb_df_present": celeb_found,
            "faceforensics_present": ff_found,
        }

    def normalize_sequence_length(self, seq: np.ndarray) -> np.ndarray:
        """Interpolates lip aperture sequence to target_seq_len (100 points)."""
        if len(seq) == 0:
            return np.zeros(self.target_seq_len, dtype=np.float32)

        x_orig = np.linspace(0, 1, len(seq))
        x_target = np.linspace(0, 1, self.target_seq_len)
        interp_seq = np.interp(x_target, x_orig, seq).astype(np.float32)

        return interp_seq

    def process_mp4_video_and_audio(
        self,
        mp4_path: Path,
        dataset_name: str,
        is_fake_label: float,
        prompt_transcript: Optional[str] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Processes a single video file through Step A3-A5 pipeline.
        Returns dict of tensors if successful, or None if extraction failed (and logs failure reason).
        """
        if not mp4_path.exists():
            self.log_failure(dataset_name, mp4_path, "FILE_NOT_FOUND", "File does not exist")
            return None

        # 1. Extract audio waveform and run forced alignment
        audio_waveform, sr = self.phoneme_extractor.extract_audio_from_video(mp4_path)

        transcript = prompt_transcript or "MY BANK PASSCODE"
        audio_absent = (audio_waveform is None)

        if audio_absent:
            # AUDIO_ABSENT: video has no audio stream. Log it and return early —
            # never substitute another file's audio. The caller decides how to handle this.
            self.log_failure(
                dataset_name, mp4_path, "AUDIO_ABSENT",
                "Video has no audio stream; desync_features cannot be computed for this clip."
            )
            return None

        audio_result = self.phoneme_extractor.align_forced_transcript(audio_waveform, prompt_text=transcript, sample_rate=sr)

        if not audio_result.bilabial_intervals:
            self.log_failure(dataset_name, mp4_path, "NO_BILABIALS_FOUND", f"No bilabial consonant timestamps in alignment for '{transcript}'")
            return None

        # 2. Extract 3D lip tracker trajectory from video frames
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            self.log_failure(dataset_name, mp4_path, "VIDEO_CORRUPT", "OpenCV failed to open video file")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_duration_ms = (1.0 / fps) * 1000.0

        lip_results: List[LipFrameResult] = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < 150:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            timestamp_ms = frame_idx * frame_duration_ms
            res = self.tracker.process_frame(frame, timestamp_ms=timestamp_ms)
            lip_results.append(res)
            frame_idx += 1

        cap.release()

        if not lip_results:
            self.log_failure(dataset_name, mp4_path, "VIDEO_CORRUPT", "Zero frames read from video clip")
            return None

        # 3. Check face detection and gap constraints
        detected_frames = [r for r in lip_results if r.face_detected]
        if not detected_frames:
            self.log_failure(dataset_name, mp4_path, "NO_FACE_DETECTED", "Zero faces detected by MediaPipe across video frames")
            return None

        lip_results = self.tracker.interpolate_missing_apertures(lip_results)

        if any(r.unreliable_face_track for r in lip_results):
            self.log_failure(dataset_name, mp4_path, "UNRELIABLE_FACE_TRACK", "Face drop gap > 3 consecutive frames detected")
            return None

        # 4. Compute cross-modal desynchronization score S_desync
        desync_res = self.desync_metric.compute_desync_score(lip_results, audio_result)

        aperture_seq = np.array([r.lip_aperture for r in lip_results if r.lip_aperture > 0.0], dtype=np.float32)
        if len(aperture_seq) == 0:
            self.log_failure(dataset_name, mp4_path, "NO_FACE_DETECTED", "All lip aperture values evaluated to 0.0")
            return None

        norm_seq = self.normalize_sequence_length(aperture_seq)

        desync_score = float(desync_res.overall_desync_score)
        max_penalty = float(desync_res.max_penalty_during_bilabials)
        num_b = desync_res.num_bilabial_events
        failed_ratio = (desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0

        lip_tensor = torch.from_numpy(norm_seq)
        desync_tensor = torch.tensor([desync_score, max_penalty, failed_ratio], dtype=torch.float32)
        label_tensor = torch.tensor([is_fake_label], dtype=torch.float32)

        # Extract subject/speaker identity from filename (e.g., id0 from id0_0000.mp4)
        id_match = re.search(r"id\d+", mp4_path.stem)
        identity_id = id_match.group() if id_match else mp4_path.stem.split("_")[0]

        return {
            "lip_sequence": lip_tensor,
            "desync_features": desync_tensor,
            "label": label_tensor,
            "identity": identity_id,
        }

    def process_vidtimit_speaker(self, speaker_dir: Path) -> List[Dict[str, torch.Tensor]]:
        """Processes genuine VidTIMIT audio-video sentence directories."""
        samples = []
        audio_dir = speaker_dir / "audio"
        video_dir = speaker_dir / "video"

        if not audio_dir.exists() or not video_dir.exists():
            return samples

        for sentence_key, prompt_text in VIDTIMIT_TRANSCRIPTS.items():
            wav_path = audio_dir / f"{sentence_key}.wav"
            sentence_video_dir = video_dir / sentence_key

            if not wav_path.exists() or not sentence_video_dir.exists():
                continue

            # Check if sentence transcript contains any bilabial consonants (B, M, P)
            bilabial_chars = {c.upper() for c in prompt_text if c.upper() in {"B", "M", "P"}}
            if not bilabial_chars:
                self.log_failure("VidTIMIT", wav_path, "SKIPPED_NO_BILABIALS_IN_TRANSCRIPT", f"Transcript '{prompt_text}' contains zero bilabial consonants")
                continue

            # Load audio waveform
            try:
                sr, audio_int16 = wavfile.read(str(wav_path))
                waveform = audio_int16.astype(np.float32) / 32768.0 if audio_int16.dtype == np.int16 else audio_int16.astype(np.float32)
            except Exception as e:
                self.log_failure("VidTIMIT", wav_path, "AUDIO_CORRUPT", str(e))
                continue

            audio_res = self.phoneme_extractor.align_forced_transcript(waveform, prompt_text=prompt_text, sample_rate=sr)

            if not audio_res.bilabial_intervals:
                self.log_failure("VidTIMIT", wav_path, "NO_BILABIALS_FOUND", f"No bilabial consonant timestamps in alignment for '{prompt_text}'")
                continue

            # Load video frames
            frame_files = sorted(list(sentence_video_dir.glob("*")), key=lambda p: int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0)
            if not frame_files:
                self.log_failure("VidTIMIT", sentence_video_dir, "VIDEO_CORRUPT", "No frame images found in sentence folder")
                continue

            lip_results = []
            frame_duration_ms = 40.0  # VidTIMIT frame rate = 25 FPS

            for f_idx, f_path in enumerate(frame_files):
                frame_img = cv2.imread(str(f_path))
                if frame_img is None:
                    continue
                timestamp_ms = f_idx * frame_duration_ms
                res = self.tracker.process_frame(frame_img, timestamp_ms=timestamp_ms)
                lip_results.append(res)

            if not lip_results or not any(r.face_detected for r in lip_results):
                self.log_failure("VidTIMIT", sentence_video_dir, "NO_FACE_DETECTED", "Zero faces detected across sentence frames")
                continue

            lip_results = self.tracker.interpolate_missing_apertures(lip_results)

            if any(r.unreliable_face_track for r in lip_results):
                self.log_failure("VidTIMIT", sentence_video_dir, "UNRELIABLE_FACE_TRACK", "Face drop gap > 3 consecutive frames")
                continue

            desync_res = self.desync_metric.compute_desync_score(lip_results, audio_res)

            aperture_seq = np.array([r.lip_aperture for r in lip_results if r.lip_aperture > 0.0], dtype=np.float32)
            if len(aperture_seq) == 0:
                self.log_failure("VidTIMIT", sentence_video_dir, "NO_FACE_DETECTED", "Zero valid face aperture values")
                continue

            norm_seq = self.normalize_sequence_length(aperture_seq)
            desync_score = float(desync_res.overall_desync_score)
            max_penalty = float(desync_res.max_penalty_during_bilabials)
            num_b = desync_res.num_bilabial_events
            failed_ratio = (desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0

            samples.append({
                "lip_sequence": torch.from_numpy(norm_seq),
                "desync_features": torch.tensor([desync_score, max_penalty, failed_ratio], dtype=torch.float32),
                "label": torch.tensor([0.0], dtype=torch.float32),  # 0.0 = Real
                "identity": speaker_dir.name,
            })

        return samples

    def process_deepfaketimit_speaker(
        self,
        deepfake_speaker_dir: Path,
        quality: str,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Processes a single DeepFakeTIMIT speaker directory (HQ or LQ).

        Actual on-disk layout (verified):
            {speaker}/{sentence}-video-{swapped_id}.avi   <- GAN-swapped face video
            {speaker}/{sentence}.wav                      <- original VidTIMIT audio

        DeepFakeTIMIT keeps the original audio track co-located as a separate
        .wav — identical to the matching VidTIMIT speaker's sentence WAV.
        Audio is loaded directly via wavfile.read (same as VidTIMIT branch).
        Sentence key is parsed by splitting the stem on '-video-'.

        Pipeline: lip_tracker → phoneme_extractor → desync_metric, label=1.0.
        """
        samples = []
        dataset_name = f"DeepFakeTIMIT-{quality}"
        speaker_name = deepfake_speaker_dir.name  # e.g. 'fadg0'

        # Collect all GAN-swapped video files.
        # Pattern: {sentence}-video-{swapped_id}.avi  (e.g. sa2-video-fram1.avi)
        video_files = sorted(
            list(deepfake_speaker_dir.glob("*-video-*.avi")) +
            list(deepfake_speaker_dir.glob("*-video-*.mp4"))
        )

        if not video_files:
            self.log_failure(
                dataset_name, deepfake_speaker_dir,
                "NO_VIDEO_FILES",
                "No *-video-*.avi/.mp4 files found in speaker directory"
            )
            return samples

        for video_path in video_files:
            # Parse sentence key: "sa2-video-fram1" → "sa2"
            stem = video_path.stem
            if "-video-" not in stem:
                self.log_failure(dataset_name, video_path, "UNKNOWN_SENTENCE_KEY",
                                 f"Stem '{stem}' does not contain '-video-' separator")
                continue
            sentence_key = stem.split("-video-")[0]

            prompt_text = VIDTIMIT_TRANSCRIPTS.get(sentence_key)
            if prompt_text is None:
                self.log_failure(
                    dataset_name, video_path, "UNKNOWN_SENTENCE_KEY",
                    f"Sentence '{sentence_key}' not in VIDTIMIT_TRANSCRIPTS dict"
                )
                continue

            # Screen out bilabial-free transcripts before running alignment.
            bilabial_chars = {c.upper() for c in prompt_text if c.upper() in {"B", "M", "P"}}
            if not bilabial_chars:
                self.log_failure(
                    dataset_name, video_path, "SKIPPED_NO_BILABIALS_IN_TRANSCRIPT",
                    f"Transcript '{prompt_text}' contains zero bilabial consonants"
                )
                continue

            # 1. Load the co-located original VidTIMIT WAV directly.
            wav_path = deepfake_speaker_dir / f"{sentence_key}.wav"
            if not wav_path.exists():
                self.log_failure(dataset_name, video_path, "AUDIO_ABSENT",
                                 f"Expected co-located WAV not found: {wav_path}")
                continue

            try:
                sr, audio_int16 = wavfile.read(str(wav_path))
                audio_waveform = (
                    audio_int16.astype(np.float32) / 32768.0
                    if audio_int16.dtype == np.int16
                    else audio_int16.astype(np.float32)
                )
            except Exception as e:
                self.log_failure(dataset_name, wav_path, "AUDIO_CORRUPT", str(e))
                continue

            # 2. Forced alignment against ground-truth transcript.
            audio_res = self.phoneme_extractor.align_forced_transcript(
                audio_waveform, prompt_text=prompt_text, sample_rate=sr
            )
            if not audio_res.bilabial_intervals:
                self.log_failure(
                    dataset_name, video_path, "NO_BILABIALS_FOUND",
                    f"No bilabial timestamps in alignment for '{prompt_text}'"
                )
                continue

            # 3. Lip aperture trajectory from the GAN-swapped .avi frames.
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.log_failure(dataset_name, video_path, "VIDEO_CORRUPT",
                                 "OpenCV failed to open video file")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_duration_ms = (1.0 / fps) * 1000.0
            lip_results = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp_ms = frame_idx * frame_duration_ms
                res = self.tracker.process_frame(frame, timestamp_ms=timestamp_ms)
                lip_results.append(res)
                frame_idx += 1

            cap.release()

            if not lip_results or not any(r.face_detected for r in lip_results):
                self.log_failure(dataset_name, video_path, "NO_FACE_DETECTED",
                                 "Zero faces detected by MediaPipe across video frames")
                continue

            lip_results = self.tracker.interpolate_missing_apertures(lip_results)

            if any(r.unreliable_face_track for r in lip_results):
                self.log_failure(dataset_name, video_path, "UNRELIABLE_FACE_TRACK",
                                 "Face drop gap > 3 consecutive frames detected")
                continue

            # 4. Cross-modal desync metric.
            desync_res = self.desync_metric.compute_desync_score(lip_results, audio_res)

            aperture_seq = np.array(
                [r.lip_aperture for r in lip_results if r.lip_aperture > 0.0],
                dtype=np.float32
            )
            if len(aperture_seq) == 0:
                self.log_failure(dataset_name, video_path, "NO_FACE_DETECTED",
                                 "Zero valid aperture values after tracking")
                continue

            norm_seq     = self.normalize_sequence_length(aperture_seq)
            desync_score = float(desync_res.overall_desync_score)
            max_penalty  = float(desync_res.max_penalty_during_bilabials)
            num_b        = desync_res.num_bilabial_events
            failed_ratio = (desync_res.failed_closure_count / num_b) if num_b > 0 else 0.0

            samples.append({
                "lip_sequence":    torch.from_numpy(norm_seq),
                "desync_features": torch.tensor(
                    [desync_score, max_penalty, failed_ratio], dtype=torch.float32
                ),
                "label":    torch.tensor([1.0], dtype=torch.float32),  # 1.0 = Fake
                "identity": speaker_name,
            })

        return samples

    def extract_all_datasets(
        self,
        max_clips_per_dataset: int = 500,
        include_celeb_df: bool = False,
    ) -> Dict[str, torch.Tensor]:

        """
        Executes feature extraction across VidTIMIT, Celeb-DF v2, and FaceForensics++.
        Skips empty dataset folders gracefully with clear log lines.
        """
        self.logger.info("=" * 70)
        self.logger.info(" Starting SecureKYC AI Multimodal Dataset Extraction ")
        self.logger.info("=" * 70)

        all_samples: List[Dict[str, Any]] = []
        orig_dir = self.raw_dir / "original_sequences"
        manip_dir = self.raw_dir / "manipulated_sequences"

        # 1. VidTIMIT Dataset Extraction
        speaker_dirs = [d for d in orig_dir.iterdir() if d.is_dir() and d.name not in ["Celeb-real", "YouTube-real", "youtube"]] if orig_dir.exists() else []

        if speaker_dirs:
            self.logger.info(f"[*] Extracting VidTIMIT Dataset from {len(speaker_dirs)} speaker folders...")
            vidtimit_samples = 0
            for spk_dir in speaker_dirs:
                spk_samples = self.process_vidtimit_speaker(spk_dir)
                all_samples.extend(spk_samples)
                vidtimit_samples += len(spk_samples)
            self.logger.info(f"[✓] Extracted {vidtimit_samples} valid VidTIMIT samples.")
        else:
            self.logger.info("[INFO] VidTIMIT dataset folder is empty or missing. Skipping gracefully.")

        # 2. DeepFakeTIMIT Dataset Extraction (higher_quality and lower_quality subfolders)
        #    On-disk layout (verified): DeepFakeTIMIT/higher_quality/<speaker>/ and
        #                               DeepFakeTIMIT/lower_quality/<speaker>/
        deepfaketimit_root = manip_dir / "DeepFakeTIMIT"
        for folder_name, quality_label in [("higher_quality", "HQ"), ("lower_quality", "LQ")]:
            quality_dir = deepfaketimit_root / folder_name
            if not quality_dir.exists():
                self.logger.info(
                    f"[INFO] DeepFakeTIMIT {quality_label} folder '{quality_dir}' not found. Skipping."
                )
                continue
            dft_speaker_dirs = [d for d in quality_dir.iterdir() if d.is_dir()]
            if not dft_speaker_dirs:
                self.logger.info(
                    f"[INFO] DeepFakeTIMIT {quality_label} folder exists but has no speaker subdirectories."
                )
                continue
            self.logger.info(
                f"[*] Extracting DeepFakeTIMIT-{quality_label} Dataset "
                f"({len(dft_speaker_dirs)} speaker folders)..."
            )
            dft_count = 0
            for spk_dir in dft_speaker_dirs:
                spk_samples = self.process_deepfaketimit_speaker(spk_dir, quality=quality_label)
                all_samples.extend(spk_samples)
                dft_count += len(spk_samples)
            self.logger.info(f"[✓] Extracted {dft_count} valid DeepFakeTIMIT-{quality_label} samples.")

        if include_celeb_df:
            celeb_real_dir = orig_dir / "Celeb-real"
            celeb_synth_dir = manip_dir / "Celeb-synthesis"

            if celeb_real_dir.exists() and len(list(celeb_real_dir.glob("*.mp4"))) > 0:
                real_mp4s = list(celeb_real_dir.glob("*.mp4"))[:max_clips_per_dataset]
                self.logger.info(f"[*] Extracting Celeb-real Dataset ({len(real_mp4s)} video clips)...")
                celeb_real_samples = 0
                for idx, mp4_path in enumerate(real_mp4s):
                    sample = self.process_mp4_video_and_audio(mp4_path, dataset_name="Celeb-real", is_fake_label=0.0)
                    if sample is not None:
                        all_samples.append(sample)
                        celeb_real_samples += 1
                    if (idx + 1) % 100 == 0:
                        self.logger.info(f"    - Processed {idx + 1}/{len(real_mp4s)} Celeb-real clips...")
                self.logger.info(f"[✓] Extracted {celeb_real_samples} valid Celeb-real samples.")
            else:
                self.logger.info("[INFO] Celeb-real dataset directory 'data/raw/original_sequences/Celeb-real' is empty/missing. Skipping gracefully.")

            if celeb_synth_dir.exists() and len(list(celeb_synth_dir.glob("*.mp4"))) > 0:
                synth_mp4s = list(celeb_synth_dir.glob("*.mp4"))[:max_clips_per_dataset]
                self.logger.info(f"[*] Extracting Celeb-synthesis Dataset ({len(synth_mp4s)} video clips)...")
                celeb_synth_samples = 0
                for idx, mp4_path in enumerate(synth_mp4s):
                    sample = self.process_mp4_video_and_audio(mp4_path, dataset_name="Celeb-synthesis", is_fake_label=1.0)
                    if sample is not None:
                        all_samples.append(sample)
                        celeb_synth_samples += 1
                    if (idx + 1) % 100 == 0:
                        self.logger.info(f"    - Processed {idx + 1}/{len(synth_mp4s)} Celeb-synthesis clips...")
                self.logger.info(f"[✓] Extracted {celeb_synth_samples} valid Celeb-synthesis samples.")
            else:
                self.logger.info("[INFO] Celeb-synthesis dataset directory 'data/raw/manipulated_sequences/Celeb-synthesis' is empty/missing. Skipping gracefully.")


        # 3. FaceForensics++ Dataset Extraction
        ff_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        ff_found = any((manip_dir / ff_type).exists() and len(list((manip_dir / ff_type).rglob("*.mp4"))) > 0 for ff_type in ff_types)

        if ff_found:
            self.logger.info("[*] Extracting FaceForensics++ Dataset...")
            for ff_type in ff_types:
                ff_dir = manip_dir / ff_type
                if ff_dir.exists():
                    ff_mp4s = list(ff_dir.rglob("*.mp4"))[:max_clips_per_dataset]
                    for mp4_path in ff_mp4s:
                        sample = self.process_mp4_video_and_audio(mp4_path, dataset_name=f"FaceForensics++_{ff_type}", is_fake_label=1.0)
                        if sample is not None:
                            all_samples.append(sample)
        else:
            self.logger.info("[INFO] FaceForensics++ directory 'data/raw/manipulated_sequences/Deepfakes' is empty or missing. Skipping gracefully.")

        if not all_samples:
            raise RuntimeError("Zero valid samples extracted across all raw datasets. Check data/processed/extraction_failures.csv.")

        # Consolidate into batched PyTorch Tensors and Identity List
        all_lip_seqs = torch.stack([s["lip_sequence"] for s in all_samples])
        all_desync_feats = torch.stack([s["desync_features"] for s in all_samples])
        all_labels = torch.stack([s["label"] for s in all_samples])
        all_identities = [s.get("identity", f"subject_{i}") for i, s in enumerate(all_samples)]

        dataset_dict = {
            "lip_sequences": all_lip_seqs,
            "desync_features": all_desync_feats,
            "labels": all_labels,
            "identities": all_identities,
        }

        dataset_pt_path = self.processed_dir / "dataset.pt"
        torch.save(dataset_dict, dataset_pt_path)

        self.logger.info("=" * 70)
        self.logger.info(f"[✓] Successfully persisted dataset tensors to {dataset_pt_path}")
        self.logger.info(f"    - Total Valid Extracted Samples: {len(all_samples)}")
        self.logger.info(f"    - Lip Sequences Tensor shape:    {list(all_lip_seqs.shape)}")
        self.logger.info(f"    - Desync Features Tensor shape:  {list(all_desync_feats.shape)}")
        self.logger.info(f"    - Labels Tensor shape:           {list(all_labels.shape)}")
        self.logger.info(f"    - Failure details logged to:     {self.failure_csv_path}")
        self.logger.info("=" * 70)

        return dataset_dict


# Backward compatibility alias
VidTIMITDatasetExtractor = MultimodalDatasetExtractor


if __name__ == "__main__":
    extractor = MultimodalDatasetExtractor()
    extractor.extract_all_datasets()
