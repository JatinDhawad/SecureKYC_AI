"""
SecureKYC AI - Audio Phoneme Extractor Module
Uses torchaudio.pipelines.MMS_FA and torchaudio.functional.forced_align to align
known challenge prompt transcripts against audio emissions and extract exact
millisecond timestamps T_bilabial for bilabial consonants (/b/, /m/, /p/).
Also includes an optional secondary phoneme recognition path using HuggingFace Wav2Vec2.
"""

import re
import sys
import time
import ssl
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import torchaudio
    import torchaudio.pipelines as P
    import torchaudio.functional as F
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AUDIO_SAMPLE_RATE,
    WAV2VEC2_MODEL_NAME,
    BILABIAL_PHONEMES,
)


@dataclass
class PhonemeInterval:
    """Represents a timestamped character/phoneme interval."""
    phoneme: str
    start_time_ms: float
    end_time_ms: float
    is_bilabial: bool
    confidence: float = 1.0


@dataclass
class AudioPhonemeResult:
    """Holds extracted character/phoneme intervals and isolated bilabial timestamps."""
    audio_duration_ms: float
    phoneme_sequence: List[PhonemeInterval] = field(default_factory=list)
    bilabial_intervals: List[PhonemeInterval] = field(default_factory=list)
    processing_latency_ms: float = 0.0
    alignment_score: float = 1.0


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Computes standard Word Error Rate (WER) between reference and hypothesis text.
    WER = (Substitutions + Deletions + Insertions) / len(Reference_Words)
    """
    ref_words = re.sub(r"[^\w\s]", "", reference.lower()).split()
    hyp_words = re.sub(r"[^\w\s]", "", hypothesis.lower()).split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0

    m, n = len(ref_words), len(hyp_words)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[m, n] / len(ref_words))


def compute_cer(s1: str, s2: str) -> float:
    """
    Computes normalized Character Error Rate (CER / normalized Levenshtein distance) in [0.0, 1.0].
    """
    s1_clean = re.sub(r"[^\w\s]", "", s1.lower()).strip()
    s2_clean = re.sub(r"[^\w\s]", "", s2.lower()).strip()
    if not s1_clean and not s2_clean:
        return 0.0
    max_len = max(len(s1_clean), len(s2_clean))
    if max_len == 0:
        return 0.0

    m, n = len(s1_clean), len(s2_clean)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1_clean[i - 1] == s2_clean[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[m, n] / max_len)


class PhonemeExtractor:
    """
    Wav2Vec2 Forced Alignment Phoneme Extractor.
    Aligns known challenge prompt text against raw audio waveforms to produce
    millisecond-level bilabial consonant timestamps (T_bilabial).
    """

    def __init__(
        self,
        bilabial_set: Set[str] = BILABIAL_PHONEMES,
        lazy_load: bool = True,
    ):
        self.bilabial_set = {p.upper() for p in bilabial_set}
        self.bundle = None
        self.model = None
        self.dictionary = None
        self.id_to_token = None
        self.has_torchaudio = HAS_TORCHAUDIO
        self._secondary_model = None
        self._secondary_processor = None

        if not lazy_load and HAS_TORCHAUDIO:
            self._load_forced_alignment_model()

    def _load_forced_alignment_model(self):
        """Loads torchaudio MMS_FA forced alignment model bundle."""
        if not HAS_TORCHAUDIO:
            raise RuntimeError("Torchaudio is not installed or failed to import.")
        if self.model is None:
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
                self.bundle = P.MMS_FA
                self.model = self.bundle.get_model()
                self.model.eval()
                self.dictionary = self.bundle.get_dict()
                self.id_to_token = {v: k for k, v in self.dictionary.items()}
            except Exception as err:
                logger.exception("Failed to load torchaudio MMS_FA forced alignment model: %s", err)
                self.model = None
                self.dictionary = None
                self.id_to_token = None
                raise RuntimeError(f"Failed to load torchaudio MMS_FA model bundle: {err}") from err

    def extract_audio_from_video(self, video_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """
        Extracts 1D float32 normalized waveform [-1.0, 1.0] from video file using PyAV.

        Returns:
            (waveform, sample_rate) on success.
            (None, 0)              as the AUDIO_ABSENT sentinel when the video has no audio
                                   stream — callers MUST check for None and MUST NOT
                                   substitute any other file's audio as a fallback.
        Raises:
            RuntimeError on file decoding/processing failure with full exception stack trace.
        """
        audio_samples = []
        try:
            import av
            container = av.open(str(video_path))
            if container.streams.audio:
                stream = container.streams.audio[0]
                resampler = av.AudioResampler(format="flt", layout="mono", rate=AUDIO_SAMPLE_RATE)
                for frame in container.decode(stream):
                    resampled = resampler.resample(frame)
                    for rf in resampled:
                        audio_samples.append(rf.to_ndarray())
            else:
                container.close()
                return None, 0
            container.close()

            if audio_samples:
                waveform = np.concatenate(audio_samples, axis=1).flatten().astype(np.float32)
                return waveform, AUDIO_SAMPLE_RATE
            else:
                return None, 0
        except Exception as e:
            logger.exception("Audio extraction failed from video %s: %s", video_path, e)
            raise RuntimeError(f"Audio extraction failed for {video_path}: {e}") from e

    def align_forced_transcript(
        self,
        audio_waveform: np.ndarray,
        prompt_text: str,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> AudioPhonemeResult:
        """
        Executes Forced Alignment of known prompt transcript against audio waveform.
        Extracts millisecond T_bilabial timestamps for /b/, /m/, /p/.
        Requires prompt_text to be explicitly provided — fails loudly if missing or empty.
        """
        if not prompt_text or not prompt_text.strip():
            raise ValueError(
                "Explicit challenge prompt transcript must be provided to align_forced_transcript(); "
                "received empty or missing prompt_text."
            )

        start_time = time.perf_counter()

        if audio_waveform is None or len(audio_waveform) == 0:
            latency = (time.perf_counter() - start_time) * 1000.0
            return AudioPhonemeResult(audio_duration_ms=0.0, processing_latency_ms=latency)

        duration_ms = (len(audio_waveform) / sample_rate) * 1000.0

        if self.model is None and HAS_TORCHAUDIO:
            self._load_forced_alignment_model()

        if self.model is None or self.dictionary is None:
            raise RuntimeError(
                "Torchaudio MMS_FA model or dictionary failed to load. Cannot execute forced alignment."
            )

        # 1. Clean and normalize prompt text
        clean_text = "".join([c.lower() for c in prompt_text if c.isalpha() or c.isspace()])
        words = clean_text.split()
        normalized_text = "*".join(words)  # Word boundary token in MMS_FA dictionary ('*')

        # 2. Tokenize prompt characters into dictionary IDs
        target_tokens = []
        char_list = []
        for c in normalized_text:
            if c in self.dictionary:
                target_tokens.append(self.dictionary[c])
                char_list.append(c)

        if not target_tokens:
            latency = (time.perf_counter() - start_time) * 1000.0
            return AudioPhonemeResult(audio_duration_ms=duration_ms, processing_latency_ms=latency)

        # 3. Resample audio if needed and compute emissions
        waveform_tensor = torch.from_numpy(audio_waveform).unsqueeze(0)  # (1, N)
        if sample_rate != self.bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.bundle.sample_rate)
            waveform_tensor = resampler(waveform_tensor)

        with torch.inference_mode():
            emissions, _ = self.model(waveform_tensor)
            emissions = torch.log_softmax(emissions, dim=-1)  # (1, T, C)

        # 4. Forced Alignment
        targets_tensor = torch.tensor([target_tokens], dtype=torch.int32)
        input_lengths = torch.tensor([emissions.shape[1]], dtype=torch.int32)
        target_lengths = torch.tensor([len(target_tokens)], dtype=torch.int32)

        blank_id = 0
        alignment, scores = F.forced_align(
            emissions,
            targets_tensor,
            input_lengths,
            target_lengths,
            blank=blank_id,
        )

        # 5. Merge token frame alignments into time spans
        token_spans = F.merge_tokens(alignment[0], scores[0])
        num_frames = emissions.shape[1]
        frame_duration_ms = duration_ms / max(num_frames, 1)

        intervals: List[PhonemeInterval] = []
        bilabials: List[PhonemeInterval] = []
        total_score = 0.0

        for span in token_spans:
            if self.id_to_token is not None and span.token in self.id_to_token:
                char = self.id_to_token[span.token].upper()
            else:
                continue

            if char == "*":
                continue  # Skip word boundary star tokens
            start_ms = span.start * frame_duration_ms
            end_ms = span.end * frame_duration_ms
            confidence = float(np.exp(span.score)) if isinstance(span.score, (float, int)) else float(torch.exp(span.score).item())
            is_b = char in self.bilabial_set

            interval = PhonemeInterval(
                phoneme=char,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                is_bilabial=is_b,
                confidence=confidence,
            )
            intervals.append(interval)
            if is_b:
                bilabials.append(interval)
            total_score += confidence

        avg_score = (total_score / len(intervals)) if intervals else 0.0
        latency = (time.perf_counter() - start_time) * 1000.0

        return AudioPhonemeResult(
            audio_duration_ms=duration_ms,
            phoneme_sequence=intervals,
            bilabial_intervals=bilabials,
            processing_latency_ms=latency,
            alignment_score=avg_score,
        )

    def extract_phonemes(
        self,
        audio_waveform: np.ndarray,
        prompt_text: str,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> AudioPhonemeResult:
        """
        Executes forced alignment of audio waveform against the issued challenge prompt transcript.
        Requires prompt_text to be explicitly provided — fails loudly if missing or empty.
        """
        if not prompt_text or not prompt_text.strip():
            raise ValueError(
                "Explicit challenge prompt transcript must be provided to extract_phonemes(); "
                "hardcoded fallback transcripts are strictly forbidden."
            )
        return self.align_forced_transcript(
            audio_waveform, prompt_text=prompt_text, sample_rate=sample_rate
        )

    def extract_phonemes_wav2vec2_phoneme(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> AudioPhonemeResult:
        """
        Optional stretch goal secondary path: Phoneme recognition via TIMIT-finetuned Wav2Vec2 model.
        Used for secondary unaligned phoneme acoustic validation.
        """
        start_time = time.perf_counter()
        if audio_waveform is None or len(audio_waveform) == 0:
            latency = (time.perf_counter() - start_time) * 1000.0
            return AudioPhonemeResult(audio_duration_ms=0.0, processing_latency_ms=latency)

        duration_ms = (len(audio_waveform) / sample_rate) * 1000.0

        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            if self._secondary_model is None:
                phoneme_model_name = "facebook/wav2vec2-lv-60-timit-phoneme"
                self._secondary_processor = Wav2Vec2Processor.from_pretrained(phoneme_model_name)
                self._secondary_model = Wav2Vec2ForCTC.from_pretrained(phoneme_model_name)
                self._secondary_model.eval()

            inputs = self._secondary_processor(audio_waveform, sampling_rate=sample_rate, return_tensors="pt")
            with torch.no_grad():
                logits = self._secondary_model(inputs.input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)[0].numpy()
            transcription = self._secondary_processor.decode(predicted_ids)

            latency = (time.perf_counter() - start_time) * 1000.0
            return AudioPhonemeResult(
                audio_duration_ms=duration_ms,
                processing_latency_ms=latency,
            )
        except Exception as e:
            logger.exception("Secondary phoneme model error: %s", e)
            raise RuntimeError(f"Secondary phoneme model failed: {e}") from e

    def is_bilabial_active(
        self,
        bilabial_intervals: List[PhonemeInterval],
        timestamp_ms: float,
        window_ms: float = 50.0,
    ) -> bool:
        """Checks if a bilabial phoneme is active near the given millisecond timestamp."""
        for interval in bilabial_intervals:
            if (interval.start_time_ms - window_ms) <= timestamp_ms <= (interval.end_time_ms + window_ms):
                return True
        return False

    def load_asr_model(self):
        """Loads secondary Wav2Vec2ForCTC ASR model and processor for free-form acoustic transcription.

        Tries loading from the local HuggingFace cache snapshot first (avoids SSL/network errors
        when the model is already cached). Falls back to standard from_pretrained if not cached.
        """
        if self._secondary_model is None:
            try:
                import os
                from pathlib import Path as _Path
                from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

                # Attempt to resolve a fully-cached local snapshot to avoid SSL failures.
                local_snapshot: Optional[str] = None
                hf_cache_root = _Path(os.environ.get("HF_HOME", _Path.home() / ".cache" / "huggingface" / "hub"))
                safe_model_name = WAV2VEC2_MODEL_NAME.replace("/", "--")
                model_cache_dir = hf_cache_root / f"models--{safe_model_name}" / "snapshots"
                if model_cache_dir.exists():
                    snapshots = sorted(model_cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                    for snap in snapshots:
                        # Require at minimum config.json + model weights to be present
                        if (snap / "config.json").exists() and any(snap.glob("model*")):
                            local_snapshot = str(snap)
                            break

                if local_snapshot:
                    logger.info("Loading Wav2Vec2 from local cache: %s", local_snapshot)
                    self._secondary_processor = Wav2Vec2Processor.from_pretrained(
                        local_snapshot, local_files_only=True
                    )
                    self._secondary_model = Wav2Vec2ForCTC.from_pretrained(
                        local_snapshot, local_files_only=True
                    )
                else:
                    logger.info("Local cache not found; downloading Wav2Vec2 from HuggingFace Hub.")
                    self._secondary_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
                    self._secondary_model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_NAME)

                self._secondary_model.eval()
                logger.info("Wav2Vec2 ASR model loaded successfully.")
            except Exception as err:
                logger.exception("Failed to load secondary Wav2Vec2 ASR model: %s", err)
                raise RuntimeError(f"Failed to load secondary Wav2Vec2 ASR model: {err}") from err

    def decode_transcription(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> str:
        """
        Decodes free-form transcription from raw audio waveform using Wav2Vec2ForCTC.
        Decodes audio independent of any assumed prompt text.
        """
        if audio_waveform is None or len(audio_waveform) == 0:
            return ""
        if self._secondary_model is None:
            self.load_asr_model()

        if sample_rate != 16000 and HAS_TORCHAUDIO:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            wf_tensor = resampler(torch.from_numpy(audio_waveform).unsqueeze(0)).squeeze(0)
            waveform_16k = wf_tensor.numpy()
        else:
            waveform_16k = audio_waveform

        inputs = self._secondary_processor(waveform_16k, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self._secondary_model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        return str(self._secondary_processor.decode(pred_ids))

    def verify_prompt_acoustic_match(
        self,
        audio_waveform: np.ndarray,
        prompt_text: str,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        max_cer: float = 0.60,
    ) -> Tuple[bool, float, float, str]:
        """
        Acoustic prompt verification via free-form ASR decoding.
        Decodes audio independently and measures normalized character Levenshtein distance (CER)
        and word error rate (WER) against the issued challenge prompt.

        Returns:
            (is_matched, cer_score, wer_score, decoded_transcription)
        """
        if audio_waveform is None or len(audio_waveform) == 0:
            return False, 1.0, 1.0, ""

        decoded = self.decode_transcription(audio_waveform, sample_rate=sample_rate)
        if not decoded.strip():
            return False, 1.0, 1.0, ""

        cer = compute_cer(prompt_text, decoded)
        wer = compute_wer(prompt_text, decoded)
        is_matched = cer <= max_cer
        return is_matched, cer, wer, decoded
