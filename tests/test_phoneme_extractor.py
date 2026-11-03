"""
SecureKYC AI - Unit & Empirical Integration Tests for Audio Phoneme Extractor
Verifies torchaudio forced alignment (MMS_FA) on real recorded audio clips,
extracting T_bilabial timestamps for /b/, /m/, /p/ consonant occurrences.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, AUDIO_SAMPLE_RATE
from src.audio.phoneme_extractor import PhonemeExtractor, AudioPhonemeResult


def get_real_sample_audio_clips():
    """Finds real VidTIMIT WAV audio clips and their corresponding known transcripts."""
    orig_dir = RAW_DATA_DIR / "original_sequences"

    samples = [
        {
            "path": orig_dir / "fadg0" / "audio" / "sx199.wav",
            "transcript": "The small baby came home.",
            "expected_bilabials": ["M", "B", "B", "M", "M"],
        },
        {
            "path": orig_dir / "fadg0" / "audio" / "sx19.wav",
            "transcript": "The paper boy delivered the newspaper.",
            "expected_bilabials": ["P", "P", "B", "P", "P"],
        },
        {
            "path": orig_dir / "fadg0" / "audio" / "sa2.wav",
            "transcript": "Don't ask me to carry an oily rag like that.",
            "expected_bilabials": ["M"],
        },
    ]

    available = [s for s in samples if s["path"].exists()]
    assert len(available) > 0, f"Expected real audio sample WAV files in '{orig_dir}'"
    return available


def test_phoneme_extractor_initialization():
    """Verifies initialization of PhonemeExtractor."""
    extractor = PhonemeExtractor(lazy_load=True)
    assert extractor.bilabial_set == {"M", "B", "P"}


def test_phoneme_extractor_empty_audio():
    """Verifies handling of empty or zero-length audio arrays."""
    extractor = PhonemeExtractor(lazy_load=True)
    empty_audio = np.zeros(0, dtype=np.float32)
    res = extractor.align_forced_transcript(empty_audio, prompt_text="MY BANK PASSCODE")

    assert res.audio_duration_ms == 0.0
    assert len(res.phoneme_sequence) == 0
    assert len(res.bilabial_intervals) == 0


def test_extract_phonemes_requires_explicit_prompt():
    """Verifies that extract_phonemes raises ValueError when prompt_text is missing or empty."""
    extractor = PhonemeExtractor(lazy_load=True)
    with pytest.raises(ValueError, match="Explicit challenge prompt transcript must be provided"):
        extractor.extract_phonemes(np.zeros(100, dtype=np.float32), prompt_text="")

    with pytest.raises(ValueError, match="Explicit challenge prompt transcript must be provided"):
        extractor.extract_phonemes(np.zeros(100, dtype=np.float32), prompt_text="   ")


def test_bilabial_active_lookup():
    """Verifies millisecond time window lookup for active bilabials."""
    extractor = PhonemeExtractor(lazy_load=True)
    res = extractor.align_forced_transcript(np.zeros(0, dtype=np.float32), prompt_text="MY BANK PASSCODE")

    from src.audio.phoneme_extractor import PhonemeInterval
    intervals = [
        PhonemeInterval(phoneme="B", start_time_ms=500.0, end_time_ms=600.0, is_bilabial=True),
        PhonemeInterval(phoneme="M", start_time_ms=1200.0, end_time_ms=1300.0, is_bilabial=True),
    ]

    assert extractor.is_bilabial_active(intervals, timestamp_ms=520.0) is True
    assert extractor.is_bilabial_active(intervals, timestamp_ms=480.0, window_ms=50.0) is True
    assert extractor.is_bilabial_active(intervals, timestamp_ms=900.0) is False


def test_real_audio_forced_alignment_empirical():
    """
    Empirical test running torchaudio forced alignment against real recorded clips.
    Recovers T_bilabial timestamps and compares them with expected reference characters.
    """
    clips = get_real_sample_audio_clips()
    extractor = PhonemeExtractor(lazy_load=False)

    print("\n" + "=" * 75)
    print(" torchaudio MMS_FA Forced Alignment Empirical Verification ")
    print("=" * 75)

    for idx, clip_info in enumerate(clips):
        path = clip_info["path"]
        transcript = clip_info["transcript"]
        sr, audio_int16 = wavfile.read(str(path))

        # Convert int16 audio to float32 [-1.0, 1.0]
        if audio_int16.dtype == np.int16:
            waveform = audio_int16.astype(np.float32) / 32768.0
        else:
            waveform = audio_int16.astype(np.float32)

        res = extractor.align_forced_transcript(waveform, prompt_text=transcript, sample_rate=sr)

        print(f"[*] Clip {idx + 1}/{len(clips)}: {path.name}")
        print(f"    - Prompt Transcript: '{transcript}'")
        print(f"    - Audio Duration:   {res.audio_duration_ms:.1f} ms")
        print(f"    - Alignment Score:  {res.alignment_score:.4f}")
        print(f"    - Latency:          {res.processing_latency_ms:.2f} ms")
        print(f"    - Total Characters Aligned: {len(res.phoneme_sequence)}")
        print(f"    - Isolated T_bilabial Intervals ({len(res.bilabial_intervals)} found):")

        recovered_phonemes = []
        for b_interval in res.bilabial_intervals:
            recovered_phonemes.append(b_interval.phoneme)
            print(
                f"      • Bilabial [{b_interval.phoneme}]: "
                f"T_start = {b_interval.start_time_ms:6.1f} ms | "
                f"T_end = {b_interval.end_time_ms:6.1f} ms | "
                f"Confidence = {b_interval.confidence:.3f}"
            )

        assert recovered_phonemes == clip_info["expected_bilabials"], (
            f"Phoneme extraction mismatch for '{transcript}': "
            f"got {recovered_phonemes}, expected {clip_info['expected_bilabials']}"
        )
        assert res.audio_duration_ms > 0.0

    print("=" * 75)


def test_ground_truth_sa2_single_bilabial_regression():
    """
    Regression test for span.token indexing bug.
    Verifies that 'Don't ask me to carry an oily rag like that.' produces EXACTLY ONE
    bilabial consonant ('M' in 'me' around ~1306 ms), and does NOT misclassify rhotic
    consonants ('r' in 'carry', 'rag') or vowels as bilabials.
    """
    orig_dir = RAW_DATA_DIR / "original_sequences"
    wav_path = orig_dir / "fadg0" / "audio" / "sa2.wav"
    assert wav_path.exists(), f"Reference WAV file not found: {wav_path}"

    sr, audio_int16 = wavfile.read(str(wav_path))
    waveform = audio_int16.astype(np.float32) / 32768.0 if audio_int16.dtype == np.int16 else audio_int16.astype(np.float32)

    extractor = PhonemeExtractor(lazy_load=False)
    res = extractor.align_forced_transcript(
        waveform,
        prompt_text="Don't ask me to carry an oily rag like that.",
        sample_rate=sr,
    )

    # Must recover exactly 1 bilabial event: 'M' in 'me'
    assert len(res.bilabial_intervals) == 1, (
        f"Expected exactly 1 bilabial interval, got {len(res.bilabial_intervals)}: "
        f"{[b.phoneme for b in res.bilabial_intervals]}"
    )
    b = res.bilabial_intervals[0]
    assert b.phoneme == "M"
    # Verify timestamp is around 1306 ms (window 1200 - 1400 ms)
    assert 1200.0 <= b.start_time_ms <= 1400.0, f"Expected M start around ~1306 ms, got {b.start_time_ms:.1f} ms"
    assert b.end_time_ms > b.start_time_ms


def test_compute_wer_and_cer():
    """Verifies Levenshtein distance and WER metric computations."""
    from src.audio.phoneme_extractor import compute_wer, compute_cer
    assert compute_wer("hello world", "hello world") == 0.0
    assert compute_cer("hello world", "hello world") == 0.0

    # 1 word substituted out of 2 = WER 0.5
    assert abs(compute_wer("hello world", "hello there") - 0.5) < 1e-5
    # Completely disjoint
    assert compute_wer("apple", "banana orange") >= 1.0
    assert compute_cer("abc", "xyz") == 1.0


def test_verify_prompt_acoustic_match_real_clip():
    """Verifies free-form ASR decoding and prompt correspondence check on a real audio clip."""
    orig_dir = RAW_DATA_DIR / "original_sequences"
    wav_path = orig_dir / "fadg0" / "audio" / "sa2.wav"
    assert wav_path.exists()

    sr, audio_int16 = wavfile.read(str(wav_path))
    waveform = audio_int16.astype(np.float32) / 32768.0

    extractor = PhonemeExtractor(lazy_load=False)
    
    # 1. Matching prompt: should pass CER <= 0.60
    is_match, cer, wer, hyp = extractor.verify_prompt_acoustic_match(
        waveform, prompt_text="Don't ask me to carry an oily rag like that.", sample_rate=sr, max_cer=0.60
    )
    assert is_match is True
    assert cer <= 0.50
    assert len(hyp) > 0

    # 2. Mismatching prompt: should fail CER > 0.60
    is_match_fake, cer_fake, wer_fake, hyp_fake = extractor.verify_prompt_acoustic_match(
        waveform, prompt_text="Bumble Bee Password", sample_rate=sr, max_cer=0.60
    )
    assert is_match_fake is False
    assert cer_fake > 0.60


