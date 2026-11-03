"""
Unit tests for app/prompt_generator.py
Verifies bilabial-rich phrase bank, session-level non-repeating sampling,
and explicit session prompt logging.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.prompt_generator import ActivePromptGenerator, BILABIAL_PHRASE_BANK


def test_bilabial_phrase_bank_density():
    """Verifies that all phrases in phrase bank contain at least 2 bilabial consonants (/m/, /b/, /p/)."""
    generator = ActivePromptGenerator()
    assert len(generator.phrase_bank) >= 10, "Phrase bank must contain at least 10 bilabial phrases."

    for phrase in generator.phrase_bank:
        count = generator.count_bilabials(phrase)
        assert count >= 2, f"Phrase '{phrase}' must contain at least 2 bilabial consonants (got {count})."


def test_session_non_repeating_phrases(tmp_path):
    """Verifies that sequential calls for the same session_id return non-repeating phrases until bank is exhausted."""
    generator = ActivePromptGenerator(processed_dir=tmp_path)
    session_id = "test_session_1"

    issued_phrases = []
    bank_size = len(generator.phrase_bank)

    for _ in range(bank_size):
        challenge = generator.generate_challenge(session_id=session_id)
        phrase = challenge["prompt"]
        assert phrase not in issued_phrases, f"Phrase '{phrase}' repeated within session '{session_id}' before bank exhaustion!"
        issued_phrases.append(phrase)

    assert len(issued_phrases) == bank_size


def test_get_issued_prompt_lookup(tmp_path):
    """Verifies that get_issued_prompt returns the exact phrase issued to that session_id."""
    generator = ActivePromptGenerator(processed_dir=tmp_path)
    session_id = "session_xyz"

    challenge = generator.generate_challenge(session_id=session_id)
    issued = challenge["prompt"]

    retrieved = generator.get_issued_prompt(session_id)
    assert retrieved == issued


def test_session_prompt_logging_file(tmp_path):
    """Verifies that issued prompts are explicitly written to data/processed/session_prompts.log."""
    generator = ActivePromptGenerator(processed_dir=tmp_path)
    session_id = "session_log_test"

    generator.generate_challenge(session_id=session_id)

    log_path = tmp_path / "session_prompts.log"
    assert log_path.exists(), "session_prompts.log file must be created."

    content = log_path.read_text(encoding="utf-8")
    assert session_id in content
    assert "Issued Phrase:" in content
