"""
SecureKYC AI - Active Bilabial Prompt Generator Module (app/prompt_generator.py)
Generates dynamic, bilabial-rich challenge phrases dense in /m/, /b/, /p/ sounds.

Tracks issued phrases per session_id to guarantee NO REPEATS within a session.
Logs exact issued prompt phrases to data/processed/session_prompts.log for inference alignment.
"""

import sys
import random
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import PROCESSED_DATA_DIR, BILABIAL_PHONEMES

# Extended Bilabial-Rich Phrase Bank (/m/, /b/, /p/ heavy)
BILABIAL_PHRASE_BANK: List[str] = [
    "My Bank Passcode",
    "Bumble Bee Password",
    "Mobile Payment Protocol",
    "Prime Member Pass",
    "Big Blue Map",
    "Bob Met Pam By The Pier",
    "Paper Bag On My Bench",
    "Public Payment Passbook",
    "Map My Mobile Path",
    "Best Member Passcode",
    "Big Apple Mobile Bank",
    "Please Push My Button",
    "Peter Bought Primary Bond",
    "Blue Palm Tree Pass",
    "Pop Music Beat Play",
]


class ActivePromptGenerator:
    """
    Generates dynamic bilabial-dense challenge phrases.
    Enforces session-level non-repeating sampling and logs issued phrase text.
    """

    def __init__(self, phrase_bank: Optional[List[str]] = None, processed_dir: Path = PROCESSED_DATA_DIR):
        self.phrase_bank = list(phrase_bank or BILABIAL_PHRASE_BANK)
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.log_file_path = self.processed_dir / "session_prompts.log"
        self._init_logger()

        # Session tracking state
        self.session_history: Dict[str, List[str]] = {}  # session_id -> list of used phrases
        self.issued_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> active prompt payload

    def _init_logger(self):
        """Initializes logging file for recording session prompt phrases."""
        self.logger = logging.getLogger(f"PromptGenerator_{id(self)}")
        self.logger.setLevel(logging.INFO)

        self.logger.handlers.clear()
        file_handler = logging.FileHandler(str(self.log_file_path), encoding="utf-8")
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def count_bilabials(self, phrase: str) -> int:
        """Counts occurrence of bilabial consonants (/m/, /b/, /p/) in phrase."""
        return sum(1 for c in phrase.upper() if c in BILABIAL_PHONEMES)

    def generate_challenge(self, session_id: str = "default_session") -> Dict[str, Any]:
        """
        Generates a bilabial-dense challenge phrase randomized per session with NO REPEATS within session.
        Logs the exact issued phrase to data/processed/session_prompts.log.
        """
        used_phrases = self.session_history.get(session_id, [])

        # Filter candidates not yet used in current session
        available_candidates = [p for p in self.phrase_bank if p not in used_phrases]

        if not available_candidates:
            # If all phrases exhausted for this session, reset session history and resample
            self.logger.info(f"[*] Session '{session_id}' exhausted all phrase bank entries. Resetting session phrase history.")
            used_phrases = []
            available_candidates = list(self.phrase_bank)

        # Randomly select a non-repeated phrase
        selected_phrase = random.choice(available_candidates)
        used_phrases.append(selected_phrase)
        self.session_history[session_id] = used_phrases

        bilabial_count = self.count_bilabials(selected_phrase)
        timestamp_ms = time.time() * 1000.0

        payload = {
            "session_id": session_id,
            "prompt": selected_phrase,
            "bilabial_target_count": bilabial_count,
            "target_phonemes": ["/b/", "/m/", "/p/"],
            "timestamp_ms": timestamp_ms,
        }

        self.issued_sessions[session_id] = payload

        # Explicit logging of issued phrase for inference forced-alignment lookup
        log_msg = f"Session: {session_id} | Issued Phrase: '{selected_phrase}' | Bilabials: {bilabial_count}"
        self.logger.info(log_msg)
        print(f"[PROMPT_ISSUED] {log_msg}")

        return payload

    def get_issued_prompt(self, session_id: str) -> str:
        """
        Retrieves the exact challenge phrase issued to a given session_id.
        Ensures Step B1 forced-alignment step receives the exact phrase at inference time.
        """
        session_info = self.issued_sessions.get(session_id)
        if session_info and "prompt" in session_info:
            return session_info["prompt"]
        
        # Fallback if session ID not found in cache
        return "My Bank Passcode"

    def clear_session(self, session_id: str):
        """Clears session tracking history upon disconnect."""
        self.session_history.pop(session_id, None)
        self.issued_sessions.pop(session_id, None)


# Global Prompt Generator Singleton
prompt_generator = ActivePromptGenerator()
