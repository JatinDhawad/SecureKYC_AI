"""
Tests for benchmark_adversarial_validation.py.

The prior test_generate_adversarial_attack_samples() has been removed because
generate_adversarial_attack_samples() was removed from the script for violating
the zero-synthetic-data rule (.agents/AGENTS.md Rule 1).

The test_evaluate_real_world_validation_set() test verifies that the script
correctly raises FileNotFoundError when the real adversarial clips file is absent,
rather than fabricating synthetic data.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_adversarial_validation import evaluate_real_world_validation_set


def test_evaluate_raises_when_no_real_adversarial_clips(tmp_path):
    """
    Verifies that evaluate_real_world_validation_set() raises FileNotFoundError
    when no real OBS/DeepFaceLive session clips exist, instead of generating
    synthetic replacements.
    """
    missing_path = tmp_path / "adversarial_attacks.pt"
    with pytest.raises(FileNotFoundError, match="pending"):
        evaluate_real_world_validation_set(adv_path=missing_path)
