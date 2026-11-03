"""
SecureKYC AI - Apple Silicon Hardware & PyTorch Environment Smoke Test
Verifies PyTorch Metal Performance Shaders (MPS) execution on Apple Silicon.
Ensures zero silent fallbacks to CPU.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config import get_device, DEVICE


def test_mps_environment():
    """
    Smoke test to verify PyTorch MPS backend availability and tensor operations.
    Raises AssertionError if MPS is unavailable or falls back to CPU.
    """
    print("=" * 65)
    print(" SecureKYC AI - Environment & MPS Hardware Smoke Test ")
    print("=" * 65)

    # 1. Verify PyTorch Version
    print(f"[1/4] PyTorch Version: {torch.__version__}")

    # 2. Check MPS Build & Availability
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"[2/4] MPS Built: {mps_built} | MPS Available: {mps_available}")

    assert mps_built, "PyTorch was not built with MPS support."
    assert mps_available, "MPS hardware acceleration is not available on this machine."

    # 3. Create Tensors on MPS Device
    device = torch.device("mps")
    print(f"[3/4] Target Compute Device: {device}")

    t0 = time.perf_counter()
    matrix_a = torch.randn((500, 500), device=device, dtype=torch.float32)
    matrix_b = torch.randn((500, 500), device=device, dtype=torch.float32)

    # 4. Perform Operation on Device
    result = torch.matmul(matrix_a, matrix_b) + 3.14159
    torch.mps.synchronize()  # Synchronize Metal stream to ensure execution
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[4/4] Tensor Operation Execution:")
    print(f"      - Operation: matrix_a (500x500) @ matrix_b (500x500) + scalar")
    print(f"      - Output Tensor Device: {result.device}")
    print(f"      - Output Tensor Shape:  {list(result.shape)}")
    print(f"      - Mean Result Value:    {result.mean().item():.4f}")
    print(f"      - Execution Latency:    {elapsed_ms:.2f} ms")

    # Strict Assertions: Confirm device is MPS and NOT CPU
    assert result.device.type == "mps", f"Expected device type 'mps', but got '{result.device.type}' (Silent CPU fallback detected!)."
    assert not result.is_cuda, "CUDA reported unexpectedly on Apple Silicon."

    print("-" * 65)
    print(" SUCCESS: PyTorch MPS acceleration is active and operational.")
    print("=" * 65)


if __name__ == "__main__":
    try:
        test_mps_environment()
        sys.exit(0)
    except AssertionError as err:
        print(f"\n FAILURE: {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n ERROR during MPS smoke test: {exc}", file=sys.stderr)
        sys.exit(1)
