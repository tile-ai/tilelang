import sys
from pathlib import Path

import pytest
import torch

import tilelang.language as T
import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "hadamard_transform"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_hadamard  # noqa: E402


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dim", [16, 256, 1024, 8192, 32768])
def test_hadamard(dim):
    """Validate sub-wave, wave, and shared-memory exchange paths on gfx942."""
    torch.manual_seed(0)
    input_tensor = torch.randn((2, dim), device="cuda", dtype=torch.float32)

    output = example_hadamard.hadamard(input_tensor, dim, T.float32)
    torch.cuda.synchronize()
    reference = example_hadamard.ref_program(input_tensor)

    torch.testing.assert_close(
        output,
        reference,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda details: f"dim={dim}: {details}",
    )
