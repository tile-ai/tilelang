import sys
from pathlib import Path

import torch

import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "grouped_gemm"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_grouped_gemm_bwd as grouped_bwd  # noqa: E402
import example_grouped_gemm_fwd as grouped_fwd  # noqa: E402


_BATCH_SIZES = [5, 9, 13]
_K = 64
_N = 96
_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_NUM_STAGES = 2
_THREADS = 256


def _inputs(*, requires_grad=False):
    torch.manual_seed(0)
    batch_sum = sum(_BATCH_SIZES)
    batch_offsets = [0]
    batch_padded_offsets = [0]
    for size in _BATCH_SIZES[:-1]:
        batch_offsets.append(batch_offsets[-1] + size)
        batch_padded_offsets.append(batch_padded_offsets[-1] + _BLOCK_M)

    a = torch.randn(batch_sum, _K, device="cuda", dtype=torch.float16)
    b = torch.randn(len(_BATCH_SIZES), _K, _N, device="cuda", dtype=torch.float16, requires_grad=requires_grad)
    sizes = torch.tensor(_BATCH_SIZES, device="cuda", dtype=torch.int32)
    offsets = torch.tensor(batch_offsets, device="cuda", dtype=torch.int32)
    padded_offsets = torch.tensor(batch_padded_offsets, device="cuda", dtype=torch.int32)
    return a, b, sizes, offsets, padded_offsets


def _reference(a, b):
    outputs = []
    offset = 0
    for group, size in enumerate(_BATCH_SIZES):
        outputs.append(a[offset : offset + size] @ b[group])
        offset += size
    return torch.cat(outputs)


@tilelang.testing.requires_rocm
def test_grouped_gemm_forward():
    """Validate irregular grouped GEMM forward tiles on ROCm."""
    a, b, sizes, offsets, padded_offsets = _inputs()
    actual = grouped_fwd.grouped_gemm(
        a,
        b,
        sizes,
        offsets,
        padded_offsets,
        tuple(_BATCH_SIZES),
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        False,
        _NUM_STAGES,
        _THREADS,
    )

    torch.testing.assert_close(actual, _reference(a, b), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_rocm
def test_grouped_gemm_forward_backward():
    """Validate the autograd example's grouped forward and weight-gradient kernels."""
    a, b, sizes, offsets, padded_offsets = _inputs(requires_grad=True)
    reference = _reference(a, b)
    output_gradient = torch.randn_like(reference)
    (expected_db,) = torch.autograd.grad(reference, b, output_gradient)

    actual = grouped_bwd.grouped_gemm_fwd(
        a,
        b,
        sizes,
        offsets,
        padded_offsets,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        _NUM_STAGES,
        _THREADS,
    )
    actual_db = grouped_bwd.grouped_gemm_bwd(
        a,
        output_gradient,
        sizes,
        offsets,
        _BLOCK_M,
        _BLOCK_N,
        _BLOCK_K,
        _NUM_STAGES,
        _THREADS,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_db, expected_db, rtol=1e-2, atol=1e-2)
