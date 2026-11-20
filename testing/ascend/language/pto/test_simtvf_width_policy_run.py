"""Device acceptance: the PTO target keeps 64-bit vector planning.

The PTO SIMT codegen lowers compute ops through x2-granular emitters
(FP8 conversions, floating min/max, ...), so the shared 128-bit default
would plan 4-lane ops it cannot emit. The pto target therefore declares
``max_vector_bits = 64``, which is exactly what every target planned
before the default change, and these kernels are the shapes that would
otherwise fail:

- an fp8 vecadd whose 32-element per-thread run plans 16 fp8 scalar
  lanes at 128 bits (``FP8TypeName`` rejects anything past 8);
- an f32 -> fp8 cast whose 4-lane conversion exceeds the x2-only FP8
  cast emitter;
- an f32 min whose 4-lane operation exceeds ``PrintFloatMinMax_``.

Each test compiles through the real PTO pipeline (PTODSL -> PTOAS ->
Bisheng) and checks the numeric result on the NPU, so a planner
regression that lifts the PTO width cap fails here instead of shipping.
"""

import os
import re
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import torch_npu  # noqa: E402,F401  (import before tilelang/ptodsl)
import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import tilelang.testing  # noqa: E402

THREADS = 128
ELEMS_PER_THREAD = 32
N = THREADS * ELEMS_PER_THREAD


def _require_npu_runtime():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch.npu is unavailable in the current environment")
    try:
        source = torch.arange(8, dtype=torch.float32)
        copied = source.to(torch.device("npu"))
        torch.npu.synchronize()
        assert torch.equal(copied.cpu(), source)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


@pytest.fixture(scope="module")
def npu():
    _require_npu_runtime()
    return torch.device("npu")


@tilelang.jit(out_idx=[2], target="pto")
def fp8_simt_vecadd(n: int, dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            b_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=THREADS):
                a_local = T.alloc_fragment((n,), dtype)
                b_local = T.alloc_fragment((n,), dtype)
                c_local = T.alloc_fragment((n,), dtype)
                T.copy(a_ub, a_local)
                T.copy(b_ub, b_local)
                for i in T.Parallel(n):
                    c_local[i] = a_local[i] + b_local[i]
                T.copy(c_local, c_ub)
            T.copy(c_ub, C)

    return main


def _lane_suffixes(source, f8):
    for match in re.finditer(rf"pto\.{f8}(?:x(\d+))?", source):
        yield int(match.group(1) or "1")


@pytest.mark.pto
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_fp8_simt_width_stays_capped_on_pto(dtype, npu):
    kernel = fp8_simt_vecadd(N, dtype)
    source = kernel.get_kernel_source()
    for f8 in ("f8e4m3", "f8e5m2"):
        assert f"{f8}x16" not in source
    widest = max(
        (lanes for f8 in ("f8e4m3", "f8e5m2") for lanes in _lane_suffixes(source, f8)),
        default=0,
    )
    assert widest <= 8

    td = getattr(torch, dtype)
    a = torch.linspace(-128, 128, N).to(td).to(npu)
    b = torch.linspace(64, -64, N).to(td).to(npu)
    actual = kernel(a, b).cpu().float()
    expected = (a.float() + b.float()).to(td).float().cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.jit(out_idx=[1], target="pto")
def f32_to_fp8_cast(n: int, dtype):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            with T.SimtVF(threads=THREADS):
                a_local = T.alloc_fragment((n,), "float32")
                c_local = T.alloc_fragment((n,), dtype)
                T.copy(a_ub, a_local)
                for i in T.Parallel(n):
                    c_local[i] = T.cast(a_local[i], dtype)
                T.copy(c_local, c_ub)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_f32_to_fp8_cast_stays_within_pair_width(dtype, npu):
    td = getattr(torch, dtype)
    kernel = f32_to_fp8_cast(N, dtype)
    # Values exactly representable in E4M3/E5M2 keep the comparison bitwise
    # independent of the rounding mode.
    a = (torch.arange(N, dtype=torch.float32) % 17 - 8) * 0.25
    actual = kernel(a.to(npu)).cpu()
    expected = a.to(td)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@tilelang.jit(out_idx=[2], target="pto")
def f32_min_kernel(n: int):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), "float32")
            b_ub = T.alloc_shared((n,), "float32")
            c_ub = T.alloc_shared((n,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=THREADS):
                a_local = T.alloc_fragment((n,), "float32")
                b_local = T.alloc_fragment((n,), "float32")
                c_local = T.alloc_fragment((n,), "float32")
                T.copy(a_ub, a_local)
                T.copy(b_ub, b_local)
                for i in T.Parallel(n):
                    c_local[i] = T.min(a_local[i], b_local[i])
                T.copy(c_local, c_ub)
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_f32_min_stays_within_pair_width(npu):
    kernel = f32_min_kernel(N)
    a = torch.linspace(-4.0, 4.0, N, dtype=torch.float32, device=npu)
    b = torch.linspace(4.0, -4.0, N, dtype=torch.float32, device=npu)
    actual = kernel(a, b).cpu()
    expected = torch.minimum(a.cpu(), b.cpu())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
