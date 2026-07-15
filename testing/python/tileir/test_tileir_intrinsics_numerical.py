"""Numerical tests for TileIR scalar/tile intrinsics and extern helpers."""

from __future__ import annotations

import pytest
import tilelang
import tilelang.language as T

# ---------------------------------------------------------------------------
# Guard: cuda_tile MLIR bindings required to build any kernel
# ---------------------------------------------------------------------------

try:
    from cuda_tile._mlir import ir as _ir  # noqa: F401

    _HAS_CUDA_TILE = True
except ImportError:
    _HAS_CUDA_TILE = False

skip_no_cuda_tile = pytest.mark.skipif(
    not _HAS_CUDA_TILE,
    reason="cuda_tile MLIR bindings unavailable",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_tileir_toolchain_unavailable():
    from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available

    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")


def _setup_gpu():
    """Return (torch, major, minor, target_str) or skip."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    major, minor = torch.cuda.get_device_capability()
    target_str = f"tileir -arch=sm_{major}{minor}"
    return torch, major, minor, target_str


# ---------------------------------------------------------------------------
# Test: atan2
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_atan2_numerical():
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n = 256

    @T.prim_func
    def kern(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(n):
                C[i] = T.atan2(A[i], B[i])

    kernel = tilelang.compile(kern, execution_backend="tileir")
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda") + 0.5
    c = torch.empty(n, device="cuda")
    kernel(a, b, c)
    torch.testing.assert_close(c, torch.atan2(a, b), rtol=1e-4, atol=1e-4)


@skip_no_cuda_tile
def test_atan2_scalar_context_numerical():
    """Scalar-form atan2 (C[0] = T.atan2(A[0], B[0])) exercises the expr.py
    _lower_call_scalar dispatch, unlike the T.Parallel form which lowers
    through tile_level.py's _lower_call_tile."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    @T.prim_func
    def kern(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32"), C: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=32):
            C[0] = T.atan2(A[0], B[0])

    kernel = tilelang.compile(kern, execution_backend="tileir")
    a = torch.randn(1, device="cuda")
    b = torch.randn(1, device="cuda") + 0.5
    c = torch.empty(1, device="cuda")
    kernel(a, b, c)
    torch.testing.assert_close(c, torch.atan2(a, b), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test: bf16 atomic_add
# ---------------------------------------------------------------------------


@skip_no_cuda_tile
def test_atomic_add_bf16_numerical():
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n = 128

    # Note: shape is a literal (not the `n` closure var) because `kern`'s body
    # never references `n` at runtime; with `from __future__ import
    # annotations` active in this module, T.prim_func resolves annotation
    # forward-refs via the function's __closure__ (see
    # eager/utils.py:get_func_nonlocals), and Python only captures a name as
    # a closure cell if it's referenced in the compiled body -- a
    # string-only annotation reference doesn't count. Using `n` here would
    # raise `NameError: name 'n' is not defined` for any dtype, independent
    # of the bf16 atomic path under test.
    @T.prim_func
    def kern(Src: T.Tensor((128,), "bfloat16"), Acc: T.Tensor((128,), "bfloat16")):
        with T.Kernel(4, threads=128):
            T.atomic_add(Acc, Src)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    src = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    acc = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
    kernel(src, acc)
    # 4 blocks each add Src once → Acc == 4 * Src (bf16 tolerance).
    torch.testing.assert_close(acc, 4 * src, rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# Test: decode_fp4_to_bf16_twiddling extern (structured TileIR port)
#
# The inline-PTX helper `decode_fp4_to_bf16_twiddling`
# (tilelang/quantize/mxfp.py, used by the mxfp4 dequant-GEMM examples) is a
# pure bit-twiddling elementwise conversion, so the TileIR backend lowers the
# `T.call_extern` to the structured `DecodeFp4Twiddling` IR op
# (tilelang/tileir/ir/ops/misc.py) instead of rejecting it.  The reference
# below is the half-word form of `torch_convert_bit_twiddling`
# (examples/dequantize_gemm/dequantize_utils.py): each 16-bit word
# w = (byte_even << 8) | byte_odd yields 4 bf16 bit patterns, scaled by 2^126.
# ---------------------------------------------------------------------------


def _ref_decode_fp4_twiddling(torch, src):
    """Reference decode: (4n,) uint8 -> (8n,) bf16 (matches
    torch_convert_bit_twiddling from examples/dequantize_gemm)."""
    n = src.numel() // 4
    out = torch.empty(8 * n, dtype=torch.bfloat16)
    for g in range(n):
        for half in range(2):
            w = (int(src[4 * g + 2 * half]) << 8) | int(src[4 * g + 2 * half + 1])
            res = [
                w & 0x81C0,
                (w << 3) & 0x81C0,
                (w << 6) & 0x81C0,
                ((w << 1) & 0x8000) | ((w >> 3) & 0x0180) | ((w >> 7) & 0x0040),
            ]
            for pos, bits in enumerate(res):
                value = torch.tensor([bits & 0xFFFF], dtype=torch.uint16).view(torch.bfloat16)[0]
                out[8 * g + 4 * half + pos] = value * (2.0**126)
    return out


@skip_no_cuda_tile
def test_decode_fp4_to_bf16_twiddling_numerical():
    """Single-group form (N=1): 4 packed bytes -> 8 bf16, bit-exact."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    @T.prim_func
    def kern(Src: T.Tensor((4,), "uint8"), Dst: T.Tensor((8,), "bfloat16")):
        with T.Kernel(1, threads=32):
            packed = T.alloc_local((4,), "uint8")
            decoded = T.alloc_local((8,), "bfloat16")
            T.copy(Src, packed)
            T.call_extern(
                "handle",
                "decode_fp4_to_bf16_twiddling",
                T.access_ptr(packed, "r"),
                T.access_ptr(decoded, "w"),
                1,
            )
            T.copy(decoded, Dst)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(0)
    for _ in range(4):
        src = torch.randint(0, 256, (4,), dtype=torch.uint8, device="cuda")
        dst = torch.empty(8, dtype=torch.bfloat16, device="cuda")
        kernel(src, dst)
        ref = _ref_decode_fp4_twiddling(torch, src.cpu()).cuda()
        torch.testing.assert_close(dst, ref, rtol=0, atol=0)


@skip_no_cuda_tile
def test_decode_fp4_to_bf16_twiddling_multigroup_numerical():
    """Multi-group form (N=4): 16 packed bytes -> 32 bf16, bit-exact."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    @T.prim_func
    def kern(Src: T.Tensor((16,), "uint8"), Dst: T.Tensor((32,), "bfloat16")):
        with T.Kernel(1, threads=32):
            packed = T.alloc_local((16,), "uint8")
            decoded = T.alloc_local((32,), "bfloat16")
            T.copy(Src, packed)
            T.call_extern(
                "handle",
                "decode_fp4_to_bf16_twiddling",
                T.access_ptr(packed, "r"),
                T.access_ptr(decoded, "w"),
                4,
            )
            T.copy(decoded, Dst)

    kernel = tilelang.compile(kern, execution_backend="tileir")
    torch.manual_seed(1)
    src = torch.randint(0, 256, (16,), dtype=torch.uint8, device="cuda")
    dst = torch.empty(32, dtype=torch.bfloat16, device="cuda")
    kernel(src, dst)
    ref = _ref_decode_fp4_twiddling(torch, src.cpu()).cuda()
    torch.testing.assert_close(dst, ref, rtol=0, atol=0)
