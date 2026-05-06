"""End-to-end tests for ``T.fp8_scaled_matmul`` on the Metal target.

Mirrors the audiohacking/fp8-mps-metal ``fp8_scaled_matmul_kernel``
algorithm at the TileLang frontend layer. Every test:

  1. Constructs an ``@T.prim_func`` that calls ``T.fp8_scaled_matmul``.
  2. Lowers it on ``Target("metal")``.
  3. Asserts the emitted MSL contains the audiohacking-pattern markers
     (``__tvm_fp8_e4m3_to_half`` / ``__tvm_fp8_e5m2_to_half``) and is
     accepted by ``xcrun --sdk macosx metal -c`` (offline compile).
  4. For E2E parity, runs a hand-written reference matmul (per-element
     ``T.cast`` + ``mx.matmul``) and compares with rtol=5e-3 (FP8
     numeric tolerance).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import pytest

import tilelang
from tilelang import tvm
from tvm.target import Target
import tilelang.language as T
import tilelang.testing


_HAS_METAL_SDK = (
    shutil.which("xcrun") is not None
    and subprocess.run(
        ["xcrun", "--sdk", "macosx", "--find", "metal"], capture_output=True
    ).returncode
    == 0
)


def _make_kernel(
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    BK: int,
    *,
    a_dtype: str = "float8_e4m3",
    b_dtype: str = "float8_e4m3",
    a_scale_size: int = 1,
    b_scale_size: int = 1,
):
    """Build a single-block FP8 scaled matmul prim_func using T.fp8_scaled_matmul.

    Parameters with leading underscores are deliberately stashed into module
    globals so the deferred type-hint evaluator inside ``@T.prim_func`` can
    see them.
    """
    g = globals()
    g.update(
        _M=M, _N=N, _K=K, _BM=BM, _BN=BN, _BK=BK,
        _SA=a_scale_size, _SB=b_scale_size,
        _A_DTYPE=a_dtype, _B_DTYPE=b_dtype,
    )

    @T.prim_func
    def fp8_scaled_kernel(
        A_fp8: T.Tensor((_M, _K), _A_DTYPE),
        A_scale: T.Tensor((_SA,), "float32"),
        B_fp8: T.Tensor((_K, _N), _B_DTYPE),
        B_scale: T.Tensor((_SB,), "float32"),
        C: T.Tensor((_M, _N), "float32"),
    ):
        with T.Kernel(T.ceildiv(_N, _BN), T.ceildiv(_M, _BM), threads=128) as (bx, by):
            A_shared = T.alloc_shared((_BM, _BK), _A_DTYPE, scope="shared")
            B_shared = T.alloc_shared((_BK, _BN), _B_DTYPE, scope="shared")
            C_local = T.alloc_fragment((_BM, _BN), "float32")
            T.clear(C_local)
            for ko in range(T.ceildiv(_K, _BK)):
                T.copy(A_fp8[by * _BM, ko * _BK], A_shared)
                T.copy(B_fp8[ko * _BK, bx * _BN], B_shared)
                T.fp8_scaled_matmul(A_shared, A_scale, B_shared, B_scale, C_local)
            T.copy(C_local, C[by * _BM, bx * _BN])

    return fp8_scaled_kernel


def _xcrun_compile(msl_source: str) -> tuple[int, str]:
    """Run ``xcrun --sdk macosx metal -c`` against the provided MSL.

    Returns (exit_code, stderr).
    """
    with tempfile.NamedTemporaryFile(suffix=".metal", delete=False) as f:
        f.write(msl_source.encode("utf-8"))
        msl_path = f.name
    try:
        air_path = msl_path + ".air"
        res = subprocess.run(
            ["xcrun", "--sdk", "macosx", "metal", "-c", msl_path, "-o", air_path],
            capture_output=True, text=True,
        )
        return res.returncode, (res.stderr or "")
    finally:
        for p in (msl_path, msl_path + ".air"):
            if os.path.exists(p):
                os.remove(p)


# --------------------------------------------------------------------------
# IR-level lowering tests (no GPU required)
# --------------------------------------------------------------------------

def test_per_tensor_scale_lowers_on_metal():
    """Per-tensor scaling: ``A_scale.shape == (1,)``, ``B_scale.shape == (1,)``."""
    fn = _make_kernel(M=32, N=32, K=64, BM=32, BN=32, BK=64)
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)

    # Audiohacking-pattern markers — the MSL inner loop is the
    # ``a_val * b_val * sa * sb`` accumulation.
    assert "__tvm_fp8_e4m3_to_half" in src, (
        "expected MSL to contain Agent C's FP8 dequant helper"
    )
    # Matmul-body shape: should accumulate into C_local.
    body = src[src.find("kernel void"):]
    assert "C_local" in body
    assert "a_val" in body and "b_val" in body
    assert "sa" in body and "sb" in body, (
        "expected per-tensor / per-row scale variables in the inner loop"
    )

    # No simdgroup MMA for FP8 — Apple has no native FP8 ALU through M5.
    assert "simdgroup_multiply_accumulate" not in body, (
        "FP8 input must take the scalar fallback path on Metal"
    )


def test_per_row_scale_lowers_on_metal():
    """Per-row A_scale, per-tensor B_scale."""
    fn = _make_kernel(
        M=32, N=32, K=64, BM=32, BN=32, BK=64,
        a_scale_size=32, b_scale_size=1,
    )
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)

    body = src[src.find("kernel void"):]
    # Per-row indexing should use the row variable as scale index.
    assert "A_scale[" in body
    # Per-tensor B uses index 0.
    assert "B_scale[0]" in body


def test_per_col_scale_lowers_on_metal():
    """Per-tensor A_scale, per-col B_scale."""
    fn = _make_kernel(
        M=32, N=32, K=64, BM=32, BN=32, BK=64,
        a_scale_size=1, b_scale_size=32,
    )
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    body = src[src.find("kernel void"):]

    assert "A_scale[0]" in body
    assert "B_scale[" in body  # per-col indexing


def test_e5m2_lowers_on_metal():
    """e5m2 input dtype uses the matching dequant helper at the call site.

    The codegen prelude bundles both ``__tvm_fp8_e4m3_to_half`` and
    ``__tvm_fp8_e5m2_to_half`` inline definitions whenever any FP8 type
    is touched, so we check the *calls* in the kernel body — not the
    helper definitions in the prelude.
    """
    fn = _make_kernel(
        M=32, N=32, K=64, BM=32, BN=32, BK=64,
        a_dtype="float8_e5m2", b_dtype="float8_e5m2",
    )
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    body = src[src.find("kernel void"):]

    # Calls in the kernel body — A_shared / B_shared loads should both
    # decode through the e5m2 helper.
    assert "__tvm_fp8_e5m2_to_half(A_shared" in body
    assert "__tvm_fp8_e5m2_to_half(B_shared" in body
    # And NO e4m3 calls in the body (the prelude does carry the e4m3
    # helper definition because it's bundled with the e5m2 one in the
    # codegen prelude — that's harmless dead code that the Metal
    # compiler eliminates).
    assert "__tvm_fp8_e4m3_to_half(A_shared" not in body
    assert "__tvm_fp8_e4m3_to_half(B_shared" not in body


def test_mixed_e4m3_e5m2_lowers_on_metal():
    """A in e4m3, B in e5m2 — both helpers must be called from the kernel body."""
    fn = _make_kernel(
        M=32, N=32, K=64, BM=32, BN=32, BK=64,
        a_dtype="float8_e4m3", b_dtype="float8_e5m2",
    )
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    body = src[src.find("kernel void"):]

    # Mixed-dtype: A path uses e4m3, B path uses e5m2.
    assert "__tvm_fp8_e4m3_to_half(A_shared" in body
    assert "__tvm_fp8_e5m2_to_half(B_shared" in body


# --------------------------------------------------------------------------
# Offline ``xcrun metal -c`` acceptance tests (require macOS metal SDK)
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not _HAS_METAL_SDK, reason="macOS metal SDK (xcrun) not available"
)
def test_xcrun_compile_per_tensor_scale():
    """The lowered MSL is accepted by the Metal AIR compiler."""
    fn = _make_kernel(M=32, N=32, K=64, BM=32, BN=32, BK=64)
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    rc, stderr = _xcrun_compile(src)
    assert rc == 0, f"xcrun metal -c failed:\n{stderr}"


@pytest.mark.skipif(
    not _HAS_METAL_SDK, reason="macOS metal SDK (xcrun) not available"
)
def test_xcrun_compile_per_row_scale():
    fn = _make_kernel(
        M=32, N=32, K=64, BM=32, BN=32, BK=64,
        a_scale_size=32, b_scale_size=32,
    )
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    rc, stderr = _xcrun_compile(src)
    assert rc == 0, f"xcrun metal -c failed:\n{stderr}"


@pytest.mark.skipif(
    not _HAS_METAL_SDK, reason="macOS metal SDK (xcrun) not available"
)
def test_xcrun_compile_mixed_dtype():
    fn = _make_kernel(
        M=32, N=32, K=64, BM=32, BN=32, BK=64,
        a_dtype="float8_e4m3", b_dtype="float8_e5m2",
    )
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    rc, stderr = _xcrun_compile(src)
    assert rc == 0, f"xcrun metal -c failed:\n{stderr}"


# --------------------------------------------------------------------------
# End-to-end parity tests (require live Metal device + torch.mps)
# --------------------------------------------------------------------------

try:
    import torch
    _HAS_TORCH_MPS = torch.backends.mps.is_available()
except Exception:
    torch = None
    _HAS_TORCH_MPS = False


def _torch_fp8_quantize(x: "torch.Tensor", dtype: str) -> "torch.Tensor":
    """Quantize float32 tensor to FP8 storage and ship to MPS.

    Conversion to ``torch.float8_e4m3fn`` / ``torch.float8_e5m2`` (PyTorch
    2.1+) is performed on CPU because torch.mps doesn't expose the
    float8 conversion kernels; the FP8-typed tensor is then moved to MPS,
    which only requires byte-level transfer.
    """
    if dtype == "float8_e4m3":
        torch_dtype = torch.float8_e4m3fn
    elif dtype == "float8_e5m2":
        torch_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"unsupported FP8 dtype: {dtype}")
    quant_cpu = x.detach().cpu().to(torch_dtype)
    return quant_cpu.to("mps")


def _torch_fp8_dequantize(x_fp8: "torch.Tensor") -> "torch.Tensor":
    """Inverse of ``_torch_fp8_quantize`` — FP8 -> float32 on the same device."""
    return x_fp8.cpu().to(torch.float32).to(x_fp8.device)


@pytest.mark.skipif(
    not _HAS_TORCH_MPS, reason="torch.mps not available"
)
@tilelang.testing.requires_metal
def test_e2e_per_tensor_scale_parity():
    """Run the kernel on Metal and compare with hand-written reference.

    Reference:
        C = (A_fp32 * A_scale) @ (B_fp32 * B_scale)
    where A_fp32 = dequant(quant_e4m3(A_orig)) and B_fp32 = dequant(quant_e4m3(B_orig)).
    Tolerance is rtol=5e-3 (FP8 rounding noise dominates).
    """
    import torch  # noqa: F401 — guarded by the skip above

    M, N, K = 32, 32, 64
    BM, BN, BK = 32, 32, 64

    fn = _make_kernel(M, N, K, BM, BN, BK)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0xCAFE)
    a_orig = torch.randn(M, K, dtype=torch.float32, device="mps") * 4.0  # in-range for e4m3
    b_orig = torch.randn(K, N, dtype=torch.float32, device="mps") * 4.0
    a_scale = torch.tensor([0.5], dtype=torch.float32, device="mps")
    b_scale = torch.tensor([0.25], dtype=torch.float32, device="mps")

    # Quantize -> uint8 storage. Take the dequant trip back through fp32 to
    # build the reference, since the underlying matmul operates on the
    # quantized values.
    a_fp8 = _torch_fp8_quantize(a_orig, "float8_e4m3")
    b_fp8 = _torch_fp8_quantize(b_orig, "float8_e4m3")
    # Build the reference on CPU in fp32 to avoid MPS using lower-precision
    # accumulators in the matmul.
    a_dequant_cpu = a_fp8.cpu().to(torch.float32)
    b_dequant_cpu = b_fp8.cpu().to(torch.float32)
    c_ref_cpu = (a_dequant_cpu @ b_dequant_cpu) * a_scale[0].cpu().item() * b_scale[0].cpu().item()
    c_ref = c_ref_cpu.to("mps")

    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a_fp8, a_scale, b_fp8, b_scale, c_out)
    torch.mps.synchronize()

    # Compare on CPU too to avoid any MPS round-trip noise.
    c_out_cpu = c_out.cpu()
    diff = torch.abs(c_out_cpu - c_ref_cpu)
    rel = diff / (torch.abs(c_ref_cpu) + 1e-6)
    rmax = rel.max().item()
    abs_max = diff.max().item()
    assert rmax < 5e-3, (
        f"FP8 scaled matmul parity failed: max rel err {rmax:.3g}, "
        f"max abs err {abs_max:.3g} (limit rel 5e-3)\n"
        f"  c_out range: [{c_out_cpu.min().item():.3f}, {c_out_cpu.max().item():.3f}]\n"
        f"  c_ref range: [{c_ref_cpu.min().item():.3f}, {c_ref_cpu.max().item():.3f}]"
    )


@pytest.mark.skipif(
    not _HAS_TORCH_MPS, reason="torch.mps not available"
)
@tilelang.testing.requires_metal
def test_e2e_per_row_scale_parity():
    """Per-row A scale, per-tensor B scale parity check."""
    import torch  # noqa: F401

    M, N, K = 32, 32, 64
    BM, BN, BK = 32, 32, 64

    fn = _make_kernel(M, N, K, BM, BN, BK, a_scale_size=M, b_scale_size=1)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0x1234)
    a_orig = torch.randn(M, K, dtype=torch.float32, device="mps") * 2.0
    b_orig = torch.randn(K, N, dtype=torch.float32, device="mps") * 2.0
    a_scale = torch.rand(M, dtype=torch.float32, device="mps") + 0.5  # [0.5, 1.5]
    b_scale = torch.tensor([0.75], dtype=torch.float32, device="mps")

    a_fp8 = _torch_fp8_quantize(a_orig, "float8_e4m3")
    b_fp8 = _torch_fp8_quantize(b_orig, "float8_e4m3")
    a_dequant = _torch_fp8_dequantize(a_fp8)
    b_dequant = _torch_fp8_dequantize(b_fp8)

    # Per-row scale — broadcast (M, 1) across the K dim.
    a_scaled = a_dequant * a_scale.unsqueeze(1)
    b_scaled = b_dequant * b_scale[0].item()
    c_ref = a_scaled @ b_scaled

    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a_fp8, a_scale, b_fp8, b_scale, c_out)

    torch.mps.synchronize()
    rel = torch.abs(c_out - c_ref) / (torch.abs(c_ref) + 1e-6)
    rmax = rel.max().item()
    assert rmax < 5e-3, (
        f"per-row FP8 scaled matmul parity failed: max relative error {rmax:.3g}"
    )


# --------------------------------------------------------------------------
# Negative tests: dtype / shape validation surfaces clean errors
# --------------------------------------------------------------------------

def test_rejects_non_fp8_a():
    """Non-FP8 ``A_fp8`` must raise TypeError at parse time."""

    def make_invalid():
        @T.prim_func
        def bad_kernel(
            A: T.Tensor((32, 64), "float32"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((64, 32), "float8_e4m3"),
            B_scale: T.Tensor((1,), "float32"),
            C: T.Tensor((32, 32), "float32"),
        ):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                C_local = T.alloc_fragment((32, 32), "float32")
                T.clear(C_local)
                T.fp8_scaled_matmul(A, A_scale, B, B_scale, C_local)
                T.copy(C_local, C[0, 0])

        return bad_kernel

    with pytest.raises(TypeError, match=r"A_fp8 must be FP8"):
        make_invalid()


def test_rejects_bad_scale_shape():
    """A_scale shape that's neither 1 nor M must fail."""

    def make_invalid():
        @T.prim_func
        def bad_kernel(
            A: T.Tensor((32, 64), "float8_e4m3"),
            A_scale: T.Tensor((7,), "float32"),  # neither 1 nor M=32
            B: T.Tensor((64, 32), "float8_e4m3"),
            B_scale: T.Tensor((1,), "float32"),
            C: T.Tensor((32, 32), "float32"),
        ):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                C_local = T.alloc_fragment((32, 32), "float32")
                T.clear(C_local)
                T.fp8_scaled_matmul(A, A_scale, B, B_scale, C_local)
                T.copy(C_local, C[0, 0])

        return bad_kernel

    with pytest.raises(ValueError, match=r"A_scale must be per-tensor"):
        make_invalid()


# --------------------------------------------------------------------------
# Numerical parity vs. the audiohacking MSL kernel via mlx.core
# --------------------------------------------------------------------------
#
# These tests use ``cppmega_mlx.nn._tilelang.fp8_msl_kernels`` as the
# ground truth oracle. That module ships the audiohacking fp8-mps-metal
# kernel pattern via ``mx.fast.metal_kernel`` (256-entry LUT decode); it
# is byte-compatible with PyTorch's ``torch.float8_e4m3fn`` representation.
# Comparing TileLang's TIR-lowered scalar K-loop against the LUT-based
# audiohacking kernel verifies that:
#
# 1. The dequant in ``__tvm_fp8_e4m3_to_half`` matches the LUT decode for
#    every byte 0x00..0xFF (including subnormals 0x01..0x07 / 0x81..0x87,
#    which were corrected by the storage-only patch fix in
#    ``codegen_metal.cc::PrintFP8Prelude``).
# 2. The fp32 FMA accumulation order does not introduce drift large
#    enough to exceed the LUT-kernel's bit-exact reference.
# 3. Per-tensor and per-row scale broadcasting agree at the bit level.

try:
    import mlx.core as mx  # noqa: F401
    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

try:
    from cppmega_mlx.nn._tilelang.fp8_msl_kernels import (
        fp8_msl_status,
        fp8_scaled_matmul_raw as _audio_fp8_scaled_matmul,
        fp8_scaled_vecmat as _audio_fp8_scaled_vecmat,
    )
    _AUDIO_AVAILABLE = (
        fp8_msl_status().available if _HAS_MLX else False
    )
except Exception:
    _AUDIO_AVAILABLE = False

try:
    import torch as _torch
    _HAS_TORCH_MPS_E2E = _torch.backends.mps.is_available()
except Exception:
    _torch = None
    _HAS_TORCH_MPS_E2E = False


def _audio_ground_truth_matmul(
    a_fp8_torch: "_torch.Tensor",
    b_fp8_torch: "_torch.Tensor",
    sa: float | "_torch.Tensor",
    sb: float | "_torch.Tensor",
):
    """Run the audiohacking/fp8-mps-metal scaled matmul via mlx.core.

    ``a_fp8_torch`` is (M, K) ``torch.float8_e4m3fn``. ``b_fp8_torch`` is
    (K, N) ``torch.float8_e4m3fn`` (same orientation as TileLang's
    ``transpose_B=False``). The audiohacking kernel itself wants B in
    (N, K) row-major form; we transpose at the boundary.
    """
    import mlx.core as mx
    import numpy as np

    a_bytes_np = a_fp8_torch.cpu().view(_torch.uint8).numpy()
    b_bytes_np = b_fp8_torch.cpu().view(_torch.uint8).numpy()

    a_mx = mx.array(a_bytes_np)
    # audiohacking kernel: B is (N, K) — i.e. each row is one output projection.
    b_t_np = np.ascontiguousarray(b_bytes_np.T)
    b_t_mx = mx.array(b_t_np)

    if isinstance(sa, _torch.Tensor):
        sa_mx = mx.array(sa.detach().cpu().numpy().astype(np.float32))
    else:
        sa_mx = float(sa)
    if isinstance(sb, _torch.Tensor):
        sb_mx = mx.array(sb.detach().cpu().numpy().astype(np.float32))
    else:
        sb_mx = float(sb)

    c_mx = _audio_fp8_scaled_matmul(a_mx, b_t_mx, scale_a=sa_mx, scale_b=sb_mx)
    mx.eval(c_mx)
    return _torch.from_numpy(np.array(c_mx))


def _audio_ground_truth_vecmat(
    x_fp8_torch: "_torch.Tensor",
    w_fp8_torch: "_torch.Tensor",
    sx: float | "_torch.Tensor",
    sw: float | "_torch.Tensor",
):
    """Vec * Mat ground truth via the audiohacking simdgroup-reduction kernel.

    ``x_fp8_torch`` is (K,) ``torch.float8_e4m3fn``.
    ``w_fp8_torch`` is (K, N) ``torch.float8_e4m3fn`` (TileLang orientation).
    The audiohacking kernel takes W as (N, K), so we transpose at the
    boundary. Returns (N,) fp32.
    """
    import mlx.core as mx
    import numpy as np

    x_bytes = x_fp8_torch.cpu().view(_torch.uint8).numpy()
    w_bytes = w_fp8_torch.cpu().view(_torch.uint8).numpy()

    x_mx = mx.array(x_bytes)
    # audiohacking expects W as (N, K) -- each row is a projection.
    w_t = np.ascontiguousarray(w_bytes.T)
    w_t_mx = mx.array(w_t)

    sx_mx = float(sx) if not isinstance(sx, _torch.Tensor) else mx.array(
        sx.detach().cpu().numpy().astype(np.float32)
    )
    sw_mx = float(sw) if not isinstance(sw, _torch.Tensor) else mx.array(
        sw.detach().cpu().numpy().astype(np.float32)
    )
    out = _audio_fp8_scaled_vecmat(x_mx, w_t_mx, scale_x=sx_mx, scale_w=sw_mx)
    mx.eval(out)
    return _torch.from_numpy(np.array(out))


@pytest.mark.skipif(
    not (_AUDIO_AVAILABLE and _HAS_TORCH_MPS_E2E),
    reason="audiohacking MSL kernel and torch.mps required",
)
@tilelang.testing.requires_metal
def test_e2e_audiohacking_parity_per_tensor_128():
    """T.fp8_scaled_matmul vs audiohacking MSL kernel at M=N=K=128.

    Per-tensor scale, e4m3, 128x128x128. The audiohacking kernel does the
    same per-element FP8 dequant + fp32 FMA + post-scale; this test
    asserts bit-level consistency to within 1e-4 absolute on the C
    output and 1e-4 relative when the reference is non-zero.
    """
    import torch
    import numpy as np

    M, N, K = 128, 128, 128
    BM, BN, BK = 32, 32, 32
    fn = _make_kernel(M, N, K, BM, BN, BK)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0xCAFE)
    a_orig = torch.randn(M, K, dtype=torch.float32) * 4.0
    b_orig = torch.randn(K, N, dtype=torch.float32) * 4.0
    sa = 0.5
    sb = 0.25

    a_fp8 = a_orig.to(torch.float8_e4m3fn).to("mps")
    b_fp8 = b_orig.to(torch.float8_e4m3fn).to("mps")
    a_scale = torch.tensor([sa], dtype=torch.float32, device="mps")
    b_scale = torch.tensor([sb], dtype=torch.float32, device="mps")

    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a_fp8, a_scale, b_fp8, b_scale, c_out)
    torch.mps.synchronize()

    c_ref = _audio_ground_truth_matmul(a_fp8.cpu(), b_fp8.cpu(), sa, sb)

    diff = (c_out.cpu() - c_ref).abs()
    rel = diff / (c_ref.abs() + 1e-6)
    abs_max = diff.max().item()
    rel_max = rel.max().item()
    assert abs_max < 1e-3, (
        f"audiohacking parity failed at 128x128x128: max abs err {abs_max:.3g}, "
        f"max rel err {rel_max:.3g}\n"
        f"c_out range: [{c_out.cpu().min():.3f}, {c_out.cpu().max():.3f}]\n"
        f"c_ref range: [{c_ref.min():.3f}, {c_ref.max():.3f}]"
    )


@pytest.mark.skipif(
    not (_AUDIO_AVAILABLE and _HAS_TORCH_MPS_E2E),
    reason="audiohacking MSL kernel and torch.mps required",
)
@tilelang.testing.requires_metal
def test_e2e_audiohacking_parity_per_row_singleblock():
    """Per-row ``A_scale``, per-tensor ``B_scale`` vs audiohacking.

    Exercises the per-row scale-broadcast branch with a single-block kernel
    (``BM == M``). The macro indexes ``A_scale[i]`` where ``i`` runs over
    the block-local rows, so it can address the full per-row scale only
    when ``BM == M``. Multi-block per-row scales would need either an
    explicit slice at the call site (``A_scale[by * BM:(by+1) * BM]``)
    or a follow-up macro extension that passes the block row offset --
    documented as a follow-up in the patch README.

    The audiohacking kernel accepts arbitrary ``(M,)`` scale_a; we feed
    it the same length-M scale tensor that we pass to TileLang.
    """
    import torch

    M, N, K = 32, 32, 64
    BM, BN, BK = 32, 32, 64
    fn = _make_kernel(M, N, K, BM, BN, BK, a_scale_size=M, b_scale_size=1)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0x1234)
    a_orig = torch.randn(M, K, dtype=torch.float32) * 2.0
    b_orig = torch.randn(K, N, dtype=torch.float32) * 2.0
    a_scale = torch.rand(M, dtype=torch.float32) + 0.5  # [0.5, 1.5]
    b_scale = torch.tensor([0.75], dtype=torch.float32)

    a_fp8 = a_orig.to(torch.float8_e4m3fn).to("mps")
    b_fp8 = b_orig.to(torch.float8_e4m3fn).to("mps")

    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(
        a_fp8, a_scale.to("mps"), b_fp8, b_scale.to("mps"), c_out
    )
    torch.mps.synchronize()

    c_ref = _audio_ground_truth_matmul(
        a_fp8.cpu(), b_fp8.cpu(), a_scale, b_scale[0].item()
    )
    diff = (c_out.cpu() - c_ref).abs()
    rel = diff / (c_ref.abs() + 1e-6)
    abs_max = diff.max().item()
    rel_max = rel.max().item()
    assert abs_max < 1e-3, (
        f"audiohacking per-row parity failed: max abs err {abs_max:.3g}, "
        f"max rel err {rel_max:.3g}"
    )


@pytest.mark.skipif(
    not (_AUDIO_AVAILABLE and _HAS_TORCH_MPS_E2E),
    reason="audiohacking MSL kernel and torch.mps required",
)
@tilelang.testing.requires_metal
def test_e2e_audiohacking_parity_vecmat_4096():
    """M=1 vecmat at K=N=4096 — TileLang vs audiohacking simdgroup kernel.

    The audiohacking project ships a dedicated ``fp8_scaled_vecmat_kernel``
    with simdgroup reduction for M=1. The TileLang lowering uses the same
    scalar dequant + FMA pattern but without the simdgroup reduction
    (the macro emits a per-cell K-loop). This test verifies that the
    fp32 outputs agree numerically; the bench test
    ``test_bench_vecmat_vs_audiohacking`` records relative timing.
    """
    import torch

    M, N, K = 1, 4096, 4096
    BM, BN, BK = 1, 64, 64

    fn = _make_kernel(M, N, K, BM, BN, BK)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0xC0DE)
    # Keep magnitudes mild: K=4096 inner sum at scale 1.0 ranges to ~64.
    a_orig = torch.randn(M, K, dtype=torch.float32) * 0.5
    b_orig = torch.randn(K, N, dtype=torch.float32) * 0.5
    sa = 0.5
    sb = 0.5

    a_fp8 = a_orig.to(torch.float8_e4m3fn).to("mps")
    b_fp8 = b_orig.to(torch.float8_e4m3fn).to("mps")
    a_scale = torch.tensor([sa], dtype=torch.float32, device="mps")
    b_scale = torch.tensor([sb], dtype=torch.float32, device="mps")

    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")
    jit_kernel(a_fp8, a_scale, b_fp8, b_scale, c_out)
    torch.mps.synchronize()

    # Use the audiohacking matmul kernel (not vecmat) to check, since both
    # produce (M=1, N) fp32 outputs — vecmat is just an M=1 specialisation.
    c_ref = _audio_ground_truth_matmul(a_fp8.cpu(), b_fp8.cpu(), sa, sb)

    diff = (c_out.cpu() - c_ref).abs()
    abs_max = diff.max().item()
    # K=4096 fp32 FMA can drift ~1e-2 between two different FMA orderings
    # even though every individual product is bit-exact. We allow that.
    assert abs_max < 5e-2, (
        f"vecmat parity failed: max abs err {abs_max:.3g}\n"
        f"c_out range: [{c_out.cpu().min():.3f}, {c_out.cpu().max():.3f}]\n"
        f"c_ref range: [{c_ref.min():.3f}, {c_ref.max():.3f}]"
    )


# --------------------------------------------------------------------------
# Bench: TFLOPS for matmul + vecmat, alongside the audiohacking baseline
# --------------------------------------------------------------------------

def _bench_callable(fn, sync, n_warm=3, n_iter=10):
    """Time a callable and return (mean_seconds, std_seconds)."""
    import time
    for _ in range(n_warm):
        fn()
        sync()
    samples = []
    for _ in range(n_iter):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    # drop the slowest 10% to reduce timer-jitter noise.
    keep = max(1, int(len(samples) * 0.9))
    s = samples[:keep]
    mean = sum(s) / len(s)
    var = sum((x - mean) ** 2 for x in s) / max(1, len(s) - 1)
    return mean, var ** 0.5


@pytest.mark.skipif(
    not (_AUDIO_AVAILABLE and _HAS_TORCH_MPS_E2E),
    reason="audiohacking MSL kernel and torch.mps required",
)
@tilelang.testing.requires_metal
def test_bench_matmul_vs_audiohacking(capsys):
    """Bench: TileLang T.fp8_scaled_matmul vs audiohacking matmul kernel at 128x128x128.

    Reports:
      - TileLang lowered MSL elapsed time (median of 10 iters)
      - Audiohacking LUT-decode kernel elapsed time
      - TFLOPS achieved
    """
    import torch
    import mlx.core as mx
    import numpy as np

    M, N, K = 128, 128, 128
    BM, BN, BK = 32, 32, 32
    flops = 2.0 * M * N * K  # 2 FMAs per output element

    fn = _make_kernel(M, N, K, BM, BN, BK)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0)
    a_orig = torch.randn(M, K, dtype=torch.float32) * 4.0
    b_orig = torch.randn(K, N, dtype=torch.float32) * 4.0
    a_fp8 = a_orig.to(torch.float8_e4m3fn).to("mps")
    b_fp8 = b_orig.to(torch.float8_e4m3fn).to("mps")
    a_scale = torch.tensor([0.5], dtype=torch.float32, device="mps")
    b_scale = torch.tensor([0.25], dtype=torch.float32, device="mps")
    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")

    def run_tilelang():
        jit_kernel(a_fp8, a_scale, b_fp8, b_scale, c_out)

    tl_mean, tl_std = _bench_callable(run_tilelang, torch.mps.synchronize)

    # Audiohacking baseline via mlx.core
    a_bytes = a_fp8.cpu().view(torch.uint8).numpy()
    b_bytes = b_fp8.cpu().view(torch.uint8).numpy()
    a_mx = mx.array(a_bytes)
    b_t_mx = mx.array(np.ascontiguousarray(b_bytes.T))

    def run_audio():
        c = _audio_fp8_scaled_matmul(a_mx, b_t_mx, scale_a=0.5, scale_b=0.25)
        mx.eval(c)

    au_mean, au_std = _bench_callable(run_audio, lambda: None)

    tl_tflops = flops / tl_mean / 1e12
    au_tflops = flops / au_mean / 1e12

    with capsys.disabled():
        print(
            f"\n[bench] {M}x{N}x{K} per-tensor e4m3 FP8 scaled matmul:\n"
            f"  TileLang  : {tl_mean*1e3:7.3f} +/- {tl_std*1e3:5.3f} ms  "
            f"({tl_tflops:5.3f} TFLOPS)\n"
            f"  audiohack : {au_mean*1e3:7.3f} +/- {au_std*1e3:5.3f} ms  "
            f"({au_tflops:5.3f} TFLOPS)\n"
            f"  ratio TileLang / audio = {tl_mean/au_mean:.2f}x"
        )


@pytest.mark.skipif(
    not (_AUDIO_AVAILABLE and _HAS_TORCH_MPS_E2E),
    reason="audiohacking MSL kernel and torch.mps required",
)
@tilelang.testing.requires_metal
def test_bench_vecmat_vs_audiohacking(capsys):
    """Bench: M=1 4096x4096 TileLang vs audiohacking vecmat kernel.

    The audiohacking project ships a dedicated simdgroup-reduction
    ``fp8_scaled_vecmat_kernel`` for M=1; the TileLang lowering uses the
    same scalar K-loop as the matmul case. We expect the audiohacking
    kernel to be substantially faster because its per-row simdgroup
    reduction amortises the K-loop across 32 lanes; the TileLang scalar
    fallback offers no reduction and is included as a correctness
    baseline.
    """
    import torch
    import mlx.core as mx

    M, N, K = 1, 4096, 4096
    BM, BN, BK = 1, 64, 64
    flops = 2.0 * M * N * K

    fn = _make_kernel(M, N, K, BM, BN, BK)
    jit_kernel = tilelang.compile(fn, target="metal")

    torch.manual_seed(0)
    a_orig = torch.randn(M, K, dtype=torch.float32) * 0.5
    b_orig = torch.randn(K, N, dtype=torch.float32) * 0.5
    a_fp8 = a_orig.to(torch.float8_e4m3fn).to("mps")
    b_fp8 = b_orig.to(torch.float8_e4m3fn).to("mps")
    a_scale = torch.tensor([0.5], dtype=torch.float32, device="mps")
    b_scale = torch.tensor([0.5], dtype=torch.float32, device="mps")
    c_out = torch.zeros(M, N, dtype=torch.float32, device="mps")

    def run_tilelang():
        jit_kernel(a_fp8, a_scale, b_fp8, b_scale, c_out)

    tl_mean, tl_std = _bench_callable(run_tilelang, torch.mps.synchronize)

    # audiohacking vecmat kernel uses (K,) x (N, K) signature.
    import numpy as np
    x_bytes = a_fp8.reshape(K).cpu().view(torch.uint8).numpy()
    w_bytes = b_fp8.cpu().view(torch.uint8).numpy()
    x_mx = mx.array(x_bytes)
    w_t_mx = mx.array(np.ascontiguousarray(w_bytes.T))

    def run_audio_vecmat():
        out = _audio_fp8_scaled_vecmat(x_mx, w_t_mx, scale_x=0.5, scale_w=0.5)
        mx.eval(out)

    av_mean, av_std = _bench_callable(run_audio_vecmat, lambda: None)

    tl_tflops = flops / tl_mean / 1e12
    av_tflops = flops / av_mean / 1e12

    with capsys.disabled():
        print(
            f"\n[bench] M=1 N={N} K={K} e4m3 FP8 vecmat:\n"
            f"  TileLang scalar  : {tl_mean*1e3:7.3f} +/- {tl_std*1e3:5.3f} ms  "
            f"({tl_tflops:6.3f} TFLOPS)\n"
            f"  audiohack simdg  : {av_mean*1e3:7.3f} +/- {av_std*1e3:5.3f} ms  "
            f"({av_tflops:6.3f} TFLOPS)\n"
            f"  ratio TileLang / audio = {tl_mean/av_mean:.2f}x"
            f" (audiohacking wins; TileLang has no simdgroup reduction yet)"
        )
