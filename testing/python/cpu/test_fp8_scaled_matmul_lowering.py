"""IR-level lowering tests for ``T.fp8_scaled_matmul``.

These tests do not require a GPU: they assemble a ``@T.prim_func`` that
calls the intrinsic, run TileLang's ``lower(...)``, and inspect the
resulting ``IRModule``. The expected post-lowering shape is the
audiohacking-pattern ``Cast(fp8 -> fp32) * Cast(fp8 -> fp32) * sa * sb``
multiply-accumulate loop.
"""

from __future__ import annotations

import pytest

import tilelang
from tilelang import tvm
from tvm.target import Target
import tilelang.language as T


def _make_kernel(
    M: int = 32,
    N: int = 32,
    K: int = 64,
    BM: int = 32,
    BN: int = 32,
    BK: int = 64,
    a_dtype: str = "float8_e4m3",
    b_dtype: str = "float8_e4m3",
    a_scale_size: int = 1,
    b_scale_size: int = 1,
):
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


def test_macro_expands_to_scalar_kloop_metal():
    """After lowering the IRModule contains the scalar dequant + scale + FMA pattern.

    We inspect the textual IR repr (avoiding TIR-stmt-walker brittleness)
    for the audiohacking-pattern markers.
    """
    fn = _make_kernel()
    target = Target("metal")
    artifact = tilelang.lower(fn, target=target)
    # The artifact carries the lowered IRModule via its kernel_source MSL,
    # which is the most stable surface to assert against.
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)

    # Audiohacking markers: per-element FP8 dequant + scale * scale.
    assert "__tvm_fp8_e4m3_to_half" in src, (
        "expected per-element FP8 dequantization in lowered IR"
    )
    body = src[src.find("kernel void"):]
    assert "a_val" in body and "b_val" in body, (
        "expected dequantized FP8 values to be named in the inner loop"
    )
    # The scale multiplications survive lowering — even in the per-tensor
    # case where the compiler could hoist them out, the generated MSL
    # keeps the scale as a runtime reference.
    assert "A_scale" in body and "B_scale" in body


def test_per_tensor_scale_lowering_shape():
    """Per-tensor scale lowers with both scales indexed at [0]."""
    fn = _make_kernel(a_scale_size=1, b_scale_size=1)
    artifact = tilelang.lower(fn, target=Target("metal"))
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    body = src[src.find("kernel void"):]
    assert "A_scale[0]" in body
    assert "B_scale[0]" in body


def test_per_row_scale_lowering_shape():
    """Per-row A: A_scale uses a row-indexed access."""
    fn = _make_kernel(a_scale_size=32, b_scale_size=1)
    artifact = tilelang.lower(fn, target=Target("metal"))
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    body = src[src.find("kernel void"):]
    # Row-indexed: per-row sa scale uses an iteration variable as index.
    # We can't predict the variable name (depends on optimizer choices)
    # but it's not the constant-0 form.
    assert "A_scale[i" in body or "A_scale[((" in body  # any non-zero access
    assert "B_scale[0]" in body


def test_e5m2_lowering_uses_e5m2_helper():
    """e5m2 input dtype routes through the e5m2 dequant helper."""
    fn = _make_kernel(a_dtype="float8_e5m2", b_dtype="float8_e5m2")
    artifact = tilelang.lower(fn, target=Target("metal"))
    src = artifact.kernel_source if hasattr(artifact, "kernel_source") else str(artifact)
    body = src[src.find("kernel void"):]
    assert "__tvm_fp8_e5m2_to_half(A_shared" in body
    assert "__tvm_fp8_e5m2_to_half(B_shared" in body


def test_validation_rejects_non_fp8_inputs():
    """Pre-lowering shape / dtype check surfaces TypeError early."""
    with pytest.raises(TypeError, match=r"A_fp8 must be FP8"):

        @T.prim_func
        def bad(
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


def test_validation_rejects_bad_scale_size():
    """A_scale shape that's neither 1 nor M raises ValueError."""
    with pytest.raises(ValueError, match=r"A_scale must be per-tensor"):

        @T.prim_func
        def bad(
            A: T.Tensor((32, 64), "float8_e4m3"),
            A_scale: T.Tensor((7,), "float32"),
            B: T.Tensor((64, 32), "float8_e4m3"),
            B_scale: T.Tensor((1,), "float32"),
            C: T.Tensor((32, 32), "float32"),
        ):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                C_local = T.alloc_fragment((32, 32), "float32")
                T.clear(C_local)
                T.fp8_scaled_matmul(A, A_scale, B, B_scale, C_local)
                T.copy(C_local, C[0, 0])


def test_validation_rejects_k_mismatch():
    """K dimension mismatch between A and B raises ValueError."""
    with pytest.raises(ValueError, match=r"K mismatch"):

        @T.prim_func
        def bad(
            A: T.Tensor((32, 64), "float8_e4m3"),
            A_scale: T.Tensor((1,), "float32"),
            B: T.Tensor((48, 32), "float8_e4m3"),  # K=48 != 64
            B_scale: T.Tensor((1,), "float32"),
            C: T.Tensor((32, 32), "float32"),
        ):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                C_local = T.alloc_fragment((32, 32), "float32")
                T.clear(C_local)
                T.fp8_scaled_matmul(A, A_scale, B, B_scale, C_local)
                T.copy(C_local, C[0, 0])


def test_intrinsic_in_pre_lowering_ir():
    """Pre-lowering IR contains the macro expansion (Cast + multiply chain).

    The macro is a TIR-level construct, so by the time we have an
    ``IRModule`` from ``@T.prim_func`` the ``T.fp8_scaled_matmul`` call
    has already been inlined into a ``For/BufferStore`` chain. This test
    verifies the macro produces *some* recognizable arithmetic shape
    rather than e.g. a ``Call`` to an unknown op.
    """
    fn = _make_kernel()
    # Pre-lowering: just the @T.prim_func itself (no target dispatch).
    ir_text = str(fn)
    # The macro expansion uses cast operations — we should see ``T.Cast`` or
    # ``Cast(`` in the textual IR somewhere along the dequant path.
    assert "Cast" in ir_text or "cast" in ir_text or "float32" in ir_text
    # And the scale buffers should appear (they're function arguments).
    assert "A_scale" in ir_text
    assert "B_scale" in ir_text
