"""PTO scalar int<->bool cast normalization.

TIR cast-to-bool means ``value != 0`` and cast-from-bool materializes a
normalized 0/1 integer.  The PTO codegen used to pass such operands through
raw, so ``int32(bool(x))`` kept ``x == 2`` as 2 and non-0/1 backing bytes of
bool buffers survived round-trips, diverging from the Ascend C++ backend
(``(int32_t)((bool)x)``) and from every other TileLang backend.  These tests
pin the device-side normalization: int/uint -> bool lowers to
``tl.as_logical_bool(value)`` and bool -> int/uint lowers to
``pto.select(cond, pto.const(1), pto.const(0))`` with immediates folded.
"""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.backend.target import determine_target
from tilelang.engine.lower import lower

N = 8
THREADS = 4


def _int_bool_int_kernel():
    @T.prim_func
    def main(A: T.Tensor((N,), "int32"), B: T.Tensor((N,), "int32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "int32")
            b_ub = T.alloc_shared((N,), "int32")
            T.copy(A, a_ub)
            with T.SimtVF(threads=THREADS):
                tx = T.get_thread_binding()
                for i in T.serial(N // THREADS):
                    b_ub[i * THREADS + tx] = T.cast(T.cast(a_ub[i * THREADS + tx], "bool"), "int32")
            T.copy(b_ub, B)

    return main


def _bool_roundtrip_kernel():
    @T.prim_func
    def main(A: T.Tensor((N,), "bool"), B: T.Tensor((N,), "bool")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "bool")
            b_ub = T.alloc_shared((N,), "bool")
            T.copy(A, a_ub)
            with T.SimtVF(threads=THREADS):
                tx = T.get_thread_binding()
                for i in T.serial(N // THREADS):
                    b_ub[i * THREADS + tx] = a_ub[i * THREADS + tx]
            T.copy(b_ub, B)

    return main


def _vectorized_bool_roundtrip_kernel():
    @T.prim_func
    def main(A: T.Tensor((N,), "bool"), B: T.Tensor((N,), "bool")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((N,), "bool")
            b_ub = T.alloc_shared((N,), "bool")
            T.copy(A, a_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(N):
                    b_ub[i] = a_ub[i]
            T.copy(b_ub, B)

    return main


def _pto_source(func):
    with determine_target("pto", return_object=True):
        return lower(func, target="pto").kernel_source


@pytest.mark.pto
def test_pto_int_to_bool_cast_normalizes_in_source():
    source = _pto_source(_int_bool_int_kernel())
    assert "tl.as_logical_bool(" in source
    assert "pto.select(tl.as_logical_bool(" in source
    assert "pto.const(1, dtype=pto.i32)" in source
    assert "pto.const(0, dtype=pto.i32)" in source
    compile(source, "<pto-int-bool-cast>", "exec")


@pytest.mark.pto
def test_pto_bool_store_normalizes_in_source():
    source = _pto_source(_bool_roundtrip_kernel())
    assert "tl.as_logical_bool(" in source
    assert "pto.const(1, dtype=pto.i8)" in source
    assert "pto.const(0, dtype=pto.i8)" in source
    compile(source, "<pto-bool-roundtrip>", "exec")


@pytest.mark.pto
def test_pto_vectorized_bool_roundtrip_stays_passthrough():
    # Vector bool casts share the flattened int8 lane storage (bool is a
    # same-width 8-bit type), so the vectorized round-trip must keep the raw
    # contiguous load/store instead of per-lane normalization.
    source = _pto_source(_vectorized_bool_roundtrip_kernel())
    assert "contiguous=" in source
    assert "pto.select(" not in source
    assert "tl.as_logical_bool(" not in source
    compile(source, "<pto-vector-bool-roundtrip>", "exec")


@pytest.mark.pto
def test_pto_int_bool_int_matches_ascend_backend():
    A = torch.tensor([2, -3, 0, 7, 1, 100, -1, 0], dtype=torch.int32, device="npu")
    ref = (A != 0).to(torch.int32)
    for target in ("ascend", "pto"):
        kernel = tilelang.compile(_int_bool_int_kernel(), target=target)
        B = torch.zeros(N, dtype=torch.int32, device="npu")
        kernel(A, B)
        torch.npu.synchronize()
        assert torch.equal(B.cpu(), ref.cpu())


@pytest.mark.pto
def test_pto_bool_roundtrip_normalizes_backing_bytes():
    # Bind an int8 tensor with non-0/1 bytes through a bool parameter; both
    # backends must normalize the stored bytes to 0/1.
    raw = torch.tensor([2, -1, 0, 3, -128, 1, 0, 5], dtype=torch.int8, device="npu")
    A = raw.view(torch.bool)
    ref = (raw != 0).to(torch.int8)
    for target in ("ascend", "pto"):
        kernel = tilelang.compile(_bool_roundtrip_kernel(), target=target)
        B = torch.zeros(N, dtype=torch.bool, device="npu")
        kernel(A, B)
        torch.npu.synchronize()
        assert torch.equal(B.view(torch.int8).cpu(), ref.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
