import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.engine.lower import lower
from tvm import tirx


@T.prim_func
def merging_assignment(
    A: T.Buffer((64,), "float32"),
    B: T.Buffer((64,), "float32"),
    C: T.Buffer((64,), "float32"),
):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        b_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)
        T.copy(B, b_ub)

        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_VL8")
            full = T.simd.pset(32)
            src = T.simd.vld(a_ub[0])
            dst = T.simd.alloc_local((1,), "float32")
            dst[0] = T.simd.vld(b_ub[0])
            dst[0] = T.simd.vadds(
                src,
                T.float32(1),
                mask,
                mode="MODE_MERGING",
            )
            T.simd.vsts(b_ub[0], dst[0], full)

        T.copy(b_ub, C)


@T.prim_func
def merging_without_mutable_destination(
    A: T.Buffer((64,), "float32"),
    C: T.Buffer((64,), "float32"),
):
    with T.Kernel(1):
        a_ub = T.alloc_shared((64,), "float32")
        c_ub = T.alloc_shared((64,), "float32")
        T.copy(A, a_ub)

        with T.SimdVF():
            mask = T.simd.pset(32, "PAT_VL8")
            full = T.simd.pset(32)
            src = T.simd.vld(a_ub[0])
            merged = T.simd.vadds(
                src,
                T.float32(1),
                mask,
                mode="MODE_MERGING",
            )
            T.simd.vsts(c_ub[0], merged, full)

        T.copy(c_ub, C)


def _as_module(func):
    return tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))


def test_legalize_simd_merging_marks_destination_read_write():
    after = transform.LegalizeSimdMerging()(_as_module(merging_assignment))["main"]
    merging_calls = []

    def collect(node):
        if isinstance(node, tirx.Call) and getattr(node.op, "name", None) == "tl.simd.vadds" and str(node.args[-1]) == '"MODE_MERGING"':
            merging_calls.append(node)

    tirx.stmt_functor.post_order_visit(after.body, collect)

    assert len(merging_calls) == 1
    call = merging_calls[0]
    assert call.dtype == tvm.DataType("")
    assert len(call.args) == 5
    destination = call.args[0]
    assert isinstance(destination, tirx.Call)
    assert destination.op.name == "tl.access_ptr"
    assert destination.args[1].value == 1
    assert destination.args[2].value == 3
    destination_load = destination.args[0]
    assert isinstance(destination_load, tirx.BufferLoad)
    assert destination_load.buffer.name == "dst"


def test_simd_merging_codegen_updates_existing_destination():
    source = lower(merging_assignment, target="ascend").kernel_source

    assert "::vadds(*((&(dst[0]))), src," in source
    assert "MODE_MERGING" in source
    assert "simd_inst::vadds(" not in source
    assert source.index("dst[0] = simd_inst::vlds") < source.index("::vadds") < source.index("simd_inst::vsts")


def test_simd_merging_requires_mutable_destination():
    with pytest.raises(tvm.error.InternalError, match="read-modify-write"):
        lower(merging_without_mutable_destination, target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
