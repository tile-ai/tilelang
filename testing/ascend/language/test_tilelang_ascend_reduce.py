"""Test Ascend reduce with explicit fragment layout + generated-code verification."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing

DTYPE_MAP = {"float32": torch.float32, "int32": torch.int32, "int64": torch.int64}
REDUCER = {"sum": "SumOp", "max": "MaxOp", "min": "MinOp"}


def make_reduce_kernel(extent, scale, dtype, op):
    threads = extent * scale

    @T.prim_func
    def kernel(
        A: T.Tensor((scale, extent), dtype),
        B: T.Tensor((scale,), dtype),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=threads):
            A_local = T.alloc_fragment((scale, extent), dtype)
            B_local = T.alloc_fragment((scale,), dtype)
            T.copy(A, A_local)
            T.annotate_layout(
                {
                    A_local: T.Fragment(
                        (scale, extent),
                        forward_fn=lambda g, e: (e * scale + g, 0),
                    )
                }
            )
            getattr(T, f"reduce_{op}")(A_local, B_local, dim=1)
            T.copy(B_local, B)

    return kernel


def ref_program(A, op):
    return {"sum": A.sum(dim=1), "max": A.max(dim=1).values, "min": A.min(dim=1).values}[op]


def _test_one(extent, scale, dtype, op):
    td = DTYPE_MAP[dtype]
    threads = extent * scale
    expected = f"AscendAllReduce<tl::{REDUCER[op]}, {threads}, {scale}, 0>::run"

    kernel = tilelang.compile(
        make_reduce_kernel(extent, scale, dtype, op),
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    assert expected in kernel.get_kernel_source(), f"Missing {expected}"

    device = torch.device("npu")
    a = (
        torch.randn(scale, extent, dtype=td, device=device)
        if td.is_floating_point
        else torch.randint(-100, 100, (scale, extent), dtype=td, device=device)
    )
    b = kernel(a)
    torch.npu.synchronize()

    max_diff = (b.float() - ref_program(a, op).float()).abs().max().item()
    assert max_diff < 1e-3, f"extent={extent} scale={scale} dtype={dtype} op={op}: max_diff={max_diff:.2e}"


@pytest.mark.parametrize(
    "extent,scale,dtype,op",
    [
        # float32  — hw_reduce + ShflXor
        (32, 1, "float32", "sum"),
        (8, 4, "float32", "sum"),
        (16, 2, "float32", "sum"),
        (8, 4, "float32", "max"),
        (8, 4, "float32", "min"),
        (64, 1, "float32", "sum"),
        (16, 4, "float32", "sum"),
        (6, 5, "float32", "sum"),
        (7, 3, "float32", "sum"),
        # int32 — hw_reduce + ShflXor
        (32, 1, "int32", "sum"),
        (8, 4, "int32", "sum"),
        (8, 4, "int32", "max"),
        (8, 4, "int32", "min"),
        (64, 1, "int32", "sum"),
        (6, 5, "int32", "sum"),
        # float16: commented out — bisheng ICE (Peephole segfault)
        # (32, 1, "float16", "sum"), (6, 5, "float16", "sum"), (8, 4, "float16", "max"),
        # bfloat16: commented out — bisheng ICE (Peephole segfault)
        # (32, 1, "bfloat16", "sum"), (8, 4, "bfloat16", "min"),
        # int64 — ShflXor only, no hw_reduce
        (32, 1, "int64", "sum"),
        (8, 4, "int64", "sum"),
    ],
)
def test_reduce(extent, scale, dtype, op):
    _test_one(extent, scale, dtype, op)


@pytest.mark.parametrize("coalesced_width", [None, 2])
def test_reducer_v2_multidim_narrow_plan(coalesced_width):
    width = None if coalesced_width is None else T.int32(coalesced_width)

    @T.prim_func
    def kernel(A: T.Tensor((8, 128), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1) as _, T.SimtVF(threads=128):
            partial = T.alloc_reducer((128,), "float32", op="sum")
            T.reducer_init(partial)
            for i, j in T.Parallel(8, 128, coalesced_width=width):
                T.reducer_update(partial[j], A[i, j])
            result = T.alloc_fragment((128,), "float32")
            T.finalize_reducer(partial, result)
            T.copy(result, B)

    compiled = tilelang.compile(
        kernel,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    source = compiled.get_kernel_source()
    if coalesced_width is None:
        # Scalar column ownership avoids unnecessary cross-thread reduction.
        assert "float partial[1];" in source
        assert "AscendAllReduce" not in source
    else:
        # An explicit vector width still takes priority over candidate search.
        assert "float partial[2];" in source
        assert "AscendAllReduce<tl::SumOp, 128, 64, 0>::run" in source

    a = torch.randn((8, 128), dtype=torch.float32, device="npu")
    actual = compiled(a)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, a.sum(dim=0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op", ["bitand", "bitor", "bitxor"])
@pytest.mark.parametrize("target", ["ascend"])
def test_reducer_v2_rejects_unsupported_bitwise_collectives(op, target):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1) as _, T.SimtVF(threads=32):
            partial = T.alloc_reducer((1,), "int32", op=op)
            T.reducer_init(partial)
            for i in T.Parallel(32):
                T.reducer_update(partial[0], A[i])
            result = T.alloc_fragment((1,), "int32")
            T.finalize_reducer(partial, result)
            T.copy(result, B)

    # tilelang.lower expects the caller to hold the target scope; the
    # vectorize planner consults Target.current().
    with tilelang.tvm.target.Target(target), pytest.raises(Exception, match="bitand, bitor, and bitxor are not supported"):
        tilelang.lower(kernel, target=target)


if __name__ == "__main__":
    tilelang.testing.main()
