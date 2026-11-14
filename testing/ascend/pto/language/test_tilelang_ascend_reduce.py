"""Ascend collective reductions across explicit fragment layouts."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.backend.target import determine_target

DTYPE_MAP = {"float32": torch.float32, "int32": torch.int32, "int64": torch.int64}


def ref_program(A, op):
    return {"sum": A.sum(dim=1), "max": A.max(dim=1).values, "min": A.min(dim=1).values}[op]


def make_default_layout_reduce_kernel(rows, extent, dtype, op):
    # No annotate_layout: the default layout places each row on a consecutive
    # thread range, so one allreduce call covers `rows` reduction groups.
    @T.prim_func
    def kernel(A: T.Tensor((rows, extent), dtype), B: T.Tensor((rows,), dtype)):
        with T.Kernel(1) as _, T.SimtVF(threads=rows * extent):
            a = T.alloc_fragment((rows, extent), dtype)
            b = T.alloc_fragment((rows,), dtype)
            T.copy(A, a)
            getattr(T, f"reduce_{op}")(a, b, dim=1)
            T.copy(b, B)

    return kernel


@pytest.mark.pto
@pytest.mark.parametrize(
    "rows,extent,dtype,op",
    [
        # extent > 32: one allreduce call spans several cross-warp groups
        (2, 64, "int32", "sum"),
        (2, 64, "float32", "min"),
        (4, 64, "float32", "sum"),
        (2, 128, "int32", "max"),
    ],
)
@pytest.mark.parametrize("target", ["pto"])
def test_reduce_default_layout_multiple_groups(target, rows, extent, dtype, op):
    kernel = tilelang.compile(
        make_default_layout_reduce_kernel(rows, extent, dtype, op),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )

    device = torch.device("npu")
    td = DTYPE_MAP[dtype]
    a = (
        torch.randn(rows, extent, dtype=td, device=device)
        if td.is_floating_point
        else torch.randint(-100, 100, (rows, extent), dtype=td, device=device)
    )
    b = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(b, ref_program(a, op).to(b.dtype), rtol=0, atol=1e-3)


def make_multigroup_scale_kernel(rows, group_rows, extent, dtype, op):
    # Annotated layout: each group of `group_rows * extent` threads hosts `group_rows` interleaved rows (scale > 1).
    @T.prim_func
    def kernel(A: T.Tensor((rows, extent), dtype), B: T.Tensor((rows,), dtype)):
        with T.Kernel(1) as _, T.SimtVF(threads=rows * extent):
            a = T.alloc_fragment((rows, extent), dtype)
            b = T.alloc_fragment((rows,), dtype)
            T.copy(A, a)
            T.annotate_layout(
                {
                    a: T.Fragment(
                        (rows, extent),
                        forward_fn=lambda i, j: (
                            (i // group_rows) * group_rows * extent + j * group_rows + (i % group_rows),
                            0,
                        ),
                    )
                }
            )
            getattr(T, f"reduce_{op}")(a, b, dim=1)
            T.copy(b, B)

    return kernel


@pytest.mark.pto
@pytest.mark.parametrize(
    "rows,group_rows,extent",
    [
        # threads=256: 4 groups of 64, scale=16
        (64, 16, 4),
        # threads=512: 4 groups of 128, scale=16, serial cross-warp combine
        (64, 16, 8),
    ],
)
@pytest.mark.parametrize("target", ["pto"])
def test_reduce_multiple_groups_scale_greater_than_one(target, rows, group_rows, extent):
    kernel = tilelang.compile(
        make_multigroup_scale_kernel(rows, group_rows, extent, "int32", "sum"),
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )

    device = torch.device("npu")
    a = torch.arange(1, rows + 1, dtype=torch.int32, device=device).unsqueeze(1).expand(rows, extent).contiguous()
    b = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(b, ref_program(a, "sum").to(b.dtype), rtol=0, atol=1e-3)


@pytest.mark.pto
@pytest.mark.parametrize("op", ["bitand", "bitor", "bitxor"])
@pytest.mark.parametrize("target", ["pto"])
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
    with determine_target(target, return_object=True), pytest.raises(Exception, match="bitand, bitor, and bitxor are not supported"):
        tilelang.lower(kernel, target=target)


@pytest.mark.parametrize(
    "op,extent",
    [
        ("sum", 32),
        ("max", 32),
        ("min", 32),
        ("sum", 8),
        ("sum", 6),
        ("sum", 64),
    ],
)
@pytest.mark.pto
def test_pto_reducer_v2_int32(op, extent):
    @T.prim_func
    def kernel(A: T.Tensor((extent,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1) as _, T.SimtVF(threads=extent):
            partial = T.alloc_reducer((1,), "int32", op=op)
            T.reducer_init(partial)
            for i in T.Parallel(extent):
                T.reducer_update(partial[0], A[i])
            result = T.alloc_fragment((1,), "int32")
            T.finalize_reducer(partial, result)
            T.copy(result, B)

    compiled = tilelang.compile(kernel, target="pto", out_idx=-1)
    a = torch.randint(-100, 100, (extent,), dtype=torch.int32, device="npu")
    actual = compiled(a)
    torch.npu.synchronize()
    expected = {
        "sum": a.sum(dtype=torch.int32),
        "max": a.max(),
        "min": a.min(),
    }[op].reshape(1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.pto
def test_pto_reducer_v2_rejects_bfloat16_allreduce():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "bfloat16"), B: T.Tensor((1,), "bfloat16")):
        with T.Kernel(1) as _, T.SimtVF(threads=32):
            partial = T.alloc_reducer((1,), "bfloat16", op="sum")
            T.reducer_init(partial)
            for i in T.Parallel(32):
                T.reducer_update(partial[0], A[i])
            result = T.alloc_fragment((1,), "bfloat16")
            T.finalize_reducer(partial, result)
            T.copy(result, B)

    with determine_target("pto", return_object=True), pytest.raises(Exception, match="PTO cross-thread allreduce.*got bfloat16"):
        tilelang.lower(kernel, target="pto")


if __name__ == "__main__":
    tilelang.testing.main()
