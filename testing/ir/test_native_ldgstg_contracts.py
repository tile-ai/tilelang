"""Explicit PTX access requires physical alignment and a uniform lazy guard."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.testing.ir import collect_calls, collect_nodes
from tilelang.transform import PassConfigKey
from tvm_tirx import ir


def _lower(function, *, vectorize=True):
    module = tvm.IRModule.from_expr(function.with_attr("global_symbol", "main").with_attr("tirx.noalias", True))
    module = tvm.tirx.transform.BindTarget(tvm.target.Target("cuda"))(module)
    if vectorize:
        module = tilelang.transform.FlattenBuffer()(module)
        module = tilelang.transform.VectorizeLoop()(module)
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
        }
    ):
        return tilelang.cuda.transform.LowerLDGSTG()(module)


@pytest.mark.parametrize(
    "offset,stride,base,alignment,eligible",
    [(0, 1, 0, 64, True), (4, 1, 0, 64, True), (1, 1, 0, 64, False), (0, 1, 1, 64, False), (0, 2, 0, 64, False), (0, 1, 0, 4, False)],
)
def test_ldgstg_requires_physical_alignment(offset, stride, base, alignment, eligible):
    @T.prim_func
    def function(
        A: T.Buffer((128,), "float32", strides=(stride,), elem_offset=offset, align=alignment),
        B: T.Buffer((128,), "float32", strides=(stride,), elem_offset=offset, align=alignment),
    ):
        B[T.Ramp(base, 1, 4)] = A[T.Ramp(base, 1, 4)]

    # This input already contains a vector transaction and a one-dimensional
    # tensor view. The pass must consume the view's offset and physical stride.
    lowered = _lower(function, vectorize=False)
    assert len(collect_calls(lowered, op="tl.ldg128")) == int(eligible)
    assert len(collect_calls(lowered, op="tl.stg128")) == int(eligible)
    if not eligible:
        assert any(load.ty == ir.PrimType("float32x4") for load in collect_nodes(lowered, ir.TensorLoad))


@pytest.mark.parametrize("condition_kind", ["lane", "read_state", "division"])
@pytest.mark.parametrize("statement", [False, True])
def test_nonuniform_or_stateful_condition_stays_scalar(condition_kind, statement):
    @T.prim_func
    def function(
        A: T.Buffer((128,), "float32", align=64),
        B: T.Buffer((128,), "float32", align=64),
        P: T.Buffer((1,), "int32", align=64),
        limit: T.int32,
    ):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                if condition_kind == "lane":
                    condition = j < limit
                elif condition_kind == "read_state":
                    condition = P[0] > 0
                else:
                    condition = T.floordiv(16, limit) > 0
                if statement:
                    with T.If(condition), T.Then():
                        B[i * 4 + j] = A[i * 4 + j]
                else:
                    B[i * 4 + j] = T.if_then_else(condition, A[i * 4 + j], T.float32(0))

    lowered = _lower(function)
    assert not collect_calls(lowered, op="tl.ldg128")
    assert not collect_calls(lowered, op="tl.stg128")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("predicated", [False, True])
def test_explicitly_aligned_copy_uses_ptx_and_runs(predicated):
    import torch

    @T.prim_func
    def function(A: T.Buffer((128,), "float32", align=64), B: T.Buffer((128,), "float32", align=64)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            for j in T.vectorized(4):
                if predicated:
                    B[tx * 4 + j] = T.if_then_else(tx < 16, A[tx * 4 + j], T.float32(0))
                else:
                    B[tx * 4 + j] = A[tx * 4 + j]

    kernel = tilelang.compile(
        function,
        target="cuda",
        pass_configs={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: predicated,
        },
    )
    values = torch.arange(128, device="cuda", dtype=torch.float32) / 8 - 4
    result = torch.full_like(values, float("nan"))
    assert values.data_ptr() % 64 == 0 and result.data_ptr() % 64 == 0
    kernel(values, result)
    expected = values.clone()
    if predicated:
        expected[64:] = 0
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    source = kernel.get_kernel_source()
    assert "load_global_128_conditional" in source if predicated else "load_global_128" in source
    assert "store_global_128" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("flag", [0, 1])
def test_uniform_if_else_retains_both_arms(flag):
    import torch

    @T.prim_func
    def function(A: T.Buffer((128,), "float32", align=64), B: T.Buffer((128,), "float32", align=64), pred: T.int32):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            for j in T.vectorized(4):
                if pred > 0:
                    B[tx * 4 + j] = A[tx * 4 + j] + 1
                else:
                    B[tx * 4 + j] = A[tx * 4 + j] - 1

    kernel = tilelang.compile(function, target="cuda", pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True})
    values = torch.arange(128, device="cuda", dtype=torch.float32)
    result = torch.full_like(values, float("nan"))
    kernel(values, result, flag)
    torch.testing.assert_close(result, values + (1 if flag else -1), atol=0, rtol=0)
    source = kernel.get_kernel_source()
    assert "load_global_128" in source and "store_global_128" in source
