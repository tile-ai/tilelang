import pytest
import torch

import tilelang
import tilelang.language as T
from tvm_tirx import tirx
from tvm_tirx.target import Target


@pytest.mark.parametrize("begin", [0, 2])
def test_native_dsl_block_init_predicate_and_scope(begin):
    @T.prim_func
    def function(A: T.Tensor((2, 3), "int32"), C: T.Tensor((2,), "int32")):
        T.func_attr({"tirx.noalias": True})
        for i in T.serial(2):
            for k in T.serial(begin, begin + 3):
                with T.sblock("sum"):
                    vi = T.axis.spatial(2, i)
                    vk = T.axis.reduce((begin, begin + 3), k)
                    T.reads(A[vi, vk - begin])
                    T.writes(C[vi])
                    T.where(vi < 1)
                    with T.init():
                        tmp = T.alloc_local((1,), "int32")
                        tmp[0] = 7
                        C[vi] = tmp[0]
                    C[vi] = C[vi] + A[vi, vk - begin]

    tirx.analysis.check_well_formed(function)
    target = Target("c")
    kernel = tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")
    inp = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    out = torch.full((2,), -100, dtype=torch.int32)
    kernel(inp, out)
    torch.testing.assert_close(out, torch.tensor([10, -100], dtype=torch.int32), atol=0, rtol=0)


def test_init_without_reduction_axes():
    @T.prim_func
    def function(C: T.Tensor((2,), "int32")):
        for i in T.serial(2):
            with T.sblock("pointwise"):
                vi = T.axis.spatial(2, i)
                with T.init():
                    tmp = T.alloc_local((1,), "int32")
                    tmp[0] = vi + 7
                    C[vi] = tmp[0]
                C[vi] = C[vi] + 1

    tirx.analysis.check_well_formed(function)
    target = Target("c")
    kernel = tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")
    out = torch.full((2,), -100, dtype=torch.int32)
    kernel(out)
    torch.testing.assert_close(out, torch.tensor([8, 9], dtype=torch.int32), atol=0, rtol=0)
