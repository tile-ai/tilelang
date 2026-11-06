"""Public C JIT executes dynamic pipelines with direct, non-identity layouts."""

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.target import Target

import tilelang
import tilelang.cpu.language as T
from tilelang.layout import Layout


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("size", [0, 1, 5])
def test_dynamic_pipeline_with_transposed_local_layout(late, size):
    n = T.dynamic("n", "int64")
    layout = Layout((2, 3), lambda row, col: (col, row))

    @T.prim_func
    def main(A: T.Tensor((n, 6), "int32"), B: T.Tensor((n, 6), "int32")):
        with T.Kernel(1):
            temporary = T.alloc_buffer((2, 3), "int32", scope="local")
            if not late:
                T.annotate_layout({temporary: layout})
            for k in T.Pipelined(n, num_stages=2):
                for i in T.serial(2):
                    for j in T.serial(3):
                        temporary[i, j] = A[k, i * 3 + j] + 1
                for i in T.serial(2):
                    for j in T.serial(3):
                        B[k, i * 3 + j] = temporary[i, j] * 3
            if late:
                T.annotate_layout({temporary: layout})

    calls = []
    tvm_ffi.structural_walk(main, (ir.Call, calls.append))
    declarations = [call for call in calls if call.op.same_as(ir.Op.get("tl.layout_map"))]
    assert len(declarations) == 1
    assert next(iter(declarations[0].attrs["layout_map"].values())).same_as(layout)
    compiled = tilelang.compile(main, out_idx=[1], target=Target("c"), target_host=Target("c"), execution_backend="tvm_ffi")
    source = torch.arange(size * 6, dtype=torch.int32).reshape(size, 6)
    output = compiled(source)
    torch.testing.assert_close(output, (source + 1) * 3, rtol=0, atol=0)
    assert output.shape == source.shape
