"""Regression tests for scalar C expression code generation."""

import torch

import tilelang
import tilelang.cpu.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tvm.target import Target


def test_cpu_scalar_min_max_use_updated_buffer_values():
    @T.prim_func
    def main(A: T.Tensor((2,), T.float32), B: T.Tensor((4,), T.float32)):
        with T.Kernel(1):
            B[0] = T.min(A[0], T.float32(0))
            B[1] = T.max(A[1], T.float32(0))
            A[0] = T.float32(1)
            A[1] = T.float32(-1)
            B[2] = T.min(A[0], T.float32(0))
            B[3] = T.max(A[1], T.float32(0))

    kernel = tilelang.compile(main, target="c", target_host="c", execution_backend="cython")
    a = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    b = torch.empty(4, dtype=torch.float32)

    kernel(a, b)

    torch.testing.assert_close(b, torch.tensor([-2.0, 2.0, 0.0, 0.0], dtype=torch.float32))


def test_c_host_scalar_min_rereads_updated_buffer():
    a = tirx.decl_buffer((1,), "float32", name="A")
    b = tirx.decl_buffer((2,), "float32", name="B")
    zero = tirx.FloatImm("float32", 0)
    body = tirx.SeqStmt(
        [
            tirx.BufferStore(b, tirx.Min(a[0], zero), [0]),
            tirx.BufferStore(a, tirx.FloatImm("float32", 1), [0]),
            tirx.BufferStore(b, tirx.Min(a[0], zero), [1]),
        ]
    )
    func = tirx.PrimFunc([a.data, b.data], body, buffer_map={a.data: a, b.data: b}).with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})

    source = tvm.ffi.get_global_func("target.build.tilelang_c_host")(mod, Target("c")).inspect_source()

    assert source.count("= A[0];") == 2


if __name__ == "__main__":
    tilelang.testing.main()
