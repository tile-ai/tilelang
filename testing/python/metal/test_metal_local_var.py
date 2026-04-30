"""Focused Metal support tests for local.var scalar code generation."""

import re

import torch

import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing


def _make_local_var_func():
    @T.prim_func
    def local_var_kernel(A: T.Tensor((2,), T.int32)):
        with T.Kernel(1, threads=1) as _:
            x = T.alloc_var(T.int32, init=3)
            y = T.alloc_var(T.int32)
            y = x + 4
            A[0] = x
            A[1] = y

    return local_var_kernel


def test_metal_local_var_scalar_codegen_uses_thread_scalars():
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(_make_local_var_func(), target="metal")

    src = artifact.kernel_source
    assert src is not None
    assert "kernel void" in src

    # local.var should lower to scalar declarations/stores rather than arrays or
    # an unsupported storage scope.
    assert re.search(r"\bint\s+\w+\s*=\s*3;", src), src
    assert re.search(r"\bint\s+\w+\s*=\s*0;", src), src
    assert re.search(r"\w+\s*=\s*\(\w+ \+ 4\);", src), src
    assert "local.var" not in src
    assert "thread int" not in src


def test_metal_local_var_codegen_has_scalar_loads_for_outputs():
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(_make_local_var_func(), target="metal")

    src = artifact.kernel_source
    assert src is not None
    output_lines = [line.strip() for line in src.splitlines() if line.strip().startswith("A[")]
    assert len(output_lines) == 2, src
    assert all("[0]" not in line.split("=", 1)[1] for line in output_lines), output_lines


@tilelang.testing.requires_metal
def test_metal_local_var_runtime_scalar_load_store():
    kernel = tilelang.compile(_make_local_var_func(), target="metal")
    out = torch.empty(2, dtype=torch.int32, device="mps")

    kernel(out)
    torch.mps.synchronize()

    assert out.cpu().tolist() == [3, 7]
