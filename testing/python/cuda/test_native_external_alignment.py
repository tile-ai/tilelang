"""Unknown external alignment and explicit vector promises are distinct contracts."""

import tvm_tirx

import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import collect_nodes


def _atomic_names(function):
    return [
        node.op.name for node in collect_nodes(function, ir.Call) if hasattr(node.op, "name") and node.op.name.startswith("tl.atomic_add")
    ]


@pytest.mark.parametrize("dtype,extent", [("float32", 4), ("float16", 2), ("bfloat16", 2)])
def test_default_tensor_atomic_plan_has_no_unproven_vector_alignment(dtype, extent):
    @T.prim_func
    def add(A: T.Tensor((extent,), dtype), B: T.Tensor((extent,), dtype)):
        for i in T.vectorized(extent):
            T.atomic_add(B[i], A[i])

    assert all(parameter.ty.data_alignment == 1 for parameter in add.params)
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"})
    module = tvm_tirx.IRModule({"main": add})
    with target:
        module = tirx.transform.BindTarget(target)(module)
        module = tilelang.transform.LowerAccessPtr()(module)
        module = tilelang.transform.FlattenBuffer()(module)
        module = tilelang.transform.VectorizeLoop()(module)
    assert _atomic_names(module["main"]) == ["tl.atomic_add_elem_op"]
    assert not tirx.analysis.undefined_vars(module["main"])


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("source_offset,output_offset", [(0, 0), (1, 3), (3, 1)])
def test_default_dsl_atomic_add_accepts_aligned_and_sliced_storage(dtype, source_offset, output_offset):
    extent = 4

    @T.prim_func
    def add(A: T.Tensor((extent,), dtype), B: T.Tensor((extent,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.vectorized(extent):
                T.atomic_add(B[i], A[i])

    kernel = tilelang.compile(add, target="cuda", execution_backend="tvm_ffi")
    torch_dtype = getattr(torch, dtype)
    source_storage = torch.arange(extent + 8, device="cuda", dtype=torch_dtype)
    source_before = source_storage.clone()
    output_storage = torch.full((extent + 8,), -16, device="cuda", dtype=torch_dtype)
    source = source_storage[source_offset : source_offset + extent]
    output = output_storage[output_offset : output_offset + extent]
    output.zero_()
    expected_storage = output_storage.clone()
    expected_storage[output_offset : output_offset + extent] = 32 * source
    kernel(source, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(output_storage, expected_storage, rtol=0, atol=0)
    torch.testing.assert_close(source_storage, source_before, rtol=0, atol=0)


def test_owned_allocation_and_explicit_alignment_keep_their_promises():
    @T.prim_func
    def copy(A: T.Buffer((4,), "float32", strides=(1,), align=16), B: T.Tensor((4,), "float32")):
        shared = T.alloc_shared((4,), "float32")
        for i in T.serial(4):
            shared[i] = A[i]
            B[i] = shared[i]

    assert copy.params[0].ty.data_alignment == 16
    assert copy.params[1].ty.data_alignment == 1
    allocations = [node for node in collect_nodes(copy, ir.Call) if hasattr(node.op, "name") and node.op.name == "tirx.alloc_tensor"]
    assert len(allocations) == 1
    assert allocations[0].ty.data_alignment == 64
