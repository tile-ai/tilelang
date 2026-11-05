"""Backend helper consumers retain native tensor and primitive type contracts."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

from tilelang.ascend import language as T
from tilelang.cuda import language as C
from tilelang.testing.ir import collect_nodes


def test_manual_multi_buffer_uses_tensor_shape_and_explicit_count():
    @T.prim_func
    def kernel():
        with T.sblock("owner"):
            ring = T.alloc_shared((2, 32), "float32")
            flat = T.alloc_shared((96,), "float32")
            T.annotate_manual_multi_buffer(ring, {flat: 3})

    bindings = [node.var for node in collect_nodes(kernel, ir.Bind) if isinstance(node.var.ty, tirx.TensorType)]
    (annotation,) = [node for node in collect_nodes(kernel, ir.Call) if node.op.name == "tl.annotation"]
    payload = tvm_ffi.get_global_func("tl.DecodeAnnotations")(annotation.args[0], annotation.attrs["annotation_schema"])
    versions = payload["tl.manual_multi_buffer"]
    assert int(versions[bindings[0]]) == 2
    assert int(versions[bindings[1]]) == 3
    assert payload["tl.source_block_annotation"]
    assert tirx.analysis.check_well_formed(kernel)


def test_manual_multi_buffer_requires_a_source_annotation_owner():
    with pytest.raises(ValueError, match="outside of any T.sblock"):

        @T.prim_func
        def invalid():
            ring = T.alloc_shared((2, 32), "float32")
            T.annotate_manual_multi_buffer(ring)


def test_manual_multi_buffer_requires_static_default_count():
    dimension = ir.Var("dimension", "int32")
    tensor = ir.Var("ring", tirx.TensorType((dimension, 4), "float32", scope="shared"))
    with pytest.raises(ValueError, match="constant leading"):
        T.annotate_manual_multi_buffer(tensor)


@pytest.mark.parametrize("scope", ["local", "shared", "global"])
def test_ascend_print_reads_native_shape_and_scope(scope):
    @T.prim_func
    def kernel(input: T.Tensor((2, 3), "int32", scope=scope)):
        T.print(input, "values")

    (call,) = [node for node in collect_nodes(kernel, ir.Call) if node.op.name == "tirx.call_extern"]
    assert call.args[0].value == "debug_print_buffer_value"
    assert call.args[1].value == "values"
    assert isinstance(call.args[-1], ir.TensorLoad)
    assert len(call.args[-1].indices) == 2
    loops = collect_nodes(kernel, ir.For)
    assert len(loops) == 1 and int(loops[0].end) == 6
    assert tirx.analysis.check_well_formed(kernel)


@pytest.mark.parametrize("dtype", ["int32", "uint16", "bool"])
def test_ascend_simd_bitwise_reduce_accepts_integer_types(dtype):
    @T.prim_func
    def kernel():
        source = T.alloc_shared((4, 8), dtype)
        output = T.alloc_shared((4,), dtype)
        with T.SimdVF():
            T.reduce_bitand(source, output, dim=1)

    calls = [node for node in collect_nodes(kernel, ir.Call) if node.op.name == "tl.tileop.reduce"]
    assert len(calls) == 1
    assert calls[0].args[2].value == "bitand"
    assert tirx.analysis.check_well_formed(kernel)


def test_ascend_simd_bitwise_reduce_rejects_float_input():
    with pytest.raises(ValueError, match="integer/bool input buffer"):

        @T.prim_func
        def kernel():
            source = T.alloc_shared((4, 8), "float32")
            output = T.alloc_shared((4,), "float32")
            with T.SimdVF():
                T.reduce_bitand(source, output, dim=1)


def test_cuda_shared_print_keeps_all_thread_axis_predicates():
    @C.prim_func
    def kernel():
        with C.Kernel(1, threads=(64, 2, 2)):
            shared = C.alloc_shared((2, 3), "int32")
            C.print(shared, "one thread", warp_id=1)

    branches = collect_nodes(kernel, ir.If)
    assert len(branches) == 1
    # Macro arguments are explicit Bind statements in the native IR.
    bindings = {node.var: node.value for node in collect_nodes(kernel, ir.Bind)}
    condition = branches[0].condition
    while isinstance(condition, ir.Var) and condition in bindings:
        condition = bindings[condition]
    names = {var.name for var in collect_nodes(condition, ir.Var)}
    assert names == {"tx", "ty", "tz"}
    assert len(collect_nodes(condition, prim.And)) == 2
    assert tirx.analysis.check_well_formed(kernel)
