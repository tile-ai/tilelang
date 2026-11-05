"""Compile native CUDA shuffle construction, including packed reduction lanes."""

import pytest
import tvm_tirx

from testing.python.cuda.test_native_codegen_cuda import _nvcc_source, _raw_kernel, _source_only


def _shuffle_kernel(dtype, lanes):
    ir = tvm_tirx.ir
    # The emitter sees pointer parameters after native tensor flattening.
    pointer_type = ir.PtrType(ir.PrimType(dtype), "global")
    source_ptr = ir.Var("source_ptr", pointer_type)
    output_ptr = ir.Var("output_ptr", pointer_type)
    tensor_type = tvm_tirx.tirx.TensorType((lanes,), dtype, scope="global", layout="default")
    source = ir.Var("source", tensor_type)
    output = ir.Var("output", tensor_type)
    indices = [ir.const(index) for index in range(lanes)]
    shuffle = ir.prim.Shuffle([ir.TensorLoad(source, [index]) for index in indices], indices)
    packed = ir.Var("packed", shuffle.ty)
    body = [
        ir.Bind(source, ir.Call("tirx.decl_tensor", [source_ptr], ty_args=[tensor_type])),
        ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[tensor_type])),
        ir.Bind(packed, shuffle),
    ]
    for index in indices:
        body.append(ir.TensorStore(output, [index], ir.prim.Shuffle([packed], [index])))
    return _raw_kernel("shuffle_lanes", body, params=[source_ptr, output_ptr])


@pytest.mark.parametrize(
    "dtype,lanes,constructor",
    [
        ("float32", 2, "make_float2("),
        ("float32", 4, "make_float4("),
        ("int32", 2, "make_int2("),
        ("float16", 2, "tl::pack_half2("),
        ("bfloat16", 2, "uint1{__pack_nv_bfloat162("),
    ],
)
def test_native_cuda_shuffle_lanes_compile(dtype, lanes, constructor, tmp_path):
    function = _shuffle_kernel(dtype, lanes)
    module = _source_only({"shuffle_lanes": function})
    source = module.inspect_source("cuda")
    assert constructor in source
    (tmp_path / "input.json").write_text(tvm_tirx.ir.save_json(function))
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    binary = _nvcc_source(source, tmp_path, target)
    assert binary
