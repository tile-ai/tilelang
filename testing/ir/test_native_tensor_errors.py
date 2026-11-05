"""Packed tensor diagnostics preserve the frontend's exception contract."""

import tvm_tirx

import numpy as np
import pytest
import torch
from tvm_tirx import ir, runtime, tirx

import tilelang
import tilelang.cpu.language as C
from tilelang.testing.packed import compile_packed_function


def _native_function():
    source = ir.Var("A", tirx.TensorType([4], "float32"))
    output = ir.Var("B", tirx.TensorType([4], "float32"))
    return tirx.Function([source, output], [ir.TensorStore(output, [0], ir.TensorLoad(source, [0]))]).with_attr("global_symbol", "main")


@pytest.mark.parametrize("tilelang_owned", [False, True])
def test_packed_tensor_error_kind_belongs_to_its_frontend(tilelang_owned):
    function = _native_function()
    executable = compile_packed_function(function, tilelang_owned=tilelang_owned)
    output = runtime.tensor(np.zeros(4, dtype="float32"))
    dtype_error = RuntimeError if tilelang_owned else TypeError
    value_error = RuntimeError if tilelang_owned else ValueError
    with pytest.raises(dtype_error, match="dtype"):
        executable(runtime.tensor(np.ones(4, dtype="int32")), output)
    with pytest.raises(value_error, match="ndim"):
        executable(runtime.tensor(np.ones((2, 2), dtype="float32")), output)
    with pytest.raises(value_error, match="shape"):
        executable(runtime.tensor(np.ones(3, dtype="float32")), output)
    # Packed value kinds and argument count are not tensor metadata errors.
    with pytest.raises(RuntimeError if tilelang_owned else TypeError):
        executable(3, output)
    with pytest.raises(TypeError):
        executable(output)
    executable(runtime.tensor(np.full(4, 7, dtype="float32")), output)
    assert output.numpy()[0] == 7


def test_tilelang_packed_signature_and_policies_are_self_contained():
    function = _native_function().with_attr("target", tvm_tirx.target.Target("c", host="c"))
    before = ir.save_json(function)
    packed = tilelang.transform.MakePackedAPI()(ir.IRModule({"main": function}))["main"]
    assert len(packed.params) == len(packed.ty.arg_types) == 4
    ir.assert_structural_equal(packed.ty.ret_type, ir.PrimType("int32"))
    for parameter, actual in zip(packed.params, packed.ty.arg_types):
        ir.assert_structural_equal(parameter.ty, actual)
    assert ir.save_json(function) == before
    assert not any(
        name in packed.attrs
        for name in (
            "tirx.tensor_argument_error_kind",
            "tirx.type_index_error_kind",
            "tirx.allow_null_unused_tensors",
            "tirx.allow_packed_subbyte_tensors",
            "tirx.allow_fp8_dtype_aliases",
        )
    )


def test_tilelang_argument_errors_preserve_user_assertion_kind():
    function = _native_function()
    body = [
        ir.AssertStmt(
            ir.TensorLoad(function.params[0], [0]) > 0,
            ir.StringImm("AssertionError"),
            [ir.StringImm("user assertion")],
        ),
        *function.body.seq,
    ]
    function = tirx.Function(function.params, body, function.ret_type, function.attrs)
    executable = compile_packed_function(function)
    with pytest.raises(AssertionError, match="user assertion"):
        executable(runtime.tensor(np.zeros(4, dtype="float32")), runtime.tensor(np.zeros(4, dtype="float32")))


@pytest.mark.parametrize("kind", ["llvm", "c"])
def test_tilelang_tensor_metadata_errors_match_original_runtimeerror(kind):
    if kind == "llvm" and not runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")

    @C.prim_func
    def main(A: C.Tensor((4,), "float32"), B: C.Tensor((4,), "float32")):
        with C.Kernel(1):
            for i in C.Parallel(4):
                B[i] = A[i]

    kernel = tilelang.compile(
        main,
        execution_backend="tvm_ffi",
        target={"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})},
        target_host={"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})},
    )
    output = torch.empty(4, dtype=torch.float32)
    for value, field in [
        (torch.ones(4, dtype=torch.int32), "dtype"),
        (torch.ones((2, 2), dtype=torch.float32), "ndim"),
        (torch.ones(3, dtype=torch.float32), "shape"),
        (torch.arange(8, dtype=torch.float32)[::2], "strides"),
    ]:
        with pytest.raises(RuntimeError, match=field):
            kernel(value, output)
    source = torch.arange(4, dtype=torch.float32)
    kernel(source, output)
    torch.testing.assert_close(output, source, rtol=0, atol=0)
