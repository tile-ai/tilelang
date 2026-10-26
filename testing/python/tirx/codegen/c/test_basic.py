# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.support import utils


def test_native_unit_return_is_void():
    import tvm_ffi

    from tvm_tirx import ir, target, tirx

    flag = ir.Var("flag", "int32")
    body = [
        ir.If(flag > 0, ir.Return(ir.Tuple([])), None),
        ir.Return(ir.Tuple([])),
    ]
    function = tirx.Function([flag], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "unit_return")
    source = tvm_ffi.get_global_func("tl.codegen.build_c")(ir.IRModule({"unit_return": function}), target.Target("c")).inspect_source()
    assert "void unit_return(" in source
    assert source.count("return;") == 2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_floating_constants_preserve_special_values_and_precision(dtype):
    from tvm_tirx import ir, runtime, tirx

    values = np.array(
        [np.inf, -np.inf, np.nan, 0.0, -0.0, np.pi, np.finfo(dtype).tiny, np.finfo(dtype).max],
        dtype=dtype,
    )
    output = ir.Var("output", tirx.TensorType((len(values),), dtype))
    body = ir.SeqStmt([ir.TensorStore(output, [index], ir.const(float(value), dtype)) for index, value in enumerate(values)])
    function = tirx.Function([output], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "floating_constants")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((len(values),), dtype)
    executable(actual)
    result = actual.numpy()
    np.testing.assert_array_equal(result, values)
    np.testing.assert_array_equal(np.signbit(result[3:5]), np.signbit(values[3:5]))


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_dynamic_loop_step_respects_direction(target):
    from tvm_tirx import ir, runtime, tirx

    if target == "llvm" and not runtime.enabled(target):
        pytest.skip("LLVM is not enabled")
    begin = ir.Var("begin", "int32")
    end = ir.Var("end", "int32")
    step = ir.Var("step", "int32")
    index = ir.Var("index", "int32")
    output = ir.Var("output", tirx.TensorType([16], "int32"))
    body = ir.For(index, begin, end, step, ir.ForKind.DEFAULT, ir.TensorStore(output, [index], index + 1))
    function = tirx.Function([output, begin, end, step], body, ret_type=ir.TupleType([]))
    executable = tirx.compile(function.with_attr("global_symbol", "dynamic_loop"), target=target)
    for first, last, increment in [(1, 11, 3), (9, -1, -3), (3, 7, -1), (8, 2, 1)]:
        actual = runtime.tensor(np.full(16, -99, dtype="int32"))
        executable(actual, first, last, increment)
        expected = np.full(16, -99, dtype="int32")
        for iteration in range(first, last, increment):
            expected[iteration] = iteration + 1
        np.testing.assert_array_equal(actual.numpy(), expected)


@pytest.mark.parametrize("target", ["c", "llvm"])
def test_allocation_without_optional_attributes(target):
    from tvm_tirx import ir, runtime, tirx

    if target == "llvm" and not runtime.enabled(target):
        pytest.skip(f"{target} is not enabled")
    output = ir.Var("output", tirx.TensorType((4,), "float32"))
    local_type = tirx.TensorType((4,), "float32", scope="local")
    temporary = ir.Var("temporary", local_type)
    allocation = ir.Bind(temporary, ir.Call("tirx.alloc_tensor", [], ty_args=[local_type]))
    initialize = [ir.TensorStore(temporary, [index], ir.const(index + 0.5, "float32")) for index in range(4)]
    copy = [ir.TensorStore(output, [index], ir.TensorLoad(temporary, [index])) for index in range(4)]
    body = ir.SeqStmt([allocation, *initialize, *copy])
    function = tirx.Function([output], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "allocation_without_attributes")
    executable = tirx.compile(function, target=target)
    actual = runtime.empty((4,), "float32")
    executable(actual)
    np.testing.assert_array_equal(actual.numpy(), np.arange(4, dtype="float32") + 0.5)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_host_pointers_do_not_require_device_scalar_types(dtype):
    from tvm_tirx import ir, runtime, tirx

    source = ir.Var("source", tirx.TensorType((4,), dtype))
    output = ir.Var("output", tirx.TensorType((1,), "uint64"))
    first = ir.Call("tirx.address_of", [ir.TensorLoad(source, [0])])
    second = ir.Call("tirx.address_of", [ir.TensorLoad(source, [1])])
    first_address = ir.Call("tirx.reinterpret", [first], ty="uint64")
    second_address = ir.Call("tirx.reinterpret", [second], ty="uint64")
    body = ir.TensorStore(output, [0], second_address - first_address)
    function = tirx.Function([source, output], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "host_pointer_storage")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((1,), "uint64")
    executable(runtime.empty((4,), dtype), actual)
    np.testing.assert_array_equal(actual.numpy(), [2])


@pytest.mark.parametrize("output_dtype", ["float16", "float32"])
def test_float16_loads_and_stores_use_numeric_values(output_dtype):
    from tvm_tirx import ir, runtime, tirx
    from tvm_tirx.ir import prim

    values = np.array([-3.5, -0.25, 0.0, 0.5, 17.0, np.finfo("float16").tiny], dtype="float16")
    source = ir.Var("source", tirx.TensorType((len(values),), "float16"))
    output = ir.Var("output", tirx.TensorType((len(values),), output_dtype))
    body = ir.SeqStmt(
        [
            ir.TensorStore(
                output,
                [index],
                prim.Cast(output_dtype, ir.TensorLoad(source, [index])) * ir.const(2, output_dtype),
            )
            for index in range(len(values))
        ]
    )
    function = tirx.Function([source, output], body, ret_type=ir.TupleType([]))
    function = function.with_attr("global_symbol", "numeric_half_storage")
    executable = tirx.compile(function, target="c")
    actual = runtime.empty((len(values),), output_dtype)
    executable(runtime.tensor(values), actual)
    np.testing.assert_array_equal(actual.numpy(), values.astype(output_dtype) * 2)


def test_add():
    nn = 1024

    @I.ir_module
    class Module:
        @T.function
        def test_fadd(
            A: T.Tensor((1024,), "float32"),
            B: T.Tensor((1024,), "float32"),
            C: T.Tensor((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                C[i0] = A[i0] + B[i0]

    def check_c():
        mhost = tvm_tirx.tirx.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm_tirx.runtime.load_module(path_dso)
        fadd = m["test_fadd"]
        dev = tvm_tirx.cpu(0)
        n = nn
        a = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        b = tvm_tirx.runtime.tensor(np.random.uniform(size=n).astype("float32"), dev)
        c = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fadd(a, b, c)
        tvm_tirx.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    check_c()


def test_reinterpret():
    nn = 1024

    @I.ir_module
    class Module:
        @T.function
        def test_reinterpret(
            A: T.Tensor((1024,), "int32"),
            B: T.Tensor((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                B[i0] = T.reinterpret(A[i0] + 2, ty="float32")

    def check_c():
        mhost = tvm_tirx.tirx.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm_tirx.runtime.load_module(path_dso)
        fadd = m["test_reinterpret"]
        dev = tvm_tirx.cpu(0)
        n = nn
        a = tvm_tirx.runtime.tensor(np.random.randint(-(2**30), 2**30, size=n).astype("int32"), dev)
        b = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fadd(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), (2 + a.numpy()).view("float32"))

    check_c()


def test_ceil():
    nn = 1024

    @I.ir_module
    class Module:
        @T.function
        def test_ceil(
            A: T.Tensor((1024,), "float32"),
            B: T.Tensor((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                B[i0] = T.ceil(A[i0])

    def check_c():
        mhost = tvm_tirx.tirx.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm_tirx.runtime.load_module(path_dso)
        fceil = m["test_ceil"]
        dev = tvm_tirx.cpu(0)
        n = nn
        a = tvm_tirx.runtime.tensor(np.random.rand(n).astype("float32"), dev)
        b = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fceil(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), (np.ceil(a.numpy()).view("float32")))

    check_c()


def test_floor():
    nn = 1024

    @I.ir_module
    class Module:
        @T.function
        def test_floor(
            A: T.Tensor((1024,), "float32"),
            B: T.Tensor((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                B[i0] = T.floor(A[i0])

    def check_c():
        mhost = tvm_tirx.tirx.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm_tirx.runtime.load_module(path_dso)
        ffloor = m["test_floor"]
        dev = tvm_tirx.cpu(0)
        n = nn
        a = tvm_tirx.runtime.tensor(np.random.rand(n).astype("float32"), dev)
        b = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        ffloor(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), (np.floor(a.numpy()).view("float32")))

    check_c()


def test_round():
    nn = 1024

    @I.ir_module
    class Module:
        @T.function
        def test_round(
            A: T.Tensor((1024,), "float32"),
            B: T.Tensor((1024,), "float32"),
        ):
            T.func_attr({"tirx.noalias": True})
            for i0 in range(1024):
                B[i0] = T.round(A[i0])

    def check_c():
        mhost = tvm_tirx.tirx.compile(Module, target="c")
        temp = utils.tempdir()
        path_dso = temp.relpath("temp.so")
        mhost.export_library(path_dso)
        m = tvm_tirx.runtime.load_module(path_dso)
        fround = m["test_round"]
        dev = tvm_tirx.cpu(0)
        n = nn
        # Exact midpoints first: this is where ties-to-even (np.round, and the
        # semantics every other backend and the constant folder use) differs
        # from ties-away-from-zero. np.random.rand never produces them, so the
        # random tail alone cannot exercise the tie rule.
        midpoints = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype="float32")
        a_np = np.concatenate([midpoints, np.random.rand(n - len(midpoints)).astype("float32")]).astype("float32")
        a = tvm_tirx.runtime.tensor(a_np, dev)
        b = tvm_tirx.runtime.tensor(np.zeros(n, dtype="float32"), dev)
        fround(a, b)
        tvm_tirx.testing.assert_allclose(b.numpy(), (np.round(a.numpy()).view("float32")))

    check_c()


def test_subroutine_call():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor(1, dtype="float32")):
            Module.subroutine(A.data)

        @T.function(private=True)
        def subroutine(A_data: T.handle("float32")):
            A = T.decl_tensor(A_data, ty_args=[T.Tensor(1, "float32", data=A_data)])
            A[0] = 42.0

    built = tvm_tirx.tirx.compile(Module, target="c").mod

    source = built.inspect_source()
    assert source.count("__tvm_ffi_main(void*") == 2, "Expected two occurrences, for forward-declaration and definition"
    assert source.count("subroutine(float*") == 2, "Expected two occurrences, for forward-declaration and definition"
    assert source.count("subroutine(") == 3, "Expected three occurrences, for forward-declaration, definition, and call from main."


def test_workspace_allocation_cast():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((256,), "float32")):
            workspace = T.alloc_tensor(ty_args=[T.Tensor((256,), "float32", scope="global")])
            for i in range(256):
                workspace[i] = A[i]
            for i in range(256):
                A[i] = workspace[i]

    built = tvm_tirx.tirx.compile(Module, target="c").mod
    assert "((float*)TVMBackendAllocWorkspace(" in built.inspect_source()

    temp = utils.tempdir()
    tvm_tirx.codegen.Executable(built).export_library(temp.relpath("workspace.so"))


def test_local_alloc_buffer_uses_plain_c_pointer():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((1,), "float32")):
            B = T.alloc_tensor(ty_args=[T.Tensor((1,), "float32", scope="local")])
            for i in range(1):
                B[i] = A[i] + T.float32(1)
                A[i] = B[i]

    built = tvm_tirx.tirx.compile(Module, target="c").mod
    assert "local float*" not in built.inspect_source()

    temp = utils.tempdir()
    path_dso = temp.relpath("local_alloc.so")
    tvm_tirx.codegen.Executable(built).export_library(path_dso)
    loaded = tvm_tirx.runtime.load_module(path_dso)

    data = tvm_tirx.runtime.tensor(np.array([1.0], dtype="float32"))
    loaded["main"](data)
    tvm_tirx.testing.assert_allclose(data.numpy(), np.array([2.0], dtype="float32"))


def test_vector_pointer_uses_byte_offset():
    buffer = tvm_tirx.tirx.Var("A", tvm_tirx.tirx.TensorType((8,), "float32x2"))
    pointer = tvm_tirx.tirx.ptr_byte_offset(buffer.data, 16, ty=buffer.data.ty)
    body = tvm_tirx.ir.Evaluate(tvm_tirx.tirx.call_extern("consume", pointer, ty="void"))
    func = tvm_tirx.tirx.Function([buffer], body).with_attr("global_symbol", "main")

    source = tvm_tirx.tirx.compile(tvm_tirx.IRModule.from_expr(func), target="c").mod.inspect_source()
    call = next(line.strip() for line in source.splitlines() if line.strip().startswith("consume("))
    assert "int32_t2" not in call
    assert "float2*" in call
    assert " + 16" in call


def test_if_then_else_avoids_extraneous_parentheses():
    @I.ir_module
    class Module:
        @T.function
        def main(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
            for i in range(8):
                B[i] = T.if_then_else(i == 0, 1, A[i])

    built = tvm_tirx.tirx.compile(Module, target="c").mod
    source = built.inspect_source()
    assert "if ((" not in source, (
        "Generated code contains extraneous parentheses in the if condition, which triggers clang's -Wparentheses-equality warning"
    )

    temp = utils.tempdir()
    path_dso = temp.relpath("if_then_else.so")
    tvm_tirx.codegen.Executable(built).export_library(path_dso)
    loaded = tvm_tirx.runtime.load_module(path_dso)

    a = tvm_tirx.runtime.tensor(np.arange(8, dtype="int32"))
    b = tvm_tirx.runtime.tensor(np.zeros(8, dtype="int32"))
    loaded["main"](a, b)
    tvm_tirx.testing.assert_allclose(b.numpy(), np.where(np.arange(8) == 0, 1, np.arange(8)).astype("int32"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
