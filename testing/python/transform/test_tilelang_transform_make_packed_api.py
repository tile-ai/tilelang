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
# ruff: noqa

import pytest
import numpy as np
import tilelang
from tvm_tirx.script import tirx as T
import tilelang.testing
import tvm_tirx
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.script import ir as I
from tilelang.testing.ir import collect_nodes, collect_calls
from tilelang.testing.packed import compile_packed_function


def _find_assignment(stmt, var_name):
    return next((node for node in collect_nodes(stmt, ir.Bind) if node.var.name == var_name), None)


def _find_compute_scope(func):
    return next((node for node in collect_nodes(func, ir.RegionStmt) if node.op.name == "tirx.compute_scope"), None)


def _collect_nodes(stmt, node_type):
    return collect_nodes(stmt, node_type)


def _marked_assume(condition, message):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
        {
            "node": condition,
            "key": "tl.assume_requires_runtime_check",
            "value": message,
        }
    )
    return ir.RegionStmt(
        tvm_tirx.ir.Op.get("tl.attr"),
        [payload],
        [],
        {"schema_version": 1, "annotation_schema": schema},
        [ir.Evaluate(ir.Call("prim.assume", [condition], ty="void"))],
        [],
    )


def _prepare_packed(function):
    # The native pipeline materializes checked markers at the host/device
    # boundary, before the packed binder. Keep that explicit pass contract.
    module = tilelang.transform.SplitHostDevice()(tvm_tirx.IRModule.from_expr(function))
    return tilelang.transform.MakePackedAPI()(module)


@pytest.mark.parametrize("use_global_symbol", [False])
def test_no_op_when_global_symbol_is_absent(use_global_symbol):
    func_attr = {"target": tvm_tirx.target.Target("llvm", host="llvm")}

    @T.function(private=True)
    def before():
        T.func_attr(func_attr)
        T.evaluate(0)

    if use_global_symbol:
        before = before.with_attr("global_symbol", "main")

    after = tilelang.transform.MakePackedAPI()(tvm_tirx.IRModule.from_expr(before))["main"]
    if use_global_symbol:
        assert len(after.params) == 4
    else:
        tvm_tirx.ir.assert_structural_equal(before, after)


def test_target_host_removed():
    """After MakePackedAPI, host-side target should be the host

    MakePackedAPI is the last transform that requires both the device
    and the host.  After MakePackedAPI, the target attribute should
    only contain the host-side target.
    """

    host = tvm_tirx.target.Target("llvm")

    @I.ir_module
    class before:
        @T.function
        def main(A: T.Tensor((1,), "float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("cuda", host=host)})
            T.evaluate(0)

    after = tilelang.transform.MakePackedAPI()(before)
    target_attr = after["main"].attrs["target"]
    assert str(host) == str(target_attr)


def test_internal_subroutine_call():
    """Internal subroutines should not use the PackedFunc API

    A subroutine without the "global_symbol" attribute is an internal
    subroutine, and is not directly exposed to a user of the generated
    `runtime.Module`.  Therefore, it doesn't need to follow the
    PackedFunc API.
    """

    @I.ir_module
    class before:
        @T.function
        def main(A: T.Tensor((1,), "float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            before.subroutine(A.data)

        # this test fails if it's made public
        @T.function(private=True)
        def subroutine(A_data: T.Ptr("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    after = tilelang.transform.MakePackedAPI()(before)
    tvm_tirx.ir.assert_structural_equal(before["subroutine"], after["subroutine"])

    compute_scope = _find_compute_scope(after["main"])
    subroutine_call_op = next(call.op for call in collect_nodes(compute_scope, ir.Call) if isinstance(call.op, tvm_tirx.ir.GlobalVar))
    assert isinstance(subroutine_call_op, tvm_tirx.ir.GlobalVar), (
        f"The main function's CallObj should use the subroutine's GLobalVar as the operation, "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_assume_requires_runtime_check_is_lowered_to_assert():
    n = ir.Var("n", "int32")
    condition = n % 4 == 0
    message = "n must be divisible by 4"
    # A checked user fact becomes available after its check. Native assume is
    # a sequential operation, so a duplicate optimizer-only fact follows it.
    body = [_marked_assume(condition, message), ir.Evaluate(ir.Call("prim.assume", [condition], ty="void"))]
    before = tirx.Function([n], body, ir.PrimType("void")).with_attr("global_symbol", "main")
    before = before.with_attr("target", tvm_tirx.target.Target("cuda", host="llvm"))
    after_mod = _prepare_packed(before)
    after = after_mod["main"]
    assert not [stmt for stmt in collect_nodes(after, ir.RegionStmt) if stmt.op.name == "tl.attr"]
    assert collect_calls(after, op="prim.assume")
    for function in [after, tilelang.transform.Simplify()(after_mod)["main"]]:
        matching_asserts = [
            stmt
            for stmt in collect_nodes(function, ir.AssertStmt)
            if any(isinstance(part, ir.StringImm) and part.value == message for part in stmt.message_parts)
        ]
        assert len(matching_asserts) == 1
        tvm_tirx.ir.assert_structural_equal(matching_asserts[0].condition, condition, map_free_vars=True)


def test_optimizer_only_assume_is_not_lowered_to_assert():
    n = ir.Var("n", "int32")
    condition = n > 0
    before = tirx.Function([n], [ir.Evaluate(ir.Call("prim.assume", [condition], ty="void"))], ir.PrimType("void"))
    before = before.with_attr("global_symbol", "main").with_attr("target", tvm_tirx.target.Target("cuda", host="llvm"))
    after = _prepare_packed(before)["main"]
    assert collect_calls(after, op="prim.assume")
    assert not [
        stmt
        for stmt in collect_nodes(after, ir.AssertStmt)
        if any(isinstance(part, ir.StringImm) and part.value.startswith("Assume:") for part in stmt.message_parts)
    ]


def test_subroutine_call_to_externally_visible_subroutine():
    """Externally-visible subroutines should use the PackedFunc API

    Because the subroutine may be called directly by a user, it must
    use the PackedFunc API.  Its signature should be updated to the
    PackedFunc signature, and call sites should be updated to use
    `T.tvm_call_cpacked`.
    """

    @I.ir_module
    class before:
        @T.function
        def main(A: T.Tensor((1,), "float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
            before.subroutine(A.data)

        @T.function
        def subroutine(A_data: T.Ptr("float32")):
            T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)

    after = tilelang.transform.MakePackedAPI()(before)

    main_compute_scope = _find_compute_scope(after["main"])
    assert main_compute_scope is not None
    subroutine_compute_scope = _find_compute_scope(after["subroutine"])
    assert subroutine_compute_scope is not None

    subroutine_call_op = collect_calls(main_compute_scope, op="tirx.call_cpacked")[0].op
    assert isinstance(subroutine_call_op, tvm_tirx.ir.Op) and subroutine_call_op.name == "tirx.call_cpacked", (
        f"The main function's CallObj should be lowered to the builtin 'tirx.call_cpacked', "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_function_call_with_wrong_argument_count():
    """Argument counts must be checked before accessing the type codes"""

    @T.function
    def func(
        A: T.Tensor([16, 16], "int32"),
        B: T.Tensor([16, 16], "int32"),
        C: T.Tensor([16, 16], "int32"),
        D: T.Tensor([16, 16], "int32"),
    ):
        pass

    built = compile_packed_function(func)

    with pytest.raises(TypeError, match="Expected 4 arguments"):
        built()


def test_function_call_with_wrong_type_code():
    """Type codes must be checked before accessing the arguments"""

    @T.function
    def func(A: T.Tensor([16, 16], "int32")):
        pass

    built = compile_packed_function(func)

    with pytest.raises(RuntimeError, match="Mismatched type"):
        built(0)


def test_function_call_with_null_data_pointer():
    """The data pointer must be checked before accessing the array"""

    @T.function
    def func(A: T.Tensor([16, 16], "int32"), B: T.Tensor([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

    built = compile_packed_function(func)

    A = tvm_tirx.runtime.tensor(np.zeros([16], dtype="int32"))
    B = tvm_tirx.runtime.empty([16, 16], "int32", tvm_tirx.cpu())

    with pytest.raises(RuntimeError, match="ndim"):
        built(A, B)


def test_function_call_with_wrong_dimensionality():
    """The dimensionality must be checked before validating the shape"""

    @T.function
    def func(A: T.Tensor([16, 16], "int32"), B: T.Tensor([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

    built = compile_packed_function(func)

    A = tvm_tirx.runtime.tensor(np.zeros([16], dtype="int32"))
    B = tvm_tirx.runtime.empty([16], "int32", tvm_tirx.cpu())

    with pytest.raises(RuntimeError, match="ndim"):
        built(A, B)


if __name__ == "__main__":
    tilelang.testing.main()
