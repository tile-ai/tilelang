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
from tilelang import tvm as tvm
import tvm
import tilelang.testing
from tvm import tir
from tvm.script import tir as T, ir as I


def _find_assignment(stmt, var_name):
    while not isinstance(stmt, tvm.tir.LetStmt):
        stmt = stmt.body

    if stmt.var.name != var_name:
        return _find_assignment(stmt.body, var_name)

    return stmt


def _find_compute_scope(func):
    result = None

    def _visitor(stmt):
        if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "compute_scope":
            nonlocal result
            result = stmt

    tir.stmt_functor.post_order_visit(func.body, _visitor)

    return result


@pytest.mark.parametrize("use_global_symbol", [False])
def test_no_op_when_global_symbol_is_absent(use_global_symbol):
    func_attr = {"target": tvm.target.Target("llvm", host="llvm")}

    @T.prim_func(private=True)
    def before():
        T.func_attr(func_attr)
        T.evaluate(0)

    if use_global_symbol:
        before = before.with_attr("global_symbol", "main")

    after = tilelang.transform.MakePackedAPI()(tvm.IRModule.from_expr(before))["main"]
    if use_global_symbol:
        assert len(after.params) == 4
    else:
        tvm.ir.assert_structural_equal(before, after)


def test_target_host_removed():
    """After MakePackedAPI, host-side target should be the host

    MakePackedAPI is the last transform that requires both the device
    and the host.  After MakePackedAPI, the target attribute should
    only contain the host-side target.
    """

    host = tvm.target.Target("llvm")

    @I.ir_module
    class before:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
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
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"target": T.target("llvm", host="llvm")})
            before.subroutine(A.data)

        # this test fails if it's made public
        @T.prim_func(private=True)
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A_data)

    after = tilelang.transform.MakePackedAPI()(before)
    tvm.ir.assert_structural_equal(before["subroutine"], after["subroutine"])

    compute_scope = _find_compute_scope(after["main"])
    subroutine_call_op = compute_scope.body.value.op
    assert isinstance(subroutine_call_op, tvm.ir.GlobalVar), (
        f"The main function's CallNode should use the subroutine's GLobalVar as the operation, "
        f"but instead has an operation of type {subroutine_call_op}"
    )


def test_subroutine_call_to_externally_visible_subroutine():
    """Externally-visible subroutines should use the PackedFunc API

    Because the subroutine may be called directly by a user, it must
    use the PackedFunc API.  Its signature should be updated to the
    PackedFunc signature, and call sites should be updated to use
    `T.tvm_call_cpacked`.
    """

    @I.ir_module
    class before:
        @T.prim_func
        def main(A: T.Buffer(1, "float32")):
            T.func_attr({"global_symbol": "main", "target": T.target("llvm", host="llvm")})
            before.subroutine(A.data)

        @T.prim_func
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)

    after = tilelang.transform.MakePackedAPI()(before)

    main_compute_scope = _find_compute_scope(after["main"])
    assert main_compute_scope is not None
    subroutine_compute_scope = _find_compute_scope(after["subroutine"])
    assert subroutine_compute_scope is not None

    subroutine_call_op = main_compute_scope.body.value.op
    assert isinstance(subroutine_call_op, tvm.ir.Op) and subroutine_call_op.name == "tir.tvm_call_cpacked", (
        f"The main function's CallNode should be lowered to the builtin 'tir.tvm_call_cpacked', "
        f"but instead has an operation of type {subroutine_call_op}"
    )


@tilelang.testing.requires_llvm
def test_function_call_with_wrong_argument_count():
    """Argument counts must be checked before accessing the type codes"""

    @T.prim_func
    def func(
        A: T.Buffer([16, 16], "int32"),
        B: T.Buffer([16, 16], "int32"),
        C: T.Buffer([16, 16], "int32"),
        D: T.Buffer([16, 16], "int32"),
    ):
        pass

    built = tvm.compile(func, target="llvm")

    with pytest.raises(tvm.TVMError):
        built()


@tilelang.testing.requires_llvm
def test_function_call_with_wrong_type_code():
    """Type codes must be checked before accessing the arguments"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32")):
        pass

    built = tvm.compile(func, target="llvm")

    with pytest.raises(tvm.TVMError):
        built(0)


@tilelang.testing.requires_llvm
def test_function_call_with_null_data_pointer():
    """The data pointer must be checked before accessing the array"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32"), B: T.Buffer([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

    built = tvm.compile(func, target="llvm")

    A = tvm.nd.array(np.zeros([16], dtype="int32"))
    B = tvm.nd.empty([16, 16], "int32", tvm.cpu())

    with pytest.raises(tvm.TVMError):
        built(A, B)


@tilelang.testing.requires_llvm
def test_function_call_with_wrong_dimensionality():
    """The dimensionality must be checked before validating the shape"""

    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32"), B: T.Buffer([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]

    built = tvm.compile(func, target="llvm")

    A = tvm.nd.array(np.zeros([16], dtype="int32"))
    B = tvm.nd.empty([16], "int32", tvm.cpu())

    with pytest.raises(tvm.TVMError):
        built(A, B)


if __name__ == "__main__":
    tilelang.testing.main()
