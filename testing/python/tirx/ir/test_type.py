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
"""Test type nodes in the IR"""

import json
import uuid

import pytest
import tvm_ffi

import tvm_tirx
from tilelang._compat import ir as tlir
from tvm_tirx.runtime import MemoryScope


def check_json_roundtrip(node):
    json_str = tlir.save_json(node)
    back = tlir.load_json(json_str)
    tvm_tirx.ir.assert_structural_equal(back, node, map_free_vars=True)


def test_missing_type():
    missing = tvm_tirx.ir.Type.missing()

    assert isinstance(missing, tvm_tirx.ir.Type)
    assert isinstance(missing, tvm_tirx.ir.MissingType)
    none = tvm_tirx.ir.NoneType()
    assert isinstance(none, tvm_tirx.ir.Type)
    assert not tvm_ffi.structural_equal(none, missing)
    assert not tvm_ffi.structural_equal(none, tvm_tirx.ir.TupleType([]))
    check_json_roundtrip(none)


def test_prim_type():
    x = tvm_tirx.ir.PrimType("int32")
    assert isinstance(x, tvm_tirx.ir.PrimType)
    assert x.dtype == "int32"
    with pytest.raises(TypeError, match="PtrType::VoidPtrType"):
        tvm_tirx.ir.PrimType("handle")


@pytest.mark.parametrize(
    "encoding",
    [
        (0, 32, 1),
        (2, 16, 8),
        (4, 16, 16),
        (17, 4, 1),
        (17, 8, 1),
        (17, 8, 16),
        (2, 32, -4),
        (3, 0, 0),
    ],
)
def test_prim_type_json_preserves_dtype_encoding(encoding):
    primitive = tvm_tirx.ir.PrimType(tvm_ffi.dtype.from_dlpack_data_type(encoding))
    restored = tlir.load_json(tlir.save_json(primitive))
    assert restored.same_as(primitive)
    assert restored.dtype.type_code == encoding[0]
    assert restored.dtype.bits == encoding[1]
    assert restored.dtype.lanes == primitive.dtype.lanes
    check_json_roundtrip(tvm_tirx.ir.FuncType([primitive], primitive))
    if encoding[2] == 1:
        tensor_type = tvm_tirx.tirx.TensorType([16], primitive)
        check_json_roundtrip(tensor_type)
        parameter = tvm_tirx.ir.Var("source", tensor_type)
        check_json_roundtrip(tvm_tirx.tirx.Function([parameter], tvm_tirx.ir.Evaluate(0)))


@pytest.mark.parametrize("dtype", ["int32", "float32x4", "float4_e2m1fn"])
def test_prim_type_json_loads_legacy_dtype_strings(dtype):
    graph = json.loads(tlir.save_json(tvm_tirx.ir.PrimType(dtype)))
    graph["nodes"][graph["root_index"]]["type"] = "ir.PrimType"
    graph["nodes"][graph["root_index"]]["data"] = {"dtype": dtype}
    assert tlir.load_json(json.dumps(graph)).same_as(tvm_tirx.ir.PrimType(dtype))


@pytest.mark.parametrize(
    "encoding",
    [[2, 32], [-1, 32, 1], [256, 32, 1], [2, -1, 1], [2, 256, 1], [2, 32, -1], [2, 32, 65536]],
)
def test_prim_type_json_rejects_out_of_range_encoding(encoding):
    graph = json.loads(tlir.save_json(tvm_tirx.ir.PrimType("float32")))
    graph["nodes"][graph["root_index"]]["data"] = encoding
    with pytest.raises(ValueError, match="PrimType"):
        tlir.load_json(json.dumps(graph))


@pytest.mark.parametrize("bits", [4, 8])
def test_fp4_script_preserves_storage_width(bits):
    primitive = tlir.prim_type_from_dlpack((17, bits, 1))
    namespaces = {"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx, "tlir": tlir}
    for original in (primitive, tvm_tirx.ir.DataTypeImm(primitive.dtype)):
        restored = eval(tlir.script(original), namespaces)
        tvm_tirx.ir.assert_structural_equal(original, restored)
    tensor_type = tvm_tirx.tirx.TensorType([16], primitive)
    parameter = tvm_tirx.ir.Var("output", tensor_type)
    source = tvm_tirx.ir.Var("source", tensor_type)
    store = tvm_tirx.ir.TensorStore(parameter, [0], tvm_tirx.ir.TensorLoad(source, [0]))
    original = tvm_tirx.tirx.Function([source, parameter], store).with_attr("global_symbol", "main")
    restored = tvm_tirx.script.from_source(tlir.script(original), extra_vars=namespaces)
    tvm_tirx.ir.assert_structural_equal(original, restored)


def test_fp4_storage_widths_have_distinct_script_cache_keys():
    packed = tlir.prim_type_from_dlpack((17, 4, 1))
    unpacked = tlir.prim_type_from_dlpack((17, 8, 1))
    assert tlir.script(packed) != tlir.script(unpacked)


def test_func_type():
    arg_types = tvm_ffi.convert([])
    ret_type = tvm_tirx.ir.PrimType("float32")
    tf = tvm_tirx.ir.FuncType(arg_types, ret_type)
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert isinstance(tf.loc, tvm_tirx.ir.UnknownLoc)
    assert tf.loc.same_as(tvm_tirx.ir.UnknownLoc())
    # TODO make sure we can set loc
    str(tf)
    check_json_roundtrip(tf)


def test_tuple_type():
    tf = tvm_tirx.ir.FuncType([], tvm_tirx.ir.TupleType([]))
    tt = tvm_tirx.ir.PrimType("float32")
    fields = tvm_ffi.convert([tf, tt])

    tup_ty = tvm_tirx.ir.TupleType(fields)
    assert tup_ty.loc.same_as(tvm_tirx.ir.UnknownLoc())
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


def test_memory_scope_json_roundtrip():
    custom = MemoryScope.register("test.ir.scope." + uuid.uuid4().hex)
    for scope in (MemoryScope("global"), MemoryScope("shared.dyn"), custom):
        assert tlir.load_json(tlir.save_json(scope)).same_as(scope)


if __name__ == "__main__":
    test_func_type()
    test_tuple_type()
