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

import tvm_ffi

import tvm_tirx.testing
from tvm_tirx import tirx
from tilelang._compat import ir, structural


def test_mixed_iter_type_detection_interrupts_walk():
    spatial = tirx.Var("spatial", "int32")
    reduction = tirx.Var("reduction", "int32")
    unreachable = tirx.Var("unreachable", "int32")
    visited = []

    def detect(var):
        visited.append(var)
        if var.same_as(reduction):
            return tvm_ffi.VisitInterrupt()
        return None

    result = structural.structural_walk(
        (spatial + reduction) + unreachable,
        (tirx.Var, detect),
        order="post",
    )

    assert isinstance(result, tvm_ffi.VisitInterrupt)
    assert any(var.same_as(spatial) for var in visited)
    assert any(var.same_as(reduction) for var in visited)
    assert not any(var.same_as(unreachable) for var in visited)


def test_region_body_metadata_tracks_definition_renewal():
    storage = ir.Var("storage", tirx.TensorType([4], "float32", scope="local"))
    allocation = ir.Bind(storage, ir.Call("tirx.alloc_tensor", [], ty_args=[storage.ty]))
    region = ir.RegionStmt("tl.scope", [], [], {"versions": {storage: 3}}, [allocation])
    function = tirx.Function([], [region], ret_type=ir.TupleType([]))
    restored = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(function, restored)
    assert ir.structural_hash(function) == ir.structural_hash(restored)
    renewed = ir.renew_def(function)
    new_region = renewed.body.seq[0]
    new_storage = new_region.body.seq[0].var
    assert not new_storage.same_as(storage)
    assert int(new_region.attrs["versions"][new_storage]) == 3
    ir.assert_structural_equal(function, renewed)
    assert ir.check_well_formed(renewed)


def test_index_map_metadata_has_local_parameters_and_lexical_captures():
    parameter = ir.Var("parameter", "int32")
    coordinate = ir.Var("coordinate", "int32")
    mapping = tirx.IndexMap([coordinate], [coordinate * 2 + parameter], None)
    region = ir.RegionStmt("tl.scope", [], [], {"maps": [mapping, mapping]}, [])
    function = tirx.Function([parameter], [region], ret_type=ir.TupleType([]))
    assert ir.check_well_formed(function)
    assert ir.check_well_formed(ir.load_json(ir.save_json(function)))
    assert ir.check_well_formed(ir.renew_def(function))
    missing_capture = tirx.Function([], [region], ret_type=ir.TupleType([]))
    assert not ir.check_well_formed(missing_capture, assert_mode=False)
    escaped_parameter = tirx.Function([parameter], [region, ir.Evaluate(coordinate)], ret_type=ir.TupleType([]))
    assert not ir.check_well_formed(escaped_parameter, assert_mode=False)


def test_function_and_loop_metadata_see_only_visible_definitions():
    parameter = ir.Var("parameter", "int32")
    local = ir.Var("local", "int32")
    index = ir.Var("index", "int32")
    body = ir.Bind(local, index + parameter)
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, body, {"locals": {local: index}})
    function = tirx.Function([parameter], [loop], ret_type=ir.TupleType([]))
    assert ir.check_well_formed(function.with_attr("parameter", parameter))
    assert not ir.check_well_formed(function.with_attr("escaped", local), assert_mode=False)
    defined = tirx.Function([parameter], ir.Bind(local, parameter + 1), ret_type=ir.TupleType([]))
    assert ir.check_well_formed(defined.with_attr("defined", {local: parameter}))


def test_metadata_equality_tracks_the_referenced_definition():
    def function(prefix, selected):
        first = ir.Var(prefix + "_first", "int32")
        second = ir.Var(prefix + "_second", "int32")
        body = [ir.Bind(first, ir.const(1)), ir.Bind(second, ir.const(2))]
        return tirx.Function([], body).with_attr("selected", [first, second][selected])

    original = function("original", 0)
    renamed = function("renamed", 0)
    wrong_reference = function("renamed", 1)
    assert ir.structural_equal(original, renamed)
    assert ir.structural_hash(original) == ir.structural_hash(renamed)
    assert not ir.structural_equal(original, wrong_reference)
    assert not ir.structural_equal(original, wrong_reference, map_free_vars=True)


def test_metadata_map_keys_are_rewritten_without_changing_the_owner():
    captured = ir.Var("captured", "int32")
    replacement = ir.Var("replacement", "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 4, 1, ir.ForKind.DEFAULT, [], {"keys": {captured: index}})
    mapped = structural.structural_map(loop, (ir.Var, lambda node: replacement if node.same_as(captured) else node))
    assert list(loop.annotations["keys"])[0].same_as(captured)
    assert list(mapped.annotations["keys"])[0].same_as(replacement)


def test_region_metadata_walk_propagates_early_interrupts():
    first = ir.Var("first", "int32")
    unreachable = ir.Var("unreachable", "int32")
    region = ir.RegionStmt("tl.scope", [], [], {"last": unreachable}, [ir.Evaluate(first)])
    visited = []

    def interrupt(variable):
        visited.append(variable)
        return tvm_ffi.VisitInterrupt()

    result = structural.structural_walk(region, (ir.Var, interrupt))
    assert isinstance(result, tvm_ffi.VisitInterrupt)
    assert len(visited) == 1 and visited[0].same_as(first)


if __name__ == "__main__":
    tvm_tirx.testing.main()
