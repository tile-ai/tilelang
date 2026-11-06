"""Persistent layout coordinate binders remain distinct from kernel captures."""

import json
import os
import subprocess
import sys

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.script import tirx as T

import tilelang
from tilelang.layout import Fragment, Layout, PartialFragment


KINDS = [
    "layout",
    "fragment",
    "partial",
    pytest.param(
        "ascend",
        marks=pytest.mark.skipif(
            tvm_ffi.get_global_func("tl.make_ascend_sf_layout", allow_missing=True) is None,
            reason="The compiler provider does not include Ascend layout support",
        ),
    ),
]


def make_layout(kind, n):
    shape = (n, 16, 32)
    if kind == "layout":
        return Layout(shape, lambda i, j, k: (i, j, (k + n) % 32))
    if kind == "ascend":
        return tilelang.layout.make_ascend_sf_layout(ir.Var("A", tirx.TensorType(shape, "uint8", scope="shared.l1")))
    factory = PartialFragment if kind == "partial" else Fragment
    return factory(
        shape,
        forward_index_fn=lambda i, j, k: (i, j),
        forward_thread_fn=lambda i, j, k, rep: k + rep * 32,
        replicate=2,
    )


def coords(layout):
    return [0, 1, 2, 3] if isinstance(layout, Fragment) else [1, 2, 3]


def concrete(expressions, n, value=8):
    expressions = tvm_ffi.structural_map(
        expressions,
        [(ir.Var, lambda var: ir.IntImm("int32", value) if var.same_as(n) else var)],
    )
    analyzer = sym.Analyzer()
    return [int(analyzer.simplify(expr)) for expr in expressions]


@pytest.mark.parametrize("kind", KINDS)
def test_layout_json_keeps_coordinate_binding(kind):
    n = ir.Var("i", "int32")  # Same spelling as an internal coordinate is harmless.
    original = make_layout(kind, n)
    restored = ir.load_json(ir.save_json(original))
    restored_n = restored.get_input_shape()[0]
    assert type(restored) is type(original)
    assert concrete(original.map_forward_index(coords(original)), n) == concrete(restored.map_forward_index(coords(restored)), restored_n)
    assert tvm_ffi.structural_equal(original, restored, map_free_vars=True)
    assert tvm_ffi.structural_hash(original, map_free_vars=True) == tvm_ffi.structural_hash(restored, map_free_vars=True)
    if isinstance(original, Fragment):
        assert concrete(original.map_forward_thread([1, 1, 2, 3]), n) == concrete(restored.map_forward_thread([1, 1, 2, 3]), restored_n)
    if isinstance(original, PartialFragment):
        assert int(restored.combine_size) == int(original.combine_size)
    if kind == "ascend":
        assert restored.kind == original.kind


def test_json_concrete_coordinate_regression():
    original = Layout((16,), lambda i: i + 1)
    restored = ir.load_json(ir.save_json(original))
    assert int(sym.Analyzer().simplify(restored.map_forward_index([3])[0])) == 4


@pytest.mark.parametrize("kind", KINDS)
def test_layout_json_inside_call_attrs_retains_external_identities(kind):
    n = ir.Var("n", "int32")
    tensor = ir.Var("A", tirx.TensorType((n, 16, 32), "uint8"))
    layout = make_layout(kind, n)
    # Any Call can carry the object: this test isolates persistence from the
    # production layout annotation op's own validation and transform hooks.
    call = ir.Call("tirx.assume_always", [ir.IntImm("bool", 1)], attrs={"layout_map": {tensor: layout}})
    before = tirx.Function([n, tensor], [ir.Evaluate(call)], ir.PrimType("void"))
    after = ir.load_json(ir.save_json(before))
    mapping = after.body.seq[0].value.attrs["layout_map"]
    key = next(iter(mapping))
    value = mapping[key]
    assert key.same_as(after.params[1])
    assert value.get_input_shape()[0].same_as(after.params[0])
    ir.assert_structural_equal(before, after)
    assert tvm_ffi.structural_hash(before) == tvm_ffi.structural_hash(after)


@pytest.mark.parametrize("kind", KINDS)
def test_layout_script_roundtrip(kind):
    original = make_layout(kind, ir.IntImm("int32", 8))
    # The standard IR script namespace and the explicit TileLang namespace
    # are enough; no private payload decoder or metadata side table is needed.
    script = original.script()
    assert "from_mapping" in script
    restored = eval(script, {"tilelang": tilelang, "T": T})
    assert type(restored) is type(original)
    ir.assert_structural_equal(original, restored)
    assert concrete(restored.map_forward_index(coords(restored)), ir.Var("unused", "int32")) == concrete(
        original.map_forward_index(coords(original)), ir.Var("unused", "int32")
    )


def test_json_recovery_in_new_process(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(ir.save_json(Layout((16,), lambda i: i + 1)))
    source = """
import sys
from tilelang import Layout
from tvm_tirx import ir, sym
# Register the owning Python wrapper before deserializing its reflected type.
# The root package deliberately keeps these wrappers lazy.
layout = ir.load_json(open(sys.argv[1]).read())
assert type(layout) is Layout
assert int(sym.Analyzer().simplify(layout.map_forward_index([3])[0])) == 4
"""
    subprocess.run([sys.executable, "-c", source, str(path)], check=True, env=os.environ.copy())


@pytest.mark.parametrize("kind", KINDS)
def test_script_recovery_in_new_process(kind, tmp_path):
    original = make_layout(kind, ir.IntImm("int32", 8))
    path = tmp_path / "layout.py"
    path.write_text(original.script())
    expected = concrete(original.map_forward_index(coords(original)), ir.Var("unused", "int32"))
    source = """
import json
import sys
import tilelang
from tvm_tirx import sym
from tvm_tirx.script import tirx as T

# Do not import backend layout modules here: the printed expression must
# resolve its own constructor through the ordinary TileLang namespace.
layout = eval(open(sys.argv[1]).read(), {"tilelang": tilelang, "T": T})
assert type(layout).__name__ == sys.argv[2]
coords = [0, 1, 2, 3] if isinstance(layout, tilelang.layout.Fragment) else [1, 2, 3]
actual = [int(sym.Analyzer().simplify(expr)) for expr in layout.map_forward_index(coords)]
assert actual == json.loads(sys.argv[3]), (actual, sys.argv[3])
"""
    subprocess.run(
        [sys.executable, "-c", source, str(path), type(original).__name__, json.dumps(expected)],
        check=True,
        env=os.environ.copy(),
    )


def test_reloaded_layout_algebra():
    layout = ir.load_json(ir.save_json(Layout((4, 8), lambda i, j: (j, i))))
    inverse = layout.inverse()
    assert list(map(int, inverse.map_forward_index([3, 2]))) == [2, 3]
    expanded = layout.expand([2])
    assert list(map(int, expanded.map_forward_index([1, 2, 3]))) == [1, 3, 2]
    reshaped = layout.reshape((2, 2, 8))
    assert len(reshaped.get_input_shape()) == 3
    assert all(int(x) > 0 for x in reshaped.get_output_shape())
