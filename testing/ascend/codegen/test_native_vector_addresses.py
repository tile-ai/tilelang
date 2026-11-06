"""Vector addresses preserve lane order and source-width cast semantics."""

import ast
import re
import shutil
import subprocess
import types

import pytest
from tvm_tirx import ir, prim, tirx

from testing.ascend.codegen import test_native_codegen_ascend as cce
from testing.ascend.codegen import test_native_codegen_pto as pto


@pytest.mark.parametrize("backend", [cce, pto], ids=["cce", "pto"])
@pytest.mark.parametrize("kind", ["cast", "shuffle", "call"])
def test_native_vector_indices_reach_memory_without_ramp_recovery(backend, kind, tmp_path):
    x = ir.Var("offset", "uint32")
    ramp = prim.Ramp(x, ir.const(1, "uint32"), 4)
    if kind == "cast":
        index = prim.Cast("int64x4", prim.Cast("uint8x4", ramp))
    elif kind == "shuffle":
        index = prim.Shuffle([ramp], [3, 1, 3, 0])
    else:
        index = prim.Ramp(tirx.call_extern("next_offset", ty="int32"), 2, 4)
    a = backend._tensor("source", scope="shared.dyn", shape=(256,))
    b = backend._tensor("destination", scope="shared.dyn", shape=(256,))
    ap = backend._ptr("ap", scope="shared.dyn")
    bp = backend._ptr("bp", scope="shared.dyn")
    snapshot = ir.save_json(index)
    body = [backend._decl(a, ap), backend._decl(b, bp), ir.TensorStore(b, [index], ir.TensorLoad(a, [index]))]
    code = backend._source([ap, bp, x], body, tmp_path)
    assert ir.save_json(index) == snapshot
    if kind == "call":
        # One evaluation for the load address, one for the store address.
        assert code.count("next_offset()") == 2
    if backend is cce:
        assert "int64x4" not in code
        assert len(re.findall(r"bp\[[^\n]+\] =", code)) == 4
        if kind == "cast":
            assert "uint8_t" in code and "int64_t" in code
    else:
        assert "tl.vector_to_list(" in code
        assert code.count("pto.load(") == 4
        assert code.count("pto.store(") == 4
        assert "contiguous=" not in code
        if kind == "cast":
            assert "pto.cast(lane, pto.ui32), pto.ui8)" in code
            assert "pto.cast(lane, pto.ui8), pto.i64)" in code


@pytest.mark.parametrize("backend", [cce, pto], ids=["cce", "pto"])
def test_address_of_vector_uses_first_lane_without_loading(backend, tmp_path):
    a = backend._tensor("source", scope="shared.dyn", shape=(256,))
    ap = backend._ptr("ap", scope="shared.dyn")
    pointer = backend._ptr("result", scope="shared.dyn")
    index = prim.Shuffle([prim.Ramp(4, 1, 4)], [2, 0, 3, 1])
    code = backend._source([ap], [backend._decl(a, ap), ir.Bind(pointer, tirx.address_of(a[index]))], tmp_path)
    assert "pto.load(" not in code
    if backend is pto:
        assert "pto.addptr(" in code
    else:
        assert "&(ap[" in code


@pytest.mark.parametrize("backend", [cce, pto], ids=["cce", "pto"])
@pytest.mark.parametrize("direction", ["load", "store"])
def test_vector_element_index_flattens_each_slot(backend, direction, tmp_path):
    a = backend._tensor("a", "float32x2", "local", (8,))
    b = backend._tensor("b", scope="shared.dyn", shape=(32,))
    bp = backend._ptr("bp", scope="shared.dyn")
    index = prim.Ramp(1, 2, 2)
    contiguous = prim.Ramp(0, 1, 4)
    value = ir.TensorLoad(a if direction == "load" else b, [index if direction == "load" else contiguous])
    assert value.ty == ir.PrimType("float32x4")
    op = ir.TensorStore(b if direction == "load" else a, [contiguous if direction == "load" else index], value)
    tx = ir.Var("tx", "int32")
    body = [backend._decl(b, bp), backend._block("SIMT_VF", [backend._thread(tx, 1, [backend._alloc(a), op])])]
    # A validated native tensor access yields four scalar lanes from two slots.
    # The old CCE implementation failed with `indices.size() == lanes (2 vs. 4)`.
    source = backend._source([bp], body, tmp_path)
    assert source


def _execute_pto_expression(source, *args, extra_globals=None):
    """Execute generated expression code, checking Python scope and call order only."""
    module = ast.parse(source)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef))
    function.decorator_list = []
    for arg in function.args.args:
        arg.annotation = None
    function.returns = None
    function.body.append(ast.Return(ast.Name("answer", ast.Load())))
    module.body = [function]
    ast.fix_missing_locations(module)

    def load(pointer, index, contiguous=None):
        if not isinstance(pointer, (list, tuple)):
            raise AssertionError("index tensor read outside the selected branch")
        if contiguous:
            return tuple(pointer[index : index + contiguous])
        return pointer[index]

    fake_pto = types.SimpleNamespace(
        init_core=lambda: None,
        cast=lambda value, dtype: value,
        load=load,
        Vec=lambda dtype, lanes, init: tuple(init) if isinstance(init, (list, tuple)) else (init,) * lanes,
        i32="i32",
        f32="f32",
    )
    fake_tl = types.SimpleNamespace(
        vector_from_list=lambda dtype, values: tuple(values),
        vector_to_list=list,
        if_then_else=lambda cond, then, otherwise: then() if cond else otherwise(),
        as_logical_bool=bool,
    )
    env = {"pto": fake_pto, "tl": fake_tl, **(extra_globals or {})}
    exec(compile(module, "<generated PTO source, expression semantics only>", "exec"), env)
    return env[function.name](*args)


def test_pto_gather_keeps_let_variable_in_scope_and_evaluates_value_once(tmp_path):
    a = pto._tensor("a", scope="shared.dyn", shape=(32,))
    ap = pto._ptr("ap", scope="shared.dyn")
    offset = ir.Var("let_offset", "int32")
    value = prim.Let(offset, tirx.call_extern("next_offset", ty="int32"), ir.TensorLoad(a, [prim.Ramp(offset, 2, 4)]))
    answer = ir.Var("answer", "float32x4")
    source = pto._source([ap], [pto._decl(a, ap), ir.Bind(answer, value)], tmp_path)
    calls = []

    def next_offset():
        calls.append("next_offset")
        return 3

    assert _execute_pto_expression(source, list(range(32)), extra_globals={"next_offset": next_offset}) == (3, 5, 7, 9)
    assert calls == ["next_offset"]


def test_pto_lazy_gather_does_not_read_unselected_index_tensor(tmp_path):
    a = pto._tensor("a", scope="shared.dyn", shape=(32,))
    ap = pto._ptr("ap", scope="shared.dyn")
    indices = pto._tensor("indices", "int32", "shared.dyn", (4,))
    ip = pto._ptr("ip", "int32", "shared.dyn")
    take = ir.Var("take", "bool")
    gather = ir.TensorLoad(a, [ir.TensorLoad(indices, [prim.Ramp(0, 1, 4)])])
    choice = ir.Call("prim.if_then_else", [take, gather, prim.Broadcast(ir.const(0, "float32"), 4)], ty=ir.PrimType("float32x4"))
    answer = ir.Var("answer", "float32x4")
    source = pto._source([ap, ip, take], [pto._decl(a, ap), pto._decl(indices, ip), ir.Bind(answer, choice)], tmp_path)
    assert _execute_pto_expression(source, list(range(32)), object(), False) == (0, 0, 0, 0)
    assert _execute_pto_expression(source, list(range(32)), [3, 1, 6, 2], True) == (3, 1, 6, 2)


_CPP_HEADER = r"""
#include <cstdint>
#include <cstdio>
#define __global__
#define __vector__
#define __ubuf__
#define __gm__
inline void asc_init() {}
struct alignas(8) float2 { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };
"""


def _run_generated_cce(source, main, tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the generated-address host oracle")
    source = "\n".join(line for line in source.splitlines() if not line.startswith("#include"))
    cpp = tmp_path / "check.cc"
    cpp.write_text(_CPP_HEADER + source + main)
    command = [
        compiler,
        "-std=c++17",
        "-O1",
        "-fno-strict-aliasing",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        str(cpp),
        "-o",
        str(tmp_path / "check"),
    ]
    built = subprocess.run(command, capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(tmp_path / "check")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stderr


def test_cce_broadcast_store_preserves_neighbor_slots(tmp_path):
    local = cce._tensor("local", scope="local", shape=(8,))
    data = cce._tensor("data", scope="shared.dyn", shape=(16,))
    pointer = cce._ptr("data_ptr", scope="shared.dyn")
    body = [cce._alloc(local), cce._decl(data, pointer)]
    body += [ir.TensorStore(local, [i], ir.const(-1, "float32")) for i in range(8)]
    body += [ir.TensorStore(local, [prim.Broadcast(2, 4)], ir.TensorLoad(data, [prim.Ramp(0, 1, 4)]))]
    body += [ir.TensorStore(data, [8 + i], ir.TensorLoad(local, [i])) for i in range(8)]
    source = cce._source([pointer], body, tmp_path, name="broadcast_store")
    _run_generated_cce(
        source,
        r"""
int main() {
  alignas(16) float data[16] = {1,2,3,4};
  broadcast_store(data);
  for (int i=0; i<8; ++i) {
    float expected = i == 2 ? 4 : -1;
    if (data[8+i] != expected) return 1;
  }
  return 0;
}
""",
        tmp_path,
    )


def test_cce_u16_index_multiply_wraps_without_signed_promotion_overflow(tmp_path):
    x, y = ir.Var("x", "uint16"), ir.Var("y", "uint32")
    index = prim.Cast("int64x4", prim.Broadcast(x, 4) * prim.Cast("uint16x4", prim.Ramp(y, ir.const(1, "uint32"), 4)))
    a = cce._tensor("a", scope="shared.dyn", shape=(65536,))
    b = cce._tensor("b", scope="shared.dyn", shape=(4,))
    ap, bp = cce._ptr("ap", scope="shared.dyn"), cce._ptr("bp", scope="shared.dyn")
    source = cce._source(
        [ap, bp, x, y],
        [cce._decl(a, ap), cce._decl(b, bp), ir.TensorStore(b, [prim.Ramp(0, 1, 4)], ir.TensorLoad(a, [index]))],
        tmp_path,
        name="multiply",
    )
    _run_generated_cce(
        source,
        r"""
int main() {
  alignas(16) float a[65536], b[4];
  for (int i=0; i<65536; ++i) a[i] = float(i);
  multiply(a,b,65535,65535);
  const int expected[4] = {1,0,65535,65534};
  for (int i=0; i<4; ++i) if (b[i] != expected[i]) return 1;
  return 0;
}
""",
        tmp_path,
    )


@pytest.mark.parametrize("backend", [cce, pto], ids=["cce", "pto"])
def test_pointer_first_lane_evaluates_effectful_ramp_operands_once(backend, tmp_path):
    a = backend._tensor("a", scope="shared.dyn", shape=(32,))
    ap = backend._ptr("ap", scope="shared.dyn")
    index = prim.Ramp(tirx.call_extern("next_base", ty="int32"), tirx.call_extern("next_stride", ty="int32"), 4)
    pointer = backend._ptr("answer_ptr", scope="shared.dyn")
    source = backend._source([ap], [backend._decl(a, ap), ir.Bind(pointer, tirx.address_of(a[index]))], tmp_path)
    assert source.count("next_base()") == 1
    assert source.count("next_stride()") == 1
    assert "pto.load(" not in source


def test_cce_scalar_core_broadcast_store_evaluates_address_and_value_once(tmp_path):
    a = cce._tensor("a", scope="shared.dyn", shape=(16,))
    ap = cce._ptr("ap", scope="shared.dyn")
    index = prim.Ramp(tirx.call_extern("next_offset", ty="int32"), 1, 4)
    value = prim.Broadcast(tirx.call_extern("next_value", ty="float32"), 4)
    source = cce._source([ap], [cce._decl(a, ap), ir.TensorStore(a, [index], value)], tmp_path, name="effect_broadcast")
    functions = r"""
static int address_calls, value_calls;
int32_t next_offset() { ++address_calls; return 4; }
float next_value() { ++value_calls; return 9.0f; }
"""
    _run_generated_cce(
        functions + source,
        r"""
int main() {
  alignas(16) float a[16] = {};
  effect_broadcast(a);
  if (address_calls != 1 || value_calls != 1) return 1;
  for (int i=0; i<16; ++i) if (a[i] != (i >= 4 && i < 8 ? 9.0f : 0.0f)) return 2;
  return 0;
}
""",
        tmp_path,
    )


def test_cce_vector_element_gather_scatter_lane_order(tmp_path):
    local = cce._tensor("local", "float32x2", "local", (4,))
    a = cce._tensor("a", scope="shared.dyn", shape=(12,))
    b = cce._tensor("b", scope="shared.dyn", shape=(12,))
    ap, bp = cce._ptr("ap", scope="shared.dyn"), cce._ptr("bp", scope="shared.dyn")
    body = [cce._alloc(local), cce._decl(a, ap), cce._decl(b, bp)]
    body += [ir.TensorStore(local, [i], ir.TensorLoad(a, [prim.Ramp(i * 2, 1, 2)])) for i in range(4)]
    index = prim.Ramp(1, 2, 2)
    body += [ir.TensorStore(b, [prim.Ramp(0, 1, 4)], ir.TensorLoad(local, [index]))]
    body += [ir.TensorStore(local, [index], ir.TensorLoad(a, [prim.Ramp(8, 1, 4)]))]
    body += [ir.TensorStore(b, [prim.Ramp(4 + i * 2, 1, 2)], ir.TensorLoad(local, [i])) for i in range(4)]
    source = cce._source([ap, bp], body, tmp_path, name="vector_slots")
    _run_generated_cce(
        source,
        r"""
int main() {
  alignas(16) float a[12], b[12];
  for (int i=0; i<12; ++i) a[i] = float(i);
  vector_slots(a,b);
  const int expected[12] = {2,3,6,7,0,1,8,9,4,5,10,11};
  for (int i=0; i<12; ++i) if (b[i] != expected[i]) return 1;
  return 0;
}
""",
        tmp_path,
    )


@pytest.mark.parametrize("direction", ["load", "store"])
def test_pto_narrow_vector_slots_scale_in_address_width(direction, tmp_path):
    offset = ir.Var("offset", "uint32")
    a = pto._tensor("a", "float32x2", "local", (256,))
    b = pto._tensor("b", scope="shared.dyn", shape=(4,))
    bp = pto._ptr("bp", scope="shared.dyn")
    index = prim.Cast("uint8x2", prim.Ramp(offset, ir.const(2, "uint32"), 2))
    contiguous = prim.Ramp(0, 1, 4)
    op = ir.TensorStore(
        b if direction == "load" else a,
        [contiguous if direction == "load" else index],
        ir.TensorLoad(a if direction == "load" else b, [index if direction == "load" else contiguous]),
    )
    body = [pto._decl(b, bp), pto._block("SIMT_VF", [pto._thread(ir.Var("tx", "int32"), 1, [pto._alloc(a), op])])]
    source = pto._source([bp, offset], body, tmp_path)
    # A uint8 slot 200 means physical scalar offsets 400/401, not 144/145.
    for lane in range(2):
        assert f"pto.cast(pto.cast(tl_indices[{lane}], pto.ui8), pto.i64) * 2" in source


def test_cce_integer_index_cast_preserves_floating_source_arithmetic(tmp_path):
    x, y = ir.Var("x", "float32x4"), ir.Var("y", "float32x4")
    index = prim.Cast("int64x4", x + y)
    a = cce._tensor("a", scope="shared.dyn", shape=(8,))
    b = cce._tensor("b", scope="shared.dyn", shape=(4,))
    ap, bp = cce._ptr("ap", scope="shared.dyn"), cce._ptr("bp", scope="shared.dyn")
    source = cce._source(
        [ap, bp, x, y],
        [cce._decl(a, ap), cce._decl(b, bp), ir.TensorStore(b, [prim.Ramp(0, 1, 4)], ir.TensorLoad(a, [index]))],
        tmp_path,
        name="float_index",
    )
    _run_generated_cce(
        source,
        r"""
int main() {
  alignas(16) float a[8], b[4];
  for (int i=0; i<8; ++i) a[i] = float(i);
  float_index(a,b,float4{1.5f,2.5f,3.25f,4.25f},float4{1.5f,1.5f,1.75f,1.75f});
  for (int i=0; i<4; ++i) if (b[i] != float(i+3)) return 1;
  return 0;
}
""",
        tmp_path,
    )
