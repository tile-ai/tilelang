"""Opaque addresses preserve the ownership of native Parallel iterations."""

from collections import Counter
import json
import operator

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import collect_calls


_SOURCE = """
__device__ int record_once(int* pointer, int logical_index) {
  atomicAdd(pointer, 1);
  return logical_index;
}
__device__ int record_local(int* pointer, int logical_index) {
  *pointer += 1;
  return logical_index;
}
"""
_NONLOCAL = ("direct", "inner_bind", "outer_bind", "offset_alias", "strided_view", "shared", "fragment")
_LOCAL = ("local", "local_alias", "scalar", "scalar_alias", "allocated_scalar")


def _byte_offset(pointer, offset):
    return tirx.ptr_byte_offset(pointer, offset, ty=pointer.ty)


def _kernel(kind):
    fragment_layout = T.Fragment((16,), forward_fn=lambda i: (i % 4, i // 4))

    @T.prim_func
    def main(A: T.Tensor((36,), "int32"), Out: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=4):
            T.import_source(_SOURCE)
            if kind == "direct":
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_once", T.address_of(A[i]), i))
            if kind == "inner_bind":
                for i in T.Parallel(16):
                    pointer = T.address_of(A[i])
                    T.evaluate(T.call_extern("int32", "record_once", pointer, i))
            if kind == "outer_bind":
                pointer = T.address_of(A[0])
                alias = pointer
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_once", _byte_offset(alias, i * 4), i))
            if kind == "offset_alias":
                pointer = _byte_offset(T.address_of(A[0]), 4)
                alias = T.reinterpret(pointer, pointer.ty)
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_once", _byte_offset(alias, i * 4), i))
            if kind == "strided_view":
                view = T.decl_buffer((16,), "int32", data=A.data, strides=(2,), elem_offset=1, align=4)
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_once", T.address_of(view[i]), i))
            if kind == "shared":
                shared = T.alloc_shared((16,), "int32")
                T.clear(shared)
                T.sync_threads()
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_once", T.address_of(shared[i]), i))
                T.sync_threads()
                T.copy(shared, A[0:16])
            if kind == "fragment":
                fragment = T.alloc_fragment((16,), "int32")
                T.annotate_layout({fragment: fragment_layout})
                T.clear(fragment)
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_local", T.address_of(fragment[i]), i))
                T.copy(fragment, A[0:16])
            if kind == "local":
                local = T.alloc_local((1,), "int32")
                local[0] = 0
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_local", T.address_of(local[0]), i))
                Out[T.get_thread_binding(0)] = local[0]
            if kind == "local_alias":
                local = T.alloc_local((1,), "int32")
                local[0] = 0
                pointer = T.address_of(local[0])
                alias = _byte_offset(pointer, 0)
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_local", alias, i))
                Out[T.get_thread_binding(0)] = local[0]
            if kind == "scalar":
                scalar = A[0]
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_local", T.address_of(scalar), i))
                Out[T.get_thread_binding(0)] = scalar
            if kind == "scalar_alias":
                scalar = A[0]
                pointer = T.address_of(scalar)
                alias = _byte_offset(pointer, 0)
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_local", alias, i))
                Out[T.get_thread_binding(0)] = scalar
            if kind == "allocated_scalar":
                scalar = T.alloc_var("int32", init=0)
                for i in T.Parallel(16):
                    T.evaluate(T.call_extern("int32", "record_local", T.address_of(scalar), i))
                Out[T.get_thread_binding(0)] = scalar

    return main


def _integer(expr, values):
    """Evaluate this test's small integer domains independently of Analyzer."""
    if isinstance(expr, ir.IntImm):
        return int(expr)
    if isinstance(expr, ir.Var):
        return values[expr]
    binary = {
        "Add": operator.add,
        "Sub": operator.sub,
        "Mul": operator.mul,
        "FloorDiv": operator.floordiv,
        "FloorMod": operator.mod,
        "LT": operator.lt,
        "LE": operator.le,
        "GT": operator.gt,
        "GE": operator.ge,
        "EQ": operator.eq,
        "NE": operator.ne,
        "And": lambda a, b: a and b,
        "Or": lambda a, b: a or b,
    }
    name = type(expr).__name__
    if name in binary:
        return binary[name](_integer(expr.a, values), _integer(expr.b, values))
    if name == "Cast":
        # All exercised values fit both index widths. No symbolic rewrite or
        # unsigned/narrowing equivalence is needed for this event oracle.
        value = _integer(expr.value, values)
        assert -(2**31) <= value < 2**31
        return value
    if name == "Not":
        return not _integer(expr.a, values)
    raise AssertionError(f"Unhandled event-oracle expression: {type(expr)} {expr}")


def _events(function):
    events = []

    def visit(stmt, values, thread=0):
        if isinstance(stmt, ir.SeqStmt):
            for child in stmt.seq:
                visit(child, values, thread)
        elif isinstance(stmt, ir.RegionStmt):
            if stmt.op.name == "tirx.launch_thread":
                tag, extent = stmt.args
                (var,) = stmt.body_params
                for value in range(_integer(extent, values)):
                    visit(stmt.body, {**values, var: value}, value if tag.value == "threadIdx.x" else thread)
            else:
                assert not stmt.body_params and not stmt.result_vars, stmt
                visit(stmt.body, dict(values), thread)
        elif isinstance(stmt, ir.For):
            begin, end, step = (_integer(value, values) for value in (stmt.begin, stmt.end, stmt.step))
            assert step > 0 and 0 <= end - begin <= 64
            for value in range(begin, end, step):
                visit(stmt.body, {**values, stmt.var: value}, thread)
        elif isinstance(stmt, ir.If):
            branch = stmt.then_case if _integer(stmt.condition, values) else stmt.else_case
            if branch is not None:
                visit(branch, dict(values), thread)
        elif isinstance(stmt, ir.Bind):
            if isinstance(stmt.var.ty, ir.PrimType) and not isinstance(stmt.value, ir.TensorLoad):
                values[stmt.var] = _integer(stmt.value, values)
        elif isinstance(stmt, ir.Evaluate):
            call = stmt.value
            if isinstance(call, ir.Call) and call.op.name == "tirx.call_extern":
                assert call.args[0].value in ("record_once", "record_local")
                events.append((thread, _integer(call.args[-1], values)))
        elif isinstance(stmt, ir.TensorStore):
            # Initializers and the result copy are outside the opaque call's
            # Parallel loop. They do not contribute an observed call event.
            pass
        else:
            raise AssertionError(f"Unhandled event-oracle statement: {type(stmt)} {stmt}")

    visit(function.body, {})
    return events


@pytest.mark.parametrize("kind", (*_NONLOCAL, *_LOCAL))
def test_parallel_opaque_address_has_correct_iteration_ownership(kind, tmp_path):
    function = _kernel(kind)
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"})
    module = tvm_tirx.IRModule({"main": function.with_attr("target", target)})
    module = tilelang.transform.MaterializeKernelLaunch()(module)
    with target:
        module = tilelang.transform.LayoutInference()(module)
        (tmp_path / "before.txt").write_text(str(module))
        module = tilelang.transform.LowerTileOp()(module)
    (tmp_path / "after.txt").write_text(str(module))
    assert tirx.analysis.check_well_formed(module)
    if kind in ("scalar", "scalar_alias"):
        addresses = collect_calls(function, op="tirx.address_of")
        assert len(addresses) == 1 and isinstance(addresses[0].args[0], ir.Var)
    events = _events(module["main"])
    (tmp_path / "events.json").write_text(json.dumps(events) + "\n")
    if kind in _NONLOCAL:
        assert Counter(index for thread, index in events) == Counter(range(16))
        assert {thread for thread, index in events} == set(range(4))
    else:
        assert Counter(events) == Counter((thread, index) for thread in range(4) for index in range(16))


def _native_pointer_function(scope, *, fragment_alias=None):
    """Construct the exact already-inferred input contract of LowerTileOp."""
    index = ir.Var("i", "int32")
    thread = ir.Var("thread", "int32")
    pointer = ir.Var("pointer", ir.PtrType("int32", scope))
    layout = T.Fragment((16,), forward_fn=lambda i: (i % 4, i // 4))
    prefix = []
    params = [pointer]
    if fragment_alias:
        tensor_type = tirx.TensorType([16], "int32", scope="local.fragment")
        tensor = ir.Var("fragment", tensor_type)
        prefix.append(ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty=tensor_type, ty_args=[tensor_type])))
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"layout_map": {tensor: layout}})
        prefix.append(ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema})))
        address = (
            tirx.address_of(tensor[0])
            if fragment_alias == "address_of"
            else ir.Call("tl.access_ptr", [tensor[0], ir.const(1), ir.const(3)], ty=pointer.ty)
        )
        prefix.append(ir.Bind(pointer, address))
        params = []
    # A raw pointer offset names physical elements. Fragment ownership maps
    # logical i to (thread=i%4, slot=i//4), so every pointer stays in bounds.
    slot = index // 4 if fragment_alias else index
    call = ir.Evaluate(tirx.call_extern("record_once", _byte_offset(pointer, slot * 4), index, ty="int32"))
    loop = ir.For(index, ir.const(0), ir.const(16), ir.const(1), ir.ForKind.PARALLEL, [call], annotations={"parallel_loop_layout": layout})
    region = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(4)], [thread], {}, [*prefix, loop])
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"})
    return tirx.Function(params, [region], ir.PrimType("void"), ir.DictAttrs({"target": target}))


@pytest.mark.parametrize("scope", ["global", "local"])
def test_parallel_raw_pointer_scope_without_tensor_binding(scope, tmp_path):
    module = tvm_tirx.IRModule({"main": _native_pointer_function(scope)})
    assert tirx.analysis.check_well_formed(module)
    module = tilelang.transform.LowerTileOp()(module)
    assert tirx.analysis.check_well_formed(module)
    (tmp_path / "after.txt").write_text(str(module))
    events = _events(module["main"])
    if scope == "global":
        assert Counter(index for thread, index in events) == Counter(range(16))
    else:
        assert Counter(events) == Counter((thread, index) for thread in range(4) for index in range(16))


@pytest.mark.parametrize("address_kind", ["address_of", "access_ptr"])
def test_parallel_fragment_pointer_alias_keeps_original_ownership_after_local_remap(address_kind, tmp_path):
    module = tvm_tirx.IRModule({"main": _native_pointer_function("local.fragment", fragment_alias=address_kind)})
    assert tirx.analysis.check_well_formed(module)
    module = tilelang.transform.LowerTileOp()(module)
    assert tirx.analysis.check_well_formed(module)
    (tmp_path / "after.txt").write_text(str(module))
    op = "tirx.address_of" if address_kind == "address_of" else "tl.access_ptr"
    (address,) = collect_calls(module["main"], op=op)
    physical = address.args[0].source
    assert physical.ty.scope.name == "local"
    assert tuple(int(size) for size in physical.ty.shape) == (4,)
    if address_kind == "access_ptr":
        # This path really changes PtrType at the outer Bind. Ownership must
        # still be found through the original logical fragment definition.
        assert address.ty.scope.name == "local"
    events = _events(module["main"])
    assert Counter(index for thread, index in events) == Counter(range(16))


@pytest.mark.parametrize("inline_lets", [False, True])
@pytest.mark.parametrize("kind", (*_NONLOCAL, *_LOCAL))
@tilelang.testing.requires_cuda
def test_parallel_opaque_address_counts_real_side_effects_cuda(kind, inline_lets, tmp_path):
    import torch

    config = {tilelang.PassConfigKey.TL_SIMPLIFY: {tilelang.PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE: inline_lets}}
    kernel = tilelang.compile(_kernel(kind), target="cuda", execution_backend="tvm_ffi", pass_configs=config)
    counters = torch.zeros(36, dtype=torch.int32, device="cuda")
    outputs = torch.zeros(4, dtype=torch.int32, device="cuda")
    kernel(counters, outputs)
    (tmp_path / "kernel.cu").write_text(kernel.get_kernel_source())
    expected = torch.zeros_like(counters)
    if kind not in _LOCAL:
        if kind == "strided_view":
            expected[1:33:2] = 1
        elif kind == "offset_alias":
            expected[1:17] = 1
        else:
            expected[:16] = 1
    (tmp_path / "counts.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "inline_lets": inline_lets,
                "counters": counters.cpu().tolist(),
                "outputs": outputs.cpu().tolist(),
                "expected_counters": expected.cpu().tolist(),
                "expected_outputs": [16] * 4 if kind in _LOCAL else None,
                "device": torch.cuda.get_device_name(),
                "capability": torch.cuda.get_device_capability(),
                "torch": torch.__version__,
                "oracle": "Each nonlocal logical event once; each local event once per thread",
            },
            indent=2,
        )
        + "\n"
    )
    if kind in _LOCAL:
        torch.testing.assert_close(outputs, torch.full_like(outputs, 16), rtol=0, atol=0)
    torch.testing.assert_close(counters, expected, rtol=0, atol=0)
