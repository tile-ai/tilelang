"""Native PC warp specialization preserves definitions and barrier transactions.

The protocol interpreter below executes generated integer control flow and
counter memory only. It is an independent oracle for slot/parity/order; it does
not substitute for numerical GPU tests in the existing PC test modules.
"""

import importlib.util
import math
import operator
from pathlib import Path

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.layout import Layout, make_swizzled_layout
from tilelang.testing.ir import collect_calls, collect_nodes


def _kernel_module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ordinary = _kernel_module("test_tilelang_transform_producer_consumer_ws")
persistent = _kernel_module("test_tilelang_transform_producer_consumer_ws_persistent")


def _apply(function, *, annotate_device_bound=False):
    function = function.with_attr({"global_symbol": "main", "target": Target({"kind": "cuda", "arch": "sm_90"})})
    module = tilelang.transform.MaterializeKernelLaunch()(ir.IRModule.from_expr(function))
    if annotate_device_bound:
        module = tilelang.cuda.transform.AnnotateDeviceBoundTmaCopies()(module)
    assert tirx.analysis.check_well_formed(module["main"])
    before = ir.save_json(module)
    result = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(module)["main"]
    assert ir.save_json(module) == before
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    assert tirx.analysis.check_well_formed(tirx.renew_def(result))
    assert not collect_calls(result, op="tl.mvb_stage_index")
    return result


_KERNEL_CASES = [
    (ordinary.matmul_pipelined, (64, 64, 128, 64, 32, 64, 2), {}),
    (ordinary.dual_gemm_shared_accumulator, (64, 64, 64, 128, 64, 64, 32, 2), {}),
    (ordinary.matmul_windowed_pipelined, (64, 128, 64, 64, 32, 64, 1), {}),
    (ordinary.prelude_tma_wait_sink, (), {}),
    (ordinary.prelude_tma_bound_index, (), {}),
    (ordinary.guarded_prelude_tma_postloop_scalar, (), {}),
    (ordinary.explicit_cp_async_wait_position, (), {}),
    (ordinary.grouped_gemm_padded_pipelined, ((63, 77), 128, 128), {}),
    (persistent.persistent_matmul_primitive, (), {}),
    (persistent.persistent_matmul_serial, (), {}),
    (persistent.persistent_matmul_serial_dynamic_bound, (), {}),
    (persistent.persistent_matmul_misaligned_k, (128, 128, 96, 64, 64, 32, 2, 1), {}),
    (persistent.persistent_matmul_guarded_body, ("lt",), {}),
    (persistent.persistent_matmul_guarded_body, ("eq",), {}),
    (persistent.shifted_mod_guard_matmul, (False,), {}),
    (persistent.shifted_mod_guard_matmul, (True,), {}),
    (persistent.persistent_matmul_variable_stages, (128, 128, 96, 64, 64, 32, 1, 1), {}),
    (persistent.persistent_matmul_variable_stages, (128, 128, 96, 64, 64, 32, 3, 1), {}),
    (persistent.persistent_matmul_guarded_deep_pipeline, (), {}),
]


@pytest.mark.parametrize("factory,args,kwargs", _KERNEL_CASES, ids=[f"{f.__name__}-{i}" for i, (f, _, _) in enumerate(_KERNEL_CASES)])
def test_existing_kernel_definitions_survive_both_roles(factory, args, kwargs):
    result = _apply(factory(*args, **kwargs))
    assert int(result.attrs.get("tl_tiled_ws_applied", 0)) == 1
    assert collect_calls(result, op="tl.tileop.tma_copy")
    assert collect_calls(result, op="tl.mbarrier_wait_parity")
    _, branch = ordinary._ws_partition(result)
    producer_defs = {bind.var for bind in collect_nodes(branch.then_case, ir.Bind)}
    consumer_defs = {bind.var for bind in collect_nodes(branch.else_case, ir.Bind)}
    assert producer_defs.isdisjoint(consumer_defs)
    repeated = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(ir.IRModule.from_expr(result))["main"]
    assert tvm_ffi.structural_equal(result, repeated)


def test_device_pointer_table_is_not_automatically_promoted_to_tma():
    result = _apply(ordinary.device_bound_copy_pipelined(), annotate_device_bound=True)
    assert int(result.attrs.get("tl_tiled_ws_applied", 0)) == 0
    assert not collect_calls(result, op="tl.tileop.tma_copy")
    assert ordinary._count_device_bound_copy_annotations(result) == (1, 2)


def test_private_fragment_clone_does_not_clone_shared_prelude_local_vars():
    result = _apply(ordinary.grouped_gemm_padded_pipelined((63, 77), 128, 128))
    tensors = [bind.var for bind in collect_nodes(result, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)]
    for name in ("cur_batch_idx", "cur_batch_size"):
        assert len([var for var in tensors if var.name == name]) == 1
        assert not any(var.name == name + "_producer_ws" for var in tensors)
    assert tirx.analysis.check_well_formed(result)


def _int(expr, values, counters):
    if isinstance(expr, ir.IntImm):
        return int(expr)
    if isinstance(expr, ir.Var):
        return values[expr]
    if isinstance(expr, ir.TensorLoad):
        return counters[(expr.source, tuple(_int(index, values, counters) for index in expr.indices))]
    if isinstance(expr, prim.Cast):
        result = _int(expr.value, values, counters)
    elif isinstance(expr, prim.Select):
        return _int(expr.true_value if _int(expr.condition, values, counters) else expr.false_value, values, counters)
    elif isinstance(expr, prim.And):
        return bool(_int(expr.a, values, counters) and _int(expr.b, values, counters))
    elif isinstance(expr, prim.Or):
        return bool(_int(expr.a, values, counters) or _int(expr.b, values, counters))
    elif isinstance(expr, prim.Not):
        return not _int(expr.a, values, counters)
    else:
        a, b = _int(expr.a, values, counters), _int(expr.b, values, counters)
        ops = {
            prim.Add: operator.add,
            prim.Sub: operator.sub,
            prim.Mul: operator.mul,
            prim.FloorDiv: operator.floordiv,
            prim.FloorMod: operator.mod,
            prim.BitwiseXor: operator.xor,
            prim.BitwiseAnd: operator.and_,
            prim.BitwiseOr: operator.or_,
            prim.Min: min,
            prim.Max: max,
            prim.LT: operator.lt,
            prim.LE: operator.le,
            prim.GT: operator.gt,
            prim.GE: operator.ge,
            prim.EQ: operator.eq,
            prim.NE: operator.ne,
        }
        if isinstance(expr, (prim.Div, prim.Mod)):
            quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
            result = quotient if isinstance(expr, prim.Div) else a - quotient * b
        else:
            result = ops[type(expr)](a, b)
    dtype = str(expr.ty)
    if dtype.startswith(("int", "uint")):
        unsigned = dtype.startswith("u")
        bits = int(dtype[4:] if unsigned else dtype[3:])
        result %= 1 << bits
        if not unsigned and result >= 1 << (bits - 1):
            result -= 1 << bits
    return result


def _protocol_trace(function, thread):
    values, counters, trace = {}, {}, []

    def value(expr):
        return _int(expr, values, counters)

    def visit(stmt):
        if isinstance(stmt, ir.SeqStmt):
            for part in stmt.seq:
                visit(part)
        elif isinstance(stmt, ir.RegionStmt):
            if stmt.op.name == "tirx.launch_thread":
                values[stmt.body_params[0]] = thread if stmt.args[0].value == "threadIdx.x" else 0
            else:
                assert stmt.op.name == "tl.attr"
                assert not stmt.body_params and not stmt.result_vars
            visit(stmt.body)
        elif isinstance(stmt, ir.For):
            for i in range(value(stmt.begin), value(stmt.end), value(stmt.step)):
                values[stmt.var] = i
                visit(stmt.body)
        elif isinstance(stmt, ir.If):
            branch = stmt.then_case if value(stmt.condition) else stmt.else_case
            if branch is not None:
                visit(branch)
        elif isinstance(stmt, ir.Bind):
            if isinstance(stmt.var.ty, tirx.TensorType):
                assert isinstance(stmt.value, ir.Call) and stmt.value.op.name == "tirx.alloc_tensor"
            else:
                values[stmt.var] = value(stmt.value)
        elif isinstance(stmt, ir.TensorStore):
            # Arithmetic tensor data is checked by the GPU numerical tests.
            # The protocol's counters are actual generated mutable storage.
            if stmt.dest.name in {"producer_phase_cnt", "consumer_phase_cnt"}:
                counters[(stmt.dest, tuple(map(value, stmt.indices)))] = value(stmt.value)
        elif isinstance(stmt, ir.Evaluate):
            call = stmt.value
            assert isinstance(call, ir.Call)
            name = call.op.name
            if name == "tl.mbarrier_wait_parity":
                trace.append(("wait", value(call.args[0].indices[0]), value(call.args[1])))
            elif name == "tl.ptx_arrive_barrier":
                trace.append(("arrive", value(call.args[0].indices[0])))
            elif name == "tl.tileop.tma_copy":
                source, dest = call.args[:2]
                metadata = ordinary._annotations(call)
                trace.append(
                    (
                        "tma",
                        source.source.name,
                        tuple(value(r.begin) for r in source.region),
                        value(dest.region[0].begin),
                        value(metadata["barrier"].indices[0]),
                        value(metadata["tl.pipeline_mbar_phase_expr"]),
                    )
                )
            elif name == "tl.tileop.gemm":
                metadata = ordinary._annotations(call)
                trace.append(
                    (
                        "gemm",
                        value(call.args[0].region[0].begin),
                        value(call.args[1].region[0].begin),
                        value(metadata["tl.pipeline_mbar_phase_expr"]),
                    )
                )
            else:
                assert call.op.same_as(ir.Op.get("tl.layout_map")) or name in {"tl.annotation", "tl.tileop.copy", "tl.tileop.fill"}
        else:
            raise AssertionError(type(stmt))

    visit(function.body)
    return trace


_PROTOCOL_CASES = [
    (lambda: persistent.persistent_matmul_guarded_body("lt"), 2, [(tile, k) for tile in range(2) for k in (0, 1)]),
    (lambda: persistent.persistent_matmul_guarded_body("eq"), 2, [(tile, k) for tile in range(2) for k in (0, 2)]),
    (lambda: persistent.shifted_mod_guard_matmul(False), 2, [(tile, k) for tile in range(2) for k in (1, 3)]),
    (lambda: persistent.shifted_mod_guard_matmul(True), 2, [(tile, k) for tile in range(2) for k in (1, 3)]),
    (persistent.persistent_matmul_guarded_deep_pipeline, 3, [(tile, k) for tile in range(3) for k in range(5)]),
]


@pytest.mark.parametrize(
    "factory,stages,transactions", _PROTOCOL_CASES, ids=["guard-lt", "guard-eq", "shifted-serial", "shifted-persistent", "three-stage"]
)
def test_protocol_uses_executed_transaction_count_and_preserves_user_guards(factory, stages, transactions):
    result = _apply(factory())
    producer, consumer = _protocol_trace(result, 0), _protocol_trace(result, 128)
    expected = [(i % stages, (i // stages) % 2) for i in range(len(transactions))]
    actual_a = [(entry[2][0], entry[2][-1] // 32, entry[3], entry[4], entry[5]) for entry in producer if entry[:2] == ("tma", "A")]
    assert actual_a == [(tile, k, slot, slot, parity) for (tile, k), (slot, parity) in zip(transactions, expected, strict=True)]
    actual_b = [(entry[2][0], entry[2][1] // 32, entry[3], entry[4], entry[5]) for entry in producer if entry[:2] == ("tma", "B")]
    assert actual_b == [(tile, k, slot, slot, parity) for (tile, k), (slot, parity) in zip(transactions, expected, strict=True)]
    assert [entry[1:] for entry in consumer if entry[0] == "gemm"] == [(slot, slot, phase) for slot, phase in expected]
    assert [entry[1:] for entry in producer if entry[0] == "wait"] == [(stages + slot, phase ^ 1) for slot, phase in expected]
    assert [entry[1:] for entry in consumer if entry[0] == "wait"] == expected
    assert [entry[1] for entry in consumer if entry[0] == "arrive"] == [stages + slot for slot, _ in expected]


def _layouts(function):
    result = {}
    for call in collect_nodes(function, ir.Call):
        if isinstance(call.op, ir.Op) and call.op.same_as(ir.Op.get("tl.layout_map")):
            result.update(call.attrs["layout_map"])
    return result


def test_swizzle_annotation_expansion_preserves_each_slot_address():
    @T.prim_func
    def kernel(A: T.Tensor((48, 64), T.float16), B: T.Tensor((48, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 64), T.float16)
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            for i in T.Pipelined(3, num_stages=3):
                T.copy(A[i * 16, 0], shared)
                T.copy(shared, B[i * 16, 0])

    (original,) = _layouts(kernel).values()
    result = _apply(kernel)
    ((expanded_tensor, expanded),) = _layouts(result).items()
    assert [int(x) for x in expanded_tensor.ty.shape] == [3, 16, 64]
    assert [int(x) for x in expanded.get_input_shape()] == [3, 16, 64]
    original_shape = list(map(int, original.get_output_shape()))
    expanded_shape = list(map(int, expanded.get_output_shape()))

    def address(layout, shape, indices):
        offset = 0
        for index, extent in zip(layout.map_forward_index(indices), shape, strict=True):
            offset = offset * extent + _int(index, {}, {})
        return offset

    for slot in range(3):
        for row in range(16):
            for col in range(64):
                assert address(expanded, expanded_shape, [slot, row, col]) == slot * math.prod(original_shape) + address(
                    original, original_shape, [row, col]
                )


def test_non_tma_padded_layout_keeps_the_unsplit_kernel():
    padded = Layout((16, 128), lambda i, j: i * 136 + j)

    @T.prim_func
    def kernel(A: T.Tensor((32, 128), T.float16), B: T.Tensor((32, 128), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 128), T.float16)
            T.annotate_layout({shared: padded})
            for i in T.Pipelined(2, num_stages=2):
                T.copy(A[i * 16, 0], shared)
                T.copy(shared, B[i * 16, 0])

    result = _apply(kernel)
    assert int(result.attrs.get("tl_tiled_ws_applied", 0)) == 0
    assert not collect_calls(result, op="tl.tileop.tma_copy")
    assert len(collect_nodes(result, ir.For)) == 1
