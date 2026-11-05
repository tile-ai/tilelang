"""Public reduction spellings build native calls without ambient IR metadata."""

import numpy as np
import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.script.ir_builder import IRBuilder

import tilelang
import tilelang.language as T
import tilelang.testing


def _call(*, count=1, axes=True):
    output = ir.Var("output", tirx.TensorType((1,), "int32", scope="local"))
    thread = ir.Var("thread", "int32")
    args = [count, ir.const(7), True, ir.TensorLoad(output, [0])]
    if axes:
        args.append(thread)
    return T.tvm_thread_allreduce(*args, dtype="handle")


def _combine_value(call, left, right):
    result = call.args[0].apply([ir.const(left), ir.const(right)])
    return int(sym.Analyzer().simplify(result.fields[0]))


def test_reducer_scope_nesting_and_exception_restore():
    total = T.comm_reducer(lambda a, b: a + b, [0])
    maximum = T.comm_reducer(lambda a, b: T.max(a, b), [T.int32(-(2**31))])
    with IRBuilder(), T.attr(total, "reduce_scope", T.reinterpret(T.uint64(0), "handle")):
        assert _combine_value(_call(), 2, 3) == 5
        with (
            pytest.raises(RuntimeError, match="abort nested scope"),
            T.attr(maximum, "reduce_scope", T.reinterpret(T.uint64(0), "handle")),
        ):
            assert _combine_value(_call(), 2, 3) == 3
            raise RuntimeError("abort nested scope")
        assert _combine_value(_call(), 2, 3) == 5
    with IRBuilder(), pytest.raises(ValueError, match="enclosing reduce_scope"):
        _call()


def test_reducer_scope_does_not_capture_another_ir_builder():
    reducer = T.comm_reducer(lambda a, b: a + b, [0])
    with IRBuilder(), T.attr(reducer, "reduce_scope", T.reinterpret(T.uint64(0), "handle")):
        with IRBuilder(), pytest.raises(ValueError, match="same function"):
            _call()
        assert _combine_value(_call(), 4, 5) == 9


@pytest.mark.parametrize("count", [0, -1, 2, ir.Var("count", "int32")])
def test_reducer_result_count_is_checked_before_native_lowering(count):
    reducer = T.comm_reducer(lambda a, b: a + b, [0])
    with (
        IRBuilder(),
        T.attr(reducer, "reduce_scope", T.reinterpret(T.uint64(0), "handle")),
        pytest.raises(ValueError, match="result count"),
    ):
        _call(count=count)


def test_reducer_requires_a_thread_axis():
    reducer = T.comm_reducer(lambda a, b: a + b, [0])
    with (
        IRBuilder(),
        T.attr(reducer, "reduce_scope", T.reinterpret(T.uint64(0), "handle")),
        pytest.raises(ValueError, match="thread axes"),
    ):
        _call(axes=False)


def test_reducer_identity_and_combiner_types_are_not_coerced():
    with pytest.raises(ValueError, match="non-empty"):
        T.comm_reducer(lambda a, b: a + b, [])
    with pytest.raises(TypeError, match="primitive types"):
        T.comm_reducer(lambda a, b: a, [ir.Var("tensor", tirx.TensorType((1,), "int32"))])
    with pytest.raises(TypeError, match="return annotation"):
        T.comm_reducer(lambda a, b: T.Cast("float32", a + b), [0])
    with pytest.raises(TypeError, match="return annotation"):
        T.comm_reducer(lambda a, b: (a, b), [0])
    with pytest.raises(TypeError, match="requires a comm_reducer"):
        T.attr(ir.const(0), "reduce_scope", ir.const(0))


def _tuple_kernel(participating):
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), S: T.Tensor((1,), "int32"), M: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=32):
            thread = T.get_thread_binding()
            total = T.alloc_local((1,), "int32")
            maximum = T.alloc_local((1,), "int32")
            with T.attr(
                T.comm_reducer(lambda a, b, c, d: (a + c, T.max(b, d)), [0, T.int32(-(2**31))]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(2), A[thread], A[thread], thread < participating, total[0], maximum[0], thread, dtype="handle"
                    )
                )
            if thread == 0:
                S[0] = total[0]
                M[0] = maximum[0]

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("participating", [0, 1, 17, 32])
def test_tuple_reducer_predicate_and_identity_execute(participating, tmp_path):
    program = _tuple_kernel(participating)
    assert tirx.analysis.check_well_formed(program)
    calls = []
    tvm_ffi.structural_walk(program, (ir.Call, calls.append), order="pre")
    reductions = [call for call in calls if call.op.name == "tirx.gpu_thread_allreduce"]
    assert len(reductions) == 1
    assert reductions[0].ty == ir.PrimType("void")
    (tmp_path / "before.json").write_text(ir.save_json(program))
    kernel = tilelang.compile(program, out_idx=[1, 2], target="cuda")
    (tmp_path / "kernel.cu").write_text(kernel.get_kernel_source())
    source = np.arange(-97, -65, dtype="int32")
    total, maximum = kernel(torch.from_numpy(source).cuda())
    assert int(total.cpu().item()) == int(source[:participating].sum(dtype=np.int64))
    expected_max = int(source[:participating].max()) if participating else -(2**31)
    assert int(maximum.cpu().item()) == expected_max


@tilelang.testing.requires_cuda
def test_dp4a_preserves_read_write_accumulation_and_four_input_lanes(tmp_path):
    @T.prim_func
    def main(A: T.Tensor((8,), "int8"), B: T.Tensor((8,), "int8"), C: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                T.dp4a(A[0], B[0], C[0])
                T.dp4a(A[4], B[4], C[1])

    calls = []
    tvm_ffi.structural_walk(main, (ir.Call, calls.append), order="pre")
    pointers = [call for call in calls if call.op.name == "tl.access_ptr"]
    assert sorted((int(call.args[1]), int(call.args[2])) for call in pointers) == [(1, 3)] * 2 + [(4, 1)] * 4
    kernel = tilelang.compile(main, target="cuda")
    (tmp_path / "kernel.cu").write_text(kernel.get_kernel_source())
    left = np.array([-128, 127, -3, 2, 11, -13, 17, -19], dtype="int8")
    right = np.array([1, -2, 3, -4, -5, 6, -7, 8], dtype="int8")
    initial = np.array([29, -31], dtype="int32")
    result = torch.from_numpy(initial).cuda()
    kernel(torch.from_numpy(left).cuda(), torch.from_numpy(right).cuda(), result)
    expected = initial + (left.astype("int32") * right.astype("int32")).reshape(2, 4).sum(axis=1)
    np.testing.assert_array_equal(result.cpu().numpy(), expected)
