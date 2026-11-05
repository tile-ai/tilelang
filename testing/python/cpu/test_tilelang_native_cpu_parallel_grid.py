"""CPU grid independence and allocation privacy on the native object model."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.cpu.language as T
from tilelang.testing.packed import compile_packed_function


def walk(value, kind):
    nodes = []
    tvm_ffi.structural_walk(value, (kind, nodes.append), order="pre")
    return nodes


def materialize(function, target="llvm", minimum=0, attrs=None):
    function = function.with_attr({"target": tvm_tirx.target.Target(target), **(attrs or {})})
    module = ir.IRModule.from_expr(function)
    with ir.transform.PassContext(config={"tl.cpu_parallel": True, "tl.cpu_parallel_min_trip": minimum}):
        module = tvm_ffi.get_global_func("tl.transform.MaterializeKernelLaunch")(False, False, None, ["cluster_dims"], None)(module)
        module = tvm_ffi.get_global_func("tl.cpu.transform.LowerCPUKernelLaunch")()(module)
        module = tvm_ffi.get_global_func("tl.cpu.transform.MaterializeCPUParallelGrid")()(module)
    assert tirx.analysis.check_well_formed(module)
    assert all("tl.cpu_grid_dim" not in loop.annotations for loop in walk(module, ir.For))
    return module


def parallel_count(module):
    return sum(loop.kind == ir.ForKind.PARALLEL for loop in walk(module, ir.For))


def execute(module, values):
    tensors = [tvm_tirx.runtime.tensor(value) for value in values]
    tirx.compile(module, target="llvm")(*tensors)
    return [value.numpy() for value in tensors]


@pytest.mark.parametrize("target,count", [("c", 3), ("llvm", 1)])
def test_perfect_grid_nest_and_unit_outer_axis(target, count):
    @T.prim_func
    def kernel(A: T.Tensor((20,), "int32")):
        with T.Kernel(1, 4, 5, cpu_num_threads=3) as (unit, bx, by):
            A[unit + bx * 5 + by] = bx * 5 + by

    result = materialize(kernel, target)
    assert parallel_count(result) == count
    loops = walk(result, ir.For)
    assert loops[0].annotations["tl.cpu_num_threads"] == 3
    if target == "llvm":
        assert loops[0].kind == ir.ForKind.DEFAULT
        assert loops[1].kind == ir.ForKind.PARALLEL
        np.testing.assert_array_equal(execute(result, [np.zeros(20, "int32")])[0], np.arange(20, dtype="int32"))


@pytest.mark.parametrize("minimum,parallel", [(0, True), (8, True), (9, False)])
def test_minimum_trip_threshold(minimum, parallel):
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32")):
        with T.Kernel(8) as bx:
            A[bx] = bx

    assert bool(parallel_count(materialize(kernel, minimum=minimum))) == parallel


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("contract", ["missing", "enabled", "disabled", "both_non_restrict", "one_non_restrict", "read_only"])
def test_parameter_alias_contract(target, contract):
    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
        with T.Kernel(16) as bx:
            if contract == "read_only":
                T.evaluate(A[bx] + B[bx])
            else:
                B[bx] = A[bx] + 1

    attrs = {}
    if contract != "missing":
        attrs["tirx.noalias"] = contract != "disabled"
    if contract in ("both_non_restrict", "one_non_restrict"):
        attrs["tl.non_restrict_params"] = list(kernel.params[: 2 if contract == "both_non_restrict" else 1])
    result = materialize(kernel, target, attrs=attrs)
    assert bool(parallel_count(result)) == (contract not in ("disabled", "both_non_restrict"))


@pytest.mark.parametrize("collision", ["constant", "modulo", "cross_store", "carried_read", "mixed_cast", "raw_cast"])
def test_overlapping_write_or_carried_read_stays_serial(collision):
    @T.prim_func
    def kernel(A: T.Tensor((17,), "int32")):
        with T.Kernel(16) as bx:
            if collision == "constant":
                A[0] = bx
            elif collision == "modulo":
                A[bx % 2] = bx
            elif collision == "cross_store":
                A[bx + 1] = bx
                A[bx] = bx
            elif collision == "carried_read":
                A[bx + 1] = A[bx] + 1
            elif collision == "mixed_cast":
                A[T.int32(T.int16(bx)) - T.int32(T.int8(bx))] = bx
            else:
                A[T.int32(T.int16(bx)) - bx] = bx

    result = materialize(kernel)
    assert parallel_count(result) == 0
    if collision == "carried_read":
        output = tvm_tirx.runtime.tensor(np.zeros(17, "int32"))
        compile_packed_function(result)(output)
        np.testing.assert_array_equal(output.numpy(), np.arange(17, dtype="int32"))


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("cast_dtype,extent,parallel", [("int8", 512, False), ("int8", 128, True), ("int64", 512, True)])
def test_cast_range_proof_preserves_exclusive_upper_bound(target, cast_dtype, extent, parallel):
    @T.prim_func
    def kernel(A: T.Tensor((512,), "int32")):
        with T.Kernel(extent) as bx:
            index = T.cast(bx, cast_dtype)
            if index >= 0:
                A[index] = bx

    result = materialize(kernel, target)
    assert bool(parallel_count(result)) == parallel


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("view_dtype,width,parallel", [("uint32", 1, True), ("uint8", 4, False)])
def test_same_storage_views_preserve_element_width(target, view_dtype, width, parallel):
    @T.prim_func
    def kernel(B: T.Tensor((16,), "int32")):
        view = T.view(B, (16 * width,), dtype=view_dtype)
        with T.Kernel(16) as bx:
            B[bx] = bx + 100
            view[bx] = T.cast(255, view_dtype)

    result = materialize(kernel, target)
    assert bool(parallel_count(result)) == parallel
    if target == "llvm":
        output = execute(result, [np.zeros(16, "int32")])[0]
        expected = np.full(16, 255, "int32") if parallel else np.arange(16, dtype="int32") + 100
        if not parallel:
            expected[:4] = -1
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("target", ["c", "llvm"])
@pytest.mark.parametrize("reset", ["full", "full_byte_view", "partial", "conditional", "while", "empty_outer", "read_before_complete"])
def test_local_allocation_reset_and_iteration_privacy(target, reset):
    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        state = T.alloc_buffer((2,), "int32", scope="local")
        view = T.view(state, (8,), dtype="uint8")
        with T.Kernel(16) as bx:
            if bx == 0:
                state[0] = 0
                state[1] = 0
            if reset == "full":
                for j in T.serial(2):
                    state[j] = 0
            elif reset == "full_byte_view":
                for j in T.serial(8):
                    view[j] = T.uint8(0)
            elif reset == "partial":
                state[0] = 0
            elif reset == "conditional":
                if bx == 0:
                    for j in T.serial(2):
                        state[j] = 0
            elif reset == "while":
                k = T.alloc_buffer((1,), "int32", scope="local")
                k[0] = bx
                while k[0] == 0:
                    for j in T.serial(2):
                        state[j] = 0
                    k[0] = 1
            elif reset == "empty_outer":
                for _k in T.serial(0):
                    for j in T.serial(2):
                        state[j] = 0
            else:
                for j in T.serial(2):
                    state[j] = 0
                    A[bx] = state[(j + 1) % 2]
            state[1] = state[1] + 1
            A[bx] = state[1]

    result = materialize(kernel, target)
    complete = reset in ("full", "full_byte_view")
    assert bool(parallel_count(result)) == complete
    if complete:
        parallel = next(loop for loop in walk(result, ir.For) if loop.kind == ir.ForKind.PARALLEL)
        binds = walk(parallel.body, ir.Bind)
        assert any(isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor" for binding in binds)
        # The byte view created before the grid must move with its backing
        # allocation even when that particular branch did not use it.
        assert any(isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.decl_tensor" for binding in binds)
    if target == "llvm" and reset != "read_before_complete":
        output = execute(result, [np.zeros(16, "int32")])[0]
        expected = np.ones(16, "int32") if complete else np.arange(1, 17, dtype="int32")
        np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("pure,parallel", [(False, False), (True, True)])
def test_external_effects_without_pointer_arguments(pure, parallel):
    emit = T.call_pure_extern if pure else T.call_extern

    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        with T.Kernel(16) as bx:
            A[bx] = emit("int32", "scalar_external", bx)

    assert bool(parallel_count(materialize(kernel))) == parallel


def test_atomic_marker_keeps_grid_serial():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        with T.Kernel(16) as bx:
            A[bx] = bx

    assert parallel_count(materialize(kernel, attrs={"tl.cpu_had_atomics": True})) == 0


def test_native_control_flow_cannot_certify_full_reset():
    tensor = ir.Var("A", tirx.TensorType([16], "int32"))
    bx = ir.Var("bx", ir.PrimType("int32"))
    loop = ir.For(
        bx,
        T.int32(0),
        T.int32(16),
        T.int32(1),
        ir.ForKind.DEFAULT,
        ir.SeqStmt([ir.TensorStore(tensor, [bx], bx), ir.Break()]),
        annotations={"tl.cpu_grid_dim": 0},
    )
    function = tirx.Function([tensor], ir.SeqStmt([loop]), ir.PrimType("void"))
    assert parallel_count(materialize(function)) == 0


def test_serial_outer_grid_cannot_share_state_with_parallel_inner_grid():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        with T.Kernel(2) as block:
            state = T.alloc_buffer((1,), "int32", scope="local")
            state[0] = 0
            T.call_extern("void", "mark_epoch", block)
            with T.Kernel(8) as inner:
                state[0] = state[0] + 1
                A[block * 8 + inner] = state[0]

    # The outer grid is serial because of mark_epoch. Its allocation is
    # therefore shared by the candidate inner grid and carries state.
    assert parallel_count(materialize(kernel)) == 0


def test_serial_outer_grid_can_privatize_a_fully_reset_inner_allocation():
    @T.prim_func
    def kernel(A: T.Tensor((16,), "int32")):
        with T.Kernel(2) as block:
            state = T.alloc_buffer((1,), "int32", scope="local")
            T.call_extern("void", "mark_epoch", block)
            with T.Kernel(8) as inner:
                state[0] = inner + 1
                A[block * 8 + inner] = state[0]

    result = materialize(kernel)
    assert parallel_count(result) == 1
    parallel = next(loop for loop in walk(result, ir.For) if loop.kind == ir.ForKind.PARALLEL)
    assert any(
        isinstance(binding.value, ir.Call) and binding.value.op.name == "tirx.alloc_tensor" for binding in walk(parallel.body, ir.Bind)
    )
