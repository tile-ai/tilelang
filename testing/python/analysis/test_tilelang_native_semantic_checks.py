"""Semantic checks on native IR preserve accepted and rejected TileLang programs."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.language as T
from tilelang.analysis import ASTPrinter, FragmentLoopChecker, NestedLoopChecker, ParallelLocalIndexChecker
from tilelang.analysis.fragment_loop_checker import collect_fragment_accesses


def _check(checker, function):
    module = ir.IRModule.from_expr(function)
    result = checker()(module)
    assert tvm_ffi.structural_equal(module, result)


def test_consecutive_parallel_nests_and_sequential_siblings_are_valid():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(2):
                for j in T.Parallel(4):
                    for k in T.Parallel(4):
                        A[i * 16 + j * 4 + k] = i + j + k
            for i in T.Parallel(32):
                for j in T.serial(2):
                    A[i] = A[i] + j

    _check(NestedLoopChecker, kernel)


@pytest.mark.parametrize("interruption", ["statement", "serial", "conditional"])
def test_nonconsecutive_parallel_nests_are_rejected(interruption):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32")):
        for i in T.Parallel(4):
            if interruption == "statement":
                A[i] = 0
                for j in T.Parallel(8):
                    A[i * 8 + j] = j
            elif interruption == "serial":
                for k in T.serial(2):
                    for j in T.Parallel(8):
                        A[i * 8 + j] = k
            else:
                if A[i] > 0:
                    for j in T.Parallel(8):
                        A[i * 8 + j] = j

    with pytest.raises(ValueError, match="Nested parallel loops are not allowed"):
        _check(NestedLoopChecker, kernel)


def test_pipeline_inside_parallel_is_rejected_but_pipeline_around_parallel_is_valid():
    @T.prim_func
    def invalid(A: T.Tensor((32,), "int32")):
        for i in T.Parallel(32):
            for j in T.Pipelined(4, num_stages=2):
                A[i] = A[i] + j

    @T.prim_func
    def valid(A: T.Tensor((32,), "int32")):
        for j in T.Pipelined(4, num_stages=2):
            for i in T.Parallel(32):
                A[i] = A[i] + j

    with pytest.raises(ValueError, match="Pipelined loop cannot be nested inside a parallel loop"):
        _check(NestedLoopChecker, invalid)
    _check(NestedLoopChecker, valid)


def test_collective_copy_inside_parallel_uses_real_operator_registration():
    @T.prim_func
    def invalid(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        for i in T.Parallel(4):
            T.copy(A[i * 8 : i * 8 + 8], B[i * 8 : i * 8 + 8])

    with pytest.raises(ValueError, match="Only elementwise operations are allowed"):
        _check(NestedLoopChecker, invalid)


def test_per_iteration_intrinsics_are_allowed():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32")):
        for i in T.Parallel(32):
            A[i] = T.max(A[i], 0.0)
            T.atomic_add(A[i], 1.0)

    _check(NestedLoopChecker, kernel)


@pytest.mark.parametrize("access", ["load", "store", "indirect"])
def test_local_index_depends_on_parallel_loop_is_rejected(access):
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32")):
        local = T.alloc_local((32,), "int32")
        for i in T.Parallel(32):
            if access == "store":
                local[i] = 1
            elif access == "load":
                A[i] = local[i]
            else:
                index = (i + 1) % 32
                local[index] = 1

    with pytest.raises(ValueError, match="Local buffer.*is indexed by T.Parallel loop variable"):
        _check(ParallelLocalIndexChecker, kernel)


def test_per_thread_local_index_and_replicated_scalar_are_valid():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "int32")):
        local = T.alloc_local((4,), "int32")
        for i in T.Parallel(32):
            for lane in T.vectorized(4):
                local[lane] = lane
            A[i] = local[0]

    _check(ParallelLocalIndexChecker, kernel)


def test_flat_bind_alias_keeps_parallel_dependency_and_var_names_do_not_alias():
    local = ir.Var("local", T.Tensor((32,), "int32", scope="local").as_type())
    parallel = ir.Var("i", ir.PrimType("int32"))
    unrelated = ir.Var("i", ir.PrimType("int32"))
    index = ir.Var("index", ir.PrimType("int32"))

    def make(value):
        body = [ir.Bind(index, value), ir.TensorStore(local, [index], ir.const(1))]
        loop = ir.For(parallel, ir.const(0), ir.const(32), ir.const(1), ir.ForKind.PARALLEL, body)
        return tirx.Function([local, unrelated], [loop])

    with pytest.raises(ValueError, match="Local buffer.*is indexed by T.Parallel loop variable"):
        _check(ParallelLocalIndexChecker, make(parallel + 1))
    _check(ParallelLocalIndexChecker, make(unrelated))


@pytest.mark.parametrize("scope", ["local.fragment", "shared.dyn"])
@pytest.mark.parametrize("depends_on_symbolic_loop", [False, True])
def test_fragment_restriction_uses_scope_and_symbolic_loop_dependency(scope, depends_on_symbolic_loop):
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int32")):
        fragment = T.alloc_buffer((32,), "int32", scope=scope)
        for i in T.Parallel(n):
            for j in T.Parallel(32):
                if depends_on_symbolic_loop:
                    fragment[(i + j) % 32] = A[i]
                else:
                    fragment[j] = j

    if scope == "local.fragment" and depends_on_symbolic_loop:
        with pytest.raises(ValueError, match="T.Parallel loop with symbolic range.*fragment buffer"):
            _check(FragmentLoopChecker, kernel)
    else:
        _check(FragmentLoopChecker, kernel)
    accesses = collect_fragment_accesses(kernel.body)
    assert len(accesses) == (1 if scope == "local.fragment" else 0)


def test_native_ast_printer_handles_bind_regions_and_empty_else(capsys):
    @T.prim_func
    def kernel(A: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=8):
            local = T.alloc_local((1,), "int32")
            for i in T.Parallel(8):
                if A[i] > 0:
                    local[0] = A[i]

    _check(ASTPrinter, kernel)
    output = capsys.readouterr().out
    for name in ("Function", "SeqStmt", "RegionStmt", "Bind", "For", "If", "TensorStore"):
        assert name in output
    assert "buffer_map" not in output
