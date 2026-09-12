import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.rocm.target import with_rocm_target_attrs
from tvm.tirx import op
from tvm.tirx.stmt_functor import post_order_visit


_CUDA_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
_ROCM_TARGET = with_rocm_target_attrs(tvm.target.Target({"kind": "hip", "mcpu": "gfx90a"}))
_ROCM_WAVE32_TARGET = with_rocm_target_attrs(tvm.target.Target({"kind": "hip", "mcpu": "gfx1100"}))


def _make_any_of_kernel(condition_kind):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()

            if condition_kind == "none":
                result[tx] = T.any_of(shared)
            elif condition_kind == "single_lane":
                if tx == 0:
                    result[tx] = T.any_of(shared)
            elif condition_kind == "warp_group":
                if tx // 32 == 0:
                    result[tx] = T.any_of(shared)
            elif condition_kind == "alternating":
                if tx % 2 == 0:
                    result[tx] = T.any_of(shared)
            else:
                raise ValueError(condition_kind)

    return main


def _make_all_of_kernel(divergent):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            if divergent:
                if tx == 0:
                    result[tx] = T.all_of(shared)
            else:
                result[tx] = T.all_of(shared)

    return main


def _make_nested_any_of_kernel():
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            if tx == 0:
                result[tx] = T.any_of(shared)
            result[tx] = T.any_of(shared)

    return main


def _make_wave_partitioned_kernel(divisor):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((128,), "bool"),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            if tx // divisor == 0:
                result[tx] = T.any_of(shared)

    return main


def _make_loop_any_of_kernel(loop_kind):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()

            if loop_kind == "constant":
                for _ in T.serial(4):
                    result[tx] = T.any_of(shared)
            elif loop_kind == "lane_extent":
                for _ in T.serial(tx + 1):
                    result[tx] = T.any_of(shared)
            elif loop_kind == "warp_extent":
                for _ in T.serial(tx // 32 + 1):
                    result[tx] = T.any_of(shared)
            elif loop_kind == "alternating_extent":
                for _ in T.serial(tx % 2 + 1):
                    result[tx] = T.any_of(shared)
            else:
                raise ValueError(loop_kind)

    return main


def _make_bound_condition_kernel(binding_kind):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()

            if binding_kind == "direct":
                lane = T.bind(tx % 2)
                if lane == 0:
                    result[tx] = T.any_of(shared)
            elif binding_kind == "chained":
                lane = T.bind(tx % 2)
                predicate = T.bind(lane == 0)
                if predicate:
                    result[tx] = T.any_of(shared)
            elif binding_kind == "warp_group":
                warp = T.bind(tx // 32)
                if warp == 0:
                    result[tx] = T.any_of(shared)
            else:
                raise ValueError(binding_kind)

    return main


def _make_if_then_else_any_of_kernel(condition_kind):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()

            if condition_kind == "single_lane":
                result[tx] = T.if_then_else(tx == 0, T.any_of(shared), False)
            elif condition_kind == "warp_group":
                result[tx] = T.if_then_else(tx // 32 == 0, T.any_of(shared), False)
            else:
                raise ValueError(condition_kind)

            result[tx] = T.any_of(shared)

    return main


def _make_let_condition_kernel(condition_kind):
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            predicate = tvm.tirx.Var("predicate", "bool")

            if condition_kind == "single_lane":
                value = tx == 0
            elif condition_kind == "warp_group":
                value = tx // 32 == 0
            else:
                raise ValueError(condition_kind)

            result[tx] = tvm.tirx.Let(
                predicate,
                value,
                T.if_then_else(predicate, T.any_of(shared), False),
            )

    return main


def _resolve(func, target=_CUDA_TARGET):
    mod = tvm.IRModule({"main": func})
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return tilelang.transform.ResolveLogicalScope()(mod)["main"]


def _logical_scopes(func, op_name="tl.any_of"):
    scopes = []
    logical_op = op.Op.get(op_name)

    def collect(node):
        if isinstance(node, tvm.tirx.Call) and node.op.same_as(logical_op):
            assert len(node.args) == 3
            scope = node.args[2]
            assert isinstance(scope, tvm.tirx.StringImm)
            scopes.append(scope.value)

    post_order_visit(func.body, collect)
    return scopes


@pytest.mark.parametrize(
    ("condition_kind", "expected_scope"),
    [
        ("none", "warp"),
        ("single_lane", "thread"),
        ("warp_group", "warp"),
        ("alternating", "thread"),
    ],
)
def test_resolve_any_of_auto_scope(condition_kind, expected_scope):
    resolved = _resolve(_make_any_of_kernel(condition_kind))
    assert _logical_scopes(resolved) == [expected_scope]


@pytest.mark.parametrize(
    ("divergent", "expected_scope"),
    [(False, "warp"), (True, "thread")],
)
def test_resolve_all_of_auto_scope(divergent, expected_scope):
    resolved = _resolve(_make_all_of_kernel(divergent))
    assert _logical_scopes(resolved, "tl.all_of") == [expected_scope]


def test_resolve_restores_scope_after_divergent_branch():
    resolved = _resolve(_make_nested_any_of_kernel())
    assert _logical_scopes(resolved) == ["thread", "warp"]


@pytest.mark.parametrize(
    ("target", "divisor", "expected_scope"),
    [
        (_ROCM_TARGET, 64, "warp"),
        (_ROCM_TARGET, 32, "thread"),
        (_ROCM_WAVE32_TARGET, 32, "warp"),
        (_ROCM_WAVE32_TARGET, 16, "thread"),
    ],
)
def test_resolve_uses_target_warp_size(target, divisor, expected_scope):
    resolved = _resolve(_make_wave_partitioned_kernel(divisor), target)
    assert _logical_scopes(resolved) == [expected_scope]


def test_resolve_global_uniform_range_uses_warp():
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            result[tx] = T.any_of(source)

    resolved = _resolve(main)
    assert _logical_scopes(resolved) == ["warp"]


def test_resolve_lane_dependent_shared_range_stays_thread():
    @T.prim_func
    def main(
        source: T.Tensor((128,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((128,), "int32")
            tx = T.get_thread_binding()
            result[tx] = T.any_of(shared[tx : tx + 8])

    resolved = _resolve(main)
    assert _logical_scopes(resolved) == ["thread"]


def test_frontend_logical_call_keeps_two_argument_form():
    func = _make_any_of_kernel("none")
    argument_counts = []
    logical_op = op.Op.get("tl.any_of")

    def collect(node):
        if isinstance(node, tvm.tirx.Call) and node.op.same_as(logical_op):
            argument_counts.append(len(node.args))

    post_order_visit(func.body, collect)
    assert argument_counts == [2]


def test_resolve_partial_warp_launch_stays_thread():
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((48,), "bool"),
    ):
        with T.Kernel(1, threads=48):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            result[tx] = T.any_of(shared)

    resolved = _resolve(main)
    assert _logical_scopes(resolved) == ["thread"]


@pytest.mark.parametrize(
    ("loop_kind", "expected_scope"),
    [
        ("constant", "warp"),
        ("lane_extent", "thread"),
        ("warp_extent", "warp"),
        ("alternating_extent", "thread"),
    ],
)
def test_resolve_auto_scope_in_loop(loop_kind, expected_scope):
    resolved = _resolve(_make_loop_any_of_kernel(loop_kind))
    assert _logical_scopes(resolved) == [expected_scope]


@pytest.mark.parametrize(
    ("binding_kind", "expected_scope"),
    [
        ("direct", "thread"),
        ("chained", "thread"),
        ("warp_group", "warp"),
    ],
)
def test_resolve_auto_scope_through_bindings(binding_kind, expected_scope):
    resolved = _resolve(_make_bound_condition_kernel(binding_kind))
    assert _logical_scopes(resolved) == [expected_scope]


@pytest.mark.parametrize(
    ("condition_kind", "expected_scope"),
    [
        ("single_lane", "thread"),
        ("warp_group", "warp"),
    ],
)
def test_resolve_auto_scope_in_if_then_else(condition_kind, expected_scope):
    resolved = _resolve(_make_if_then_else_any_of_kernel(condition_kind))
    assert _logical_scopes(resolved) == [expected_scope, "warp"]


def test_resolve_local_buffer_condition_stays_thread():
    @T.prim_func
    def main(
        source: T.Tensor((70,), "int32"),
        result: T.Tensor((64,), "bool"),
    ):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            predicate = T.alloc_local((1,), "bool")
            tx = T.get_thread_binding()

            predicate[0] = tx == 0
            if predicate[0]:
                result[tx] = T.any_of(shared)

    resolved = _resolve(main)
    assert _logical_scopes(resolved) == ["thread"]


@pytest.mark.parametrize(
    ("condition_kind", "expected_scope"),
    [
        ("single_lane", "thread"),
        ("warp_group", "warp"),
    ],
)
def test_resolve_auto_scope_through_let(condition_kind, expected_scope):
    resolved = _resolve(_make_let_condition_kernel(condition_kind))
    assert _logical_scopes(resolved) == [expected_scope]
