import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.rocm.target import with_rocm_target_attrs
from tilelang.testing.ir import collect_calls, collect_extern_calls


_CUDA_TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
_ROCM_TARGET = with_rocm_target_attrs(tvm.target.Target({"kind": "hip", "mcpu": "gfx90a"}))
_ROCM_WAVE32_TARGET = with_rocm_target_attrs(tvm.target.Target({"kind": "hip", "mcpu": "gfx1100"}))


def _make_logical_kernel(reduction="any_of", threads=64, scope="shared", lane_dependent=False):
    logical_op = getattr(T, reduction)

    @T.prim_func
    def main(result: T.Tensor((threads,), "bool")):
        with T.Kernel(1, threads=threads):
            values = T.alloc_buffer((128,), "int32", scope=scope)
            tx = T.get_thread_binding()
            if lane_dependent:
                result[tx] = logical_op(values[tx : tx + 8])
            else:
                result[tx] = logical_op(values)

    return main


def _make_conditional_kernel(form, divisor):
    @T.prim_func
    def main(result: T.Tensor((128,), "bool")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            if form == "branch":
                if tx // divisor == 0:
                    result[tx] = T.any_of(shared)
            elif form == "bind":
                group = T.bind(tx // divisor)
                predicate = T.bind(group == 0)
                if predicate:
                    result[tx] = T.any_of(shared)
            elif form == "let":
                predicate = tvm.tirx.Var("predicate", "bool")
                result[tx] = tvm.tirx.Let(predicate, tx // divisor == 0, T.if_then_else(predicate, T.any_of(shared), False))
            elif form == "select":
                result[tx] = T.if_then_else(tx // divisor == 0, T.any_of(shared), False)
            elif form == "local_predicate":
                predicate = T.alloc_local((1,), "bool")
                predicate[0] = tx == 0
                if predicate[0]:
                    result[tx] = T.any_of(shared)

            # Check that divergence does not leak into the following statement.
            result[tx] = T.any_of(shared)

    return main


def _make_loop_kernel(loop_kind):
    @T.prim_func
    def main(result: T.Tensor((64,), "bool")):
        with T.Kernel(1, threads=64):
            shared = T.alloc_shared((70,), "int32")
            tx = T.get_thread_binding()
            if loop_kind == "uniform":
                for _ in T.serial(tx // 32 + 1):
                    result[tx] = T.any_of(shared)
            elif loop_kind == "divergent":
                for _ in T.serial(tx + 1):
                    result[tx] = T.any_of(shared)
            elif loop_kind == "while":
                iteration = T.alloc_local((1,), "int32")
                iteration[0] = 0
                while iteration[0] < tx + 1:
                    result[tx] = T.any_of(shared)
                    iteration[0] += 1
            result[tx] = T.any_of(shared)

    return main


def _resolve(func, target=_CUDA_TARGET):
    mod = tvm.IRModule({"main": func})
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return tilelang.transform.ResolveLogicalScope()(mod)["main"]


def _logical_scopes(func, op_name="tl.any_of"):
    scopes = []
    for call in collect_calls(func, op=op_name):
        assert len(call.args) == 3
        assert isinstance(call.args[2], tvm.tirx.StringImm)
        scopes.append(call.args[2].value)
    return scopes


@pytest.mark.parametrize("reduction", ["any_of", "all_of"])
@pytest.mark.parametrize(("threads", "expected_scope"), [(64, "warp"), (48, "thread")])
def test_resolve_logical_scope(reduction, threads, expected_scope):
    func = _make_logical_kernel(reduction, threads)
    assert _logical_scopes(func, f"tl.{reduction}") == ["auto"]
    assert _logical_scopes(_resolve(func), f"tl.{reduction}") == [expected_scope]


@pytest.mark.parametrize(
    ("form", "divisor", "expected_scope"),
    [
        ("branch", 1, "thread"),
        ("branch", 32, "warp"),
        ("bind", 1, "thread"),
        ("bind", 32, "warp"),
        ("let", 1, "thread"),
        ("let", 32, "warp"),
        ("select", 1, "thread"),
        ("select", 32, "warp"),
        ("local_predicate", 1, "thread"),
    ],
)
def test_resolve_conditional_scope_and_restore(form, divisor, expected_scope):
    resolved = _resolve(_make_conditional_kernel(form, divisor))
    assert _logical_scopes(resolved) == [expected_scope, "warp"]


@pytest.mark.parametrize(
    ("target", "divisor", "expected_scope"),
    [
        (_ROCM_TARGET, 64, "warp"),
        (_ROCM_TARGET, 32, "thread"),
        (_ROCM_WAVE32_TARGET, 32, "warp"),
    ],
)
def test_resolve_uses_target_warp_size(target, divisor, expected_scope):
    resolved = _resolve(_make_conditional_kernel("branch", divisor), target)
    assert _logical_scopes(resolved) == [expected_scope, "warp"]


def test_resolve_without_target_warp_size_lowers_to_thread_helper():
    resolved = _resolve(_make_logical_kernel(), tvm.target.Target("hip"))
    mod = tvm.IRModule({"main": resolved})
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.LowerIntrin()(mod)
    assert len(collect_extern_calls(mod["main"], symbol="tl::Any")) == 1
    assert collect_extern_calls(mod["main"], symbol="tl::AnyWarp") == []


@pytest.mark.parametrize(
    ("loop_kind", "expected_scope"),
    [("uniform", "warp"), ("divergent", "thread"), ("while", "thread")],
)
def test_resolve_loop_scope_and_restore(loop_kind, expected_scope):
    resolved = _resolve(_make_loop_kernel(loop_kind))
    assert _logical_scopes(resolved) == [expected_scope, "warp"]


@pytest.mark.parametrize(
    ("scope", "lane_dependent"),
    [("shared", True), ("local", False), ("local.fragment", False)],
)
def test_resolve_nonuniform_access_stays_thread(scope, lane_dependent):
    resolved = _resolve(_make_logical_kernel(scope=scope, lane_dependent=lane_dependent))
    assert _logical_scopes(resolved) == ["thread"]


def test_resolve_global_uniform_range_uses_warp():
    @T.prim_func
    def main(source: T.Tensor((70,), "int32"), result: T.Tensor((64,), "bool")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            result[tx] = T.any_of(source)

    assert _logical_scopes(_resolve(main)) == ["warp"]
