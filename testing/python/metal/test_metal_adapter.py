"""Metal adapter and ABI regression coverage.

Covers the runtime contracts required by local transformer workloads:

- scalar binding: ``out_idx`` + scalar interleave (scalar first/middle/tail x
  single/multi output) must not crash and must bind correctly on real MPS.
- dynamic shape: dynamic ``out_idx`` shape resolution keyed by ``tirx.Var`` identity
  (shared ``T.const``, expression dims like ``N + 1``, multiple outputs,
  alias/non-alias inputs, undetermined symbol -> explicit error).
- global buffer: compiler-generated buffers (``T.alloc_global``) bind from the lowered
  host allocation semantics; ``T.const`` + ``T.empty`` + ``alloc_global``
  compiles and runs on Metal.
- launch order: multi-kernel launch plan follows host call-site order (producer ->
  consumer, reversed function map, repeated same symbol).
- keepalive: an exception after the first successful enqueue still establishes a
  completion fence and keeps submitted buffers pinned.
- completion: completed batches release their strong refs without a second launch
  (background reaper), including the adapter destruction path.

Every test executes on the real MPS device; failing = ABI/adapter regression.
"""

import gc
import threading
import time
import weakref
from unittest import mock

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
from tvm import tirx

from tilelang import tvm as tvm
from tilelang.jit.adapter.torch import metal as metal_mod
from tilelang.jit.adapter.torch.metal import MetalKernelAdapter

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="PyTorch MPS device is required",
)

MPS = torch.device("mps")


def _wait_keepalive_drained(timeout: float = 15.0) -> bool:
    """Wait until the module-global keepalive queue is empty."""
    deadline = time.time() + timeout
    while metal_mod._pending_keepalive:
        if time.time() > deadline:
            return False
        time.sleep(0.02)
    return True


# ---------------------------------------------------------------------------
# scalar binding: out_idx + scalar interleave — scalar first / middle / tail x
# single / multi output. All six cases must bind and run on real MPS.
# ---------------------------------------------------------------------------
@T.prim_func
def scalar_binding_scalar_first(S: T.int32, A: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] * T.cast(S, T.float32)


@T.prim_func
def scalar_binding_scalar_middle(A: T.Tensor((64,), "float32"), S: T.int32, B: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] * T.cast(S, T.float32) + B[bx]


@T.prim_func
def scalar_binding_scalar_tail(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32"), S: T.int32, OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] + B[bx] + T.cast(S, T.float32)


@T.prim_func
def scalar_binding_multi_scalar_first(
    S: T.int32,
    X: T.Tensor((32,), "float32"),
    Y: T.Tensor((32,), "float32"),
    OUT1: T.Tensor((32,), "float32"),
    OUT2: T.Tensor((32,), "float32"),
):
    with T.Kernel(32) as bx:
        OUT1[bx] = X[bx] * T.cast(S, T.float32)
        OUT2[bx] = Y[bx] / (T.cast(S, T.float32) + 1.0)


@T.prim_func
def scalar_binding_multi_scalar_middle(
    X: T.Tensor((32,), "float32"),
    S: T.int32,
    Y: T.Tensor((32,), "float32"),
    OUT1: T.Tensor((32,), "float32"),
    OUT2: T.Tensor((32,), "float32"),
):
    with T.Kernel(32) as bx:
        OUT1[bx] = X[bx] + T.cast(S, T.float32)
        OUT2[bx] = X[bx] * Y[bx] - T.cast(S, T.float32)


@T.prim_func
def scalar_binding_multi_scalar_tail(
    X: T.Tensor((32,), "float32"),
    Y: T.Tensor((32,), "float32"),
    S: T.int32,
    OUT1: T.Tensor((32,), "float32"),
    OUT2: T.Tensor((32,), "float32"),
):
    with T.Kernel(32) as bx:
        OUT1[bx] = X[bx] * 2.0 + T.cast(S, T.float32)
        OUT2[bx] = Y[bx] * 3.0 - T.cast(S, T.float32)


def _scalar_binding_first(S, g):
    A = g.normal(size=(64,)).astype(np.float32)
    return (scalar_binding_scalar_first, [2], (S, torch.from_numpy(A).to(MPS)), (A * S,))


def _scalar_binding_middle(S, g):
    A = g.normal(size=(64,)).astype(np.float32)
    B = g.normal(size=(64,)).astype(np.float32)
    return (scalar_binding_scalar_middle, [3], (torch.from_numpy(A).to(MPS), S, torch.from_numpy(B).to(MPS)), (A * S + B,))


def _scalar_binding_tail(S, g):
    A = g.normal(size=(64,)).astype(np.float32)
    B = g.normal(size=(64,)).astype(np.float32)
    return (scalar_binding_scalar_tail, [3], (torch.from_numpy(A).to(MPS), torch.from_numpy(B).to(MPS), S), (A + B + S,))


def _scalar_binding_multi_first(S, g):
    X = g.normal(size=(32,)).astype(np.float32)
    Y = g.normal(size=(32,)).astype(np.float32)
    return (
        scalar_binding_multi_scalar_first,
        [3, 4],
        (S, torch.from_numpy(X).to(MPS), torch.from_numpy(Y).to(MPS)),
        (X * S, Y / (S + 1.0)),
    )


def _scalar_binding_multi_middle(S, g):
    X = g.normal(size=(32,)).astype(np.float32)
    Y = g.normal(size=(32,)).astype(np.float32)
    return (scalar_binding_multi_scalar_middle, [3, 4], (torch.from_numpy(X).to(MPS), S, torch.from_numpy(Y).to(MPS)), (X + S, X * Y - S))


def _scalar_binding_multi_tail(S, g):
    X = g.normal(size=(32,)).astype(np.float32)
    Y = g.normal(size=(32,)).astype(np.float32)
    return (
        scalar_binding_multi_scalar_tail,
        [3, 4],
        (torch.from_numpy(X).to(MPS), torch.from_numpy(Y).to(MPS), S),
        (X * 2.0 + S, Y * 3.0 - S),
    )


SCALAR_BINDING_CASES = [
    _scalar_binding_first,
    _scalar_binding_middle,
    _scalar_binding_tail,
    _scalar_binding_multi_first,
    _scalar_binding_multi_middle,
    _scalar_binding_multi_tail,
]


@pytest.mark.parametrize("build", SCALAR_BINDING_CASES, ids=["first", "middle", "tail", "multi_first", "multi_middle", "multi_tail"])
def test_scalar_binding_scalar_interleave_out_idx(build):
    S = 3
    g = np.random.default_rng(23)
    prim, out_idx, call_args, expected = build(S, g)
    kern = tilelang.compile(prim, out_idx=out_idx, execution_backend="torch", target="metal")
    got = kern(*call_args)
    if isinstance(got, (list, tuple)):
        got = tuple(got)
    else:
        got = (got,)
    torch.mps.synchronize()
    for g_, e_ in zip(got, expected):
        err = np.abs(g_.cpu().numpy() - e_).max()
        assert err < 1e-4, f"[scalar binding {prim.attrs['global_symbol']}] binding mismatch: max_abs_err={err}"


# ---------------------------------------------------------------------------
# dynamic shape: dynamic out_idx shape resolution (Var identity, PrimExpr dims,
# shared symbols, multiple outputs, alias/non-alias, explicit errors).
# ---------------------------------------------------------------------------
@tilelang.jit
def dynamic_shape_shared_const(A, block: int = 128):
    N = T.const("N")
    A: T.Tensor[[N], T.float32]
    OUT = T.empty([N], dtype=T.float32)
    with T.Kernel(N, threads=block) as bx:
        OUT[bx] = A[bx] * 2.0
    return OUT


@tilelang.jit
def dynamic_shape_expr_dim(A, block: int = 128):
    N = T.const("N")
    A: T.Tensor[[N], T.float32]
    OUT = T.empty([N + 1], dtype=T.float32)
    with T.Kernel(N + 1, threads=block) as bx:
        OUT[bx] = A[T.min(bx, N - 1)] * 2.0
    return OUT


@tilelang.jit
def dynamic_shape_multi_output(X, Y, block: int = 128):
    N = T.const("N")
    X: T.Tensor[[N], T.float32]
    Y: T.Tensor[[N], T.float32]
    OUT1 = T.empty([N], dtype=T.float32)
    OUT2 = T.empty([N + 1], dtype=T.float32)
    with T.Kernel(N, threads=block) as bx:
        OUT1[bx] = X[bx] + Y[bx]
    with T.Kernel(N + 1, threads=block) as bx:
        OUT2[bx] = X[T.min(bx, N - 1)] * Y[T.min(bx, N - 1)]
    return (OUT1, OUT2)


@tilelang.jit
def dynamic_shape_alias_inputs(X, Y, block: int = 128):
    N = T.const("N")
    X: T.Tensor[[N], T.float32]
    Y: T.Tensor[[N], T.float32]
    OUT = T.empty([N], dtype=T.float32)
    with T.Kernel(N, threads=block) as bx:
        OUT[bx] = X[bx] * Y[bx] + X[bx]
    return OUT


def test_dynamic_shape_dynamic_out_idx_shared_const():
    a = torch.randn(97, device=MPS)
    out = dynamic_shape_shared_const(a)
    torch.mps.synchronize()
    assert out.shape == (97,)
    assert torch.allclose(out, a * 2.0, atol=1e-5)


def test_dynamic_shape_dynamic_out_idx_expr_dim():
    a = torch.randn(97, device=MPS)
    out = dynamic_shape_expr_dim(a)
    torch.mps.synchronize()
    assert out.shape == (98,), f"expected N+1=98, got {out.shape}"
    assert torch.allclose(out[:97], a * 2.0, atol=1e-5)
    assert out[97].item() == pytest.approx(a[96].item() * 2.0, abs=1e-5)


def test_dynamic_shape_dynamic_out_idx_multi_output():
    x = torch.randn(65, device=MPS)
    y = torch.randn(65, device=MPS)
    o1, o2 = dynamic_shape_multi_output(x, y)
    torch.mps.synchronize()
    assert o1.shape == (65,)
    assert o2.shape == (66,)
    assert torch.allclose(o1, x + y, atol=1e-5)
    assert torch.allclose(o2[:65], x * y, atol=1e-5)


def test_dynamic_shape_dynamic_out_idx_alias_and_non_alias():
    x = torch.randn(73, device=MPS)
    # alias: same tensor bound to both inputs
    out = dynamic_shape_alias_inputs(x, x)
    torch.mps.synchronize()
    assert torch.allclose(out, x * x + x, atol=1e-5)
    # non-alias: distinct tensors
    y = torch.randn(73, device=MPS)
    out = dynamic_shape_alias_inputs(x, y)
    torch.mps.synchronize()
    assert torch.allclose(out, x * y + x, atol=1e-5)


def test_dynamic_shape_undetermined_symbol_raises():
    # Lazy-style prim_func whose output dim M is not tied to any input:
    # the launch must fail with an explicit error naming the symbol.
    N = tirx.Var("N", "int32")
    M = tirx.Var("M", "int32")

    @T.prim_func
    def f(A: T.Tensor((N,), "float32"), OUT: T.Tensor((M,), "float32")):
        with T.Kernel(N) as bx:
            OUT[bx] = A[bx] * 2.0

    kern = tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")
    a = torch.randn(31, device=MPS)
    with pytest.raises(RuntimeError, match="not determined by any caller-supplied tensor input shape"):
        kern(a)
    torch.mps.synchronize()


def test_dynamic_shape_conflicting_symbol_bindings_raise():
    # The same symbol bound to two inputs with different sizes must be
    # rejected explicitly instead of silently resolving to one of them.
    N = tirx.Var("N", "int32")

    @T.prim_func
    def f(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float32")):
        with T.Kernel(N) as bx:
            OUT[bx] = A[bx] + B[bx]

    kern = tilelang.compile(f, out_idx=[2], execution_backend="torch", target="metal")
    a = torch.randn(17, device=MPS)
    b = torch.randn(19, device=MPS)
    with pytest.raises(RuntimeError, match="conflicting sizes"):
        kern(a, b)
    torch.mps.synchronize()


# ---------------------------------------------------------------------------
# global buffer: compiler-generated buffers (T.alloc_global) — lazy + eagerjit.
# ---------------------------------------------------------------------------
@T.prim_func
def global_buffer_alloc_global_lazy(A: T.Tensor((97,), "float32"), B: T.Tensor((97,), "float32")):
    C = T.alloc_global((97,), "float32")
    with T.Kernel(97) as bx:
        C[bx] = A[bx] + 1.0
        B[bx] = C[bx] * 2.0


def test_global_buffer_alloc_global_lazy():
    a = torch.randn(97, device=MPS)
    kern = tilelang.compile(global_buffer_alloc_global_lazy, out_idx=[1], execution_backend="torch", target="metal")
    b = kern(a)
    torch.mps.synchronize()
    assert torch.allclose(b, (a + 1.0) * 2.0, atol=1e-5)


@tilelang.jit
def global_buffer_alloc_global_eagerjit(A, block_N, dtype):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    B = T.empty([N], dtype=dtype)
    C = T.alloc_global([N], dtype)
    with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
        T.copy(A[bx * block_N : (bx + 1) * block_N], C[bx * block_N : (bx + 1) * block_N])
        T.copy(C[bx * block_N : (bx + 1) * block_N], B[bx * block_N : (bx + 1) * block_N])
    return B


def test_global_buffer_alloc_global_eagerjit():
    a = torch.randn(1024, device=MPS, dtype=torch.float16)
    b = global_buffer_alloc_global_eagerjit(a, 128, "float16")
    torch.mps.synchronize()
    assert torch.allclose(b, a, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# launch order: launch plan follows host call-site order (producer -> consumer,
# reversed function map, repeated same symbol).
# ---------------------------------------------------------------------------
@T.prim_func
def launch_order_two_stage(A: T.Tensor((97,), "float32"), B: T.Tensor((97,), "float32"), C: T.Tensor((97,), "float32")):
    with T.Kernel(97) as bx:
        B[bx] = A[bx] + 1.0
    with T.Kernel(97) as bx:
        C[bx] = B[bx] * 2.0


def test_launch_order_producer_consumer_order():
    a = torch.randn(97, device=MPS)
    kern = tilelang.compile(launch_order_two_stage, out_idx=[1, 2], execution_backend="torch", target="metal")
    plan = kern.adapter._launch_plan()
    assert [s.symbol for s in plan] == ["launch_order_two_stage_kernel", "launch_order_two_stage_kernel_1"]
    b, c = kern(a)
    torch.mps.synchronize()
    assert torch.allclose(b, a + 1.0, atol=1e-5)
    assert torch.allclose(c, (a + 1.0) * 2.0, atol=1e-5)


def test_launch_order_reversed_function_map_still_host_order():
    """device_mod function-map order reversed vs host call sites: the launch
    plan must follow the HOST order; following the map would compute C from
    uninitialized B (NaN) and fail."""
    kern = tilelang.compile(launch_order_two_stage, out_idx=[1, 2], execution_backend="torch", target="metal")
    art = kern.artifact
    funcs = list(art.device_mod.functions.items())
    assert len(funcs) == 2
    assert [v.name_hint for v in art.device_mod.functions.keys()] == ["launch_order_two_stage_kernel", "launch_order_two_stage_kernel_1"]
    rev = tvm.IRModule()
    rev[funcs[1][0]] = funcs[1][1]  # insert kernel_1 FIRST -> reversed map
    rev[funcs[0][0]] = funcs[0][1]
    assert [v.name_hint for v in rev.functions.keys()] == ["launch_order_two_stage_kernel_1", "launch_order_two_stage_kernel"]

    adapter = MetalKernelAdapter(
        params=art.params,
        result_idx=[1, 2],
        func_or_mod=launch_order_two_stage,
        host_mod=art.host_mod,
        device_mod=rev,
        kernel_global_source=art.kernel_source,
    )
    plan = adapter._launch_plan()
    assert [s.symbol for s in plan] == ["launch_order_two_stage_kernel", "launch_order_two_stage_kernel_1"]

    a = torch.randn(97, device=MPS)
    b, c = adapter(a)
    torch.mps.synchronize()
    assert torch.allclose(b, a + 1.0, atol=1e-5)
    assert torch.allclose(c, (a + 1.0) * 2.0, atol=1e-5)


@T.prim_func
def launch_order_repeated_symbol(A: T.Tensor((97,), "float32"), OUT: T.Tensor((97,), "float32")):
    for _it in T.serial(2):
        with T.Kernel(97) as bx:
            OUT[bx] = A[bx] + 1.0


def test_launch_order_repeated_same_symbol():
    kern = tilelang.compile(launch_order_repeated_symbol, out_idx=[1], execution_backend="torch", target="metal")
    plan = kern.adapter._launch_plan()
    assert len(plan) == 2, "duplicate host call sites must be preserved"
    assert plan[0].symbol == plan[1].symbol == "launch_order_repeated_symbol_kernel"
    a = torch.randn(97, device=MPS)
    out = kern(a)
    torch.mps.synchronize()
    assert torch.allclose(out, a + 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# keepalive: exception after the first successful enqueue still establishes a
# completion fence and pins the submitted buffers.
# ---------------------------------------------------------------------------
def test_keepalive_aborted_launch_pins_submitted_buffers():
    real_compile_shader = torch.mps.compile_shader
    call_count = {"n": 0}

    class _RaisingModule:
        def __init__(self, mod):
            self._mod = mod

        def __getattr__(self, name):
            fn = getattr(self._mod, name)

            def _wrapped(*args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("injected second-launch failure")
                return fn(*args, **kwargs)

            return _wrapped

    def _compile_wrapper(source):
        return _RaisingModule(real_compile_shader(source))

    with mock.patch("torch.mps.compile_shader", side_effect=_compile_wrapper):
        kern = tilelang.compile(launch_order_two_stage, out_idx=[1, 2], execution_backend="torch", target="metal")

    a = torch.randn(97, device=MPS)
    with pytest.raises(RuntimeError, match="injected second-launch failure"):
        kern(a)
    # The first kernel was enqueued successfully: the batch must be pinned
    # (fence + strong refs) even though the launcher raised.
    assert metal_mod._pending_keepalive, "submitted buffers must stay pinned after an aborted launch"
    assert _wait_keepalive_drained(), "completion fence must fire and release the pinned batch"
    torch.mps.synchronize()


def test_keepalive_temporary_output_stress():
    """Caller-supplied outputs dropped immediately after launch:
    every batch must stay pinned across many launches, with no caller strong
    reference left behind."""
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    a = torch.randn(64, device=MPS)
    # Pause the reaper and neutralize the release helper so pinning is
    # observable while batches are pending: no drain can release the current
    # batch before the assertion runs.
    orig_release = metal_mod._release_finished_work
    orig_poll = metal_mod._KEEPALIVE_POLL_SECONDS
    metal_mod._release_finished_work = lambda: None
    metal_mod._KEEPALIVE_POLL_SECONDS = 3600.0
    weaks = []
    try:
        for i in range(30):
            # Vary the input to defeat allocator-caching aliasing.
            a.mul_(1.0 + i * 1e-6)
            a_i = a.clone()
            out = torch.empty(64, device=MPS)
            weaks.append(weakref.ref(out))
            kern(a, out)
            # No caller strong reference remains: the keepalive batch is the
            # ONLY strong ref. While the batch is pending the temp must be
            # alive (proves the allocator cannot reuse its memory), and it
            # must be the *current* batch (the pre-launch drain released the
            # previous one).
            del out
            assert metal_mod._pending_keepalive, f"iter {i}: batch must be pending"
            assert weaks[-1]() is not None, f"iter {i}: temp dropped while pending"
            assert torch.allclose(a, a_i, atol=0.0), f"iter {i}: input drift"
    finally:
        metal_mod._release_finished_work = orig_release
        metal_mod._KEEPALIVE_POLL_SECONDS = orig_poll
        metal_mod._reaper_wakeup.set()
    # Restored reaper: every batch releases once its event fires, and the
    # temps are freed without any second launch.
    assert _wait_keepalive_drained(), "all batches must be released after completion"
    torch.mps.synchronize()
    gc.collect()
    for i, w in enumerate(weaks):
        assert w() is None, f"iter {i}: temp must be released after completion"
    # The last output is still numerically correct end to end.
    out_last = torch.empty(64, device=MPS)
    kern(a, out_last)
    torch.mps.synchronize()
    assert torch.allclose(out_last, a + 1.0, atol=1e-5)
    del weaks
    gc.collect()


# ---------------------------------------------------------------------------
# completion: completed batches release strong refs without a second launch.
# ---------------------------------------------------------------------------
@T.prim_func
def completion_inplace(A: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] + 1.0


def test_completion_releases_without_second_launch():
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    a = torch.randn(64, device=MPS)
    out = torch.zeros(64, device=MPS)
    weak = weakref.ref(out)
    kern(a, out)
    torch.mps.synchronize()
    del out
    gc.collect()
    # No second launch: the background reaper must release the batch.
    assert _wait_keepalive_drained(), "batch must be released by the reaper, not by a next launch"
    gc.collect()
    assert weak() is None, "adapter must not hold the output tensor after completion"


def test_completion_adapter_destruction_path():
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    a = torch.randn(64, device=MPS)
    out = torch.zeros(64, device=MPS)
    weak = weakref.ref(out)
    kern(a, out)
    torch.mps.synchronize()
    del out
    del kern
    gc.collect()
    assert _wait_keepalive_drained(), "global queue must be reaped after adapter destruction"
    gc.collect()
    assert weak() is None


def test_completion_sequential_launches_do_not_accumulate():
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    a = torch.randn(64, device=MPS)
    for i in range(5):
        out = torch.zeros(64, device=MPS)
        kern(a, out)
        assert out[0].item() == pytest.approx(a[0].item() + 1.0, abs=1e-5)
        del out
        gc.collect()
        assert _wait_keepalive_drained(), f"queue must drain after launch {i}"
    torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Host call-site args are consumed per site,
# keepalive reaping is batch-atomic, host control flow is static-only, and
# the destruction path has weakref/finalizer evidence.
# ---------------------------------------------------------------------------
@T.prim_func
def host_plan_loop_carried(A: T.Tensor((97,), "float32"), OUT: T.Tensor((98,), "float32")):
    for _it in T.serial(2):
        with T.Kernel(96 + _it) as bx:
            OUT[bx] = A[bx] + T.cast(_it, T.float32)


def test_host_plan_loop_carried_args_and_geometry():
    """A repeated symbol called with different loop-carried
    scalar arguments and different per-site geometry. The loop variable must
    be substituted into the call-site function args and launch args, and the
    writes are non-idempotent (iter 1 overwrites iter 0 with +1.0)."""
    # No out_idx: the caller supplies OUT explicitly, so the never-written
    # tail element can be deterministically initialized.
    kern = tilelang.compile(host_plan_loop_carried, execution_backend="torch", target="metal")
    plan = kern.adapter._launch_plan()
    assert len(plan) == 2, "constant loop must expand to two call sites"
    assert plan[0].symbol == plan[1].symbol == "host_plan_loop_carried_kernel"
    # Per-site geometry from the substituted launch args: 96 then 97
    # (evaluated at launch time by the same resolver the launcher uses).
    analyzer = tvm.arith.Analyzer()
    assert metal_mod._resolve_int_value(plan[0].grid[0], {}, analyzer) == 96
    assert metal_mod._resolve_int_value(plan[1].grid[0], {}, analyzer) == 97
    # Per-site scalar args: the loop var becomes const 0 then const 1.
    assert [b.value for b in plan[0].bindings if b.kind == "const"] == [0]
    assert [b.value for b in plan[1].bindings if b.kind == "const"] == [1]
    # Numeric: the second launch (grid 97) overwrites [0, 97) with A+1.
    a = torch.randn(97, device=MPS)
    # Deterministic tail element: the adapter-owned
    # `torch.empty` output would leave out[97] uninitialized, so pass an
    # explicitly zeroed OUT; the never-written element then has defined
    # contents and the assertion cannot flake on allocator reuse.
    out = torch.zeros(98, device=MPS)
    kern(a, out)
    torch.mps.synchronize()
    assert torch.allclose(out[:97], a + 1.0, atol=1e-5)
    assert out[97].item() == 0.0  # never written: defined by the caller init


HOST_PLAN_N1 = tirx.Var("N", "int32")  # same name, first identity
HOST_PLAN_N2 = tirx.Var("N", "int32")  # same name, second identity


@T.prim_func
def host_plan_same_name_vars(
    A: T.Tensor((HOST_PLAN_N2,), "float32"),
    B: T.Tensor((HOST_PLAN_N1,), "float32"),
    OUT: T.Tensor((HOST_PLAN_N1,), "float32"),
):
    with T.Kernel(HOST_PLAN_N1) as bx:
        OUT[bx] = B[bx] * 2.0


def test_host_plan_same_name_distinct_identity_vars():
    """Two tirx.Vars with the same name but different identity.
    The symbol binding must use Var identity (grid + device scalar resolve to
    HOST_PLAN_N1 = 5), not the first string match (which would pick HOST_PLAN_N2 = 9)."""
    kern = tilelang.compile(host_plan_same_name_vars, out_idx=[2], execution_backend="torch", target="metal")
    plan = kern.adapter._launch_plan()
    site = plan[0]
    sym_bindings = [b for b in site.bindings if b.kind == "symbol"]
    assert len(sym_bindings) == 1
    assert sym_bindings[0].symbol.same_as(HOST_PLAN_N1), "symbol must keep Var identity"
    assert site.grid[0].same_as(HOST_PLAN_N1), "geometry must carry Var identity"

    # Mocked launch: the geometry must be 5 (HOST_PLAN_N1), not 9 (HOST_PLAN_N2).
    captured = {}

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            captured["args"] = fn_args
            captured["threads"] = kwargs["threads"]
            captured["group_size"] = kwargs["group_size"]

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with mock.patch("torch.mps.compile_shader", return_value=_MockModule()):
        kern_mock = tilelang.compile(host_plan_same_name_vars, out_idx=[2], execution_backend="torch", target="metal")
    a9 = torch.randn(9, device=MPS)
    b5 = torch.randn(5, device=MPS)
    kern_mock(a9, b5)
    assert captured["threads"] == [5 * 128, 1, 1], f"geometry must use HOST_PLAN_N1=5, got {captured['threads']}"
    packed = [x for x in captured["args"] if isinstance(x, torch.Tensor) and x.dtype == torch.uint8]
    assert len(packed) == 1, "exactly one packed args_t scalar buffer expected"
    slot0 = int(packed[0].cpu().numpy().view("<i4")[0])
    assert slot0 == 5, f"device scalar must be HOST_PLAN_N1=5, got {slot0}"

    # Real MPS end to end: same geometry, correct values.
    out = kern(a9, b5)
    torch.mps.synchronize()
    assert out.shape == (5,)
    assert torch.allclose(out, b5 * 2.0, atol=1e-5)


@T.prim_func
def host_plan_cond_branch(A: T.Tensor((97,), "float32"), OUT: T.Tensor((97,), "float32")):
    for _it in T.serial(2):
        if _it == 0:
            with T.Kernel(97) as bx:
                OUT[bx] = A[bx] + 1.0
        else:
            with T.Kernel(97) as bx:
                OUT[bx] = A[bx] * 2.0 + 1.0


def test_host_plan_static_conditional_taken_branch_only():
    """An IfThenElse whose condition is resolved by the substituted
    loop variable must walk only its taken branch (2 sites, not 4)."""
    kern = tilelang.compile(host_plan_cond_branch, out_idx=[1], execution_backend="torch", target="metal")
    plan = kern.adapter._launch_plan()
    assert len(plan) == 2, f"one branch per iteration expected, got {len(plan)} sites"
    assert plan[0].symbol == "host_plan_cond_branch_kernel"  # iter 0: then (A+1)
    assert plan[1].symbol == "host_plan_cond_branch_kernel_1"  # iter 1: else (A*2+1)
    a = torch.randn(97, device=MPS)
    out = kern(a)
    torch.mps.synchronize()
    assert torch.allclose(out, a * 2.0 + 1.0, atol=1e-5)


def test_host_plan_runtime_conditional_plan_build_error():
    """A conditional depending on a runtime value must be rejected
    at plan build time, never silently executed as both branches."""
    N = tirx.Var("N", "int32")

    @T.prim_func
    def f(A: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float32")):
        if N > 10:
            with T.Kernel(N) as bx:
                OUT[bx] = A[bx] + 1.0
        else:
            with T.Kernel(N) as bx:
                OUT[bx] = A[bx] * 2.0

    with pytest.raises(RuntimeError, match="cannot be resolved statically"):
        tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")


def test_host_plan_runtime_loop_plan_build_error():
    """A loop with runtime-dependent bounds must be rejected at
    plan build time."""
    N = tirx.Var("N", "int32")

    @T.prim_func
    def f(A: T.Tensor((N,), "float32"), OUT: T.Tensor((N,), "float32")):
        for _i in T.serial(N):
            with T.Kernel(N) as bx:
                OUT[bx] = A[bx] + 1.0

    with pytest.raises(RuntimeError, match="non-constant bounds"):
        tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")


def test_host_plan_call_site_args_swapped_binding():
    """The host calls the kernel with swapped buffer
    arguments (OUT, A) while the device parameters are (A, OUT). The plan
    must bind device param 0 to the call site's first actual argument (OUT,
    slot 1) and device param 1 to A (slot 0) -- never by device-parameter
    name."""
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    art = kern.artifact
    device_mod = art.device_mod
    assert "completion_inplace_kernel" in device_mod

    # Synthetic host module: FFI-style prologue binds handle Vars to args
    # slots, then calls the kernel with swapped buffers (OUT first, A second).
    struct_get = tvm.ir.Op.get("tirx.tvm_struct_get")
    call_packed = tvm.ir.Op.get("tirx.tvm_call_packed")
    args_var = tirx.Var("args", "handle")

    def sg(struct, index, field, dtype="handle"):
        index_e = index if isinstance(index, tirx.Var) else tirx.IntImm("int32", index)
        return tirx.Call(
            tvm.DataType(dtype),
            struct_get,
            [struct, index_e, tirx.IntImm("int32", field)],
        )

    a_h = tirx.Var("A_handle", "handle")
    out_h = tirx.Var("OUT_handle", "handle")
    a = tirx.Var("A", "handle")
    out = tirx.Var("OUT", "handle")
    call = tirx.Call(
        tvm.DataType("int32"),
        call_packed,
        [
            tirx.StringImm("completion_inplace_kernel"),
            out,  # swapped: device param 0 (A) receives OUT's buffer
            a,  # swapped: device param 1 (OUT) receives A's buffer
            tirx.IntImm("int32", 64),
            tirx.IntImm("int32", 128),
            tirx.IntImm("int32", 1),
            tirx.IntImm("int32", 1),
        ],
    )
    host_func = tirx.PrimFunc(
        [
            tirx.Var("self_handle", "handle"),
            args_var,
            tirx.Var("num_args", "int32"),
            tirx.Var("result", "handle"),
        ],
        tirx.SeqStmt(
            [
                tirx.Bind(a_h, sg(args_var, 0, 15)),
                tirx.Bind(out_h, sg(args_var, 1, 15)),
                tirx.Bind(a, sg(a_h, 0, 1)),
                tirx.Bind(out, sg(out_h, 0, 1)),
                tirx.Evaluate(call),
            ]
        ),
    ).with_attr("tirx.is_entry_func", True)
    host_mod = tvm.IRModule({tvm.ir.GlobalVar("completion_inplace_swapped"): host_func})

    adapter = MetalKernelAdapter(
        params=art.params,
        result_idx=[1],
        func_or_mod=completion_inplace,
        host_mod=host_mod,
        device_mod=device_mod,
        kernel_global_source=art.kernel_source,
    )
    plan = adapter._launch_plan()
    assert len(plan) == 1
    bindings = plan[0].bindings
    assert len(bindings) == 2, "device params (A, OUT)"
    assert bindings[0].kind == "user" and bindings[0].param_index == 1, (
        "device param A must bind the call site's first actual arg (OUT slot 1)"
    )
    assert bindings[1].kind == "user" and bindings[1].param_index == 0, (
        "device param OUT must bind the call site's second actual arg (A slot 0)"
    )


class _RendezvousEvent:
    """Fake event: query() blocks until two threads have both arrived, then
    returns a fixed answer. Extra arrivals (e.g. a racing reaper) are
    harmless."""

    def __init__(self, done):
        self._done = done
        self._lock = threading.Lock()
        self._arrived = 0
        self._go = threading.Event()

    def query(self):
        with self._lock:
            self._arrived += 1
            if self._arrived >= 2:
                self._go.set()
        assert self._go.wait(timeout=10), "rendezvous timeout"
        return self._done


def test_host_plan_concurrent_release_race():
    """Two threads release finished work concurrently. Both observe
    the completed head batch before either removes it; the still-running next
    batch must survive (never double-popleft)."""
    orig_release = metal_mod._release_finished_work
    orig_poll = metal_mod._KEEPALIVE_POLL_SECONDS
    # Pause/neutralize the background reaper so only our two threads act.
    metal_mod._release_finished_work = lambda: None
    metal_mod._KEEPALIVE_POLL_SECONDS = 3600.0
    saved = list(metal_mod._pending_keepalive)
    try:
        ev0 = _RendezvousEvent(True)  # head batch: finished
        ev1 = _RendezvousEvent(False)  # next batch: still running
        metal_mod._pending_keepalive.clear()
        metal_mod._pending_keepalive.append(((object(),), ev0))
        metal_mod._pending_keepalive.append(((object(),), ev1))

        errors = []

        def worker():
            try:
                orig_release()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"release threads raised: {errors}"
        remaining = list(metal_mod._pending_keepalive)
        assert len(remaining) == 1, f"exactly the unfinished batch must remain, got {len(remaining)}"
        assert remaining[0][1] is ev1, "the unfinished batch was released early"
    finally:
        metal_mod._pending_keepalive.clear()
        metal_mod._pending_keepalive.extend(saved)
        metal_mod._release_finished_work = orig_release
        metal_mod._KEEPALIVE_POLL_SECONDS = orig_poll


class _RaisingEvent:
    def query(self):
        raise RuntimeError("injected query failure")


def test_host_plan_query_error_sync_transition_drops_batch():
    """A head-of-line query exception must not pin the queue
    forever -- synchronize() proves completion and the batch is dropped."""
    orig_release = metal_mod._release_finished_work
    orig_poll = metal_mod._KEEPALIVE_POLL_SECONDS
    metal_mod._release_finished_work = lambda: None
    metal_mod._KEEPALIVE_POLL_SECONDS = 3600.0
    saved = list(metal_mod._pending_keepalive)
    try:
        metal_mod._pending_keepalive.clear()
        metal_mod._pending_keepalive.append(((object(),), _RaisingEvent()))
        with mock.patch("torch.mps.synchronize", return_value=None):
            orig_release()
        assert not metal_mod._pending_keepalive, "query error + sync must drop the batch"
        assert not metal_mod._stuck_keepalive
    finally:
        metal_mod._pending_keepalive.clear()
        metal_mod._pending_keepalive.extend(saved)
        metal_mod._release_finished_work = orig_release
        metal_mod._KEEPALIVE_POLL_SECONDS = orig_poll


class _FlakyEvent:
    """query() fails twice (teardown-style), then reports completion."""

    def __init__(self):
        self.calls = 0

    def query(self):
        self.calls += 1
        if self.calls <= 2:
            raise RuntimeError("transient query failure")
        return True


def test_host_plan_query_error_sync_failure_pins_stuck_then_retries():
    """When both query() and synchronize() fail during MPS teardown,
    the batch must stay pinned but leave the head-of-line path (stuck list);
    a later successful query releases it."""
    orig_release = metal_mod._release_finished_work
    orig_poll = metal_mod._KEEPALIVE_POLL_SECONDS
    metal_mod._release_finished_work = lambda: None
    metal_mod._KEEPALIVE_POLL_SECONDS = 3600.0
    saved = list(metal_mod._pending_keepalive)
    try:
        metal_mod._pending_keepalive.clear()
        flaky = _FlakyEvent()
        metal_mod._pending_keepalive.append(((object(),), flaky))
        with mock.patch("torch.mps.synchronize", side_effect=RuntimeError("sync failed")):
            orig_release()
        assert not metal_mod._pending_keepalive, "stuck batch leaves the active queue"
        assert len(metal_mod._stuck_keepalive) == 1, "batch stays pinned in the stuck list"
        # Next drain: query now succeeds -> released from the stuck list.
        orig_release()
        assert not metal_mod._stuck_keepalive, "stuck batch must be released on retry"
        assert flaky.calls >= 3
    finally:
        metal_mod._pending_keepalive.clear()
        metal_mod._stuck_keepalive.clear()
        metal_mod._pending_keepalive.extend(saved)
        metal_mod._release_finished_work = orig_release
        metal_mod._KEEPALIVE_POLL_SECONDS = orig_poll


def test_host_plan_adapter_destruction_weakref_finalizer():
    """The adapter's destruction path is provable: the object
    must be collectable after `del kern` with weakref + finalizer evidence
    (the launcher closure must not capture the adapter)."""
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    adapter = kern.adapter
    weak = weakref.ref(adapter)
    fired = []
    finalizer = weakref.finalize(adapter, lambda: fired.append(True))
    del adapter
    del kern
    gc.collect()
    assert weak() is None, "adapter must be collectable after kernel deletion"
    assert fired == [True], "adapter finalizer must run (provable __del__ path)"
    del finalizer
    gc.collect()


@tilelang.jit
def host_plan_k1_broadcast(w, block: int = 64):
    T_ = T.const("T_")
    w: T.Tensor[[T_, 1], T.float32]
    OUT = T.empty([T_], dtype=T.float32)
    with T.Kernel(T_, threads=block) as bx:
        OUT[bx] = w[bx, 0] * 2.0
    return OUT


def test_host_plan_rank1_fed_to_trailing_singleton_param():
    """A flat (m,) tensor fed to a declared (T_, 1) param is valid (torch
    right-aligned broadcasting; flat memory identical) and must not trip the
    scalar binding rank check."""
    w = torch.randn(7, device=MPS)
    out = host_plan_k1_broadcast(w)
    torch.mps.synchronize()
    assert out.shape == (7,)
    assert torch.allclose(out, w * 2.0, atol=1e-5)


def test_host_plan_rank_mismatch_non_singleton_still_raises():
    """The trailing-singleton relaxation must NOT accept a rank-1 tensor for
    a declared (T_, 2) param."""
    T_ = tirx.Var("T_", "int32")

    @T.prim_func
    def f(W: T.Tensor((T_, 2), "float32"), OUT: T.Tensor((T_,), "float32")):
        with T.Kernel(T_) as bx:
            OUT[bx] = W[bx, 0] * 2.0

    kern = tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")
    w = torch.randn(7, device=MPS)
    with pytest.raises(RuntimeError, match="declared rank 2"):
        kern(w)
    torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Retained-prefix dimension validation for trailing-singleton rank relaxation,
# nested static host loops, and deterministic output initialization.
# ---------------------------------------------------------------------------
def test_adapter_expr_rank_relax_rejects_retained_prefix_mismatch():
    """A declared (7, 1) parameter fed a (3,) tensor must
    be REJECTED.  The trailing 1 may be implicit, but the retained prefix (7)
    must equal the actual extent (3); accepting it would launch static
    geometry 7 over a three-element buffer (GPU out-of-bounds).  The
    rejection happens BEFORE any kernel launch: the mocked module observes
    zero calls (out-of-bounds guard)."""

    @T.prim_func
    def f(W: T.Tensor((7, 1), "float32"), OUT: T.Tensor((7,), "float32")):
        with T.Kernel(7) as bx:
            OUT[bx] = W[bx, 0] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with mock.patch("torch.mps.compile_shader", return_value=_MockModule()):
        kern = tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")
    w = torch.randn(3, device=MPS)
    with pytest.raises(RuntimeError, match="declared dimension"):
        kern(w)
    torch.mps.synchronize()


def test_adapter_expr_rank_matched_capacity_pattern_allowed():
    """Qwen MoE scope guard: a rank-matched declared
    capacity upper bound (64) fed a smaller actual tensor (15) is the legal
    padded/masked pattern (the kernel's accesses are internally
    masked/offset-guarded, so declared >= actual is safe and must NOT be
    rejected). The capacity dim must be explicitly declared
    (``T.annotate_capacity_dims``); without the declaration the same call
    is rejected (see ``test_capacity_eager_scalar_param_dim_unmarked_rejected``).
    Numerically: the masked kernel computes exactly the actual region and
    never touches the rest."""

    @tilelang.jit
    def masked_kernel(a, cap, m, block: int = 32):
        a: T.Tensor[[cap], T.float32]
        OUT = T.empty([cap], dtype=T.float32)
        T.annotate_capacity_dims({"a": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            if bx < m:  # masked: only the actual m elements are accessed
                OUT[bx] = a[bx] * 2.0
        return OUT

    a = torch.randn(15, device=MPS)
    out = masked_kernel(a, 64, 15)  # declared (64,), actual (15,)
    torch.mps.synchronize()
    assert out.shape == (64,), "declared capacity shape is kept"
    assert torch.allclose(out[:15], a * 2.0, atol=1e-5)
    # No scalar-arg packing involved (all baked at compile time): the tail
    # is never written, so a second call with the full capacity is also fine.
    a64 = torch.randn(64, device=MPS)
    out64 = masked_kernel(a64, 64, 64)
    torch.mps.synchronize()
    assert torch.allclose(out64, a64 * 2.0, atol=1e-5)


def test_adapter_expr_rank_relax_expression_prefix_validation():
    """A retained general-expression dimension (N + 1) must be
    validated against the actual extent after N is bound by another input:
    with A binding N=3, a (3,) tensor for W (declared N+1=4 != 3) is
    rejected, and a (4,) tensor (N+1=4) is accepted end to end."""
    N = tirx.Var("N", "int32")

    @T.prim_func
    def f(
        A: T.Tensor((N,), "float32"),
        W: T.Tensor((N + 1, 1), "float32"),
        OUT: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N) as bx:
            OUT[bx] = A[bx] + W[bx, 0]

    kern = tilelang.compile(f, out_idx=[2], execution_backend="torch", target="metal")
    a = torch.randn(3, device=MPS)  # binds N = 3 -> declared W prefix N+1 = 4
    # Wrong extent: N+1 = 4 != 3 must be rejected.
    w_bad = torch.randn(3, device=MPS)
    with pytest.raises(RuntimeError, match="declared dimension"):
        kern(a, w_bad)
    torch.mps.synchronize()
    # Legal: N=3 -> declared N+1=4 matches a (4,) tensor.
    w_ok = torch.randn(4, device=MPS)
    out = kern(a, w_ok)
    torch.mps.synchronize()
    assert out.shape == (3,)
    assert torch.allclose(out, a + w_ok[:3], atol=1e-5)


def test_adapter_expr_legal_broadcast_variants_pass():
    """The legal torch right-aligned broadcast shapes
    for a declared (T_, 1) param must all pass: (m,) and (m, 1)."""
    w = torch.randn(7, device=MPS)
    assert torch.allclose(host_plan_k1_broadcast(w), w * 2.0, atol=1e-5)
    torch.mps.synchronize()
    w2 = w.view(7, 1)
    assert torch.allclose(host_plan_k1_broadcast(w2), w * 2.0, atol=1e-5)
    torch.mps.synchronize()


@T.prim_func
def adapter_expr_nested_static_loops(A: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
    # Loop vars are carried through the LAUNCH GEOMETRY only (not as scalar
    # kernel args): per-site grid = 8 + i + j.  (Runtime multi-scalar args
    # are tested separately below; this kernel isolates static expansion and
    # per-site substitution through launch geometry.)
    for _i in T.serial(2, 4):  # outer: min=2 (nonzero), extent=2
        for _j in T.serial(_i + 1):  # inner bound references the outer var
            with T.Kernel(8 + _i + _j) as bx:
                OUT[bx] = A[bx] + 1.0


def test_adapter_expr_nested_static_host_loops_expand():
    """A nested loop whose inner bound
    references the outer constant iteration variable is statically
    enumerable.  Outer min=2 (nonzero) x inner extents {3, 4} -> exactly 7
    call sites, with BOTH loop variables substituted into the per-site
    launch geometry (grid = 8 + i + j -> 10,11,12,11,12,13,14); the old
    walker rejected this as a 'non-constant bounds' runtime loop."""
    # No out_idx: the caller supplies OUT explicitly (deterministic init,
    # deterministic caller initialization).
    kern = tilelang.compile(adapter_expr_nested_static_loops, execution_backend="torch", target="metal")
    plan = kern.adapter._launch_plan()
    assert len(plan) == 7, f"nested static expansion expected 7 sites, got {len(plan)}"
    analyzer = tvm.arith.Analyzer()
    grids = [metal_mod._resolve_int_value(site.grid[0], {}, analyzer) for site in plan]
    assert grids == [10, 11, 12, 11, 12, 13, 14], f"outer/inner loop vars must substitute into per-site geometry, got {grids}"
    # Numeric end to end: every site writes A+1 over [0, 8+i+j); the last
    # site (i=3, j=3, grid 14) wins, and out[14] stays caller-initialized
    # (a wrong expansion or wrong grid would leave a different pattern).
    a = torch.randn(64, device=MPS)
    out = torch.zeros(64, device=MPS)
    kern(a, out)
    torch.mps.synchronize()
    assert torch.allclose(out[:14], a[:14] + 1.0, atol=1e-5)
    assert out[14:].abs().max().item() == 0.0, "sites must not write past grid 14"


# ---------------------------------------------------------------------------
# Exact-versus-capacity declaration discipline.
# declaration discipline.  Rank matching is NO LONGER a validation
# criterion: every declared dimension of every tensor input is validated
# exactly (constants and general expressions must equal the caller's actual
# extent; Var dims bind by identity) UNLESS the dimension is explicitly
# declared as a capacity dimension in the compiled contract
# (``tilelang_capacity_dims`` PrimFunc attr).
#
# Capacity marking is explicit opt-in only.
# EXPLICIT opt-in ONLY.  The syntactic auto-inference from tensor
# annotations (a dim that directly references a scalar function parameter
# was auto-exempted) is removed: ``B_q(E, N, (K+1)//2)`` had its ordinary
# exact dims E/N auto-exempted, which could let a smaller caller buffer
# through.  Eager kernels declare capacity dims in the body via
# ``T.annotate_capacity_dims({"A_q": (0,)})``; lazy kernels opt in via
# ``func.with_attr("tilelang_capacity_dims", {"W": (0,)})``.  Everything
# unmarked stays strictly validated.  Explicit capacity dims accept EITHER
# mismatch direction at launch -- declared > actual (padded/masked, the
# Qwen MoE QMM pattern) or declared < actual (active-prefix processing of a
# larger allocation, the activation-quantization pattern) -- and the adapter runs an
# advisory guard audit (warning when a marked dim's accesses show no
# mask/offset guard evidence).
# ---------------------------------------------------------------------------
def test_strict_shape_rank_matched_constant_mismatch_rejected():
    """A plain unmarked rank-matched declared
    (7,) fed a (3,) tensor MUST be rejected BEFORE any kernel launch (mock
    module observes zero calls).  Under adapter expression the rank-matched branch was
    blanket-exempt, so this launched 7 elements over a three-element buffer
    (GPU out-of-bounds)."""

    @T.prim_func
    def f(W: T.Tensor((7,), "float32"), OUT: T.Tensor((7,), "float32")):
        with T.Kernel(7) as bx:
            OUT[bx] = W[bx] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with mock.patch("torch.mps.compile_shader", return_value=_MockModule()):
        kern = tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")
        with pytest.raises(RuntimeError, match="declared dimension"):
            kern(torch.randn(3))  # CPU tensor: rejection precedes any device use
    torch.mps.synchronize()


def test_strict_shape_rank_matched_rank1_constant_mismatch_rejected():
    """Rank-matched (7, 1) fed (3, 1) must be rejected: no trailing-singleton
    relaxation applies (ranks match), and the unmarked constant dim 7 != 3 is
    a real out-of-bounds hazard."""

    @T.prim_func
    def f(W: T.Tensor((7, 1), "float32"), OUT: T.Tensor((7,), "float32")):
        with T.Kernel(7) as bx:
            OUT[bx] = W[bx, 0] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with mock.patch("torch.mps.compile_shader", return_value=_MockModule()):
        kern = tilelang.compile(f, out_idx=[1], execution_backend="torch", target="metal")
        with pytest.raises(RuntimeError, match="declared dimension"):
            kern(torch.randn(3, 1))
    torch.mps.synchronize()


def test_strict_shape_rank_matched_expression_mismatch_rejected():
    """Rank-matched general-expression dim (N + 1) fed a (3,) tensor with
    N=3 must be rejected (declared N+1=4 != 3), and the matching (4,) tensor
    must pass end to end.  This is the rank-matched twin of the adapter expression
    relaxed-prefix expression test: rank matching must NOT exempt it."""

    N = tirx.Var("N", "int32")

    @T.prim_func
    def f(
        A: T.Tensor((N,), "float32"),
        W: T.Tensor((N + 1,), "float32"),
        OUT: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(N) as bx:
            OUT[bx] = A[bx] + W[bx]

    kern = tilelang.compile(f, out_idx=[2], execution_backend="torch", target="metal")
    a = torch.randn(3, device=MPS)  # binds N = 3 -> declared W = N + 1 = 4
    w_bad = torch.randn(3, device=MPS)
    with pytest.raises(RuntimeError, match="declared dimension"):
        kern(a, w_bad)
    torch.mps.synchronize()
    w_ok = torch.randn(4, device=MPS)
    out = kern(a, w_ok)
    torch.mps.synchronize()
    assert out.shape == (3,)
    assert torch.allclose(out, a + w_ok[:3], atol=1e-5)


def test_strict_shape_eager_literal_dim_mismatch_rejected():
    """An eagerjit tensor annotation with a LITERAL dim (7) is an exact
    declaration (no scalar-parameter reference), so a (3,) tensor must be
    rejected before launch.  (Two separate wrappers: the eagerjit kernel
    cache keys on compile-time values only, so a kernel whose shader module
    was built under the mock must not be reused for the real launch.)"""

    @tilelang.jit
    def lit_kernel_bad(a):
        a: T.Tensor[[7], T.float32]
        OUT = T.empty([7], dtype=T.float32)
        with T.Kernel(7) as bx:
            OUT[bx] = a[bx]
        return OUT

    @tilelang.jit
    def lit_kernel_ok(a):
        a: T.Tensor[[7], T.float32]
        OUT = T.empty([7], dtype=T.float32)
        with T.Kernel(7) as bx:
            OUT[bx] = a[bx]
        return OUT

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        pytest.raises(RuntimeError, match="declared dimension"),
    ):
        lit_kernel_bad(torch.randn(3))  # CPU tensor: rejection precedes device use
    torch.mps.synchronize()
    # Real MPS launch with the matching literal extent.
    a = torch.randn(7, device=MPS)
    out = lit_kernel_ok(a)
    torch.mps.synchronize()
    assert out.shape == (7,)
    assert torch.allclose(out, a, atol=1e-6)


def test_strict_shape_eager_literal_rank1_mismatch_rejected():
    """Round-2 HIGH guard in the eager path: an eagerjit (7, 1) literal
    declaration fed a flat (3,) tensor is rank-relaxed but the retained
    literal prefix 7 != 3 -> rejected (the relaxation must not weaken the
    retained-prefix validation for eager kernels either)."""

    @tilelang.jit
    def lit_kernel2(w):
        w: T.Tensor[[7, 1], T.float32]
        OUT = T.empty([7], dtype=T.float32)
        with T.Kernel(7) as bx:
            OUT[bx] = w[bx, 0] * 2.0
        return OUT

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        pytest.raises(RuntimeError, match="declared dimension"),
    ):
        lit_kernel2(torch.randn(3))
    torch.mps.synchronize()


def test_strict_shape_eager_capacity_scalar_param_2d_allowed():
    """The Qwen MoE capacity pattern in 2D: declared (cap, k) with cap/k direct
    scalar-parameter references, actual (15, 8) -> accepted; the masked
    kernel computes exactly the actual region and never touches the rest
    (numeric verification on real MPS).  capacity: the capacity dims are
    declared EXPLICITLY via ``T.annotate_capacity_dims`` (the annotation
    alone no longer exempts anything -- see
    ``test_capacity_eager_scalar_param_dim_unmarked_rejected``)."""

    @tilelang.jit
    def masked_kernel2d(a, out, cap, k, m, block: int = 32):
        a: T.Tensor[[cap, k], T.float32]
        out: T.Tensor[[cap, k], T.float32]
        T.annotate_capacity_dims({"a": (0,), "out": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            if bx < m:
                for j in T.serial(k):
                    out[bx, j] = a[bx, j] * 2.0

    a = torch.randn(15, 8, device=MPS)
    # Caller-supplied OUT keeps the masked tail deterministically initialized:
    # the masked tail must stay caller-zeros.
    out = torch.zeros(64, 8, device=MPS)
    masked_kernel2d(a, out, 64, 8, 15)  # declared (64, 8), actual (15, 8)
    torch.mps.synchronize()
    assert out.shape == (64, 8), "declared capacity shape is kept"
    assert torch.allclose(out[:15], a * 2.0, atol=1e-5)
    assert out[15:].abs().max().item() == 0.0, "masked tail must never be written"


def test_strict_shape_capacity_dim_scoped_other_dims_strict():
    """Per-dim scoping: (cap, 8) declares dim0 as an EXPLICIT capacity dim
    (``T.annotate_capacity_dims``, capacity) but dim1 as an exact literal.  A
    mismatched dim1 is rejected even though dim0 is a legal capacity
    mismatch; a matching dim1 (with any actual dim0 <= cap) is accepted."""

    @tilelang.jit
    def scope_kernel(a, cap, m, block: int = 32):
        a: T.Tensor[[cap, 8], T.float32]
        OUT = T.empty([cap, 8], dtype=T.float32)
        T.annotate_capacity_dims({"a": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            if bx < m:
                for j in T.serial(8):
                    OUT[bx, j] = a[bx, j] * 2.0
        return OUT

    out = scope_kernel(torch.randn(15, 8, device=MPS), 64, 15)
    torch.mps.synchronize()
    assert out.shape == (64, 8)
    with pytest.raises(RuntimeError, match="declared dimension"):
        scope_kernel(torch.randn(15, 4, device=MPS), 64, 15)
    torch.mps.synchronize()
    with pytest.raises(RuntimeError, match="declared dimension"):
        scope_kernel(torch.randn(15, 16, device=MPS), 64, 15)
    torch.mps.synchronize()


def test_strict_shape_explicit_capacity_attr_lazy_allowed():
    """Explicit opt-in channel for lazy @T.prim_func kernels: a param whose
    dim is declared as a capacity dim via
    ``func.with_attr("tilelang_capacity_dims", {"W": (0,)})`` accepts an
    actual (3,) buffer (kernel internally masked to m=3); the identical
    unmarked kernel rejects it."""

    @T.prim_func
    def fcap(W: T.Tensor((7,), "float32"), OUT: T.Tensor((7,), "float32")):
        with T.Kernel(7) as bx:
            if bx < 3:  # masked: only the actual 3 elements are accessed
                OUT[bx] = W[bx] * 2.0

    marked = fcap.with_attr("tilelang_capacity_dims", {"W": (0,)})
    # No out_idx: the caller supplies OUT explicitly (deterministic init,
    # the masked tail must stay caller-initialized to zero).
    kern = tilelang.compile(marked, execution_backend="torch", target="metal")
    w = torch.randn(3, device=MPS)
    out = torch.zeros(7, device=MPS)
    kern(w, out)
    torch.mps.synchronize()
    assert torch.allclose(out[:3], w * 2.0, atol=1e-5), "masked region must be computed"
    assert out[3:].abs().max().item() == 0.0, "masked tail must never be written"
    # The unmarked twin must reject the same call.
    kern2 = tilelang.compile(fcap, execution_backend="torch", target="metal")
    with pytest.raises(RuntimeError, match="declared dimension"):
        kern2(torch.randn(3, device=MPS), torch.zeros(7, device=MPS))
    torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Capacity marking is explicit opt-in only.
# EXPLICIT opt-in only.  The syntactic auto-inference (a tensor annotation
# dim that directly references a scalar function parameter was auto-marked
# as a capacity dim, exempting ordinary exact dims such as B_q.E/N from
# validation) is removed.  Regression matrix:
#   1. unmarked annotation dims that reference scalar params -> validated
#      exactly (Qwen MoE A_q rows=64 vs actual 15 -> REJECT);
#   2. explicitly declared capacity + mask/offset guard -> accepted (64 vs
#      15, numeric, tail stays caller-zeros);
#   3. explicit capacity WITHOUT guard evidence -> adapter warning;
#   4. explicit capacity with actual > declared -> accepted (declared
#      R=15 active prefix over a larger 64-row allocation; rows beyond
#      declared are never touched, see
#      test_capacity_eager_capacity_actual_larger_than_declared_allowed);
#   5. lazy ``with_attr`` opt-in unchanged (test_strict_shape_explicit_capacity_attr_lazy_allowed).
# ---------------------------------------------------------------------------
def test_capacity_eager_scalar_param_dim_unmarked_rejected():
    """An eager-JIT tensor annotation
    whose dim references a scalar function parameter is an EXACT
    declaration unless the author explicitly declares capacity.  Declared
    (64, 8) fed an actual (15, 8) tensor MUST be rejected BEFORE any launch
    (mock observes zero calls).  This is the exact Qwen MoE hazard: A_q's
    ordinary rows dim (and B_q's E/N in the round-4 evidence dump) must be
    validated per-dim."""

    @tilelang.jit
    def unmarked_kernel(a, out, cap, k, m, block: int = 32):
        a: T.Tensor[[cap, k], T.float32]
        out: T.Tensor[[cap, k], T.float32]
        with T.Kernel(cap, threads=block) as bx:
            if bx < m:
                for j in T.serial(k):
                    out[bx, j] = a[bx, j] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        pytest.raises(RuntimeError, match="declared dimension"),
    ):
        unmarked_kernel(torch.randn(15, 8), torch.zeros(64, 8), 64, 8, 15)
    torch.mps.synchronize()


def test_capacity_eager_explicit_capacity_masked_allowed():
    """The same kernel with an explicit
    ``T.annotate_capacity_dims`` declaration accepts declared (64, 8) vs
    actual (15, 8), computes exactly the masked region, and never touches
    the caller-zeros tail (real MPS launch)."""

    @tilelang.jit
    def marked_kernel(a, out, cap, k, m, block: int = 32):
        a: T.Tensor[[cap, k], T.float32]
        out: T.Tensor[[cap, k], T.float32]
        T.annotate_capacity_dims({"a": (0,), "out": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            if bx < m:
                for j in T.serial(k):
                    out[bx, j] = a[bx, j] * 2.0

    a = torch.randn(15, 8, device=MPS)
    out = torch.zeros(64, 8, device=MPS)
    marked_kernel(a, out, 64, 8, 15)  # declared (64, 8), actual (15, 8)
    torch.mps.synchronize()
    assert out.shape == (64, 8), "declared capacity shape is kept"
    assert torch.allclose(out[:15], a * 2.0, atol=1e-5)
    assert out[15:].abs().max().item() == 0.0, "masked tail must never be written"


def test_capacity_qwen_moe_shape_probe():
    """Qwen MoE representative shape: A_q declared with rows=64,
    K=11) via scalar params, caller supplies (15, 11).  Unmarked -> REJECT
    (per-dim validation, no exemption); explicitly marked (capacity dim 0)
    -> ACCEPT with numeric verification.  Two separate wrappers: the
    eagerjit kernel cache keys on compile-time values only, so the mock-
    compiled (rejecting) wrapper must never be reused for the real launch."""

    @tilelang.jit
    def qwen_moe_unmarked(A_q, OUT, rows, K, m, block: int = 32):
        A_q: T.Tensor[[rows, K], T.float32]
        OUT: T.Tensor[[rows, K], T.float32]
        with T.Kernel(rows, threads=block) as bx:
            if bx < m:
                for j in T.serial(K):
                    OUT[bx, j] = A_q[bx, j] * 2.0

    @tilelang.jit
    def qwen_moe_marked(A_q, OUT, rows, K, m, block: int = 32):
        A_q: T.Tensor[[rows, K], T.float32]
        OUT: T.Tensor[[rows, K], T.float32]
        T.annotate_capacity_dims({"A_q": (0,), "OUT": (0,)})
        with T.Kernel(rows, threads=block) as bx:
            if bx < m:
                for j in T.serial(K):
                    OUT[bx, j] = A_q[bx, j] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched for a mismatched shape")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        pytest.raises(RuntimeError, match="declared dimension"),
    ):
        qwen_moe_unmarked(torch.randn(15, 11), torch.zeros(64, 11), 64, 11, 15)
    torch.mps.synchronize()
    # Explicit capacity declaration -> 64 vs 15 is legal (masked access).
    a = torch.randn(15, 11, device=MPS)
    out = torch.zeros(64, 11, device=MPS)
    qwen_moe_marked(a, out, 64, 11, 15)
    torch.mps.synchronize()
    assert torch.allclose(out[:15], a * 2.0, atol=1e-5)
    assert out[15:].abs().max().item() == 0.0, "masked tail must never be written"


def test_capacity_eager_capacity_actual_larger_than_declared_allowed():
    """The activation-quantization pattern (declared < actual): a capacity dim whose
    declared extent is the ACTIVE prefix (grid = declared, so every access
    is bounded by declared) may receive a LARGER actual buffer (the padded
    allocation) -- the rows beyond declared are simply never touched.
    Real MPS launch: declared 15, actual 64-row buffer; out[:15] computed,
    out[15:] stays caller-zeros.  (The guard audit is advisory and may warn
    here: the accesses are bounded by the declared grid, not by an extra
    mask.)"""

    @tilelang.jit
    def prefix_kernel(x, out, R, cap, block: int = 32):
        x: T.Tensor[[R], T.float32]
        out: T.Tensor[[R], T.float32]
        T.annotate_capacity_dims({"x": (0,), "out": (0,)})
        with T.Kernel(R, threads=block) as bx:
            out[bx] = x[bx] * 2.0

    # declared R=15 (active prefix): the kernel grid is R, so every access
    # is bounded by declared; the caller passes LARGER 64-row allocations
    # (rows >= 15 are garbage / caller-zeros and are never touched).
    x = torch.randn(64, device=MPS)
    out = torch.zeros(64, device=MPS)
    prefix_kernel(x, out, 15, 64)
    torch.mps.synchronize()
    assert torch.allclose(out[:15], x[:15] * 2.0, atol=1e-5)
    assert out[15:].abs().max().item() == 0.0, "rows beyond declared must stay untouched"


def test_capacity_eager_capacity_guard_audit_warns_unguarded(caplog):
    """An explicitly declared capacity dim whose accesses show no
    mask/offset guard evidence (unconditional full sweep over the declared
    extent) is a hazard: the adapter emits an advisory warning at
    compilation time (the explicit declaration is the trust boundary, so
    the audit warns rather than rejects)."""

    @tilelang.jit
    def unguarded_kernel(a, out, cap, m, block: int = 32):
        a: T.Tensor[[cap], T.float32]
        out: T.Tensor[[cap], T.float32]
        T.annotate_capacity_dims({"a": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            out[bx] = a[bx] * 2.0  # NO guard: sweeps the full declared extent

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    import logging

    # Compile-only path: the audit runs at adapter construction (compile
    # time), so no launch is attempted and the mock observes zero calls.
    pf = unguarded_kernel.get_tir(torch.zeros(15), torch.zeros(64), 64, 15)
    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        caplog.at_level(logging.WARNING, logger="tilelang.jit.adapter.torch.metal"),
    ):
        tilelang.compile(pf, execution_backend="torch", target="metal")
    assert any("no mask/offset guard evidence" in r.message and "'a'" in r.message for r in caplog.records), (
        f"expected guard-audit warning, got {[r.message for r in caplog.records]}"
    )


def test_capacity_eager_capacity_guard_audit_silent_when_guarded(caplog):
    """A masked access conditioned on a runtime
    bound ``m``) is structural guard evidence, so no warning is emitted."""

    @tilelang.jit
    def guarded_kernel(a, out, cap, m, block: int = 32):
        a: T.Tensor[[cap], T.float32]
        out: T.Tensor[[cap], T.float32]
        T.annotate_capacity_dims({"a": (0,), "out": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            if bx < m:  # runtime-bounded mask
                out[bx] = a[bx] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    import logging

    pf = guarded_kernel.get_tir(torch.zeros(15), torch.zeros(64), 64, 15)
    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        caplog.at_level(logging.WARNING, logger="tilelang.jit.adapter.torch.metal"),
    ):
        tilelang.compile(pf, execution_backend="torch", target="metal")
    assert not any("no mask/offset guard evidence" in r.message for r in caplog.records), (
        f"unexpected guard-audit warning: {[r.message for r in caplog.records]}"
    )


def test_capacity_eager_capacity_guard_against_declared_extent_warns(caplog):
    """Audit adversarial: a guard against the DECLARED extent itself (``bx
    < cap``, i.e. a no-op for any smaller actual buffer) is not guard
    evidence and must still warn.  Guards only count when their bound is
    not the declared extent (a runtime bound or a smaller baked bound)."""

    @tilelang.jit
    def noop_guard_kernel(a, out, cap, m, block: int = 32):
        a: T.Tensor[[cap], T.float32]
        out: T.Tensor[[cap], T.float32]
        T.annotate_capacity_dims({"a": (0,)})
        with T.Kernel(cap, threads=block) as bx:
            if bx < cap:  # bound == declared extent: no-op guard
                out[bx] = a[bx] * 2.0

    class _MockFn:
        def __call__(self, *fn_args, **kwargs):
            raise AssertionError("kernel must not be launched")

    class _MockModule:
        def __getattr__(self, name):
            return _MockFn()

    import logging

    pf = noop_guard_kernel.get_tir(torch.zeros(15), torch.zeros(64), 64, 15)
    with (
        mock.patch("torch.mps.compile_shader", return_value=_MockModule()),
        caplog.at_level(logging.WARNING, logger="tilelang.jit.adapter.torch.metal"),
    ):
        tilelang.compile(pf, execution_backend="torch", target="metal")
    assert any("no mask/offset guard evidence" in r.message and "'a'" in r.message for r in caplog.records), (
        f"expected guard-audit warning, got {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# adapter expression §4: multi-runtime-scalar packing (≥2 runtime scalar kernel
# arguments).  The Metal codegen packs every scalar parameter into a single
# ``args_t`` struct at ``buffer(num_buffer)``, but the
# ``torch.mps.compile_shader`` launcher binds each positional argument to
# its own buffer slot, so pre-fix only the first scalar landed in the struct
# and the rest silently read stale bytes (``kern(a, 5, 7)`` -> ``a + 5000``
# instead of ``a + 5007``).  The adapter must pack all runtime scalars into
# one buffer reproducing the struct layout.  All cases run on real MPS.
# ---------------------------------------------------------------------------
@T.prim_func
def adapter_expr_two_scalar_middle(A: T.Tensor((64,), "float32"), scalar_i: T.int32, J: T.int32, OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] + T.cast(scalar_i, T.float32) * 1000.0 + T.cast(J, T.float32)


@T.prim_func
def adapter_expr_two_scalar_first(scalar_i: T.int32, J: T.int32, A: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] + T.cast(scalar_i, T.float32) * 1000.0 + T.cast(J, T.float32)


@T.prim_func
def adapter_expr_int_float_scalar_tail(A: T.Tensor((64,), "float32"), scalar_i: T.int32, F: T.float32, OUT: T.Tensor((64,), "float32")):
    with T.Kernel(64) as bx:
        OUT[bx] = A[bx] + T.cast(scalar_i, T.float32) * 1000.0 + F


def _adapter_expr_run(prim, out_idx, call_args, expected):
    kern = tilelang.compile(prim, out_idx=out_idx, execution_backend="torch", target="metal")
    got = kern(*call_args)
    torch.mps.synchronize()
    err = np.abs(got.cpu().numpy() - expected).max()
    assert err < 1e-4, f"[{prim.attrs['global_symbol']}] multi-runtime-scalar mismatch: max_abs_err={err}"


def test_adapter_expr_two_runtime_scalars_middle():
    """Two runtime scalars in middle positions must produce ``a + 5007``."""
    a = torch.from_numpy(np.arange(64, dtype=np.float32)).to(MPS)
    _adapter_expr_run(adapter_expr_two_scalar_middle, [3], (a, 5, 7), np.arange(64, dtype=np.float32) + 5007.0)


def test_adapter_expr_two_runtime_scalars_first():
    """Two runtime scalars in the leading public positions (FFI slot chain)."""
    a = torch.from_numpy(np.arange(64, dtype=np.float32)).to(MPS)
    _adapter_expr_run(adapter_expr_two_scalar_first, [3], (5, 7, a), np.arange(64, dtype=np.float32) + 5007.0)


def test_adapter_expr_int_and_float_runtime_scalars_tail():
    """int32 + float32 runtime scalars pack into their 8-byte slots."""
    a = torch.from_numpy(np.arange(64, dtype=np.float32)).to(MPS)
    _adapter_expr_run(adapter_expr_int_float_scalar_tail, [3], (a, 5, 7.5), np.arange(64, dtype=np.float32) + 5007.5)


# ---------------------------------------------------------------------------
# Adapter boundary-condition regressions
# ---------------------------------------------------------------------------
def test_loop_local_bind_snapshot():
    """A tirx.Bind inside a static host loop must be snapshotted per call
    site. The loop rebinds OUT to args slot 0 then
    slot 1; without the snapshot both sites resolve the final (slot 1)
    binding and the plan would enqueue two identical launches."""
    kern = tilelang.compile(completion_inplace, execution_backend="torch", target="metal")
    art = kern.artifact
    device_mod = art.device_mod
    assert "completion_inplace_kernel" in device_mod

    struct_get = tvm.ir.Op.get("tirx.tvm_struct_get")
    call_packed = tvm.ir.Op.get("tirx.tvm_call_packed")
    args_var = tirx.Var("args", "handle")

    def sg(struct, index, field, dtype="handle"):
        index_e = index if isinstance(index, tirx.Var) else tirx.IntImm("int32", index)
        return tirx.Call(
            tvm.DataType(dtype),
            struct_get,
            [struct, index_e, tirx.IntImm("int32", field)],
        )

    a_h = tirx.Var("A_handle", "handle")
    v_h = tirx.Var("V_handle", "handle")
    a = tirx.Var("A", "handle")
    out = tirx.Var("OUT", "handle")
    it = tirx.Var("it", "int32")
    call = tirx.Call(
        tvm.DataType("int32"),
        call_packed,
        [
            tirx.StringImm("completion_inplace_kernel"),
            a,
            out,
            tirx.IntImm("int32", 64),
            tirx.IntImm("int32", 128),
            tirx.IntImm("int32", 1),
            tirx.IntImm("int32", 1),
        ],
    )
    host_func = tirx.PrimFunc(
        [
            tirx.Var("self_handle", "handle"),
            args_var,
            tirx.Var("num_args", "int32"),
            tirx.Var("result", "handle"),
        ],
        tirx.SeqStmt(
            [
                tirx.Bind(a_h, sg(args_var, 0, 15)),
                tirx.Bind(a, sg(a_h, 0, 1)),
                tirx.For(
                    it,
                    0,
                    2,
                    tirx.ForKind.SERIAL,
                    tirx.SeqStmt(
                        [
                            tirx.Bind(v_h, sg(args_var, it, 15)),
                            tirx.Bind(out, sg(v_h, 0, 1)),
                            tirx.Evaluate(call),
                        ]
                    ),
                ),
            ]
        ),
    ).with_attr("tirx.is_entry_func", True)
    host_mod = tvm.IRModule({tvm.ir.GlobalVar("completion_inplace_loop_bind"): host_func})

    adapter = MetalKernelAdapter(
        params=art.params,
        result_idx=[1],
        func_or_mod=completion_inplace,
        host_mod=host_mod,
        device_mod=device_mod,
        kernel_global_source=art.kernel_source,
    )
    plan = adapter._launch_plan()
    assert len(plan) == 2, "static loop must expand to two call sites"
    # Device params (A, OUT): OUT binds args slot 0 on iteration 0 and slot
    # 1 on iteration 1; A is constant (slot 0) on both sites.
    assert plan[0].bindings[0].param_index == 0
    assert plan[1].bindings[0].param_index == 0
    assert plan[0].bindings[1].kind == "user" and plan[0].bindings[1].param_index == 0
    assert plan[1].bindings[1].kind == "user" and plan[1].bindings[1].param_index == 1


def test_resolve_int_value_preserves_declared_dtype():
    """General PrimExpr substitution must build each integer replacement
    with the variable's declared dtype. Hardcoded int32
    replacements made an int64 shape symbol fail with
    InternalError: substituting n:int64 -> 41:int32 before
    simplification."""
    analyzer = tvm.arith.Analyzer()
    n64 = tirx.Var("n", "int64")
    assert metal_mod._resolve_int_value(n64 + tirx.IntImm("int64", 1), {n64: 41}, analyzer) == 42
    s64 = tirx.Var("s", "int64")
    assert metal_mod._resolve_int_value(s64 * 2, {}, analyzer, full=[5], scalar_vars={s64: 0}) == 10
    u64 = tirx.Var("u", "uint64")
    assert metal_mod._resolve_int_value(u64 + tirx.IntImm("uint64", 3), {u64: 7}, analyzer) == 10
    i32 = tirx.Var("i", "int32")
    assert metal_mod._resolve_int_value(i32 + 5, {i32: 3}, analyzer) == 8


def test_annotate_capacity_dims_rejects_unknown_name():
    """Unknown capacity-dimension names must raise at declaration time
    instead of being silently dropped by the adapter (a
    typo would otherwise leave the intended dim under strict validation and
    fail later with a confusing declared-dimension mismatch)."""

    @tilelang.jit
    def bad_capacity(a, m):
        a: T.Tensor((m, 8), "float32")
        T.annotate_capacity_dims({"a_q": (0,)})  # typo: parameter is a

    with pytest.raises(ValueError, match="unknown tensor parameter"):
        bad_capacity(torch.randn(64, 8, device=MPS), 64)

    with pytest.raises(ValueError, match="unknown tensor parameter"):

        @T.prim_func
        def bad_capacity_lazy(A: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
            T.annotate_capacity_dims({"B": (0,)})
            for i in T.serial(64):
                OUT[i] = A[i]

    @T.prim_func
    def valid_capacity_attr(A: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
        with T.Kernel(64) as i:
            OUT[i] = A[i]

    bad_attr = valid_capacity_attr.with_attr("tilelang_capacity_dims", {"B": (0,)})
    with pytest.raises(ValueError, match="unknown tensor parameter"):
        tilelang.compile(bad_attr, out_idx=[1], execution_backend="torch", target="metal")
