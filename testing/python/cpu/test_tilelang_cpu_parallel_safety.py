"""CPU parallel safety regressions; shared proofs avoid a target cross product.

GEMM and atomic integration tests cover both execution backends. These tests
retain the P1 counterexamples and direct-TIR boundaries missing from those tests.
"""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm
from tvm import tirx


def _compile_serial(func):
    kernel = tilelang.compile(
        func,
        target="c",
        execution_backend="cython",
        out_idx=-1,
        pass_configs={"tl.cpu_parallel": True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()
    return kernel


def _apply(func, target="c"):
    mod = tvm.IRModule.from_expr(func.with_attr({"target": tvm.target.Target(target), "global_symbol": "main"}))
    result = tilelang.cpu.transform.MaterializeCPUParallelGrid()(mod)["main"]
    kinds = []
    tirx.stmt_functor.post_order_visit(
        result.body, lambda n: kinds.append(n.kind) if isinstance(n, tirx.For) and n.loop_var.name == "bx" else None
    )
    assert len(kinds) == 1
    return kinds[0]


def test_while_reset_stays_serial(capfd):
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        s = T.alloc_buffer((1,), "int32", scope="local")
        with T.Kernel(32, cpu_num_threads=4) as bx:
            k = T.alloc_buffer((1,), "int32", scope="local")
            k[0] = bx
            while k[0] == 0:
                s[0] = 0
                k[0] = 1
            s[0] = s[0] + A[bx]
            B[bx] = s[0]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.ones(32, dtype=torch.int32)), torch.arange(1, 33, dtype=torch.int32))
    assert "stays serial" in capfd.readouterr().err


def test_read_before_covering_loop_finishes_stays_serial():
    @T.prim_func
    def main(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        s = T.alloc_buffer((2,), "int32", scope="local")
        with T.Kernel(16, cpu_num_threads=4) as bx:
            if bx == 0:
                s[0] = 0
                s[1] = 0
            for j in T.serial(2):
                s[j] = A[bx * 2 + j]
                B[bx * 2 + j] = s[(j + 1) % 2]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.arange(1, 33, dtype=torch.int32)), torch.arange(32, dtype=torch.int32))


@pytest.mark.parametrize("reset_kind", ["empty_outer", "predicate", "block_predicate", "complete"])
def test_direct_reset_control_flow(reset_kind):
    # Initialization in a nested complete loop must not escape a possibly
    # empty outer loop or a predicated region. While and partial initialization
    # already have numerical regressions through the full CPU pipeline.
    bx = tirx.Var("bx", "int32")
    j = tirx.Var("j", "int32")
    t = tirx.Var("t", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    s = tirx.decl_buffer((4,), "int32", name="s", scope="local")
    zero = tirx.IntImm("int32", 0)

    def fill(value, predicate=None):
        return tirx.For(j, 0, 4, tirx.ForKind.SERIAL, tirx.BufferStore(s, value, [j], predicate=predicate))

    reset = fill(zero)
    if reset_kind == "empty_outer":
        reset = tirx.For(t, 0, bx, tirx.ForKind.SERIAL, reset)
    elif reset_kind == "predicate":
        reset = fill(zero, bx == 0)
    elif reset_kind == "block_predicate":
        reset = tirx.SBlockRealize([], bx == 0, tirx.SBlock([], [], [], "reset", reset))
    grid = tirx.For(
        bx,
        0,
        16,
        tirx.ForKind.SERIAL,
        tirx.SeqStmt(
            [
                tirx.IfThenElse(bx == 0, fill(zero), None),
                reset,
                tirx.BufferStore(B, tirx.BufferLoad(s, [1]), [bx]),
                fill(bx + 1),
            ]
        ),
        annotations={"tl.cpu_grid_dim": 0},
    )
    func = tirx.PrimFunc([B.data], tirx.SeqStmt([tirx.AllocBuffer(s), grid]), buffer_map={B.data: B})
    expected = tirx.ForKind.PARALLEL if reset_kind == "complete" else tirx.ForKind.SERIAL
    assert _apply(func) == expected


def test_stateful_extern_without_pointer_stays_serial(capfd):
    @T.prim_func
    def main(A: T.Tensor((1024,), "int32"), B: T.Tensor((1024,), "int32")):
        with T.Kernel(
            1024,
            cpu_num_threads=4,
            prelude='extern "C" int cpu_grid_counter() { static int value = 0; return value++; }\n',
        ) as bx:
            B[bx] = T.call_extern("int32", "cpu_grid_counter") + A[bx]

    kernel = _compile_serial(main)
    torch.testing.assert_close(kernel(torch.zeros(1024, dtype=torch.int32)), torch.arange(1024, dtype=torch.int32))
    assert "effects that cannot be proven safe" in capfd.readouterr().err


@pytest.mark.parametrize("pure", [False, True])
def test_direct_extern_effects(pure, capfd):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    call = tirx.call_pure_extern("int32", "pure_scalar", bx) if pure else tirx.call_extern("int32", "counter")
    grid = tirx.For(bx, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(B, call, [bx]), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data], grid, buffer_map={B.data: B})
    assert _apply(func, "llvm") == (tirx.ForKind.PARALLEL if pure else tirx.ForKind.SERIAL)
    if not pure:
        assert "effects that cannot be proven safe" in capfd.readouterr().err


@pytest.mark.parametrize(
    "target,cast_dtype,extent,parallel",
    [
        ("c", "int8", 512, False),
        ("llvm", "int8", 512, False),
        ("c", "int8", 128, True),
        ("c", "int8", None, False),
        ("llvm", "int64", 512, True),
    ],
)
def test_cast_range_proof(target, cast_dtype, extent, parallel):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((512,), "int32", name="B")
    n = tirx.Var("n", "int32")
    index = tirx.Cast(cast_dtype, bx)
    body = tirx.IfThenElse(index >= 0, tirx.BufferStore(B, bx, [index]), None)
    grid = tirx.For(bx, 0, n if extent is None else extent, tirx.ForKind.SERIAL, body, annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data, n] if extent is None else [B.data], grid, buffer_map={B.data: B})
    assert _apply(func, target) == (tirx.ForKind.PARALLEL if parallel else tirx.ForKind.SERIAL)


@pytest.mark.parametrize("mixed_casts", [False, True])
def test_casts_preserve_iterator_identity(mixed_casts):
    bx = tirx.Var("bx", "int32")
    B = tirx.decl_buffer((16,), "int32", name="B")
    cast_bx = tirx.Cast("int16", bx)
    if mixed_casts:
        # Distinct cast dtypes of one iterator must not become independent.
        index = tirx.Cast("int32", cast_bx) - tirx.Cast("int32", tirx.Cast("int8", bx))
    else:
        # Retaining raw bx as a free parameter would hide this collision.
        index = tirx.Cast("int32", cast_bx) - bx
    grid = tirx.For(bx, 0, 16, tirx.ForKind.SERIAL, tirx.BufferStore(B, bx, [index]), annotations={"tl.cpu_grid_dim": 0})
    func = tirx.PrimFunc([B.data], grid, buffer_map={B.data: B})
    assert _apply(func) == tirx.ForKind.SERIAL
