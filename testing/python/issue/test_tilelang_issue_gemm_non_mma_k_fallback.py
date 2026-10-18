import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_swizzled_layout


def _check_fma_shared_bounds(mod):
    tir = tvm.tirx

    @tir.functor.visitor
    class BoundsChecker(tir.PyStmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.analyzer = tvm.arith.Analyzer()
            self.checked = 0

        def visit_attr_stmt_(self, op):
            if op.attr_key == "thread_extent":
                self.analyzer.bind(op.node.var, tvm.ir.Range(0, op.value))
            self.visit_stmt(op.body)

        def visit_for_(self, op):
            self.analyzer.bind(op.loop_var, tvm.ir.Range.from_min_extent(op.min, op.extent))
            self.visit_stmt(op.body)

        def visit_if_then_else_(self, op):
            with self.analyzer.constraint_scope(op.condition):
                self.visit_stmt(op.then_case)
            if op.else_case is not None:
                with self.analyzer.constraint_scope(tir.Not(op.condition)):
                    self.visit_stmt(op.else_case)

        def visit_buffer_store_(self, op):
            if op.buffer.name in ("A_stage", "B_stage", "C_shared"):
                for index, extent in zip(op.indices, op.buffer.shape, strict=True):
                    assert self.analyzer.can_prove(index >= 0)
                    assert self.analyzer.can_prove(index < extent), f"{op.buffer.name}: {index} exceeds {extent}"
                self.checked += 1

    checker = BoundsChecker()
    for func in mod.functions.values():
        checker.visit_stmt(func.body)
    assert checker.checked > 0


def _make_gemm(m, n, k):
    @tilelang.jit
    def gemm(
        A: T.Tensor((m, k), T.float16),
        B: T.Tensor((k, n), T.float16),
        C: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((m, k), T.float16)
            B_shared = T.alloc_shared((k, n), T.float16)
            C_local = T.alloc_fragment((m, n), T.float16)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C)

    return gemm


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("k", [4, 8, 16, 20, 24, 28])
def test_gemm_falls_back_for_partial_mma_atom(k):
    m = n = 64
    kernel = _make_gemm(m, n, k)

    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    c = torch.empty(m, n, device="cuda", dtype=torch.float16)
    kernel(a, b, c)

    torch.testing.assert_close(c, a @ b, rtol=2e-2, atol=2e-2)


@tilelang.testing.requires_cuda
def test_fma_gemm_with_padded_shared_layout():
    m = n = 16
    k = 17

    @T.prim_func
    def main(A: T.Tensor((m, k), T.float16), B: T.Tensor((k, n), T.float16), C: T.Tensor((m, n), T.float32)):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((m, k), T.float16)
            B_shared = T.alloc_shared((k, n), T.float16)
            C_local = T.alloc_fragment((m, n), T.float32)
            # Keep the layout selector in the test even when GEMM selects FMA.
            T.annotate_layout({A_shared: make_swizzled_layout(A_shared), B_shared: make_swizzled_layout(B_shared)})
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.copy(C_local, C)

    kernel = tilelang.compile(main, target="cuda", pass_configs={"tl.disable_wgmma": True})
    a = ((torch.arange(m * k, device="cuda") % 7 - 3) / 4).reshape(m, k).half()
    b = ((torch.arange(k * n, device="cuda") % 5 - 2) / 4).reshape(k, n).half()
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    kernel(a, b, c)
    expected = (a.cpu().double() @ b.cpu().double()).float()
    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "a_scope,b_scope",
    [("shared", "shared"), ("local.fragment", "shared"), ("shared", "local.fragment"), ("local.fragment", "local.fragment")],
)
@pytest.mark.parametrize("size", [8, 16])
@pytest.mark.parametrize("clear_accum", [False, True])
def test_fma_gemm_nonzero_thread_partition(a_scope, b_scope, size, clear_accum):
    @T.prim_func
    def main(A: T.Tensor((size, 4), T.float16), B: T.Tensor((4, size), T.float16), C: T.Tensor((size, size), T.float32)):
        with T.Kernel(1, threads=256):
            A_tile = T.alloc_buffer((size, 4), T.float16, scope=a_scope)
            B_tile = T.alloc_buffer((4, size), T.float16, scope=b_scope)
            C_shared = T.alloc_shared((size, size), T.float32)
            T.fill(C_shared, -123.0)
            tx = T.get_thread_binding()
            if tx >= 128:
                T.copy(A, A_tile)
                T.copy(B, B_tile)
                T.gemm(A_tile, B_tile, C_shared, clear_accum=clear_accum)
            T.copy(C_shared, C)

    kernel = tilelang.compile(main, target="cuda")
    _check_fma_shared_bounds(kernel.artifact.device_mod)
    a = torch.arange(size * 4, device="cuda", dtype=torch.float16).view(size, 4) / 8
    b = torch.arange(4 * size, device="cuda", dtype=torch.float16).view(4, size) / 8
    c = torch.empty((size, size), device="cuda", dtype=torch.float32)
    kernel(a, b, c)
    expected = a.float() @ b.float()
    if not clear_accum:
        expected -= 123
    torch.testing.assert_close(c, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
