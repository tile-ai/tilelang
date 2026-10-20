"""Logical reductions preserve layouts, ownership and expression evaluation."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout, make_swizzled_layout


def compile_kernel(func, serial=False):
    return tilelang.compile(
        func,
        out_idx=-1,
        target="cuda",
        pass_configs={"tl.disable_parallel_logical_reduction": serial},
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["any_of", "all_of"])
@pytest.mark.parametrize("threads", [32, 128])
@pytest.mark.parametrize("serial", [False, True])
def test_swizzled_rows(kind, threads, serial):
    op = getattr(T, kind)

    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), Out: T.Tensor((16,), "bool")):
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(A, shared)
            for r in T.serial(16):
                Out[r] = op(shared[r, :])

    kernel = compile_kernel(main, serial)
    a = torch.zeros((16, 64), dtype=torch.float16, device="cuda")
    a[8] = 1
    for data in (a, 1 - a, torch.ones_like(a)):
        expected = (torch.any if kind == "any_of" else torch.all)(data != 0, dim=1)
        torch.testing.assert_close(kernel(data), expected)
    assert ("tl::AllReduce<" in kernel.get_kernel_source()) == (not serial)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("mode", ["thread_row", "parallel_row", "single_thread", "local"])
def test_thread_local_execution(mode):
    @T.prim_func
    def main(A: T.Tensor((32, 64), "float16"), Out: T.Tensor((32,), "bool")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32, 64), "float16")
            local = T.alloc_local((4,), "int32")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(A, shared)
            tid = T.get_thread_binding()
            if mode == "parallel_row":
                for r in T.Parallel(32):
                    Out[r] = T.any_of(shared[r, :])
            elif mode == "single_thread":
                if tid == 0:
                    for r in T.serial(32):
                        Out[r] = T.any_of(shared[r, :])
            elif mode == "local":
                for i in T.serial(4):
                    local[i] = T.if_then_else(tid % 2 == 0, 1 << i, 0)
                Out[tid] = T.all_of(local)
            else:
                Out[tid] = T.any_of(shared[tid, :])

    a = torch.zeros((32, 64), dtype=torch.float16, device="cuda")
    a[::2, 31] = 1
    kernel = compile_kernel(main)
    torch.testing.assert_close(kernel(a), torch.any(a != 0, dim=1))
    assert "tl::AllReduce<" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["any_of", "all_of"])
@pytest.mark.parametrize("length", [0, 17, 61])
def test_padded_layout_and_tail(kind, length):
    op = getattr(T, kind)

    @T.prim_func
    def main(A: T.Tensor((4, 61), "int32"), Out: T.Tensor((4,), "bool")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((4, 61), "int32")
            T.annotate_layout({shared: Layout((4, 61), lambda r, c: (r, 2 * c))})
            T.copy(A, shared)
            for r in T.serial(4):
                Out[r] = op(shared[r, 0:length])

    a = torch.ones((4, 61), dtype=torch.int32, device="cuda")
    a[0] = 0
    a[1, -1] = 0
    a[2, 16] = 0
    expected = (torch.any if kind == "any_of" else torch.all)(a[:, :length] != 0, dim=1)
    torch.testing.assert_close(compile_kernel(main)(a), expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("expression", ["and", "or", "select"])
def test_lazy_expression(expression):
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), flag: T.int32, Out: T.Tensor((1,), "bool")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(A, shared)
            if expression == "and":
                Out[0] = flag != 0 and T.any_of(shared[8, :])
            elif expression == "or":
                Out[0] = flag != 0 or T.all_of(shared[8, :])
            else:
                Out[0] = T.if_then_else(flag != 0, T.any_of(shared[8, :]), T.all_of(shared[7, :]))

    a = torch.zeros((16, 64), dtype=torch.float16, device="cuda")
    a[8, 31] = 1
    kernel = compile_kernel(main)
    for flag in (0, 1):
        expected = bool(flag)
        torch.testing.assert_close(kernel(a, flag), torch.tensor([expected], device="cuda"))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("use_continue", [False, True])
def test_while_recomputes_reduction(use_continue):
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(A, shared)
            iterations = T.alloc_var("int32", init=0)
            while T.any_of(shared[7, :]):
                iterations += 1
                T.clear(shared)
                if use_continue:
                    continue
            Out[0] = iterations

    a = torch.ones((16, 64), dtype=torch.float16, device="cuda")
    kernel = compile_kernel(main)
    torch.testing.assert_close(kernel(a), torch.tensor([1], dtype=torch.int32, device="cuda"))
    assert "tl::AllReduce<" in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_divergent_loop_exit_keeps_local_scans():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), Out: T.Tensor((32,), "bool")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(A, shared)
            tid = T.get_thread_binding()
            Out[tid] = False
            for _step in T.serial(2):
                if tid == 0:
                    break
                Out[tid] = T.any_of(shared[7, :])

    a = torch.ones((16, 64), dtype=torch.float16, device="cuda")
    expected = torch.ones((32,), dtype=torch.bool, device="cuda")
    expected[0] = False
    kernel = compile_kernel(main)
    torch.testing.assert_close(kernel(a), expected)
    assert "tl::AllReduce<" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_dynamic_slice_and_float_truth():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float32"), n: T.int32, Out: T.Tensor((2,), "bool")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 64), "float32")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            T.copy(A, shared)
            Out[0] = T.any_of(shared[7, :n])
            Out[1] = T.all_of(shared[7, :n])

    a = torch.ones((16, 64), dtype=torch.float32, device="cuda")
    a[7, 0] = float("nan")
    a[7, 32] = -0.0
    kernel = compile_kernel(main)
    for n in (0, 1, 33, 64):
        expected = torch.stack((torch.any(a[7, :n] != 0), torch.all(a[7, :n] != 0)))
        torch.testing.assert_close(kernel(a, n), expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["any_of", "all_of"])
@pytest.mark.parametrize("warp_specialized", [False, True])
def test_pipelined_shared_query(kind, warp_specialized):
    op = getattr(T, kind)

    @T.prim_func
    def main(A: T.Tensor((4, 16, 64), "float16"), Out: T.Tensor((4, 16), "bool")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_swizzled_layout(shared)})
            for k in T.Pipelined(4, num_stages=2):
                T.copy(A[k, :, :], shared)
                for row in T.serial(16):
                    Out[k, row] = op(shared[row, :])

    a = torch.zeros((4, 16, 64), dtype=torch.float16, device="cuda")
    a[1, ::2, :] = 1
    a[2, :, 63] = 1
    a[3] = 1
    expected = (torch.any if kind == "any_of" else torch.all)(a != 0, dim=2)
    kernel = tilelang.compile(
        main,
        out_idx=-1,
        target="cuda",
        pass_configs={"tl.disable_warp_specialized": not warp_specialized},
    )
    torch.testing.assert_close(kernel(a), expected)
    source = kernel.get_kernel_source()
    if "warpgroup_reg_alloc" in source:
        # The consumer warp group cannot enter a full-CTA collective.
        assert "tl::AllReduce<" not in source
    else:
        assert "tl::AllReduce<" in source


@tilelang.testing.requires_cuda
def test_fragment_and_value_snapshot():
    @T.prim_func
    def main(A: T.Tensor((256,), "int32"), Out: T.Tensor((3,), "bool")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((256,), "int32")
            T.copy(A, fragment)
            before = T.any_of(fragment)
            Out[0] = T.all_of(fragment)
            T.clear(fragment)
            Out[1] = before
            Out[2] = T.any_of(fragment)

    kernel = compile_kernel(main)
    for values in ([0] * 256, [1] * 256, [0] * 255 + [1]):
        a = torch.tensor(values, dtype=torch.int32, device="cuda")
        expected = torch.stack((torch.all(a != 0), torch.any(a != 0), torch.tensor(False, device="cuda")))
        torch.testing.assert_close(kernel(a), expected)
    assert "tl::AllReduce<" in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_fragment_query_requires_collective_context():
    @T.prim_func
    def main(A: T.Tensor((64,), "int32"), Out: T.Tensor((1,), "bool")):
        with T.Kernel(1, threads=32):
            fragment = T.alloc_fragment((64,), "int32")
            T.copy(A, fragment)
            if T.get_thread_binding() == 0:
                Out[0] = T.any_of(fragment)

    with pytest.raises(Exception, match="distributed fragments require a uniform collective context"):
        compile_kernel(main)


if __name__ == "__main__":
    tilelang.testing.main()
