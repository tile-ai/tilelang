import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit
def _nested_guard_cast_store(
    A: T.Tensor((16,), T.float8_e4m3fn),
    B: T.Tensor((16,), T.float8_e4m3fn),
):
    with T.Kernel(1, threads=32):
        a_local = T.alloc_local((16,), T.float32)
        T.copy(A, a_local)
        for i in T.vectorized(16):
            if i < 12:  # noqa: SIM102
                if i < 8:
                    B[i] = T.cast(a_local[i], T.float8_e4m3fn)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
def test_nested_guard_cast_store_preserves_unwritten_lanes():
    a = (torch.arange(16, device="cuda", dtype=torch.float32) * 0.5).to(torch.float8_e4m3fn)
    b = torch.full((16,), 7.0, device="cuda").to(torch.float8_e4m3fn)

    _nested_guard_cast_store(a, b)

    expected = torch.full((16,), 7.0, device="cuda", dtype=torch.float32)
    expected[:8] = a[:8].to(torch.float32)
    torch.testing.assert_close(b.to(torch.float32), expected, rtol=0, atol=0)


def _check_guard_reads_output(target, device):
    @T.prim_func
    def main(A: T.Tensor((16,), T.float16), B: T.Tensor((16,), T.float16)):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((16,), T.float32)
            for j in T.serial(16):
                local[j] = A[j]
            for i in T.vectorized(16):
                if i < 12:  # noqa: SIM102
                    if B[i] > T.float16(0):
                        B[i] = local[i] + 1

    kernel = tilelang.compile(main, target=target, execution_backend="cython")
    a = torch.arange(16, device=device, dtype=torch.float16)
    b = torch.where(torch.arange(16, device=device) % 2 == 0, -1.0, 7.0).to(torch.float16)
    expected = b.clone()
    expected[1:12:2] = a[1:12:2] + 1
    kernel(a, b)
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def test_guard_reads_output_cpu():
    _check_guard_reads_output("c", "cpu")


@tilelang.testing.requires_cuda
def test_guard_reads_output_cuda():
    _check_guard_reads_output("cuda", "cuda")


def _check_nonzero_min_nested_guards_staging(target, device):
    @T.prim_func
    def main(A: T.Tensor((8,), T.float16), B: T.Tensor((8,), T.float16)):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((8,), T.float32)
            for i in T.vectorized(2, 6):
                if i != 4:  # noqa: SIM102
                    if i < 5:
                        local[i] = A[i]
            for i in T.vectorized(2, 6):
                if i != 4:  # noqa: SIM102
                    if i < 5:
                        B[i] = local[i] + 1

    # Preserve the nested guards until decoupling; earlier simplification could
    # otherwise hide a missing guard in either generated copy direction.
    mod = tilelang.transform.DecoupleTypeCast()(tvm.IRModule.from_expr(main))
    analyzer = tvm.arith.Analyzer()

    def bind_loop(node):
        if isinstance(node, tvm.tirx.For):
            analyzer.bind(node.loop_var, tvm.ir.Range.from_min_extent(node.min, node.extent))

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, bind_loop)
    buffers = {}

    def check_staging(node):
        if isinstance(node, (tvm.tirx.BufferLoad, tvm.tirx.BufferStore)) and "_local_cast" in node.buffer.name:
            buffers[node.buffer.name.split("_local_cast")[0]] = [int(dim) for dim in node.buffer.shape]
            for index, extent in zip(node.indices, node.buffer.shape, strict=True):
                assert analyzer.can_prove(index >= 0)
                assert analyzer.can_prove(index < extent)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, check_staging)
    assert buffers == {"A": [4], "B": [4]}
    kernel = tilelang.compile(mod["main"], target=target, execution_backend="cython")
    a = torch.arange(8, device=device, dtype=torch.float16)
    b = torch.full_like(a, -1)
    expected = b.clone()
    expected[2:4] = a[2:4] + 1
    kernel(a, b)
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def test_nonzero_min_nested_guards_staging_cpu():
    _check_nonzero_min_nested_guards_staging("c", "cpu")


@tilelang.testing.requires_cuda
def test_nonzero_min_nested_guards_staging_cuda():
    _check_nonzero_min_nested_guards_staging("cuda", "cuda")


if __name__ == "__main__":
    tilelang.testing.main()
