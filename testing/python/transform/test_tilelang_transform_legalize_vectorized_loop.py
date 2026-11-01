import pytest
import torch

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def vectorize_access_legalize(M: int = 64, N: int = 64):
    dtype = T.float32
    vec_len = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                for v in T.vectorized(vec_len):
                    A_shared[tid, j, v] = A[tid, j, v]

    @T.prim_func
    def expected(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j, v_2 in T.grid(M, vec_len // 4):
                for vec in T.vectorized(4):
                    A_shared[tid, j, v_2 * 4 + vec] = A[tid, j, v_2 * 4 + vec]

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


@tilelang.testing.requires_cuda
def test_vectorized_nonzero_min_uses_aligned_vector_width():
    @T.prim_func
    def before(
        A: T.Tensor((8,), T.int32),
        B: T.Tensor((8,), T.int32),
    ):
        for value in T.vectorized(2, 6):
            B[value] = A[value] + 1

    @T.prim_func
    def expected(
        A: T.Tensor((8,), T.int32),
        B: T.Tensor((8,), T.int32),
    ):
        for value_outer in T.serial(2):
            for vec in T.vectorized(2):
                B[2 + (value_outer * 2 + vec)] = A[2 + (value_outer * 2 + vec)] + 1

    mod = tvm.IRModule.from_expr(before)
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(transformed["before"].body, expected.body)


@tilelang.testing.requires_cuda
def test_vectorized_split_preserves_large_int64_extent():
    tir = tvm.tirx
    start, extent = 2, 1 << 32
    a = tir.decl_buffer((start + extent,), "int32", name="A")
    i = tir.Var("i", "int64")
    body = tir.BufferStore(a, a[i] + 1, [i])
    loop = tir.For(i, start, extent, tir.ForKind.VECTORIZED, body)
    func = tir.PrimFunc([a.data], loop, buffer_map={a.data: a})
    with tvm.target.Target("cuda"):
        mod = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule.from_expr(func))
    loops = []
    tir.stmt_functor.post_order_visit(mod["main"].body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
    assert len(loops) == 2
    assert all(loop.loop_var.dtype == "int64" for loop in loops)
    assert int(loops[0].extent) * int(loops[1].extent) == extent


def _guarded_cast_store(memory_guard):
    @T.prim_func
    def main(
        A: T.Tensor((512,), T.float32),
        Flags: T.Tensor((1,), T.int32),
        Out: T.Tensor((512,), T.bfloat16),
        n: T.int32,
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            local = T.alloc_local((16,), T.float32)
            for i in T.serial(16):
                local[i] = A[tx * 16 + i]
            for i in T.vectorized(16):
                if (Flags[0] if memory_guard else n) > 0:
                    Out[T.cast(tx * 16 + i, T.int64)] = local[i] / 2

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("memory_guard", [True, False])
def test_guarded_cast_store_vector_width(memory_guard):
    mod = tvm.IRModule.from_expr(_guarded_cast_store(memory_guard))
    with tvm.target.Target({"kind": "cuda", "arch": "sm_100"}):
        mod = tl.transform.DecoupleTypeCast()(mod)
        mod = tl.transform.LegalizeVectorizedLoop()(mod)

    store_widths = []

    def collect_loop(node):
        if isinstance(node, tvm.tirx.For) and node.kind == tvm.tirx.ForKind.VECTORIZED:

            def collect_store(stmt):
                if isinstance(stmt, tvm.tirx.BufferStore) and stmt.buffer.name == "Out":
                    store_widths.append(int(node.extent))

            tvm.tirx.stmt_functor.post_order_visit(node.body, collect_store)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, collect_loop)
    # A memory guard prevents cast staging, so FP32 limits the width to eight.
    # A scalar guard permits staging and the BF16 copy should retain 16 lanes.
    # The int64 address cast must not restrict either data vector width.
    assert store_widths == [8 if memory_guard else 16]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10)
def test_memory_guarded_cast_store_cuda():
    kernel = tl.compile(_guarded_cast_store(True), target={"kind": "cuda", "arch": "sm_100"})
    a = torch.arange(512, device="cuda", dtype=torch.float32)
    flags = torch.zeros((1,), device="cuda", dtype=torch.int32)
    out = torch.full((512,), -1, device="cuda", dtype=torch.bfloat16)
    expected = out.clone()

    kernel(a, flags, out, 1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    flags.fill_(1)
    kernel(a, flags, out, 1)
    torch.testing.assert_close(out, (a / 2).to(torch.bfloat16), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
