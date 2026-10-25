from testing.python.transform._native_ir import make_function

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
            for j, v_2 in T.grid(N, vec_len // 4):
                for vec in T.vectorized(4):
                    A_shared[tid, j, v_2 * 4 + vec] = A[tid, j, v_2 * 4 + vec]

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body, map_free_vars=True)
    assert tvm.tirx.analysis.check_well_formed(transformed)


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
    transformed = tvm.tirx.transform.StmtSimplify()(transformed)
    reference = tvm.tirx.transform.StmtSimplify()(tvm.IRModule.from_expr(expected))
    tvm.ir.assert_structural_equal(transformed["before"].body, reference["expected"].body, map_free_vars=True)
    assert tvm.tirx.analysis.check_well_formed(transformed)


@tilelang.testing.requires_cuda
def test_vectorized_split_preserves_large_int64_extent():
    tir = tvm.tirx
    start, extent = 2, 1 << 32
    a = tir.decl_buffer((start + extent,), "int32", name="A")
    i = tir.Var("i", "int64")
    body = tir.BufferStore(a, a[i] + 1, [i])
    loop = tir.For(i, start, start + extent, 1, tir.ForKind.VECTORIZED, body)
    func = make_function([a.data], loop, buffer_map={a.data: a})
    with tvm.target.Target("cuda"):
        mod = tl.transform.LegalizeVectorizedLoop()(tvm.IRModule.from_expr(func))
    loops = []
    tir.stmt_functor.post_order_visit(mod["main"].body, lambda node: loops.append(node) if isinstance(node, tir.For) else None)
    assert len(loops) == 2
    assert all(loop.var.ty == "int64" for loop in loops)
    assert int(loops[0].end) * int(loops[1].end) == extent
    assert all(int(loop.begin) == 0 and int(loop.step) == 1 for loop in loops)
    assert tvm.tirx.analysis.check_well_formed(mod)
    tvm.ir.assert_structural_equal(mod, tvm.ir.load_json(tvm.ir.save_json(mod)))


if __name__ == "__main__":
    tilelang.testing.main()
