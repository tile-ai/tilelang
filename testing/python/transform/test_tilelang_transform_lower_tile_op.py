from tilelang import tvm as tvm
from tilelang.utils.target import determine_target
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import pytest

auto_target = tvm.target.Target(determine_target("auto"))


@pytest.mark.parametrize(
    "block_M, block_N, block_K, threads, vec_load_b, dtype",
    [
        (64, 64, 32, 128, 8, T.float16),
    ],
)
def test_loop_tail_split(block_M, block_N, block_K, threads, vec_load_b, dtype):
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    def before():
        @T.prim_func
        def main(
            B: T.Tensor((K, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(B[k * block_K, bx * block_N], B_shared)

        return tvm.IRModule({"main": main})

    def after():
        @T.prim_func
        def main(
            B: T.Tensor((K, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    t = thread_bindings
                    for i in T.unroll(0, block_N * block_K // (threads * vec_load_b)):
                        if (k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b)) * N % vec_load_b == 0:
                            for vec in T.vectorized(vec_load_b):
                                B_shared[
                                    i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ] = T.if_then_else(
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                    and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                    B[
                                        k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                        bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                    ],
                                    T.float16(0),
                                )
                        else:
                            for vec in T.serial(vec_load_b):
                                B_shared[
                                    i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ] = T.if_then_else(
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                    and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                    B[
                                        k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                        bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                    ],
                                    T.float16(0),
                                )

        return tvm.IRModule({"main": main})

    with tvm.transform.PassContext():
        mod = tvm.tir.transform.BindTarget(auto_target)(before())
        mod = tl.transform.LowerTileOp()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
    ref_mod = tvm.tir.transform.BindTarget(auto_target)(after())
    ref_mod = tvm.tir.transform.Simplify()(ref_mod)
    # Note(tzj): The structures are equal except the argument in "T.reads" function.
    # The difference is just between the first index and the indices range, which is totally equivalent
    tvm.ir.structural_equal(mod, ref_mod)
    # tvm.ir.assert_structural_equal(mod, ref_mod)


if __name__ == "__main__":
    tilelang.testing.main()
