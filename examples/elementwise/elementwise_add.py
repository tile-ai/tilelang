import tilelang
import tilelang.language as T


# copied from https://github.com/tile-ai/tilelang/blob/main/testing/python/kernel/test_tilelang_kernel_element_wise_add.py
def elementwise_add(
    M,
    N,
    block_M,
    block_N,
    in_dtype,
    out_dtype,
    threads,
):

    @T.prim_func
    def main(
            A: T.Buffer((M, N), in_dtype),
            B: T.Buffer((M, N), in_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for (local_y, local_x) in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                C[y, x] = A[y, x] + B[y, x]

    return main


def ref_program(x, y):
    return x + y


if __name__ == "__main__":
    program = elementwise_add(512, 1024, 128, 256, "float32", "float32", 128)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")
    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
