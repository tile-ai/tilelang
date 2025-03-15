import tilelang
import tilelang.language as T


def elewise_add(N, NUM_ELE_PER_THREAD=8, threads=256, dtype="bfloat16"):

    @T.prim_func
    def main(A: T.Buffer((N), dtype), B: T.Buffer((N), dtype), C: T.Buffer((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * NUM_ELE_PER_THREAD), threads=threads) as (b_x):
            A_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            B_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            C_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)

            s_start = b_x * threads * NUM_ELE_PER_THREAD
            s_end = (b_x + 1) * threads * NUM_ELE_PER_THREAD

            # LDG. 128
            T.copy(
                A[s_start:s_end],
                A_register,
            )
            T.copy(
                B[s_start:s_end],
                B_register,
            )

            # vector add.
            for tid, i in T.Parallel(threads, NUM_ELE_PER_THREAD):
                C_register[tid * NUM_ELE_PER_THREAD + i] = (
                    A_register[tid * NUM_ELE_PER_THREAD + i] +
                    B_register[tid * NUM_ELE_PER_THREAD + i])

            # STG. 128
            T.copy(
                C_register,
                C[s_start:s_end],
            )

    return main


def ref_program(x, y):
    return x + y


if __name__ == "__main__":
    N = 8192**2
    program = elewise_add(N)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    print("All checks pass.")
    latency = profiler.do_bench(ref_program, warmup=500)
    print("Ref: {:.2f} ms".format(latency))
    latency = profiler.do_bench(profiler.mod, warmup=500)
    print("Tile-lang: {:.2f} ms".format(latency))
