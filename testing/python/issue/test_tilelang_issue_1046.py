import tilelang
import tilelang.language as T

tilelang.disable_cache()

FP8 = "float8_e4m3"
BF16 = "bfloat16"


@tilelang.jit
def test_kernel(N, in_dtype=BF16, out_dtype=FP8):
    M = T.symbolic("M")
    blk_m = 128
    group_size = 128

    @T.prim_func
    def test_kernel_(X: T.Tensor[(M, N), in_dtype], Y: T.Tensor[(M, N), out_dtype]):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (pid_m, pid_n):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
            T.copy(x_shared, Y[pid_m * blk_m, pid_n * group_size])

    return test_kernel_


kernel = test_kernel(128)

print(kernel.get_kernel_source())
