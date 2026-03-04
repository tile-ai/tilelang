import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def simple_kernel(M=128, N=128):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float32"), B: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, 32), T.ceildiv(M, 32), threads=128) as (bx, by):
            A_shared = T.alloc_shared((32, 32), "float32")
            T.copy(A[by * 32 : (by + 1) * 32, bx * 32 : (bx + 1) * 32], A_shared)
            T.copy(A_shared, B[by * 32 : (by + 1) * 32, bx * 32 : (bx + 1) * 32])

    return kernel


if __name__ == "__main__":
    kernel = simple_kernel()
    print("Kernel compiled!")
