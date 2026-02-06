import tilelang
import tilelang.language as T
from tilelang.profiler import TensorSupplyType
from tilelang.distributed import init_distributed


def simple_shift(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def main(
        A: T.Buffer((M, N), dtype),
        B: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            peer = T.alloc_local([1], "int32")
            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()
            peer[0] = (mype[0] + 1) % npes[0]

            T.putmem_nbi_block(T.address_of(B[0, 0]), T.address_of(A[0, 0]), block_M * block_N * 2, peer[0])

    return main


WORLD_SIZE, RANK, LOCAL_RANK = init_distributed()

func = simple_shift(128, 128, 128, 128)
# Auto-selects cython backend when TILELANG_USE_DISTRIBUTED=1 is set
kernel = tilelang.compile(func, out_idx=-1)

# Get CUDA Source
if RANK == 0:
    print(kernel.get_kernel_source())

profiler = kernel.get_profiler(tensor_supply_type=TensorSupplyType.Randn)
out = profiler.run_once()

print(out)
