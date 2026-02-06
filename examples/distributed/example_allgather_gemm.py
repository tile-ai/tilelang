import torch
import pynvshmem
import os
import tilelang
import tilelang.language as T
from tilelang.profiler import TensorSupplyType
from tilelang.distributed import init_distributed


def allgather_gemm(PE_num, M, N, K, block_M, block_N, block_K, dtype="float16"):
    accum_dtype = "float"

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        A_ag: T.Buffer((M * PE_num, K), dtype),
        B: T.Buffer((K, N), dtype),
        signal: T.Buffer((PE_num,), "uint64"),
        C: T.Buffer((M * PE_num, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            mype = T.alloc_local([1], "int32")
            npes = T.alloc_local([1], "int32")
            peer = T.alloc_local([1], "int32")

            mype[0] = T.get_pe()
            npes[0] = T.get_pe_num()

            T.copy(A[by * block_M, bx * block_K], A_shared)
            T.copy(A_shared, A_ag[mype[0] * M, bx * block_K])
            for k in T.serial(PE_num - 1):
                peer[0] = (mype[0] + 1 + k) % npes[0]
                T.putmem_signal_nbi_block(
                    T.address_of(A_ag[mype[0] * M, 0]),
                    T.address_of(A[0, 0]),
                    block_M * block_K * 2,
                    T.address_of(signal[k]),
                    k + 1,
                    9,
                    peer[0],
                )
            for k in T.serial(PE_num - 1):
                T.signal_wait_until(T.address_of(signal[k]), 0, k + 1)

            for bk in T.serial(PE_num):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                    T.copy(A_ag[bk * M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[bk * M, bx * block_N])

    return main


tilelang.disable_cache()
M, N, K, block_M, block_N, block_K = 64, 64, 64, 64, 64, 64
dtype = torch.float16

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP = init_distributed(return_tp_group=True)
PE_num = WORLD_SIZE
func = allgather_gemm(PE_num, M, N, K, block_M, block_N, block_K)
kernel = tilelang.compile(func, out_idx=-1, pass_configs={"tl.disable_tma_lower": True, "tl.disable_warp_specialized": True})

# Get CUDA Source
if RANK == 0:
    print(kernel.get_kernel_source())

profiler = kernel.get_profiler(tensor_supply_type=TensorSupplyType.Randn)

A_tensor = torch.arange(M * PE_num * K, dtype=dtype).cuda() * 0.001
A_tensor = A_tensor.reshape(M * PE_num, K)
B_tensor = torch.arange(K * N, dtype=dtype).cuda() * 0.001
B_tensor = B_tensor.reshape(K, N)

print("A_tensor:", A_tensor)
print("B_tensor:", B_tensor)


def ref_program(A, B):
    return A @ B


C_ref = ref_program(A_tensor, B_tensor)
print("C_ref:", C_ref)

# profiler.init_distributed()
A_local = pynvshmem.nvshmem_create_tensor([M, K], dtype)
A_local[:].copy_(A_tensor[M * RANK : M * (RANK + 1), :])

A_ag_local = pynvshmem.nvshmem_create_tensor([M * PE_num, K], dtype)
A_ag_local.fill_(0)

B_local = pynvshmem.nvshmem_create_tensor([K, N], dtype)
B_local[:].copy_(B_tensor)

signal_local = pynvshmem.nvshmem_create_tensor([PE_num], torch.uint64)
signal_local.fill_(0)

out = kernel(A_local, A_ag_local, B_local, signal_local)
print("out:", out)

ref_cpu = C_ref.cpu()
for i in range(PE_num):
    if i == RANK:
        out_cpu = out.cpu()
        assert torch.allclose(out_cpu, ref_cpu, atol=1e-2, rtol=1e-2)
        print(f"rank {i} check passed.")
