import torch
import torch.distributed as dist
import pynvshmem
import tilelang
import tilelang.language as T
import os
from tilelang.distributed.utils import init_distributed
from tilelang.env import env
from packaging import version
import importlib.metadata

cuda_python_version = importlib.metadata.version("cuda-python")
if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import runtime as cudart
else:
    from cuda import cudart
# NODES=2 NODE_RANK=0 ARNOLD_WORKER_0_HOST=ip0 bash tilelang/distributed/launch.sh ./examples/distributed/example_overlapping_allgather.py
# NODES=2 NODE_RANK=1 ARNOLD_WORKER_0_HOST=ip0 bash tilelang/distributed/launch.sh ./examples/distributed/example_overlapping_allgather.py


def internode_gather(M, local_world_size, block_M, threads):
    @T.prim_func
    def main(
        dst: T.Tensor((M), "float32"),
        src: T.Tensor((M), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            rank[0] = (T.get_pe() + local_world_size) % (2 * local_world_size)  # 2 nodes
            T.putmem_nbi_block(T.address_of(dst[bx * block_M]), T.address_of(src[bx * block_M]), block_M * 4, rank[0])

    return main


def intranode_gather(M, world_size, block_M, threads):
    @T.prim_func
    def main(
        dst: T.Tensor((M * world_size), "float32"),
        src: T.Tensor((M * 2), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
            rank = T.alloc_local([1], "uint64")
            num_rank = T.alloc_local([1], "uint64")
            rank[0] = T.get_rank()
            num_rank[0] = T.get_num_ranks()
            tid = T.get_thread_binding()
            if tid == 0:
                T.print(T.cast(rank[0], "int32"), msg="signal")
                T.print(T.cast(num_rank[0], "int32"), msg="signal")
            for k in T.serial(world_size // 2):  # 2 node
                T.put_block(
                    src=T.address_of(src[bx * block_M]),
                    dst=T.address_of(dst[bx * block_M + rank[0] * M]),
                    size=block_M,
                    dst_pe=k,
                )
                T.put_block(
                    src=T.address_of(src[bx * block_M + M]),
                    dst=T.address_of(dst[bx * block_M + M * num_rank[0] + rank[0] * M]),
                    size=block_M,
                    dst_pe=k,
                )

    return main


if __name__ == "__main__":
    tilelang.disable_cache()

    M = 2
    K = 12288
    # for 2 node(16 GPUs), world_size=16,rank is 0-15,local rank is 0-7
    WORLD_SIZE, RANK, LOCAL_RANK, TP_GROUP, LC_GROUP = init_distributed(return_tp_group=True, return_lc_group=True)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    allocator = tilelang.get_allocator(
        size=2**25, device="cuda", is_distributed=True, local_rank=LOCAL_RANK, num_local_ranks=local_world_size, group=LC_GROUP
    )
    print(local_world_size, LOCAL_RANK)

    # Each rank sends the local_tensor to ranks of other nodes with the same local_rank
    # Assuming there are 2 nodes, each with 4 workers
    # 0-th local tensor ([0] -> [4]), 4-th local tensor ([4] -> [0])
    # 1-th local tensor ([1] -> [5]), 5-th local tensor ([5] -> [1])
    # 2-th local tensor ([2] -> [6]), 6-th local tensor ([6] -> [2])
    # 3-th local tensor ([3] -> [7]), 7-th local tensor ([7] -> [3])
    interkernel = tilelang.compile(internode_gather(M, local_world_size, M, 128))
    if LOCAL_RANK == 0:
        print(interkernel.get_kernel_source())
    src = pynvshmem.nvshmem_create_tensor([M], torch.float32)
    dst = pynvshmem.nvshmem_create_tensor([M], torch.float32)
    input_data = torch.ones([M], dtype=torch.float32, device="cuda") * RANK
    src.copy_(input_data)

    pynvshmem.nvshmem_barrier_all()
    dist.barrier(TP_GROUP)
    interkernel(dst, src)
    pynvshmem.nvshmem_barrier_all()

    # Each rank sends the local_tensor and the received internode tensors to intranode ranks.
    # 0-th and 4-th local tensors ([0]->[1,2,3])
    # 1-th and 5-th local tensors ([1]->[0,2,3])
    # 2-th and 6-th local tensors ([2]->[0,1,3])
    # 3-th and 7-th local tensors ([3]->[0,1,2])
    # 0-th and 4-th local tensors ([4]->[5,6,7])
    # 1-th and 5-th local tensors ([5]->[4,6,7])
    # 2-th and 6-th local tensors ([6]->[4,5,7])
    # 3-th and 7-th local tensors ([7]->[4,5,6])
    src_intra = tilelang.tensor((M * 2), torch.float32, allocator=allocator).normal_()
    dst_intra = tilelang.tensor((M * WORLD_SIZE), torch.float32, allocator=allocator)
    if RANK < WORLD_SIZE / 2:
        cudart.cudaMemcpy(src_intra.data_ptr(), src.data_ptr(), M * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(src_intra.data_ptr() + M * 4, dst.data_ptr(), M * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    else:
        cudart.cudaMemcpy(src_intra.data_ptr(), dst.data_ptr(), M * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        cudart.cudaMemcpy(src_intra.data_ptr() + M * 4, src.data_ptr(), M * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)

    env.USE_NVSHMEM = False
    intrakernel = tilelang.compile(intranode_gather(M, WORLD_SIZE, M, 128), pass_configs={tilelang.PassConfigKey.TL_DISABLE_RDC: True})
    intrakernel.initialize(allocator=allocator)
    if LOCAL_RANK == 0:
        print(intrakernel.get_kernel_source())
    torch.cuda.synchronize()
    torch.distributed.barrier(LC_GROUP)
    intrakernel(dst_intra, src_intra)
    torch.cuda.synchronize()
    torch.distributed.barrier(LC_GROUP)

    print(dst_intra)
