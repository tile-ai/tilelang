# Test for creating tensors

import torch
import torch.distributed
import pynvshmem
import os
import datetime


# TODO: Add more checks, e.g. tensor manipulations.
def test_nvshmem_create_tensor(N, dtype):
    t = pynvshmem.nvshmem_create_tensor((N,), dtype)
    assert t.numel() == N and t.dtype == dtype and t.device == torch.device("cuda", RANK) and t.is_contiguous()
    if RANK == 0:
        print("nvshmem_create_tensor test passed!✅")


def test_nvshmem_create_tensor_list_intra_node(N, dtype):
    tensor_list = pynvshmem.nvshmem_create_tensor_list_intra_node((N,), dtype)
    t = tensor_list[RANK]
    assert t.numel() == N and t.dtype == dtype and t.device == torch.device("cuda", RANK) and t.is_contiguous()
    if RANK == 0:
        print("nvshmem_create_tensor_list_intra_node test passed!✅")


if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    assert 2 <= WORLD_SIZE <= 8, "This test is for intra-node multi-GPU"
    RANK = int(os.environ.get("RANK", 0))

    torch.cuda.set_device(RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    N = 1024
    dtype = torch.float16

    test_nvshmem_create_tensor(N, dtype)
    test_nvshmem_create_tensor_list_intra_node(N, dtype)
