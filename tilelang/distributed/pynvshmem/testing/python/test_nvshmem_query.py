# Test for basic nvshmem queries

import torch
import torch.distributed
import pynvshmem
import os
import datetime

if __name__ == "__main__":
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    assert WORLD_SIZE > 2
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(LOCAL_RANK)
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

    assert pynvshmem.nvshmem_my_pe() == RANK
    assert pynvshmem.nvshmem_n_pes() == WORLD_SIZE
    if RANK == 0:
        print("Test for basic queries passed!✅")
