import ctypes

lib = ctypes.CDLL("./libnvshmem_example.so")

lib.run_simple_shift.restype = ctypes.c_int

import torch
import pynvshmem


def init_distributed():
    import os
    import datetime

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
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


def run_simple_shift():
    result = lib.run_simple_shift()
    print(f"message: {result}")
    return result


if __name__ == "__main__":
    init_distributed()
    run_simple_shift()
