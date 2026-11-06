"""NPU cache maintenance shared by event and msprof timing."""

import ctypes
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _get_device_l2_cache_flush():
    # Resolve lazily so importing the profiler does not require CANN.
    try:
        acl = ctypes.CDLL("libascendcl.so")
        flush = acl.aclrtDeviceL2CacheFlush
    except (OSError, AttributeError) as error:
        raise RuntimeError("NPU benchmarking requires a CANN runtime providing aclrtDeviceL2CacheFlush") from error
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int
    return flush


def flush_l2_cache() -> None:
    """Synchronize and flush the current NPU's L2 before a timed invocation.

    The runtime call is synchronous and device-wide. Its cost, including any
    deferred output writeback from the previous invocation, is not kernel time.
    """
    flush = _get_device_l2_cache_flush()
    torch.npu.synchronize()
    status = flush(None)
    if status != 0:
        raise RuntimeError(
            f"aclrtDeviceL2CacheFlush failed with error code {status}; "
            "NPU benchmarking requires device and runtime support for L2 cache flushing"
        )
