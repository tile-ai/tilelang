import pytest
import torch

import tilelang.testing
from tilelang.profiler.msprof import _get_cache_read_kernel, bench_with_msprof
from tilelang.profiler.torch_bench import _allocate_cache


@tilelang.testing.requires_ascend
def test_msprof_keeps_user_launches_of_the_cache_kernel():
    device = torch.device("npu", torch.npu.current_device())
    cache = _allocate_cache(device, 4)
    if not cache.numel():
        pytest.skip("The native L2 flush has no scratch kernel to exclude")
    work = torch.empty_like(cache)
    read = _get_cache_read_kernel(work.numel())
    other_stream = torch.npu.Stream(device=device)

    # Both user launches have exactly the same name as the profiler's cache read.
    def workload():
        read(work)
        with torch.npu.stream(other_stream):
            read(work)

    with pytest.raises(RuntimeError, match="No profiled kernels were recorded"):
        bench_with_msprof(lambda: None, cache, n_repeat=3)
    workload()
    torch.npu.synchronize()
    assert bench_with_msprof(workload, cache, n_repeat=3, detailed=True).dur_ns > 0
