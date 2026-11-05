"""The default worker query reaches a real persistent kernel without old TVM."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.carver.arch import driver


@pytest.mark.parametrize("stateful", [False, True])
@tilelang.testing.requires_cuda
def test_default_scheduler_worker_count_executes(stateful):
    workers = driver.get_num_sms()
    rows, columns = 2 * workers + 1, 3

    @T.prim_func
    def kernel(output: T.Tensor((rows, columns), "int32")):
        with T.Kernel(workers, threads=32) as worker:
            scheduler = T.PersistentTileScheduler(rows, columns, stateful=stateful, swizzle_size=2)
            if stateful:
                scheduler.init(worker)
                while scheduler.valid():
                    x, y = scheduler.m_idx, scheduler.n_idx
                    if T.get_thread_binding() == 0:
                        output[x, y] = x * columns + y
                    scheduler.next_tile()
            else:
                for wave in T.serial(T.ceildiv(rows * columns, workers)):
                    x, y = scheduler.coord(wave * workers + worker)
                    if x < rows and y < columns and T.get_thread_binding() == 0:
                        output[x, y] = x * columns + y

    compiled = tilelang.compile(kernel, out_idx=[0], target="cuda", execution_backend="tvm_ffi")
    output = compiled()
    torch.testing.assert_close(
        output, torch.arange(rows * columns, device="cuda", dtype=torch.int32).reshape(rows, columns), atol=0, rtol=0
    )
