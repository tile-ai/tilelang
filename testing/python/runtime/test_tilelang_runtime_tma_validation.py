import ctypes
import re

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


CUDA_SM90_TARGET = {"kind": "cuda", "arch": "sm_90"}


def _compile_tvm_ffi(func, *, target=CUDA_SM90_TARGET, target_host="c", pass_configs=None):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target=target,
            target_host=target_host,
            execution_backend="tvm_ffi",
            pass_configs=pass_configs or {},
        )
    finally:
        tilelang.enable_cache()


def _get_tma_create_tiled():
    func = tvm.ffi.get_global_func("__tvm_tensormap_create_tiled", allow_missing=True)
    if func is None:
        pytest.skip("__tvm_tensormap_create_tiled is unavailable in this build.")
    return func


@tilelang.testing.requires_cuda
def test_tma_runtime_validation_surfaces_invalid_argument_constraints():
    create_tiled = _get_tma_create_tiled()

    with pytest.raises(RuntimeError) as exc_info:
        create_tiled(
            ctypes.c_void_p(0x10),
            7,
            2,
            ctypes.c_void_p(0x20),
            1733,
            4,
            4,
            6932,
            64,
            1,
            1,
            1,
            0,
            0,
            2,
            0,
        )

    message = str(exc_info.value)
    assert "Invalid TMA descriptor arguments for __tvm_tensormap_create_tiled" in message
    assert "tensorMap address must be 64-byte aligned" in message
    assert "effective cuda globalStrides[0]" in message
    assert "globalStridesRaw [4, 6932]" in message
    assert "cudaGlobalStrides [6932]" in message
    assert "format         7 (CU_TENSOR_MAP_DATA_TYPE_FLOAT32)" in message


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_host_codegen_aligns_native_tensor_map_storage():
    import torch
    from tvm_tirx import runtime

    m, k = 16, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy_2d_desc(
        x: T.Tensor((m, k), T.float16),
        y: T.Tensor((m, k), T.float16),
    ):
        with T.Kernel(T.ceildiv(m, block_m), T.ceildiv(k, block_k), threads=threads) as (pid_m, pid_k):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            mbar = T.alloc_barrier(threads)
            T.tma_copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
                barrier=mbar,
            )
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(
                x_shared,
                y[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
            )

    kernel = _compile_tvm_ffi(
        tma_copy_2d_desc,
        target={"kind": "cuda", "arch": f"sm_{runtime.cuda(0).compute_version.replace('.', '')}"},
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )

    source = kernel.get_host_source()
    assert "runtime.cuTensorMapEncodeTiled" in source
    assert re.search(r"uint64_t data\[16\];.*aligned\(64\).*TVMTensorMap", source)
    assert re.search(r"TVMTensorMap stack_\d+\[1\];", source)
    values = torch.arange(m * k, dtype=torch.float32, device="cuda").reshape(m, k).to(torch.float16)
    output = torch.empty_like(values)
    kernel(values, output)
    torch.testing.assert_close(output, values, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
