import torch

import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=None,
    execution_backend="tvm_ffi",
    pass_configs={"tl.disable_data_race_check": True},
)
def _fp4_eager_kernel(A):
    M, K = T.const("M K")
    A: T.Tensor((M, K), T.float4_e2m1fn)
    with T.Kernel(1, threads=32):
        T.evaluate(A[0, 0])


@tilelang.jit(
    out_idx=None,
    execution_backend="tvm_ffi",
    pass_configs={"tl.disable_data_race_check": True},
)
def _fp4_eager_strided_kernel(A):
    M, K, S = T.const("M K S")
    A: T.StridedTensor((M, K), (S, 1), T.float4_e2m1fn)
    with T.Kernel(1, threads=32):
        T.evaluate(A[0, 0])


@tilelang.jit(
    out_idx=None,
    execution_backend="tvm_ffi",
    pass_configs={"tl.disable_data_race_check": True},
)
def _fp4_lazy_kernel():
    @T.prim_func
    def main(A: T.Tensor((256, 512), T.float4_e2m1fn)):
        with T.Kernel(1, threads=32):
            T.evaluate(A[0, 0])

    return main


def _assert_runs_packed_fp4_storage(kernel, dtype):
    tensor = torch.empty((256, 256), dtype=dtype, device="cuda")
    kernel(tensor)
    torch.cuda.synchronize()


def _assert_accepts_packed_fp4_storage(kernel):
    assert torch.cuda.is_available()
    assert hasattr(torch, "float4_e2m1fn_x2")

    _assert_runs_packed_fp4_storage(kernel, torch.int8)
    _assert_runs_packed_fp4_storage(kernel, torch.uint8)
    _assert_runs_packed_fp4_storage(kernel, torch.float4_e2m1fn_x2)


def test_eager_jit_binds_logical_shape_for_packed_fp4_tensor():
    _assert_accepts_packed_fp4_storage(_fp4_eager_kernel)


def test_lazy_jit_accepts_packed_fp4_tensor():
    _assert_accepts_packed_fp4_storage(_fp4_lazy_kernel())


def test_eager_jit_binds_logical_stride_for_packed_fp4_tensor():
    packed = torch.empty((512, 256), dtype=torch.int8)[::2, :]

    prim_func = _fp4_eager_strided_kernel.get_tir(packed)
    buffer = next(iter(prim_func.buffer_map.values()))

    assert list(buffer.shape) == [256, 512]
    assert list(buffer.strides) == [1024, 1]
