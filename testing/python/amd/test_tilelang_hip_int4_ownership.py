import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import language as T


@pytest.fixture(autouse=True)
def _disable_tilelang_cache():
    tilelang.disable_cache()
    try:
        yield
    finally:
        tilelang.enable_cache()


def _copy_kernel(dtype, n, threads, *, through_shared=False):
    @T.prim_func
    def kernel(
        source: T.Tensor((n,), dtype),
        output: T.Tensor((n,), dtype),
    ):
        with T.Kernel(1, threads=threads):
            if through_shared:
                shared = T.alloc_shared((n,), dtype)
                T.copy(source, shared)
                T.copy(shared, output)
            else:
                T.copy(source, output)

    return kernel


def _odd_nibble_copy_kernel(dtype, n):
    @T.prim_func
    def kernel(
        source: T.Tensor((n,), dtype),
        output: T.Tensor((n,), dtype),
    ):
        with T.Kernel(1, threads=n // 2):
            for i in T.Parallel(n // 2):
                output[2 * i + 1] = source[2 * i + 1]

    return kernel


def _local_to_packed_global_copy_kernel(dtype, values_per_thread=1):
    n = 128 * values_per_thread

    @T.prim_func
    def kernel(output: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            local = T.alloc_local((values_per_thread,), dtype)
            for i in T.serial(values_per_thread):
                local[i] = 0
            T.copy(local, output[tx * values_per_thread])

    return kernel


def _im2col_to_packed_shared_kernel(dtype):
    @T.prim_func
    def kernel(source: T.Tensor((1, 1, 1, 128), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((1, 128), dtype)
            T.im2col(source, shared, 0, 0, 1, 1, 1, 0)

    return kernel


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize("through_shared", [False, True])
def test_packed_int4_copy_uses_byte_owned_x2_stores(dtype, through_shared):
    n = 256
    compiled = tilelang.compile(
        _copy_kernel(dtype, n, threads=128, through_shared=through_shared),
        out_idx=[1],
        target="hip",
    )
    source = torch.arange(n // 2, dtype=torch.uint8, device="cuda")
    if dtype == "int4":
        source = source.view(torch.int8)

    result = compiled(source)

    assert torch.equal(result.view(torch.uint8), source.view(torch.uint8))
    assert f"tl_{dtype}_packed_store(" not in compiled.get_kernel_source()


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4_copy_rejects_split_byte_ownership(dtype):
    with pytest.raises(
        Exception,
        match="logical elements that share a writable byte",
    ):
        tilelang.compile(
            _copy_kernel(dtype, 128, threads=128),
            out_idx=[1],
            target="hip",
        )


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_local_to_packed_global_copy_rejects_split_byte_ownership(dtype):
    with pytest.raises(
        Exception,
        match="logical elements that share a writable byte",
    ):
        tilelang.compile(
            _local_to_packed_global_copy_kernel(dtype),
            target="hip",
        )


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_local_to_packed_global_copy_accepts_byte_ownership(dtype):
    compiled = tilelang.compile(
        _local_to_packed_global_copy_kernel(dtype, values_per_thread=2),
        out_idx=[0],
        target="hip",
    )
    result = compiled()

    assert torch.all(result.view(torch.uint8) == 0)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_im2col_to_packed_shared_rejects_split_byte_ownership(dtype):
    with pytest.raises(
        Exception,
        match="logical elements that share a writable byte",
    ):
        tilelang.compile(
            _im2col_to_packed_shared_kernel(dtype),
            target="hip",
        )


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_packed_int4_single_nibble_per_byte_has_one_owner(dtype):
    n = 128
    compiled = tilelang.compile(
        _odd_nibble_copy_kernel(dtype, n),
        target="hip",
    )
    source = torch.arange(n // 2, dtype=torch.uint8, device="cuda")
    output = torch.zeros_like(source)
    if dtype == "int4":
        source = source.view(torch.int8)
        output = output.view(torch.int8)

    compiled(source, output)

    assert torch.equal(output.view(torch.uint8) & 0xF0, source.view(torch.uint8) & 0xF0)
    assert torch.all((output.view(torch.uint8) & 0x0F) == 0)


if __name__ == "__main__":
    tilelang.testing.main()
