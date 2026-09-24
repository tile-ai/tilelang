import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.jit.adapter.nvrtc import is_nvrtc_available
from tvm import tirx


FP8_DTYPES = ["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("vector_element", [False, True])
def test_packed_store_codegen(dtype, lanes, vector_element):
    # Cover both a Ramp into a scalar buffer and a vector-typed buffer element.
    buffer_dtype = f"{dtype}x{lanes}" if vector_element else dtype
    src = tirx.decl_buffer((128,), buffer_dtype, name="src")
    dst = tirx.decl_buffer((128,), buffer_dtype, name="dst")
    index = 0 if vector_element else tirx.Ramp(0, 1, lanes)
    body = tirx.BufferStore(dst, tirx.BufferLoad(src, [index]), [index])
    func = tirx.PrimFunc([src.data, dst.data], body, buffer_map={src.data: src, dst.data: dst})
    func = func.with_attr("global_symbol", "copy_vector")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile")
    source = build(tvm.IRModule({"copy_vector": func}), tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})).inspect_source()
    if lanes == 32:
        # Preserve the native 256-bit global-memory path.
        assert "tl::store_global_256" in source
        assert "tl::store_packed_vector" not in source
    else:
        assert "tl::store_packed_vector" in source


def copy_through_shared(dtype, lanes):
    threads, blocks = 128, 4
    size = threads * blocks * lanes

    @T.prim_func
    def main(src: T.Tensor((size,), dtype), dst: T.Tensor((size,), dtype)):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding()
            local = T.alloc_local((lanes,), dtype)
            shared = T.alloc_shared((threads * lanes,), dtype)
            for i in T.vectorized(lanes):
                local[i] = src[(bx * threads + tx) * lanes + i]
            for i in T.vectorized(lanes):
                shared[tx * lanes + i] = local[i]
            T.sync_threads()
            for i in T.vectorized(lanes):
                dst[(bx * threads + tx) * lanes + i] = shared[((tx + 1) % threads) * lanes + i]

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_packed_copy_bits(dtype, lanes):
    _check_copy_bits(dtype, lanes, "auto")


def _check_copy_bits(dtype, lanes, execution_backend):
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        pytest.skip(f"PyTorch {dtype} is unavailable")
    kernel = tilelang.compile(copy_through_shared(dtype, lanes), target="cuda", execution_backend=execution_backend)
    # Include every byte pattern (NaNs, signed zero, etc.) and rotate across
    # threads so that shared-memory loads cannot be forwarded from registers.
    bits = torch.arange(4 * 128 * lanes, device="cuda").to(torch.uint8)
    output = torch.empty_like(bits)
    kernel(bits.view(torch_dtype), output.view(torch_dtype))
    expected = bits.reshape(4, 128, lanes).roll(-1, dims=1).flatten()
    assert torch.equal(output, expected)
    assert "tl::store_packed_vector" in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
@pytest.mark.skipif(not is_nvrtc_available, reason="NVRTC is unavailable")
@pytest.mark.parametrize("lanes", [2, 8, 32])
def test_packed_copy_nvrtc(lanes):
    _check_copy_bits("float8_e4m3fn", lanes, "nvrtc")


def repeated_cast(dtype, size=65536, repeats=16):
    lanes, threads = 8, 128

    @T.prim_func
    def main(src: T.Tensor((size,), T.bfloat16), dst: T.Tensor((repeats, size), dtype)):
        with T.Kernel(size // (threads * lanes), threads=threads) as bx:
            tx = T.get_thread_binding()
            local = T.alloc_local((lanes,), T.bfloat16)
            shared = T.alloc_shared((repeats, threads * lanes), dtype)
            for i in T.vectorized(lanes):
                local[i] = src[(bx * threads + tx) * lanes + i]
            for r in T.unroll(repeats):
                for i in T.vectorized(lanes):
                    shared[r, tx * lanes + i] = local[i] * T.bfloat16(1.0 / (r + 1))
            for r in T.unroll(repeats):
                for i in T.vectorized(lanes):
                    dst[r, (bx * threads + tx) * lanes + i] = shared[r, tx * lanes + i]

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_repeated_bf16_cast(dtype):
    kernel = tilelang.compile(repeated_cast(dtype), target="cuda")
    # Exercise all finite BF16 encodings, including subnormals and signed zero.
    src = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    src = torch.where(torch.isfinite(src), src, 0)
    output = torch.empty((16, 65536), dtype=getattr(torch, dtype), device="cuda")
    kernel(src, output)
    for r in range(16):
        scale = torch.tensor(1.0 / (r + 1), dtype=torch.bfloat16, device="cuda")
        # Match CUDA's SATFINITE conversion rather than PyTorch's overflow
        # to NaN/Inf. Multiplication still rounds to BF16 before conversion.
        limit = torch.finfo(output.dtype).max
        expected = (src * scale).float().clamp(-limit, limit).to(output.dtype)
        assert torch.equal(output[r].view(torch.uint8), expected.view(torch.uint8))
    source = kernel.get_kernel_source()
    assert "tl::store_packed_vector<uint>" in source
    assert "tl::store_packed_vector<uint2>" in source


if __name__ == "__main__":
    tilelang.testing.main()
