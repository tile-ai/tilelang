"""On-device T.fill inside SimdVF, including multi-buffered pipeline slots."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@pytest.mark.parametrize("pass_name", ["LowerSimdVFFill", "AscendSimdVFLowerParallel"])
def test_simdvf_fill_fp4_rejected(pass_name):
    from tilelang import tvm
    from tilelang.ascend import transform as ascend

    @T.prim_func
    def kernel():
        temp = T.alloc_shared((64,), "float4_e2m1fn")
        with T.SimdVF():
            T.fill(temp, 0.0)

    with pytest.raises(ValueError, match="T.fill inside SimdVF.*float4_e2m1fn"):
        getattr(ascend, pass_name)()(tvm.IRModule({"main": kernel}))


@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_simdvf_fill_full_unversioned_ub_npu(target):
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                T.fill(temp, 1.0)
            T.copy(temp, O)

    compiled = tilelang.compile(kernel, target=target, out_idx=-1)
    out = compiled()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.full((64,), 1.0, dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_simdvf_fill_2d_ub_npu(target):
    @T.prim_func
    def kernel(O: T.Buffer((32, 64), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((32, 64), "float32")
            with T.SimdVF():
                T.fill(temp, 0.0)
            T.copy(temp, O)

    compiled = tilelang.compile(kernel, target=target, out_idx=-1)
    out = compiled()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.zeros((32, 64), dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_simdvf_fill_multibuffered_ub_inside_pipeline_npu(target):
    tile = 64

    @T.prim_func
    def kernel(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            acc = T.alloc_shared((tile,), "float32")
            tmp = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({acc: 2, tmp: 2})
            for w in T.Pipelined(8, num_stages=2):
                T.copy(A[w * tile : (w + 1) * tile], tmp)
                with T.SimdVF():
                    T.fill(acc, 1.0)
                    for i in T.Parallel(tile):
                        tmp[i] = tmp[i] + acc[i]
                T.copy(tmp, C[w * tile : (w + 1) * tile])

    compiled = tilelang.compile(kernel, target=target, out_idx=-1)
    device = torch.device("npu")
    torch.manual_seed(0)
    a = torch.randn(8 * tile, dtype=torch.float32, device=device)
    c = compiled(a)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + 1.0, rtol=0, atol=1e-5)


@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_simdvf_fill_aligned_subregion_ub_npu(target):
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((128,), "float32")
            with T.SimdVF():
                T.fill(temp[8:72], 1.0)
            T.copy(temp[8:72], O)

    compiled = tilelang.compile(kernel, target=target, out_idx=-1)
    out = compiled()
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), torch.full((64,), 1.0, dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("start", [1, 4])
@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_simdvf_fill_misaligned_subregion_rejected(start, target):
    @T.prim_func
    def kernel(O: T.Buffer((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((128,), "float32")
            with T.SimdVF():
                T.fill(temp[start : start + 64], 1.0)
            T.copy(temp[start : start + 64], O)

    with pytest.raises(ValueError, match="32-byte-aligned"):
        tilelang.compile(kernel, target=target, out_idx=-1)


if __name__ == "__main__":
    tilelang.testing.main()


@pytest.mark.parametrize("value", [0, -7])
@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_pto_fill_integer_and_float(value, dtype):
    @T.prim_func
    def kernel(O: T.Buffer((128,), dtype)):
        with T.Kernel(1):
            temp = T.alloc_shared((128,), dtype)
            with T.SimdVF():
                T.fill(temp, value)
            T.copy(temp, O)

    result = tilelang.compile(kernel, target="pto", out_idx=-1)()
    torch.testing.assert_close(result.cpu(), torch.full((128,), value, dtype=getattr(torch, dtype)), rtol=0, atol=0)


@pytest.mark.pto
@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "src_dtype,dst_dtype,size,narrow_dtype",
    [("int32", "float32", 64, None), ("float32", "float16", 128, None), ("int32", "int32", 64, "int8")],
)
def test_pto_fill_runtime_scalar_cast(src_dtype, dst_dtype, size, narrow_dtype):
    scalar_type = getattr(T, src_dtype)
    narrow = narrow_dtype is not None

    @T.prim_func
    def kernel(value: scalar_type, O: T.Tensor((size,), dst_dtype)):
        with T.Kernel(1):
            temp = T.alloc_shared((size,), dst_dtype)
            with T.SimdVF():
                if narrow:
                    T.fill(temp, T.Cast(dst_dtype, T.Cast(narrow_dtype, value)))
                else:
                    T.fill(temp, value)
            T.copy(temp, O)

    compiled = tilelang.compile(kernel, target="pto", out_idx=-1)
    for value in (257, -257, 127, 128) if narrow else (7, -7):
        scalar = float(value) if src_dtype.startswith("float") else value
        output = compiled(scalar)
        converted = torch.tensor(scalar, dtype=getattr(torch, src_dtype))
        if narrow:
            converted = converted.to(getattr(torch, narrow_dtype))
        expected = converted.to(getattr(torch, dst_dtype)).expand(size)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@tilelang.testing.requires_ascend
def test_pto_vdup_runtime_narrowing_cast():
    @T.prim_func
    def kernel(value: T.float32, O: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                vector = T.simd.vdup(T.Cast("float16", value), "float32")
                T.simd.vsts(temp[0], vector)
            T.copy(temp, O)

    compiled = tilelang.compile(kernel, target="pto", out_idx=-1)
    for value in (1.0003, 1.0007, -1.0003):
        output = compiled(value)
        expected = torch.tensor(value, dtype=torch.float32).half().float().expand(64)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.pto
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("strided", [False, True], ids=["short", "non-contiguous"])
def test_pto_fill_legacy_regions(strided):
    shape = (2, 64) if strided else (32,)

    @T.prim_func
    def kernel(O: T.Tensor(shape, "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared(shape, "float32")
            if strided:
                T.fill(temp, 123.0)
            with T.SimdVF():
                if strided:
                    T.fill(temp[:, :32], -7.0)
                else:
                    T.fill(temp, -7.0)
            T.copy(temp, O)

    output = tilelang.compile(kernel, target="pto", out_idx=-1)()
    expected = torch.full(shape, -7.0)
    if strided:
        expected[:, 32:] = 123.0
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
