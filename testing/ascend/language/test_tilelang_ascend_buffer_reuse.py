"""Runtime regressions for on-chip buffer lifetime and alias proofs."""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_dma_tail_is_not_overwritten_by_an_earlier_allocation(disable_reuse, padded):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    width = 7 if padded else 8

    @T.prim_func
    def main(x: T.Tensor((8192,), "int32"), y: T.Tensor((width,), "int32"), out: T.Tensor((width,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8192,), "int32")
            b = T.alloc_shared((8192,), "int32")
            with T.Stage(0):
                T.copy(x, a)
            if padded:
                T.copy(y, b[8184 : 8184 + width], pad_value=0)
            else:
                T.copy(y, b[8184 : 8184 + width])
            T.copy(b[8184 : 8184 + width], out)

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    x = torch.full((8192,), 100, dtype=torch.int32, device="npu")
    y = torch.full((width,), 200, dtype=torch.int32, device="npu")
    expected = torch.full((width,), 200, dtype=torch.int32)
    for _ in range(10):
        actual = kernel(x, y)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("stride", [1, 16])
def test_padded_copy_preserves_values_outside_physical_rows(disable_reuse, stride):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    @T.prim_func
    def main(A: T.Tensor((30,), "float32"), out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((512,), "float32")
            b = T.alloc_shared((512,), "float32")
            view = T.StridedTensor((32,), (stride,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                view[31] = 100.0
                b[31 * stride] = 200.0
                T.ascend_set_copy_pad_value(b[31 * stride], dtype="float32")
                # Order b's completion before the copy through a DMA round
                # trip; A is scratch and is never read back by scalar code.
                with T.Task():
                    for i in T.serial(30):
                        b[i] = 1.0
                T.copy(b[:30], A)
                T.copy(A, view[:30], data_select=True)
                out[0] = view[0] + view[31]

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    source = torch.ones((30,), dtype=torch.float32, device="npu")
    expected = torch.tensor([201.0 if stride == 1 else 101.0], dtype=torch.float32)
    for _ in range(10):
        actual = kernel(source)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_changed_index_keeps_the_original_buffer_value_live(disable_reuse):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            indices = T.alloc_shared((1,), "int32")
            with T.Stage(0):
                a[0] = 100
                b[0] = 200
                out[1] = b[0]
                indices[0] = 1
                a[indices[0]] = 300
                indices[0] = 0
                out[0] = a[indices[0]]

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.tensor([100, 200], dtype=torch.int32)
    for _ in range(10):
        actual = kernel()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_pointer_store_outside_view_waits_before_scalar_read(disable_reuse, strided):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    @T.prim_func
    def main(out: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            allocation = T.alloc_shared((256,), "float32")
            if strided:
                view = T.StridedTensor((80,), (2,), "float32", data=allocation.data, scope="shared.dyn")
            else:
                view = tvm.tirx.decl_buffer((96,), "float32", data=allocation.data, scope="shared.dyn")
            with T.Stage(0):
                with T.SimdVF():
                    T.simd.vsts(view[64], T.simd.vdup(T.float32(1), "float32"))
                out[0] = view[88 if strided else 96]

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.ones((1,), dtype=torch.float32)
    for _ in range(10):
        actual = kernel()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_pointer_reads_physical_elements_outside_the_logical_view(disable_reuse, strided):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    view_size = 8 if strided else 1

    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            T.import_source("__aicore__ inline int32_t read_view_second(__ubuf__ int32_t* p) { return p[1]; }\n")
            a = T.alloc_shared((16,), "int32")
            b = T.alloc_shared((16,), "int32")
            if strided:
                view = T.StridedTensor((view_size,), (2,), "int32", data=a.data, scope="shared.dyn")
            else:
                view = tvm.tirx.decl_buffer((view_size,), "int32", data=a.data, scope="shared.dyn")
            with T.Stage(0):
                a[1] = 100
                b[1] = 200
                out[1] = b[1]
                with T.Task():
                    for i in T.serial(view_size):
                        view[i] = 300
                out[0] = T.call_extern("int32", "read_view_second", T.access_ptr(view[0], "r", extent=2))

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.tensor([100, 200], dtype=torch.int32)
    for _ in range(10):
        actual = kernel()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
@pytest.mark.parametrize("metadata_kind", ["stride", "offset", "shape"])
def test_changed_layout_metadata_keeps_the_original_physical_value(disable_reuse, metadata_kind):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((16,), "int32")
            b = T.alloc_shared((16,), "int32")
            metadata = T.alloc_shared((1,), "int32")
            with T.Stage(0):
                metadata[0] = 1
                if metadata_kind == "stride":
                    view = T.StridedTensor((8,), (metadata[0],), "int32", data=a.data, scope="shared.dyn")
                elif metadata_kind == "offset":
                    view = tvm.tirx.decl_buffer((8,), "int32", data=a.data, elem_offset=metadata[0], scope="shared.dyn")
                else:
                    view = tvm.tirx.decl_buffer((2, metadata[0]), "int32", data=a.data, scope="shared.dyn")
                a[1] = 100
                b[1] = 200
                out[1] = b[1]
                metadata[0] = 2
                if metadata_kind == "shape":
                    view[1, 0] = 300
                elif metadata_kind == "offset":
                    view[0] = 300
                else:
                    view[1] = 300
                metadata[0] = 1
                if metadata_kind == "shape":
                    out[0] = view[1, 0]
                elif metadata_kind == "offset":
                    out[0] = view[0]
                else:
                    out[0] = view[1]

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.tensor([100, 200], dtype=torch.int32)
    for _ in range(10):
        actual = kernel()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_layout_metadata_storage_is_not_reused_before_its_reader(disable_reuse):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((16,), "int32")
            b = T.alloc_shared((16,), "int32")
            metadata = T.alloc_shared((1,), "int32")
            with T.Stage(0):
                metadata[0] = 1
                view = T.StridedTensor((8,), (metadata[0],), "int32", data=a.data, scope="shared.dyn")
                a[1] = 100
                a[2] = 300
                b[0] = 2
                out[1] = b[0]
                out[0] = view[1]

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.tensor([100, 2], dtype=torch.int32)
    for _ in range(10):
        actual = kernel()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_gather_reads_beyond_the_declared_compact_pointer_region(disable_reuse):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")

    @T.prim_func
    def main(out: T.Tensor((65,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((512,), "float32")
            b = T.alloc_shared((512,), "float32")
            c = T.alloc_shared((64,), "float32")
            with T.Stage(0):
                with T.Task():
                    for i in T.serial(512):
                        a[i] = T.float32(100)
                with T.Task():
                    for i in T.serial(512):
                        b[i] = T.float32(200)
                out[64] = b[256]
                with T.Task():
                    for i in T.serial(64):
                        a[i] = T.float32(300)
                with T.SimdVF():
                    value = T.simd.vgather2(a[0], T.simd.vdup(T.uint32(256), "uint32"))
                    T.simd.vsts(c[0], value)
                T.copy(c, out[:64])

    kernel = tilelang.compile(main, out_idx=-1, target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.full((65,), 100, dtype=torch.float32)
    expected[64] = 200
    for _ in range(10):
        actual = kernel()
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_nd2nz_post_copy_preserves_data_beyond_its_base_element(disable_reuse):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    from tilelang.layout import make_ascend_nz_layout

    @T.prim_func
    def main(B: T.Tensor((16, 32), "float16"), out: T.Tensor((16, 16), "float32"), out_b: T.Tensor((1,), "float16")):
        with T.Kernel(1):
            a = T.alloc_shared((17 * 32,), "float16")
            b = T.alloc_shared((17 * 32,), "float16")
            dst = T.alloc_l1((16, 32), "float16")
            rhs = T.alloc_l1((16, 32), "float16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.annotate_layout({dst: make_ascend_nz_layout(dst)})
            with T.Stage(0):
                with T.Task():
                    for i in T.serial(17 * 32):
                        a[i] = T.float16(100)
                with T.Task():
                    for i in T.serial(17 * 32):
                        b[i] = T.float16(200)
                out_b[0] = b[32]
                a[0] = T.float16(300)
                T.ascend_nd2nz_post_copy(dst[0, 0], a[0], 16, 32, 16, "half")
            T.copy(B, rhs)
            T.gemm(dst, rhs, accum, clear_accum=True, transpose_B=True)
            T.copy(accum, out)

    rhs = torch.zeros((16, 32), dtype=torch.float16)
    rhs[:, 16:] = torch.eye(16, dtype=torch.float16)
    rhs = rhs.to("npu")
    kernel = tilelang.compile(main, out_idx=[1, 2], target="ascend", pass_configs={"tl.disable_shared_memory_reuse": disable_reuse})
    expected = torch.full((16, 16), 100, dtype=torch.float32)
    expected_b = torch.tensor([200], dtype=torch.float16)
    for _ in range(10):
        actual, actual_b = kernel(rhs)
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_b.cpu(), expected_b, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
