"""Runtime regression for 32-byte alignment between physical UB versions."""

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_ascend_nz_layout


@tilelang.jit()
def kernel_builder():
    @T.prim_func
    def kernel(OUT: T.Tensor((8, 4), "uint16")):
        with T.Kernel(1):
            # One logical version is 8 uint16 values = 16 bytes. Physical
            # versions must nevertheless begin at 32-byte-aligned addresses.
            ub = T.alloc_shared((4, 2), "uint16")
            T.annotate_buffer_versions({ub: 2})
            for w in T.serial(2):
                for i in T.serial(4):
                    for j in T.serial(2):
                        ub[i, j] = T.uint16(1)
                T.copy(ub[:, :], OUT[w * 4, 0])

    return kernel


def test_multibuffer_copy_uses_32byte_aligned_version_stride():
    kernel = kernel_builder()
    source = kernel.get_kernel_source()
    assert "(&(ub[(w * 16)]))" in source, source

    out = torch.zeros((8, 4), dtype=torch.int16, device="npu").view(torch.uint16)
    kernel(out)
    torch.npu.synchronize()
    expected = torch.zeros((8, 4), dtype=torch.int16, device="npu")
    expected[:, :2] = 1
    assert torch.equal(out.view(torch.int16), expected)


@tilelang.jit()
def dtype_view_kernel_builder():
    @T.prim_func
    def kernel(A: T.Tensor((8,), "uint16"), OUT: T.Tensor((16,), "uint8")):
        with T.Kernel(1):
            # One logical version is 4 uint16 = 8 bytes. A uint8 view aliases the
            # same storage with a different element dtype; both must resolve to
            # the same 32-byte-aligned physical version slot.
            ub = T.alloc_shared((4,), "uint16")
            v8 = T.view(ub, (8,), dtype="uint8")
            T.annotate_buffer_versions({ub: 2})
            for w in T.serial(2):
                for i in T.serial(8):
                    v8[i] = T.uint8(0)
                for i in T.serial(4):
                    ub[i] = A[w * 4 + i]
                T.copy(v8[:], OUT[w * 8])

    return kernel


def test_multibuffer_dtype_changing_view_shares_aligned_storage():
    kernel = dtype_view_kernel_builder()
    source = kernel.get_kernel_source()
    # 8-byte slot padded to a 32-byte version stride, shared by both dtypes.
    assert "[(w * 32)]" in source, source

    a = (torch.arange(8, device="npu") + 1).to(torch.int16).view(torch.uint16)
    out = torch.zeros(16, dtype=torch.uint8, device="npu")
    kernel(a, out)
    torch.npu.synchronize()
    expected = torch.tensor(list(a.view(torch.int16).cpu().numpy().tobytes()), dtype=torch.uint8)
    assert torch.equal(out.cpu(), expected), (out.cpu().tolist(), expected.tolist())


def layout_remap_kernel_builder():
    @T.prim_func
    def kernel(A: T.Tensor((2, 16, 16), "bfloat16"), OUT: T.Tensor((2, 16, 16), "bfloat16")):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 16), "bfloat16")
            T.annotate_layout({ub: make_ascend_nz_layout(ub)})
            T.annotate_buffer_versions({ub: 2})
            for w in T.serial(2):
                for i in T.serial(16):
                    for j in T.serial(16):
                        ub[i, j] = A[w, i, j]
                for i in T.serial(16):
                    for j in T.serial(16):
                        OUT[w, i, j] = ub[i, j]

    return kernel


def test_layout_remap_preserves_multibuffer_version_owner():
    kernel = tilelang.compile(layout_remap_kernel_builder())
    assert kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
