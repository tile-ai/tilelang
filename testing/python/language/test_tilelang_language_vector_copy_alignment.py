import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("src_offset,dst_offset", [(0, 0), (4, 0), (0, 4), (4, 4)])
def test_vector_copy_256_alignment(src_offset, dst_offset):
    @T.prim_func
    def copy8(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (8,), dtype=T.float32, align=16)
        B = T.match_buffer(B_ptr, (8,), dtype=T.float32, align=16)
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                for i in T.vectorized(8):
                    B[i] = A[i]

    kernel = tilelang.compile(copy8, execution_backend="tvm_ffi")
    source = kernel.get_kernel_source()
    assert "load_global_256" in source
    assert "store_global_256" in source

    a = torch.arange(16, device="cuda", dtype=torch.float32)
    b = torch.full((16,), -1.0, device="cuda")
    src = a[src_offset : src_offset + 8]
    dst = b[dst_offset : dst_offset + 8]
    assert src.data_ptr() % 32 == src_offset * 4
    assert dst.data_ptr() % 32 == dst_offset * 4

    kernel(src, dst)
    expected = torch.full_like(b, -1.0)
    expected[dst_offset : dst_offset + 8] = src
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
