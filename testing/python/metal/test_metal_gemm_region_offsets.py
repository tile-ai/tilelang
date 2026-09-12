import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


@T.prim_func
def region_gemm(
    a: T.Tensor((48, 48), T.float16),
    b: T.Tensor((80, 48), T.float16),
    output: T.Tensor((32, 32), T.float32),
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((48, 48), T.float16)
        b_shared = T.alloc_shared((80, 48), T.float16)
        accum = T.alloc_fragment((32, 32), T.float32)
        T.copy(a, a_shared)
        T.copy(b, b_shared)
        T.clear(accum)
        T.gemm(a_shared[8:40, 16:48], b_shared[24:56, 8:40], accum, transpose_B=True)
        T.copy(accum, output)


def test_region_offsets_lower_for_transposed_rhs():
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        tilelang.lower(region_gemm, target="metal")


@tilelang.testing.requires_metal
def test_region_offsets_address_physical_buffer_axes():
    compiled = tilelang.compile(
        region_gemm,
        out_idx=[],
        target="metal",
        target_host="c",
        execution_backend="tvm_ffi",
    )
    a = torch.randn((48, 48), dtype=torch.float16, device="mps")
    b = torch.randn((80, 48), dtype=torch.float16, device="mps")
    output = torch.empty((32, 32), dtype=torch.float32, device="mps")
    compiled(a, b, output)
    torch.mps.synchronize()
    expected = a[8:40, 16:48].float() @ b[24:56, 8:40].float().T
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
