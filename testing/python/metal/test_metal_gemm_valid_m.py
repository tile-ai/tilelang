import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
import torch
from tilelang import tvm


M, N, K = 64, 32, 16


@T.prim_func
def partial_m_gemm(
    a: T.Tensor((M, K), T.float16),
    b: T.Tensor((K, N), T.float16),
    output: T.Tensor((M, N), T.float32),
    valid_m: T.int32,
):
    with T.Kernel(1, threads=128):
        a_shared = T.alloc_shared((M, K), T.float16)
        b_shared = T.alloc_shared((K, N), T.float16)
        accum = T.alloc_fragment((M, N), T.float32)
        T.copy(a, a_shared)
        T.copy(b, b_shared)
        T.fill(accum, -7.0)
        T.gemm(a_shared, b_shared, accum, clear_accum=True, valid_m=valid_m)
        T.copy(accum, output)


def test_runtime_valid_m_is_preserved_in_metal_codegen():
    with tvm.transform.PassContext(), tvm.target.Target("metal"):
        artifact = tilelang.lower(
            partial_m_gemm,
            target="metal",
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    source = artifact.kernel_source
    assert "< arg.bound[0]" in source
    assert "<= arg.bound[0]" in source
    assert "simdgroup_multiply_accumulate" in source


def test_static_valid_m_is_validated_at_construction():
    with pytest.raises(ValueError, match="valid_m must be in"):

        @T.prim_func
        def invalid(
            a: T.Tensor((M, K), T.float16),
            b: T.Tensor((K, N), T.float16),
            output: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(1, threads=128):
                T.gemm(a, b, output, clear_accum=True, valid_m=M + 1)


@tilelang.testing.requires_metal
def test_runtime_valid_m_skips_wholly_invalid_instruction_rows():
    compiled = tilelang.compile(
        partial_m_gemm,
        out_idx=[],
        target="metal",
        target_host="c",
        execution_backend="tvm_ffi",
    )
    a = torch.randn((M, K), dtype=torch.float16, device="mps")
    b = torch.randn((K, N), dtype=torch.float16, device="mps")
    output = torch.full((M, N), -7.0, dtype=torch.float32, device="mps")
    valid_m = 17
    compiled(a, b, output, valid_m)
    torch.mps.synchronize()

    expected = a.float() @ b.float()
    torch.testing.assert_close(output[:valid_m].cpu(), expected[:valid_m].cpu(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(output[32:].cpu(), torch.full((M - 32, N), -7.0))


if __name__ == "__main__":
    tilelang.testing.main()
