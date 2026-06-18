import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
def test_semantic_fp4_gemm_uses_unpacked_carrier_fragments():
    def assert_unpacked_shared_carrier(tir, name):
        assert f'{name}: T.handle("custom[float4_e2m1_unpacked]", "shared.dyn")' in tir

    cases = [
        (T.float4_e2m1fn, T.float4_e2m1fn, "A", "B", "e2m1", "e2m1"),
        (T.float8_e4m3fn, T.float4_e2m1fn, None, "B", "e4m3", "e2m1"),
        (T.float4_e2m1fn, T.float8_e4m3fn, "A", None, "e2m1", "e4m3"),
    ]

    def lower_gemm(a_dtype, b_dtype):
        @T.prim_func
        def main(
            A: T.Tensor((128, 128), a_dtype),
            B: T.Tensor((128, 128), b_dtype),
            C: T.Tensor((128, 128), T.float32),
        ):
            with T.Kernel(1, 1, threads=128):
                A_shared = T.alloc_shared((128, 128), a_dtype)
                B_shared = T.alloc_shared((128, 128), b_dtype)
                C_local = T.alloc_fragment((128, 128), T.float32)

                T.clear(C_local)
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[0, 0])

        with tvm.target.Target("cuda"):
            artifact = tilelang.lower(main, target="cuda")
        return str(artifact.device_mod), artifact.kernel_source

    for a_dtype, b_dtype, a_carrier, b_carrier, a_mma, b_mma in cases:
        tir, src = lower_gemm(a_dtype, b_dtype)
        if a_carrier:
            assert 'A: T.handle("float4_e2m1fn", "global")' in tir
            assert_unpacked_shared_carrier(tir, "A_shared")
            assert 'A_local = T.alloc_buffer((64,), "custom[float4_e2m1_unpacked]", scope="local")' in tir
        if b_carrier:
            assert 'B: T.handle("float4_e2m1fn", "global")' in tir
            assert_unpacked_shared_carrier(tir, "B_shared")
            assert 'B_local = T.alloc_buffer((64,), "custom[float4_e2m1_unpacked]", scope="local")' in tir
        assert f'T.ptx_mma("float32", "m16n8k32", "row", "col", "{a_mma}", "{b_mma}", "fp32"' in tir
        assert "tl::ptx_ldmatrix_x4" in src
        assert "tl::ptx_ldmatrix_b4x16" not in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
def test_semantic_fp4_gemm_pipeline_uses_unpacked_carrier_cp_async():
    @T.prim_func
    def main(
        A: T.Tensor((256, 256), T.float4_e2m1fn),
        B: T.Tensor((256, 256), T.float4_e2m1fn),
        C: T.Tensor((256, 256), T.float32),
    ):
        with T.Kernel(2, 2, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), T.float4_e2m1fn)
            B_shared = T.alloc_shared((128, 64), T.float4_e2m1fn)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)
            for ko in T.Pipelined(4, num_stages=2):
                T.copy(A[by * 128, ko * 64], A_shared)
                T.copy(B[bx * 128, ko * 64], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * 128, bx * 128])

    with tvm.target.Target("cuda"):
        artifact = tilelang.lower(main, target="cuda")

    tir = str(artifact.device_mod)
    src = artifact.kernel_source
    assert 'A: T.handle("float4_e2m1fn", "global")' in tir
    assert 'B: T.handle("float4_e2m1fn", "global")' in tir
    assert "custom[float4_e2m1_unpacked]" in tir
    assert "tl::cp_async_gs" in src
    assert "tl::ptx_ldmatrix_x4" in src
    assert "tl::ptx_ldmatrix_b4x16" not in src
    compact_src = src.replace(" ", "")
    assert "((((((int)threadIdx.x)&15)>>3)+((((int)threadIdx.x)&3)>>1))&1)*16" in compact_src
