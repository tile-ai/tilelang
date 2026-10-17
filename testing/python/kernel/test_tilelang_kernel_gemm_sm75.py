import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def _make_gemm_kernel(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _assert_mma_sync_shape(source, a_type, c_type, m, n, k):
    expected = f"tl::mma_sync<tl::DataType::{a_type}, tl::DataType::{a_type}, tl::DataType::{c_type}, {m}, {n}, {k}, false, true>"
    assert expected in source
    assert "mma_sync_sm70" not in source


def _pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    tensor_i16 = tensor.to(torch.int16)
    packed = (tensor_i16[..., ::2] & 0x0F) | ((tensor_i16[..., 1::2] & 0x0F) << 4)
    return packed.to(torch.int8).contiguous()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_f16_gemm_uses_m16n8k8_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 32, T.float16, T.float32, T.float32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kFloat16", "kFloat32", 16, 8, 8)

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref = a.float() @ b.float().T

    tilelang.testing.torch_assert_close(c, ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_f16_gemm_f16_accum_uses_m16n8k8_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 32, T.float16, T.float16, T.float16),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kFloat16", "kFloat16", 16, 8, 8)

    a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.1
    c = kernel(a, b)
    ref = (a.float() @ b.float().T).to(torch.float16)

    tilelang.testing.torch_assert_close(c, ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_uint8_gemm_uses_m8n8k16_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 64, T.uint8, T.int32, T.int32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kUInt8", "kInt32", 8, 8, 16)

    a = torch.randint(0, 256, (M, K), device="cuda", dtype=torch.uint8)
    b = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8)
    c = kernel(a, b)
    ref = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda")

    tilelang.testing.torch_assert_close(c, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_int8_gemm_uses_m8n8k16_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 64, T.int8, T.int32, T.int32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kInt8", "kInt32", 8, 8, 16)

    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
    c = kernel(a, b)
    ref = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda")

    tilelang.testing.torch_assert_close(c, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
def test_sm75_int4_gemm_uses_m8n8k32_and_matches_torch():
    M = N = K = 128
    kernel = tilelang.compile(
        _make_gemm_kernel(M, N, K, 64, 64, 64, T.int4, T.int32, T.int32),
        target="cuda",
        out_idx=[2],
    )
    _assert_mma_sync_shape(kernel.get_kernel_source(), "kInt4", "kInt32", 8, 8, 32)

    a = torch.randint(-8, 8, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-8, 8, (N, K), device="cuda", dtype=torch.int8)
    c = kernel(_pack_int4(a), _pack_int4(b))
    ref = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda")

    tilelang.testing.torch_assert_close(c, ref, rtol=0, atol=0)


def _make_pipelined_gemm_kernel(
    M, N, K, block_M, block_N, block_K, num_stages, transpose_B, in_dtype=T.float16, out_dtype=T.float32, accum_dtype=T.float32
):
    B_shape = (N, K) if transpose_B else (K, N)
    B_shared_shape = (block_N, block_K) if transpose_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                if transpose_B:
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                else:
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(7, 5)
@pytest.mark.parametrize("transpose_B", [True, False])
def test_sm75_f16_gemm_pipelined_multi_stage_matches_torch(transpose_B):
    M = N = K = 128
    kernel = tilelang.compile(
        _make_pipelined_gemm_kernel(M, N, K, 64, 64, 32, 2, transpose_B),
        target="cuda",
        out_idx=[2],
    )
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b_shape = (N, K) if transpose_B else (K, N)
    b = torch.randn(b_shape, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref = a.float() @ (b.float().T if transpose_B else b.float())

    tilelang.testing.torch_assert_close(c, ref, rtol=1e-2, atol=1e-2, max_mismatched_ratio=0.01)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "in_dtype, accum_dtype, block_K, transpose_B",
    [
        (T.float16, T.float32, 32, True),
        (T.float16, T.float32, 32, False),
        (T.int8, T.int32, 64, True),
        (T.int4, T.int32, 64, True),
    ],
)
def test_sm75_ldmatrix_shared_reads_stay_in_bounds(in_dtype, accum_dtype, block_K, transpose_B):
    # Cross-compiles for sm_75, so it guards the Turing address math on any
    # CUDA runner. The ldmatrix lane maps step past the shared tile unless the
    # emitter wraps them (see mma_macro_generator.py).
    M = N = K = 128
    kernel = tilelang.compile(
        _make_pipelined_gemm_kernel(M, N, K, 64, 64, block_K, 0, transpose_B, in_dtype, accum_dtype, accum_dtype),
        target={"kind": "cuda", "arch": "sm_75"},
        out_idx=[2],
    )
    (func,) = kernel.artifact.device_mod.functions.values()

    analyzer = tvm.arith.Analyzer()
    buffer_sizes = {}
    ldmatrix_calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.AttrStmt) and str(node.attr_key) == "thread_extent":
            analyzer.update(node.node.var, tvm.arith.ConstIntBound(0, int(node.value) - 1))
        elif isinstance(node, tvm.tirx.For):
            analyzer.update(
                node.loop_var,
                tvm.arith.ConstIntBound(int(node.min), int(node.min) + int(node.extent) - 1),
            )
        elif isinstance(node, (tvm.tirx.AllocBuffer, tvm.tirx.DeclBuffer)):
            size = 1
            for extent in node.buffer.shape:
                size *= int(extent)
            buffer_sizes[node.buffer.data] = size
        elif isinstance(node, tvm.tirx.Call) and str(getattr(node.op, "name", "")) == "tl.ptx_ldmatrix":
            ldmatrix_calls.append(node)

    post_order_visit(func.body, visit)
    assert ldmatrix_calls, "expected ldmatrix lowering for sm_75"

    for call in ldmatrix_calls:
        src = call.args[2]
        assert str(src.op.name) == "tirx.tvm_access_ptr"
        data, offset, extent = src.args[1], src.args[2], int(src.args[3])
        assert data in buffer_sizes, f"ldmatrix source {data.name} has no visible allocation"
        bound = analyzer.const_int_bound(offset)
        assert bound.min_value >= 0
        assert bound.max_value + extent <= buffer_sizes[data], (
            f"ldmatrix reads past the shared tile: offset up to {bound.max_value} + extent {extent} exceeds {buffer_sizes[data]} elements"
        )


if __name__ == "__main__":
    tilelang.testing.main()
