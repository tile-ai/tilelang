import torch
import torch.backends
import tilelang.testing
from tilelang import tvm as tvm
from tvm import DataType
import tilelang.language as T
from tilelang import JITKernel
from tilelang.transform.simplify import apply_simplify
from tilelang.utils.tensor import map_torch_type
from typing import Optional

tilelang.testing.set_random_seed(0)


def gemv_simt(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    trans_A: bool,
    trans_B: bool,
    with_bias: bool = False,
    n_partition: Optional[int] = 4,
    reduce_thread: Optional[int] = 32,
):
    assert n_partition is not None, "n_partition must be provided"
    assert reduce_thread is not None, (
        "reduce_thread must be provided currently, as related bitblas.gpu.gemv.GEMVsch_outer_reduction_with_config is not implemented"
    )

    assert isinstance(N, int) and isinstance(K, int), "Do not support dynamic N and K Currently"

    assert trans_A is False, "Dequantize only implement for trans_A=False currently"
    assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"

    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits

    block_K = reduce_thread * micro_size_k

    A_shape = (M, K)
    B_shape = (N, K)
    Bias_shape = (N,)
    C_shape = (M, N)

    dp4a_size = 4
    use_dp4a = in_dtype == T.int8 and accum_dtype == T.int32

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        Bias: T.Tensor(Bias_shape, out_dtype),
        C: T.Tensor(C_shape, out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, n_partition), M, threads=(reduce_thread, n_partition)) as (
            bx,
            by,
        ):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_local = T.alloc_local((micro_size_k,), in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.get_thread_binding(0)
            ni = T.get_thread_binding(1)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k):
                    B_local[v] = B[
                        bx * n_partition + ni,
                        ko * block_K + kr * micro_size_k + v,
                    ]

                if use_dp4a:
                    for ki in T.serial(micro_size_k // dp4a_size):
                        T.dp4a(
                            A_local[ki * dp4a_size],
                            B_local[ki * dp4a_size],
                            accum_res[0],
                        )
                else:
                    for ki in T.serial(micro_size_k):
                        accum_res[0] += A_local[ki].astype(accum_dtype) * B_local[ki].astype(accum_dtype)

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                if with_bias:
                    C[by, bx * n_partition + ni] = reduced_accum_res[0] + Bias[bx * n_partition + ni]
                else:
                    C[by, bx * n_partition + ni] = reduced_accum_res[0]

    return apply_simplify(main)


def evaluate_gemv_simt(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    trans_A: bool = False,
    trans_B: bool = True,
    with_bias: bool = False,
):
    program = gemv_simt(M, N, K, in_dtype, out_dtype, accum_dtype, trans_A, trans_B, with_bias)

    kernel = JITKernel(program, target="cuda")

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)
    accum_dtype = map_torch_type(accum_dtype)

    if in_dtype in {torch.int8, torch.int32}:
        A = torch.randint(-128, 128, (M, K), dtype=torch.int8).to(in_dtype).cuda()
        B = torch.randint(-128, 128, (N, K), dtype=torch.int8).to(in_dtype).cuda()
        Bias = torch.randint(-128, 128, (N,), dtype=torch.int32).to(accum_dtype).cuda()
    elif in_dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
        A = torch.randn(M, K).to(in_dtype).cuda()
        B = torch.randn(N, K).to(in_dtype).cuda()
        Bias = torch.randn(N).to(accum_dtype).cuda()
    else:
        A = torch.randn(M, K).to(in_dtype).cuda() - 0.5
        B = torch.randn(N, K).to(in_dtype).cuda() - 0.5
        Bias = torch.randn(N).to(accum_dtype).cuda() - 0.5

    C = torch.zeros(M, N).to(out_dtype).cuda()

    if with_bias:
        kernel(A, B, Bias, C)
    else:
        kernel(A, B, C)

    ref_c = torch.mm(A.to(torch.float32), B.T.to(torch.float32))
    if with_bias:
        ref_c += Bias.to(torch.float32)

    print(C)
    print(ref_c)
    tilelang.testing.torch_assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 9)
def test_gemv_simt():
    evaluate_gemv_simt(1, 1024, 1024, T.float8_e4m3fn, T.float32, T.float32, with_bias=False)
    evaluate_gemv_simt(1, 1024, 1024, T.float8_e5m2, T.float32, T.float32, with_bias=False)


if __name__ == "__main__":
    tilelang.testing.main()
