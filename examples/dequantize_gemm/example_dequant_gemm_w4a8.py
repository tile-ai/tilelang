import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tvm import tir
import itertools
import torch
import argparse


def _tir_u8_to_i4_to_i8(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == T.int8
    assert val.dtype == T.uint8

    mask = tir.const((1 << nbit) - 1, T.uint8)

    i4 = (val >> (pos.astype(T.uint8) * tir.const(nbit, T.uint8))) & mask

    i8_shifted = tir.reinterpret(T.int8, i4 << tir.const(4, T.uint8))
    i8 = i8_shifted >> tir.const(4, T.int8)
    return i8


def get_configs():
    iter_params = dict(
        block_M=[64, 128],
        block_N=[64, 128],
        block_K=[128, 256],
        num_stages=[1, 2],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.jit(out_idx=[1])
def _convert_test(N, K, block_N, block_K, in_dtype, num_bits=4, threads=128):
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.uint8
    B_shape = (N, K // num_elems_per_byte)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    @T.prim_func
    def main(
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((N, K), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)

            for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=1):
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                for i, j in T.Parallel(block_N, block_K):
                    B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
                        num_bits,
                        B_local[i, j // num_elems_per_byte],
                        j % num_elems_per_byte,
                        dtype=in_dtype,
                    )
                T.copy(B_dequantize_local, C[bx * block_N, k * block_K])

    return main


def torch_convert(tensor):
    def _convert(val, pos):
        assert val.dtype == torch.uint8
        val = val.view(torch.int8)
        mask = (1 << 4) - 1
        i4_shifted = (val >> (pos * 4)) & mask
        i4 = (i4_shifted << 4) >> 4

        return i4.view(torch.int8)

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.int8, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor


def ref_program(A, qB):
    dtypeC = T.int32
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C.transpose(0, 1)


def matmul_int8xint4(M, N, K, in_dtype, out_dtype, accum_dtype, num_bits=4, tune=False):
    @tilelang.jit(out_idx=[2])
    def kernel_func(block_M, block_N, block_K, num_stages, threads):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = T.uint8
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_local_shape = (block_N, block_K)

        assert K % (block_K) == 0

        @T.prim_func
        def main(
            A: T.Tensor(A_shape, in_dtype),
            B: T.Tensor(B_shape, storage_dtype),
            Ct: T.Tensor((N, M), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_local_shape, in_dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_local_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout(
                    {
                        B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    }
                )

                T.clear(Ct_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, Ct_shared)
                T.copy(Ct_shared, Ct[bx * block_N : (bx + 1) * block_N, by * block_M : (by + 1) * block_M])

        return main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[2])
        def kernel(block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads).prim_func

        return kernel()

    else:

        def kernel(block_M, block_N, block_K, num_stages, threads):
            return kernel_func(block_M, block_N, block_K, num_stages, threads)

        return kernel


def main(m=128, n=256, k=256, tune=False):
    total_flops = 2 * m * n * k
    if not tune:
        kernel = matmul_int8xint4(m, n, k, T.int8, T.int32, T.int32, num_bits=4, tune=tune)(
            block_M=32, block_N=32, block_K=128, num_stages=1, threads=128
        )
        profiler = kernel.get_profiler()
        profiler.assert_allclose(ref_program, rtol=1e-2, atol=1e-2)
        print("All checks pass.")

        latency = profiler.do_bench(warmup=50)
        print(f"Tilelang: {latency} ms")

    else:
        best_result = matmul_int8xint4(m, n, k, T.int8, T.int32, T.int32, num_bits=4, tune=tune)
        best_latency = best_result.latency
        best_config = best_result.config
        print(f"Bset latency: {best_latency}")
        print(f"Best config: {best_config}")
        print(f"Best tflops: {total_flops / best_latency * 1e-9}")


def run_regression_perf(m=128, n=256, k=256):
    kernel = matmul_int8xint4(m, n, k, "int8", "int32", "int32", num_bits=4, tune=False)(
        block_M=32, block_N=32, block_K=128, num_stages=1, threads=128
    )
    profiler = kernel.get_profiler()
    return profiler.do_bench(warmup=10, rep=100, backend="cupti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=512, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=512, help="Matrix dimension K")
    parser.add_argument("--tune", action="store_true", help="Enable tuning")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    main(M, N, K, args.tune)
    # main(M, N, K, True)
