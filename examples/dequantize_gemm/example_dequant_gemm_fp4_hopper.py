# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from tvm import tir
import itertools
import torch
import argparse


def _tir_u8_to_f4_to_f16(nbit: int, val: tir.PrimExpr, pos: tir.PrimExpr, dtype: str):
    assert nbit == 4
    assert dtype == "float16"
    assert val.dtype == "uint8"
    # e_f4 == 0 -> e_f16 = 0
    # e_f4 != 0 -> e_f16 = e_f4 + 8 = e_f4 | (1000)_2
    # s1e2n1
    mask = tir.const((1 << nbit) - 1, "uint16")
    f4 = (val >> (pos.astype("uint16") * tir.const(nbit, "uint16"))) & mask
    s = f4 >> tir.const(3, "uint16")
    e_f4 = f4 & tir.const(7, "uint16")
    e_f16 = e_f4 | tir.const(8, "uint16")
    val_f16 = tir.reinterpret(
        "float16",
        ((e_f16 | (s << tir.const(5, "uint16"))) << tir.const(10, "uint16")).astype("uint16"))
    # return tir.Select(e_f4 == tir.const(0, "uint32"), tir.const(0, "float16"), val_f16)
    return val_f16


def torch_convert(tensor):

    def print_bit(name, val):
        val_cpu = val.cpu().item()
        binary_repr = f'{val_cpu:032b}'
        print(name, binary_repr)

    def _convert(val, pos):
        assert val.dtype == torch.uint8
        val = val.view(torch.int8)
        mask = (1 << 4) - 1
        f4 = ((val >> (pos * 4)) & mask).to(torch.int16)
        s = f4 >> 3
        e_f4 = f4 & 7
        e_f16 = e_f4 | 8
        val_f16 = ((e_f16 | (s << 5)) << 10) & 0xFFFF
        lower_16_bits = (val_f16 & 0xFFFF).to(torch.uint16)
        return lower_16_bits.view(torch.float16)

    N = tensor.shape[0]
    K = tensor.shape[1]
    new_tensor = torch.empty(N, K * 2, dtype=torch.float16, device=tensor.device)
    for i in range(new_tensor.shape[0]):
        for j in range(new_tensor.shape[1]):
            new_tensor[i][j] = _convert(tensor[i][j // 2], j % 2)
    return new_tensor


def test_convert(N, K, block_N, block_K, in_dtype, num_bits=4, threads=128):
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
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

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                for i, j in T.Parallel(block_N, block_K):
                    B_dequantize_local[i, j] = _tir_u8_to_f4_to_f16(
                        num_bits,
                        B_local[i, j // num_elems_per_byte],
                        j % num_elems_per_byte,
                        dtype=in_dtype,
                    )
                T.copy(B_dequantize_local, C[bx * block_N, k * block_K])

    return main


def test_fp4_fp16_convert_close():
    N, K = 256, 256
    block_N, block_K = 64, 64
    program = test_convert(
        N,
        K,
        block_N,
        block_K,
        "float16",
    )

    kernel = tilelang.compile(program, out_idx=[1])

    B = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda").to(torch.uint8)
    tl_out = kernel(B)
    ref_out = torch_convert(B)
    assert torch.allclose(tl_out, ref_out, rtol=0.01, atol=0.01), (tl_out, ref_out)
    print("Pass")


def get_configs():
    block_M = [128]
    block_N = [128, 256]
    block_K = [128]
    num_stages = [2]
    threads = [256]
    splits = [1]
    _configs = list(itertools.product(block_M, block_N, block_K, num_stages, threads, splits))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'block_K': c[2],
        'num_stages': c[3],
        'threads': c[4],
        'split': c[5]
    } for c in _configs]
    return configs


def matmul(M, N, K, in_dtype, out_dtype, accum_dtype, num_bits=4, tune=False):

    def kernel_func(block_M, block_N, block_K, num_stages, threads, split=1):
        num_elems_per_byte = 8 // num_bits
        storage_dtype = "uint8"
        A_shape = (M, K)
        B_shape = (N, K // num_elems_per_byte)
        A_shared_shape = (block_M, block_K)
        B_shared_shape = (block_N, block_K // num_elems_per_byte)
        B_dequantize_shared_shape = (block_N, block_K)
        assert K % (block_K * split) == 0
        KK = K // split

        @T.prim_func
        def main_split(
                A: T.Tensor(A_shape, in_dtype),
                B: T.Tensor(B_shape, storage_dtype),
                Ct: T.Tensor((N, M), out_dtype),
        ):
            SplitC = T.alloc_buffer([
                split, (N + block_N - 1) // block_N * block_N,
                (M + block_M - 1) // block_M * block_M
            ], out_dtype)
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), split,
                    threads=threads) as (bx, by, bz):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout({
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    Ct_shared: tilelang.layout.make_swizzled_layout(Ct_shared),
                })

                T.clear(Ct_local)
                for k in T.Pipelined(K // (block_K * split), num_stages=num_stages):
                    T.copy(A[by * block_M, KK * bz + k * block_K], A_shared)
                    T.copy(B[bx * block_N, (KK * bz + k * block_K) // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_f4_to_f16(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, SplitC[bz, bx * block_N:(bx + 1) * block_N,
                                        by * block_M:(by + 1) * block_M])
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
                acc = T.alloc_fragment((block_N, block_M), out_dtype)
                T.clear(acc)
                for k in range(split):
                    for i, j in T.Parallel(block_N, block_M):
                        acc[i, j] += SplitC[k, bx * block_N + i, by * block_M + j]
                T.copy(acc, Ct[bx * block_N, by * block_M])

        @T.prim_func
        def main(
                A: T.Tensor(A_shape, in_dtype),
                B: T.Tensor(B_shape, storage_dtype),
                Ct: T.Tensor((N, M), out_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(A_shared_shape, in_dtype)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
                B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
                B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                B_dequantize_prev_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
                Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
                Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

                T.annotate_layout({
                    B_shared: tilelang.layout.make_swizzled_layout(B_shared),
                    Ct_shared: tilelang.layout.make_swizzled_layout(Ct_shared),
                })

                T.clear(Ct_local)
                for k in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_dequantize_local[i, j] = _tir_u8_to_f4_to_f16(
                            num_bits,
                            B_local[i, j // num_elems_per_byte],
                            j % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    T.copy(B_dequantize_local, B_dequantize_prev_local)
                    T.gemm(B_dequantize_prev_local, A_shared, Ct_local, transpose_B=True)
                T.copy(Ct_local, Ct_shared)
                T.copy(Ct_shared, Ct[bx * block_N:(bx + 1) * block_N,
                                     by * block_M:(by + 1) * block_M])

        if split == 1:
            return main
        else:
            return main_split

    if tune:

        @autotune(
            configs=get_configs(),
            keys=["block_M", "block_N", "block_K", "num_stages", "threads", "split"],
            warmup=10,
            rep=10)
        @jit(out_idx=[2], supply_type=tilelang.TensorSupplyType.Integer, ref_prog=None)
        def kernel(block_M=None,
                   block_N=None,
                   block_K=None,
                   num_stages=None,
                   threads=None,
                   split=None):
            return kernel_func(block_M, block_N, block_K, num_stages, threads, split)

        return kernel()
    else:

        def kernel(block_M, block_N, block_K, num_stages, threads, split=1):
            return kernel_func(block_M, block_N, block_K, num_stages, threads, split)

        return kernel


def ref_program(A, qB):
    dtypeC = "float16"
    B = torch_convert(qB)
    C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
    C = C.to(torch.__getattribute__(dtypeC))
    return C.transpose(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=256, help='M')
    parser.add_argument('--n', type=int, default=256, help='N')
    parser.add_argument('--k', type=int, default=256, help='K')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    total_flops = 2 * M * N * K

    if (not args.tune):
        program = matmul(
            M, N, K, "float16", "float16", "float32", num_bits=4, tune=args.tune)(
                block_M=128, block_N=128, block_K=128, num_stages=2, threads=256, split=1)
        kernel = tilelang.compile(program, out_idx=[2])
        profiler = kernel.get_profiler(tilelang.TensorSupplyType.Integer)
        profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(ref_program, warmup=500)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(warmup=500)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = matmul(M, N, K, "float16", "float16", "float32", num_bits=4, tune=args.tune)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
