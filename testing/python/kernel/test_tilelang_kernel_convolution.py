# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T


def convolution(N, C, H, W, F, K, S, D, P, in_dtype, out_dtype, dtypeAccum, block_M, block_N,
                block_K, num_stages, threads):
    KH, KW = K, K
    OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
    OW = (W + 2 * P - D * (K - 1) - 1) // S + 1

    @T.prim_func
    def main(
            data: T.Buffer((N, H, W, C), in_dtype),
            kernel: T.Buffer((KH, KW, C, F), in_dtype),
            out: T.Buffer((N, OH, OW, F), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M),
                threads=threads) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), in_dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), in_dtype)
            out_local = T.alloc_fragment((block_M, block_N), dtypeAccum)

            kernel_flat = T.Buffer((KH * KW * C, F), in_dtype, kernel.data)
            out_flat = T.Buffer((N * OH * OW, F), out_dtype, out.data)

            T.clear(out_local)
            for k_iter in T.Pipelined(T.ceildiv(KH * KW * C, block_K), num_stages=num_stages):
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    access_h = m % (OH * OW) // OW * S + k // (KW * C) * D - P
                    access_w = m % OW * S + k // C % KW * D - P
                    in_bound = ((access_h >= 0) and (access_w >= 0) and (access_h < H) and
                                (access_w < W))
                    data_shared[i,
                                j] = T.if_then_else(in_bound, data[m // (OH * OW), access_h,
                                                                   access_w, k % C], 0)
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                T.gemm(data_shared, kernel_shared, out_local)

            T.copy(out_local, out_flat[by * block_M, bx * block_N])

    return main


def run_conv(N,
             C,
             H,
             W,
             F,
             K,
             S,
             D,
             P,
             in_dtype,
             out_dtype,
             dtypeAccum,
             block_M,
             block_N,
             block_K,
             num_stages=2,
             threads=128):
    program = convolution(N, C, H, W, F, K, S, D, P, in_dtype, out_dtype, dtypeAccum, block_M,
                          block_N, block_K, num_stages, threads)

    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

    def ref_program(A, B):
        import torch

        A = A.permute(0, 3, 1, 2).to(torch.float)  # N, H, W, C -> N, C, H, W
        B = B.permute(3, 2, 0, 1).to(torch.float)  # H, W, C, F -> F, C, H, W
        C = torch.conv2d(A, B, stride=S, padding=P, dilation=D)
        C = C.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        return C.to(torch.__getattribute__(out_dtype))

    mod.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_conv_f16f16f16_k3s1d1p1():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        3,
        1,
        1,
        1,
        "float16",
        "float16",
        "float16",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f16_k3s2d1p1():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        3,
        2,
        1,
        1,
        "float16",
        "float16",
        "float16",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f16_k1s1d1p0():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        1,
        1,
        1,
        0,
        "float16",
        "float16",
        "float16",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f16_k1s2d1p0():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        1,
        2,
        1,
        0,
        "float16",
        "float16",
        "float16",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f32_k3s1d1p1():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        3,
        1,
        1,
        1,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f32_k3s2d1p1():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        3,
        2,
        1,
        1,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f32_k1s1d1p0():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        1,
        1,
        1,
        0,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_f16f16f32_k1s2d1p0():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        1,
        2,
        1,
        0,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_bf16bf16f32_k3s1d1p1():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        3,
        1,
        1,
        1,
        "bfloat16",
        "bfloat16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_bf16bf16f32_k3s2d1p1():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        3,
        2,
        1,
        1,
        "bfloat16",
        "bfloat16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_bf16bf16f32_k1s1d1p0():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        1,
        1,
        1,
        0,
        "bfloat16",
        "bfloat16",
        "float32",
        128,
        128,
        32,
        2,
    )


def test_conv_bf16bf16f32_k1s2d1p0():
    run_conv(
        1,
        128,
        64,
        64,
        128,
        1,
        2,
        1,
        0,
        "bfloat16",
        "bfloat16",
        "float32",
        128,
        128,
        32,
        2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
