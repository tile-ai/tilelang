import torch
import torch.backends
import tilelang.testing
from tilelang import tvm as tvm
from tvm import DataType
import tilelang.language as T

tilelang.testing.set_random_seed(0)


@tilelang.jit(out_idx=[2])
def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    num_bits=4,
):
    from tilelang.quantize import _tir_packed_to_unsigned_convert

    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.int8
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    storage_type = str("".join(c for c in storage_dtype if not c.isdigit()))
    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    local_size_compressed = local_size // num_elems_per_byte

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_local([local_size_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([local_size], in_dtype)
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            tx = T.get_thread_binding()

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * local_size_compressed)):
                    for v in T.vectorized(0, local_size_compressed):
                        index = i * threads * local_size_compressed + tx * local_size_compressed + v
                        vi = index // (block_K // num_elems_per_byte)
                        vj = index % (block_K // num_elems_per_byte)
                        B_local[v] = B_shared[vi, vj]
                    for v in T.serial(0, local_size):
                        B_dequantize_local[v] = _tir_packed_to_unsigned_convert(storage_type, storage_nbit)(
                            num_bits,
                            B_local[v // num_elems_per_byte],
                            v % num_elems_per_byte,
                            dtype=in_dtype,
                        )
                    for v in T.vectorized(0, local_size):
                        index = i * threads * local_size + tx * local_size + v
                        vi = index // block_K
                        vj = index % block_K
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    kernel = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Integer)

    out = profiler.run_once()
    assert out is not None

    def ref_program(A, qB):
        import torch

        B = torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4, dtype=torch.half).to(torch.half).to(A.device)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
        C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program)


@tvm.testing.requires_package("bitblas")
def tl_matmul_with_ladder_weight_only_transform_block_reduce_int4(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    transform_b,
):
    from tilelang.intrinsics.mma_layout import make_mma_swizzle_layout as make_swizzle_layout
    from tilelang.intrinsics.mma_macro_generator import (
        TensorCoreIntrinEmitterWithLadderTransform,
    )

    from bitblas.gpu.intrin.lop3 import decode_i4_to_f16

    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.int8

    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == T.int32:
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2

    warp_rows = 4
    warp_cols = 4
    warp_row_tiles = micro_size_x * warp_rows
    warp_col_tiles = micro_size_y * warp_cols
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2
    reduce_k = 1

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = 32 if in_dtype == T.float16 else 64
    chunk = block_K // reduce_k

    is_smooth_a = False
    can_swizzle = block_K * DataType(in_dtype).bits == 512
    apply_pad_a = not (is_smooth_a or can_swizzle)
    pad_factor = 8

    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k // num_elems_per_byte)
    A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
    B_shared_shape = (
        block_N // micro_size_y,
        block_K // micro_size_k,
        micro_size_y,
        micro_size_k // num_elems_per_byte,
    )
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitterWithLadderTransform(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
        reduce_k=reduce_k,
        transform_kind_b=transform_b,
        num_elems_per_byte=num_elems_per_byte,
    )

    vec_load_qb = 16
    if block_N * (block_K // reduce_k) // num_elems_per_byte // threads < vec_load_qb:
        vec_load_qb = block_N * (block_K // reduce_k) // num_elems_per_byte // threads

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads, prelude=decode_i4_to_f16) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size // num_elems_per_byte), storage_dtype)
            B_dequantize_local = T.alloc_local((warp_cols * local_size), in_dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size), accum_dtype)
            reduced_accum_res = T.alloc_local(0, accum_dtype)
            thread_binding = T.get_thread_binding(0)
            rk = T.get_thread_binding(1)

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                }
            )

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):
                # Load A into shared memory
                for i, k in T.Parallel(block_M, (block_K // reduce_k)):
                    vk = rk * (block_K // reduce_k) + k
                    A_shared[i, vk] = A[by * block_M + i, ko * block_K + vk]

                # TODO(lei): Layout Inference Pass is not efficient to handle the four dims int8 load
                for i in T.serial(block_N * (block_K // reduce_k) // num_elems_per_byte // (threads * vec_load_qb)):
                    for v in T.vectorized(0, vec_load_qb):
                        t = thread_binding
                        idx = i * threads * vec_load_qb * reduce_k + rk * threads * vec_load_qb + t * vec_load_qb + v
                        vkk = idx % (micro_size_k // num_elems_per_byte)
                        vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                        vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (block_K // micro_size_k)
                        vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y // (block_K // micro_size_k)) % (
                            block_N // micro_size_y
                        )
                        B_shared[vj, vk, vjj, vkk] = B[bx * (block_N // micro_size_y) + vj, ko * (block_K // micro_size_k) + vk, vjj, vkk]

                for ki in T.serial(0, (block_K // (micro_size_k * reduce_k))):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                        rk=rk,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_shared,
                        ki,
                        rk=rk,
                    )

                    for j in T.serial(warp_cols):
                        local_size_b = mma_emitter.local_size_b
                        T.call_extern(
                            "handle",
                            "decode_i4u_to_f16",
                            T.address_of(B_local[j * local_size_b // num_elems_per_byte]),
                            T.address_of(B_dequantize_local[j * local_size_b]),
                            8,
                        )

                    mma_emitter.mma(A_local, B_dequantize_local, C_local)

            if reduce_k > 1:
                for n in T.serial(warp_rows * warp_cols * local_size):
                    T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    )
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_local[n],
                            True,
                            reduced_accum_res[0],
                            rk,
                            dtype="handle",
                        )
                    )
                    if rk == 0:
                        C_local[n] = reduced_accum_res[0]

            if rk == 0:
                mma_emitter.stmatrix(
                    C_local,
                    C_shared,
                )

            for i, j in T.Parallel(block_M, (block_N // reduce_k)):
                vj = rk * (block_N // reduce_k) + j
                C[by * block_M + i, bx * block_N + vj] = C_shared[
                    i // micro_size_x, vj // micro_size_y, i % micro_size_x, vj % micro_size_y
                ]

    return main


def assert_tl_matmul_with_ladder_weight_only_transform_block_reduce_int4_correctness(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    transform_b,
):
    import bitblas

    matmul = tl_matmul_with_ladder_weight_only_transform_block_reduce_int4(M, N, K, in_dtype, out_dtype, accum_dtype, transform_b)

    kernel = tilelang.compile(matmul, out_idx=[2])
    src_code = kernel.get_kernel_source()
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Integer)

    # src_code is the generated cuda source
    assert src_code is not None
    num_bits = 4
    num_elems_per_byte = 8 // num_bits
    storage_dtype = T.int8

    A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
    qB = torch.randint(0, 127, (N, K // num_elems_per_byte), device="cuda", dtype=getattr(torch, storage_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
        M=N,
        N=K,
        transform_kind=transform_b,
        transpose_matrix=True,
        dequantize_bits=num_bits,
        storage_dtype=storage_dtype,
    )

    ladder_permutate = bitblas.ops.LadderPermutate(ladder_permutate_config)

    lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
        M=N,
        N=K,
        datatype=in_dtype,
        dequantize_bits=num_bits,
        storage_dtype=storage_dtype,
    )
    lop3_permutate = bitblas.ops.LOP3Permutate(
        config=lop3_permutate_config,
        target=tvm.target.Target("llvm"),
    )
    QLB = ladder_permutate(qB.cpu()).cuda()
    QLB = lop3_permutate(QLB.cpu()).cuda()

    kernel(A, QLB, C)

    latency = profiler.do_bench(warmup=25)

    # Ensure that the latency is not None
    assert latency is not None

    B = torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4, dtype=torch.half).to(torch.half).to(A.device)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)

    # Get Reference Result
    ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
    print("Ref C: ", ref_c)
    print("C: ", C)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_package("bitblas")
def test_run_dequantize_gemm():
    run_gemm(256, 256, 256, T.float16, T.float16, T.float16, 128, 128, 32, num_threads=128)
    run_gemm(256, 256, 256, T.int8, T.int32, T.int32, 128, 128, 32, num_threads=128)


@tilelang.testing.requires_package("bitblas")
def test_assert_tl_matmul_with_ladder_weight_only_transform_block_reduce_int4():
    assert_tl_matmul_with_ladder_weight_only_transform_block_reduce_int4_correctness(256, 1024, 512, T.float16, T.float16, T.float16, 3)


def main():
    test_run_dequantize_gemm()


if __name__ == "__main__":
    main()
