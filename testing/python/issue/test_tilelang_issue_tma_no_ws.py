import re

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.layout import make_cutlass_metadata_layout
import torch


def _compile_tvm_ffi(func, pass_configs, **kwargs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="cuda",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs,
            **kwargs,
        )
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_lower_no_warp_specialized_injects_mbarrier():
    """Regression for Hopper TMA lowering when warp specialization is disabled.

    When `tl.disable_tma_lower=False` but `tl.disable_warp_specialized=True`, the
    optimization pipeline must still run the TMA barrier allocation/injection
    passes so generated CUDA source defines and uses `mbarrier[...]` correctly.
    """

    M, K = 16, 128
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=threads) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            T.fill(x_shared, 0)
            T.copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
            )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    kernel = _compile_tvm_ffi(tma_copy, pass_configs)

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "mbarrier_mem" in src
    assert "arrive_and_expect_tx" in src
    assert "expect_transaction" not in src
    assert ".arrive();" not in src

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    kernel(x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_lower_no_warp_specialized_2d_descriptor_uses_args1_barrier():
    """Cover the 2D-descriptor TMA barrier rewrite path (barrier at args[1])."""

    M, K = 16, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy_2d_desc(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=threads) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            T.fill(x_shared, 0)
            T.copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
            )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

    kernel = _compile_tvm_ffi(tma_copy_2d_desc, pass_configs)

    src = kernel.get_kernel_source()
    assert "CUtensorMap" in src
    assert "tl::tma_load" in src

    flat_src = " ".join(src.split())
    pattern = r"tl::tma_load\([^,]+,\s*mbarrier\[[0-9]+\]"
    assert re.search(pattern, flat_src), (
        f"Expected regex {pattern!r} to match flattened CUDA source. Generated source (truncated):\n{src[:1000]}"
    )

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    kernel(x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_num_stages_zero_pure_tma_does_not_auto_warp_specialize():
    """num_stages=0 should keep pure TMA loops out of auto-WS."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def copy_loop_num_stages_zero(
        x: T.Tensor((M, K), T.float16),
        y: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=0):
                T.copy(
                    x[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    x_shared,
                )
                T.copy(
                    x_shared,
                    y[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(copy_loop_num_stages_zero, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "__launch_bounds__(160, 1)" not in src
    assert "if (32 <= ((int)threadIdx.x))" not in src

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    y = kernel(x)
    torch.testing.assert_close(y, x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_num_stages_one_pure_tma_keeps_auto_warp_specialize():
    """Pure TMA loops should auto-WS when num_stages is explicitly enabled."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def copy_loop_num_stages_one(
        x: T.Tensor((M, K), T.float16),
        y: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=1):
                T.copy(
                    x[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    x_shared,
                )
                T.copy(
                    x_shared,
                    y[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(copy_loop_num_stages_one, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "__launch_bounds__(160, 1)" in src
    assert "if (32 <= ((int)threadIdx.x))" in src

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    y = kernel(x)
    torch.testing.assert_close(y, x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_num_stages_zero_cp_async_only_does_not_auto_warp_specialize():
    """num_stages=0 should keep cp.async-only loops out of auto-WS."""

    bytes_per_copy = 16
    threads = 32

    @T.prim_func
    def cp_async_only_num_stages_zero(
        x: T.Tensor((4 * bytes_per_copy,), T.uint8),
        y: T.Tensor((4 * bytes_per_copy,), T.uint8),
    ):
        with T.Kernel(1, threads=threads):
            x_shared = T.alloc_shared((bytes_per_copy,), dtype=T.uint8)
            for ko in T.Pipelined(4, num_stages=0):
                T.ptx_cp_async(
                    T.access_ptr(x_shared[0], "w", bytes_per_copy),
                    T.access_ptr(x[ko * bytes_per_copy], "r", bytes_per_copy),
                    bytes_per_copy,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                for i in T.serial(bytes_per_copy):
                    y[ko * bytes_per_copy + i] = x_shared[i]

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(cp_async_only_num_stages_zero, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "cp_async_gs<16>" in src
    assert "__launch_bounds__(32, 1)" in src
    assert "__launch_bounds__(160, 1)" not in src
    assert "if (32 <= ((int)threadIdx.x))" not in src

    x = torch.randint(0, 256, (4 * bytes_per_copy,), device="cuda", dtype=torch.uint8)
    y = kernel(x)
    torch.testing.assert_close(y, x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_num_stages_one_mixed_tma_cp_async_keeps_auto_ws():
    """Mixed TMA+cp.async loops should auto-WS when num_stages is enabled."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 128
    cp_async_bytes = 16

    @T.prim_func
    def mixed_async_num_stages_one(
        x: T.Tensor((M, K), T.float16),
        meta: T.Tensor((2 * cp_async_bytes,), T.uint8),
        y: T.Tensor((M, K), T.float16),
        meta_out: T.Tensor((2 * cp_async_bytes,), T.uint8),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            meta_shared = T.alloc_shared((cp_async_bytes,), dtype=T.uint8)

            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=1):
                T.copy(
                    x[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    x_shared,
                )
                T.ptx_cp_async(
                    T.access_ptr(meta_shared[0], "w", cp_async_bytes),
                    T.access_ptr(meta[ko * cp_async_bytes], "r", cp_async_bytes),
                    cp_async_bytes,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                T.copy(
                    x_shared,
                    y[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                )
                for i in T.serial(cp_async_bytes):
                    meta_out[ko * cp_async_bytes + i] = meta_shared[i]

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(mixed_async_num_stages_one, pass_configs, out_idx=[2, 3])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    cp_async_idx = src.index("cp_async_gs<16>")

    assert producer_idx < cp_async_idx < consumer_idx
    assert "cp_async_gs<16>" not in src[consumer_idx:]


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_mixed_tma_cp_async_shared_stage_barriers():
    """Mixed TMA+cp.async groups should share one forward/bp barrier set."""

    M = N = K = 256
    block_m = block_n = 128
    block_k = 32
    num_stages = 3
    threads = 128

    @T.prim_func
    def mixed_gemm_shared_barrier(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                for k, j in T.Parallel(block_k, block_n):
                    B_shared[k, j] = B[ko * block_k + k, bx * block_n + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(mixed_gemm_shared_barrier, pass_configs, out_idx=[2])

    src = kernel.get_kernel_source()
    flat_src = " ".join(src.split())

    assert "tl::tma_load" in src
    assert "cp_async_gs<16>" in src
    assert "uint64_t mbarrier_mem[6]" in src
    assert "arrive_and_expect_tx" not in src
    assert ".expect_transaction(8192);" in src
    assert src.count(".init(128);") == 6
    assert ".init(1);" not in src
    assert "tl::mbarrier_cp_async_arrive_noinc(mbarrier[(ko % 3)])" in flat_src
    assert "tl::mbarrier_cp_async_arrive_noinc(mbarrier[((ko % 3) + 4)])" not in flat_src
    assert "mbarrier[((ko % 3) + 7)]" not in flat_src
    assert "mbarrier[((ko % 3) + 10)]" not in flat_src

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    ref = a @ b
    c = kernel(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_sparse_ws_regular_metadata_copy_stays_in_producer():
    """Ordinary global->shared metadata copies should stay in the producer."""

    M, N, K = 128, 128, 256
    block_m = block_n = 128
    block_k = 128
    num_stages = 2
    threads = 128

    @T.prim_func
    def sparse_tensorcore_metadata_copy(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // 8), "uint8"),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k // 2), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            E_shared = T.alloc_shared((block_m, block_k // 8), "uint8")
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.annotate_layout(
                {
                    E: make_cutlass_metadata_layout(E, mma_dtype=T.float16, arch="9.0", block_k=block_k),
                    E_shared: make_cutlass_metadata_layout(E_shared, mma_dtype=T.float16, arch="9.0", block_k=block_k),
                }
            )

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(E[by * block_m, k * block_k // 8], E_shared)
                T.copy(A_sparse[by * block_m, k * block_k // 2], A_shared)
                T.copy(B[k * block_k, bx * block_n], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, False, False)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(sparse_tensorcore_metadata_copy, pass_configs, out_idx=[3])

    src = kernel.get_kernel_source()
    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    metadata_copy_idx = src.index("*(uchar2*)(E +")
    gemm_idx = src.index("tl::gemm_sp_ss<")

    assert producer_idx < metadata_copy_idx < consumer_idx
    assert consumer_idx < gemm_idx
    assert "*(uchar2*)(E +" not in src[consumer_idx:]


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_pure_tma_consumer_local_init_does_not_leak_into_producer():
    """Consumer-only pre-loop local init should not be duplicated into producer."""

    batch = heads = 1
    seq_len = dim = 64
    block_m = block_n = 64
    downsample_len = 1
    num_stages = 1
    threads = 128
    scale = (1.0 / dim) ** 0.5 * 1.44269504

    shape = [batch, heads, seq_len, dim]
    block_mask_shape = [batch, heads, downsample_len, downsample_len]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def sparse_flash_attn(
        Q: T.Tensor(shape, dtype),
        K: T.Tensor(shape, dtype),
        V: T.Tensor(shape, dtype),
        BlockSparseMask: T.Tensor(block_mask_shape, "bool"),
        Output: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            V_shared = T.alloc_shared([block_n, dim], dtype)
            O_shared = T.alloc_shared([block_m, dim], dtype)
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
            acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_m], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
            scores_scale = T.alloc_fragment([block_m], accum_dtype)
            scores_sum = T.alloc_fragment([block_m], accum_dtype)
            logsum = T.alloc_fragment([block_m], accum_dtype)
            block_mask = T.alloc_local([downsample_len], "bool")

            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for vj in T.serial(downsample_len):
                block_mask[vj] = BlockSparseMask[bz, by, bx, vj]

            for k in T.Pipelined(downsample_len, num_stages=num_stages):
                if block_mask[k] != 0:
                    T.copy(K[bz, by, k * block_n : (k + 1) * block_n, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[bz, by, k * block_n : (k + 1) * block_n, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_m : (bx + 1) * block_m, :])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(sparse_flash_attn, pass_configs, out_idx=[4])

    src = kernel.get_kernel_source()
    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    prelude_src = src[:producer_idx]
    producer_src = src[producer_idx:consumer_idx]
    consumer_src = src[consumer_idx:]
    flat_src = " ".join(src.split())

    assert src.count(".init(1);") == 3
    assert src.count(".init(128);") == 2
    assert "tl::tma_load(K_desc, mbarrier[0]" in src
    assert "tl::tma_load(V_desc, mbarrier[1]" in src
    assert re.search(r"mbarrier\[2\]\.wait\([^;]+\);", flat_src)
    assert re.search(r"mbarrier\[3\]\.wait\([^;]+\);", flat_src)
    assert re.search(r"mbarrier\[0\]\.wait\([^;]+\);", flat_src)
    assert re.search(r"mbarrier\[1\]\.wait\([^;]+\);", flat_src)
    assert "mbarrier[2].arrive();" in consumer_src
    assert "mbarrier[3].arrive();" in consumer_src
    assert "block_mask" in producer_src
    assert "block_mask" in consumer_src
    assert "acc_o" not in producer_src
    assert "logsum" not in producer_src
    assert "scores_max" not in producer_src
    assert "*(float4*)(acc_o + (i * 4)) = make_float4" not in prelude_src
    assert "*(float4*)(acc_o + (i * 4)) = make_float4" in consumer_src
    assert "*(float2*)(logsum + 0) = make_float2" not in prelude_src
    assert "*(float2*)(logsum + 0) = make_float2" in consumer_src


if __name__ == "__main__":
    tilelang.testing.main()
