import re

import tilelang
import tilelang.testing
from tilelang import language as T
import torch

from examples.gemm_sp.sparse_utils import get_e_factor
from tvm_ffi import structural_equal
from tvm_tirx import ir, sym
from tilelang._compat.stmt_functor import post_order_visit


def _nodes(root, node_type):
    result = []
    post_order_visit(root, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


def _calls(root, name):
    return [call for call in _nodes(root, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == name]


def _roles(kernel):
    groups = [region for region in _nodes(kernel.artifact.device_mod, ir.RegionStmt) if region.attrs.get("tl.ws_concurrent")]
    if not groups:
        return []
    assert len(groups) == 1
    roles = []
    for statement in groups[0].body.seq:
        if isinstance(statement, ir.Bind):
            assert isinstance(statement.value, ir.Call) and statement.value.op.name == "tirx.alloc_tensor"
            continue
        if isinstance(statement, ir.Evaluate):
            assert isinstance(statement.value, ir.Call) and statement.value.op.name == "tl.fence_proxy_async"
            continue
        assert isinstance(statement, ir.If) and statement.else_case is None
        for condition, bodies in roles:
            if structural_equal(condition, statement.condition):
                bodies.append(statement.then_case)
                break
        else:
            roles.append((statement.condition, [statement.then_case]))
    return [ir.SeqStmt(bodies) for _, bodies in roles]


def _launch_threads(kernel):
    return {
        int(region.args[1])
        for region in _nodes(kernel.artifact.device_mod, ir.RegionStmt)
        if region.op.name == "tirx.launch_thread" and str(region.args[0].value) == "threadIdx.x"
    }


def _assert_async_copy(source):
    assert re.search(r"^\s+ptx_cp_async_legacy_\w*16_\w*\(", source, re.MULTILINE)


def _assert_barrier_protocol(kernel):
    module = kernel.artifact.device_mod
    initializes = _calls(module, "tl.mbarrier_init")
    assert initializes
    initialized = {(call.args[0].source, int(call.args[0].indices[0])) for call in initializes}
    assert len(initialized) == len(initializes)
    assert all(int(call.args[1]) > 0 for call in initializes)
    analyzer = sym.Analyzer()
    barriers = {tensor for tensor, _ in initialized}
    for name in ("tl.mbarrier_wait_parity", "tl.mbarrier_arrive", "tl.ptx_cp_async_barrier_noinc"):
        for call in _calls(module, name):
            access = call.args[0]
            assert isinstance(access, ir.TensorLoad) and access.source in barriers
            assert analyzer.can_prove(access.indices[0] >= 0)
            assert analyzer.can_prove(access.indices[0] < access.source.ty.shape[0])
    for call in _calls(module, "tl.tma_load"):
        assert isinstance(call.args[1], ir.TensorLoad) and call.args[1].source in barriers


def _stores(root, prefix):
    return [store for store in _nodes(root, ir.TensorStore) if store.dest.name.startswith(prefix)]


def _zero(value):
    if isinstance(value, ir.prim.Broadcast):
        return _zero(value.value)
    return isinstance(value, (ir.IntImm, ir.FloatImm)) and value.value == 0


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
def test_num_stages_zero_pure_tma_does_not_auto_warp_specialize():
    """num_stages=0 should keep ordinary T.copy on the synchronous path."""

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

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(copy_loop_num_stages_zero, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" not in src
    assert not _roles(kernel)

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

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(copy_loop_num_stages_one, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert len(_roles(kernel)) >= 2
    _assert_barrier_protocol(kernel)

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    y = kernel(x)
    torch.testing.assert_close(y, x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_guarded_pure_tma_pipeline_keeps_auto_warp_specialize():
    """A pipeline loop nested under a runtime guard should still auto-WS."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def guarded_copy_loop(
        x: T.Tensor((M, K), T.float16),
        y: T.Tensor((M, K), T.float16),
        valid_m: T.int32,
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            if pid_m * block_m < valid_m:
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

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(guarded_copy_loop, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert len(_roles(kernel)) >= 2
    _assert_barrier_protocol(kernel)

    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    y = kernel(x, M)
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

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(cp_async_only_num_stages_zero, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    _assert_async_copy(src)
    assert _launch_threads(kernel) == {threads}
    assert not _roles(kernel)

    x = torch.randint(0, 256, (4 * bytes_per_copy,), device="cuda", dtype=torch.uint8)
    y = kernel(x)
    torch.testing.assert_close(y, x)
    torch.cuda.synchronize()


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_num_stages_one_cp_async_only_keeps_non_ws_launch_shape():
    """Stage-1 cp.async-only loops should stay non-WS on Hopper."""

    bytes_per_copy = 16
    threads = 32

    @T.prim_func
    def cp_async_only_num_stages_one(
        x: T.Tensor((4 * bytes_per_copy,), T.uint8),
        y: T.Tensor((4 * bytes_per_copy,), T.uint8),
    ):
        with T.Kernel(1, threads=threads):
            x_shared = T.alloc_shared((bytes_per_copy,), dtype=T.uint8)
            for ko in T.Pipelined(4, num_stages=1):
                T.ptx_cp_async(
                    T.access_ptr(x_shared[0], "w", bytes_per_copy),
                    T.access_ptr(x[ko * bytes_per_copy], "r", bytes_per_copy),
                    bytes_per_copy,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                for i in T.serial(bytes_per_copy):
                    y[ko * bytes_per_copy + i] = x_shared[i]

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(cp_async_only_num_stages_one, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    _assert_async_copy(src)
    assert _launch_threads(kernel) == {threads}
    assert not _roles(kernel)

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

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(mixed_async_num_stages_one, pass_configs, out_idx=[2, 3])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    _assert_async_copy(src)
    roles = _roles(kernel)
    producers = [role for role in roles if _calls(role, "tl.tma_load")]
    assert len(producers) == 1
    assert _calls(producers[0], "tl.ptx_cp_async")
    assert all(not _calls(role, "tl.ptx_cp_async") for role in roles if not role.same_as(producers[0]))
    _assert_barrier_protocol(kernel)
    source = torch.randn((M, K), dtype=torch.float16, device="cuda")
    metadata = torch.randint(0, 256, (2 * cp_async_bytes,), dtype=torch.uint8, device="cuda")
    actual, actual_metadata = kernel(source, metadata)
    torch.testing.assert_close(actual, source, rtol=0, atol=0)
    torch.testing.assert_close(actual_metadata, metadata, rtol=0, atol=0)


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
                T.copy(B[ko * block_k : (ko + 1) * block_k, bx * block_n : (bx + 1) * block_n], B_shared, prefer_instruction="cp_async")
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(mixed_gemm_shared_barrier, pass_configs, out_idx=[2])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    _assert_async_copy(src)
    assert "arrive_and_expect_tx" not in src
    module = kernel.artifact.device_mod
    assert any(int(call.args[1]) == 8192 for call in _calls(module, "tl.mbarrier_expect_tx"))
    assert all(int(call.args[1]) == 128 for call in _calls(module, "tl.mbarrier_init"))
    assert _calls(module, "tl.ptx_cp_async_barrier_noinc")
    _assert_barrier_protocol(kernel)

    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    ref = a @ b
    c = kernel(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda_compute_version_eq(9, 0)
def test_sparse_ws_regular_metadata_copy_stays_in_producer():
    """Ordinary global->shared metadata copies should stay in the producer."""

    M, N, K = 128, 128, 256
    block_m = block_n = 128
    block_k = 128
    num_stages = 2
    threads = 128

    e_factor = get_e_factor(T.float16, T.uint8)

    @T.prim_func
    def sparse_tensorcore_metadata_copy(
        A_sparse: T.Tensor((M, K // 2), T.float16),
        E: T.Tensor((M, K // e_factor), T.uint8),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k // 2), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            E_shared = T.alloc_shared((block_m, block_k // e_factor), T.uint8)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(E[by * block_m, k * block_k // e_factor], E_shared)
                T.copy(A_sparse[by * block_m, k * block_k // 2], A_shared)
                T.copy(B[k * block_k, bx * block_n], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_local, transpose_A=False, transpose_E=False, transpose_B=False)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(sparse_tensorcore_metadata_copy, pass_configs, out_idx=[3])

    src = kernel.get_kernel_source()
    producer_idx = src.index("if (((int)threadIdx.x) < 128)")
    consumer_idx = src.index("} else {", producer_idx)
    metadata_copy_idx = src.index("tl::tma_load(E_desc")

    assert producer_idx < metadata_copy_idx < consumer_idx
    assert "tl::tma_load(E_desc" not in src[consumer_idx:]


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

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(sparse_flash_attn, pass_configs, out_idx=[4])

    roles = _roles(kernel)
    producers = [role for role in roles if _calls(role, "tl.tma_load")]
    consumers = [role for role in roles if _stores(role, "acc_o")]
    assert len(producers) == len(consumers) == 1
    producer, consumer = producers[0], consumers[0]
    for role in (producer, consumer):
        assert _stores(role, "block_mask")
        assert any(load.source.name.startswith("BlockSparseMask") for load in _nodes(role, ir.TensorLoad))
    for prefix in ("acc_o", "logsum", "scores_max"):
        assert not _stores(producer, prefix)
    for prefix in ("acc_o", "logsum"):
        initializes = [store for store in _stores(kernel.artifact.device_mod, prefix) if _zero(store.value)]
        assert initializes
        assert all(store in _stores(consumer, prefix) for store in initializes)
    assert _calls(consumer, "tl.mbarrier_wait_parity")
    assert _calls(consumer, "tl.mbarrier_arrive")
    calls = _nodes(consumer, ir.Call)
    handoffs = [
        index
        for index, call in enumerate(calls)
        if call.op.name == "tl.mbarrier_arrive" and call.args[0].source.name.startswith("ws_barrier_O_shared")
    ]
    assert len(handoffs) == 1
    assert calls[handoffs[0] - 1].op.name == "tl.fence_proxy_async"
    _assert_barrier_protocol(kernel)
    mask = torch.ones(block_mask_shape, dtype=torch.bool, device="cuda")
    for _ in range(3):
        query, key, value = [torch.randn(shape, dtype=torch.float16, device="cuda") for _ in range(3)]
        actual = kernel(query, key, value, mask)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_pure_tma_fragment_mask_copy_initializes_consumer_branch():
    """A pre-loop fragment mask copy must be duplicated into both WS branches."""

    batch = heads = 1
    seq_q = block_m = block_n = dim = 64
    downsample_len = 4
    seq_kv = downsample_len * block_n
    threads = 128

    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    block_mask_shape = [batch, heads, 1, downsample_len]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def sparse_attention_mask_copy(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        BlockSparseMask: T.Tensor(block_mask_shape, "int8"),
        Output: T.Tensor((batch, heads, seq_q, block_n), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_m), heads, batch, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([block_m, dim], dtype)
            K_shared = T.alloc_shared([block_n, dim], dtype)
            acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
            block_mask = T.alloc_fragment([downsample_len], "int8")

            T.copy(Q[bz, by, bx * block_m : (bx + 1) * block_m, :], Q_shared)
            T.copy(BlockSparseMask[bz, by, bx, :], block_mask)

            for k in T.Pipelined(downsample_len, num_stages=1):
                if block_mask[k] != 0:
                    T.copy(K[bz, by, k * block_n : (k + 1) * block_n, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_s, Output[bz, by, bx * block_m : (bx + 1) * block_m, :])

    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}
    kernel = _compile_tvm_ffi(sparse_attention_mask_copy, pass_configs, out_idx=[3])

    roles = _roles(kernel)
    producers = [role for role in roles if _calls(role, "tl.tma_load")]
    consumers = [role for role in roles if _stores(role, "acc_s")]
    assert len(producers) == len(consumers) == 1
    for role in (producers[0], consumers[0]):
        assert _stores(role, "block_mask")
        assert any(load.source.name.startswith("BlockSparseMask") for load in _nodes(role, ir.TensorLoad))
    _assert_barrier_protocol(kernel)
    query = torch.randn(q_shape, dtype=torch.float16, device="cuda")
    key = torch.randn(kv_shape, dtype=torch.float16, device="cuda")
    mask = torch.ones(block_mask_shape, dtype=torch.int8, device="cuda")
    actual = kernel(query, key, mask)
    expected = query.float() @ key[..., -block_n:, :].float().transpose(-1, -2)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
