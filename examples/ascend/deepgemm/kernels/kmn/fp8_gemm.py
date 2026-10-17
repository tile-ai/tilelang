"""KMN MXFP8 GEMM: dense, batched, M-grouped and K-grouped variants."""

import tilelang
import tilelang.ascend.language as T

from ...config import GemmType, Major
from ..common import COMPILE_FLAGS, MAX_EPILOGUE_STAGES, MK_ALIGNMENT, MX_SF_DIVISOR, physical_shape
from ..copy import make_gm_to_l1_copy, make_scale_copy
from ..epilogue import make_store_output
from .compute import make_compute


def _build_kernel(m, n, k, major_a, major_b, out_dtype, accumulate, config, gemm_type, with_alpha=False):
    shape_m = m if m is not None else T.dynamic("shape_m")
    shape_n = n if n is not None else T.dynamic("shape_n")
    shape_k = k if k is not None else T.dynamic("shape_k")
    BLOCK_M, BLOCK_N, BLOCK_K = config.block_m, config.block_n, config.block_k
    MAD_M, MAD_N, MAD_K = config.mad_m, config.mad_n, config.mad_k
    ub_dtype = "float32" if with_alpha else out_dtype
    output_m = MAD_M * (2 if with_alpha and out_dtype == "bfloat16" else 1)
    num_mad_m_tiles = BLOCK_M // MAD_M
    num_mad_n_tiles = BLOCK_N // MAD_N

    shape_sf_k = T.ceildiv(shape_k, MX_SF_DIVISOR)
    sf_pairs_per_chunk = config.sf_k_blocks * (BLOCK_K // MX_SF_DIVISOR)

    copy_a = make_gm_to_l1_copy(major_a, config.l2_ctrl_a)
    copy_b = make_gm_to_l1_copy(major_b, config.l2_ctrl_b)
    copy_sfa = make_scale_copy(config.l2_ctrl_a)
    copy_sfb = make_scale_copy(config.l2_ctrl_b)
    compute_k_block = make_compute(config, major_a, major_b, blockscaled=True)
    store_output = make_store_output(config, major_a, major_b, blockscaled=True, with_alpha=with_alpha)

    batched = gemm_type == GemmType.Batched
    m_grouped = gemm_type == GemmType.MGroupedContiguousWithPsumLayout
    k_grouped = gemm_type == GemmType.KGroupedContiguousWithPsumLayout
    num_groups = T.dynamic("num_groups") if gemm_type != GemmType.Normal else None

    def tensor(name, shape, dtype, grouped=False):
        strides = (T.dynamic(f"{name}_outer_stride", dtype="int64"), 1)
        if grouped:
            group_stride = "batch" if batched else "group"
            shape = (num_groups, *shape)
            strides = (T.dynamic(f"{name}_{group_stride}_stride", dtype="int64"), *strides)
        return T.StridedTensor(shape, strides, dtype)

    a_type = tensor("a", physical_shape(shape_m, shape_k, major_a), "float8_e4m3fn", batched)
    b_type = tensor("b", physical_shape(shape_n, shape_k, major_b), "float8_e4m3fn", batched or m_grouped)
    sfa_type = tensor("sfa", (shape_sf_k, shape_m), "int16", batched)
    sfb_type = tensor("sfb", (shape_sf_k, shape_n), "int16", batched or m_grouped)
    d_type = tensor("d", (shape_m, shape_n), out_dtype, batched or k_grouped)
    scheduler_type = {
        GemmType.Normal: T.AscendTileScheduler,
        GemmType.Batched: T.AscendBatchedTileScheduler,
        GemmType.MGroupedContiguousWithPsumLayout: T.AscendMGroupedTileScheduler,
        GemmType.KGroupedContiguousWithPsumLayout: T.AscendKGroupedTileScheduler,
    }[gemm_type]
    scheduler_kwargs = dict(block_m=BLOCK_M, block_n=BLOCK_N, num_cores=config.num_blocks, shape_m=shape_m, shape_n=shape_n)
    if num_groups is not None:
        scheduler_kwargs["num_groups"] = num_groups
    if k_grouped:
        scheduler_kwargs["shape_k"] = shape_k
    if m_grouped or k_grouped:
        scheduler_kwargs["alignment"] = MK_ALIGNMENT

    @T.macro
    def body(gm_a, gm_b, gm_sfa, gm_sfb, gm_d, grouped_layout=None, alpha=1.0):
        with T.MixedKernel(config.num_blocks, sids=1) as (block_idx, _sub_id):
            if m_grouped or k_grouped:
                scheduler = scheduler_type(**scheduler_kwargs, grouped_layout=grouped_layout)
            else:
                scheduler = scheduler_type(**scheduler_kwargs)
            l1a = T.alloc_l1(physical_shape(BLOCK_M, BLOCK_K, major_a), "float8_e4m3fn")
            l1b = T.alloc_l1(physical_shape(BLOCK_N, BLOCK_K, major_b), "float8_e4m3fn")
            l1_sfa = T.alloc_l1((BLOCK_M, sf_pairs_per_chunk), "int16")
            l1_sfb = T.alloc_l1((BLOCK_N, sf_pairs_per_chunk), "int16")
            l0a = T.alloc_l0a((num_mad_m_tiles, MAD_M, MAD_K), "float8_e4m3fn")
            l0b = T.alloc_l0b((num_mad_n_tiles, MAD_N, MAD_K), "float8_e4m3fn")
            l0_sfa = T.alloc_l0a_sf(l0a, sf_dtype="int16", sf_shape=(num_mad_m_tiles, MAD_M, MAD_K // MX_SF_DIVISOR))
            l0_sfb = T.alloc_l0b_sf(l0b, sf_dtype="int16", sf_shape=(num_mad_n_tiles, MAD_N, MAD_K // MX_SF_DIVISOR))
            l0c = T.alloc_l0c((num_mad_m_tiles, num_mad_n_tiles, MAD_M, MAD_N), "float32")
            epilogue_ub = T.alloc_shared((MAD_M, MAD_N), ub_dtype)
            if gemm_type == GemmType.Normal:
                output_ub = T.view(epilogue_ub, (output_m, MAD_N), dtype=out_dtype)
            T.annotate_buffer_versions(
                {
                    l1a: config.num_l1_stages,
                    l1b: config.num_l1_stages,
                    l1_sfa: config.num_l1_sf_stages,
                    l1_sfb: config.num_l1_sf_stages,
                    l0a: max(config.num_l0_stages // num_mad_m_tiles, 1),
                    l0b: max(config.num_l0_stages // num_mad_n_tiles, 1),
                    epilogue_ub: min(config.num_epilogue_stages, MAX_EPILOGUE_STAGES),
                }
            )

            T.set_mmad_direction("n")
            if accumulate:
                T.set_atomic("add", out_dtype)
            scheduler.init(block_idx)
            while scheduler.valid():
                if batched:
                    batch_idx = scheduler.batch()
                else:
                    batch_idx = 0
                if m_grouped:
                    group_idx = scheduler.group()
                elif k_grouped:
                    group_idx, k_idx_base, eff_shape_k, sf_idx_base = scheduler.group()
                else:
                    group_idx = 0
                num_k_blocks = T.ceildiv(eff_shape_k if k_grouped else shape_k, BLOCK_K)
                num_sf_chunks = T.ceildiv(num_k_blocks, config.sf_k_blocks)
                m_idx, n_idx, actual_m, actual_n = scheduler.tile()
                for sf_chunk_idx in range(num_sf_chunks):
                    sf_pair_idx = (sf_idx_base if k_grouped else 0) + sf_chunk_idx * sf_pairs_per_chunk
                    chunk_k_idx = sf_chunk_idx * config.sf_k_blocks * BLOCK_K
                    chunk_k_end = T.min(chunk_k_idx + config.sf_k_blocks * BLOCK_K, eff_shape_k if k_grouped else shape_k)
                    chunk_sf_pairs = T.ceildiv(chunk_k_end - chunk_k_idx, MX_SF_DIVISOR)
                    with T.Task():
                        copy_sfa(gm_sfa, l1_sfa, m_idx, sf_pair_idx, actual_m, chunk_sf_pairs, batch_idx)
                        copy_sfb(gm_sfb, l1_sfb, n_idx, sf_pair_idx, actual_n, chunk_sf_pairs, group_idx if m_grouped else batch_idx)
                    for k_within_chunk in range(T.ceildiv(chunk_k_end - chunk_k_idx, BLOCK_K)):
                        k_block_idx = sf_chunk_idx * config.sf_k_blocks + k_within_chunk
                        if k_grouped:
                            local_k_idx = k_block_idx * BLOCK_K
                            k_idx = k_idx_base + local_k_idx
                            actual_k = T.min(eff_shape_k - local_k_idx, BLOCK_K)
                        else:
                            k_idx = k_block_idx * BLOCK_K
                            actual_k = T.min(shape_k - k_idx, BLOCK_K)
                        with T.Task():
                            copy_a(gm_a, l1a, m_idx, k_idx, actual_m, actual_k, batch_idx)
                            copy_b(gm_b, l1b, n_idx, k_idx, actual_n, actual_k, group_idx if m_grouped else batch_idx)
                        compute_k_block(
                            l1a,
                            l1b,
                            l0a,
                            l0b,
                            l0c,
                            k_block_idx,
                            num_k_blocks,
                            actual_m,
                            actual_n,
                            actual_k,
                            l1_sfa=l1_sfa,
                            l1_sfb=l1_sfb,
                            l0_sfa=l0_sfa,
                            l0_sfb=l0_sfb,
                            k_within_chunk=k_within_chunk,
                        )
                store_output(
                    l0c,
                    epilogue_ub,
                    output_ub if gemm_type == GemmType.Normal else epilogue_ub,
                    gm_d,
                    m_idx,
                    n_idx,
                    actual_m,
                    actual_n,
                    batch_idx=group_idx if k_grouped else batch_idx,
                    alpha=alpha,
                )
                scheduler.next_block()
            if accumulate:
                T.set_atomic_none()

    if gemm_type == GemmType.Normal:

        @T.prim_func
        def kernel(gm_a: a_type, gm_b: b_type, gm_sfa: sfa_type, gm_sfb: sfb_type, gm_d: d_type, alpha: T.float32):
            body(gm_a, gm_b, gm_sfa, gm_sfb, gm_d, alpha=alpha)

    elif batched:

        @T.prim_func
        def kernel(gm_a: a_type, gm_b: b_type, gm_sfa: sfa_type, gm_sfb: sfb_type, gm_d: d_type):
            body(gm_a, gm_b, gm_sfa, gm_sfb, gm_d)

    else:

        @T.prim_func
        def kernel(
            gm_a: a_type, gm_b: b_type, gm_sfa: sfa_type, gm_sfb: sfb_type, gm_d: d_type, grouped_layout: T.Buffer((num_groups,), "int32")
        ):
            body(gm_a, gm_b, gm_sfa, gm_sfb, gm_d, grouped_layout=grouped_layout)

    return kernel


@tilelang.jit(compile_flags=COMPILE_FLAGS)
def build_fp8_gemm(m, n, k, major_a, major_b, out_dtype, accumulate, config, with_alpha=False):
    return _build_kernel(m, n, k, major_a, major_b, out_dtype, accumulate, config, GemmType.Normal, with_alpha)


@tilelang.jit(compile_flags=COMPILE_FLAGS)
def build_fp8_batched_gemm(m, n, k, major_a, major_b, out_dtype, accumulate, config):
    return _build_kernel(m, n, k, major_a, major_b, out_dtype, accumulate, config, GemmType.Batched)


@tilelang.jit(compile_flags=COMPILE_FLAGS)
def build_fp8_m_grouped_gemm(total_m, n, k, major_b, out_dtype, config):
    return _build_kernel(total_m, n, k, Major.K, major_b, out_dtype, False, config, GemmType.MGroupedContiguousWithPsumLayout)


@tilelang.jit(compile_flags=COMPILE_FLAGS)
def build_fp8_k_grouped_gemm(m, n, total_k, out_dtype, config):
    return _build_kernel(m, n, total_k, Major.MN, Major.MN, out_dtype, True, config, GemmType.KGroupedContiguousWithPsumLayout)
