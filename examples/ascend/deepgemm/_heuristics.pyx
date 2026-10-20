# cython: language_level=3, cdivision=True
"""GEMM configuration selection for DeepGEMM-Ascend kernels.

Candidate fitting, cost estimation, pruning, and deterministic tie-breaking
use C structs and integer arithmetic.
The public entry point converts Python metadata and constructs one result.
Very large load-cost products use Python integers to avoid intermediate overflow.
"""

from libc.stdint cimport int64_t
from libc.string cimport memset

cdef const int64_t L1_SIZE_BYTES = 512 * 1024
cdef const int64_t L0A_SIZE_BYTES = 64 * 1024
cdef const int64_t L0B_SIZE_BYTES = 64 * 1024
cdef const int64_t L0C_SIZE_BYTES = 256 * 1024
cdef const int64_t UB_SIZE_BYTES = 256 * 1024
cdef const int64_t FRAC_MN = 16
cdef const int64_t MK_ALIGNMENT = 256
cdef const int64_t _INT64_MAX = (1 << 63) - 1
cdef const int64_t L2_LD_NORMAL_FV = 0
cdef const int64_t L2_LD_NOTALLOC_KEEP = 4
cdef const int64_t L2_ST_NORMAL_FV = 0
cdef const int64_t L2_ST_NOTALLOC_CLEAN = 4


# Keep values in sync with the public enums in config.py.
cdef enum Major:
    K = 0
    MN = 1


cdef enum GemmType:
    Normal = 0
    MGroupedContiguousWithPsumLayout = 1
    KGroupedContiguousWithPsumLayout = 2
    Batched = 3


cdef struct GemmDesc:
    int64_t m
    int64_t n
    int64_t k
    Major major_a
    Major major_b
    bint acc
    GemmType gemm_type
    int64_t num_groups
    int64_t expected_m
    int64_t expected_k
    int64_t num_cores
    int64_t outer_stride_a
    int64_t outer_stride_b
    int64_t outer_stride_d
    int64_t output_alignment
    bint compiled_k
    bint compiled_n
    bint identity_epilogue
    bint with_bias
    int64_t stride_batch_a
    int64_t stride_batch_b


cdef struct GemmConfig:
    int64_t block_m
    int64_t block_n
    int64_t block_k
    int64_t mad_m
    int64_t mad_n
    int64_t mad_k
    int64_t num_l1_stages
    int64_t num_l0_stages
    int64_t num_epilogue_stages
    int64_t num_l1_sf_stages
    int64_t sf_k_blocks
    int64_t l2_ctrl_a
    int64_t l2_ctrl_b
    int64_t l2_ctrl_store_cd
    int64_t num_blocks
    bint direct_store
    bint dual_aiv_dequant
    bint resident_a
    bint resident_b
    bint resident_sf


cdef struct GemmDtype:
    int64_t elem_bits
    int64_t epilogue_elem_bits
    int64_t output_elem_bits
    bint with_fp8_cast
    bint has_sf
    bint dequant_b


cdef struct GemmComputeCost:
    int64_t mflops
    int64_t time_ns
    int64_t num_tail_blocks


cdef struct HWDesc:
    int64_t gm_bw_gbps
    int64_t l2_bw_gbps
    int64_t aic_clock_mhz
    int64_t bf16_flops_per_cycle
    int64_t l1_to_l0_bytes_per_cycle
    int64_t l1_to_l0_sf_bytes_per_cycle
    int64_t ub_to_l1_bytes_per_cycle
    int64_t l0c_to_ub_bytes_per_cycle
    int64_t fp4_decode_bytes_per_cycle
    int64_t num_pipe_events
    int64_t num_intra_block_events
    int64_t num_ai_cores
    int64_t l2_size_bytes
    int64_t l2_sector_bytes
    int64_t sf_elements_per_pair
    int64_t fractal_row_bytes


cdef inline GemmConfig empty_config() noexcept:
    cdef GemmConfig c
    memset(&c, 0, sizeof(GemmConfig))
    return c


cdef inline HWDesc get_hw_desc() noexcept:
    return HWDesc(
        gm_bw_gbps=3500,
        l2_bw_gbps=4500,
        aic_clock_mhz=1650,
        bf16_flops_per_cycle=8192,
        l1_to_l0_bytes_per_cycle=256,
        l1_to_l0_sf_bytes_per_cycle=32,
        ub_to_l1_bytes_per_cycle=128,
        l0c_to_ub_bytes_per_cycle=128,
        fp4_decode_bytes_per_cycle=64,
        num_pipe_events=8,
        num_intra_block_events=16,
        num_ai_cores=32,
        l2_size_bytes=128 * 1024 * 1024,
        l2_sector_bytes=128,
        sf_elements_per_pair=64,
        fractal_row_bytes=32,
    )


cdef inline GemmDtype get_gemm_dtype(object desc):
    cdef GemmDtype dtype
    if desc.cd_dtype not in ("bfloat16", "float32", "float8_e4m3fn"):
        raise ValueError(f"Unsupported output dtype: {desc.cd_dtype}")
    dtype = GemmDtype(elem_bits=0, epilogue_elem_bits=32, output_elem_bits=32, with_fp8_cast=False, has_sf=False, dequant_b=False)
    dtype.with_fp8_cast = desc.cd_dtype == "float8_e4m3fn"
    dtype.output_elem_bits = 8 if dtype.with_fp8_cast else 16 if desc.cd_dtype == "bfloat16" else 32
    dtype.epilogue_elem_bits = 16 if dtype.with_fp8_cast or (desc.cd_dtype == "bfloat16" and desc.epilogue_kind == "identity") else 32
    if desc.a_dtype == "bfloat16" and desc.b_dtype == "bfloat16":
        dtype.elem_bits = 16
    elif desc.a_dtype == "float4_e2m1fn_x2" and desc.b_dtype == "float4_e2m1fn_x2":
        dtype.elem_bits = 4
        dtype.has_sf = True
    elif desc.a_dtype == "float8_e4m3fn" and desc.b_dtype in ("float8_e4m3fn", "float4_e2m1fn_x2"):
        dtype.elem_bits = 8
        dtype.has_sf = True
        dtype.dequant_b = desc.b_dtype == "float4_e2m1fn_x2"
    else:
        raise ValueError(f"Unsupported dtype pair: A={desc.a_dtype} B={desc.b_dtype}")
    return dtype


cdef inline int64_t ceil_mul_div(int64_t a, int64_t b, int64_t divisor):
    # Use Python integers when a load-cost product would overflow int64.
    if (a | b) < (<int64_t>1 << 31) or a <= (_INT64_MAX - divisor + 1) // b:
        return (a * b + divisor - 1) // divisor
    return (<object>a * b + divisor - 1) // divisor


cdef inline int64_t ceil_div(int64_t a, int64_t b) noexcept:
    return (a + b - 1) // b


cdef inline int64_t align(int64_t value, int64_t alignment) noexcept:
    return ceil_div(value, alignment) * alignment


cdef inline int64_t gcd(int64_t a, int64_t b) noexcept:
    while b:
        a, b = b, a % b
    return a


cdef inline bint is_k_major(Major major) noexcept:
    return major == Major.K


cdef inline bint is_mn_major(Major major) noexcept:
    return major == Major.MN


cdef inline bint is_m_grouped(GemmType gemm_type) noexcept:
    return gemm_type == GemmType.MGroupedContiguousWithPsumLayout


cdef inline bint is_k_grouped(GemmType gemm_type) noexcept:
    return gemm_type == GemmType.KGroupedContiguousWithPsumLayout


cdef inline bint is_batched(GemmType gemm_type) noexcept:
    return gemm_type == GemmType.Batched


cdef inline int64_t get_mad_alignment_k(int64_t elem_bits) noexcept:
    return get_hw_desc().fractal_row_bytes * 8 // elem_bits


cdef inline int64_t get_mad_flops_per_cycle(int64_t elem_bits) noexcept:
    return get_hw_desc().bf16_flops_per_cycle * 16 // elem_bits


cdef inline int64_t gm_load_bytes(
    int64_t mn, int64_t extent_k, int64_t total_mn, int64_t elem_bits,
    Major major, int64_t outer_stride, int64_t block_k
) noexcept:
    cdef HWDesc hw
    cdef int64_t block_bytes, boundary_period, mn_major, num_row_blocks, row_bytes, row_load_bytes, stride_bytes
    cdef int64_t stride_gcd
    hw = get_hw_desc()
    mn_major = is_mn_major(major)
    row_bytes = ceil_div((total_mn if mn_major else extent_k) * elem_bits, 8)
    block_bytes = ceil_div((mn if mn_major else block_k) * elem_bits, 8)
    # FP4 strides count packed bytes; logical extents count nibbles.
    stride_bytes = outer_stride * max(elem_bits, 8) // 8 if outer_stride else row_bytes
    num_row_blocks = ceil_div(row_bytes, block_bytes)
    stride_gcd = gcd(stride_bytes, hw.l2_sector_bytes)
    boundary_period = stride_gcd // gcd(block_bytes, stride_gcd)
    row_load_bytes = (
        align(row_bytes, stride_gcd) + num_row_blocks * hw.l2_sector_bytes
        - stride_gcd * (1 + (num_row_blocks - 1) // boundary_period)
    )
    return ceil_div(extent_k * row_load_bytes, num_row_blocks) if mn_major else mn * row_load_bytes


cdef inline int64_t sf_load_bytes(int64_t mn, int64_t extent_k, int64_t total_mn, bint has_sf, int64_t block_k) noexcept:
    if not has_sf:
        return 0
    return gm_load_bytes(mn, ceil_div(extent_k, get_hw_desc().sf_elements_per_pair), total_mn, 16, Major.MN, total_mn, block_k)


cdef inline int64_t load_time_ns(
    int64_t total_gm_bytes, int64_t total_load_bytes,
    int64_t num_waves, int64_t num_blocks, int64_t num_batches
):
    cdef HWDesc hw
    cdef int64_t gm_time, l2_bytes, l2_time, wave_cores
    hw = get_hw_desc()
    total_gm_bytes = min(total_gm_bytes, total_load_bytes)
    l2_bytes = total_load_bytes - total_gm_bytes
    wave_cores = num_waves * hw.num_ai_cores
    gm_time = ceil_mul_div(total_gm_bytes, wave_cores, num_blocks * num_batches * hw.gm_bw_gbps)
    l2_time = ceil_mul_div(l2_bytes, wave_cores, num_blocks * num_batches * hw.l2_bw_gbps)
    return gm_time + l2_time


cdef inline int64_t mn_tail_bytes(int64_t num_rows, int64_t elem_bits, int64_t stride_bytes, int64_t m, int64_t tail_m) noexcept:
    cdef HWDesc hw
    cdef int64_t offset_bytes, row_bytes, stride_gcd
    hw = get_hw_desc()
    stride_gcd = gcd(stride_bytes, hw.l2_sector_bytes)
    offset_bytes = (m - tail_m) * elem_bits // 8
    row_bytes = ceil_div(tail_m * elem_bits, 8)
    return num_rows * (align(offset_bytes % stride_gcd + row_bytes, stride_gcd) + hw.l2_sector_bytes - stride_gcd)


cdef inline bint make_gemm_candidate(
    GemmDesc d, GemmDtype dtype, int64_t block_m, int64_t block_n, int64_t block_k, GemmConfig* result
) noexcept:
    """Fit MAD tiles and all buffer stages together; reject an infeasible block."""
    cdef HWDesc hw
    cdef int64_t ab_bytes, align_m, align_n, desired_sf_blocks, k, k_alignment, l0a_k, l0b_k, mad_k, mad_m, mad_n
    cdef int64_t max_l1_stages, mte1_copy_k, num_batches, num_blocks, num_cd_stages, num_epilogue_stages, num_k_blocks
    cdef int64_t num_l0_stages, num_l1_sf_stages, num_l1_stages, num_m_blocks, num_n_blocks, row_elements, sf_blocks_fit
    cdef int64_t sf_bytes, sf_k_blocks, split, split_m, split_n
    hw = get_hw_desc()
    if block_m * block_n * 4 > L0C_SIZE_BYTES:
        return False
    if is_m_grouped(d.gemm_type) and block_m != MK_ALIGNMENT:
        return False
    k_alignment = hw.sf_elements_per_pair if dtype.has_sf else get_mad_alignment_k(dtype.elem_bits)
    if block_k % k_alignment:
        return False

    num_l0_stages = 2
    align_m = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_a) else FRAC_MN
    align_n = 64 if d.with_bias else 128 if dtype.with_fp8_cast else get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_b) else FRAC_MN
    for split in range(3):
        split_m = 2 if split == 2 else 1
        split_n = 2 if split == 1 else 1
        if block_m % (split_m * align_m) or block_n % (split_n * align_n):
            continue
        mad_m, mad_n = block_m // split_m, block_n // split_n
        if mad_m * mad_n * 4 > L0C_SIZE_BYTES // 2:
            continue
        l0a_k = L0A_SIZE_BYTES * 8 // (num_l0_stages * mad_m * dtype.elem_bits)
        l0b_k = L0B_SIZE_BYTES * 8 // (num_l0_stages * mad_n * dtype.elem_bits)
        mad_k = min(l0a_k, l0b_k, max(k_alignment, block_k // num_l0_stages))
        mad_k = mad_k // k_alignment * k_alignment
        while mad_k and block_k % mad_k:
            mad_k -= k_alignment
        # Split MN before shortening K enough to amplify the L1 -> L0 copies.
        mte1_copy_k = hw.l1_to_l0_bytes_per_cycle * 8 // dtype.elem_bits
        if mad_k < min(block_k, mte1_copy_k) // num_l0_stages:
            continue
        break
    else:
        return False
    if not mad_k:
        return False

    if dtype.dequant_b:
        row_elements = block_k if is_k_major(d.major_b) else block_n
        if block_k > 512 or row_elements not in (256, 512):
            return False

    k = d.num_groups * d.expected_k if is_k_grouped(d.gemm_type) else d.k
    sf_k_blocks = min(4, ceil_div(k, block_k)) if dtype.has_sf else 0
    num_l1_sf_stages = 2 if dtype.has_sf else 0
    max_l1_stages = min(4, hw.num_pipe_events - num_l1_sf_stages) if dtype.has_sf else hw.num_pipe_events
    sf_bytes = (block_m + block_n) * (block_k // hw.sf_elements_per_pair) * 2 * sf_k_blocks * num_l1_sf_stages
    ab_bytes = (block_m + block_n) * block_k * dtype.elem_bits // 8
    num_l1_stages = min(max_l1_stages, (L1_SIZE_BYTES - sf_bytes) // ab_bytes)
    if num_l1_stages < 2:
        return False
    if dtype.has_sf:
        num_k_blocks = ceil_div(k, block_k)
        desired_sf_blocks = ceil_div(num_k_blocks, num_l1_sf_stages) if dtype.dequant_b else num_k_blocks
        sf_blocks_fit = (L1_SIZE_BYTES - ab_bytes * num_l1_stages) // (sf_bytes // sf_k_blocks)
        sf_k_blocks = min(max(desired_sf_blocks, sf_k_blocks), sf_blocks_fit)

    num_cd_stages = (block_m // mad_m) * (block_n // mad_n)
    num_batches = d.num_groups if is_batched(d.gemm_type) or is_k_grouped(d.gemm_type) else 1
    num_blocks = ceil_div(d.m, block_m) * ceil_div(d.n, block_n) * num_batches
    num_epilogue_stages = min(
        hw.num_intra_block_events,
        UB_SIZE_BYTES // (mad_m * mad_n * dtype.epilogue_elem_bits // 8 + (block_n * dtype.output_elem_bits // 8 if d.with_bias else 0) + (mad_m * mad_n // 32 if dtype.with_fp8_cast else 0)),
        ceil_div(num_blocks, d.num_cores) * num_cd_stages,
    )
    if num_epilogue_stages * (1 if dtype.dequant_b else 2) < num_cd_stages:
        return False

    num_m_blocks, num_n_blocks = ceil_div(d.m, block_m), ceil_div(d.n, block_n)
    result[0] = GemmConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        mad_m=mad_m,
        mad_n=mad_n,
        mad_k=mad_k,
        num_l1_stages=num_l1_stages,
        num_l0_stages=num_l0_stages,
        num_epilogue_stages=num_epilogue_stages,
        num_l1_sf_stages=num_l1_sf_stages,
        sf_k_blocks=sf_k_blocks,
        l2_ctrl_a=L2_LD_NOTALLOC_KEEP if num_n_blocks <= 2 and num_m_blocks > num_n_blocks else L2_LD_NORMAL_FV,
        l2_ctrl_b=L2_LD_NOTALLOC_KEEP if num_m_blocks <= 2 and num_n_blocks > num_m_blocks else L2_LD_NORMAL_FV,
        l2_ctrl_store_cd=L2_ST_NORMAL_FV,
        num_blocks=d.num_cores,
        direct_store=False,
        dual_aiv_dequant=False,
        resident_a=False, resident_b=False, resident_sf=False,
    )

    return True


cdef inline GemmComputeCost get_gemm_compute_cost(GemmDesc d, GemmDtype dtype, int64_t block_m, int64_t block_n) noexcept:
    cdef HWDesc hw
    cdef int64_t align_m, align_n, compute_flops, compute_m, compute_mflops, compute_mn, compute_n, k, m, n, num_batches
    cdef int64_t num_blocks, num_core_classes, num_full_blocks, num_full_per_group, num_full_per_period
    cdef int64_t num_full_remaining, num_groups_per_period, num_m_blocks, num_n_blocks, num_remaining_groups
    cdef int64_t num_tail_blocks, num_waves, tail_m, tail_remainder, tail_stride, tail_suffix
    hw = get_hw_desc()
    m, n = d.m, d.n
    k = d.expected_k if is_k_grouped(d.gemm_type) else d.k
    num_m_blocks, num_n_blocks = ceil_div(m, block_m), ceil_div(n, block_n)
    num_blocks = num_m_blocks * num_n_blocks
    num_batches = d.num_groups if is_batched(d.gemm_type) or is_k_grouped(d.gemm_type) else 1
    num_waves = ceil_div(num_blocks * num_batches, d.num_cores)
    align_m = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_a) else FRAC_MN
    align_n = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_b) else FRAC_MN
    compute_m = block_m if is_m_grouped(d.gemm_type) else align(min(m, block_m), align_m)
    compute_n = align(min(n, block_n), align_n)
    compute_mn = num_waves * compute_m * compute_n
    num_tail_blocks = 0
    tail_stride = 1 if num_m_blocks % 4 else 4
    if not is_m_grouped(d.gemm_type) and gcd(tail_stride, d.num_cores) == 1:
        num_core_classes = gcd(num_blocks, d.num_cores)
        num_groups_per_period = d.num_cores // num_core_classes
        num_remaining_groups = num_batches % num_groups_per_period
        num_full_per_period = ceil_div(num_blocks - num_n_blocks, num_core_classes)
        tail_remainder = num_n_blocks % d.num_cores
        tail_suffix = tail_remainder if tail_stride == 1 else min(tail_remainder, 1)
        num_full_per_group = ceil_div(num_blocks - tail_suffix, d.num_cores) - num_n_blocks // d.num_cores
        num_full_remaining = min(num_remaining_groups * num_full_per_group, num_full_per_period)
        num_full_blocks = num_batches // num_groups_per_period * num_full_per_period + num_full_remaining
        num_tail_blocks = max(0, num_waves - num_full_blocks)
        tail_m = align(m - (num_m_blocks - 1) * block_m, align_m)
        compute_mn -= num_tail_blocks * (compute_m - tail_m) * compute_n
    compute_flops = compute_mn * 2 * k
    compute_mflops = get_mad_flops_per_cycle(dtype.elem_bits) * hw.aic_clock_mhz
    return GemmComputeCost(compute_mflops, ceil_div(compute_flops * 1000, compute_mflops), num_tail_blocks)


cdef inline int64_t get_gemm_cost(GemmDesc d, GemmDtype dtype, GemmConfig c, GemmComputeCost compute_cost, int64_t best_cost):
    """Estimate overlapped compute, cold GM/L2 traffic, and L1 -> L0 time in ns."""
    cdef HWDesc hw
    cdef int64_t a_block_bytes, a_bytes, a_contiguous_bytes, a_stride_bytes, ab_bytes, align_m, align_n, b_block_bytes
    cdef int64_t b_bytes, b_contiguous_bytes, b_elem_bits, block_m, block_n, block_swizzle_m, block_swizzle_n
    cdef int64_t compute_bound_time_ns, compute_m, compute_mflops, compute_n, compute_time_ns, contiguous_bytes
    cdef int64_t dequant_bytes, dequant_time_ns, discount_ns, first_a_bytes, first_b_bytes, first_k, first_load_time_ns
    cdef int64_t first_sf_k, first_sfa_bytes, first_sfb_bytes, first_stage_time_ns, k, l0_time_ns, l2_a_row_bytes
    cdef int64_t l2_b_row_bytes, last_compute_time_ns, last_k, load_a_time_ns, load_b_time_ns, load_m, m, memory_time_ns
    cdef int64_t n, num_a_gm_blocks, num_b_gm_blocks, num_batches, num_blocks, num_cold_tail_waves, num_l0_rows
    cdef int64_t num_l0a_loads, num_l0b_loads, num_m_blocks, num_n_blocks, num_tail_blocks, num_tail_wave_blocks
    cdef int64_t num_waves, prefetch_compute_ns, prefetch_k, reuse_b, sf_bytes, sf_overlap_time_ns, sf_row_bytes
    cdef int64_t sf_startup_time_ns, sfa_block_bytes, sfb_block_bytes, swizzle_a_bytes, swizzle_b_bytes, tail_a_bytes
    cdef int64_t tail_compute_m, tail_compute_ns, tail_m, tail_memory_delay_ns, tail_memory_ns, tail_sfa_bytes
    cdef int64_t wa_block_bytes, wb_block_bytes, a_load_blocks, b_load_blocks, sfa_load_blocks, sfb_load_blocks
    hw = get_hw_desc()
    m, n = d.m, d.n
    k = d.expected_k if is_k_grouped(d.gemm_type) else d.k
    block_m, block_n = c.block_m, c.block_n
    num_m_blocks, num_n_blocks = ceil_div(m, block_m), ceil_div(n, block_n)
    num_blocks = num_m_blocks * num_n_blocks
    num_batches = d.num_groups if is_batched(d.gemm_type) or is_k_grouped(d.gemm_type) else 1
    num_waves = ceil_div(num_blocks * num_batches, d.num_cores)
    b_elem_bits = dtype.elem_bits // 2 if dtype.dequant_b else dtype.elem_bits
    sf_row_bytes = ceil_div(k, hw.sf_elements_per_pair) * 2 if dtype.has_sf else 0
    align_m = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_a) else FRAC_MN
    align_n = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_b) else FRAC_MN
    compute_m = block_m if is_m_grouped(d.gemm_type) else align(min(m, block_m), align_m)
    compute_n = align(min(n, block_n), align_n)
    compute_mflops = compute_cost.mflops
    compute_time_ns = compute_cost.time_ns
    num_tail_blocks = compute_cost.num_tail_blocks

    load_m = num_m_blocks * block_m if is_m_grouped(d.gemm_type) else m
    sfa_block_bytes = sf_load_bytes(block_m, k, load_m, dtype.has_sf, c.block_k)
    sfb_block_bytes = sf_load_bytes(block_n, k, n, dtype.has_sf, c.block_k)
    wa_block_bytes = gm_load_bytes(block_m, k, load_m, dtype.elem_bits, d.major_a, d.outer_stride_a, c.block_k)
    wb_block_bytes = gm_load_bytes(block_n, k, n, b_elem_bits, d.major_b, d.outer_stride_b, c.block_k)
    if dtype.elem_bits == 16 and c.resident_a and is_k_major(d.major_a):
        wa_block_bytes = block_m * align(k * 2, hw.l2_sector_bytes)
    a_block_bytes = wa_block_bytes + sfa_block_bytes
    b_block_bytes = wb_block_bytes + sfb_block_bytes
    l2_a_row_bytes = 0 if c.l2_ctrl_a == L2_LD_NOTALLOC_KEEP else sf_row_bytes + ceil_div(k * dtype.elem_bits, 8)
    l2_b_row_bytes = 0 if c.l2_ctrl_b == L2_LD_NOTALLOC_KEEP else sf_row_bytes + ceil_div(k * b_elem_bits, 8)

    # First use is cold; evicted operands are reloaded at swizzle boundaries.
    block_swizzle_m = min(num_m_blocks, 4)
    block_swizzle_n = min(num_n_blocks, hw.num_ai_cores // block_swizzle_m)
    swizzle_a_bytes = min(m, block_swizzle_m * block_m) * l2_a_row_bytes
    swizzle_b_bytes = min(n, block_swizzle_n * block_n) * l2_b_row_bytes
    num_a_gm_blocks = num_m_blocks * num_batches
    num_b_gm_blocks = num_n_blocks * num_batches
    if swizzle_a_bytes + swizzle_b_bytes > hw.l2_size_bytes:
        num_a_gm_blocks *= ceil_div(num_n_blocks, block_swizzle_n)
    if swizzle_a_bytes + n * l2_b_row_bytes > hw.l2_size_bytes:
        num_b_gm_blocks *= ceil_div(num_m_blocks, block_swizzle_m)
    if is_m_grouped(d.gemm_type):
        num_b_gm_blocks = min(num_blocks, max(num_b_gm_blocks, num_n_blocks * d.num_groups))
    if c.l2_ctrl_a == L2_LD_NOTALLOC_KEEP:
        num_a_gm_blocks = num_blocks * num_batches
    if c.l2_ctrl_b == L2_LD_NOTALLOC_KEEP:
        num_b_gm_blocks = num_blocks * num_batches

    a_load_blocks = min(num_blocks, d.num_cores) * num_batches if c.resident_a and num_m_blocks == 1 else num_blocks * num_batches
    b_load_blocks = (min(num_blocks, d.num_cores * (d.num_groups if is_m_grouped(d.gemm_type) else 1)) * num_batches
                     if c.resident_b and num_n_blocks == 1 else num_blocks * num_batches)
    sfa_load_blocks = a_load_blocks if c.resident_sf else num_blocks * num_batches
    sfb_load_blocks = b_load_blocks if c.resident_sf else num_blocks * num_batches
    # Broadcast operands can be retained across batches; SF is batch-specific.
    if dtype.has_sf and is_batched(d.gemm_type):
        if c.resident_a and d.stride_batch_a == 0:
            a_load_blocks //= num_batches
        if c.resident_b and d.stride_batch_b == 0:
            b_load_blocks //= num_batches
    load_a_time_ns = load_time_ns(
        num_a_gm_blocks * a_block_bytes, a_load_blocks * wa_block_bytes + sfa_load_blocks * sfa_block_bytes,
        num_waves, num_blocks, num_batches,
    )
    load_b_time_ns = load_time_ns(
        num_b_gm_blocks * b_block_bytes, b_load_blocks * wb_block_bytes + sfb_load_blocks * sfb_block_bytes,
        num_waves, num_blocks, num_batches,
    )
    contiguous_bytes = (block_n if is_k_major(d.major_b) else c.block_k) // 2 * hw.fractal_row_bytes
    dequant_bytes = (
        ceil_div(b_load_blocks, d.num_cores) * block_n * k * dtype.elem_bits // 8 * align(contiguous_bytes, hw.ub_to_l1_bytes_per_cycle) // contiguous_bytes
        if dtype.dequant_b
        else 0
    )
    dequant_time_ns = ceil_div(dequant_bytes * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz)
    memory_time_ns = load_a_time_ns + load_b_time_ns + dequant_time_ns
    if memory_time_ns > best_cost:
        return memory_time_ns

    # A tail cannot save more compute time than its cold-memory service permits.
    tail_m = m - (num_m_blocks - 1) * block_m
    tail_compute_m = align(tail_m, align_m)
    discount_ns = ceil_div(num_tail_blocks * (compute_m - tail_compute_m) * compute_n * 2 * k * 1000, compute_mflops)
    tail_memory_delay_ns = 0
    if discount_ns:

        a_stride_bytes = d.outer_stride_a * max(dtype.elem_bits, 8) // 8 if d.outer_stride_a else ceil_div(m * dtype.elem_bits, 8)
        tail_a_bytes = (
            mn_tail_bytes(k, dtype.elem_bits, a_stride_bytes, m, tail_m)
            if is_mn_major(d.major_a)
            else gm_load_bytes(tail_m, k, m, dtype.elem_bits, d.major_a, d.outer_stride_a, c.block_k)
        )
        tail_sfa_bytes = mn_tail_bytes(ceil_div(k, hw.sf_elements_per_pair), 16, m * 2, m, tail_m) if dtype.has_sf else 0
        reuse_b = (
            num_m_blocks > 1
            and c.l2_ctrl_b != L2_LD_NOTALLOC_KEEP
            and min(m, 4 * block_m) * l2_a_row_bytes + n * l2_b_row_bytes <= hw.l2_size_bytes
        )
        num_tail_wave_blocks = min(num_n_blocks, d.num_cores)
        tail_memory_ns = (
            ceil_div((tail_a_bytes + tail_sfa_bytes) * num_tail_wave_blocks, hw.gm_bw_gbps)
            + ceil_div(b_block_bytes * num_tail_wave_blocks, hw.l2_bw_gbps if reuse_b else hw.gm_bw_gbps)
            + ceil_div(dequant_time_ns, num_waves)
        )
        tail_compute_ns = ceil_div(tail_compute_m * compute_n * 2 * k * 1000, compute_mflops)
        num_cold_tail_waves = num_tail_blocks if c.l2_ctrl_a == L2_LD_NOTALLOC_KEEP else min(num_batches, num_tail_blocks)
        tail_memory_delay_ns = min(discount_ns, num_cold_tail_waves * max(0, tail_memory_ns - tail_compute_ns))

    # Initial A/B/SF fill is exposed; later SF loads can overlap MADs.
    first_k = min(k, c.block_k)
    first_sf_k = min(k, c.block_k * c.sf_k_blocks)
    first_sfa_bytes = sf_load_bytes(block_m, first_sf_k, load_m, dtype.has_sf, c.block_k)
    first_sfb_bytes = sf_load_bytes(block_n, first_sf_k, n, dtype.has_sf, c.block_k)
    first_a_bytes = (a_block_bytes if dtype.elem_bits == 16 and c.resident_a else
                     gm_load_bytes(block_m, first_k, load_m, dtype.elem_bits, d.major_a, d.outer_stride_a, c.block_k) + first_sfa_bytes)
    first_b_bytes = gm_load_bytes(block_n, first_k, n, b_elem_bits, d.major_b, d.outer_stride_b, c.block_k) + first_sfb_bytes
    first_stage_time_ns = (
        ceil_div(load_a_time_ns * first_a_bytes, a_block_bytes)
        + ceil_div(load_b_time_ns * first_b_bytes, b_block_bytes)
        + ceil_div(dequant_bytes * first_k // k * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz)
    )
    sf_startup_time_ns = ceil_div(load_a_time_ns * first_sfa_bytes, a_block_bytes) + ceil_div(
        load_b_time_ns * first_sfb_bytes, b_block_bytes
    )
    prefetch_k = min(k, (c.num_l1_stages - 1) * c.block_k)
    prefetch_compute_ns = ceil_div(prefetch_k * 2 * compute_m * compute_n * 1000, compute_mflops)
    sf_overlap_time_ns = min(sf_startup_time_ns // num_waves, prefetch_compute_ns) if c.num_l1_sf_stages > 1 else 0
    first_load_time_ns = first_stage_time_ns - (num_waves - 1) * sf_overlap_time_ns

    # Model L0 operand reuse and transfer-width amplification.
    num_l0a_loads = 1 if c.block_k <= c.mad_k * c.num_l0_stages else block_n // c.mad_n
    num_l0b_loads = 1 if block_n == c.mad_n and c.block_k <= c.mad_k * c.num_l0_stages else block_m // c.mad_m
    num_l0_rows = num_waves * (block_m * num_l0a_loads + block_n * num_l0b_loads)
    a_bytes = num_waves * block_m * num_l0a_loads * k * dtype.elem_bits // 8
    b_bytes = (1 if dtype.elem_bits == 16 and c.resident_b else num_waves) * block_n * num_l0b_loads * k * dtype.elem_bits // 8
    a_contiguous_bytes = (c.mad_m if is_k_major(d.major_a) else c.mad_k) * hw.fractal_row_bytes
    b_contiguous_bytes = (c.mad_n if is_k_major(d.major_b) else c.mad_k) * hw.fractal_row_bytes
    ab_bytes = (
        a_bytes * align(a_contiguous_bytes, hw.l1_to_l0_bytes_per_cycle) // a_contiguous_bytes
        + b_bytes * align(b_contiguous_bytes, hw.l1_to_l0_bytes_per_cycle) // b_contiguous_bytes
    )
    sf_bytes = num_l0_rows * sf_row_bytes
    l0_time_ns = ceil_div(ab_bytes * 1000, hw.l1_to_l0_bytes_per_cycle * hw.aic_clock_mhz) + ceil_div(
        sf_bytes * 1000, hw.l1_to_l0_sf_bytes_per_cycle * hw.aic_clock_mhz
    )
    compute_bound_time_ns = first_load_time_ns + max(l0_time_ns, compute_time_ns + tail_memory_delay_ns)
    last_k = k % c.block_k or c.block_k
    last_compute_time_ns = ceil_div(last_k * compute_m * compute_n * 2 * 1000, compute_mflops)
    return max(compute_bound_time_ns, memory_time_ns + last_compute_time_ns)


cdef inline int64_t get_gemm_output_cost(GemmDesc d, GemmDtype dtype, GemmConfig c) noexcept:
    cdef HWDesc hw = get_hw_desc()
    cdef int64_t waves = ceil_div(ceil_div(d.m, c.block_m) * ceil_div(d.n, c.block_n), d.num_cores)
    cdef int64_t values = min(d.m, c.block_m) * min(d.n, c.block_n)
    cdef int64_t fix_ns = ceil_div(values * dtype.epilogue_elem_bits // 8 * 1000, hw.l0c_to_ub_bytes_per_cycle * hw.aic_clock_mhz)
    cdef int64_t gm_bytes = values * dtype.output_elem_bits // 8 * (1 + d.acc) + (values // 32 if dtype.with_fp8_cast else 0)
    cdef int64_t gm_ns = ceil_div(gm_bytes * hw.num_ai_cores, hw.gm_bw_gbps)
    cdef int64_t bias_bytes = (1 if dtype.dequant_b else 2) * min(d.n, c.block_n) * dtype.output_elem_bits // 8 if d.with_bias else 0
    cdef int64_t bias_ns = ceil_div(bias_bytes * hw.num_ai_cores, hw.gm_bw_gbps)
    cdef int64_t vector_values = align(min(d.m, c.block_m), c.mad_m) * align(min(d.n, c.block_n), c.mad_n)
    cdef int64_t vector_ns = ceil_div(ceil_div(vector_values * 4, 256) * 1000, hw.aic_clock_mhz) if d.with_bias else 0
    if d.with_bias and c.num_epilogue_stages >= 2 and waves >= 2:
        return waves * max(fix_ns, bias_ns + gm_ns + vector_ns)
    return waves * (max(fix_ns, gm_ns) if c.direct_store else max(fix_ns, bias_ns) + gm_ns + vector_ns)


cdef inline int64_t get_gemm_refinement_cost(GemmDesc d, GemmDtype dtype, GemmConfig c):
    cdef HWDesc hw = get_hw_desc()
    cdef GemmComputeCost compute = get_gemm_compute_cost(d, dtype, c.block_m, c.block_n)
    cdef int64_t cost = get_gemm_cost(d, dtype, c, compute, _INT64_MAX)
    cdef int64_t waves, loads, blocks, batches, decoded_bytes, decode_ns, store_ns, b_load_blocks
    if dtype.elem_bits == 16 and d.m < c.block_m:
        waves = ceil_div(ceil_div(d.n, c.block_n), d.num_cores)
        loads = (1 if c.resident_a else waves * ceil_div(d.k, c.block_k)) + waves * ceil_div(d.k, c.block_k)
        cost += ceil_div(loads * 64 * 1000, hw.aic_clock_mhz)
    if dtype.dequant_b:
        blocks = ceil_div(d.m, c.block_m) * ceil_div(d.n, c.block_n)
        batches = d.num_groups if is_batched(d.gemm_type) or is_k_grouped(d.gemm_type) else 1
        b_load_blocks = (min(blocks, d.num_cores * (d.num_groups if is_m_grouped(d.gemm_type) else 1)) * batches
                         if c.resident_b else blocks * batches)
        if c.resident_b and is_batched(d.gemm_type) and d.stride_batch_b == 0:
            b_load_blocks //= batches
        decoded_bytes = ceil_div(b_load_blocks, d.num_cores) * c.block_n * (d.expected_k if is_k_grouped(d.gemm_type) else d.k)
        decode_ns = ceil_div(decoded_bytes * 1000, (2 if c.dual_aiv_dequant else 1) * hw.fp4_decode_bytes_per_cycle * hw.aic_clock_mhz)
        store_ns = ceil_div(decoded_bytes * 1000, hw.ub_to_l1_bytes_per_cycle * hw.aic_clock_mhz)
        cost += max(0, decode_ns - max(compute.time_ns, store_ns))
    return cost + get_gemm_output_cost(d, dtype, c)


cdef inline GemmConfig refine_gemm_pipeline(GemmDesc d, GemmDtype dtype, GemmConfig c):
    cdef HWDesc hw = get_hw_desc()
    cdef int64_t stride_d = d.outer_stride_d or d.n
    cdef int64_t align_m = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_a) else 1
    cdef int64_t align_n = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_b) else FRAC_MN
    cdef GemmConfig candidate
    cdef int64_t num_m_blocks = ceil_div(d.m, c.block_m)
    cdef int64_t num_n_blocks = ceil_div(d.n, c.block_n)
    cdef int64_t num_waves = ceil_div(num_m_blocks * num_n_blocks, d.num_cores)
    cdef int64_t resident_k = d.expected_k if is_k_grouped(d.gemm_type) else d.k
    cdef int64_t a_bytes, b_bytes, num_l1_stages
    if (d.gemm_type == GemmType.Normal and d.identity_epilogue
        and (dtype.elem_bits != 4 or (is_k_major(d.major_a) and is_k_major(d.major_b)))
        and (not d.acc or dtype.output_elem_bits == 32) and d.m % align_m == 0 and d.n % align_n == 0
        and d.output_alignment == 0 and stride_d * dtype.epilogue_elem_bits // 8 % 32 == 0
        and (ceil_div(d.m, c.block_m) * ceil_div(d.n, c.block_n) <= d.num_cores
             or (d.m <= c.block_m and c.mad_n * dtype.epilogue_elem_bits // 8 % hw.l2_sector_bytes == 0))):
        candidate = c
        candidate.direct_store = True
        if get_gemm_output_cost(d, dtype, candidate) <= get_gemm_output_cost(d, dtype, c):
            c = candidate
    if (d.gemm_type == GemmType.Normal and d.compiled_k) or (dtype.has_sf and (d.compiled_k or is_k_grouped(d.gemm_type))):
        candidate = c
        if dtype.elem_bits == 16:
            a_bytes = c.block_m * align(d.k, FRAC_MN) * 2
            b_bytes = c.block_n * c.block_k * 2
            num_l1_stages = min(hw.num_pipe_events, (L1_SIZE_BYTES - a_bytes) // b_bytes) if a_bytes < L1_SIZE_BYTES else 0
            if d.m <= c.block_m and num_l1_stages > c.num_l0_stages:
                candidate.resident_a = True
                candidate.num_l1_stages = num_l1_stages
            candidate.resident_b = (num_n_blocks == 1 and num_waves > 1 and c.block_n == c.mad_n
                                    and d.k <= c.mad_k * c.num_l0_stages)
        elif resident_k <= c.num_l1_stages * c.block_k and num_waves >= 2:
            candidate.resident_a = num_m_blocks == 1
            candidate.resident_b = (num_n_blocks == 1 and (not dtype.dequant_b or not is_m_grouped(d.gemm_type)
                                                         or ceil_div(resident_k, c.block_k) >= 2))
            candidate.resident_sf = (candidate.resident_a or candidate.resident_b) and resident_k <= c.sf_k_blocks * c.block_k
        if get_gemm_refinement_cost(d, dtype, candidate) <= get_gemm_refinement_cost(d, dtype, c):
            c = candidate
    if dtype.dequant_b and c.direct_store:
        candidate = c
        candidate.dual_aiv_dequant = True
        if get_gemm_refinement_cost(d, dtype, candidate) < get_gemm_refinement_cost(d, dtype, c):
            c = candidate
    return c


cdef inline int64_t get_gemm_k_cost(GemmDesc d, GemmDtype dtype, GemmConfig c):
    cdef HWDesc hw = get_hw_desc()
    cdef int64_t waves = ceil_div(ceil_div(d.m, c.block_m) * ceil_div(d.n, c.block_n), d.num_cores)
    cdef int64_t mad_cycles = ceil_div(2 * c.mad_m * c.mad_n * c.mad_k, get_mad_flops_per_cycle(dtype.elem_bits))
    cdef int64_t load_cycles = ceil_div((c.mad_m + c.mad_n) * c.mad_k * dtype.elem_bits, 8 * hw.l1_to_l0_bytes_per_cycle)
    cdef int64_t num_mads = c.block_m // c.mad_m * (c.block_n // c.mad_n)
    cdef int64_t issue_cycles = 64 if dtype.elem_bits == 16 else 40 if dtype.elem_bits == 4 else 116
    cdef int64_t issue_ns = ceil_div(waves * num_mads * ceil_div(d.k, c.mad_k) * issue_cycles * 1000, hw.aic_clock_mhz)
    cdef int64_t turnovers = d.k // c.block_k * max(0, c.block_k // c.mad_k - c.num_l0_stages) + max(0, ceil_div(d.k % c.block_k, c.mad_k) - c.num_l0_stages)
    if dtype.has_sf:
        load_cycles += ceil_div((c.mad_m + c.mad_n) * c.mad_k * 2, hw.sf_elements_per_pair * hw.l1_to_l0_sf_bytes_per_cycle)
    return get_gemm_refinement_cost(d, dtype, c) + issue_ns + ceil_div(waves * num_mads * turnovers * max(mad_cycles, load_cycles) * 1000, hw.aic_clock_mhz)


cdef inline GemmConfig select_gemm_config_impl(GemmDesc d, GemmDtype dtype):
    """Select legal tiles using integer costs, pruning, and deterministic tie-breaking."""
    cdef GemmComputeCost compute_cost
    cdef GemmConfig config
    cdef GemmConfig fitted
    cdef HWDesc hw
    cdef GemmConfig wider, primary, candidate
    cdef int64_t k_limit, best_k_cost, block_k, mad_cycles, load_cycles, full_epilogue_bytes, resident_row_bytes
    cdef int64_t a_matrix_bytes, avg_block_bytes, b_elem_bits, b_matrix_bytes, best_cost, block_cap, block_m, block_n
    cdef int64_t compute_k, cost, k, m_index, n_index, m_alignment, m_end, m_start, max_block_k, min_block_k
    cdef int64_t min_first_load_time_ns, min_memory_time_ns, n_alignment, n_end, n_limit, n_start, num_batches
    cdef int64_t num_m_blocks, num_n_blocks, num_waves, store_hint, choice
    hw = get_hw_desc()
    k = d.num_groups * d.expected_k if is_k_grouped(d.gemm_type) else d.k
    compute_k = d.expected_k if is_k_grouped(d.gemm_type) else d.k
    num_batches = d.num_groups if is_batched(d.gemm_type) or is_k_grouped(d.gemm_type) else 1
    b_elem_bits = dtype.elem_bits // 2 if dtype.dequant_b else dtype.elem_bits
    a_matrix_bytes = d.m * compute_k * dtype.elem_bits // 8
    b_matrix_bytes = d.n * compute_k * b_elem_bits // 8
    block_cap = 256 if dtype.elem_bits == 4 or dtype.dequant_b else 512
    m_alignment = get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_a) else FRAC_MN
    n_alignment = 128 if dtype.with_fp8_cast else get_mad_alignment_k(dtype.elem_bits) if is_mn_major(d.major_b) else FRAC_MN
    m_start, m_end = m_alignment, min(block_cap, align(d.m, m_alignment))
    n_start, n_end = n_alignment, block_cap
    if is_m_grouped(d.gemm_type):
        m_start = m_end = MK_ALIGNMENT
        block_n = min(L0C_SIZE_BYTES // (4 * m_start), align(d.n, n_alignment))
        if dtype.dequant_b and is_mn_major(d.major_b):
            block_n = block_cap
        n_start = n_end = block_n

    min_block_k = hw.l1_to_l0_bytes_per_cycle * 8 // dtype.elem_bits
    max_block_k = 2 * min_block_k
    config = empty_config()
    best_cost = _INT64_MAX
    # Descending search establishes a pruning bound; equal costs prefer smaller M/N.
    # A constant range step is required for Cython to emit a C loop.
    for m_index in range(m_end // m_alignment, m_start // m_alignment - 1, -1):
        block_m = m_index * m_alignment
        n_limit = min(n_end, L0C_SIZE_BYTES // (4 * block_m))
        for n_index in range(ceil_div(n_limit, n_alignment), n_start // n_alignment - 1, -1):
            block_n = n_index * n_alignment
            num_n_blocks, num_m_blocks = ceil_div(d.n, block_n), ceil_div(d.m, block_m)
            num_waves = ceil_div(num_m_blocks * num_n_blocks * num_batches, d.num_cores)
            avg_block_bytes = a_matrix_bytes // num_m_blocks + b_matrix_bytes // num_n_blocks
            min_memory_time_ns = num_waves * hw.num_ai_cores * avg_block_bytes // hw.l2_bw_gbps
            if min_memory_time_ns > best_cost:
                continue
            compute_cost = get_gemm_compute_cost(d, dtype, block_m, block_n)
            min_first_load_time_ns = min_memory_time_ns * min(compute_k, min_block_k) // compute_k
            if compute_cost.time_ns + min_first_load_time_ns > best_cost:
                continue
            if not make_gemm_candidate(d, dtype, block_m, block_n, min_block_k, &fitted):
                continue
            # Prefer wider K blocks when at least three L1 stages fit.
            if k % max_block_k == 0:
                if make_gemm_candidate(d, dtype, block_m, block_n, max_block_k, &wider) and wider.num_l1_stages >= 3:
                    fitted = wider
            cost = get_gemm_cost(d, dtype, fitted, compute_cost, best_cost)
            if cost > best_cost:
                continue
            best_cost, config = cost, fitted
    if config.block_m == 0:
        return config

    num_m_blocks, num_n_blocks = ceil_div(d.m, config.block_m), ceil_div(d.n, config.block_n)
    num_waves = ceil_div(num_m_blocks * num_n_blocks, d.num_cores)
    store_hint = L2_ST_NORMAL_FV if d.acc and num_waves >= 4 and ceil_div(d.k, config.block_k) >= 16 else L2_ST_NOTALLOC_CLEAN
    config.l2_ctrl_store_cd = store_hint
    config = refine_gemm_pipeline(d, dtype, config)
    if (d.gemm_type == GemmType.Normal and d.compiled_k
        and d.k % (hw.sf_elements_per_pair if dtype.has_sf else FRAC_MN) == 0
        and (not dtype.dequant_b or d.k <= config.num_l1_stages * config.block_k)):
        primary = config
        best_k_cost = get_gemm_k_cost(d, dtype, config)
        k_limit = min(d.k, L1_SIZE_BYTES * 8 // ((primary.block_m + primary.block_n) * dtype.elem_bits))
        block_k = min_block_k
        while block_k <= k_limit:
            if block_k != primary.block_k and not (d.k <= primary.block_k and block_k < d.k):
                if make_gemm_candidate(d, dtype, primary.block_m, primary.block_n, block_k, &candidate):
                    mad_cycles = ceil_div(2 * primary.mad_m * primary.mad_n * primary.mad_k, get_mad_flops_per_cycle(dtype.elem_bits))
                    load_cycles = ceil_div(min(primary.mad_m, primary.mad_n) * primary.mad_k * dtype.elem_bits, 8 * hw.l1_to_l0_bytes_per_cycle)
                    if (candidate.mad_m == primary.mad_m and candidate.mad_n == primary.mad_n
                        and (not dtype.has_sf or candidate.num_l1_stages >= primary.num_l1_stages)
                        and (candidate.mad_k == primary.mad_k or mad_cycles <= load_cycles)):
                        candidate.l2_ctrl_store_cd = primary.l2_ctrl_store_cd
                        candidate = refine_gemm_pipeline(d, dtype, candidate)
                        cost = get_gemm_k_cost(d, dtype, candidate)
                        if cost < best_k_cost:
                            best_k_cost, config = cost, candidate
            block_k *= 2
    # Bias may need two output stages to overlap the drain with the next block.
    if d.with_bias and config.num_epilogue_stages == 1 and get_gemm_output_cost(d, dtype, config) > get_gemm_cost(d, dtype, config, get_gemm_compute_cost(d, dtype, config.block_m, config.block_n), _INT64_MAX):
        primary = config
        for choice in range(2):
            block_m = primary.block_m - m_alignment if choice == 0 else primary.block_m
            block_n = primary.block_n if choice == 0 else primary.block_n - max(n_alignment, 64)
            if (block_m < m_start or block_n < n_start
                or (block_m != primary.block_m and is_mn_major(d.major_a))
                or (block_n != primary.block_n and is_mn_major(d.major_b))):
                continue
            if not make_gemm_candidate(d, dtype, block_m, block_n, primary.block_k, &candidate):
                continue
            if candidate.num_epilogue_stages < 2 or candidate.mad_k != primary.mad_k or candidate.num_l1_stages < primary.num_l1_stages:
                continue
            candidate.l2_ctrl_store_cd = primary.l2_ctrl_store_cd
            candidate = refine_gemm_pipeline(d, dtype, candidate)
            if get_gemm_k_cost(d, dtype, candidate) < get_gemm_k_cost(d, dtype, config):
                config = candidate
    full_epilogue_bytes = (config.block_m * config.block_n * dtype.epilogue_elem_bits // 8
                           + (config.block_n * dtype.output_elem_bits // 8 if d.with_bias else 0)
                           + (config.block_m * config.block_n // 32 if dtype.with_fp8_cast else 0))
    if (dtype.elem_bits == 4 and config.block_m * config.mad_k * config.num_l0_stages * dtype.elem_bits <= L0A_SIZE_BYTES * 8
        and config.block_n * config.mad_k * config.num_l0_stages * dtype.elem_bits <= L0B_SIZE_BYTES * 8
        and full_epilogue_bytes * config.num_epilogue_stages <= UB_SIZE_BYTES):
        candidate = config
        candidate.mad_m, candidate.mad_n = config.block_m, config.block_n
        if get_gemm_k_cost(d, dtype, candidate) <= get_gemm_k_cost(d, dtype, config):
            config = candidate
    if (dtype.dequant_b and config.direct_store and not config.dual_aiv_dequant and d.gemm_type == GemmType.Normal
        and num_m_blocks == 1 and num_waves > 1 and not d.acc and config.block_m == config.mad_m
        and 2 * config.block_m * hw.fp4_decode_bytes_per_cycle <= get_mad_flops_per_cycle(dtype.elem_bits)
        and d.n * dtype.epilogue_elem_bits // 8 % hw.l2_sector_bytes == 0
        and config.block_n * dtype.epilogue_elem_bits // 8 % hw.l2_sector_bytes == 0
        and 2 * config.block_m * config.block_n * 4 <= L0C_SIZE_BYTES
        and is_k_major(d.major_a) and is_k_major(d.major_b) and d.compiled_n
        and d.compiled_k and d.k > config.block_k and d.k % config.block_k == 0
        and ceil_div(d.n, config.block_n * d.num_cores) * (d.k // config.block_k) > config.num_l1_stages):
        config.dual_aiv_dequant = True
    resident_row_bytes = ceil_div(d.k * dtype.elem_bits, 8) + ceil_div(d.k, hw.sf_elements_per_pair) * 2
    if (dtype.elem_bits == 4 and d.gemm_type == GemmType.Normal and is_mn_major(d.major_a) and is_mn_major(d.major_b)
        and config.block_n * dtype.elem_bits == hw.fractal_row_bytes * 8 and d.k <= config.block_k
        and config.l2_ctrl_b == L2_LD_NOTALLOC_KEEP and (d.m + d.n) * resident_row_bytes <= hw.l2_size_bytes):
        candidate = config
        candidate.l2_ctrl_b = L2_LD_NORMAL_FV
        if get_gemm_k_cost(d, dtype, candidate) <= get_gemm_k_cost(d, dtype, config):
            config = candidate
    return config


def select_gemm_config(desc, config_type):
    """Convert Python metadata once, run the native search, and box one result."""
    cdef GemmDesc d
    cdef GemmDtype dtype
    cdef GemmConfig c
    d = GemmDesc(
        m=desc.m,
        n=desc.n,
        k=desc.k,
        major_a=desc.major_a.value,
        major_b=desc.major_b.value,
        acc=desc.acc,
        gemm_type=desc.gemm_type.value,
        num_groups=desc.num_groups,
        expected_m=desc.expected_m,
        expected_k=desc.expected_k,
        num_cores=desc.num_cores,
        outer_stride_a=desc.outer_stride_a,
        outer_stride_b=desc.outer_stride_b,
        outer_stride_d=desc.outer_stride_d,
        output_alignment=desc.output_alignment,
        compiled_k="k" in desc.compiled_dims,
        compiled_n="n" in desc.compiled_dims,
        identity_epilogue=desc.epilogue_kind == "identity",
        with_bias=desc.epilogue_kind == "bias",
        stride_batch_a=desc.stride_batch_a,
        stride_batch_b=desc.stride_batch_b,
    )
    # The tensor API handles empty GEMMs before calling the heuristic. Keep
    # invalid direct calls away from C division, with a useful error instead.
    if d.m <= 0 or d.n <= 0 or d.k <= 0 or d.num_cores <= 0:
        raise ValueError("m, n, k, and num_cores must be positive")
    if d.outer_stride_a < 0 or d.outer_stride_b < 0:
        raise ValueError("operand strides must be nonnegative")
    if is_batched(d.gemm_type) or is_m_grouped(d.gemm_type) or is_k_grouped(d.gemm_type):
        if d.num_groups <= 0:
            raise ValueError("grouped and batched GEMMs require positive num_groups")
    if is_k_grouped(d.gemm_type) and d.expected_k <= 0:
        raise ValueError("K-grouped GEMMs require positive expected_k")

    dtype = get_gemm_dtype(desc)

    c = select_gemm_config_impl(d, dtype)
    if c.block_m == 0:
        raise AssertionError(f"No legal GEMM configuration for {desc}")
    return config_type(
        c.block_m,
        c.block_n,
        c.block_k,
        c.mad_m,
        c.mad_n,
        c.mad_k,
        c.num_l1_stages,
        c.num_l0_stages,
        c.num_epilogue_stages,
        c.num_l1_sf_stages,
        c.sf_k_blocks,
        c.l2_ctrl_a,
        c.l2_ctrl_b,
        c.l2_ctrl_store_cd,
        c.num_blocks,
        bool(c.direct_store),
        bool(c.dual_aiv_dequant),
        bool(c.resident_a),
        bool(c.resident_b),
        bool(c.resident_sf),
    )
