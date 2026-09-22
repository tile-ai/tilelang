"""Ascend (NPU) implementations for MHC norm_fn operations using TileLang."""
import math
from functools import lru_cache

import torch

import tilelang
from tilelang import language as T
from tilelang.language import simd as S
from tilelang.layout import make_ascend_compact_nz_layout

from tile_kernels.config import get_num_cube_cores, get_num_sms, get_num_vec_cores
from tile_kernels.utils import align, ceil_div

# fp32 vector register width on Ascend (2048-bit vreg / 32-bit lane).
_VEC = 64

_RMSNORM_TOKEN_BLOCK = 8
_RMSNORM_NUM_STAGES = 2

_UB_BYTES = 248 * 1024

# Column-strip alignment (a multiple of _VEC=64) so n_blk // _VEC stays whole.
_N_BLK_ALIGN = 128

# Cap rows per tile to bound unroll / register pressure.
_MAX_BLOCK_M = 32


@lru_cache(maxsize=256)
def _choose_tile(n: int, num_stages: int) -> tuple[int, int, bool]:
    n_blk = min(align(n, _N_BLK_ALIGN), 2048)
    n_align_full = align(n, _N_BLK_ALIGN)

    # Try resident normw: full strip (single buffer) + staged fn/out tiles.
    per_row = 2 * n_blk * 4 * num_stages  # fn_ub + out_ub, double buffered
    budget_res = _UB_BYTES - n_align_full * 4 - 8 * 1024
    block_m_res = budget_res // per_row
    if block_m_res >= 1:
        block_m = max(1, min(_MAX_BLOCK_M, block_m_res))
        return block_m, n_blk, True

    # Fallback: per-tile normw strip (staged), for n too large to hold resident.
    budget = _UB_BYTES - (num_stages * n_blk * 4) - 8 * 1024
    block_m = budget // per_row
    block_m = max(1, min(_MAX_BLOCK_M, block_m))
    return block_m, n_blk, False


def _cap_num_cores(num_tasks: int, max_num_cores: int) -> int:
    """Avoid launching idle core blocks when the tile count is small."""
    return max(1, min(max_num_cores, num_tasks))


@lru_cache(maxsize=256)
def _choose_fwd_num_cores(m: int, n: int, num_stages: int, max_num_cores: int) -> int:
    if n == 1:
        num_tasks = ceil_div(m, 8192)
    else:
        block_m, n_blk, _ = _choose_tile(n, num_stages)
        num_tasks = ceil_div(m, block_m) * ceil_div(n, n_blk)
    return _cap_num_cores(num_tasks, max_num_cores)


@tilelang.jit
def get_mhc_fn_normw_merge_fwd_kernel_asc(
    n: int, num_stages: int = 2, num_cores: int | None = None
):
    max_num_cores = get_num_vec_cores()
    num_cores = max_num_cores if num_cores is None else num_cores
    assert 0 < num_cores <= max_num_cores
    m = T.dynamic('m')

    if n == 1:
        m_blk = 8192  # flat elements per UB tile (VL-aligned)
        m_vregs = m_blk // _VEC

        @T.prim_func
        def _mhc_fn_normw_merge_fwd(
            fn: T.Tensor[(m, 1), T.float32],
            normw: T.Tensor[(1,), T.float32],
            out_fn: T.Tensor[(m, 1), T.float32],
        ):
            with T.Kernel(num_cores) as core_id:
                fn_ub = T.alloc_shared((m_blk,), T.float32)
                out_ub = T.alloc_shared((m_blk,), T.float32)
                w_ub = T.alloc_shared((_VEC,), T.float32)
                T.annotate_buffer_versions({fn_ub: num_stages, out_ub: num_stages, w_ub: 1})

                # Load the single scalar into lane 0 and broadcast to a vreg.
                T.copy(normw[0:1], w_ub[0:1])

                for blk in T.Persistent(
                    [T.ceildiv(m, m_blk)], num_cores, core_id, group_size=1, num_stages=num_stages
                ):
                    row0 = blk * m_blk
                    valid = T.min(m_blk, m - row0)
                    T.copy(fn[row0:row0 + valid, 0], fn_ub[:valid])
                    with T.SimdVF():
                        w_reg = S.vld(w_ub[0], dist='BRC_B32')  # broadcast normw[0] to all lanes
                        for v in range(m_vregs):
                            col = v * _VEC
                            S.vsts(out_ub[col], S.vmul(S.vld(fn_ub[col]), w_reg))
                    T.copy(out_ub[:valid], out_fn[row0:row0 + valid, 0])

        return _mhc_fn_normw_merge_fwd

    block_m, n_blk, normw_resident = _choose_tile(n, num_stages)
    n_col_tiles = ceil_div(n, n_blk)
    n_vregs = n_blk // _VEC          # literal trip count for the VF inner loop
    n_align_full = align(n, _N_BLK_ALIGN)

    if normw_resident:

        @T.prim_func
        def _mhc_fn_normw_merge_fwd(
            fn: T.Tensor[(m, n), T.float32],
            normw: T.Tensor[(n,), T.float32],
            out_fn: T.Tensor[(m, n), T.float32],
        ):
            with T.Kernel(num_cores) as core_id:
                fn_ub = T.alloc_shared((block_m, n_blk), T.float32)
                out_ub = T.alloc_shared((block_m, n_blk), T.float32)
                normw_ub = T.alloc_shared((n_align_full,), T.float32)
                T.annotate_buffer_versions({fn_ub: num_stages, out_ub: num_stages, normw_ub: 1})

                T.copy(normw[0:n], normw_ub[0:n])

                for row_blk, col_blk in T.Persistent(
                    [T.ceildiv(m, block_m), n_col_tiles], num_cores, core_id, group_size=1, num_stages=num_stages
                ):
                    row0 = row_blk * block_m
                    col0 = col_blk * n_blk
                    valid_rows = T.min(block_m, m - row0)
                    valid_cols = T.min(n_blk, n - col0)

                    T.copy(fn[row0 : row0 + valid_rows, col0 : col0 + valid_cols], fn_ub[:valid_rows, :valid_cols])

                    with T.SimdVF():
                        for i in range(block_m):
                            for v in range(n_vregs):
                                col = v * _VEC
                                fn_reg = S.vld(fn_ub[i, col])
                                w_reg = S.vld(normw_ub[col0 + col])
                                S.vsts(out_ub[i, col], S.vmul(fn_reg, w_reg))

                    T.copy(out_ub[:valid_rows, :valid_cols], out_fn[row0 : row0 + valid_rows, col0 : col0 + valid_cols])

        return _mhc_fn_normw_merge_fwd

    @T.prim_func
    def _mhc_fn_normw_merge_fwd(
        fn: T.Tensor[(m, n), T.float32],
        normw: T.Tensor[(n,), T.float32],
        out_fn: T.Tensor[(m, n), T.float32],
    ):
        with T.Kernel(num_cores) as core_id:
            fn_ub = T.alloc_shared((block_m, n_blk), T.float32)
            normw_ub = T.alloc_shared((n_blk,), T.float32)
            out_ub = T.alloc_shared((block_m, n_blk), T.float32)
            T.annotate_buffer_versions({fn_ub: num_stages, normw_ub: num_stages, out_ub: num_stages})

            for row_blk, col_blk in T.Persistent(
                [T.ceildiv(m, block_m), n_col_tiles], num_cores, core_id, group_size=1, num_stages=num_stages
            ):
                row0 = row_blk * block_m
                col0 = col_blk * n_blk
                valid_rows = T.min(block_m, m - row0)
                valid_cols = T.min(n_blk, n - col0)

                T.copy(fn[row0 : row0 + valid_rows, col0 : col0 + valid_cols], fn_ub[:valid_rows, :valid_cols])
                T.copy(normw[col0 : col0 + valid_cols], normw_ub[:valid_cols])

                with T.SimdVF():
                    for i in range(block_m):
                        for v in range(n_vregs):
                            col = v * _VEC
                            fn_reg = S.vld(fn_ub[i, col])
                            w_reg = S.vld(normw_ub[col])
                            S.vsts(out_ub[i, col], S.vmul(fn_reg, w_reg))

                T.copy(out_ub[:valid_rows, :valid_cols], out_fn[row0 : row0 + valid_rows, col0 : col0 + valid_cols])

    return _mhc_fn_normw_merge_fwd


@lru_cache(maxsize=256)
def _get_fwd_kernel(n: int, num_stages: int, num_cores: int):
    return get_mhc_fn_normw_merge_fwd_kernel_asc(
        n, num_stages=num_stages, num_cores=num_cores
    )


@lru_cache(maxsize=256)
def _choose_tile_bwd(n: int, num_stages: int, num_cores: int) -> tuple[int, int, bool]:
    occupancy_blk = align(max(1, ceil_div(n, num_cores)), _N_BLK_ALIGN)
    n_blk = min(max(_N_BLK_ALIGN, occupancy_blk), 2048)

    # Three staged 2D tiles (fn, g, fn_grad), each block_m x n_blk fp32, num_stages deep.
    per_row = 3 * n_blk * 4 * num_stages
    # Resident single-buffered strips: normw + normw_grad accumulator (carry-in/out).
    resident = 2 * n_blk * 4
    budget = _UB_BYTES - resident - 8 * 1024
    block_m = budget // per_row
    block_m = max(1, min(_MAX_BLOCK_M, block_m))
    return block_m, n_blk, True


@lru_cache(maxsize=256)
def _choose_bwd_num_cores(n: int, num_stages: int, max_num_cores: int) -> int:
    _, n_blk, _ = _choose_tile_bwd(n, num_stages, num_cores=max_num_cores)
    return _cap_num_cores(ceil_div(n, n_blk), max_num_cores)


@tilelang.jit
def get_mhc_fn_normw_merge_bwd_kernel_asc(
    n: int, num_stages: int = 2, num_cores: int | None = None
):
    max_num_cores = get_num_vec_cores()
    num_cores = max_num_cores if num_cores is None else num_cores
    assert 0 < num_cores <= max_num_cores
    m = T.dynamic('m')

    # Keep the tile shape stable across launch widths; only the core count is
    # reduced when the resulting column-tile count does not need full occupancy.
    block_m, n_blk, _ = _choose_tile_bwd(n, num_stages, num_cores=max_num_cores)
    n_col_tiles = ceil_div(n, n_blk)
    n_vregs = n_blk // _VEC

    @T.prim_func
    def _mhc_fn_normw_merge_bwd(
        fn: T.Tensor[(m, n), T.float32],
        normw: T.Tensor[(n,), T.float32],
        out_fn_grad: T.Tensor[(m, n), T.float32],
        fn_grad: T.Tensor[(m, n), T.float32],
        normw_grad: T.Tensor[(n,), T.float32],
    ):
        with T.Kernel(num_cores) as core_id:
            fn_ub = T.alloc_shared((block_m, n_blk), T.float32)
            g_ub = T.alloc_shared((block_m, n_blk), T.float32)
            fng_ub = T.alloc_shared((block_m, n_blk), T.float32)
            normw_ub = T.alloc_shared((n_blk,), T.float32)
            nwg_ub = T.alloc_shared((n_blk,), T.float32)       # normw_grad accumulator (carry-in/out)
            T.annotate_buffer_versions({
                fn_ub: num_stages, g_ub: num_stages, fng_ub: num_stages,
                normw_ub: 1, nwg_ub: 1,
            })

            for col_blk in T.Persistent(
                [n_col_tiles], num_cores, core_id, group_size=1, num_stages=num_stages
            ):
                col0 = col_blk * n_blk
                valid_cols = T.min(n_blk, n - col0)

                T.copy(normw[col0 : col0 + valid_cols], normw_ub[:valid_cols])
                T.copy(normw_grad[col0 : col0 + valid_cols], nwg_ub[:valid_cols])

                for row_blk in T.serial(T.ceildiv(m, block_m)):
                    row0 = row_blk * block_m
                    valid_rows = T.min(block_m, m - row0)

                    T.copy(fn_grad[row0 : row0 + valid_rows, col0 : col0 + valid_cols], fng_ub[:valid_rows, :valid_cols])

                    T.copy(out_fn_grad[row0 : row0 + valid_rows, col0 : col0 + valid_cols], g_ub[:valid_rows, :valid_cols])
                    T.copy(fn[row0 : row0 + valid_rows, col0 : col0 + valid_cols], fn_ub[:valid_rows, :valid_cols])

                    if valid_rows == block_m:
                        with T.SimdVF():
                            for v in T.unroll(n_vregs, explicit=True):
                                col = v * _VEC
                                w_reg = S.vld(normw_ub[col])
                                acc = S.alloc_var(T.float32)
                                acc = S.vld(nwg_ub[col])  # carry-in
                                for i in range(block_m):
                                    g_reg = S.vld(g_ub[i, col])
                                    fn_reg = S.vld(fn_ub[i, col])
                                    fng_reg = S.vld(fng_ub[i, col])
                                    S.vsts(fng_ub[i, col], S.vadd(fng_reg, S.vmul(g_reg, w_reg)))
                                    S.vmula(acc, g_reg, fn_reg)
                                S.vsts(nwg_ub[col], acc)
                    else:
                        with T.SimdVF():
                            for v in T.unroll(n_vregs, explicit=True):
                                col = v * _VEC
                                w_reg = S.vld(normw_ub[col])
                                acc = S.alloc_var(T.float32)
                                acc = S.vld(nwg_ub[col])  # carry-in
                                for i in T.serial(valid_rows):
                                    g_reg = S.vld(g_ub[i, col])
                                    fn_reg = S.vld(fn_ub[i, col])
                                    fng_reg = S.vld(fng_ub[i, col])
                                    S.vsts(fng_ub[i, col], S.vadd(fng_reg, S.vmul(g_reg, w_reg)))
                                    S.vmula(acc, g_reg, fn_reg)
                                S.vsts(nwg_ub[col], acc)

                    T.copy(fng_ub[:valid_rows, :valid_cols], fn_grad[row0 : row0 + valid_rows, col0 : col0 + valid_cols])

                # Accumulator already holds old + sum_i (carry-in seeded), store directly.
                T.copy(nwg_ub[:valid_cols], normw_grad[col0 : col0 + valid_cols])

    return _mhc_fn_normw_merge_bwd


@lru_cache(maxsize=256)
def _get_bwd_kernel(n: int, num_stages: int, num_cores: int):
    return get_mhc_fn_normw_merge_bwd_kernel_asc(
        n, num_stages=num_stages, num_cores=num_cores
    )


def mhc_fn_normw_merge_bwd_asc(
    fn,
    normw,
    out_fn_grad,
    fn_grad,
    normw_grad,
    use_pdl: bool = False,
    num_stages: int = 2,
):
    del use_pdl  # no PDL on Ascend; accepted for signature parity with CUDA.
    assert fn.dim() == 2 and out_fn_grad.shape == fn.shape and fn_grad.shape == fn.shape
    assert normw.dim() == 1 and normw.shape[0] == fn.shape[1]
    assert normw_grad.shape == normw.shape

    if fn.shape[0] == 0:
        return fn_grad, normw_grad

    n = int(fn.shape[1])
    max_num_cores = get_num_vec_cores()
    num_cores = _choose_bwd_num_cores(n, num_stages, max_num_cores)
    kernel = _get_bwd_kernel(n, num_stages, num_cores)
    kernel(fn, normw, out_fn_grad, fn_grad, normw_grad)
    return fn_grad, normw_grad


def mhc_fn_normw_merge_fwd_asc(fn, normw, out_fn, use_pdl: bool = False, num_stages: int = 2):
    del use_pdl  # no PDL on Ascend; accepted for signature parity with CUDA.
    assert fn.dim() == 2 and out_fn.shape == fn.shape
    assert normw.dim() == 1 and normw.shape[0] == fn.shape[1]

    if fn.shape[0] == 0:
        return out_fn

    n = int(fn.shape[1])
    max_num_cores = get_num_vec_cores()
    num_cores = _choose_fwd_num_cores(int(fn.shape[0]), n, num_stages, max_num_cores)
    kernel = _get_fwd_kernel(n, num_stages, num_cores)
    kernel(fn, normw, out_fn)
    return out_fn


@tilelang.jit
def mhc_reduce_partials_and_rmsnorm_fwd_asc(
    out_mul_splitted,
    sqrsum_splitted,
    out_mul,
    sqrsum,
    out,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')
    assert n_rms_group == 1, 'Ascend reduce/rmsnorm fwd currently supports one RMS group'

    out_mul_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out: T.Tensor[(num_tokens, mhc_mult3), T.float32]

    n_cores = get_num_vec_cores()

    # Preserve the original per-token SIMD kernel.  The single-partial path
    # groups 16 FP32 sqrsums into one contiguous 64B GM-to-UB transfer.
    token_block_size = 16 if n_splits == 1 else _RMSNORM_TOKEN_BLOCK

    with T.Kernel(n_cores) as core_id:
        partial_mul_ub = T.alloc_shared(
            (
                _RMSNORM_NUM_STAGES,
                token_block_size,
                n_splits,
                n_rms_group,
                _VEC,
            ),
            T.float32,
        )
        partial_sqrsum_ub = T.alloc_shared(
            (
                _RMSNORM_NUM_STAGES,
                token_block_size,
                n_splits,
                n_rms_group,
            ),
            T.float32,
        )
        out_mul_ub = T.alloc_shared(
            (token_block_size, n_rms_group, _VEC),
            T.float32,
        )
        sqrsum_ub = T.alloc_shared(
            (token_block_size, n_rms_group),
            T.float32,
        )
        out_ub = T.alloc_shared((token_block_size, _VEC), T.float32)
        T.annotate_manual_multi_buffer(partial_mul_ub, partial_sqrsum_ub)
        T.annotate_buffer_versions(
            {
                out_mul_ub: _RMSNORM_NUM_STAGES,
                sqrsum_ub: _RMSNORM_NUM_STAGES,
                out_ub: _RMSNORM_NUM_STAGES,
            }
        )

        for token_block in T.Persistent(
            [T.ceildiv(num_tokens, token_block_size)],
            n_cores,
            core_id,
            group_size=1,
            num_stages=_RMSNORM_NUM_STAGES,
            annotations={'enable_offset': True},
        ):
            token_start = token_block * token_block_size
            stage = (token_block // n_cores) % _RMSNORM_NUM_STAGES
            valid_tokens = T.min(token_block_size, num_tokens - token_start)
            if n_splits == 1:
                # The fixed 16-token view gives MTE a 64B transfer.  T.copy
                # clamps the final ragged block to the dynamic tensor extent.
                T.copy(
                    sqrsum_splitted[0, token_start : token_start + token_block_size, 0],
                    partial_sqrsum_ub[stage, 0:token_block_size, 0, 0],
                )

            for row in T.serial(valid_tokens):
                pid = token_start + row
                for k in T.serial(n_rms_group):
                    T.copy(
                        out_mul_splitted[0:n_splits, pid, k, 0:mhc_mult3],
                        partial_mul_ub[stage, row, 0:n_splits, k, 0:mhc_mult3],
                    )
                    if n_splits != 1:
                        T.copy(
                            sqrsum_splitted[0:n_splits, pid, k],
                            partial_sqrsum_ub[stage, row, 0:n_splits, k],
                        )

            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                one = S.vdup(1.0, T.float32)
                one_lane = S.pset(32, 'PAT_VL1')

                for row in range(token_block_size):
                    out_vec = S.alloc_var(T.float32)
                    out_vec = zero
                    for k in range(n_rms_group):
                        sqrsum_vec = S.alloc_var(T.float32)
                        sqrsum_vec = zero
                        for split in range(n_splits):
                            sqrsum_vec = S.vadd(
                                sqrsum_vec,
                                S.vld(
                                    partial_sqrsum_ub[stage, row, split, k],
                                    dist='BRC_B32',
                                ),
                            )
                        rms_vec = S.vdiv(
                            one,
                            S.vsqrt(
                                S.vadds(
                                    S.vmuls(sqrsum_vec, 1.0 / rms_group_size),
                                    rms_eps,
                                )
                            ),
                        )
                        S.vsts(
                            sqrsum_ub[row, k],
                            sqrsum_vec,
                            one_lane,
                            dist='ONEPT_B32',
                        )
                        out_mul_vec = S.alloc_var(T.float32)
                        out_mul_vec = zero
                        for split in range(n_splits):
                            out_mul_vec = S.vadd(
                                out_mul_vec,
                                S.vld(partial_mul_ub[stage, row, split, k, 0]),
                            )
                        S.vsts(out_mul_ub[row, k, 0], out_mul_vec)
                        out_vec = S.vadd(
                            out_vec,
                            S.vmul(out_mul_vec, rms_vec),
                        )
                    S.vsts(out_ub[row, 0], out_vec)

            T.copy(
                out_ub[0:valid_tokens, 0:mhc_mult3],
                out[token_start:token_start + valid_tokens, 0:mhc_mult3],
            )
            T.copy(
                out_mul_ub[0:valid_tokens, 0:n_rms_group, 0:mhc_mult3],
                out_mul[
                    token_start:token_start + valid_tokens,
                    0:n_rms_group,
                    0:mhc_mult3,
                ],
            )
            T.copy(
                sqrsum_ub[0:valid_tokens, 0:n_rms_group],
                sqrsum[token_start:token_start + valid_tokens, 0:n_rms_group],
            )

@tilelang.jit
def mhc_reduce_partials_and_rmsnorm_bwd_asc(
    out_grad,
    out_mul,
    sqrsum,
    out_mul_grad,
    sqrsum_grad,
    rms_group_size: int,
    rms_eps: float,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    n_rms_group = T.const('n_rms_group')

    out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32]
    out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32]
    sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32]

    n_cores = get_num_vec_cores()

    with T.Kernel(n_cores) as core_id:
        out_grad_ub = T.alloc_shared(
            (_RMSNORM_TOKEN_BLOCK, _VEC),
            T.float32,
        )
        out_mul_ub = T.alloc_shared(
            (_RMSNORM_TOKEN_BLOCK, n_rms_group, _VEC),
            T.float32,
        )
        sqrsum_ub = T.alloc_shared(
            (_RMSNORM_TOKEN_BLOCK, n_rms_group),
            T.float32,
        )
        out_mul_grad_ub = T.alloc_shared(
            (_RMSNORM_TOKEN_BLOCK, n_rms_group, _VEC),
            T.float32,
        )
        sqrsum_grad_ub = T.alloc_shared(
            (_RMSNORM_TOKEN_BLOCK, n_rms_group),
            T.float32,
        )
        T.annotate_buffer_versions(
            {
                out_grad_ub: _RMSNORM_NUM_STAGES,
                out_mul_ub: _RMSNORM_NUM_STAGES,
                sqrsum_ub: _RMSNORM_NUM_STAGES,
                out_mul_grad_ub: _RMSNORM_NUM_STAGES,
                sqrsum_grad_ub: _RMSNORM_NUM_STAGES,
            }
        )

        for token_block in T.Persistent(
            [T.ceildiv(num_tokens, _RMSNORM_TOKEN_BLOCK)],
            n_cores,
            core_id,
            group_size=1,
            num_stages=_RMSNORM_NUM_STAGES,
            annotations={'enable_offset': True},
        ):
            token_start = token_block * _RMSNORM_TOKEN_BLOCK
            valid_tokens = T.min(_RMSNORM_TOKEN_BLOCK, num_tokens - token_start)
            T.copy(
                out_grad[token_start:token_start + valid_tokens, 0:mhc_mult3],
                out_grad_ub[0:valid_tokens, 0:mhc_mult3],
            )
            T.copy(
                out_mul[
                    token_start:token_start + valid_tokens,
                    0:n_rms_group,
                    0:mhc_mult3,
                ],
                out_mul_ub[0:valid_tokens, 0:n_rms_group, 0:mhc_mult3],
            )
            T.copy(
                sqrsum[token_start:token_start + valid_tokens, 0:n_rms_group],
                sqrsum_ub[0:valid_tokens, 0:n_rms_group],
            )

            with T.SimdVF():
                one = S.vdup(1.0, T.float32)
                one_lane = S.pset(32, 'PAT_VL1')
                lane_id = S.vci(T.int32(0), T.int32)
                valid_lanes = S.vcmp(
                    lane_id,
                    S.vdup(mhc_mult3, T.int32),
                    op='lt',
                )
                for row in range(_RMSNORM_TOKEN_BLOCK):
                    out_grad_vec = S.vld(out_grad_ub[row, 0])

                    for k in range(n_rms_group):
                        sqrsum_vec = S.vld(
                            sqrsum_ub[row, k],
                            dist='BRC_B32',
                        )
                        rms_vec = S.vdiv(
                            one,
                            S.vsqrt(
                                S.vadds(
                                    S.vmuls(sqrsum_vec, 1.0 / rms_group_size),
                                    rms_eps,
                                )
                            ),
                        )
                        out_mul_vec = S.vld(out_mul_ub[row, k, 0])
                        rms_grad_vec = S.vdupv(
                            S.vcadd(
                                S.vmul(out_grad_vec, out_mul_vec, valid_lanes),
                                valid_lanes,
                            ),
                            valid_lanes,
                        )
                        sqrsum_grad_vec = S.vmuls(
                            S.vdiv(
                                S.vmul(rms_grad_vec, rms_vec),
                                S.vadds(
                                    sqrsum_vec,
                                    rms_eps * rms_group_size,
                                ),
                            ),
                            -0.5,
                        )
                        S.vsts(
                            out_mul_grad_ub[row, k, 0],
                            S.vmul(out_grad_vec, rms_vec, valid_lanes),
                            valid_lanes,
                        )
                        S.vsts(
                            sqrsum_grad_ub[row, k],
                            sqrsum_grad_vec,
                            one_lane,
                            dist='ONEPT_B32',
                        )

            T.copy(
                out_mul_grad_ub[
                    0:valid_tokens,
                    0:n_rms_group,
                    0:mhc_mult3,
                ],
                out_mul_grad[
                    token_start:token_start + valid_tokens,
                    0:n_rms_group,
                    0:mhc_mult3,
                ],
            )
            T.copy(
                sqrsum_grad_ub[0:valid_tokens, 0:n_rms_group],
                sqrsum_grad[
                    token_start:token_start + valid_tokens,
                    0:n_rms_group,
                ],
            )


_SQRSUM_VL = 64  # fp32 vector-register lane count


@tilelang.jit
def get_mhc_gemm_with_sqrsum_fwd_kernel_asc(
    num_tokens: int,
    mhc_mult3: int,
    rms_group_size: int,
    n_rms_group: int,
    split_size: int,
    hidden_block: int,
    n_splits: int,
    n_block: int,
    token_block: int = 32,
    num_stages: int = 2,
    num_cores: int | None = None,
):
    max_num_cores = get_num_cube_cores()
    num_cores = max_num_cores if num_cores is None else num_cores
    assert 0 < num_cores <= max_num_cores
    VL = _SQRSUM_VL
    assert hidden_block % VL == 0, f'hidden_block {hidden_block} must be a multiple of {VL}'
    nchunk = hidden_block // VL
    assert nchunk in (1, 2, 4), (
        f'hidden_block {hidden_block} -> nchunk {nchunk}; the square-sum tree-reduce '
        'only handles nchunk in {1,2,4} (hidden_block in {64,128,256})'
    )
    k_tiles = split_size // hidden_block
    token_tiles = ceil_div(num_tokens, token_block)
    total_blocks = token_tiles * n_rms_group * n_splits

    @T.prim_func
    def _mhc_gemm_with_sqrsum_fwd(
        x: T.Tensor[(num_tokens, rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, rms_group_size), T.float32],
        out: T.Tensor[(n_splits, num_tokens, n_rms_group * mhc_mult3), T.float32],
        sqrsum: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
    ):
        with T.MixedKernel(num_cores, sids=2) as (core_id, sid):
            # Use HF32 (round-to-nearest-even) for the FP32 Cube GEMM.
            T.set_hf32_mode("nearest_even")
            x_ub_b = T.alloc_shared((token_block // 2, hidden_block), T.bfloat16)
            x_ub_f = T.alloc_shared((token_block // 2, hidden_block), T.float32)
            x_l1 = T.alloc_l1((token_block, hidden_block), T.float32)
            fn_l1 = T.alloc_l1((n_block, hidden_block), T.float32)
            acc = T.alloc_l0c((token_block, n_block), T.float32)
            sqacc = T.alloc_shared((token_block // 2, nchunk * VL), T.float32)
            sqr_final = T.alloc_shared((token_block // 2,), T.float32)

            for block_id in T.Persistent([total_blocks], num_cores, core_id, group_size=1):
                pid_z = block_id // (token_tiles * n_rms_group)
                tmp = block_id % (token_tiles * n_rms_group)
                pid_y = tmp // token_tiles
                pid_x = tmp % token_tiles

                token_start = pid_x * token_block
                k_offset = pid_z * split_size + pid_y * rms_group_size

                # Zero the square-sum accumulator once per output tile.
                with T.SimdVF():
                    T.fill(sqacc, 0.0)

                # ---- GEMM (fp32) fused with square-sum: read x once, cast once ----
                for kt in T.Pipelined(k_tiles, num_stages=num_stages):
                    k_start = k_offset + kt * hidden_block
                    T.dual_copy(x[token_start:token_start + token_block, k_start:k_start + hidden_block], x_ub_b)
                    T.copy(fn[0:mhc_mult3, k_start:k_start + hidden_block], fn_l1[0:mhc_mult3, :])
                    with T.SimdVF():
                        for i in range(token_block // 2):
                            for jj in range(nchunk):
                                b = T.simd.vld(x_ub_b[i, jj * VL], dist='UNPK_B16')
                                f = T.simd.vcvt(b, 'float32', part=0)
                                T.simd.vsts(x_ub_f[i, jj * VL], f)
                                sq = T.simd.vmul(f, f)
                                prev = T.simd.vld(sqacc[i, jj * VL])
                                T.simd.vsts(sqacc[i, jj * VL], T.simd.vadd(prev, sq))
                    T.dual_copy(x_ub_f, x_l1)
                    T.gemm(
                        x_l1,
                        fn_l1,
                        acc,
                        transpose_B=True,
                        clear_accum=(kt == 0),
                        unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2),
                    )
                # Consume the final GEMM unit flag while committing the L0C tile.
                T.copy(
                    acc[:, 0:mhc_mult3],
                    out[pid_z, token_start:token_start + token_block,
                        pid_y * mhc_mult3:(pid_y + 1) * mhc_mult3],
                    unit_flag_ctrl=3,
                )
                # fold the nchunk VL-slices and horizontally reduce -> per-row scalar
                with T.SimdVF():
                    full = T.simd.pset(32, 'PAT_ALL')
                    one = T.simd.pset(32, 'PAT_VL1')
                    for i in range(token_block // 2):
                        # tree-reduce nchunk (1,2,4) VL-slices with distinct SSA values,
                        # then horizontal reduce -> per-row scalar (single ONEPT store).
                        if nchunk == 1:
                            red = T.simd.vld(sqacc[i, 0])
                        elif nchunk == 2:
                            red = T.simd.vadd(T.simd.vld(sqacc[i, 0]), T.simd.vld(sqacc[i, VL]))
                        else:  # nchunk == 4
                            s01 = T.simd.vadd(T.simd.vld(sqacc[i, 0]), T.simd.vld(sqacc[i, VL]))
                            s23 = T.simd.vadd(T.simd.vld(sqacc[i, 2 * VL]), T.simd.vld(sqacc[i, 3 * VL]))
                            red = T.simd.vadd(s01, s23)
                        T.simd.vsts(sqr_final[i], T.simd.vcadd(red, full), one, 'ONEPT_B32')
                half = token_block // 2
                T.copy(sqr_final, sqrsum[pid_z, token_start + sid * half : token_start + sid * half + half, pid_y])

    return _mhc_gemm_with_sqrsum_fwd


@lru_cache(maxsize=256)
def _choose_gemm_num_cores(
    num_tokens: int,
    n_rms_group: int,
    n_splits: int,
    token_block: int,
    max_num_cores: int,
) -> int:
    total_blocks = ceil_div(num_tokens, token_block) * n_rms_group * n_splits
    return _cap_num_cores(total_blocks, max_num_cores)


@lru_cache(maxsize=256)
def _get_gemm_with_sqrsum_kernel(
    num_tokens: int,
    mhc_mult3: int,
    rms_group_size: int,
    n_rms_group: int,
    split_size: int,
    hidden_block: int,
    n_splits: int,
    n_block: int,
    token_block: int,
    num_stages: int,
    num_cores: int,
):
    return get_mhc_gemm_with_sqrsum_fwd_kernel_asc(
        num_tokens=num_tokens,
        mhc_mult3=mhc_mult3,
        rms_group_size=rms_group_size,
        n_rms_group=n_rms_group,
        split_size=split_size,
        hidden_block=hidden_block,
        n_splits=n_splits,
        n_block=n_block,
        token_block=token_block,
        num_stages=num_stages,
        num_cores=num_cores,
    )


def _mhc_gemm_with_sqrsum_fwd_asc_legacy(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    split_size: int,
    hidden_block: int,
    token_block: int = 32,
    n_splits: int = 1,
    use_pdl: bool = False,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    del use_pdl
    assert x.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert out.dtype == torch.float32
    assert sqrsum.dtype == torch.float32

    num_tokens, rms_group_size = int(x.shape[0]), int(x.shape[1])
    mhc_mult3 = int(fn.shape[0])
    n_rms_group = int(out.shape[2])
    n_block = max(align(mhc_mult3, 16), 16)

    if num_tokens == 0:
        return out, sqrsum

    max_num_cores = get_num_cube_cores()
    num_cores = _choose_gemm_num_cores(
        num_tokens,
        n_rms_group,
        n_splits,
        token_block,
        max_num_cores,
    )

    kernel = _get_gemm_with_sqrsum_kernel(
        num_tokens=num_tokens,
        mhc_mult3=mhc_mult3,
        rms_group_size=rms_group_size,
        n_rms_group=n_rms_group,
        split_size=split_size,
        hidden_block=hidden_block,
        n_splits=n_splits,
        n_block=n_block,
        token_block=token_block,
        num_stages=num_stages,
        num_cores=num_cores,
    )

    assert fn.is_contiguous() and fn.dtype == torch.float32

    out_3d = out.reshape(n_splits, num_tokens, n_rms_group * mhc_mult3)
    kernel(x, fn, out_3d, sqrsum)
    return out, sqrsum

@tilelang.jit
def mhc_gemm_with_sqrsum_fwd_asc(
    x,
    fn,
    out_mul,
    sqrsum,
    block_mhc_mult3: int,
    num_aic_cores: int,
    num_k_splits: int,
    cores_per_split: int,
    num_hidden_blocks_per_split: int,
    num_l0b_stages: int,
    num_l0c_stages: int,
    num_token_blocks_per_chunk: int,
    deterministic: bool = False,
):
    block_num_tokens = 128
    block_hidden = 256
    mad_hidden = 64
    num_aivs = 2
    half_block_num_tokens = block_num_tokens // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    num_load_stages = 3
    num_cast_stages = 2
    num_l1_x_stages = 3
    num_mad_stages = 2

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')

    x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    out_mul: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens,), T.float32]

    num_store_phases = num_k_splits if deterministic else (2 if num_k_splits > 1 else 1)
    # In non-deterministic mode, phase 0 separates split 0's non-atomic store
    # from the remaining splits' atomic adds. The last phase only needs a
    # barrier when another token chunk follows. Encode that optional barrier
    # as a uniform 0/1-trip loop because AutoSchedule rejects inter-core waits
    # under conditional control flow inside T.PerCoreTask.
    num_always_synced_store_phases = num_store_phases if deterministic else (1 if num_k_splits > 1 else 0)
    num_chunk_end_syncs = 0 if deterministic or num_k_splits == 1 else 1
    nz_stage_rows = half_block_num_tokens + 1

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        num_token_blocks = T.ceildiv(num_tokens, block_num_tokens)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)

        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        sqrsum_ub = T.alloc_shared((num_l0c_stages, half_block_num_tokens), T.float32)

        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3), T.float32)

        T.annotate_buffer_versions(
            {
                x_ub: num_load_stages,
                x_nz_ub: num_cast_stages,
                x_l1: num_l1_x_stages,
                fn_l1: num_l0b_stages,
                x_l0: num_mad_stages,
                fn_l0: num_l0b_stages,
            }
        )

        if split_id != 0:
            T.set_atomic('add', 'float32')
        T.set_hf32_mode('nearest_even')
        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )
            aiv_token_begin_base = token_block_base * block_num_tokens + aiv_id * half_block_num_tokens
            num_valid_sqrsum_stages = T.min(
                num_l0c_stages,
                T.max(
                    T.ceildiv(
                        num_tokens - aiv_token_begin_base,
                        cores_per_split * block_num_tokens,
                    ),
                    0,
                ),
            )

            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                global_hidden_block = split_id * num_hidden_blocks_per_split + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden

                # Keep physical buffers block_mhc_mult3-aligned, but restrict
                # data movement and MAD regions to the logical mhc_mult3.
                T.copy(
                    fn[0:mhc_mult3, hidden_begin : hidden_begin + block_hidden],
                    fn_l1[0:mhc_mult3, :],
                )
                for hidden_mad in T.Serial(num_mads_per_block):
                    T.copy(
                        fn_l1[0:mhc_mult3, hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden],
                        fn_l0[hidden_mad, 0:mhc_mult3, :],
                    )

                for token_stage in T.Serial(
                    num_valid_token_stages,
                    annotations={'multi_buffer_eligible': [x_l1]},
                ):
                    token_block = token_block_base + token_stage * cores_per_split
                    token_begin = token_block * block_num_tokens
                    actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                    aiv_token_begin = token_begin + aiv_id * half_block_num_tokens

                    if aiv_token_begin < num_tokens:
                        actual_aiv_num_tokens = T.min(
                            num_tokens - aiv_token_begin,
                            half_block_num_tokens,
                        )
                        T.copy(
                            x[
                                aiv_token_begin : aiv_token_begin + actual_aiv_num_tokens,
                                hidden_begin : hidden_begin + block_hidden,
                            ],
                            x_ub[0:actual_aiv_num_tokens, :],
                            l2_cache_ctrl='notalloc_keep',
                            pad_value=0,
                        )

                        # Widen one AIV half-tile, pack padded NZ, and update row sums.
                        with T.SimdVF(latency=640):
                            bf16_mask = T.simd.pset(16)
                            f32_mask = T.simd.pset(32)
                            nz_stride = T.int32((nz_stage_rows << 16) | (8 * nz_stage_rows))
                            zero_bf16 = T.simd.vdup(T.bfloat16(0), 'bfloat16', bf16_mask)
                            row_indices = T.simd.vci(T.float32(0), 'float32')
                            current_row = T.simd.alloc_var('float32')
                            current_row = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                            one = T.simd.vdup(T.float32(1), 'float32', f32_mask)
                            sqr_sums = T.simd.alloc_var('float32')
                            sqr_sums = T.simd.vdup(T.float32(0), 'float32', f32_mask)

                            for row in T.Serial(half_block_num_tokens):
                                row_acc = T.simd.alloc_var('float32')
                                row_acc = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                                nz_ptr = T.simd.make_ubuf_ptr(
                                    T.access_ptr(
                                        x_nz_ub[row, 0],
                                        'w',
                                        1,
                                        block_hidden,
                                    ),
                                    'float32',
                                )

                                for pass_id in T.Serial(block_hidden // 128):
                                    packed = T.simd.vld(x_ub[row, pass_id * 128])
                                    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                    lo = T.reinterpret(lo_bits, 'float32x64')
                                    hi = T.reinterpret(hi_bits, 'float32x64')

                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(lo, lo, f32_mask),
                                        f32_mask,
                                    )
                                    row_acc = T.simd.vadd(
                                        row_acc,
                                        T.simd.vmul(hi, hi, f32_mask),
                                        f32_mask,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        lo,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )
                                    nz_ptr = T.simd.vsstb(
                                        hi,
                                        nz_ptr,
                                        nz_stride,
                                        f32_mask,
                                        update=True,
                                    )

                                row_sum = T.simd.alloc_var('float32')
                                row_sum = T.simd.vcadd(row_acc, f32_mask)
                                row_sum_broadcast = T.simd.vdupv(row_sum, f32_mask)
                                row_mask = T.simd.vcmp(
                                    row_indices,
                                    current_row,
                                    f32_mask,
                                    'eq',
                                )
                                sqr_sums = T.simd.vsel(row_sum_broadcast, sqr_sums, row_mask)
                                current_row = T.simd.vadd(current_row, one, f32_mask)

                            if local_hidden_block != 0:
                                previous = T.simd.vld(sqrsum_ub[token_stage, 0])
                                sqr_sums = T.simd.vadd(sqr_sums, previous, f32_mask)
                            T.simd.vsts(sqrsum_ub[token_stage, 0], sqr_sums, f32_mask)

                        T.dual_copy(
                            x_nz_ub[0:half_block_num_tokens, 0:block_hidden],
                            x_l1[0:block_num_tokens, 0:block_hidden],
                        )

                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(
                            x_l1[
                                0:actual_num_tokens,
                                hidden_mad * mad_hidden : (hidden_mad + 1) * mad_hidden,
                            ],
                            x_l0[0:actual_num_tokens, :],
                        )
                        T.gemm(
                            x_l0[0:actual_num_tokens, :],
                            fn_l0[hidden_mad, 0:mhc_mult3, :],
                            out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                            transpose_B=True,
                            clear_accum=(local_hidden_block == 0 and hidden_mad == 0),
                            unit_flag_ctrl=T.Select(
                                local_hidden_block == num_hidden_blocks_per_split - 1 and hidden_mad == num_mads_per_block - 1,
                                3,
                                2,
                            ),
                        )

            with T.PerCoreTask():
                for store_phase in T.Serial(num_store_phases):
                    store_this_phase = T.if_then_else(
                        deterministic,
                        split_id == store_phase,
                        T.if_then_else(store_phase == 0, split_id == 0, split_id != 0),
                    )
                    out_mul_flag = (chunk_idx * num_store_phases + store_phase) % 4

                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_token_stages):
                                token_block = token_block_base + token_stage * cores_per_split
                                token_begin = token_block * block_num_tokens
                                actual_num_tokens = T.min(num_tokens - token_begin, block_num_tokens)
                                T.copy(
                                    out_mul_l0[token_stage, 0:actual_num_tokens, 0:mhc_mult3],
                                    out_mul[
                                        token_begin : token_begin + actual_num_tokens,
                                        0:mhc_mult3,
                                    ],
                                    unit_flag_ctrl=3,
                                )
                    num_store_syncs = T.if_then_else(
                        store_phase < num_always_synced_store_phases,
                        1,
                        T.min(num_token_chunks - chunk_idx - 1, num_chunk_end_syncs),
                    )
                    for _ in T.Serial(num_store_syncs):
                        T.ascend_sync_inter_arrive('PIPE_FIX', out_mul_flag)
                        T.ascend_sync_inter_wait('PIPE_FIX', out_mul_flag)

            with T.PerCoreTask():
                for store_phase in T.Serial(num_store_phases):
                    store_this_phase = T.if_then_else(
                        deterministic,
                        split_id == store_phase,
                        T.if_then_else(store_phase == 0, split_id == 0, split_id != 0),
                    )
                    sqrsum_flag = 4 + (chunk_idx * num_store_phases + store_phase) % 4

                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_sqrsum_stages):
                                token_block = token_block_base + token_stage * cores_per_split
                                token_begin = token_block * block_num_tokens + aiv_id * half_block_num_tokens
                                actual_aiv_num_tokens = T.min(
                                    num_tokens - token_begin,
                                    half_block_num_tokens,
                                )
                                T.copy(
                                    sqrsum_ub[token_stage, 0:actual_aiv_num_tokens],
                                    sqrsum[token_begin : token_begin + actual_aiv_num_tokens],
                                    l2_cache_ctrl='normal_fv',
                                )
                    num_store_syncs = T.if_then_else(
                        store_phase < num_always_synced_store_phases,
                        1,
                        T.min(num_token_chunks - chunk_idx - 1, num_chunk_end_syncs),
                    )
                    for _ in T.Serial(num_store_syncs):
                        T.ascend_sync_inter_arrive('PIPE_MTE3', sqrsum_flag)
                        T.ascend_sync_inter_wait('PIPE_MTE3', sqrsum_flag)

        if split_id != 0:
            T.set_atomic_none()

@tilelang.jit
def get_mhc_gemm_with_sqrsum_bwd_kernel_asc(
    num_tokens: int,
    mhc_mult3: int,
    rms_group_size: int,
    n_rms_group: int,
    token_block: int,
    hidden_block: int,
    grad_block: int,
    token_sub_block: int = 64,
    num_stages: int = 2,
    num_cores: int | None = None,
):
    """Build the single-kernel backward for GEMM plus square-sum.

    The task list contains both kinds of independent work.  x-gradient tasks
    are laid out over token/group/hidden tiles to keep the Cube occupied, while
    one fn-gradient task owns each group/hidden tile and accumulates all token
    tiles into its L0C accumulator.  Keeping the two paths in one
    ``MixedKernel`` launch avoids the synchronization and launch overhead of
    the two-kernel PR implementation.
    """
    max_num_cores = get_num_cube_cores()
    num_cores = max_num_cores if num_cores is None else num_cores
    assert 0 < num_cores <= max_num_cores
    assert token_block % 2 == 0
    assert token_block % token_sub_block == 0
    assert hidden_block % _VEC == 0
    assert num_tokens % token_block == 0
    assert rms_group_size % hidden_block == 0

    token_tiles = num_tokens // token_block
    hidden_tiles = rms_group_size // hidden_block
    x_tasks = token_tiles * n_rms_group * hidden_tiles
    fn_tasks = n_rms_group * hidden_tiles
    total_tasks = x_tasks + fn_tasks
    half_tokens = token_block // 2
    hidden_chunks = hidden_block // _VEC
    token_sub_tiles = token_block // token_sub_block

    @T.prim_func
    def mhc_gemm_with_sqrsum_bwd_asc_fused(
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group * mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
        x: T.Tensor[(num_tokens, rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, rms_group_size), T.float32],
        x_grad: T.Tensor[(num_tokens, rms_group_size), T.bfloat16],
        fn_grad: T.Tensor[(mhc_mult3, rms_group_size), T.float32],
    ):
        with T.MixedKernel(num_cores, sids=2) as (core_id, sid):
            # Use HF32 (round-to-nearest-even) for both FP32 backward GEMMs.
            T.set_hf32_mode("nearest_even")
            # Cube resources for x_grad = out_mul_grad @ fn and fn_grad =
            # out_mul_grad^T @ x.  They are allocated together, but each task
            # uses only one accumulator, so the two paths remain independent.
            grad_l1 = T.alloc_l1((token_block, grad_block), T.float32)
            fn_l1 = T.alloc_l1((grad_block, hidden_block), T.float32)
            x_l1 = T.alloc_l1((token_block, hidden_block), T.float32)
            grad_l0 = T.alloc_l0a((token_block, grad_block), T.float32)
            fn_grad_l0 = T.alloc_l0a((token_sub_block, grad_block), T.float32)
            fn_l0 = T.alloc_l0b((grad_block, hidden_block), T.float32)
            x_l0 = T.alloc_l0b((token_sub_block, hidden_block), T.float32)
            x_acc = T.alloc_l0c((token_block, hidden_block), T.float32)
            fn_acc = T.alloc_l0c((grad_block, hidden_block), T.float32)

            # UB resources are shared by the two task paths.  dual_copy splits
            # the token dimension between the two vector sub-cores.
            gemm_ub = T.alloc_shared((half_tokens, hidden_block), T.float32)
            x_ub_b = T.alloc_shared((half_tokens, hidden_block), T.bfloat16)
            x_ub_f = T.alloc_shared((half_tokens, hidden_block), T.float32)
            x_grad_ub = T.alloc_shared((half_tokens, hidden_block), T.bfloat16)
            sqrsum_grad_ub = T.alloc_shared((half_tokens, n_rms_group), T.float32)

            for block_id in T.Persistent(
                [total_tasks], num_cores, core_id, group_size=1
            ):
                if block_id < x_tasks:
                    # ------------------------- x gradient -----------------
                    x_pid_z = block_id % hidden_tiles
                    x_tmp = block_id // hidden_tiles
                    x_pid_y = x_tmp % n_rms_group
                    x_pid_x = x_tmp // n_rms_group

                    x_token_start = x_pid_x * token_block
                    x_hidden_start = x_pid_y * rms_group_size + x_pid_z * hidden_block

                    T.copy(
                        out_mul_grad[
                            x_token_start:x_token_start + token_block,
                            x_pid_y * mhc_mult3:
                            x_pid_y * mhc_mult3 + grad_block,
                        ],
                        grad_l1,
                    )
                    T.copy(
                        fn[0:grad_block, x_hidden_start:x_hidden_start + hidden_block],
                        fn_l1,
                    )
                    T.copy(grad_l1, grad_l0)
                    T.copy(fn_l1, fn_l0)
                    # Both operands are in L0.  For this L0 NN GEMM,
                    # transpose_B=False is encoded by the MN-major L0B
                    # layout and the L1->L0B copy; it does not enter the
                    # direct L1 GEMM path, which is NT-only on Ascend.
                    T.gemm(
                        grad_l0,
                        fn_l0,
                        x_acc,
                        transpose_A=False,
                        transpose_B=False,
                        clear_accum=True,
                        unit_flag_ctrl=3,
                    )

                    # The vector half receives all inputs in one dual-copy
                    # sequence, then performs the bf16->fp32 correction before
                    # packing the result back to bf16.
                    T.dual_copy(x_acc, gemm_ub, unit_flag_ctrl=3)
                    T.dual_copy(
                        x[
                            x_token_start:x_token_start + token_block,
                            x_hidden_start:x_hidden_start + hidden_block,
                        ],
                        x_ub_b,
                    )
                    T.dual_copy(
                        x_grad[
                            x_token_start:x_token_start + token_block,
                            x_hidden_start:x_hidden_start + hidden_block,
                        ],
                        x_grad_ub,
                    )
                    T.dual_copy(
                        sqrsum_grad[x_token_start:x_token_start + token_block, :],
                        sqrsum_grad_ub,
                    )

                    with T.SimdVF():
                        for i in range(half_tokens):
                            sq_grad = S.vld(sqrsum_grad_ub[i, x_pid_y], dist="BRC_B32")
                            twice_sq_grad = S.vmuls(sq_grad, 2.0)
                            for chunk in range(hidden_chunks):
                                col = chunk * _VEC
                                x_f32 = S.vcvt(
                                    S.vld(x_ub_b[i, col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                old_grad_f32 = S.vcvt(
                                    S.vld(x_grad_ub[i, col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                grad_f32 = S.vadd(
                                    S.vadd(S.vld(gemm_ub[i, col]), old_grad_f32),
                                    S.vmul(x_f32, twice_sq_grad),
                                )
                                S.vsts(
                                    x_grad_ub[i, col],
                                    S.vcvt(grad_f32, T.bfloat16),
                                    dist="PK_B32",
                                )

                    T.copy(
                        x_grad_ub,
                        x_grad[
                            x_token_start + sid * half_tokens:
                            x_token_start + (sid + 1) * half_tokens,
                            x_hidden_start:x_hidden_start + hidden_block,
                        ],
                    )
                else:
                    # ------------------------- fn gradient -----------------
                    fn_block = block_id - x_tasks
                    fn_pid_y = fn_block // hidden_tiles
                    fn_pid_z = fn_block % hidden_tiles
                    fn_hidden_start = fn_pid_y * rms_group_size + fn_pid_z * hidden_block

                    for token_tile in T.serial(token_tiles):
                        fn_token_start = token_tile * token_block
                        T.copy(
                            out_mul_grad[
                                fn_token_start:fn_token_start + token_block,
                                fn_pid_y * mhc_mult3:
                                fn_pid_y * mhc_mult3 + grad_block,
                            ],
                            grad_l1,
                        )
                        T.dual_copy(
                            x[
                                fn_token_start:fn_token_start + token_block,
                                fn_hidden_start:fn_hidden_start + hidden_block,
                            ],
                            x_ub_b,
                        )
                        with T.SimdVF():
                            for i in range(half_tokens):
                                for chunk in range(hidden_chunks):
                                    col = chunk * _VEC
                                    x_f32 = S.vcvt(
                                        S.vld(x_ub_b[i, col], dist="UNPK_B16"),
                                        T.float32,
                                        part=0,
                                    )
                                    S.vsts(x_ub_f[i, col], x_f32)
                        T.dual_copy(x_ub_f, x_l1)

                        for sub_tile in T.serial(token_sub_tiles):
                            sub_start = sub_tile * token_sub_block
                            T.copy(
                                grad_l1[sub_start:sub_start + token_sub_block, :],
                                fn_grad_l0,
                            )
                            T.copy(
                                x_l1[sub_start:sub_start + token_sub_block, :],
                                x_l0,
                            )
                            # This is the L0 TN form (A^T @ B).  The B
                            # operand is again physically prepared as
                            # MN-major, so FP32 transpose_B=False is handled
                            # by layout/copy lowering rather than L1 GEMM.
                            T.gemm(
                                fn_grad_l0,
                                x_l0,
                                fn_acc,
                                transpose_A=True,
                                transpose_B=False,
                                clear_accum=(token_tile == 0 and sub_tile == 0),
                            )

                    T.copy(
                        fn_acc[0:mhc_mult3, :],
                        fn_grad[0:mhc_mult3, fn_hidden_start:fn_hidden_start + hidden_block],
                    )

    return mhc_gemm_with_sqrsum_bwd_asc_fused


@lru_cache(maxsize=256)
def _get_gemm_with_sqrsum_bwd_kernel(
    num_tokens: int,
    mhc_mult3: int,
    rms_group_size: int,
    n_rms_group: int,
    token_block: int,
    hidden_block: int,
    grad_block: int,
    num_stages: int,
    num_cores: int,
):
    return get_mhc_gemm_with_sqrsum_bwd_kernel_asc(
        num_tokens=num_tokens,
        mhc_mult3=mhc_mult3,
        rms_group_size=rms_group_size,
        n_rms_group=n_rms_group,
        token_block=token_block,
        hidden_block=hidden_block,
        grad_block=grad_block,
        num_stages=num_stages,
        num_cores=num_cores,
    )


def mhc_gemm_with_sqrsum_bwd_asc(
    out_mul_grad: torch.Tensor,
    sqrsum_grad: torch.Tensor,
    x: torch.Tensor,
    fn: torch.Tensor,
    x_grad: torch.Tensor,
    fn_grad: torch.Tensor,
    token_block: int = 128,
    hidden_block: int = 128,
    use_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused Ascend backward in one Cube+SIMD kernel launch."""
    del use_pdl  # PDL is not used by the Ascend MixedKernel implementation.
    assert out_mul_grad.dtype == torch.float32
    assert sqrsum_grad.dtype == torch.float32
    assert x.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert x_grad.dtype == torch.bfloat16
    assert fn_grad.dtype == torch.float32
    assert out_mul_grad.dim() == 3
    assert sqrsum_grad.dim() == 2
    assert x.dim() == 2 and fn.dim() == 2
    assert x_grad.shape == x.shape
    assert fn_grad.shape == fn.shape

    num_tokens = int(x.shape[0])
    rms_group_size = int(x.shape[1])
    n_rms_group = int(out_mul_grad.shape[1])
    mhc_mult3 = int(out_mul_grad.shape[2])
    grad_block = max(align(mhc_mult3, 16), 16)

    assert int(out_mul_grad.shape[0]) == num_tokens
    assert int(fn.shape[0]) == mhc_mult3
    assert int(fn.shape[1]) == rms_group_size
    assert tuple(sqrsum_grad.shape) == (num_tokens, n_rms_group)
    assert token_block % 2 == 0
    assert token_block % 64 == 0
    assert hidden_block % _VEC == 0
    assert num_tokens % token_block == 0, (
        f"num_tokens {num_tokens} must be divisible by token_block {token_block}"
    )
    assert rms_group_size % hidden_block == 0, (
        f"rms_group_size {rms_group_size} must be divisible by hidden_block {hidden_block}"
    )

    if num_tokens == 0:
        return x_grad, fn_grad

    token_tiles = num_tokens // token_block
    hidden_tiles = rms_group_size // hidden_block
    total_tasks = (token_tiles + 1) * n_rms_group * hidden_tiles
    max_num_cores = get_num_cube_cores()
    num_cores = _cap_num_cores(total_tasks, max_num_cores)

    kernel = _get_gemm_with_sqrsum_bwd_kernel(
        num_tokens=num_tokens,
        mhc_mult3=mhc_mult3,
        rms_group_size=rms_group_size,
        n_rms_group=n_rms_group,
        token_block=token_block,
        hidden_block=hidden_block,
        grad_block=grad_block,
        num_stages=2,
        num_cores=num_cores,
    )
    # Keep the external tensors unpadded.  The kernel requests grad_block-sized
    # GM->L1 tiles; Ascend OOB lowering clamps the valid mhc_mult3 rows and
    # emits zero fills for the internal Cube tail.
    out_mul_grad_2d = out_mul_grad.reshape(num_tokens, n_rms_group * mhc_mult3)
    kernel(
        out_mul_grad_2d,
        sqrsum_grad,
        x,
        fn,
        x_grad,
        fn_grad,
    )
    return x_grad, fn_grad
