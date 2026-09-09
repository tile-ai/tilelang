import re

import pytest
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.backend.target import determine_target
from tilelang.engine.lower import lower
from tvm import tirx

# Most sibling-owner fixtures in this file use explicit claims; the dedicated
# auto-annotation cases intentionally omit them.

LATER_BREAK_REASON = "later loop_break support: clean up synchronization state and advance counters on early exits"


def _make_guarded_program(mode="auto", fixed_versions=True):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            if fixed_versions:
                T.annotate_buffer_versions({ub: (2, mode)})
            else:
                T.annotate_buffer_versions({ub: mode})
            for w in T.Pipelined(8, num_stages=2):
                if enabled > 0:
                    T.copy(A[w * tile : (w + 1) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[w * tile : (w + 1) * tile])

    return main


def _make_counter_flag_shrink_program():
    tile = 64

    @T.macro
    def protocol(A, C, ub, offset):
        T.copy(A[offset : offset + tile], ub)
        with T.SimdVF():
            mask = T.simd.pset(32)
            value = T.simd.vld(ub[0])
            T.simd.vsts(ub[0], value, mask)
        T.copy(ub, C[offset : offset + tile])

    @T.prim_func
    def main(A: T.Buffer((16 * tile,), "float32"), C: T.Buffer((16 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for w in T.Pipelined(8, num_stages=2):
                base = w * 2 * tile
                protocol(A, C, ub, base)
                protocol(A, C, ub, base + tile)

    return main


def _make_cross_level_sync_program(mode, inner=1):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode)})
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                for _inner in T.serial(inner):
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_sibling_cross_level_sync_program(mode, inner=1):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode)})
            for i in T.Pipelined(
                2,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                for _inner_i in T.serial(inner):
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[i * tile : (i + 1) * tile])
            for j in T.Pipelined(
                2,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                offset = (j + 2) * tile
                T.copy(A[offset : offset + tile], ub)
                for _inner_j in T.serial(inner):
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_mixed_distance_sibling_cross_level_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((2 * tile,), "float32"), C: T.Buffer((6 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(
                2,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                for inner in T.serial(2):
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    offset = (i * 2 + inner) * tile
                    T.copy(ub, C[offset : offset + tile])
            for j in T.Pipelined(
                2,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                for _once in T.serial(1):
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    offset = (j + 4) * tile
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_mutating_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred: T.Buffer((4,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                if pred[i] > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    pred[i] = 0
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_mutable_bind_scoped_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.Buffer((2,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            outer_guard = T.bind(pred[0] > 0)
            access_guard = T.bind(pred[1] > 0)
            pred[0] = 0
            pred[1] = 0
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                if outer_guard:
                    if access_guard:
                        T.copy(A[i * tile : (i + 1) * tile], ub)
                    for repeat in T.serial(2):
                        if access_guard:
                            offset = (i * 2 + repeat) * tile
                            T.copy(ub, C[offset : offset + tile])

    return main


def _make_mutable_bind_nested_extent_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.Buffer((1,), "int32"),
        extent: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            seed_guard = T.bind(pred[0] > 0)
            repeat_count = T.bind(extent[0])
            pred[0] = 0
            extent[0] = 0
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                if seed_guard:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                for repeat in T.serial(repeat_count):
                    offset = (i * 2 + repeat) * tile
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_versioned_storage_control_program(explicit_claim=True):
    @T.prim_func
    def main(C: T.Buffer((2,), "int32")):
        with T.Kernel(1):
            ub = T.alloc_shared((2,), "int32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(
                2,
                annotations={"multi_buffer_eligible": [ub]} if explicit_claim else {},
            ):
                ub[0] = i + 1
                for j in T.serial(ub[0]):
                    C[i] = j

    return main


def _make_versioned_storage_condition_guard_program():
    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                ub[0] = A[i]
                if ub[0] > 0:
                    C[i] = ub[0]

    return main


def _make_sibling_loop_program(inner=3, mode=None):
    tile = 64
    groups = 2

    @T.prim_func
    def main(
        A: T.Buffer((2 * groups * inner * tile,), "float32"),
        C: T.Buffer((2 * groups * inner * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for outer in T.Pipelined(2, num_stages=2):
                for group in T.Unroll(groups, explicit=True):
                    for inner_idx in T.serial(inner, annotations={"multi_buffer_eligible": [ub]}):
                        offset = ((outer * groups + group) * inner + inner_idx) * tile
                        T.copy(A[offset : offset + tile], ub)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)
                        T.copy(ub, C[offset : offset + tile])

    return main


def _make_mismatched_sibling_version_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: "counter"})
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                T.copy(ub, C[i * tile : (i + 1) * tile])
            for j in T.Pipelined(4, num_stages=3, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[(j + 4) * tile : (j + 5) * tile], ub)
                T.copy(ub, C[(j + 4) * tile : (j + 5) * tile])

    return main


def _make_single_version_counter_mode_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (1, "counter")})
            for i in T.Pipelined(4, num_stages=2):
                if enabled > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_bf16_mnk_program(k_tiles=4, mode=None):
    block_m = 32
    block_n = 32
    mad_m = 16
    mad_n = 16
    mad_k = 16
    block_k = k_tiles * mad_k

    @T.prim_func
    def main(
        A: T.Buffer((2 * block_m, block_k), "bfloat16"),
        B: T.Buffer((2 * block_n, block_k), "bfloat16"),
        C: T.Buffer((2, block_m, block_n), "float32"),
    ):
        with T.Kernel(1):
            l1a = T.alloc_l1((block_m, block_k), "bfloat16")
            l1b = T.alloc_l1((block_n, block_k), "bfloat16")
            l0a = T.alloc_l0a((mad_m, mad_k), "bfloat16")
            l0b = T.alloc_l0b((mad_n, mad_k), "bfloat16")
            l0c = T.alloc_l0c((2, 2, mad_m, mad_n), "float32")
            T.annotate_buffer_versions(
                {
                    l0a: (2, mode) if mode is not None else 2,
                    l0b: (2, mode) if mode is not None else 2,
                }
            )
            for block in T.Pipelined(2, num_stages=2):
                T.copy(A[block * block_m : (block + 1) * block_m, :], l1a)
                T.copy(B[block * block_n : (block + 1) * block_n, :], l1b)
                for m_tile in T.Unroll(2, explicit=True):
                    for n_tile in T.Unroll(2, explicit=True):
                        for k_tile in T.Pipelined(
                            k_tiles,
                            num_stages=2,
                        ):
                            T.copy(
                                l1a[
                                    m_tile * mad_m : (m_tile + 1) * mad_m,
                                    k_tile * mad_k : (k_tile + 1) * mad_k,
                                ],
                                l0a,
                            )
                            T.copy(
                                l1b[
                                    n_tile * mad_n : (n_tile + 1) * mad_n,
                                    k_tile * mad_k : (k_tile + 1) * mad_k,
                                ],
                                l0b,
                            )
                            T.gemm(
                                l0a,
                                l0b,
                                l0c[m_tile, n_tile, :, :],
                                transpose_B=True,
                                clear_accum=k_tile == 0,
                            )
                for m_tile in T.Unroll(2, explicit=True):
                    for n_tile in T.Unroll(2, explicit=True):
                        T.copy(
                            l0c[m_tile, n_tile, :, :],
                            C[
                                block,
                                m_tile * mad_m : (m_tile + 1) * mad_m,
                                n_tile * mad_n : (n_tile + 1) * mad_n,
                            ],
                        )

    return main


def _make_fp8_mnk_copy_program():
    tile = 256
    m_tiles = 2
    n_tiles = 2
    k_tiles = 2
    total_tiles = m_tiles * n_tiles * k_tiles

    @T.prim_func
    def main(
        A: T.Buffer((total_tiles, tile), "float8_e4m3fn"),
        B: T.Buffer((total_tiles, tile), "float8_e4m3fn"),
        C: T.Buffer((total_tiles, tile), "float8_e4m3fn"),
        D: T.Buffer((total_tiles, tile), "float8_e4m3fn"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((tile,), "float8_e4m3fn")
            b_ub = T.alloc_shared((tile,), "float8_e4m3fn")
            T.annotate_buffer_versions({a_ub: 2, b_ub: 2})
            for m_tile in T.Unroll(m_tiles, explicit=True):
                for n_tile in T.Unroll(n_tiles, explicit=True):
                    for k_tile in T.Pipelined(k_tiles, num_stages=2):
                        offset = (m_tile * n_tiles + n_tile) * k_tiles + k_tile
                        T.copy(A[offset, :], a_ub)
                        T.copy(B[offset, :], b_ub)
                        T.copy(a_ub, C[offset, :])
                        T.copy(b_ub, D[offset, :])

    return main


def _make_row_writer_then_consumer_program():
    blocks = 2
    rows = 4
    cols = 64

    @T.prim_func
    def main(
        Score: T.Buffer((blocks, rows, cols), "float32"),
        Latent: T.Buffer((blocks, rows, cols), "float32"),
        ScoreOut: T.Buffer((blocks, rows, cols), "float32"),
        LatentOut: T.Buffer((blocks, rows, cols), "float32"),
        Pred: T.Buffer((blocks, rows), "int32"),
    ):
        with T.Kernel(1):
            score_ub = T.alloc_shared((rows, cols), "float32")
            latent_ub = T.alloc_shared((rows, cols), "float32")
            T.annotate_buffer_versions({score_ub: 2, latent_ub: 2})
            with T.SimdVF(cols):
                T.fill(score_ub, float("-inf"))
                T.fill(latent_ub, 0)
            for bdim in T.Pipelined(blocks, num_stages=2):
                for i in T.serial(rows):
                    if Pred[bdim, i] > 0:
                        T.copy(Score[bdim, i, :], score_ub[i, :])
                        T.copy(Latent[bdim, i, :], latent_ub[i, :])
                T.copy(score_ub, ScoreOut[bdim, :, :])
                T.copy(latent_ub, LatentOut[bdim, :, :])

    return main


def _make_structured_fill_initialization_program():
    blocks = 2
    rows = 4
    cols = 64

    @T.prim_func
    def main(
        A: T.Buffer((blocks, cols), "float32"),
        C: T.Buffer((blocks, rows, cols), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for r in T.serial(rows):
                T.fill(ub[r, :], 0)
            for bdim in T.Pipelined(blocks, num_stages=2):
                T.copy(A[bdim, :], ub[0, :])
                T.copy(ub, C[bdim, :, :])

    return main


def _make_fill_then_access_program():
    blocks = 2
    rows = 4
    cols = 64

    @T.prim_func
    def main(
        A: T.Buffer((blocks, cols), "float32"),
        C: T.Buffer((blocks, rows, cols), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((rows, cols), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for bdim in T.Pipelined(blocks, num_stages=2):
                T.fill(ub, 0)
                T.copy(A[bdim, :], ub[0, :])
                T.copy(ub, C[bdim, :, :])

    return main


def _make_partial_owner_with_external_consumer_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((5 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                T.copy(ub, C[i * tile : (i + 1) * tile])
            T.copy(ub, C[4 * tile : 5 * tile])

    return main


def _make_versioned_storage_assume_program(
    explicit_claim=False,
    task_wrapped=False,
    read_first=False,
    nested_control=False,
):
    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]} if explicit_claim else {},
            ):
                if task_wrapped:
                    with T.Task():
                        if read_first:
                            T.assume(ub[0] >= 0)
                        ub[0] = A[i]
                        if not read_first:
                            T.assume(ub[0] >= 0)
                        C[i] = ub[0]
                else:
                    if read_first:
                        T.assume(ub[0] >= 0)
                    ub[0] = A[i]
                    if not read_first:
                        T.assume(ub[0] >= 0)
                    if nested_control:
                        for _j in T.serial(1):
                            C[i] = ub[0]
                    else:
                        C[i] = ub[0]

    return main


def _make_sibling_assume_program():
    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(2, num_stages=2):
                ub[0] = A[i]
                T.assume(ub[0] >= 0)
                C[i] = ub[0]
            for j in T.Pipelined(2, num_stages=2):
                ub[0] = A[j + 2]
                T.assume(ub[0] >= 0)
                C[j + 2] = ub[0]

    return main


def _make_manual_storage_with_explicit_auto_claim_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((2, tile), "float32")
            T.annotate_manual_multi_buffer(ub)
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                slot = i % 2
                T.copy(A[i * tile : (i + 1) * tile], ub[slot, :])
                T.copy(ub[slot, :], C[i * tile : (i + 1) * tile])

    return main


def _make_guarded_manual_storage_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((2, tile), "float32")
            T.annotate_manual_multi_buffer(ub)
            for i in T.Pipelined(4, num_stages=2):
                if enabled > 0:
                    slot = i % 2
                    T.copy(A[i * tile : (i + 1) * tile], ub[slot, :])
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[slot, 0])
                        T.simd.vsts(ub[slot, 0], value, mask)
                    T.copy(ub[slot, :], C[i * tile : (i + 1) * tile])

    return main


def _make_counter_l1_layout_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile, tile), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0 = T.alloc_l0a((tile, tile), "bfloat16")
            T.annotate_buffer_versions({l1: (3, "counter")})
            for i in T.Pipelined(
                4,
                num_stages=3,
                annotations={"multi_buffer_eligible": [l1]},
            ):
                T.copy(A[i * tile : (i + 1) * tile, :], l1)
                T.copy(l1, l0)

    return main


def _make_l1_fill_initialization_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile, tile), "bfloat16")):
        with T.Kernel(1):
            l1 = T.alloc_l1((tile, tile), "bfloat16")
            l0 = T.alloc_l0a((tile, tile), "bfloat16")
            T.annotate_buffer_versions({l1: (2, "counter")})
            T.fill(l1, 0)
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [l1]},
            ):
                T.copy(A[i * tile : (i + 1) * tile, :], l1)
                T.copy(l1, l0)

    return main


def _make_dependent_extent_program(mode=None):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((16 * tile,), "float32"), C: T.Buffer((16 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for outer in T.Pipelined(4, num_stages=2):
                for inner in T.serial(outer + 1):
                    offset = (outer * 4 + inner) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_non_affine_offset_program(mode="iteration"):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((5 * tile,), "float32"), C: T.Buffer((5 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for outer in T.serial(2):
                for inner in T.Pipelined(
                    outer + 2,
                    num_stages=2,
                    annotations={
                        "enable_offset": True,
                        "multi_buffer_eligible": [ub],
                    },
                ):
                    offset = (outer * 3 + inner) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_non_affine_control_path_program(kind):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        enabled: T.int32,
        stop: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "iteration")})
            if kind == "guard":
                if enabled > 0:
                    for outer in T.serial(2):
                        for inner in T.serial(
                            2,
                            annotations={"multi_buffer_eligible": [ub]},
                        ):
                            offset = (outer * 2 + inner) * tile
                            T.copy(A[offset : offset + tile], ub)
            else:
                for outer in T.serial(2):
                    if outer >= stop:
                        T.loop_break()
                    for inner in T.serial(
                        2,
                        annotations={"multi_buffer_eligible": [ub]},
                    ):
                        offset = (outer * 2 + inner) * tile
                        T.copy(A[offset : offset + tile], ub)

    return main


def _make_mutable_range_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((tile,), "float32"),
        extent: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "iteration")})
            for _outer in T.serial(2):
                for _inner in T.serial(
                    extent[0],
                    annotations={"multi_buffer_eligible": [ub]},
                ):
                    T.copy(A, ub)

    return main


def _make_loop_local_range_var_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((tile,), "float32"),
        extent: T.Buffer((2,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "iteration")})
            for outer in T.serial(2):
                inner_extent = T.bind(extent[outer])
                for _inner in T.serial(
                    inner_extent,
                    annotations={"multi_buffer_eligible": [ub]},
                ):
                    T.copy(A, ub)

    return main


def _make_decreasing_dependent_extent_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((6 * tile,), "float32"), C: T.Buffer((6 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((6 * tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for outer in T.Pipelined(4, num_stages=2):
                for inner in T.serial(4 - outer):
                    # This is the old flattened iteration expression. Because
                    # the inner extent depends on outer, distinct loop tuples
                    # can map to the same offset.
                    offset = (outer * (4 - outer) + inner) * tile
                    T.copy(A[offset : offset + tile], ub[offset : offset + tile])
                    T.copy(ub[offset : offset + tile], C[offset : offset + tile])

    return main


def _make_break_program(position="entry", mode=None):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        stop: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.serial(8):
                if position == "entry":  # noqa: SIM102 -- outer branch is Python staging
                    if i >= stop:
                        T.loop_break()
                T.copy(A[i * tile : (i + 1) * tile], ub)
                if position == "mid":  # noqa: SIM102 -- outer branch is Python staging
                    if i >= stop:
                        T.loop_break()
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[i * tile : (i + 1) * tile])
                if position == "tail":  # noqa: SIM102 -- outer branch is Python staging
                    if i >= stop:
                        T.loop_break()

    return main


def _make_lexical_middle_break_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        stop: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            for i in T.serial(4):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                if i >= stop:
                    T.loop_break()
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_nested_break_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        stop: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for outer in T.serial(2):
                for inner in T.serial(4):
                    if inner >= stop:
                        T.loop_break()
                    offset = (outer * 4 + inner) * tile
                    T.copy(A[offset : offset + tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_counter_alias_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            alias = T.reshape(ub, (tile // 2, 2))
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(8, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                if enabled > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(alias[0, 0])
                        T.simd.vsts(alias[0, 0], value, mask)
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_sibling_alias_views_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((tile,), "float32")
            alias_a = T.reshape(base, (tile // 2, 2))
            alias_b = T.reshape(base, (tile // 4, 4))
            T.annotate_buffer_versions({alias_a: 2})
            T.fill(base, 0)
            for i in T.serial(2, annotations={"multi_buffer_eligible": [alias_a]}):
                alias_a[0, 0] = A[i]
                C[i] = alias_a[0, 0]
            for j in T.serial(2, annotations={"multi_buffer_eligible": [alias_b]}):
                alias_b[0, 0] = A[j + 2]
                C[j + 2] = alias_b[0, 0]

    return main


def _make_group_break_union_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        B: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        D: T.Buffer((8 * tile,), "float32"),
        stop: T.int32,
    ):
        with T.Kernel(1):
            ub_a = T.alloc_shared((tile,), "float32")
            ub_b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub_a: 2, ub_b: 2})
            for outer in T.serial(2):
                for i in T.serial(
                    4,
                    annotations={"multi_buffer_eligible": [ub_a, ub_b]},
                ):
                    offset = (outer * 4 + i) * tile
                    T.copy(A[offset : offset + tile], ub_a)
                    T.copy(ub_a, C[offset : offset + tile])
                    if i >= stop:
                        T.loop_break()
                    T.copy(B[offset : offset + tile], ub_b)
                    T.copy(ub_b, D[offset : offset + tile])

    return main


def _make_group_break_same_signature_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        B: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        D: T.Buffer((4 * tile,), "float32"),
        stop: T.int32,
    ):
        with T.Kernel(1):
            ub_a = T.alloc_shared((tile,), "float32")
            ub_b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub_a: 2, ub_b: 2})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub_a, ub_b]}):
                T.copy(A[i * tile : (i + 1) * tile], ub_a)
                T.copy(B[i * tile : (i + 1) * tile], ub_b)
                if i >= stop:
                    T.loop_break()
                T.copy(ub_a, C[i * tile : (i + 1) * tile])
                T.copy(ub_b, D[i * tile : (i + 1) * tile])

    return main


def _make_multiple_break_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        stop_a: T.int32,
        stop_b: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(4):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                if i >= stop_a:
                    T.loop_break()
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                if i >= stop_b:
                    T.loop_break()
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_scalar_middle_break_program():
    @T.prim_func
    def main(stop: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((1,), "int32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(4):
                ub[0] = i
                if i >= stop:
                    T.loop_break()
                ub[0] = i + 1

    return main


def _make_cross_core_middle_break_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(C: T.Buffer((4 * tile_m, tile_n), "bfloat16"), stop: T.int32):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0)
                if i >= stop:
                    T.loop_break()
                if sid == 0:
                    T.copy(ub, C[i * tile_m : (i + 1) * tile_m, :])

    return main


def _make_fix_unit_flag_middle_break_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(C: T.Buffer((4 * tile_m, tile_n), "bfloat16"), stop: T.int32):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if i >= stop:
                    T.loop_break()
                if sid == 0:
                    T.copy(ub, C[i * tile_m : (i + 1) * tile_m, :])

    return main


def _make_offset_program(enable_offset=True, mode="counter"):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.Pipelined(8, num_stages=2, annotations={"enable_offset": enable_offset}):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_auto_sibling_cross_stage_counter_program():
    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(2, annotations={"enable_offset": True}):
                with T.Stage(1):
                    C[i] = ub[0]
                with T.Stage(0):
                    ub[0] = A[i]
            for j in T.serial(2, annotations={"enable_offset": True}):
                with T.Stage(1):
                    C[j + 2] = ub[0]
                with T.Stage(0):
                    ub[0] = A[j + 2]

    return main


def _make_per_core_task_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(8, num_stages=2):
                with T.PerCoreTask():
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_guarded_per_core_task_program(guard_all_accesses):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(
                8,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                if enabled > 0:
                    with T.PerCoreTask():
                        # Each PerCoreTask invocation must execute exactly one candidate Task.
                        if i % 2 == 0:
                            T.copy(A[i * tile : (i + 1) * tile], ub)
                        else:
                            T.copy(A[i * tile : (i + 1) * tile], ub)
                    if guard_all_accesses:  # noqa: SIM102 -- outer branch is Python staging
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)
                        T.copy(ub, C[i * tile : (i + 1) * tile])
                if not guard_all_accesses:
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_mixed_core_memory_bind_program(mode=None):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        B: T.Buffer((8, tile), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            l1 = T.alloc_l1((1, tile), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.Pipelined(8, num_stages=2):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                T.copy(B[i : i + 1, :], l1)
                # This scalar task illegally reads Vector-local UB and
                # Cube-local L1, so AssignCore must eventually reject it.
                gate = ub[0] + l1[0, 0]
                if gate > 0:
                    T.copy(ub, C[i * tile : (i + 1) * tile])
                    T.copy(B[i : i + 1, :], l1)

    return main


def _make_unbroadcastable_counter_guard_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        pred: T.Buffer((1,), "int32"),
        A: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        C: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            pred_ub = T.alloc_shared((1,), "int32")
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.copy(pred, pred_ub)
            raw_predicate = T.bind(pred_ub[0])
            active = T.bind(raw_predicate > 0)
            T.annotate_buffer_versions({ub: 2})
            for _vector_only in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                if active:
                    T.copy(A[0:tile_m, :], ub)
                    if sid == 0:
                        T.copy(ub, C[0:tile_m, :])
            for _cross_core in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if sid == 0:
                    T.copy(ub, C[tile_m : 2 * tile_m, :])

    return main


def _make_sparse_guard_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32"), C: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(8, num_stages=2):
                if i % 2 == 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_guarded_inplace_global_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((8 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(8, num_stages=2):
                if i % 2 == 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, A[i * tile : (i + 1) * tile])

    return main


def _make_guarded_reused_global_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(8, num_stages=2):
                if i % 2 == 0:
                    T.copy(A, ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, A)

    return main


def _make_mismatched_counter_lexical_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((tile,), "float32"),
        B: T.Buffer((8 * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            other = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(8, num_stages=2):
                if i % 2 == 0:
                    T.copy(A, ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, A)
                if i % 3 == 0:
                    T.copy(A, other)
                    T.copy(other, B[i * tile : (i + 1) * tile])

    return main


def _make_single_iteration_guarded_global_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            read_ub = T.alloc_shared((tile,), "float32")
            write_ub = T.alloc_shared((tile,), "float32")
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vdup(T.float32(0), "float32", mask)
                T.simd.vsts(write_ub[0], value, mask)
            for _ in T.serial(1):
                if enabled > 0:
                    T.copy(A, read_ub)
                    T.copy(write_ub, A)

    return main


def _make_task_internal_guarded_global_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                with T.Task():
                    if enabled > 0:
                        T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.Task():
                    if enabled > 0:
                        T.copy(ub, A[i * tile : (i + 1) * tile])

    return main


def _make_task_internal_nested_loop_guarded_global_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                with T.Task():
                    for _j in T.serial(2):
                        if enabled > 0:
                            T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.Task():
                    for _j in T.serial(2):
                        if enabled > 0:
                            T.copy(ub, A[i * tile : (i + 1) * tile])

    return main


def _make_task_internal_sblock_guarded_program():
    tile = 64

    @T.prim_func
    def main(enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for _i in T.serial(4):
                with T.Task(), T.SimdVF():
                    if enabled > 0:
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)

    return main


def _make_task_internal_storage_condition_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                with T.Task():
                    if A[i * tile] > 0:
                        T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.Task():
                    if A[i * tile] > 0:
                        T.copy(ub, A[i * tile : (i + 1) * tile])

    return main


def _make_compound_task_guard_definition_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        D: T.Buffer((4,), "int32"),
        pred_a: T.int32,
        pred_b: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                with T.Task():
                    first = T.bind(pred_a > 0)
                    if first:
                        T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.Task():
                    late = T.bind(pred_b > 0)
                    D[i] = T.cast(first, "int32")
                if late:
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)

    return main


def _make_early_compound_task_guard_definition_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        D: T.Buffer((4,), "int32"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                with T.Task():
                    active = T.bind(enabled > 0)
                    D[i] = T.cast(active, "int32")
                if active:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)

    return main


def _make_nested_lexical_domain_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                if enabled > 0:
                    for _j in T.serial(1):
                        T.copy(A[i * tile : (i + 1) * tile], ub)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, A[i * tile : (i + 1) * tile])

    return main


def _make_nested_guarded_lexical_domain_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        outer_enabled: T.int32,
        inner_enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vdup(T.float32(0), "float32", mask)
                T.simd.vsts(ub[0], value, mask)
            for i in T.serial(4):
                if outer_enabled > 0:
                    for _j in T.serial(1):
                        if inner_enabled > 0:
                            T.copy(A[i * tile : (i + 1) * tile], ub)
                            with T.SimdVF():
                                mask = T.simd.pset(32)
                                value = T.simd.vld(ub[0])
                                T.simd.vsts(ub[0], value, mask)
                    T.copy(ub, A[i * tile : (i + 1) * tile])

    return main


def _make_nested_only_guarded_lexical_domain_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for _outer in T.serial(4):
                for _inner in T.serial(1):
                    if enabled > 0:
                        T.copy(A, ub)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)
                        T.copy(ub, A)

    return main


def _make_nested_iteration_dependent_guard_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for _outer in T.serial(4):
                for inner in T.serial(2):
                    if inner % 2 == 0:
                        T.copy(A, ub)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)
                        T.copy(ub, A)

    return main


def _make_mutating_nested_extent_storage_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((tile,), "float32"),
        C: T.Buffer((tile,), "float32"),
        extent: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for _outer in T.serial(4):
                for _inner in T.serial(extent[0]):
                    T.copy(A, ub)
                    extent[0] = 0
                    T.copy(ub, C)

    return main


def _make_unversioned_union_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        pred_a: T.int32,
        pred_b: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                if pred_a > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                if pred_b > 0:
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)

    return main


def _make_cross_stage_unversioned_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        pred_a: T.int32,
        pred_b: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"enable_offset": True},
            ):
                with T.Stage(0):
                    if pred_a > 0:
                        T.copy(A[i * tile : (i + 1) * tile], ub)
                with T.Stage(1):
                    if pred_b > 0:
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)

    return main


def _make_early_lexical_late_counter_snapshot_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((tile,), "float32"),
        B: T.Buffer((tile,), "float32"),
        C: T.Buffer((tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            lexical = T.alloc_shared((tile,), "float32")
            counter = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({counter: (2, "counter")})
            for _i in T.serial(4):
                if pred > 0:
                    T.copy(A, lexical)
                    T.copy(lexical, C)
                if pred > 0:
                    T.copy(B, counter)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(counter[0])
                        T.simd.vsts(counter[0], value, mask)

    return main


def _make_wide_and_narrow_same_edge_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile,), "float32"),
        C: T.Buffer((tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            scratch = T.alloc_shared((tile,), "float32")
            narrow = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({scratch: 1, narrow: 1})
            for _i in T.serial(1):
                T.copy(A[0:tile], scratch)
                if pred > 0:
                    T.copy(A[tile : 2 * tile], narrow)
                    T.copy(narrow, A[tile : 2 * tile])
            T.copy(scratch, C)

    return main


def _make_one_task_two_sparse_domains_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        pred_x: T.int32,
        pred_y: T.int32,
    ):
        with T.Kernel(1):
            x = T.alloc_shared((tile,), "float32")
            y = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({x: 1, y: 1})
            for i in T.Pipelined(4, num_stages=2):
                if pred_x > 0:
                    T.copy(A[2 * i * tile : (2 * i + 1) * tile], x)
                if pred_y > 0:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], y)
                if pred_x > 0:  # noqa: SIM102 - keep two storage guards distinct
                    if pred_y > 0:
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value_x = T.simd.vld(x[0])
                            value_y = T.simd.vld(y[0])
                            T.simd.vsts(x[0], value_x, mask)
                            T.simd.vsts(y[0], value_y, mask)

    return main


def _make_one_domain_split_into_two_parent_domains_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4, tile), "float32"),
        B: T.Buffer((4, tile), "float32"),
        common: T.int32,
        pred_x: T.int32,
        pred_y: T.int32,
    ):
        with T.Kernel(1):
            x = T.alloc_shared((tile,), "float32")
            y = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({x: 1, y: 1})
            for i in T.serial(4):
                if common > 0:
                    for _j in T.serial(1):
                        T.copy(A[i, :], x)
                        T.copy(B[i, :], y)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value_x = T.simd.vld(x[0])
                            value_y = T.simd.vld(y[0])
                            T.simd.vsts(x[0], value_x, mask)
                            T.simd.vsts(y[0], value_y, mask)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value_x = T.simd.vld(x[0])
                            value_y = T.simd.vld(y[0])
                            T.simd.vsts(x[0], value_x, mask)
                            T.simd.vsts(y[0], value_y, mask)
                if pred_x > 0:
                    T.copy(x, A[i, :])
                if pred_y > 0:
                    T.copy(y, B[i, :])

    return main


def _make_nested_loop_carried_lexical_domains_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((2, tile), "float32"),
        outer_enabled: T.int32,
        inner_enabled: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((2, tile), "float32")
            T.annotate_buffer_versions({ub: 1})
            for outer in T.serial(2):
                if outer_enabled > 0:
                    for _inner in T.Pipelined(4, num_stages=2):
                        if inner_enabled > 0:
                            T.copy(A[outer, :], ub[outer, :])
                            T.copy(ub[outer, :], A[outer, :])

    return main


def _make_union_guard_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((16 * tile,), "float32"), pred_a: T.int32, pred_b: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.Pipelined(
                8,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                if pred_a > 0:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)
                if pred_b > 0:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub)
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        value = T.simd.vld(ub[0])
                        T.simd.vsts(ub[0], value, mask)

    return main


def _make_exhaustive_nested_union_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((24 * tile,), "float32"),
        outer: T.int32,
        left: T.int32,
        right: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(
                8,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                if outer > 0:
                    if left > 0:
                        T.copy(A[(3 * i) * tile : (3 * i + 1) * tile], ub)
                    else:
                        if right > 0:
                            T.copy(A[(3 * i + 1) * tile : (3 * i + 2) * tile], ub)
                        else:
                            T.copy(A[(3 * i + 2) * tile : (3 * i + 3) * tile], ub)

    return main


def _make_independent_guard_union_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((32 * tile,), "float32"),
        a0: T.bool,
        b0: T.bool,
        a1: T.bool,
        b1: T.bool,
        a2: T.bool,
        b2: T.bool,
        a3: T.bool,
        b3: T.bool,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(
                8,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                if a0:  # noqa: SIM102 - preserve two-literal guard cube
                    if b0:
                        T.copy(A[(4 * i) * tile : (4 * i + 1) * tile], ub)
                if a1:  # noqa: SIM102 - preserve two-literal guard cube
                    if b1:
                        T.copy(A[(4 * i + 1) * tile : (4 * i + 2) * tile], ub)
                if a2:  # noqa: SIM102 - preserve two-literal guard cube
                    if b2:
                        T.copy(A[(4 * i + 2) * tile : (4 * i + 3) * tile], ub)
                if a3:  # noqa: SIM102 - preserve two-literal guard cube
                    if b3:
                        T.copy(A[(4 * i + 3) * tile : (4 * i + 4) * tile], ub)

    return main


def _make_cross_core_absorbed_guard_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(C: T.Buffer((4 * tile_m, tile_n), "bfloat16")):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(4, num_stages=2):
                if i % 2 == 0:
                    T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                    if sid == 0:
                        T.copy(ub, C[i * tile_m : (i + 1) * tile_m, :])

    return main


def _make_cross_core_implied_snapshot_guard_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        A: T.Buffer((tile_m, tile_n), "bfloat16"),
        B: T.Buffer((tile_m, tile_n), "bfloat16"),
        C: T.Buffer((4 * tile_m, tile_n), "bfloat16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile_m, tile_n), "bfloat16")
            b_l1 = T.alloc_l1((tile_m, tile_n), "bfloat16")
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                if i % 2 == 0:
                    T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if i % 4 == 0:
                    T.copy(ub, C[i * tile_m : (i + 1) * tile_m, :])

    return main


def _make_cross_core_nested_snapshot_guard_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        A: T.Buffer((tile_m, tile_n), "bfloat16"),
        B: T.Buffer((tile_m, tile_n), "bfloat16"),
        C: T.Buffer((4 * tile_m, tile_n), "bfloat16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((tile_m, tile_n), "bfloat16")
            b_l1 = T.alloc_l1((tile_m, tile_n), "bfloat16")
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, l0c, transpose_B=True, clear_accum=True)
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]},
            ):
                if i % 2 == 0:  # noqa: SIM102 - preserve nested snapshot scope
                    if i % 4 == 0:
                        T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                        T.copy(ub, C[i * tile_m : (i + 1) * tile_m, :])

    return main


def _make_crossed_partial_guard_projection_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile,), "float32"),
        C: T.Buffer((2 * tile,), "float32"),
        pred_a: T.bool,
        pred_b: T.bool,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(2):
                for _middle in T.serial(1):
                    if pred_a:
                        for _producer in T.serial(1):
                            if pred_b:
                                T.copy(A[i * tile : (i + 1) * tile], ub)
                    if pred_b:
                        for _consumer in T.serial(1):
                            if pred_a:
                                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_narrow_child_bridge_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4, 2 * tile), "float32"),
        C: T.Buffer((4, 2 * tile), "float32"),
        pred_a: T.bool,
        pred_b: T.bool,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((2 * tile,), "float32")
            T.annotate_buffer_versions({ub: 1})
            for i in T.serial(4):
                if pred_b:
                    T.copy(A[i, 0:tile], ub[0:tile])
                for _middle in T.serial(1):
                    if pred_a:
                        T.copy(A[i, tile : 2 * tile], ub[tile : 2 * tile])
                    if pred_a:
                        T.copy(ub[tile : 2 * tile], C[i, tile : 2 * tile])
                if pred_b:
                    T.copy(ub[0:tile], C[i, 0:tile])

    return main


def _make_mixed_core_sibling_program(mode=None):
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        C: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for _cross_core in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if sid == 0:
                    T.copy(ub, C[0:tile_m, :])
            for _vector_only in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[tile_m : 2 * tile_m, :], ub)
                if sid == 0:
                    T.copy(ub, C[tile_m : 2 * tile_m, :])

    return main


def _make_guarded_mixed_core_sibling_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        C: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        enabled: T.int32,
    ):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: 2})
            for _vector_only in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                active = T.bind(enabled > 0)
                if active:
                    T.copy(A[0:tile_m, :], ub)
                    if sid == 0:
                        T.copy(ub, C[0:tile_m, :])
            for _cross_core in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if sid == 0:
                    T.copy(ub, C[tile_m : 2 * tile_m, :])

    return main


def _make_lexical_union_guard_core_resolution_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        C: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        pred_cube: T.int32,
        pred_vector: T.int32,
    ):
        with T.Kernel(1):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            active_cube = T.bind(pred_cube > 0)
            active_vector = T.bind(pred_vector > 0)
            for i in T.Pipelined(2, num_stages=2):
                if active_cube:
                    T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if active_vector:
                    T.copy(ub, C[i * tile_m : (i + 1) * tile_m, :])

    return main


def _make_outer_guarded_mixed_core_sibling_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        C: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        pred: T.Buffer((1,), "int32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: 2})
            active = T.bind(pred[0] > 0)
            pred[0] = 0
            if active:
                for _vector_only in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                    T.copy(A[0:tile_m, :], ub)
                    if sid == 0:
                        T.copy(ub, C[0:tile_m, :])
            for _cross_core in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if sid == 0:
                    T.copy(ub, C[tile_m : 2 * tile_m, :])

    return main


def _make_dynamic_extent_mixed_core_sibling_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        A: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        C: T.Buffer((2 * tile_m, tile_n), "bfloat16"),
        extent: T.Buffer((1,), "int32"),
    ):
        with T.MixedKernel(1) as (_, sid):
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.annotate_buffer_versions({ub: 2})
            repeat_count = T.bind(extent[0])
            extent[0] = 0
            for _vector_only in T.serial(repeat_count, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[0:tile_m, :], ub)
                if sid == 0:
                    T.copy(ub, C[0:tile_m, :])
            for _cross_core in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(l0c, ub, sub_blockid=0, unit_flag_ctrl=3)
                if sid == 0:
                    T.copy(ub, C[tile_m : 2 * tile_m, :])

    return main


def _make_mismatched_sibling_protocol_program(mode=None):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((2 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                T.copy(ub, C[i * tile : (i + 1) * tile])
            for j in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[(j + 2) * tile : (j + 3) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)

    return main


def _make_nested_unconditional_projection_program(swapped=False):
    tile = 64

    @T.macro
    def first_pass(A, x, y, w):
        for i in T.serial(2):
            offset = (w * 8 + i * 2) * tile
            T.copy(A[offset : offset + tile], x)
            T.copy(A[offset + tile : offset + 2 * tile], y)
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vld(y[0])
                T.simd.vsts(y[0], value, mask)
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vld(x[0])
                T.simd.vsts(x[0], value, mask)

    @T.macro
    def second_pass(A, x, y, w):
        for j in T.serial(2):
            offset = (w * 8 + 4 + j * 2) * tile
            T.copy(A[offset : offset + tile], x)
            T.copy(A[offset + tile : offset + 2 * tile], y)
            with T.SimdVF():
                mask = T.simd.pset(32)
                x_value = T.simd.vld(x[0])
                y_value = T.simd.vld(y[0])
                T.simd.vsts(x[0], x_value, mask)
                T.simd.vsts(y[0], y_value, mask)

    @T.prim_func
    def main(A: T.Buffer((24 * tile,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((tile,), "float32")
            y = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({x: (2, "counter"), y: (2, "counter")})
            for w in T.serial(3):
                if swapped:
                    second_pass(A, x, y, w)
                    first_pass(A, x, y, w)
                else:
                    first_pass(A, x, y, w)
                    second_pass(A, x, y, w)

    return main


def _make_cross_sibling_region_swap_program():
    @T.prim_func
    def main(A: T.Buffer((3, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((128,), "float32")
            T.annotate_buffer_versions({ub: 2})

            for _a in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[0, :], ub[0:64])
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[64])
                    T.simd.vsts(ub[64], value, mask)

            for _b in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[1, :], ub[0:64])
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[64])
                    T.simd.vsts(ub[64], value, mask)

            for _c in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[2, :], ub[64:128])
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)

    return main


def _make_missing_channel_endpoint_program(mode=None):
    @T.prim_func
    def main():
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})

            for _vector in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vdup(T.float32(1), "float32", mask)
                    T.simd.vsts(ub[0], value, mask)

            for _scalar_0 in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                ub[0] = T.float32(2)

            for _scalar_1 in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                ub[0] = T.float32(3)

    return main


def _make_nested_missing_channel_endpoint_program(pipelined=False):
    @T.macro
    def sibling_epochs(ub):
        for _vector in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vdup(T.float32(1), "float32", mask)
                T.simd.vsts(ub[0], value, mask)

        for _scalar in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
            ub[0] = T.float32(2)

    @T.prim_func
    def main():
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({ub: 2})
            if pipelined:
                for _outer in T.Pipelined(3, num_stages=2):
                    sibling_epochs(ub)
            else:
                for _outer in T.serial(3):
                    sibling_epochs(ub)

    return main


def _make_same_pipe_transitive_elimination_program():
    @T.prim_func
    def main(A: T.Buffer((128,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((64,), "float32")
            w = T.alloc_shared((64,), "float32")
            with T.SimdVF():
                mask = T.simd.pset(32)
                value = T.simd.vdup(T.float32(1), "float32", mask)
                T.simd.vsts(x[0], value, mask)
                T.simd.vsts(w[0], value, mask)
            T.copy(A[0:64], x)
            T.copy(A[64:128], x)
            w[1] = w[0] + x[0]

    return main


def _make_guard_group_program(same_guard=True):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((16 * tile,), "float32"),
        C: T.Buffer((16 * tile,), "float32"),
        pred_a: T.int32,
        pred_b: T.int32,
    ):
        with T.Kernel(1):
            ub_a = T.alloc_shared((tile,), "float32")
            ub_b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub_a: 2, ub_b: 3})
            for i in T.Pipelined(8, num_stages=3):
                if same_guard:  # noqa: SIM102 -- Python staging choice
                    if pred_a > 0:
                        T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub_a)
                        T.copy(ub_a, C[(2 * i) * tile : (2 * i + 1) * tile])
                        T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub_b)
                        T.copy(ub_b, C[(2 * i + 1) * tile : (2 * i + 2) * tile])
                else:
                    if pred_a > 0:
                        T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub_a)
                        T.copy(ub_a, C[(2 * i) * tile : (2 * i + 1) * tile])
                    if pred_b > 0:
                        T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub_b)
                        T.copy(ub_b, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_equivalent_guard_group_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((16 * tile,), "float32"),
        C: T.Buffer((16 * tile,), "float32"),
        pred: T.Buffer((8,), "int32"),
    ):
        with T.Kernel(1):
            ub_a = T.alloc_shared((tile,), "float32")
            ub_b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub_a: 2, ub_b: 3})
            for i in T.Pipelined(8, num_stages=3):
                if pred[i] > 0:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub_a)
                    T.copy(ub_a, C[(2 * i) * tile : (2 * i + 1) * tile])
                pred[i] = 1
                if pred[i] > 0:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub_b)
                    T.copy(ub_b, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_equivalent_snapshot_group_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub_a = T.alloc_shared((tile,), "float32")
            ub_b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub_a: 2, ub_b: 3})
            for i in T.Pipelined(4, num_stages=3):
                if pred > 0:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub_a)
                    T.copy(ub_a, C[(2 * i) * tile : (2 * i + 1) * tile])
                if pred > 0:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub_b)
                    T.copy(ub_b, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_crossed_counter_group_snapshot_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((16 * tile,), "float32"),
        C: T.Buffer((16 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: 2, b: 3})
            for i in T.Pipelined(4, num_stages=3, annotations={"multi_buffer_eligible": [a, b]}):
                if pred > 0:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], a)
                    T.copy(a, C[(2 * i) * tile : (2 * i + 1) * tile])
                if pred > 0:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], b)
                    T.copy(b, C[(2 * i + 1) * tile : (2 * i + 2) * tile])
            for j in T.Pipelined(4, num_stages=3, annotations={"multi_buffer_eligible": [a, b]}):
                if pred > 0:
                    T.copy(A[(2 * j + 8) * tile : (2 * j + 9) * tile], b)
                    T.copy(b, C[(2 * j + 8) * tile : (2 * j + 9) * tile])
                if pred > 0:
                    T.copy(A[(2 * j + 9) * tile : (2 * j + 10) * tile], a)
                    T.copy(a, C[(2 * j + 9) * tile : (2 * j + 10) * tile])

    return main


def _make_mixed_core_equivalent_guard_group_program():
    tile_m = 16
    tile_n = 16

    @T.prim_func
    def main(
        predicate: T.Buffer((1,), "int32"),
        A: T.Buffer((4 * tile_m, tile_n), "bfloat16"),
        pred: T.int32,
    ):
        with T.MixedKernel(1) as (_, _sid):
            pred_ub = T.alloc_shared((1,), "int32")
            cube_l1 = T.alloc_l1((tile_m, tile_n), "bfloat16")
            l0c = T.alloc_l0c((tile_m, tile_n), "float32")
            mixed_ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            vector_ub = T.alloc_shared((tile_m, tile_n), "bfloat16")
            T.copy(predicate, pred_ub)
            vector_value = pred_ub[0]
            vector_pred = T.bind(pred + (vector_value // 2 * 2 + vector_value % 2 - vector_value))
            T.annotate_buffer_versions({mixed_ub: 3, vector_ub: 2})
            for i in T.Pipelined(
                4,
                num_stages=3,
                annotations={"multi_buffer_eligible": [vector_ub, mixed_ub]},
            ):
                if (pred > 0) and (vector_pred // 2 * 2 + vector_pred % 2 == vector_pred):
                    T.copy(A[i * tile_m : (i + 1) * tile_m, :], vector_ub)
                if pred > 0:
                    T.copy(A[i * tile_m : (i + 1) * tile_m, :], cube_l1)
                if pred > 0:
                    T.copy(l0c, mixed_ub, sub_blockid=0, unit_flag_ctrl=3)

    return main


def _make_assumed_equivalent_snapshot_group_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            T.assume(pred >= 0)
            T.assume(pred <= 1)
            ub_a = T.alloc_shared((tile,), "float32")
            ub_b = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub_a: 2, ub_b: 3})
            for i in T.Pipelined(4, num_stages=3):
                if pred > 0:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub_a)
                    T.copy(ub_a, C[(2 * i) * tile : (2 * i + 1) * tile])
                if pred == 1:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub_b)
                    T.copy(ub_b, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_implied_union_guard_program(mode="counter"):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.Pipelined(4, num_stages=2):
                if pred > 0:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub)
                    T.copy(ub, C[(2 * i) * tile : (2 * i + 1) * tile])
                if pred > 1:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub)
                    T.copy(ub, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_reverse_implied_union_guard_program(mode="counter"):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.Pipelined(4, num_stages=2):
                weak = T.bind(pred > 0)
                strong = T.bind(pred > 1)
                if strong:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub)
                    T.copy(ub, C[(2 * i) * tile : (2 * i + 1) * tile])
                if weak:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub)
                    T.copy(ub, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_mutated_snapshot_union_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred: T.Buffer((4,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.Pipelined(4, num_stages=2):
                before = T.bind(pred[i] > 0)
                pred[i] = 0
                after = T.bind(pred[i] > 0)
                if before:
                    T.copy(A[(2 * i) * tile : (2 * i + 1) * tile], ub)
                    T.copy(ub, C[(2 * i) * tile : (2 * i + 1) * tile])
                if after:
                    T.copy(A[(2 * i + 1) * tile : (2 * i + 2) * tile], ub)
                    T.copy(ub, C[(2 * i + 1) * tile : (2 * i + 2) * tile])

    return main


def _make_guarded_inner_loop_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((16 * tile,), "float32"), C: T.Buffer((16 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for outer in T.serial(4):
                if outer % 2 == 0:
                    for inner in T.Pipelined(3, num_stages=2):
                        offset = (outer * 3 + inner) * tile
                        T.copy(A[offset : offset + tile], ub)
                        with T.SimdVF():
                            mask = T.simd.pset(32)
                            value = T.simd.vld(ub[0])
                            T.simd.vsts(ub[0], value, mask)
                        T.copy(ub, C[offset : offset + tile])

    return main


def _make_guarded_nested_pipeline_program():
    block_m = 32
    block_n = 32
    mad_m = 16
    mad_n = 16
    mad_k = 16
    k_tiles = 3
    block_k = k_tiles * mad_k

    @T.prim_func
    def main(
        A: T.Buffer((2 * block_m, block_k), "bfloat16"),
        B: T.Buffer((2 * block_n, block_k), "bfloat16"),
        enabled: T.int32,
    ):
        with T.Kernel(1):
            l1a = T.alloc_l1((block_m, block_k), "bfloat16")
            l1b = T.alloc_l1((block_n, block_k), "bfloat16")
            l0a = T.alloc_l0a((mad_m, mad_k), "bfloat16")
            l0b = T.alloc_l0b((mad_n, mad_k), "bfloat16")
            l0c = T.alloc_l0c((mad_m, mad_n), "float32")
            T.annotate_buffer_versions(
                {
                    l1a: (2, "counter"),
                    l1b: (2, "counter"),
                    l0a: (2, "counter"),
                    l0b: (2, "counter"),
                }
            )
            for block in T.Pipelined(
                2,
                num_stages=2,
                annotations={"multi_buffer_eligible": [l1a, l1b]},
            ):
                if enabled > 0:
                    T.copy(A[block * block_m : (block + 1) * block_m, :], l1a)
                    T.copy(B[block * block_n : (block + 1) * block_n, :], l1b)
                    for k in T.Pipelined(
                        k_tiles,
                        num_stages=2,
                        annotations={"multi_buffer_eligible": [l0a, l0b]},
                    ):
                        T.copy(l1a[:mad_m, k * mad_k : (k + 1) * mad_k], l0a)
                        T.copy(l1b[:mad_n, k * mad_k : (k + 1) * mad_k], l0b)
                        T.gemm(
                            l0a,
                            l0b,
                            l0c,
                            transpose_B=True,
                            clear_accum=k == 0,
                        )

    return main


def _make_mismatched_sibling_guard_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred_a: T.int32,
        pred_b: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            if pred_a > 0:
                for i in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    T.copy(ub, C[i * tile : (i + 1) * tile])
            if pred_b > 0:
                for i in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                    T.copy(A[(i + 2) * tile : (i + 3) * tile], ub)
                    T.copy(ub, C[(i + 2) * tile : (i + 3) * tile])

    return main


def _make_iteration_guarded_sibling_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((8 * tile,), "float32"),
        C: T.Buffer((8 * tile,), "float32"),
        pred_a: T.Buffer((4,), "int32"),
        pred_b: T.Buffer((4,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                if pred_a[i] > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    T.copy(ub, C[i * tile : (i + 1) * tile])
            for j in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                if pred_b[j] > 0:
                    T.copy(A[(j + 4) * tile : (j + 5) * tile], ub)
                    T.copy(ub, C[(j + 4) * tile : (j + 5) * tile])

    return main


def _make_different_depth_sibling_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred_a: T.int32,
        pred_b: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                if pred_a > i:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    T.copy(ub, C[i * tile : (i + 1) * tile])
            for wrapper in T.serial(1):
                for j in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                    if pred_b > j + wrapper:
                        T.copy(A[(j + 2) * tile : (j + 3) * tile], ub)
                        T.copy(ub, C[(j + 2) * tile : (j + 3) * tile])

    return main


def _make_nested_loop_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for outer in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                for inner in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                    offset = (outer * 2 + inner) * tile
                    T.copy(A[offset : offset + tile], ub)
                    T.copy(ub, C[offset : offset + tile])

    return main


def _make_nested_alias_loop_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((tile,), "float32")
            alias = T.reshape(base, (tile // 2, 2))
            T.annotate_buffer_versions({base: 2})
            for outer in T.serial(2, annotations={"multi_buffer_eligible": [base]}):
                for inner in T.serial(2, annotations={"multi_buffer_eligible": [alias]}):
                    offset = (outer * 2 + inner) * tile
                    alias[0, 0] = A[offset]
                    C[offset] = alias[0, 0]

    return main


def _make_auto_nested_alias_loop_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        with T.Kernel(1):
            base = T.alloc_shared((tile,), "float32")
            alias = T.reshape(base, (tile // 2, 2))
            for outer in T.serial(2):
                base[0] = A[outer]
                for inner in T.serial(2):
                    offset = outer * 2 + inner
                    alias[0, 0] = A[offset]
                    C[offset] = alias[0, 0]

    return main


def _make_nested_guarded_access_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((10 * tile,), "float32"),
        C: T.Buffer((32 * tile,), "float32"),
    ):
        with T.Kernel(1):
            chunk_buf = T.alloc_shared((tile,), "float32")
            data_buf = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({chunk_buf: 2, data_buf: 3})
            for chunk in T.Pipelined(2, num_stages=2):
                T.copy(A[chunk * tile : (chunk + 1) * tile], chunk_buf)
                for k in T.Pipelined(4, num_stages=3):
                    if chunk * 4 + k < 6:
                        T.copy(A[(chunk * 4 + k + 2) * tile : (chunk * 4 + k + 3) * tile], data_buf)
                        for repeat in T.serial(2):
                            offset = ((chunk * 4 + k) * 2 + repeat) * tile
                            T.copy(data_buf, C[offset : offset + tile])
                            T.copy(chunk_buf, C[(16 * tile + offset) : (17 * tile + offset)])

    return main


def _make_nonserial_nested_guard_program(mode="counter"):
    tile = 64

    @T.prim_func
    def main():
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for _i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                with T.SimdVF():
                    for j in T.Parallel(tile):
                        if j < tile // 2:
                            ub[j] = j
                            ub[j] = ub[j] + 1.0

    return main


def _make_nested_snapshot_epoch_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred_a: T.Buffer((4,), "int32"),
        pred_b: T.Buffer((4,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                if pred_a[i] > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                for _repeat in T.serial(1):
                    if pred_b[i] > 0:
                        T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_optional_nested_endpoint_program(mode=None):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        enabled: T.int32,
        repeat_count: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, mode) if mode is not None else 2})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                if enabled > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                for _repeat in T.serial(repeat_count):
                    T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_lexical_optional_endpoint_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((tile,), "float32"), repeat_count: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.copy(A, ub)
            for _repeat in T.serial(repeat_count):
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)

    return main


def _make_unrelated_dependent_nested_extent_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            unrelated = T.alloc_shared((1,), "int32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                for outer in T.serial(2):
                    for inner in T.serial(outer + 1):
                        unrelated[0] = outer + inner
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_equivalent_sibling_snapshot_program():
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            for i in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                if pred > 0:
                    T.copy(A[i * tile : (i + 1) * tile], ub)
                    T.copy(ub, C[i * tile : (i + 1) * tile])
            for j in T.serial(2, annotations={"multi_buffer_eligible": [ub]}):
                if pred > 0:
                    T.copy(A[(j + 2) * tile : (j + 3) * tile], ub)
                    T.copy(ub, C[(j + 2) * tile : (j + 3) * tile])

    return main


def _make_fill_initialization_program(position="head", partial=False, guarded=False):
    tile = 64

    @T.prim_func
    def main(
        A: T.Buffer((4 * tile,), "float32"),
        C: T.Buffer((4 * tile,), "float32"),
        pred: T.int32,
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            if position == "head":
                if guarded:
                    if pred > 0:
                        T.fill(ub, 0)
                elif partial:
                    T.fill(ub[0 : tile // 2], 0)
                else:
                    T.fill(ub, 0)
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                T.copy(ub, C[i * tile : (i + 1) * tile])
            if position == "tail":
                T.fill(ub, 0)

    return main


def _make_scoped_fill_initialization_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), enabled: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            if enabled > 0:
                with T.SimdVF():
                    T.fill(ub, 0)
                for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                    T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_fill_with_extra_pointer_write_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            with T.Task():
                T.fill(ub, 0)
                T.evaluate(T.access_ptr(ub[0], "w", 1))
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_fill_reading_target_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            T.fill(ub, ub[0])
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_fill_with_target_dependent_region_program(bound):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            if bound == "min":
                T.fill(ub[ub[0] : ub[0] + tile // 2], 0)
            else:
                T.fill(ub[0 : ub[0]], 0)
            for i in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_assumed_fill_initialization_program(explicit_claim=True):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            T.assume(ub[0] >= 0)
            T.fill(ub, 0)
            for i in T.Pipelined(
                4,
                num_stages=2,
                annotations={"multi_buffer_eligible": [ub]} if explicit_claim else {},
            ):
                T.copy(A[i * tile : (i + 1) * tile], ub)

    return main


def _make_stepped_iteration_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (3, "iteration")})
            for outer in T.serial(2):
                for inner in T.serial(0, 4, 2, annotations={"multi_buffer_eligible": [ub]}):
                    index = outer * 2 + inner // 2
                    T.copy(A[index * tile : (index + 1) * tile], ub)
                    T.copy(ub, C[index * tile : (index + 1) * tile])

    return main


def _make_fill_with_nested_control_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((4 * tile,), "float32"), C: T.Buffer((4 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: 2})
            with T.Task():
                for j in T.serial(2):
                    T.fill(ub[j * (tile // 2) : (j + 1) * (tile // 2)], 0)
            for i in T.serial(4, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[i * tile : (i + 1) * tile], ub)
                T.copy(ub, C[i * tile : (i + 1) * tile])

    return main


def _make_head_tail_program(wrapped=False):
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((10 * tile,), "float32"), C: T.Buffer((10 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: (2, "counter")})
            if wrapped:
                for _head in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                    T.copy(A[0:tile], ub)
                    T.copy(ub, C[0:tile])
            else:
                T.copy(A[0:tile], ub)
            for i in T.Pipelined(8, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[(i + 1) * tile : (i + 2) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[(i + 1) * tile : (i + 2) * tile])
            if wrapped:
                for _tail in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                    T.copy(A[9 * tile : 10 * tile], ub)
                    T.copy(ub, C[9 * tile : 10 * tile])

    return main


def _make_for_one_tail_owner_program():
    tile = 64

    @T.prim_func
    def main(A: T.Buffer((5 * tile,), "float32"), C: T.Buffer((5 * tile,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({ub: "counter"})
            for layer in T.Pipelined(4, num_stages=2, annotations={"multi_buffer_eligible": [ub]}):
                T.copy(A[layer * tile : (layer + 1) * tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                # A regular layer performs an extra residual-style update that
                # the final tail layer does not need.
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[layer * tile : (layer + 1) * tile])
            for tail in T.serial(1, annotations={"multi_buffer_eligible": [ub]}):
                offset = (tail + 4) * tile
                T.copy(A[offset : offset + tile], ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    value = T.simd.vld(ub[0])
                    T.simd.vsts(ub[0], value, mask)
                T.copy(ub, C[offset : offset + tile])

    return main


def _device_script(program):
    return lower(program, target="ascend").device_mod.script()


def _pass_snapshots(program, pass_names):
    snapshots = {name: [] for name in pass_names}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in snapshots:
                snapshots[info.name].append(mod.script())

    with tvm.transform.PassContext(opt_level=3, instruments=[Capture()]):
        lower(program, target="ascend")
    return snapshots


def _pass_script(program, pass_name, occurrence=0):
    scripts = _pass_snapshots(program, {pass_name})[pass_name]
    assert scripts, f"Pass {pass_name} did not run"
    return scripts[occurrence]


def _prepare_script(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(determine_target("ascend"))(mod)
    for transform in (
        ascend_transform.NormalizeControlFlowForSchedule,
        ascend_transform.NormalizeNoConflictHints,
        ascend_transform.MaterializeScheduleUnits,
        ascend_transform.AnnotateMultiBufferEligible,
        ascend_transform.EstimateLatency,
        ascend_transform.AutoSchedule,
        ascend_transform.AssignCore,
        ascend_transform.PrepareMultiBuffer,
    ):
        mod = transform()(mod)
    return mod.script()


def _prepare_without_control_normalization(program):
    mod = tvm.IRModule.from_expr(program.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(determine_target("ascend"))(mod)
    mod = ascend_transform.MaterializeScheduleUnits()(mod)
    mod = ascend_transform.AnnotateMultiBufferEligible()(mod)
    mod = ascend_transform.EstimateLatency()(mod)
    mod = ascend_transform.AutoSchedule()(mod)
    mod = ascend_transform.AssignCore()(mod)
    return ascend_transform.PrepareMultiBuffer()(mod)


def _materialize_without_control_normalization(program):
    mod = _prepare_without_control_normalization(program)
    mod = ascend_transform.ResolveCore()(mod)
    mod = ascend_transform.InsertSync()(mod)
    return ascend_transform.MaterializeMultiBuffer()(mod)


def _counter_name(script):
    match = re.search(r"([A-Za-z0-9_]+_version_counter_[A-Za-z0-9_]*)\[0\]", script)
    assert match, script
    return match.group(1)


def _counter_names(script, storage=None):
    prefix = rf"{storage}_" if storage else r"[A-Za-z0-9_]+_"
    return list(dict.fromkeys(re.findall(rf"({prefix}version_counter_[A-Za-z0-9_]*)\[0\]", script)))


def _storage_domain_guards(script, storage):
    result = []
    marker = '"tl.storage_epoch_guard_map": {'
    pattern = rf"(?<![\w]){re.escape(storage)}(?:\.data)?: ([^,}}]+)"
    for line in script.splitlines():
        if marker not in line:
            continue
        guard_map = line.split(marker, 1)[1].split("}", 1)[0]
        if match := re.search(pattern, guard_map):
            result.append(match.group(1))
    return result


def _normalized_event_expressions(script, operation, hard_event, counter):
    expressions = re.findall(rf'T\.ascend_{operation}_flag\("{hard_event}", ([^\n]+)\)', script)
    counter_load = re.escape(f"{counter}[0]")
    return {re.sub(counter_load, "COUNTER[0]", expression) for expression in expressions if re.search(counter_load, expression)}


def _event_expressions(script, operation, hard_event):
    return re.findall(rf'T\.ascend_{operation}_flag\("{hard_event}", ([^\n]+)\)', script)


def _assert_no_sibling_l0_fallback(script, counter):
    assert 'T.ascend_pipe_barrier("PIPE_MTE1")' not in script
    for operation in ("set", "wait"):
        acquire = re.findall(rf'T\.ascend_{operation}_flag\("MTE1_M", ([^\n]+)\)', script)
        assert acquire and all(counter in expression for expression in acquire)

        release = re.findall(rf'T\.ascend_{operation}_flag\("M_MTE1", ([^\n]+)\)', script)
        constant_release = [expression for expression in release if counter not in expression]
        # Only the two-slot release ring's kernel prologue/epilogue remains.
        assert len(constant_release) == 2


def _direct_enclosing_if(lines, line_index):
    indent = len(lines[line_index]) - len(lines[line_index].lstrip())
    for candidate in reversed(lines[:line_index]):
        candidate_indent = len(candidate) - len(candidate.lstrip())
        if candidate_indent >= indent:
            continue
        stripped = candidate.strip()
        return stripped[3:-1] if stripped.startswith("if ") and stripped.endswith(":") else None
    return None


def _task_attr_before(script, statement):
    lines = script.splitlines()
    statement_line = next(i for i, line in enumerate(lines) if statement in line)
    return next(line for line in reversed(lines[:statement_line]) if '"tl.ascend_task"' in line)


def test_buffer_version_annotation_accepts_count_tuple_and_mode():
    @T.prim_func
    def main():
        with T.Kernel(1):
            fixed = T.alloc_shared((64,), "float32")
            explicit = T.alloc_shared((64,), "float32")
            inferred = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions(
                {
                    fixed: 2,
                    explicit: (3, "counter"),
                    inferred: "iteration",
                }
            )

    script = main.script()
    version_map = re.search(r'"tl.buffer_versions_map": \{[^}]*\}', script)
    assert version_map is not None
    assert "fixed: 2" in version_map.group(0)
    assert "explicit: 3" in version_map.group(0)
    assert "inferred" not in version_map.group(0)
    mode_map = re.search(r'"tl.buffer_version_mode": \{[^}]*\}', script)
    assert mode_map is not None
    assert 'explicit: "counter"' in mode_map.group(0)
    assert 'inferred: "iteration"' in mode_map.group(0)


def test_invalid_buffer_version_mode_is_rejected():
    with pytest.raises(ValueError, match="buffer version mode"):
        _make_guarded_program("lexical")


def test_invalid_buffer_version_tuple_is_rejected():
    with pytest.raises(ValueError, match=r"must be \(num_versions, mode\)"):

        @T.prim_func
        def main():
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "float32")
                T.annotate_buffer_versions({ub: (2, "counter", "extra")})


def test_buffer_version_tuple_requires_valid_mode():
    with pytest.raises(ValueError, match="buffer version mode"):

        @T.prim_func
        def main():
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "float32")
                T.annotate_buffer_versions({ub: (2, None)})


def test_aliases_combine_version_count_and_mode():
    @T.prim_func
    def main():
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            alias = T.reshape(ub, (32, 2))
            T.annotate_buffer_versions({ub: 2, alias: "counter"})

    script = main.script()
    assert re.search(r'"tl.buffer_versions_map": \{[^:}]+: 2\}', script)
    assert re.search(r'"tl.buffer_version_mode": \{[^:}]+: "counter"\}', script)


def test_aliases_reject_conflicting_buffer_version_counts():
    with pytest.raises(ValueError, match="same version count"):

        @T.prim_func
        def main():
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "float32")
                alias = T.reshape(ub, (32, 2))
                T.annotate_buffer_versions({ub: 2, alias: 3})


def test_aliases_reject_conflicting_buffer_version_modes():
    with pytest.raises(ValueError, match="aliases of one buffer storage"):

        @T.prim_func
        def main():
            with T.Kernel(1):
                ub = T.alloc_shared((64,), "float32")
                alias = T.reshape(ub, (32, 2))
                T.annotate_buffer_versions({ub: "counter", alias: "iteration"})


def test_iteration_mode_keeps_affine_fast_path():
    script = _device_script(_make_offset_program(enable_offset=False, mode="iteration"))
    assert "version_counter" not in script
    assert re.search(r"i(?:_\d+)? % 2", script)


def test_iteration_mode_allows_guarded_epoch():
    program = _make_guarded_program("iteration")
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    script = _pass_script(program, "tl.InsertSync")
    assert "version_counter" not in script
    assert _storage_domain_guards(prepared, "ub") == []
    lines = script.splitlines()
    syncs = [i for i, line in enumerate(lines) if "ascend_set_flag" in line or "ascend_wait_flag" in line]
    assert syncs
    assert all(_direct_enclosing_if(lines, i) is None for i in syncs)


def test_manual_multi_buffer_keeps_lexical_iteration_clock():
    program = _make_guarded_manual_storage_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    assert _storage_domain_guards(prepared, "ub") == []

    inserted = _pass_script(program, "tl.InsertSync")
    lines = inserted.splitlines()
    for hard_event in ("MTE2_V", "V_MTE3"):
        syncs = [
            i for i, line in enumerate(lines) if f'"{hard_event}"' in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
        ]
        assert len(syncs) == 2
        assert all("% 2" in lines[i] for i in syncs)
        assert all(_direct_enclosing_if(lines, i) is None for i in syncs)


def test_iteration_mode_warns_for_non_affine_loop_nest(capfd):
    capfd.readouterr()
    script = _pass_script(
        _make_dependent_extent_program(mode="iteration"),
        "tl.PrepareMultiBuffer",
    )
    assert "version_counter" not in script
    assert "uses a non-affine loop nest; consider counter mode" in capfd.readouterr().err


@pytest.mark.parametrize("kind", ["guard", "loop_break"])
def test_iteration_mode_warns_for_non_affine_control_path(kind, capfd):
    capfd.readouterr()
    script = _pass_script(
        _make_non_affine_control_path_program(kind),
        "tl.PrepareMultiBuffer",
    )
    assert "version_counter" not in script
    assert "uses a non-affine loop nest; consider counter mode" in capfd.readouterr().err


@pytest.mark.parametrize(
    "builder",
    [_make_mutable_range_program, _make_loop_local_range_var_program],
    ids=["mutable-read", "loop-local-var"],
)
def test_iteration_mode_warns_for_non_affine_loop_range(builder, capfd):
    capfd.readouterr()
    script = _pass_script(builder(), "tl.PrepareMultiBuffer")
    assert "version_counter" not in script
    assert "uses a non-affine loop nest; consider counter mode" in capfd.readouterr().err


def test_iteration_mode_does_not_recommend_unavailable_counter(capfd):
    capfd.readouterr()
    script = _pass_script(_make_non_affine_offset_program(), "tl.PrepareMultiBuffer")
    assert "version_counter" not in script
    warning = capfd.readouterr().err
    assert "uses a non-affine loop nest" in warning
    assert "consider counter mode" not in warning


def test_auto_mode_falls_back_when_counter_stage_is_unavailable(capfd):
    capfd.readouterr()
    script = _pass_script(_make_non_affine_offset_program(mode=None), "tl.PrepareMultiBuffer")
    assert "version_counter" not in script
    warning = capfd.readouterr().err
    assert "uses a non-affine loop nest" in warning
    assert "consider counter mode" not in warning


def test_regular_auto_mode_keeps_affine_fast_path():
    program = _make_offset_program(enable_offset=False, mode=None)
    script = _device_script(program)
    assert "version_counter" not in script
    assert re.search(r"i(?:_\d+)? % 2", script)


def test_counter_mode_does_not_materialize_single_version_ring():
    program = _make_single_version_counter_mode_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    inserted = _pass_script(program, "tl.InsertSync")
    assert "version_counter" not in inserted
    assert "tl.buffer_versions_map" not in prepared
    assert "tl.multi_buffer_counter_map" not in prepared
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1 and "__cond_" in guards[0]
    assert "T.sblock_alloc_buffer((64,)" in inserted
    assert "% 1" not in inserted
    lines = inserted.splitlines()
    barrier = next(i for i, line in enumerate(lines) if "ascend_pipe_barrier" in line)
    assert "__cond_" in (_direct_enclosing_if(lines, barrier) or "")


@pytest.mark.parametrize("stage, expected_counters", [("store", 1), ("compute", 2)])
def test_real_example_compress_lowers_with_counter(stage, expected_counters):
    from examples.ascend.example_compress import _compress_and_update_state_decode_tl_ascend

    program = _compress_and_update_state_decode_tl_ascend.get_tir(
        DIM=128,
        COMPRESS_RATIO=4,
        OVERLAP_RATIO=2,
        HAS_APE=True,
        STAGE=stage,
    )
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    assert "version_counter" in prepared
    assert 'T.sblock_alloc_buffer((1,), "int32", scope="local.var")' in prepared
    assert len(_counter_names(prepared)) == expected_counters


def test_real_example_compress_avoids_inner_mte2_barriers():
    from examples.ascend.example_compress import _compress_and_update_state_decode_tl_ascend

    program = _compress_and_update_state_decode_tl_ascend.get_tir(
        DIM=128,
        COMPRESS_RATIO=4,
        OVERLAP_RATIO=2,
        HAS_APE=True,
        STAGE="compute",
    )
    inserted = _pass_script(program, "tl.InsertSync")
    assert 'T.ascend_pipe_barrier("PIPE_MTE2")' not in inserted


def test_sibling_iteration_mode_is_rejected():
    with pytest.raises(Exception, match="requires exactly one owner"):
        _pass_script(_make_sibling_loop_program(mode="iteration"), "tl.PrepareMultiBuffer")


def test_sibling_version_count_uses_one_storage_level_ring():
    program = _make_mismatched_sibling_version_program()
    scheduled = _pass_script(program, "tl.AutoSchedule")
    assert re.search(r'"tl.buffer_versions_map": \{ub: 3\}', scheduled)

    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    counter = _counter_name(prepared)
    assert prepared.count(f"{counter}[0] = {counter}[0] + 1") == 2

    materialized = _pass_script(program, "tl.MaterializeMultiBuffer")
    assert f"{counter}[0] % 3" in materialized


@pytest.mark.parametrize("fixed_versions", [True, False])
def test_guarded_epoch_uses_one_counter_for_data_and_flags(fixed_versions):
    script = _device_script(_make_guarded_program(fixed_versions=fixed_versions))
    counter = _counter_name(script)
    assert f"{counter}[0] % 2" in script
    assert re.search(rf"ascend_(?:set|wait)_flag\([^\n]*{counter}\[0\]", script)
    assert f"{counter}[0] = {counter}[0] + 1" in script

    lines = script.splitlines()
    guard_line = next(i for i, line in enumerate(lines) if line.lstrip().startswith("if ") and ("enabled" in line or "__cond_0" in line))
    guard_indent = len(lines[guard_line]) - len(lines[guard_line].lstrip())
    dynamic_syncs = [
        line for line in lines[guard_line + 1 :] if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
    ]
    assert dynamic_syncs
    assert all(len(line) - len(line.lstrip()) > guard_indent for line in dynamic_syncs)


def test_counter_same_iteration_flags_shrink_independently_from_data_ring():
    snapshots = _pass_snapshots(
        _make_counter_flag_shrink_program(),
        {"tl.InsertSync", "tl.MaterializeMultiBuffer"},
    )
    script = snapshots["tl.InsertSync"][0]
    counter = _counter_name(script)

    for hard_event in ("MTE2_V", "V_MTE3"):
        for operation in ("set", "wait"):
            events = re.findall(rf'T\.ascend_{operation}_flag\("{hard_event}", ([^\n]+)\)', script)
            assert len(events) == 2
            assert all(counter not in event for event in events)

    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    assert f"{counter}[0] % 2" in materialized

    # A loop-carried release/acquire channel still follows the two-version
    # ring; only same-iteration flag rings participate in shrink.
    guarded = _pass_script(_make_guarded_program(fixed_versions=True), "tl.InsertSync")
    guarded_counter = _counter_name(guarded)
    for operation in ("set", "wait"):
        assert _normalized_event_expressions(guarded, operation, "MTE3_MTE2", guarded_counter) == {"COUNTER[0] % 2"}


def test_mutated_condition_uses_one_snapshot_for_data_flags_and_counter():
    script = _pass_script(_make_mutating_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    protocol = [
        i
        for i, line in enumerate(lines)
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line or f"{counter}[0] = {counter}[0] +" in line)
    ]
    assert protocol
    guards = {_direct_enclosing_if(lines, i) for i in protocol}
    assert all(guard and "__cond_" in guard and "pred" not in guard for guard in guards)


def test_owner_guard_uses_only_direct_child_guards():
    script = _pass_script(_make_mutable_bind_scoped_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    protocol = [
        i
        for i, line in enumerate(lines)
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line or f"{counter}[0] = {counter}[0] +" in line)
    ]
    assert protocol
    guards = [_direct_enclosing_if(lines, i) for i in protocol]
    guard_text = " ".join(guard for guard in guards if guard)
    assert "outer_guard" in guard_text
    assert "access_guard" not in guard_text
    assert "pred[" not in guard_text
    assert not re.search(r"\bfree\d+\b", script)


def test_nested_extent_does_not_contribute_to_owner_epoch():
    script = _pass_script(_make_mutable_bind_nested_extent_program(), "tl.PrepareMultiBuffer")
    assert "version_counter" not in script
    assert "tl.multi_buffer_counter_map" not in script
    assert _storage_domain_guards(script, "ub") == []
    assert not re.search(r"\bfree\d+\b", script)


def test_sibling_loops_share_monotonic_counter():
    program = _make_sibling_loop_program()
    scheduled = _pass_script(program, "tl.AutoSchedule")
    assert re.search(r'"tl.buffer_versions_map": \{ub: 2\}', scheduled)

    script = _device_script(program)
    counter = _counter_name(script)
    # The two explicit-unroll instances become sibling inner loops. Their
    # Increments update one storage counter instead of restarting at each loop.
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 2
    assert f"{counter}[0] % 2" in script


def test_bf16_mnk_auto_marks_every_k_loop():
    program = _make_bf16_mnk_program()
    eligible = _pass_script(program, "tl.AnnotateMultiBufferEligible")
    annotation_lines = [line for line in eligible.splitlines() if "multi_buffer_eligible" in line]
    k_lines = [line for line in annotation_lines if "for k_tile" in line]
    assert len(k_lines) == 4
    assert all("l0a" in line and "l0b" in line for line in k_lines)
    block_line = next(line for line in annotation_lines if "for block" in line)
    assert "l0a" not in block_line and "l0b" not in block_line


def test_fp8_mnk_auto_marks_every_k_loop():
    eligible = _pass_script(_make_fp8_mnk_copy_program(), "tl.AnnotateMultiBufferEligible")
    annotation_lines = [line for line in eligible.splitlines() if "multi_buffer_eligible" in line]
    k_lines = [line for line in annotation_lines if "for k_tile" in line]
    assert len(k_lines) == 4
    assert all("a_ub" in line and "b_ub" in line for line in k_lines)


def test_auto_promotes_partial_row_writers_to_consuming_outer_loop():
    snapshots = _pass_snapshots(
        _make_row_writer_then_consumer_program(),
        {"tl.AnnotateMultiBufferEligible", "tl.PrepareMultiBuffer"},
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    annotation_lines = [line for line in eligible.splitlines() if "multi_buffer_eligible" in line]
    outer_line = next(line for line in annotation_lines if "for bdim" in line)
    inner_line = next(line for line in annotation_lines if "for i" in line)
    assert "score_ub" in outer_line and "latent_ub" in outer_line
    assert "score_ub" not in inner_line and "latent_ub" not in inner_line
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    assert prepared.count("tl.multi_buffer_broadcast_fill") == 1


def test_auto_owner_keeps_structured_fill_loop_broadcast_only():
    snapshots = _pass_snapshots(
        _make_structured_fill_initialization_program(),
        {
            "tl.AnnotateMultiBufferEligible",
            "tl.PrepareMultiBuffer",
            "tl.MaterializeMultiBuffer",
        },
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    fill_loop = next(line for line in eligible.splitlines() if "for r" in line)
    owner_loop = next(line for line in eligible.splitlines() if "for bdim" in line)
    assert "ub" not in fill_loop
    assert "ub" in owner_loop

    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    assert prepared.count("tl.multi_buffer_broadcast_fill") == 1
    counter = _counter_name(prepared)
    assert prepared.count(f"{counter}[0] = {counter}[0] + 1") == 1

    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    assert re.search(r"T\.fill\(T\.region\(ub_\d+\[0, r(?:_\d+)?, 0\], 2, 2, 1, 64\), 0\)", materialized)


def test_auto_owner_claims_loop_that_fills_then_accesses_storage():
    snapshots = _pass_snapshots(
        _make_fill_then_access_program(),
        {
            "tl.AnnotateMultiBufferEligible",
            "tl.PrepareMultiBuffer",
            "tl.MaterializeMultiBuffer",
        },
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    owner_loop = next(line for line in eligible.splitlines() if "for bdim" in line)
    assert "ub" in owner_loop

    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    assert "tl.multi_buffer_broadcast_fill" not in prepared
    counter = _counter_name(prepared)
    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    fill_line = next(line for line in materialized.splitlines() if "T.fill" in line)
    assert f"{counter}[0] % 2" in fill_line


def test_auto_owner_keeps_conditional_write_first_promotion():
    eligible = _pass_script(_make_guarded_program(), "tl.AnnotateMultiBufferEligible")
    owner_line = next(line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for w" in line)
    assert "ub" in owner_line


def test_auto_owner_does_not_enter_atomic_task():
    eligible = _pass_script(_make_fill_with_nested_control_program(), "tl.AnnotateMultiBufferEligible")
    task_loop_claims = [line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for j" in line]
    assert not task_loop_claims


def test_auto_owner_rejects_partial_frontier_with_external_consumer():
    eligible = _pass_script(
        _make_partial_owner_with_external_consumer_program(),
        "tl.AnnotateMultiBufferEligible",
    )
    owner_line = next(line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for i" in line)
    assert "ub" not in owner_line


@pytest.mark.parametrize(
    "explicit_claim,task_wrapped",
    [(False, False), (True, False), (False, True)],
)
def test_owner_rewrites_versioned_storage_in_assume_guard(explicit_claim, task_wrapped):
    snapshots = _pass_snapshots(
        _make_versioned_storage_assume_program(explicit_claim, task_wrapped),
        {"tl.AnnotateMultiBufferEligible", "tl.MaterializeMultiBuffer"},
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    owner_line = next(line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for i" in line)
    assert "ub" in owner_line

    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    assume_lines = [line for line in materialized.splitlines() if "assume" in line and "ub_" in line]
    assert assume_lines
    assert all("% 2" in line and ", 0]" in line for line in assume_lines)


def test_owner_rejects_read_first_assume_guard():
    eligible = _pass_script(
        _make_versioned_storage_assume_program(read_first=True),
        "tl.AnnotateMultiBufferEligible",
    )
    owner_line = next(line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for i" in line)
    assert "ub" not in owner_line


def test_owner_rewrites_assume_guard_on_nested_control():
    snapshots = _pass_snapshots(
        _make_versioned_storage_assume_program(nested_control=True),
        {"tl.AnnotateMultiBufferEligible", "tl.MaterializeMultiBuffer"},
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    owner_line = next(line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for i" in line)
    assert "ub" in owner_line

    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    assume_lines = [line for line in materialized.splitlines() if "assume" in line and "ub_" in line]
    assert assume_lines
    assert all("% 2" in line and ", 0]" in line for line in assume_lines)


def test_materialize_rewrites_versioned_storage_condition_guard():
    mod = _materialize_without_control_normalization(_make_versioned_storage_condition_guard_program())
    condition = next(line for line in mod.script().splitlines() if "if " in line and "ub_" in line)
    assert "% 2" in condition and ", 0]" in condition


def test_prepare_rejects_unnormalized_versioned_storage_loop_bound():
    with pytest.raises(Exception, match="unnormalized loop bound"):
        _prepare_without_control_normalization(_make_versioned_storage_control_program(explicit_claim=False))


def test_sibling_owner_assumes_share_counter_version():
    snapshots = _pass_snapshots(
        _make_sibling_assume_program(),
        {"tl.AnnotateMultiBufferEligible", "tl.MaterializeMultiBuffer"},
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    owner_lines = [line for line in eligible.splitlines() if "multi_buffer_eligible" in line]
    assert sum("ub" in line for line in owner_lines if "for i" in line or "for j" in line) == 2

    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    counter = _counter_name(materialized)
    assume_lines = [line for line in materialized.splitlines() if "assume" in line and "ub_" in line]
    assert assume_lines
    assert all(f"{counter}[0] % 2" in line for line in assume_lines)
    assert materialized.count(f"{counter}[0] = {counter}[0] + 1") == 2


def test_manual_storage_warns_when_dropping_explicit_auto_claim(capfd):
    capfd.readouterr()
    eligible = _pass_script(
        _make_manual_storage_with_explicit_auto_claim_program(),
        "tl.AnnotateMultiBufferEligible",
    )
    owner_line = next(line for line in eligible.splitlines() if "multi_buffer_eligible" in line and "for i" in line)
    assert "ub" not in owner_line
    warning = capfd.readouterr().err
    assert "Ignoring explicit 'multi_buffer_eligible' claim for storage ub" in warning
    assert "because it is already manually multi-buffered" in warning


def test_sibling_iteration_override_is_rejected():
    with pytest.raises(Exception, match="requires exactly one owner"):
        _device_script(_make_bf16_mnk_program(mode="iteration"))


def test_bf16_mnk_sibling_loops_share_l0_counters():
    script = _device_script(_make_bf16_mnk_program())
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 4
    # L0A and L0B have the same guard and epoch stream, so all four sibling K
    # loops use one logical counter. Each MTE1<->M direction must keep one
    # canonical base across every owner, while the allocator may choose a
    # non-zero base because other flag protocols are present.
    for event in ("MTE1_M", "M_MTE1"):
        sets = _normalized_event_expressions(script, "set", event, counter)
        waits = _normalized_event_expressions(script, "wait", event, counter)
        assert len(sets) == 1
        assert waits == sets
    _assert_no_sibling_l0_fallback(script, counter)


def test_counter_data_ring_uses_int32_layout_index():
    program = _make_counter_l1_layout_program()
    snapshots = _pass_snapshots(
        program,
        {"tl.PrepareMultiBuffer", "tl.ResolveCore", "tl.MaterializeMultiBuffer"},
    )
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    resolved = snapshots["tl.ResolveCore"][0]
    materialized = snapshots["tl.MaterializeMultiBuffer"][0]
    counter = _counter_name(materialized)
    assert f"{counter}[0] % 3" in materialized
    assert 'T.Cast("int32"' not in materialized

    for statement in (f"{counter}[0] = 0", f"{counter}[0] = {counter}[0] + 1"):
        assert '"core_mask": T.int64(3)' in _task_attr_before(prepared, statement)
        assert '"core_mask": T.int64(2)' in _task_attr_before(resolved, statement)

    # LowerTileOp substitutes physical indices into int32 layout variables.
    # Full lowering must therefore accept the counter-derived L1 version slot.
    _device_script(program)


def test_bf16_mnk_single_k_tile_keeps_release_ring_across_siblings():
    script = _device_script(_make_bf16_mnk_program(k_tiles=1))
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 4
    sets = _normalized_event_expressions(script, "set", "M_MTE1", counter)
    waits = _normalized_event_expressions(script, "wait", "M_MTE1", counter)
    assert len(sets) == 1
    assert waits == sets
    _assert_no_sibling_l0_fallback(script, counter)


def test_cross_sibling_region_conflict_uses_lexical_boundary():
    script = _pass_script(_make_cross_sibling_region_swap_program(), "tl.InsertSync")
    counter = _counter_name(script)
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 3
    for operation in ("set", "wait"):
        assert f'T.ascend_{operation}_flag("V_MTE2", 0)' in script
        assert f'T.ascend_{operation}_flag("MTE2_V", 0)' in script


def test_missing_counter_channel_becomes_lexical_dependency():
    for mode in (None, "counter"):
        script = _pass_script(_make_missing_channel_endpoint_program(mode=mode), "tl.InsertSync")
        counter = _counter_name(script)
        assert script.count(f"{counter}[0] = {counter}[0] + 1") == 3
        # The V/S obligation has no common counter channel.  It is retained as
        # an unguarded, constant-ID lexical handshake between sibling owners.
        assert 'T.ascend_set_flag("V_S", 0)' in script
        assert 'T.ascend_wait_flag("V_S", 0)' in script


@pytest.mark.parametrize("pipelined", [False, True], ids=["serial", "pipelined"])
def test_missing_counter_channels_become_outer_loop_lexical_dependencies(pipelined):
    script = _pass_script(
        _make_nested_missing_channel_endpoint_program(pipelined),
        "tl.InsertSync",
    )

    # The forward V->S obligation is same-iteration.  The reverse S->V
    # obligation is loop-carried and therefore also has a prologue token and
    # epilogue drain.  Both are ordinary lexical one-slot dependencies at the
    # enclosing loop level.
    assert script.count('T.ascend_set_flag("V_S", 0)') == 1
    assert script.count('T.ascend_wait_flag("V_S", 0)') == 1
    assert script.count('T.ascend_set_flag("S_V", 0)') == 2
    assert script.count('T.ascend_wait_flag("S_V", 0)') == 2


def test_same_pipe_order_eliminates_transitive_sync():
    script = _pass_script(_make_same_pipe_transitive_elimination_program(), "tl.InsertSync")
    # V->MTE2 and MTE2->S are explicit edges.  The two MTE2 copies provide
    # the middle issue-order edge, so the wider direct V->S edge is redundant.
    for operation in ("set", "wait"):
        assert f'T.ascend_{operation}_flag("V_MTE2"' in script
        assert f'T.ascend_{operation}_flag("MTE2_S"' in script
        assert f'T.ascend_{operation}_flag("V_S"' not in script


@pytest.mark.parametrize("mode", ["auto", "counter"])
def test_cross_level_sync_inherits_outer_owner_clock(mode):
    script = _pass_script(_make_cross_level_sync_program(mode), "tl.InsertSync")
    hard_events = set(re.findall(r'T\.ascend_(?:set|wait)_flag\("([^"]+)"', script))
    # The owner has another access outside the extent-one child, so the child's
    # storage-local iteration suffix contains only that child loop.
    assert hard_events == {"MTE2_V", "V_MTE3", "MTE3_MTE2"}

    if mode == "counter":
        counter = _counter_name(script)
        expected_clock = "COUNTER[0] % 2"
        for hard_event in hard_events:
            for operation in ("set", "wait"):
                assert _normalized_event_expressions(script, operation, hard_event, counter) == {expected_clock}
    else:
        assert "version_counter" not in script
        for hard_event in hard_events:
            for operation in ("set", "wait"):
                dynamic = {expression for expression in _event_expressions(script, operation, hard_event) if "%" in expression}
                assert dynamic == {"i % 2"}


@pytest.mark.parametrize("mode", ["iteration", "counter"])
def test_cross_level_loop_carried_sync_uses_outer_owner_clock(mode):
    script = _pass_script(_make_cross_level_sync_program(mode, inner=2), "tl.InsertSync")
    hard_events = set(re.findall(r'T\.ascend_(?:set|wait)_flag\("([^"]+)"', script))
    assert hard_events == {"MTE2_V", "V_MTE3", "MTE3_V", "MTE3_MTE2"}

    if mode == "counter":
        counter = _counter_name(script)
        expected_clock = "COUNTER[0] % 2"
        for hard_event in hard_events:
            for operation in ("set", "wait"):
                assert _normalized_event_expressions(script, operation, hard_event, counter) == {expected_clock}
    else:
        assert "version_counter" not in script
        for hard_event in hard_events:
            for operation in ("set", "wait"):
                dynamic = {expression for expression in _event_expressions(script, operation, hard_event) if "%" in expression}
                assert dynamic == {"i % 2"}


@pytest.mark.parametrize("mode", ["auto", "counter"])
def test_sibling_cross_level_syncs_share_counter_channels(mode):
    script = _pass_script(_make_sibling_cross_level_sync_program(mode, inner=2), "tl.InsertSync")
    counter = _counter_name(script)
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 2
    hard_events = set(re.findall(r'T\.ascend_(?:set|wait)_flag\("([^"]+)"', script))
    assert hard_events == {"MTE2_V", "V_MTE3", "MTE3_V", "MTE3_MTE2"}
    for hard_event in hard_events:
        for operation in ("set", "wait"):
            expressions = _event_expressions(script, operation, hard_event)
            assert sum(counter in expression for expression in expressions) == 2
            assert _normalized_event_expressions(script, operation, hard_event, counter) == {"COUNTER[0] % 2"}


def test_storage_local_cross_iteration_channels_share_counter_ring():
    script = _pass_script(_make_mixed_distance_sibling_cross_level_program(), "tl.InsertSync")
    counter = _counter_name(script)
    # The first owner advances through (i, inner), while the extent-one child
    # in the second owner advances through (j, _once). Both are one step in
    # their storage-local iteration sequence and can share one counter ring.
    for operation in ("set", "wait"):
        events = _event_expressions(script, operation, "MTE3_V")
        assert sum(counter in event for event in events) == 2
        assert _normalized_event_expressions(script, operation, "MTE3_V", counter) == {"COUNTER[0] % 2"}


def test_outer_dependent_inner_extent_switches_to_counter():
    script = _device_script(_make_dependent_extent_program())
    counter = _counter_name(script)
    assert f"{counter}[0] = {counter}[0] + 1" in script
    dynamic_flags = [line for line in script.splitlines() if ("ascend_set_flag" in line or "ascend_wait_flag" in line) and "%" in line]
    assert dynamic_flags
    assert all(counter in line for line in dynamic_flags)


def test_dependent_extent_cross_iteration_uses_loop_tuple_distinctness():
    script = _pass_script(_make_decreasing_dependent_extent_program(), "tl.InsertSync")
    counter = _counter_name(script)
    for operation in ("set", "wait"):
        assert _normalized_event_expressions(script, operation, "MTE3_MTE2", counter) == {"COUNTER[0] % 2"}


def test_sparse_guard_distance_counts_only_executed_epochs():
    script = _pass_script(_make_sparse_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    version_expr = f"{counter}[0] % 2"
    assert version_expr in script
    assert not re.search(r"i(?:_\d+)?\s*%\s*2\s*%", script)
    increment_line = next(i for i, line in enumerate(script.splitlines()) if f"{counter}[0] = {counter}[0] +" in line)
    assert _direct_enclosing_if(script.splitlines(), increment_line) in ("__cond_0", "__cond_0[0]")


def test_guarded_inplace_global_dependency_is_covered_by_counter_domain(capfd):
    capfd.readouterr()
    script = _pass_script(_make_guarded_inplace_global_program(), "tl.InsertSync")
    warning = capfd.readouterr().err

    # The counter-domain MTE2->V->MTE3 path projects its transitive ordering to
    # the equivalent lexical domain and removes the direct GM handshake.
    assert "reverse ordering too loose to bound flag reuse" not in warning
    assert 'T.ascend_set_flag("MTE2_MTE3"' not in script
    assert 'T.ascend_wait_flag("MTE2_MTE3"' not in script


def test_unversioned_loop_carried_sync_counts_guarded_active_epochs():
    script = _pass_script(_make_guarded_reused_global_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    lexical_protocol = [
        i
        for i, line in enumerate(lines)
        if "T.ascend_" in line and '"MTE3_MTE2"' in line and counter not in line and _direct_enclosing_if(lines, i) is not None
    ]
    assert len(lexical_protocol) == 2
    guards = {_direct_enclosing_if(lines, i) for i in lexical_protocol}
    assert len(guards) == 1
    assert "__cond_" in next(iter(guards))


def test_equivalent_counter_and_lexical_guards_keep_counter_domain_distance():
    script = _pass_script(_make_guarded_reused_global_program(), "tl.InsertSync")
    counter = _counter_name(script)
    for hard_event in ("MTE2_V", "V_MTE3"):
        for operation in ("set", "wait"):
            expressions = _event_expressions(script, operation, hard_event)
            assert expressions == [f"{counter}[0] % 2"]


def test_non_equivalent_epoch_guards_keep_independent_loop_carried_protocols():
    program = _make_mismatched_counter_lexical_guard_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    ub_guards = _storage_domain_guards(prepared, "ub")
    assert len(ub_guards) == 1
    other_guards = _storage_domain_guards(prepared, "other")
    assert len(other_guards) == 1
    assert other_guards[0] != ub_guards[0]

    script = _pass_script(program, "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    counter_protocol = [
        i for i, line in enumerate(lines) if '"MTE3_MTE2"' in line and counter in line and _direct_enclosing_if(lines, i) == ub_guards[0]
    ]
    lexical_protocol = [
        i
        for i, line in enumerate(lines)
        if '"MTE3_MTE2"' in line and counter not in line and _direct_enclosing_if(lines, i) == other_guards[0]
    ]
    assert len(counter_protocol) == 2
    assert len(lexical_protocol) == 2


def test_non_multibuffer_sync_stays_inside_its_loop_local_guard():
    script = _pass_script(_make_single_iteration_guarded_global_program(), "tl.InsertSync")
    lines = script.splitlines()
    sync_lines = [i for i, line in enumerate(lines) if 'T.ascend_set_flag("MTE2_MTE3"' in line or 'T.ascend_wait_flag("MTE2_MTE3"' in line]
    assert len(sync_lines) == 2
    guards = {_direct_enclosing_if(lines, i) for i in sync_lines}
    assert len(guards) == 1
    guard = next(iter(guards))
    assert guard is not None and "__cond_" in guard


def test_task_internal_access_conditions_remain_atomic():
    program = _make_task_internal_guarded_global_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    assert _storage_domain_guards(prepared, "A") == []

    inserted = _pass_script(program, "tl.InsertSync")
    lines = inserted.splitlines()
    sync_lines = [i for i, line in enumerate(lines) if 'T.ascend_set_flag("MTE2_MTE3"' in line or 'T.ascend_wait_flag("MTE2_MTE3"' in line]
    assert len(sync_lines) == 2
    assert all(_direct_enclosing_if(lines, i) is None for i in sync_lines)


def test_task_internal_nested_loop_remains_atomic():
    prepared = _pass_script(
        _make_task_internal_nested_loop_guarded_global_program(),
        "tl.PrepareMultiBuffer",
    )
    assert _storage_domain_guards(prepared, "A") == []


def test_task_internal_sblock_condition_remains_atomic():
    prepared = _pass_script(
        _make_task_internal_sblock_guarded_program(),
        "tl.PrepareMultiBuffer",
    )
    assert _storage_domain_guards(prepared, "ub") == []


def test_task_internal_storage_dependent_condition_stays_unconditional():
    prepared = _pass_script(
        _make_task_internal_storage_condition_program(),
        "tl.PrepareMultiBuffer",
    )
    assert _storage_domain_guards(prepared, "A") == []


def test_compound_task_guard_definition_does_not_escape_before_definition():
    program = _make_compound_task_guard_definition_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    assert _storage_domain_guards(prepared, "ub") == []
    _device_script(program)


def test_early_compound_task_guard_definition_keeps_local_epoch():
    program = _make_early_compound_task_guard_definition_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(prepared, "ub")
    assert guards == ["active"]
    _device_script(program)


def test_unversioned_storage_uses_the_union_of_direct_access_guards():
    script = _pass_script(_make_unversioned_union_guard_program(), "tl.InsertSync")
    lines = script.splitlines()
    sync_lines = [i for i, line in enumerate(lines) if 'T.ascend_set_flag("MTE2_V"' in line or 'T.ascend_wait_flag("MTE2_V"' in line]
    assert len(sync_lines) == 2
    guards = {_direct_enclosing_if(lines, i) for i in sync_lines}
    assert len(guards) == 1
    guard = next(iter(guards))
    assert guard is not None and " or " in guard
    assert len(set(re.findall(r"__cond_\d+", guard))) == 2


def test_cross_stage_unversioned_storage_uses_unconditional_epoch():
    program = _make_cross_stage_unversioned_guard_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    assert _storage_domain_guards(prepared, "ub") == []

    inserted = _pass_script(program, "tl.InsertSync")
    lines = inserted.splitlines()
    sync_lines = [i for i, line in enumerate(lines) if '"MTE2_V"' in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)]
    assert len(sync_lines) == 2
    assert all(_direct_enclosing_if(lines, i) is None for i in sync_lines)


def test_prepare_serializes_lexical_epoch_guards():
    script = _pass_script(_make_nested_lexical_domain_program(), "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(script, "ub")
    assert len(guards) == 1 and "__cond_" in guards[0]


def test_inner_lexical_closure_projects_to_the_enclosing_domain():
    script = _pass_script(_make_nested_lexical_domain_program(), "tl.InsertSync")
    assert 'T.ascend_set_flag("MTE2_V"' in script
    assert 'T.ascend_wait_flag("MTE2_V"' in script
    assert 'T.ascend_set_flag("V_MTE3"' in script
    assert 'T.ascend_wait_flag("V_MTE3"' in script
    assert 'T.ascend_set_flag("MTE2_MTE3"' not in script
    assert 'T.ascend_wait_flag("MTE2_MTE3"' not in script


def test_nested_lexical_domains_keep_distinct_loop_local_guards():
    program = _make_nested_guarded_lexical_domain_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    prepared_guards = _storage_domain_guards(prepared, "ub")
    assert len(prepared_guards) == len(set(prepared_guards)) == 2

    script = _pass_script(program, "tl.InsertSync")
    lines = script.splitlines()
    mte2_v = [i for i, line in enumerate(lines) if 'T.ascend_set_flag("MTE2_V"' in line or 'T.ascend_wait_flag("MTE2_V"' in line]
    assert len(mte2_v) == 2
    inner_guards = {_direct_enclosing_if(lines, i) for i in mte2_v}
    assert len(inner_guards) == 1 and None not in inner_guards
    guarded_syncs = {
        guard
        for i, line in enumerate(lines)
        if "T.ascend_" in line
        if (guard := _direct_enclosing_if(lines, i)) is not None
        if "__cond_" in guard
    }
    assert len(guarded_syncs) == 2
    assert inner_guards < guarded_syncs


def test_nested_storage_guard_does_not_expand_loop_local_snapshot(capfd):
    program = _make_nested_only_guarded_lexical_domain_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    local_guards = _storage_domain_guards(prepared, "A")

    # The inner snapshot is not expanded into `enabled` at the parent loop.
    # Since it is unavailable before entering the inner loop, parent promotion
    # mechanically falls back to the unconditional lexical domain.
    assert len(local_guards) == 1
    assert "__cond_" in local_guards[0]

    capfd.readouterr()
    inserted = _pass_script(program, "tl.InsertSync")
    warning = capfd.readouterr().err
    assert "synchronization ring is malformed" not in warning
    # The inner loop has no sibling storage access, so its storage-local
    # iteration tuple includes both (_outer, _inner). Keep the real reverse
    # dependency even though _inner itself has extent one.
    for operation in ("set", "wait"):
        assert inserted.count(f'T.ascend_{operation}_flag("MTE3_MTE2", 0)') == 2


def test_nested_iteration_dependent_guard_does_not_escape_its_loop():
    prepared = _pass_script(
        _make_nested_iteration_dependent_guard_program(),
        "tl.PrepareMultiBuffer",
    )
    assert len(_storage_domain_guards(prepared, "A")) == 1


def test_mutating_nested_extent_does_not_guard_parent_epoch():
    prepared = _pass_script(
        _make_mutating_nested_extent_storage_program(),
        "tl.PrepareMultiBuffer",
    )
    assert _storage_domain_guards(prepared, "ub") == []


def test_equivalent_epoch_domain_uses_one_dominating_guard_representative():
    program = _make_early_lexical_late_counter_snapshot_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")

    counter_guards = _storage_domain_guards(prepared, "counter")
    lexical_guards = _storage_domain_guards(prepared, "lexical")
    assert len(lexical_guards) == 1
    assert len(counter_guards) == 1
    assert lexical_guards == counter_guards

    inserted = _pass_script(program, "tl.InsertSync")
    counter = _counter_name(inserted)
    lines = inserted.splitlines()
    lexical_syncs = [
        i for i, line in enumerate(lines) if 'T.ascend_set_flag("MTE2_MTE3"' in line or 'T.ascend_wait_flag("MTE2_MTE3"' in line
    ]
    assert len(lexical_syncs) == 2
    lexical_sync_guards = {_direct_enclosing_if(lines, i) for i in lexical_syncs}
    assert len(lexical_sync_guards) == 1 and None not in lexical_sync_guards
    lexical_guard = next(iter(lexical_sync_guards))
    assert lexical_guard == lexical_guards[0]
    counter_syncs = [i for i, line in enumerate(lines) if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)]
    assert counter_syncs
    assert all(_direct_enclosing_if(lines, i) == counter_guards[0] for i in counter_syncs)


def test_wide_domain_edge_covers_same_physical_narrow_edge():
    script = _pass_script(_make_wide_and_narrow_same_edge_program(), "tl.InsertSync")
    lines = script.splitlines()
    sync_lines = [i for i, line in enumerate(lines) if 'T.ascend_set_flag("MTE2_MTE3"' in line or 'T.ascend_wait_flag("MTE2_MTE3"' in line]
    assert len(sync_lines) == 2
    guards = [_direct_enclosing_if(lines, i) for i in sync_lines]
    assert guards == [None, None]


def test_one_task_can_participate_in_two_sparse_epoch_domains():
    program = _make_one_task_two_sparse_domains_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    x_guards = _storage_domain_guards(prepared, "x")
    y_guards = _storage_domain_guards(prepared, "y")
    assert len(x_guards) == len(y_guards) == 1
    assert x_guards != y_guards

    script = _pass_script(program, "tl.InsertSync")
    lines = script.splitlines()
    for hard_event, expected_count in (("MTE2_V", 4), ("V_MTE2", 8)):
        sync_lines = [
            i for i, line in enumerate(lines) if f'"{hard_event}"' in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
        ]
        assert len(sync_lines) == expected_count
        guards = {guard for i in sync_lines if (guard := _direct_enclosing_if(lines, i)) is not None}
        assert len(guards) == 2


def test_shared_inner_domain_promotes_to_multiple_parent_domains():
    program = _make_one_domain_split_into_two_parent_domains_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    x_guards = _storage_domain_guards(prepared, "x")
    y_guards = _storage_domain_guards(prepared, "y")

    # The inner loop is unconditional within its path and therefore shares one
    # lexical domain for x and y. At the outer loop their additional guarded
    # uses split that source domain into two parent domains.
    assert len(x_guards) == len(y_guards) == 1
    assert x_guards != y_guards

    script = _pass_script(program, "tl.InsertSync")
    lines = script.splitlines()
    parent_guards = {
        guard
        for i, line in enumerate(lines)
        if 'T.ascend_set_flag("V_MTE3"' in line
        if (guard := _direct_enclosing_if(lines, i)) is not None
    }
    assert parent_guards == {x_guards[0], y_guards[0]}


def test_inner_loop_carried_edge_stays_in_the_inner_active_epoch_domain():
    script = _pass_script(_make_nested_loop_carried_lexical_domains_program(), "tl.InsertSync")
    lines = script.splitlines()
    carried = [i for i, line in enumerate(lines) if '"MTE3_MTE2"' in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)]
    assert len(carried) == 4
    guarded = [_direct_enclosing_if(lines, i) for i in carried if _direct_enclosing_if(lines, i) is not None]
    assert len(guarded) == 2
    assert len(set(guarded)) == 1
    assert "__cond_0" in guarded[0]


def test_epoch_dependencies_share_one_union_guard():
    script = _pass_script(_make_union_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    guarded_protocol_lines = [
        i
        for i, line in enumerate(lines)
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line or f"{counter}[0] = {counter}[0] +" in line)
    ]
    assert guarded_protocol_lines
    guards = {_direct_enclosing_if(lines, i) for i in guarded_protocol_lines}
    # Data, flags, and the counter all use the same loop-local snapshot union.
    for guard in guards:
        assert guard is not None and " or " in guard
        snapshot_vars = set(re.findall(r"__cond_\d+", guard))
        assert len(snapshot_vars) == 2


def test_exhaustive_nested_union_guard_simplifies_to_outer_guard():
    prepared = _pass_script(_make_exhaustive_nested_union_guard_program(), "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1, prepared
    outer_snapshot = re.search(r"(__cond_\d+) = 0 < outer", prepared)
    assert outer_snapshot, prepared
    assert guards[0] == outer_snapshot.group(1)


def test_independent_guard_union_stays_linear():
    prepared = _pass_script(_make_independent_guard_union_program(), "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1, prepared
    guard = guards[0]
    # CNF distribution would turn these four cubes into 16 clauses with 64
    # literal occurrences. The bounded reducer must retain the linear DNF.
    predicates = re.findall(r"\b[ab][0-3]\b", guard)
    assert len(predicates) == len(set(predicates)) == 8
    assert guard.count(" and ") == 4
    assert guard.count(" or ") == 3


def test_cross_core_union_guard_absorbs_subcore_predicate():
    # The FIX writer executes under A, while only AIV0 consumes under A && sid.
    # Their buffer-domain guard is A. Keeping the sid snapshot in the unified
    # protocol would either leave an undefined AIV-only var in the AIC clone or
    # make AIC wait for a release that the inactive AIV never produces.
    script = _device_script(_make_cross_core_absorbed_guard_program())
    counters = _counter_names(script, "ub")
    # Like every other root SBlock allocation, one logical counter allocation
    # is shared by the mixed-kernel IR and lowered into each concrete core.
    assert len(counters) == 1
    assert "T.ascend_cross_core_set_flag" in script
    assert "T.ascend_cross_core_wait_flag" in script
    dynamic_events = [line for line in script.splitlines() if "ascend_cross_core_" in line and "version_counter" in line]
    assert dynamic_events
    assert all('T.Cast("int32"' not in line and "% 2" in line for line in dynamic_events)


def test_cross_core_implied_snapshot_guard_is_resolved_before_sync():
    snapshots = _pass_snapshots(
        _make_cross_core_implied_snapshot_guard_program(),
        {"tl.PrepareMultiBuffer", "tl.AssignCore", "tl.ResolveCore"},
    )
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1, prepared
    guard = guards[0]
    snapshot_vars = set(re.findall(r"__cond_\d+", guard))
    assert len(snapshot_vars) == 1
    assert " or " not in guard

    first_assigned = snapshots["tl.AssignCore"][0]
    first_resolved = snapshots["tl.ResolveCore"][0]
    snapshot = next(iter(snapshot_vars))
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_assigned, f"{snapshot}: T.bool")
    statement = f"{snapshot} = i % 2 == 0"
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_resolved, statement)


def test_cross_core_nested_snapshot_retains_its_definition_guard():
    program = _make_cross_core_nested_snapshot_guard_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1, prepared
    guard = guards[0]
    assert " and " in guard
    assert len(set(re.findall(r"__cond_\d+", guard))) == 2

    script = _device_script(program)
    assert "ascend_cross_core_set_flag" in script
    assert "ascend_cross_core_wait_flag" in script


def test_parent_projection_keeps_unabsorbed_active_guard():
    program = _make_crossed_partial_guard_projection_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(prepared, "ub")
    active_guards = {guard for guard in guards if "pred_a" in guard and "pred_b" in guard}
    assert len(active_guards) == 1, prepared
    active_guard = next(iter(active_guards))

    inserted = _pass_script(program, "tl.InsertSync")
    lines = inserted.splitlines()
    sync_lines = [i for i, line in enumerate(lines) if '"MTE2_MTE3"' in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)]
    assert len(sync_lines) == 2, inserted
    sync_guards = [_direct_enclosing_if(lines, i) for i in sync_lines]
    assert all(guard is not None and active_guard in guard for guard in sync_guards), inserted


def test_narrow_child_order_does_not_bridge_wider_parent_domain():
    inserted = _pass_script(
        _make_narrow_child_bridge_program(),
        "tl.InsertSync",
    )
    sync_lines = [
        line for line in inserted.splitlines() if '"MTE2_MTE3"' in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
    ]
    assert len(sync_lines) == 4, inserted


def test_counter_pass_boundaries_and_core_resolution():
    program = _make_guarded_program(mode="counter")
    snapshots = _pass_snapshots(
        program,
        {
            "tl.AutoSchedule",
            "tl.AssignCore",
            "tl.PrepareMultiBuffer",
            "tl.ResolveCore",
            "tl.InsertSync",
            "tl.MaterializeMultiBuffer",
        },
    )

    assert len(snapshots["tl.AssignCore"]) == 1
    assert len(snapshots["tl.ResolveCore"]) == 1
    scheduled = snapshots["tl.AutoSchedule"][0]
    first_assigned = snapshots["tl.AssignCore"][0]
    first_resolved = snapshots["tl.ResolveCore"][0]
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    synchronized = snapshots["tl.InsertSync"][0]
    materialized = snapshots["tl.MaterializeMultiBuffer"][0]

    assert "version_counter" not in first_assigned
    assert "tl.buffer_version_mode" in scheduled
    assert "tl.buffer_version_mode" in first_assigned

    counter = _counter_name(prepared)
    allocation = f"{counter} = T.sblock_alloc_buffer"
    initialize = f"{counter}[0] = 0"
    advance = f"{counter}[0] = {counter}[0] + 1"

    assert "tl.buffer_versions_map" in prepared
    assert "tl.buffer_version_mode" not in prepared
    assert "tl.multi_buffer_counter_map" in prepared
    assert "tl.storage_epoch_guard_map" in prepared
    assert allocation in prepared and initialize in prepared and advance in prepared
    assert "T.sblock_alloc_buffer((2, 64)" not in prepared

    assert "tl.multi_buffer_counter_map" in synchronized
    assert re.search(rf"ascend_(?:set|wait)_flag\([^\n]*{counter}\[0\]", synchronized)
    assert "T.sblock_alloc_buffer((2, 64)" not in synchronized

    assert "tl.multi_buffer_counter_map" not in materialized
    assert "tl.storage_epoch_guard_map" not in materialized
    assert "tl.buffer_versions_map" not in materialized
    assert "tl.buffer_version_mode" not in materialized
    assert "tl.manual_multi_buffer" in materialized
    assert "T.sblock_alloc_buffer((2, 64)" in materialized
    assert f"{counter}[0] % 2" in materialized

    for statement in (initialize, advance):
        assert '"core_mask": T.int64(3)' in _task_attr_before(prepared, statement)
        assert '"core_mask": T.int64(1)' in _task_attr_before(first_resolved, statement)
        assert '"core_mask": T.int64(1)' in _task_attr_before(synchronized, statement)
        assert '"core_mask": T.int64(1)' in _task_attr_before(materialized, statement)


@pytest.mark.parametrize("mode", [None, "counter"])
def test_resolve_core_broadcasts_sibling_counter_from_all_readers(mode):
    program = _make_mixed_core_sibling_program(mode=mode)
    snapshots = _pass_snapshots(program, {"tl.ResolveCore", "tl.PrepareMultiBuffer"})
    assert len(snapshots["tl.ResolveCore"]) == 1
    counter = _counter_name(snapshots["tl.PrepareMultiBuffer"][0])
    first_resolved = snapshots["tl.ResolveCore"][0]

    for statement in (f"{counter}[0] = 0", f"{counter}[0] = {counter}[0] + 1"):
        positions = [i for i, line in enumerate(first_resolved.splitlines()) if statement in line]
        assert positions
        for position in positions:
            task_attr = next(line for line in reversed(first_resolved.splitlines()[:position]) if '"tl.ascend_task"' in line)
            assert '"core_mask": T.int64(3)' in task_attr

    _device_script(program)


def test_resolve_core_accounts_for_sibling_counter_guard_producer():
    program = _make_guarded_mixed_core_sibling_program()
    snapshots = _pass_snapshots(
        program,
        {"tl.AssignCore", "tl.PrepareMultiBuffer", "tl.ResolveCore"},
    )
    first_assigned = snapshots["tl.AssignCore"][0]
    first_resolved = snapshots["tl.ResolveCore"][0]
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    counter = _counter_name(prepared)

    condition = next(line.strip().split(" = ", 1)[0] for line in first_assigned.splitlines() if " = 0 < enabled" in line)
    condition_name = condition.split(":", 1)[0]
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_assigned, f"{condition} = 0 < enabled")
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_resolved, f"{condition_name} = 0 < enabled")
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_resolved, f"{counter}[0] = 0")

    _device_script(program)


def test_resolve_core_accounts_for_lexical_storage_union_guard():
    snapshots = _pass_snapshots(
        _make_lexical_union_guard_core_resolution_program(),
        {"tl.AssignCore", "tl.PrepareMultiBuffer", "tl.ResolveCore"},
    )
    assigned = snapshots["tl.AssignCore"][0]
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    resolved = snapshots["tl.ResolveCore"][0]

    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1 and " or " in guards[0]
    for name in ("active_cube", "active_vector"):
        assert '"core_mask": T.int64(3)' in _task_attr_before(assigned, f"{name}:")
        assert '"core_mask": T.int64(3)' in _task_attr_before(resolved, f"{name}:")


@pytest.mark.parametrize(
    ("builder", "producer_rhs"),
    [
        (_make_outer_guarded_mixed_core_sibling_program, " = 0 < pred[0]"),
        (_make_dynamic_extent_mixed_core_sibling_program, " = extent[0]"),
    ],
    ids=["outer-guard", "dynamic-extent"],
)
def test_resolve_core_accounts_for_enclosing_control_producers(builder, producer_rhs):
    program = builder()
    snapshots = _pass_snapshots(program, {"tl.AssignCore", "tl.ResolveCore"})
    first_assigned = snapshots["tl.AssignCore"][0]
    first_resolved = snapshots["tl.ResolveCore"][0]

    def task_attr_before_producer(script):
        line = next(line for line in script.splitlines() if producer_rhs in line)
        return _task_attr_before(script, line.strip())

    assert '"core_mask": T.int64(3)' in task_attr_before_producer(first_assigned)
    assert '"core_mask": T.int64(3)' in task_attr_before_producer(first_resolved)

    _device_script(program)


def test_sibling_loops_with_different_ring_protocols_use_lexical_dependency():
    for mode in (None, "counter"):
        script = _pass_script(_make_mismatched_sibling_protocol_program(mode=mode), "tl.InsertSync")
        counter = _counter_name(script)
        assert script.count(f"{counter}[0] = {counter}[0] + 1") == 2
        # The incompatible owner protocols keep their local counter rings.  A
        # constant-ID MTE3->MTE2 handshake covers the protocol transition.  Its
        # surrounding unconditional edges also prove the wider MTE2->V order,
        # so no redundant direct lexical handshake is needed.
        assert 'T.ascend_set_flag("MTE3_MTE2", 2)' in script
        assert 'T.ascend_wait_flag("MTE3_MTE2", 2)' in script
        assert 'T.ascend_set_flag("MTE2_V", 2)' not in script
        assert 'T.ascend_wait_flag("MTE2_V", 2)' not in script


@pytest.mark.parametrize("swapped", [False, True])
def test_unconditional_counter_order_projects_into_lexical_closure(capfd, swapped):
    snapshots = _pass_snapshots(
        _make_nested_unconditional_projection_program(swapped),
        {"tl.AnnotateMultiBufferEligible", "tl.InsertSync"},
    )
    eligible = snapshots["tl.AnnotateMultiBufferEligible"][0]
    owner_lines = [line for line in eligible.splitlines() if "multi_buffer_eligible" in line and ("for i" in line or "for j" in line)]
    assert len(owner_lines) == 2
    assert all("x" in line and "y" in line for line in owner_lines)

    script = snapshots["tl.InsertSync"][0]
    stderr = capfd.readouterr().err

    # The owner-local MTE2->V ordering and the lexical V->V same-pipe ordering
    # form the reverse half of the outer V->MTE2 ring.  The first edge is
    # unconditional, so its counter-domain proof is also valid lexically.
    assert "the synchronization ring is malformed" not in stderr
    assert 'T.ascend_pipe_barrier("PIPE_MTE2")' not in script


def test_equal_guard_domains_share_one_counter_group():
    script = _pass_script(_make_guard_group_program(same_guard=True), "tl.MaterializeMultiBuffer")
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    assert f"{counter}[0] % 2" in script
    assert f"{counter}[0] % 3" in script
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 1


def test_equivalent_distinct_snapshots_share_counter_when_representative_dominates():
    script = _pass_script(_make_equivalent_snapshot_group_program(), "tl.MaterializeMultiBuffer")
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    assert f"{counter}[0] % 2" in script
    assert f"{counter}[0] % 3" in script


def test_counter_group_guard_representative_dominates_every_member():
    prepared = _pass_script(
        _make_crossed_counter_group_snapshot_program(),
        "tl.PrepareMultiBuffer",
    )
    snapshots = re.findall(r"(__cond_\d+)(?:: T.bool)? = 0 < pred", prepared)
    assert len(snapshots) == 4, prepared
    expected = snapshots[::2]
    assert _storage_domain_guards(prepared, "a") == expected
    assert _storage_domain_guards(prepared, "b") == expected
    counter = _counter_name(prepared)
    lines = prepared.splitlines()
    increments = [i for i, line in enumerate(lines) if f"{counter}[0] = {counter}[0] + 1" in line]
    assert [_direct_enclosing_if(lines, i) for i in increments] == expected


def test_counter_group_requires_every_equivalent_guard_on_protocol_cores():
    script = _prepare_script(_make_mixed_core_equivalent_guard_group_program())
    assert len(_counter_names(script)) == 2


def test_lexical_guard_representative_preserves_core_availability():
    program = _make_mixed_core_equivalent_guard_group_program()
    script = _prepare_script(program)
    vector_guards = _storage_domain_guards(script, "vector_ub")
    cube_guards = _storage_domain_guards(script, "cube_l1")
    assert len(vector_guards) == len(cube_guards) == 1
    assert vector_guards != cube_guards
    _device_script(program)


def test_outer_context_assumptions_participate_in_guard_equivalence():
    script = _pass_script(_make_assumed_equivalent_snapshot_group_program(), "tl.MaterializeMultiBuffer")
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    assert f"{counter}[0] % 2" in script
    assert f"{counter}[0] % 3" in script


def test_implied_union_guard_keeps_the_earliest_snapshot():
    script = _pass_script(_make_implied_union_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    protocol = [
        i
        for i, line in enumerate(lines)
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line or f"{counter}[0] = {counter}[0] +" in line)
    ]
    assert protocol
    guards = {_direct_enclosing_if(lines, i) for i in protocol}
    assert len(guards) == 1
    guard = next(iter(guards))
    assert guard and " or " not in guard
    assert len(set(re.findall(r"__cond_\d+", guard))) == 1


def test_early_union_guard_uses_one_stable_counter_snapshot():
    script = _pass_script(_make_reverse_implied_union_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    protocol = [
        i
        for i, line in enumerate(lines)
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line or f"{counter}[0] = {counter}[0] +" in line)
    ]
    assert protocol
    guards = {_direct_enclosing_if(lines, i) for i in protocol}
    assert len(guards) == 1
    guard = next(iter(guards))
    assert guard and "weak" in guard and "strong" not in guard and "pred" not in guard


def test_guard_implication_does_not_merge_mutated_read_snapshots():
    prepared = _pass_script(_make_mutated_snapshot_union_program(), "tl.PrepareMultiBuffer")
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1, prepared
    # The second mutable-read snapshot is scheduled after the first access. It
    # must not be proved equal to the first snapshot. Both a full epoch and the
    # explicit union of the two stable snapshots are safe.
    guard = guards[0]
    assert guard == "T.bool(True)" or set(guard.split(" or ")) == {"before", "after"}


@pytest.mark.parametrize(
    "program_factory",
    [_make_implied_union_guard_program, _make_reverse_implied_union_guard_program],
)
def test_auto_mode_keeps_counter_for_ordered_implied_guards(program_factory):
    prepared = _pass_script(program_factory(mode=None), "tl.PrepareMultiBuffer")
    assert "version_counter" in prepared


def test_proof_equivalent_but_distinct_snapshots_use_separate_groups():
    script = _pass_script(_make_equivalent_guard_group_program(), "tl.MaterializeMultiBuffer")
    counters = _counter_names(script)
    assert len(counters) == 2
    assert any(f"{counter}[0] % 2" in script for counter in counters)
    assert any(f"{counter}[0] % 3" in script for counter in counters)


def test_different_guard_domains_use_distinct_counter_groups():
    script = _pass_script(_make_guard_group_program(same_guard=False), "tl.MaterializeMultiBuffer")
    counters = _counter_names(script)
    assert len(counters) == 2
    for counter in counters:
        assert f"{counter}[0] = {counter}[0] + 1" in script
        assert any(counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line) for line in script.splitlines())


def test_for_if_for_uses_guarded_inner_epoch_counter():
    script = _pass_script(_make_guarded_inner_loop_program(), "tl.PrepareMultiBuffer")
    counter = _counter_name(script)
    assert re.search(r"for inner in (?:range\(3\)|T\.serial\(3)", script)
    assert f"{counter}[0] = {counter}[0] + 1" in script
    assert re.search(r"if __cond_0(?:\[0\])?:", script)


def test_outer_scope_guards_are_not_repeated_around_counter_syncs():
    script = _pass_script(_make_mismatched_sibling_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    increments = [i for i, line in enumerate(lines) if f"{counter}[0] = {counter}[0] +" in line]
    assert len(increments) == 2
    assert script.count("if __cond_0:") == 1
    assert script.count("if __cond_1:") == 1
    assert "__cond_0 = T.alloc_buffer" not in script
    assert "__cond_1 = T.alloc_buffer" not in script
    dynamic_syncs = [i for i, line in enumerate(lines) if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)]
    # The existing source-scope if encloses each complete owner loop.  Syncs
    # and advances inside that loop must not emit the same guard a second time.
    assert dynamic_syncs
    assert all(_direct_enclosing_if(lines, i) is None for i in dynamic_syncs + increments)


def test_enclosing_guard_is_not_repeated_in_nested_pipeline_sync_guards():
    script = _pass_script(_make_guarded_nested_pipeline_program(), "tl.InsertSync")
    assert re.search(r"if (__cond_\d+):\n\s+for k(?:_\d+)? in T\.serial", script)
    assert not re.search(r"if __cond_\d+ and [^:\n]*k(?:_\d+)?", script)
    counters = _counter_names(script, "l0a") + _counter_names(script, "l0b")
    lines = script.splitlines()
    inner_syncs = [
        i
        for i, line in enumerate(lines)
        if any(counter in line for counter in counters) and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
    ]
    assert inner_syncs
    assert all(_direct_enclosing_if(lines, i) is None for i in inner_syncs)


def test_sibling_loops_advance_on_their_own_iteration_guards():
    program = _make_iteration_guarded_sibling_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    counter = _counter_name(prepared)
    assert re.search(r"__cond_\d+(?:: T\.bool)? = 0 < pred_a\[i(?:_\d+)?\]", prepared)
    assert re.search(r"__cond_\d+(?:: T\.bool)? = 0 < pred_b\[j(?:_\d+)?\]", prepared)
    assert not re.search(r"__cond_\d+ = T\.alloc_buffer", prepared)
    lines = prepared.splitlines()
    increments = [i for i, line in enumerate(lines) if f"{counter}[0] = {counter}[0] +" in line]
    assert len(increments) == 2
    increment_guards = {_direct_enclosing_if(lines, i) for i in increments}
    assert len(increment_guards) == 2
    assert all(guard and "__cond_" in guard for guard in increment_guards)

    synchronized = _pass_script(program, "tl.InsertSync")
    sync_lines = synchronized.splitlines()
    dynamic_syncs = [
        i for i, line in enumerate(sync_lines) if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
    ]
    sync_guards = {_direct_enclosing_if(sync_lines, i) for i in dynamic_syncs}
    assert increment_guards <= sync_guards


def test_different_depth_sibling_loops_share_counter():
    script = _pass_script(_make_different_depth_sibling_program(), "tl.PrepareMultiBuffer")
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    lines = script.splitlines()
    increments = [i for i, line in enumerate(lines) if f"{counter}[0] = {counter}[0] +" in line]
    assert len(increments) == 2
    guards = {_direct_enclosing_if(lines, i) for i in increments}
    assert len(guards) == 2
    assert all(guard and "__cond_" in guard for guard in guards)


def test_nested_multi_buffer_loops_are_rejected():
    with pytest.raises(Exception, match="has nested owner loops"):
        _pass_script(_make_nested_loop_program(), "tl.PrepareMultiBuffer")


def test_nested_alias_multi_buffer_loops_are_rejected_by_storage():
    with pytest.raises(Exception, match="has nested owner loops"):
        _pass_script(_make_nested_alias_loop_program(), "tl.PrepareMultiBuffer")


def test_versioned_storage_control_expressions_are_snapshotted():
    script = _pass_script(_make_versioned_storage_control_program(), "tl.PrepareMultiBuffer")
    assert "__loop_bound_" in script
    inner_loop = next(line for line in script.splitlines() if "for j" in line)
    assert "__loop_bound_" in inner_loop
    assert "ub[" not in inner_loop


def test_frontend_uses_outer_owner_for_nested_alias_storage():
    script = _pass_script(_make_auto_nested_alias_loop_program(), "tl.AnnotateMultiBufferEligible")
    outer_line = next(line for line in script.splitlines() if "for outer" in line)
    inner_line = next(line for line in script.splitlines() if "for inner" in line)
    # The outer loop covers both its direct base access and the nested alias
    # access, so the current storage-level annotator chooses it as the single
    # complete owner and suppresses a nested partial claim.
    assert '"multi_buffer_eligible": [base.data]' in outer_line
    assert '"multi_buffer_eligible": []' in inner_line


def test_nested_guarded_epoch_advances_after_inner_control():
    script = _pass_script(_make_nested_guarded_access_program(), "tl.InsertSync")
    counters = _counter_names(script, "data_buf")
    assert len(counters) == 1
    counter = counters[0]
    lines = script.splitlines()
    increment = next(i for i, line in enumerate(lines) if f"{counter}[0] = {counter}[0] +" in line)
    repeat_loop = max(i for i, line in enumerate(lines[:increment]) if "for repeat" in line)
    repeat_indent = len(lines[repeat_loop]) - len(lines[repeat_loop].lstrip())
    increment_indent = len(lines[increment]) - len(lines[increment].lstrip())
    assert increment_indent < repeat_indent
    assert _direct_enclosing_if(lines, increment) in ("__cond_0", "__cond_0[0]")

    protocol = [i for i, line in enumerate(lines) if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)]
    assert protocol
    assert all(i < increment for i in protocol)
    acquire = next(i for i in protocol if 'ascend_wait_flag("MTE2_MTE3"' in lines[i])
    release = next(i for i in protocol if 'ascend_set_flag("MTE3_MTE2"' in lines[i])
    assert repeat_loop < acquire < release < increment
    assert _direct_enclosing_if(lines, acquire) == "repeat == 0"
    assert _direct_enclosing_if(lines, release) == "repeat + 1 >= 2"


def test_nonserial_nested_condition_does_not_escape_loop_guard():
    script = _pass_script(_make_nonserial_nested_guard_program(), "tl.InsertSync")
    counter = _counter_name(script)
    protocol_lines = [
        line for line in script.splitlines() if counter in line and (f"{counter}[0] = {counter}[0] +" in line or "ascend_" in line)
    ]
    assert protocol_lines
    assert all("j" not in line and "__cond" not in line for line in protocol_lines)


def test_auto_mode_ignores_nested_loop_local_condition():
    script = _pass_script(_make_nonserial_nested_guard_program(mode=None), "tl.PrepareMultiBuffer")
    assert "version_counter" not in script
    assert "tl.multi_buffer_counter_map" not in script


def test_nested_condition_snapshot_widens_to_an_empty_owner_epoch():
    script = _pass_script(_make_nested_snapshot_epoch_program(), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    protocol = [
        i
        for i, line in enumerate(lines)
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line or f"{counter}[0] = {counter}[0] +" in line)
    ]
    assert protocol
    guards = [_direct_enclosing_if(lines, i) for i in protocol]
    assert all(guard is None or "__cond_" not in guard for guard in guards)


def test_optional_nested_protocol_uses_unconditional_parent_epoch():
    for mode in (None, "counter"):
        script = _pass_script(_make_optional_nested_endpoint_program(mode=mode), "tl.InsertSync")
        lines = script.splitlines()
        repeat = next(i for i, line in enumerate(lines) if "for _repeat in T.serial(repeat_count" in line)
        acquire = max(i for i, line in enumerate(lines[:repeat]) if 'T.ascend_wait_flag("MTE2_MTE3"' in line)
        assert _direct_enclosing_if(lines, acquire) is None
        assert acquire < repeat
        if mode is None:
            assert "i % 2" in lines[acquire]
        else:
            assert "version_counter" in lines[acquire]


def test_lexical_sync_endpoint_stays_outside_possibly_empty_loop():
    script = _pass_script(_make_lexical_optional_endpoint_program(), "tl.InsertSync")
    lines = script.splitlines()
    repeat = next(i for i, line in enumerate(lines) if "for _repeat in T.serial(repeat_count" in line)
    wait = next(i for i, line in enumerate(lines) if 'T.ascend_wait_flag("MTE2_V"' in line)
    assert wait < repeat


def test_unrelated_dependent_nested_extent_does_not_disable_counter():
    script = _pass_script(_make_unrelated_dependent_nested_extent_program(), "tl.PrepareMultiBuffer")
    assert "version_counter" in script


def test_equivalent_sibling_loops_keep_loop_local_condition_binds():
    program = _make_equivalent_sibling_snapshot_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    counter = _counter_name(prepared)
    increment_lines = [i for i, line in enumerate(prepared.splitlines()) if f"{counter}[0] = {counter}[0] +" in line]
    assert len(increment_lines) == 2
    increment_guards = {_direct_enclosing_if(prepared.splitlines(), i) for i in increment_lines}
    assert any("__cond_0" in guard for guard in increment_guards if guard)
    assert any("__cond_1" in guard for guard in increment_guards if guard)

    synchronized = _pass_script(program, "tl.InsertSync")
    dynamic_syncs = [
        i
        for i, line in enumerate(synchronized.splitlines())
        if counter in line and ("ascend_set_flag" in line or "ascend_wait_flag" in line)
    ]
    sync_guards = {_direct_enclosing_if(synchronized.splitlines(), i) for i in dynamic_syncs}
    assert any("__cond_0" in guard for guard in sync_guards if guard)
    assert any("__cond_1" in guard for guard in sync_guards if guard)


def test_any_owner_external_fill_is_broadcast_to_every_version():
    cases = (
        _make_fill_initialization_program(),
        _make_fill_initialization_program(partial=True),
        _make_fill_initialization_program(position="tail"),
        _make_fill_initialization_program(guarded=True),
    )
    for program in cases:
        prepared = _pass_script(program, "tl.PrepareMultiBuffer")
        assert "tl.multi_buffer_broadcast_fill" in prepared

        script = _pass_script(program, "tl.MaterializeMultiBuffer")
        assert "tl.multi_buffer_broadcast_fill" not in script
        assert re.search(r"T\.fill\(T\.region\(ub_\d+\[0, 0\], 2, 2,", script)


def test_owner_external_l1_fill_rejects_multiple_versions():
    with pytest.raises(
        Exception,
        match=r"Broadcast fill for L1 storage .* with 2 versions is not supported",
    ):
        _pass_script(_make_l1_fill_initialization_program(), "tl.PrepareMultiBuffer")


def test_owner_external_fill_task_rejects_other_storage_accesses():
    with pytest.raises(Exception, match="only T.fill accesses that can be rewritten independently"):
        _pass_script(_make_fill_with_extra_pointer_write_program(), "tl.PrepareMultiBuffer")


def test_owner_external_fill_rejects_target_storage_reads():
    with pytest.raises(Exception, match="only T.fill accesses that can be rewritten independently"):
        _pass_script(_make_fill_reading_target_program(), "tl.PrepareMultiBuffer")


@pytest.mark.parametrize("bound", ["min", "extent"])
def test_owner_external_fill_rejects_target_dependent_region(bound):
    with pytest.raises(Exception, match="only T.fill accesses that can be rewritten independently"):
        _pass_script(
            _make_fill_with_target_dependent_region_program(bound),
            "tl.PrepareMultiBuffer",
        )


def test_owner_external_fill_rejects_target_storage_assume_guard():
    with pytest.raises(Exception, match="task guard outside its annotated owner loops"):
        _pass_script(_make_assumed_fill_initialization_program(), "tl.PrepareMultiBuffer")


def test_auto_owner_rejects_target_storage_assume_guarded_fill():
    program = _make_assumed_fill_initialization_program(explicit_claim=False)
    eligible = _pass_script(program, "tl.AnnotateMultiBufferEligible")
    owner = next(line for line in eligible.splitlines() if "for i" in line)
    assert "ub" not in owner
    _pass_script(program, "tl.PrepareMultiBuffer")


def test_iteration_mode_uses_stepped_loop_trip_count():
    script = _pass_script(_make_stepped_iteration_program(), "tl.MaterializeMultiBuffer")
    assert "version_counter" not in script
    assert re.search(r"\((?:_tmp|inner(?:_\d+)?) \+ outer(?:_\d+)? \* 2\) % 3", script)
    assert not re.search(r"outer(?:_\d+)? \* 4", script)


def test_owner_external_fill_inside_task_control_is_broadcast():
    program = _make_fill_with_nested_control_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    assert "tl.multi_buffer_broadcast_fill" in prepared

    script = _pass_script(program, "tl.MaterializeMultiBuffer")
    assert "tl.multi_buffer_broadcast_fill" not in script
    assert re.search(r"for j in (?:T\.serial\(2|range\(2\))", script)
    assert re.search(r"T\.fill\(T\.region\(ub_\d+\[0, j \* 32\], 2, 2, 32\)", script)


def test_fill_in_common_outer_guard_targets_every_version():
    program = _make_scoped_fill_initialization_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    counter = _counter_name(prepared)
    assert "tl.multi_buffer_broadcast_fill" in prepared

    script = _pass_script(program, "tl.MaterializeMultiBuffer")
    assert f"{counter}[0] % 2" in script
    assert "tl.multi_buffer_broadcast_fill" not in script
    assert re.search(r"T\.fill\(T\.region\(ub_1\[0, 0\], 2, 2, 64\)", script)
    # NormalizeControlFlow retains one source guard for each sibling statement
    # (fill and owner); synchronization must not add a third copy. The
    # unguarded snapshot remains an SSA Bind rather than mutable local storage.
    assert script.count("if __cond_0:") == 2
    assert "__cond_0 = T.alloc_buffer" not in script


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_entry_break_does_not_start_or_advance_counter_epoch():
    script = _pass_script(_make_break_program("entry"), "tl.AutoSchedule")
    counter = _counter_name(script)
    lines = script.splitlines()
    break_index = next(i for i, line in enumerate(lines) if "T.loop_break()" in line)
    guard_assignment = max(i for i, line in enumerate(lines[:break_index]) if "stop <= i" in line)
    window = lines[guard_assignment : break_index + 1]
    assert not any(counter in line for line in window)
    assert not any("ascend_set_flag" in line or "ascend_wait_flag" in line for line in window)


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_tail_break_keeps_normal_release_then_advance_order():
    script = _pass_script(_make_break_program("tail"), "tl.AutoSchedule")
    counter = _counter_name(script)
    lines = script.splitlines()
    break_index = next(i for i, line in enumerate(lines) if "T.loop_break()" in line)
    release = max(i for i, line in enumerate(lines[:break_index]) if 'T.ascend_set_flag("MTE3_MTE2"' in line and counter in line)
    advance = max(i for i, line in enumerate(lines[:break_index]) if f"{counter}[0] = {counter}[0] +" in line)
    assert release < advance < break_index
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 1


@pytest.mark.parametrize("mode", [None, "counter"])
@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_middle_break_balances_both_flag_phases_before_advancing(mode):
    # InsertSync is the owner of cleanup and still exposes the immutable break
    # snapshot before later simplification lowers the scheduled protocol.
    script = _pass_script(_make_break_program("mid", mode=mode), "tl.InsertSync")
    counter = _counter_name(script)
    lines = script.splitlines()
    break_index = next(i for i, line in enumerate(lines) if "T.loop_break()" in line)
    guard_assignment = max(i for i, line in enumerate(lines[:break_index]) if "stop <= i" in line)
    wait = next(i for i in range(guard_assignment, break_index) if 'T.ascend_wait_flag("MTE2_V"' in lines[i] and counter in lines[i])
    restore = next(i for i in range(guard_assignment, break_index) if 'T.ascend_set_flag("MTE3_MTE2"' in lines[i] and counter in lines[i])
    advance = next(i for i in range(guard_assignment, break_index) if f"{counter}[0] = {counter}[0] +" in lines[i])
    assert wait < restore < advance < break_index
    event = f"{counter}[0] % 2"
    assert event in lines[wait] and event in lines[restore]


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_middle_break_without_crossing_flag_only_advances_counter():
    script = _pass_script(_make_scalar_middle_break_program(), "tl.AutoSchedule")
    counter = _counter_name(script)
    lines = script.splitlines()
    break_index = next(i for i, line in enumerate(lines) if "T.loop_break()" in line)
    guard_assignment = max(i for i, line in enumerate(lines[:break_index]) if "stop <= i" in line)
    window = lines[guard_assignment : break_index + 1]
    assert not any("ascend_set_flag" in line or "ascend_wait_flag" in line for line in window)
    assert sum(f"{counter}[0] = {counter}[0] +" in line for line in window) == 1


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_lexical_middle_break_balances_crossing_flag_phase():
    script = _pass_script(_make_lexical_middle_break_program(), "tl.AutoSchedule")
    assert "version_counter" not in script
    lines = script.splitlines()
    break_index = next(i for i, line in enumerate(lines) if "T.loop_break()" in line)
    guard_assignment = max(i for i, line in enumerate(lines[:break_index]) if "stop <= i" in line)
    window = lines[guard_assignment : break_index + 1]
    assert any('T.ascend_wait_flag("MTE2_V"' in line for line in window)
    assert any('T.ascend_set_flag("MTE3_MTE2"' in line for line in window)


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_each_middle_break_exit_gets_its_own_cleanup_and_advance():
    script = _pass_script(_make_multiple_break_program(), "tl.AutoSchedule")
    counter = _counter_name(script)
    lines = script.splitlines()
    break_indices = [i for i, line in enumerate(lines) if "T.loop_break()" in line]
    assert len(break_indices) == 2
    assignments = [i for i, line in enumerate(lines) if "stop_a <= i" in line or "stop_b <= i" in line]
    for start, stop in zip(assignments, break_indices):
        window = lines[start : stop + 1]
        assert any("ascend_wait_flag" in line and counter in line for line in window)
        assert any('ascend_set_flag("MTE3_MTE2"' in line and counter in line for line in window)
        assert sum(f"{counter}[0] = {counter}[0] +" in line for line in window) == 1


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_cross_core_middle_break_cleans_vector_phase_and_advances_each_side_once():
    program = _make_cross_core_middle_break_program()
    inserted = _pass_script(program, "tl.InsertSync")
    counters = _counter_names(inserted, "ub")
    assert len(counters) == 1
    counter = counters[0]
    lines = inserted.splitlines()
    break_index = next(i for i, line in enumerate(lines) if "T.loop_break()" in line)
    guard = max(i for i, line in enumerate(lines[:break_index]) if "stop <= i" in line)
    window = lines[guard : break_index + 1]
    wait = next(i for i, line in enumerate(window) if "ascend_cross_core_wait_flag" in line)
    restore = next(i for i, line in enumerate(window) if "ascend_cross_core_set_flag" in line)
    advance = next(i for i, line in enumerate(window) if f"{counter}[0] = {counter}[0] +" in line)
    assert wait < restore < advance
    assert all('T.Cast("int32"' not in window[i] and "% 2" in window[i] for i in (wait, restore))

    # LowerScheduledTIR freshens the one Broadcast allocation into one local
    # physical counter for each concrete core stream.
    partitioned = _pass_script(program, "tl.LowerScheduledTIR")
    assert len(_counter_names(partitioned, "ub")) == 2
    assert partitioned.count("T.loop_break()") == 2


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_middle_break_rejects_unrestorable_fix_unit_flag_protocol():
    with pytest.raises(Exception, match="FIX unit-flag protocol crosses the exit"):
        _pass_script(_make_fix_unit_flag_middle_break_program(), "tl.AutoSchedule")


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_non_outer_break_counter_persists_across_outer_iterations():
    script = _pass_script(_make_nested_break_program(), "tl.AutoSchedule")
    counter = _counter_name(script)
    allocation = script.index(f"{counter} = T.alloc_buffer")
    loop_match = re.search(r"for outer(?:, inner)? in ", script)
    assert loop_match is not None and allocation < loop_match.start()
    assert f"{counter}[0] = {counter}[0] + 1" in script


def test_counter_domain_rewrites_storage_aliases_consistently():
    script = _device_script(_make_counter_alias_program())
    counter = _counter_name(script)
    assert script.count(f"{counter}[0] % 2") >= 3


def test_disjoint_alias_views_share_storage_counter_and_expand_owner():
    script = _device_script(_make_sibling_alias_views_program())
    assert len(_counter_names(script)) == 1
    assert '"dyn_shared_memory_buf": 512' in script
    assert re.search(r"base(?:_\d+)? = T.alloc_buffer\(\(128,\)", script)


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_group_break_uses_separate_storage_counters():
    script = _pass_script(_make_group_break_union_program(), "tl.AutoSchedule")
    counters = _counter_names(script)
    assert len(counters) == 2
    assert all(f"{counter}[0] = {counter}[0] + 1" in script for counter in counters)


@pytest.mark.skip(reason=LATER_BREAK_REASON)
def test_group_break_with_equal_signatures_shares_storage_counter():
    script = _pass_script(_make_group_break_same_signature_program(), "tl.AutoSchedule")
    counters = _counter_names(script)
    assert len(counters) == 1
    counter = counters[0]
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 2


def test_counter_rejects_enable_offset():
    with pytest.raises(Exception, match="accessed at multiple schedule stages"):
        _device_script(_make_offset_program())


def test_auto_multi_owner_counter_reports_cross_stage_accesses():
    with pytest.raises(Exception, match="Align their T.Stage values"):
        _pass_script(_make_auto_sibling_cross_stage_counter_program(), "tl.PrepareMultiBuffer")


def test_iteration_mode_allows_enable_offset():
    script = _device_script(_make_offset_program(enable_offset=True, mode="iteration"))
    assert "version_counter" not in script


def test_counter_accepts_explicitly_disabled_offset():
    script = _device_script(_make_offset_program(enable_offset=False))
    assert "version_counter" in script


def test_counter_supports_per_core_task_access():
    script = _pass_script(_make_per_core_task_program(), "tl.InsertSync")
    counter = _counter_name(script)
    assert '"tl.ascend_per_core_task"' in script
    assert re.search(rf'T\.ascend_set_flag\("MTE2_V", {counter}\[0\] % 2\)', script)
    _device_script(_make_per_core_task_program())


def test_counter_supports_guarded_per_core_task_when_sync_guard_implies_task_guard():
    script = _pass_script(_make_guarded_per_core_task_program(True), "tl.InsertSync")
    counter = _counter_name(script)
    assert '"tl.ascend_per_core_task"' in script
    assert re.search(rf'T\.ascend_set_flag\("MTE2_V", {counter}\[0\] % 2\)', script)

    lines = script.splitlines()
    per_core = next(i for i, line in enumerate(lines) if '"tl.ascend_per_core_task"' in line)
    next_unit = next(
        (i for i in range(per_core + 1, len(lines)) if '"tl.schedule_unit"' in lines[i]),
        len(lines),
    )
    task_guard = _direct_enclosing_if(lines, per_core)
    assert task_guard is not None
    assert f"if {task_guard}:" not in "\n".join(lines[per_core:next_unit])
    _device_script(_make_guarded_per_core_task_program(True))


def test_counter_rejects_guarded_per_core_task_when_sync_guard_escapes_task_guard():
    with pytest.raises(
        tvm.error.InternalError,
        match=r"synchronization guard .* does not imply the T\.PerCoreTask guard",
    ):
        _pass_script(_make_guarded_per_core_task_program(False), "tl.InsertSync")


@pytest.mark.parametrize("mode", ["counter", "iteration"])
def test_assign_core_rejects_scalar_bind_mixing_ub_and_l1(mode):
    with pytest.raises(Exception, match="incompatible core-local memories"):
        _pass_script(_make_mixed_core_memory_bind_program(mode=mode), "tl.AssignCore")


def test_prepare_consumers_do_not_widen_bind_after_sync_planning():
    snapshots = _pass_snapshots(
        _make_cross_core_implied_snapshot_guard_program(),
        {"tl.AssignCore", "tl.PrepareMultiBuffer", "tl.ResolveCore", "tl.InsertSync"},
    )
    prepared = snapshots["tl.PrepareMultiBuffer"][0]
    guards = _storage_domain_guards(prepared, "ub")
    assert len(guards) == 1
    snapshot = next(iter(re.findall(r"__cond_\d+", guards[0])))
    first_assigned = snapshots["tl.AssignCore"][0]
    first_resolved = snapshots["tl.ResolveCore"][0]
    synchronized = snapshots["tl.InsertSync"][0]

    # Resolve must see PrepareMultiBuffer's generated consumers before
    # InsertSync plans dependencies, so synchronization never observes a
    # narrower producer than final lowering does.
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_assigned, f"{snapshot}: T.bool")
    statement = f"{snapshot} = i % 2 == 0"
    assert '"core_mask": T.int64(3)' in _task_attr_before(first_resolved, statement)
    assert '"core_mask": T.int64(3)' in _task_attr_before(synchronized, statement)


def test_resolve_rejects_counter_guard_unavailable_on_protocol_core():
    with pytest.raises(
        tvm.error.InternalError,
        match="storage-epoch guard .* is unavailable on every legal execution core",
    ):
        _pass_script(_make_unbroadcastable_counter_guard_program(), "tl.ResolveCore")


def test_counter_rejects_head_tail_outside_annotated_loop():
    with pytest.raises(Exception, match="is accessed outside its annotated owner loops"):
        _pass_script(_make_head_tail_program(), "tl.PrepareMultiBuffer")


def test_serial_one_head_tail_share_counter_flag_channel():
    script = _pass_script(_make_head_tail_program(wrapped=True), "tl.InsertSync")
    counter = _counter_name(script)
    assert script.count(f"{counter}[0] = {counter}[0] + 1") == 3
    assert script.index(f"{counter}[0] = 0") < script.index("for _head in T.serial(1")
    # The extent-one head/tail owners continue the same global epoch stream as
    # the pipelined middle owner, so all three use one physical flag ring.
    for operation in ("set", "wait"):
        events = re.findall(rf'T\.ascend_{operation}_flag\("MTE3_MTE2", ([^\n]+)\)', script)
        dynamic = [event for event in events if counter in event]
        assert len(dynamic) == 3
        assert len(set(dynamic)) == 1
        assert set(events) == {"0", "1", f"{counter}[0] % 2"}


def test_for_one_tail_owner_reuses_counter_flag_channel_without_lexical_sync():
    program = _make_for_one_tail_owner_program()
    prepared = _pass_script(program, "tl.PrepareMultiBuffer")
    counter = _counter_name(prepared)
    assert prepared.count(f"{counter}[0] = {counter}[0] + 1") == 2

    inserted = _pass_script(program, "tl.InsertSync")
    assert "ascend_pipe_barrier" not in inserted
    assert set(re.findall(r'T\.ascend_(?:set|wait)_flag\("([^"]+)"', inserted)) == {
        "MTE2_V",
        "V_MTE3",
        "MTE3_MTE2",
    }
    for hard_event in ("MTE2_V", "V_MTE3", "MTE3_MTE2"):
        for operation in ("set", "wait"):
            events = _event_expressions(inserted, operation, hard_event)
            dynamic = [event for event in events if counter in event]
            assert len(dynamic) == 2
            assert _normalized_event_expressions(inserted, operation, hard_event, counter) == {"COUNTER[0] % 2"}
            expected = {f"{counter}[0] % 2"}
            if hard_event == "MTE3_MTE2":
                expected |= {"0", "1"}
            assert set(events) == expected


if __name__ == "__main__":
    tilelang.testing.main()
