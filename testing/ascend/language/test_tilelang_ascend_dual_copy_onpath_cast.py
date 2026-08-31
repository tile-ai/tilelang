"""Codegen tests for L0C fp32 -> UB f16/bf16 ``dual_copy`` on-path cast.

Quant dual lowers to two non-dual FixPipes; same-dtype stays hardware dual.
Frontend rejects unsupported casts; RewriteDualCopy enforces a 2:1 split.
"""

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang.engine.lower import lower

# quant_pre mode for the on-path cast: 1 == F322F16, 16 == F322BF16.
QUANT_PRE = {"float16": 1, "bfloat16": 16}


def gemm_dual_copy(dst_dtype, split, M=256, N=256, K=256, unit_flag_ctrl=None):
    """Single-tile gemm + ``dual_copy`` fan-out. ``split`` is ``"M"`` or ``"N"``."""
    TILE_M, TILE_N, TILE_K = M, N, K
    half_m, half_n = TILE_M // 2, TILE_N // 2
    dst_shape = (half_m, TILE_N) if split == "M" else (TILE_M, half_n)

    @T.prim_func
    def main(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((N, K), "float16"),
        C: T.Buffer(dst_shape, dst_dtype),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared(dst_shape, dst_dtype)

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(
                a_l1,
                b_l1,
                acc,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=unit_flag_ctrl,
            )
            T.dual_copy(acc, tmp, unit_flag_ctrl=unit_flag_ctrl)
            T.copy(tmp, C)

    return main


def _kernel_source(func):
    return lower(func, target="ascend").kernel_source


def _cc_to_ub_argument_lists(source):
    """Return every ``copy_matrix_cc_to_ub(...)`` call's top-level arguments."""
    needle = "copy_matrix_cc_to_ub("
    calls = []
    start = 0
    while True:
        open_pos = source.find(needle, start)
        if open_pos == -1:
            return calls
        depth = 1
        i = open_pos + len(needle)
        while i < len(source):
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        calls.append(_split_top_level_commas(source[open_pos + len(needle) : i]))
        start = i + 1


def _split_top_level_commas(text):
    parts = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return parts


def _cc_to_ub_tails(source):
    """Return the fixed integer-argument tails of every emitted cc_to_ub call."""
    tails = []
    for args in _cc_to_ub_argument_lists(source):
        assert len(args) == 26, args
        tails.append(args[-19:])
    return tails


def _cc_to_ub_sizes(source):
    """Return (copy_inner, copy_rows) for every emitted cc_to_ub call."""
    return [(args[3], args[4]) for args in _cc_to_ub_argument_lists(source)]


@pytest.mark.parametrize("split", ["M", "N"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_onpath_cast_emits_two_nondual_pipes(dst_dtype, split):
    source = _kernel_source(gemm_dual_copy(dst_dtype, split))

    tails = _cc_to_ub_tails(source)
    assert len(tails) == 2, source

    quant_pre = str(QUANT_PRE[dst_dtype])
    # Each pipe: dual_dst_ctl == 0, correct quant_pre, NZ2ND enabled.
    for tail in tails:
        assert tail[0] == "0", tail  # dual_dst_ctl
        assert tail[4] == quant_pre, tail  # quant_pre
        assert tail[7] == "1", tail  # NZ2ND_en

    # sub_blockid routes pipe 0 -> AIV 0 and pipe 1 -> AIV 1.
    assert tails[0][1] == "0", tails
    assert tails[1][1] == "1", tails

    # Same-pipe FixPipe commands are already ordered; no extra PIPE_FIX.
    assert source.count("AscendC::PipeBarrier<pipe_t::PIPE_FIX>()") == 0, source


@pytest.mark.parametrize("split", ["M", "N"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_onpath_cast_unit_flag_clears_on_last_pipe(dst_dtype, split):
    source = _kernel_source(gemm_dual_copy(dst_dtype, split, unit_flag_ctrl=3))
    tails = _cc_to_ub_tails(source)
    assert len(tails) == 2, source
    # Non-last FixPipe demotes 3 → 2 (CHECK_ONLY) so Cube cannot reuse L0C
    # until the second half-copy's CHECK_AND_CLEAR (3) completes.
    assert tails[0][3] == "2", tails
    assert tails[1][3] == "3", tails


@pytest.mark.parametrize("split", ["M", "N"])
def test_same_dtype_dual_keeps_single_hardware_dual_pipe(split):
    source = _kernel_source(gemm_dual_copy("float32", split))

    tails = _cc_to_ub_tails(source)
    assert len(tails) == 1, source
    assert tails[0][0] == ("1" if split == "M" else "2"), tails  # dual_dst_ctl
    assert tails[0][4] == "0", tails  # quant_pre == NoQuant
    assert source.count("AscendC::PipeBarrier<pipe_t::PIPE_FIX>()") == 0, source


def test_whole_tile_quant_copy_is_single_nondual_pipe():
    # A non-dual L0C fp32 -> UB f16 copy (T.copy, not dual_copy) must remain a
    # single non-dual FixPipe with the on-path quant mode.
    TILE_M, TILE_N, TILE_K = 256, 256, 256

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), "float16"),
        B: T.Buffer((TILE_N, TILE_K), "float16"),
        C: T.Buffer((TILE_M, TILE_N), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared((TILE_M, TILE_N), "float16")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, tmp)
            T.copy(tmp, C)

    source = _kernel_source(main)
    tails = _cc_to_ub_tails(source)
    assert len(tails) == 1, source
    assert tails[0][0] == "0", tails  # dual_dst_ctl == 0
    assert tails[0][4] == "1", tails  # quant_pre == F322F16


def test_frontend_rejects_int_cast():
    # L0C fp32 -> UB int32 dual_copy is not one of the supported on-path casts.
    with pytest.raises(ValueError, match="only supports fp32 -> f16/bf16"):
        gemm_dual_copy("int32", "M")


def test_frontend_rejects_l0c_to_gm_onpath_cast():
    # Build inside the raises() so the eager prim_func decorator is covered.
    TILE_M, TILE_N, TILE_K = 256, 256, 256

    def build():
        @T.prim_func
        def main(
            A: T.Buffer((TILE_M, TILE_K), "float16"),
            B: T.Buffer((TILE_N, TILE_K), "float16"),
            C: T.Buffer((TILE_M // 2, TILE_N), "float16"),
        ):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
                b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
                acc = T.alloc_l0c((TILE_M, TILE_N), "float32")

                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
                T.dual_copy(acc, C)

        return main

    with pytest.raises(ValueError, match="only supports L0C -> UB"):
        build()


def test_frontend_rejects_fp8_dual_copy():
    with pytest.raises(ValueError, match="fp8/int8 quant is not supported"):
        gemm_dual_copy("float8_e4m3fn", "M")


def compact_column_slice_dual_copy(split, alloc=64, region_n=32):
    """Compact MAD into ``acc[:, :region_n]`` on an ``alloc x alloc`` L0C."""
    region_m = alloc
    dst_shape = (region_m // 2, region_n) if split == "M" else (region_m, region_n // 2)

    @T.prim_func
    def main(
        A: T.Buffer((alloc, alloc), "float16"),
        B: T.Buffer((alloc, alloc), "float16"),
        C: T.Buffer(dst_shape, "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((alloc, alloc), "float16")
            b_l1 = T.alloc_l1((alloc, alloc), "float16")
            a_l0 = T.alloc_l0a((alloc, alloc), "float16")
            b_l0 = T.alloc_l0b((alloc, alloc), "float16")
            acc = T.alloc_l0c((alloc, alloc), "float32")
            tmp = T.alloc_shared(dst_shape, "float16")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1[0:region_m, 0:region_n], a_l0[0:region_m, 0:region_n])
            T.copy(b_l1[0:region_n, 0:region_n], b_l0[0:region_n, 0:region_n])
            T.gemm(
                a_l0[0:region_m, 0:region_n],
                b_l0[0:region_n, 0:region_n],
                acc[0:region_m, 0:region_n],
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(acc[0:region_m, 0:region_n], tmp)
            T.copy(tmp, C)

    return main


def test_onpath_dual_tail_uses_src_half_rows():
    TILE_M, TILE_N, TILE_K = 256, 256, 256
    tail_m = 192

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), "float16"),
        B: T.Buffer((TILE_N, TILE_K), "float16"),
        C: T.Buffer((TILE_M // 2, TILE_N), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared((TILE_M // 2, TILE_N), "float16")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc[0:tail_m, 0:TILE_N], tmp)
            T.copy(tmp, C)

    with pytest.raises(ValueError, match="exact 2:1 extent ratio"):
        _kernel_source(main)


def dynamic_prefix_onpath_dual_copy(split, max_units, unit):
    TILE_M = TILE_N = TILE_K = 256
    dst_shape = (TILE_M // 2, TILE_N) if split == "M" else (TILE_M, TILE_N // 2)

    if split == "M":

        @T.prim_func
        def main(
            A: T.Buffer((TILE_M, TILE_K), "float16"),
            B: T.Buffer((TILE_N, TILE_K), "float16"),
            C: T.Buffer(dst_shape, "float16"),
            sizes: T.Buffer((1,), "int32"),
        ):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
                b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
                acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
                tmp = T.alloc_shared(dst_shape, "float16")
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
                T.dual_copy(
                    acc[0 : T.max(T.min(T.int32(sizes[0]), max_units), 0) * unit, 0:TILE_N],
                    tmp,
                )
                T.copy(tmp, C)

        return main

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), "float16"),
        B: T.Buffer((TILE_N, TILE_K), "float16"),
        C: T.Buffer(dst_shape, "float16"),
        sizes: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared(dst_shape, "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(
                acc[0:TILE_M, 0 : T.max(T.min(T.int32(sizes[0]), max_units), 0) * unit],
                tmp,
            )
            T.copy(tmp, C)

    return main


def dynamic_tail_onpath_dual_copy(split):
    if split == "M":
        return dynamic_prefix_onpath_dual_copy(split, max_units=128, unit=2)
    return dynamic_prefix_onpath_dual_copy(split, max_units=8, unit=32)


@pytest.mark.parametrize("split", ["M", "N"])
def test_onpath_dual_dynamic_tail_uses_src_half(split):
    source = _kernel_source(dynamic_tail_onpath_dual_copy(split))
    sizes = _cc_to_ub_sizes(source)
    assert len(sizes) == 2, source
    tile_half = "128"
    pipe0_inner, pipe0_rows = sizes[0]
    pipe1_inner, pipe1_rows = sizes[1]
    if split == "M":
        assert pipe0_inner == "256" and pipe1_inner == "256", source
        assert "min" in pipe0_rows and tile_half in pipe0_rows, (pipe0_rows, source)
        assert tile_half in pipe1_rows, (pipe1_rows, source)
    else:
        assert pipe0_rows == "256" and pipe1_rows == "256", source
        assert "min" in pipe0_inner and pipe0_inner != tile_half, (pipe0_inner, source)
        assert tile_half in pipe1_inner, (pipe1_inner, source)


def dynamic_over_half_onpath_dual_copy(split, unit_flag_ctrl=3):
    TILE_M = TILE_N = TILE_K = 256
    tile_half = TILE_M // 2
    dst_shape = (tile_half, TILE_N) if split == "M" else (TILE_M, TILE_N // 2)
    extra_units, unit = (32, 2) if split == "M" else (2, 32)

    if split == "M":

        @T.prim_func
        def main(
            A: T.Buffer((TILE_M, TILE_K), "float16"),
            B: T.Buffer((TILE_N, TILE_K), "float16"),
            C: T.Buffer(dst_shape, "float16"),
            sizes: T.Buffer((1,), "int32"),
        ):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
                b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
                acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
                tmp = T.alloc_shared(dst_shape, "float16")
                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(
                    a_l1,
                    b_l1,
                    acc,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=unit_flag_ctrl,
                )
                T.dual_copy(
                    acc[
                        0 : tile_half + T.max(T.min(T.int32(sizes[0]), extra_units), 1) * unit,
                        0:TILE_N,
                    ],
                    tmp,
                    unit_flag_ctrl=unit_flag_ctrl,
                )
                T.copy(tmp, C)

        return main

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), "float16"),
        B: T.Buffer((TILE_N, TILE_K), "float16"),
        C: T.Buffer(dst_shape, "float16"),
        sizes: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared(dst_shape, "float16")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(
                a_l1,
                b_l1,
                acc,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=unit_flag_ctrl,
            )
            T.dual_copy(
                acc[
                    0:TILE_M,
                    0 : tile_half + T.max(T.min(T.int32(sizes[0]), extra_units), 1) * unit,
                ],
                tmp,
                unit_flag_ctrl=unit_flag_ctrl,
            )
            T.copy(tmp, C)

    return main


@pytest.mark.parametrize("split", ["M", "N"])
def test_onpath_dual_dynamic_over_half_tail_uses_tile_half(split):
    source = _kernel_source(dynamic_over_half_onpath_dual_copy(split))
    args = _cc_to_ub_argument_lists(source)
    sizes = _cc_to_ub_sizes(source)
    tails = _cc_to_ub_tails(source)
    assert len(sizes) == 2, source
    tile_half = "128"
    pipe0_inner, pipe0_rows = sizes[0]
    pipe1_inner, pipe1_rows = sizes[1]
    if split == "M":
        assert pipe0_inner == "256" and pipe1_inner == "256", source
        assert pipe0_rows == tile_half, (pipe0_rows, source)
        assert pipe1_rows != tile_half and "max" in pipe1_rows, (pipe1_rows, source)
    else:
        assert pipe0_rows == "256" and pipe1_rows == "256", source
        assert pipe0_inner == tile_half, (pipe0_inner, source)
        assert pipe1_inner != tile_half and "max" in pipe1_inner, (pipe1_inner, source)
    assert tails[0][1] == "0" and tails[1][1] == "1", tails
    assert tails[0][3] == "2" and tails[1][3] == "3", tails
    assert args[0][1] != args[1][1], (args[0][1], args[1][1], source)
    assert "+" in args[1][1], (args[1][1], source)
    # Full-width M-prefix still walks the 256-row L0C NZ pitch, not the copied 192.
    for call_args in args:
        assert call_args[6] == "256", (call_args[6], source)


@pytest.mark.parametrize("split", ["M", "N"])
def test_onpath_dual_dynamic_single_pipe_when_src_fits_tile_half(split):
    if split == "M":
        func = dynamic_prefix_onpath_dual_copy(split, max_units=64, unit=2)
    else:
        func = dynamic_prefix_onpath_dual_copy(split, max_units=3, unit=16)
    source = _kernel_source(func)
    sizes = _cc_to_ub_sizes(source)
    assert len(sizes) == 1, source
    tails = _cc_to_ub_tails(source)
    assert tails[0][1] == "0", tails
    copy_inner, copy_rows = sizes[0]
    if split == "M":
        assert copy_inner == "256", (copy_inner, source)
        assert copy_rows != "128", (copy_rows, source)
    else:
        assert copy_rows == "256", (copy_rows, source)
        assert copy_inner != "128", (copy_inner, source)
        assert "16" in copy_inner, (copy_inner, source)


@pytest.mark.parametrize("split", ["M", "N"])
def test_column_slice_second_pipe_offset(split):
    # acc[:, :32] on a 64-wide L0C; second pipe src ptr must be offset.
    alloc, region_n = 64, 32
    region_m = alloc
    source = _kernel_source(compact_column_slice_dual_copy(split, alloc, region_n))
    args = _cc_to_ub_argument_lists(source)
    assert len(args) == 2, source
    if split == "M":
        expected_inner, expected_rows = str(region_n), str(region_m // 2)
    else:
        expected_inner, expected_rows = str(region_n // 2), str(region_m)
    for copy_inner, copy_rows in _cc_to_ub_sizes(source):
        assert copy_inner == expected_inner, (copy_inner, expected_inner, source)
        assert copy_rows == expected_rows, (copy_rows, expected_rows, source)
    for call_args in args:
        assert call_args[6] == str(alloc), (call_args[6], alloc, source)
    assert args[0][1] != args[1][1], (args[0][1], args[1][1], source)
    assert "+" in args[1][1], (args[1][1], source)


def test_m_split_compact_2d_uses_alloc_row_stride():
    # Compact 32x32 MAD in a 64x64 L0C; M-split offset uses row stride 64.
    # loop_src_stride stays the compact region M because N is also sliced.
    alloc, region_m, region_n = 64, 32, 32

    @T.prim_func
    def main(
        A: T.Buffer((alloc, alloc), "float16"),
        B: T.Buffer((alloc, alloc), "float16"),
        C: T.Buffer((region_m // 2, region_n), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((alloc, alloc), "float16")
            b_l1 = T.alloc_l1((alloc, alloc), "float16")
            a_l0 = T.alloc_l0a((alloc, alloc), "float16")
            b_l0 = T.alloc_l0b((alloc, alloc), "float16")
            acc = T.alloc_l0c((alloc, alloc), "float32")
            tmp = T.alloc_shared((region_m // 2, region_n), "float16")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1[0:region_m, 0:region_n], a_l0[0:region_m, 0:region_n])
            T.copy(b_l1[0:region_n, 0:region_n], b_l0[0:region_n, 0:region_n])
            T.gemm(
                a_l0[0:region_m, 0:region_n],
                b_l0[0:region_n, 0:region_n],
                acc[0:region_m, 0:region_n],
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(acc[0:region_m, 0:region_n], tmp)
            T.copy(tmp, C)

    source = _kernel_source(main)
    args = _cc_to_ub_argument_lists(source)
    assert len(args) == 2, source
    expected_inner, expected_rows = str(region_n), str(region_m // 2)
    for copy_inner, copy_rows in _cc_to_ub_sizes(source):
        assert copy_inner == expected_inner, (copy_inner, expected_inner, source)
        assert copy_rows == expected_rows, (copy_rows, expected_rows, source)
    for call_args in args:
        assert call_args[6] == str(region_m), (call_args[6], region_m, source)
    assert args[0][1] != args[1][1], (args[0][1], args[1][1], source)
    assert "+" in args[1][1], (args[1][1], source)


def test_lowering_rejects_odd_m():
    TILE_M, TILE_N, TILE_K = 16, 32, 32
    odd_m = 15

    def build():
        @T.prim_func
        def main(
            A: T.Buffer((TILE_M, TILE_K), "float16"),
            B: T.Buffer((TILE_N, TILE_K), "float16"),
            C: T.Buffer((odd_m // 2, TILE_N), "float16"),
        ):
            with T.Kernel(1):
                a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
                b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
                acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
                tmp = T.alloc_shared((odd_m // 2, TILE_N), "float16")

                T.copy(A, a_l1)
                T.copy(B, b_l1)
                T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
                T.dual_copy(acc[0:odd_m, 0:TILE_N], tmp)
                T.copy(tmp, C)

        return main

    with pytest.raises(ValueError, match="Cannot infer dual_copy split direction"):
        build()


def test_lowering_rejects_n_not_multiple_of_32():
    with pytest.raises(ValueError, match="multiple of 32"):
        _kernel_source(gemm_dual_copy("float16", "N", N=16))


@pytest.mark.parametrize("split", ["M", "N"])
def test_lowering_rejects_dst_smaller_than_src_half(split):
    TILE_M, TILE_N, TILE_K = 256, 256, 256
    dst_rows = TILE_M // 2 if split == "M" else TILE_M
    dst_cols = TILE_N if split == "M" else TILE_N // 2
    region_rows = 64 if split == "M" else TILE_M
    region_cols = TILE_N if split == "M" else 64

    @T.prim_func
    def main(
        A: T.Buffer((TILE_M, TILE_K), "float16"),
        B: T.Buffer((TILE_N, TILE_K), "float16"),
        C: T.Buffer((dst_rows, dst_cols), "float16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared((dst_rows, dst_cols), "float16")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.dual_copy(acc, tmp[0:region_rows, 0:region_cols])
            T.copy(tmp, C)

    with pytest.raises(ValueError, match="exact 2:1 extent ratio"):
        _kernel_source(main)


def gemm_dual_copy_with_vf(dst_dtype, split, epilogue, M=256, N=256, K=256):
    """Same geometry as ``gemm_dual_copy``, plus the SimdVF bias-add epilogue used
    by the on-path vs SimdVF-cast performance kernels.

    ``epilogue="onpath"``: L0C fp32 -> UB ``dst_dtype`` (two FixPipes) then SimdVF vadd.
    ``epilogue="vf_cast"``: L0C fp32 -> UB fp32 (hardware dual) then SimdVF vcvt + vadd.
    """
    TILE_M, TILE_N, TILE_K = M, N, K
    half_m, half_n = TILE_M // 2, TILE_N // 2
    dst_rows, dst_cols = (half_m, TILE_N) if split == "M" else (TILE_M, half_n)
    dst_shape = (dst_rows, dst_cols)

    @T.macro
    def vf_bias_add(buf, bias):
        with T.SimdVF():
            for i, j in T.Parallel(dst_rows, dst_cols):
                buf[i, j] = buf[i, j] + bias[j]

    @T.macro
    def vf_cast(src, dst):
        with T.SimdVF():
            for i in range(dst_rows):
                for j in range(0, dst_cols, 64):
                    x = T.simd.vld(src[i, j])
                    y = T.simd.vcvt(x, dst_dtype)
                    T.simd.vsts(dst[i, j], y, dist="PK_B32")

    @T.prim_func
    def main(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((N, K), "float16"),
        Bias: T.Buffer((dst_cols,), dst_dtype),
        C: T.Buffer(dst_shape, dst_dtype),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")
            tmp = T.alloc_shared(dst_shape, dst_dtype)
            bias_ub = T.alloc_shared((dst_cols,), dst_dtype)
            if epilogue == "vf_cast":
                tmp_f32 = T.alloc_shared(dst_shape, "float32")

            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(Bias, bias_ub)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            if epilogue == "onpath":
                T.dual_copy(acc, tmp)
            else:
                T.dual_copy(acc, tmp_f32)
                vf_cast(tmp_f32, tmp)
            vf_bias_add(tmp, bias_ub)
            T.copy(tmp, C)

    return main


@pytest.mark.parametrize("split", ["M", "N"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_onpath_with_vf_still_two_nondual_pipes(dst_dtype, split):
    source = _kernel_source(gemm_dual_copy_with_vf(dst_dtype, split, "onpath"))
    tails = _cc_to_ub_tails(source)
    assert len(tails) == 2, source
    quant_pre = str(QUANT_PRE[dst_dtype])
    for tail in tails:
        assert tail[0] == "0", tail
        assert tail[4] == quant_pre, tail
    assert tails[0][1] == "0" and tails[1][1] == "1", tails
    assert source.count("AscendC::PipeBarrier<pipe_t::PIPE_FIX>()") == 0, source
    assert "simd_inst::vadd(" in source, source
    assert "simd_inst::vcvt<" not in source, source


@pytest.mark.parametrize("split", ["M", "N"])
@pytest.mark.parametrize("dst_dtype", ["float16", "bfloat16"])
def test_vf_cast_keeps_single_hardware_dual_pipe(dst_dtype, split):
    source = _kernel_source(gemm_dual_copy_with_vf(dst_dtype, split, "vf_cast"))
    tails = _cc_to_ub_tails(source)
    assert len(tails) == 1, source
    assert tails[0][0] == ("1" if split == "M" else "2"), tails
    assert tails[0][4] == "0", tails
    assert source.count("AscendC::PipeBarrier<pipe_t::PIPE_FIX>()") == 0, source
    assert "simd_inst::vcvt<" in source, source
    assert "simd_inst::vadd(" in source, source
    assert "PK_B32" in source, source


if __name__ == "__main__":
    tilelang.testing.main()
