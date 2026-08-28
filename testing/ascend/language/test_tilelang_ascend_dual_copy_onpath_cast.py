"""Codegen tests for L0C fp32 -> UB f16/bf16 ``dual_copy`` on-path cast.

Dual + ``quant_pre`` is illegal, so lowering emits two non-dual FixPipes
(``sub_blockid`` 0 then 1). Same-dtype dual stays one hardware-dual pipe.
Unsupported casts fail in the frontend; N-split alignment is checked in C++.
"""

import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm
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
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
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
    return [args[-19:] for args in _cc_to_ub_argument_lists(source)]


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


def test_lowering_rejects_n_not_multiple_of_32():
    # N=16 is C0-aligned but not 32-aligned, so the N-split on-path path is
    # rejected by the C++ ICHECK rather than the Python frontend.
    with pytest.raises(tvm.error.InternalError, match="N % 32 == 0"):
        _kernel_source(gemm_dual_copy("float16", "N", N=16))


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
