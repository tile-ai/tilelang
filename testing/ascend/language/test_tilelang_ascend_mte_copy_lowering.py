"""Regression tests for Ascend MTE (GM<->UBuf) copy-arg lowering.

The cases pin the exact ``copy_*_align_v2`` burst arguments emitted by codegen
and validate supported layout combinations end-to-end on NPU:

* ``test_gm2ub_single_row_copy`` — a contiguous 1D GM->UBuf load must be a single
  burst of the full byte length.
* ``test_ub2gm_partial_col_copy`` — a 2D ``buf[:, :cols]`` UBuf->GM store must be
  issued as ``n_rows`` strided bursts that skip the UBuf row pitch. A previous
  heuristic collapsed any copy with ``row_bytes < 32`` into a single contiguous
  burst, reading one padded row and silently dropping the remaining rows.
* ``test_gm2ub_1d_to_2d`` — a contiguous source is split at the destination's
  physical row boundary.
* ``test_gm2ub_2d_to_2d`` — two strided sides with matching row boundaries keep
  their independent row strides.
* ``test_gm2ub_single_element_rows`` — a coalesced strided vector is represented
  as one-element MTE rows.
* ``test_fp4_column_alignment`` — packed FP4 rows must contain whole bytes in
  both GM->UBuf and UBuf->GM copies, so odd-width column copies are rejected.
* ``test_gm2ub_dynamic_rows_middle_dim`` — a runtime-sized strided source chooses
  the physical source row boundary instead of relying on symbolic size ordering.
* ``test_mte_copy_hoists_let_bindings_before_calls`` — a sign-unknown runtime
  ``floordiv`` in a copy address must emit its temporary declarations before the
  DMA call rather than inside its pointer argument.
* ``test_gm2ub_incompatible_2d_rejected`` — two different physical row boundaries
  are rejected until outer-loop lowering is implemented.
* ``test_gm_to_l1_fixed_outer_axis_uses_effective_row_stride`` — a fixed
  batch/group axis contributes to the base pointer but not the matrix row
  stride.
* ``test_l0c_to_rank3_gm_uses_effective_row_stride`` — the same rank reduction
  applies to a grouped L0C-to-GM output copy, while a smaller destination
  region controls the transferred tail extent.
* ``test_l0c_to_oob_gm_preserves_source_geometry`` — clamping an OOB GM output
  tail reduces the transfer extent without changing the producer's L0C pitch.
* ``test_compact_l0_regions_drive_load_mad_and_store_geometry`` — an L0 region
  smaller than its allocation consistently controls MTE1, MAD, and FIX pitch.
* ``test_gm_to_l1_true_nd_rejected`` — a genuine three-dimensional transfer is
  rejected instead of being silently flattened with the wrong stride.
* ``test_gm2ub2gm_fp4_copy`` — a packed ``float4_e2m1fn`` GM->UBuf->GM copy
  must use 4-bit-based byte counts (``N`` logical elements => ``N/2`` bytes).
"""

import re

import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def single_row_copy(n: int):
    @T.prim_func
    def kernel(
        a: T.Buffer((n,), "float32"),
        out: T.Buffer((n,), "float32"),
    ):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((n,), "float32")
            T.copy(a[:n], temp)
            T.copy(temp, out[:n])

    return kernel


def test_gm2ub_single_row_copy():
    n = 1024  # 4096 bytes => lenBurst should be 128
    kernel = tilelang.compile(
        single_row_copy(n),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*1,\s*4096,\s*0,\s*0,\s*0,\s*0,\s*4096,\s*4096\);",
        source,
    )
    bad = re.search(r"copy_gm_to_ubuf\(", source)
    assert good is not None, source
    assert bad is None, source

    device = torch.device("npu")
    a = torch.arange(n, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(out, a)


ROWS = 4
WIDTH = 64
COLS = 4


def partial_col_copy():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS, WIDTH), "int32"),
        dst: T.Buffer((ROWS * COLS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, WIDTH), "int32")
            T.copy(src, buf)
            T.copy(buf[:, :COLS], dst)

    return kernel


def test_ub2gm_partial_col_copy():
    kernel = tilelang.compile(
        partial_col_copy(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    # 4 bursts of 4 int32 (16B), source stride = row pitch (64*4 = 256B),
    # dest stride = contiguous 16B. cacheMode for a store defaults to 4.
    good = re.search(
        r"copy_ubuf_to_gm_align_v2\([^;]*,\s*0,\s*4,\s*16,\s*4,\s*16,\s*256\);",
        source,
    )
    # The buggy collapse emitted a single 64B contiguous burst.
    bad = re.search(
        r"copy_ubuf_to_gm_align_v2\([^;]*,\s*0,\s*1,\s*64,\s*4,\s*64,\s*64\);",
        source,
    )
    assert good is not None, source
    assert bad is None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * WIDTH, dtype=torch.int32, device=device).reshape(ROWS, WIDTH)
    out = kernel(src)
    torch.npu.synchronize()

    expect = src[:, :COLS].reshape(-1).contiguous()
    torch.testing.assert_close(out, expect)


def gm2ub_1d_to_2d():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS * COLS,), "int32"),
        dst: T.Buffer((ROWS * COLS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, WIDTH), "int32")
            T.copy(src, buf[:, :COLS])
            T.copy(buf[:, :COLS], dst)

    return kernel


def test_gm2ub_1d_to_2d():
    kernel = tilelang.compile(
        gm2ub_1d_to_2d(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*4,\s*16,\s*0,\s*0,\s*0,\s*0,\s*16,\s*256\);",
        source,
    )
    assert good is not None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * COLS, dtype=torch.int32, device=device)
    out = kernel(src)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src)


MATCHED_SRC_WIDTH = 8
MATCHED_DST_WIDTH = 16


def gm2ub_2d_to_2d():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS, MATCHED_SRC_WIDTH), "int32"),
        dst: T.Buffer((ROWS * COLS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, MATCHED_DST_WIDTH), "int32")
            T.copy(src[:, :COLS], buf[:, :COLS])
            T.copy(buf[:, :COLS], dst)

    return kernel


def test_gm2ub_2d_to_2d():
    kernel = tilelang.compile(
        gm2ub_2d_to_2d(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*4,\s*16,\s*0,\s*0,\s*0,\s*0,\s*32,\s*64\);",
        source,
    )
    assert good is not None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * MATCHED_SRC_WIDTH, dtype=torch.int32, device=device).reshape(ROWS, MATCHED_SRC_WIDTH)
    out = kernel(src)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src[:, :COLS].reshape(-1).contiguous())


def gm2ub_single_element_rows():
    @T.prim_func
    def kernel(
        src: T.Buffer((ROWS, WIDTH), "int32"),
        dst: T.Buffer((ROWS,), "int32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, 1), "int32")
            T.copy(src[:, 0:1], buf)
            T.copy(buf, dst)

    return kernel


def test_gm2ub_single_element_rows():
    kernel = tilelang.compile(
        gm2ub_single_element_rows(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*4,\s*4,\s*0,\s*0,\s*0,\s*0,\s*256,\s*4\);",
        source,
    )
    assert good is not None, source

    device = torch.device("npu")
    src = torch.arange(ROWS * WIDTH, dtype=torch.int32, device=device).reshape(ROWS, WIDTH)
    out = kernel(src)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src[:, 0].contiguous())


def runtime_divisor_copy():
    @T.prim_func
    def kernel(
        src: T.Tensor((1024, 64), "float32"),
        dst: T.Tensor((1024, 64), "float32"),
        num_cols: T.int32,
    ):
        with T.Kernel(1):
            buf = T.alloc_shared((32, 64), "float32")
            for i in T.serial(4):
                row = i // num_cols
                T.copy(src[row * 32, 0], buf)
                T.copy(buf, dst[row * 32, 0])

    return kernel


def test_mte_copy_hoists_let_bindings_before_calls():
    source = tilelang.lower(runtime_divisor_copy(), target="ascend").kernel_source

    for call_name in ("copy_gm_to_ubuf_align_v2", "copy_ubuf_to_gm_align_v2"):
        call_start = source.index(call_name)
        call_end = source.index(");", call_start)
        call = source[call_start:call_end]
        assert "int32_t rmod" not in call, source
        assert "int32_t rdiv" not in call, source

    assert source.count("int32_t rmod") == 2, source
    assert source.count("int32_t rdiv") == 2, source


def fp4_column_copy(cols):
    @T.prim_func
    def kernel(
        src: T.Tensor((ROWS, WIDTH), T.float4_e2m1fn),
        dst: T.Tensor((ROWS, WIDTH), T.float4_e2m1fn),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((ROWS, WIDTH), T.float4_e2m1fn)
            T.copy(src[:, :cols], buf[:, :cols])
            T.copy(buf[:, :cols], dst[:, :cols])

    return kernel


@pytest.mark.parametrize("cols", [1, 3, 5])
def test_fp4_odd_column_count_rejected(cols):
    with pytest.raises(Exception, match="packed sub-byte row must be byte-aligned"):
        tilelang.compile(fp4_column_copy(cols), out_idx=-1, target="ascend")


@pytest.mark.parametrize("cols", [2, 4, 6])
def test_fp4_even_column_count_allowed(cols):
    tilelang.compile(fp4_column_copy(cols), out_idx=-1, target="ascend")


DYNAMIC_ROWS = 4
OVERLAP = 2
DYNAMIC_WIDTH = 512


def dynamic_rows_middle_dim_copy():
    @T.prim_func
    def kernel(
        src: T.Buffer((DYNAMIC_ROWS, OVERLAP, DYNAMIC_WIDTH), "float32"),
        rows: T.Buffer((1,), "int32"),
        out: T.Buffer((DYNAMIC_ROWS, DYNAMIC_WIDTH), "float32"),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((DYNAMIC_ROWS, DYNAMIC_WIDTH), "float32")
            n = T.int32(rows[0])
            if n > 0:
                T.copy(src[0:n, 0, :], buf[0:n, :])
            T.copy(buf, out)

    return kernel


def test_gm2ub_dynamic_rows_middle_dim():
    kernel = tilelang.compile(
        dynamic_rows_middle_dim_copy(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    # The dynamic row count is clamped to the buffer's row extent (DYNAMIC_ROWS)
    # so an out-of-range `n` cannot issue an OOB DMA.
    good = re.search(
        rf"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*min\(n,\s*{DYNAMIC_ROWS}\),\s*2048,\s*0,\s*0,\s*0,\s*0,\s*4096,\s*2048\);",
        source,
    )
    bad = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*1,\s*\(n \* 2048\),\s*0,\s*0,\s*0,\s*0,",
        source,
    )
    assert good is not None, source
    assert bad is None, source

    device = torch.device("npu")
    src = torch.arange(DYNAMIC_ROWS * OVERLAP * DYNAMIC_WIDTH, dtype=torch.float32, device=device).reshape(
        DYNAMIC_ROWS, OVERLAP, DYNAMIC_WIDTH
    )
    rows = torch.tensor([DYNAMIC_ROWS], dtype=torch.int32, device=device)
    out = kernel(src, rows)
    torch.npu.synchronize()

    torch.testing.assert_close(out, src[:, 0, :])


def l0c_to_strided_gm_copy():
    ldd = T.dynamic("ldd", dtype="int64")
    m, n, k = 16, 16, 64

    @T.prim_func
    def kernel(
        a: T.Buffer((m, k), "bfloat16"),
        b: T.Buffer((n, k), "bfloat16"),
        out: T.StridedTensor((m, n), (ldd, 1), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((m, k), "bfloat16")
            b_l1 = T.alloc_l1((n, k), "bfloat16")
            acc = T.alloc_l0c((m, n), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(
                a_l1,
                b_l1,
                acc,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
            T.copy(acc, out[0, 0], unit_flag_ctrl=3)

    return kernel


def test_l0c_to_strided_gm_uses_runtime_row_stride():
    kernel = tilelang.compile(l0c_to_strided_gm_copy(), target="ascend")
    source = kernel.get_kernel_source()
    assert re.search(
        r"copy_matrix_cc_to_gm\([^;]*,\s*0,\s*16,\s*16,\s*\(\(int32_t\)ldd\),\s*16,",
        source,
    ), source

    device = torch.device("npu")
    a = torch.randn((16, 64), dtype=torch.bfloat16, device=device)
    b = torch.randn((16, 64), dtype=torch.bfloat16, device=device)
    storage = torch.empty((16, 32), dtype=torch.float32, device=device)
    out = storage[:, :16]
    kernel(a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.float() @ b.float().T)


def compact_l0_region_copy():
    alloc_m, alloc_n, alloc_k = 64, 64, 64
    tile_m, tile_n, tile_k = 17, 19, 18

    @T.prim_func
    def kernel(
        a: T.Buffer((alloc_m, alloc_k), "bfloat16"),
        b: T.Buffer((alloc_n, alloc_k), "bfloat16"),
        out: T.Buffer((tile_m, tile_n), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((alloc_m, alloc_k), "bfloat16")
            b_l1 = T.alloc_l1((alloc_n, alloc_k), "bfloat16")
            a_l0 = T.alloc_l0a((alloc_m, alloc_k), "bfloat16")
            b_l0 = T.alloc_l0b((alloc_n, alloc_k), "bfloat16")
            acc = T.alloc_l0c((alloc_m, alloc_n), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.copy(a_l1[0:tile_m, 0:tile_k], a_l0[0:tile_m, 0:tile_k])
            T.copy(b_l1[0:tile_n, 0:tile_k], b_l0[0:tile_n, 0:tile_k])
            T.gemm(
                a_l0[0:tile_m, 0:tile_k],
                b_l0[0:tile_n, 0:tile_k],
                acc[0:tile_m, 0:tile_n],
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(acc[0:tile_m, 0:tile_n], out, unit_flag_ctrl=3)

    return kernel


def test_compact_l0_regions_drive_load_mad_and_store_geometry():
    artifact = tilelang.lower(compact_l0_region_copy(), target="ascend")
    source = artifact.kernel_source

    # The L1 allocation keeps its 64-row / 4-fractal pitch, while each compact
    # L0 operand uses the 17/19-row region pitch (two fractals).
    assert re.search(r"load_cbuf_to_ca\([^;]*,\s*0,\s*0,\s*2,\s*2,\s*4,\s*2,\s*0\);", source), source
    assert re.search(r"load_cbuf_to_cb\([^;]*,\s*0,\s*0,\s*2,\s*2,\s*4,\s*2,\s*0\);", source), source
    assert re.search(r"mad\([^;]*,\s*17,\s*18,\s*19,", source), source
    assert re.search(r"copy_matrix_cc_to_gm\([^;]*,\s*0,\s*19,\s*17,\s*19,\s*32,", source), source


def dynamic_compact_l0_region_copy():
    alloc = 64

    @T.prim_func
    def kernel(
        a: T.Buffer((alloc, alloc), "bfloat16"),
        b: T.Buffer((alloc, alloc), "bfloat16"),
        out: T.Buffer((alloc, alloc), "float32"),
        sizes: T.Buffer((3,), "int32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((alloc, alloc), "bfloat16")
            b_l1 = T.alloc_l1((alloc, alloc), "bfloat16")
            a_l0 = T.alloc_l0a((alloc, alloc), "bfloat16")
            b_l0 = T.alloc_l0b((alloc, alloc), "bfloat16")
            acc = T.alloc_l0c((alloc, alloc), "float32")
            m = T.max(T.min(T.int32(sizes[0]), alloc), 0)
            n = T.max(T.min(T.int32(sizes[1]), alloc), 0)
            k = T.max(T.min(T.int32(sizes[2]), alloc), 0)
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.copy(a_l1[0:m, 0:k], a_l0[0:m, 0:k])
            T.copy(b_l1[0:n, 0:k], b_l0[0:n, 0:k])
            T.gemm(
                a_l0[0:m, 0:k],
                b_l0[0:n, 0:k],
                acc[0:m, 0:n],
                transpose_B=True,
                clear_accum=True,
            )
            T.copy(acc[0:m, 0:n], out[0:m, 0:n], unit_flag_ctrl=3)

    return kernel


def test_dynamic_compact_l0_region_uses_symbolic_geometry():
    artifact = tilelang.lower(dynamic_compact_l0_region_copy(), target="ascend")
    source = artifact.kernel_source
    m_blocks = r"\(\(\(m - 1\) >> 4\) \+ 1\)"
    n_blocks = r"\(\(\(n - 1\) >> 4\) \+ 1\)"
    k_blocks = r"\(\(\(k - 1\) >> 4\) \+ 1\)"
    assert re.search(
        rf"if \(\(0 < m\) && \(0 < k\)\) \{{\s*"
        rf"load_cbuf_to_ca\([^;]*,\s*0,\s*0,\s*{m_blocks},\s*{k_blocks},\s*4,\s*{m_blocks},\s*0\);",
        source,
    ), source
    assert re.search(
        rf"if \(\(0 < n\) && \(0 < k\)\) \{{\s*"
        rf"load_cbuf_to_cb\([^;]*,\s*0,\s*0,\s*{n_blocks},\s*{k_blocks},\s*4,\s*{n_blocks},\s*0\);",
        source,
    ), source
    assert re.search(r"mad\([^;]*,\s*m,\s*k,\s*n,", source), source
    assert re.search(
        r"copy_matrix_cc_to_gm\([^;]*,\s*0,\s*min\(n, 64\),\s*min\(m, 64\),\s*64,"
        r"\s*\(\(\(\(min\(m, 64\) - 1\) >> 4\) \* 16\) \+ 16\),",
        source,
    ), source


def gm_to_l1_fixed_outer_axis_copy():
    @T.prim_func
    def kernel(
        src: T.StridedTensor((4, 16, 64), (2000, 80, 1), "bfloat16"),
        group: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((16, 64), "bfloat16")
            group_idx = T.int32(group[0])
            T.copy(src[group_idx, 0:16, 0:64], l1)

    return kernel


def test_gm_to_l1_fixed_outer_axis_uses_effective_row_stride():
    artifact = tilelang.lower(gm_to_l1_fixed_outer_axis_copy(), target="ascend")
    source = artifact.kernel_source

    # The group stride is 2000 bf16 (4000B), while adjacent matrix rows are
    # only 80 bf16 (160B) apart. The fixed group axis belongs in the pointer
    # offset, not the ND2NZ row-stride argument.
    assert re.search(
        r"copy_gm_to_cbuf_multi_nd2nz\([^;]*,\s*0,\s*160,\s*0,\s*16,\s*64,",
        source,
    ), source
    assert not re.search(
        r"copy_gm_to_cbuf_multi_nd2nz\([^;]*,\s*0,\s*4000,\s*0,\s*16,\s*64,",
        source,
    ), source


def l0c_to_rank3_gm_copy():
    @T.prim_func
    def kernel(
        a: T.Buffer((16, 64), "bfloat16"),
        b: T.Buffer((16, 64), "bfloat16"),
        out: T.StridedTensor((4, 16, 16), (1000, 40, 1), "float32"),
        group: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 64), "bfloat16")
            b_l1 = T.alloc_l1((16, 64), "bfloat16")
            acc = T.alloc_l0c((16, 16), "float32")
            group_idx = T.int32(group[0])
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, out[group_idx, 0:8, 0:12], unit_flag_ctrl=3)

    return kernel


def test_l0c_to_rank3_gm_uses_effective_row_stride():
    artifact = tilelang.lower(l0c_to_rank3_gm_copy(), target="ascend")
    source = artifact.kernel_source
    assert re.search(
        r"copy_matrix_cc_to_gm\([^;]*,\s*0,\s*12,\s*8,\s*40,\s*16,",
        source,
    ), source


def l0c_to_oob_gm_copy():
    m, n, k, tile_m = 200, 16, 16, 128

    @T.prim_func
    def kernel(
        a: T.Buffer((tile_m, k), "bfloat16"),
        b: T.Buffer((n, k), "bfloat16"),
        out: T.Buffer((m, n), "float32"),
    ):
        with T.Kernel(2) as block:
            a_l1 = T.alloc_l1((tile_m, k), "bfloat16")
            b_l1 = T.alloc_l1((n, k), "bfloat16")
            acc = T.alloc_l0c((tile_m, n), "float32")
            T.copy(a, a_l1)
            T.copy(b, b_l1)
            T.gemm(
                a_l1,
                b_l1,
                acc,
                transpose_B=True,
                clear_accum=True,
                unit_flag_ctrl=3,
            )
            T.copy(
                acc,
                out[block * tile_m : (block + 1) * tile_m, :],
                unit_flag_ctrl=3,
            )

    return kernel


def test_l0c_to_oob_gm_preserves_source_geometry():
    artifact = tilelang.lower(l0c_to_oob_gm_copy(), target="ascend")
    source = artifact.kernel_source
    copy = re.search(r"copy_matrix_cc_to_gm\([^;]+\);", source)
    assert copy is not None, source
    args = copy.group(0)
    assert re.search(r",\s*0,\s*16,\s*min\(128,", args), args
    assert re.search(r",\s*16,\s*128,\s*0,\s*0,", args), args


def gm_to_l1_true_nd_copy():
    @T.prim_func
    def kernel(
        src: T.StridedTensor((2, 3, 4), (20, 5, 1), "bfloat16"),
    ):
        with T.Kernel(1):
            l1 = T.alloc_l1((6, 4), "bfloat16")
            T.copy(src, l1)

    return kernel


def test_gm_to_l1_true_nd_rejected():
    with pytest.raises(Exception, match="Outer-loop lowering is not yet implemented"):
        tilelang.lower(gm_to_l1_true_nd_copy(), target="ascend")


INCOMPATIBLE_A = 2
INCOMPATIBLE_B = 3
INCOMPATIBLE_C = 4
INCOMPATIBLE_SRC_A_STRIDE = INCOMPATIBLE_B * INCOMPATIBLE_C + 2
INCOMPATIBLE_DST_C_PAD = 1


def gm2ub_incompatible_2d():
    @T.prim_func
    def kernel(
        src: T.StridedTensor(
            shape=[INCOMPATIBLE_A, INCOMPATIBLE_B, INCOMPATIBLE_C],
            strides=[INCOMPATIBLE_SRC_A_STRIDE, INCOMPATIBLE_C, 1],
            dtype="float32",
        ),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared(
                (INCOMPATIBLE_A, INCOMPATIBLE_B, INCOMPATIBLE_C + INCOMPATIBLE_DST_C_PAD),
                "float32",
            )
            T.copy(src, buf[:, :, :INCOMPATIBLE_C])

    return kernel


def test_gm2ub_incompatible_2d_rejected():
    with pytest.raises(Exception, match="incompatible 2D row boundaries require outer-loop lowering"):
        tilelang.compile(
            gm2ub_incompatible_2d(),
            target="ascend",
        )


FP4_N = 1024


def fp4_copy():
    @T.prim_func
    def kernel(
        src: T.Tensor((FP4_N,), T.float4_e2m1fn),
        dst: T.Tensor((FP4_N,), T.float4_e2m1fn),
    ):
        with T.Kernel(1) as _:
            buf = T.alloc_shared((FP4_N,), T.float4_e2m1fn)
            T.copy(src, buf)
            T.copy(buf, dst)

    return kernel


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_gm2ub2gm_fp4_copy():
    kernel = tilelang.compile(
        fp4_copy(),
        out_idx=-1,
        target="ascend",
    )

    source = kernel.get_kernel_source()
    good_gm2ub = re.search(
        r"copy_gm_to_ubuf_align_v2\([^;]*,\s*0,\s*1,\s*512,\s*0,\s*0,\s*0,\s*0,\s*512,\s*512\);",
        source,
    )
    good_ub2gm = re.search(
        r"copy_ubuf_to_gm_align_v2\([^;]*,\s*0,\s*1,\s*512,\s*4,\s*512,\s*512\);",
        source,
    )
    assert good_gm2ub is not None, source
    assert good_ub2gm is not None, source

    device = torch.device("npu")
    nbytes = FP4_N // 2
    src_bytes = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cpu").to(device)
    src_fp4 = src_bytes.view(torch.float4_e2m1fn_x2)
    out = kernel(src_fp4)
    torch.npu.synchronize()

    assert out.shape == (nbytes,)
    assert out.dtype == torch.float4_e2m1fn_x2
    out_bytes = out.view(torch.uint8)
    torch.testing.assert_close(out_bytes.cpu(), src_bytes.cpu())


if __name__ == "__main__":
    tilelang.testing.main()
