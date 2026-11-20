"""Numerical copy tails, including transposed Cube loads and untouched neighbors."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import calls, nodes


@pytest.mark.parametrize("rank", [1, 2], ids=["vector-tail", "matrix-column-tail"])
def test_dma_tail_roundtrip(rank):
    shape = (80,) if rank == 1 else (4, 80)
    tile_shape = (64,) if rank == 1 else (4, 64)

    @T.prim_func
    def main(src: T.Tensor(shape, "float32"), out: T.Tensor(shape, "float32")):
        with T.Kernel(2) as block:
            ub = T.alloc_shared(tile_shape, "float32")
            if rank == 1:
                T.copy(src[block * 64 : (block + 1) * 64], ub)
                T.copy(ub, out[block * 64 : (block + 1) * 64])
            else:
                T.copy(src[:, block * 64 : (block + 1) * 64], ub)
                T.copy(ub, out[:, block * 64 : (block + 1) * 64])

    src = torch.arange(80 if rank == 1 else 320, device="npu", dtype=torch.float32).reshape(shape)
    actual = tilelang.compile(main, out_idx=-1, target="ascend")(src)
    torch.testing.assert_close(actual, src)


@pytest.mark.parametrize("mutable_offset", [False, True], ids=["scalar-parameter", "local-variable"])
def test_opaque_store_tail_preserves_neighboring_storage(mutable_offset):
    @T.prim_func
    def main(src: T.Tensor((8, 32), "float32"), out: T.Tensor((12, 32), "float32"), row_offset: T.int32):
        with T.Kernel(1):
            ub = T.alloc_shared((8, 32), "float32")
            T.copy(src, ub)
            if mutable_offset:
                offset = T.alloc_var("int32", init=row_offset)
                T.copy(ub, out[offset : offset + 8, :])
            else:
                T.copy(ub, out[row_offset : row_offset + 8, :])

    src = torch.arange(1, 257, device="npu", dtype=torch.float32).reshape(8, 32)
    pool = torch.zeros((24, 32), device="npu")
    kernel = tilelang.compile(main, target="ascend")
    kernel(src, pool[:12], 8)
    expected = torch.zeros_like(pool)
    expected[8:12] = src[:4]
    torch.testing.assert_close(pool, expected)


@pytest.mark.parametrize("transpose", [False, True], ids=["direct", "transposed-nonsquare-load"])
def test_gemm_copy_tails_across_all_matrix_axes(transpose):
    m, n, k = 31, 19, 47
    tile_m, tile_n, tile_k = 16, 16, 32
    a_shape = (k, m) if transpose else (m, k)

    @T.prim_func
    def main(A: T.Tensor(a_shape, "bfloat16"), B: T.Tensor((n, k), "bfloat16"), C: T.Tensor((m, n), "float32")):
        with T.Kernel(4) as block:
            a = T.alloc_l1((tile_m, tile_k), "bfloat16")
            b = T.alloc_l1((tile_n, tile_k), "bfloat16")
            a0 = T.alloc_l0a((tile_m, tile_k), "bfloat16")
            b0 = T.alloc_l0b((tile_n, tile_k), "bfloat16")
            acc = T.alloc_l0c((tile_m, tile_n), "float32")
            for step in T.serial(2):
                if transpose:
                    T.copy(A[step * tile_k : (step + 1) * tile_k, block // 2 * tile_m : (block // 2 + 1) * tile_m], a, transpose=True)
                else:
                    T.copy(A[block // 2 * tile_m : (block // 2 + 1) * tile_m, step * tile_k : (step + 1) * tile_k], a)
                T.copy(B[block % 2 * tile_n : (block % 2 + 1) * tile_n, step * tile_k : (step + 1) * tile_k], b)
                T.copy(a, a0)
                T.copy(b, b0)
                T.gemm(a0, b0, acc, transpose_B=True, clear_accum=step == 0)
            T.copy(acc, C[block // 2 * tile_m : (block // 2 + 1) * tile_m, block % 2 * tile_n : (block % 2 + 1) * tile_n])

    a = torch.randn(a_shape, device="npu", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="npu", dtype=torch.bfloat16)
    actual = tilelang.compile(main, out_idx=-1, target="ascend")(a, b)
    expected = (a.float().T if transpose else a.float()) @ b.float().T
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("transpose", [False, True], ids=["k-contiguous", "mn-contiguous"])
@pytest.mark.parametrize("k_alloc", [64, 128])
def test_mx_dynamic_k_tail_prepares_each_operand(transpose, k_alloc):
    # Both operands use actual-sized copies, so ordinary OOB clamping alone
    # cannot provide the padding up to the MX K64 boundary.
    mn, sf_cols = 32, k_alloc // 64
    shape = (k_alloc, mn) if transpose else (mn, k_alloc)

    @T.prim_func
    def main(
        A: T.Tensor((mn, k_alloc), "float8_e4m3fn"),
        B: T.Tensor((mn, k_alloc), "float8_e4m3fn"),
        SA: T.Tensor((sf_cols, mn), "uint16"),
        SB: T.Tensor((sf_cols, mn), "uint16"),
        C: T.Tensor((mn, mn), "float32"),
        actual_k: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_l1(shape, "float8_e4m3fn")
            b = T.alloc_l1(shape, "float8_e4m3fn")
            sa = T.alloc_l1((mn, sf_cols), "uint16")
            sb = T.alloc_l1((mn, sf_cols), "uint16")
            a0 = T.alloc_l0a(shape, "float8_e4m3fn")
            b0 = T.alloc_l0b(shape, "float8_e4m3fn")
            sa0 = T.alloc_l0a_sf(a0, sf_shape=(mn, sf_cols))
            sb0 = T.alloc_l0b_sf(b0, sf_shape=(mn, sf_cols))
            acc = T.alloc_l0c((mn, mn), "float32")
            # Keep the GEMM/load geometry fixed while varying the last K64 tail.
            k = T.min(T.max(actual_k, k_alloc - 63), k_alloc)
            if transpose:
                T.copy(A[:, :k], a[:k, :], transpose=True)
                T.copy(B[:, :k], b[:k, :], transpose=True)
            else:
                T.copy(A[:, :k], a[:, :k])
                T.copy(B[:, :k], b[:, :k])
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.copy(a, a0)
            T.copy(b, b0)
            T.copy(sa, sa0)
            T.copy(sb, sb0)
            T.gemm_blockscaled(a0, b0, acc, sa0, sb0, transpose_A=transpose, transpose_B=not transpose, clear_accum=True)
            T.copy(acc, C)

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AscendInsertOOBPadding", "tl.InsertSync"):
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        tilelang.lower(main, target="ascend", enable_device_compile=False)
    tasks = [node for node in nodes(snapshots["tl.AscendInsertOOBPadding"], tirx.AttrStmt) if node.attr_key == "tl.ascend_task"]
    assert len(tasks) == 2
    for task in tasks:
        (copy,) = calls(task, "tl.tileop.ascend_copy")
        (fill,) = calls(task, "tl.tileop.fill")
        assert copy.args[1].args[0].buffer.same_as(fill.args[0].args[0].buffer)
    scheduled = snapshots["tl.InsertSync"]
    assert not [call for call in calls(scheduled, "tl.ascend_pipe_barrier") if call.args[0].value == "PIPE_MTE2"]
    for operation in ("set", "wait"):
        assert [call for call in calls(scheduled, "tl.ascend_" + operation + "_flag") if call.args[0].value == "MTE2_MTE1"]

    gen = torch.Generator().manual_seed(3)
    a = (torch.randint(-4, 5, (mn, k_alloc), generator=gen).float() / 8).to(torch.float8_e4m3fn)
    b = (torch.randint(-4, 5, (mn, k_alloc), generator=gen).float() / 8).to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (mn, k_alloc // 32), dtype=torch.uint8, generator=gen)
    sb = torch.randint(124, 131, (mn, k_alloc // 32), dtype=torch.uint8, generator=gen)
    a_scaled = a.float() * torch.pow(2.0, sa.float() - 127).repeat_interleave(32, dim=1)
    b_scaled = b.float() * torch.pow(2.0, sb.float() - 127).repeat_interleave(32, dim=1)
    compiled = tilelang.compile(main, out_idx=4, target="ascend")
    args = (a.npu(), b.npu(), sa.view(torch.uint16).T.contiguous().npu(), sb.view(torch.uint16).T.contiguous().npu())
    # Reuse the kernel across full tiles and both kinds of partial K64 group.
    for k in (k_alloc, k_alloc - 63, k_alloc - 32, k_alloc - 31, k_alloc - 1, k_alloc):
        actual = compiled(*args, k).cpu()
        expected = a_scaled[:, :k] @ b_scaled[:, :k].T
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5, msg=f"actual_k={k}")
