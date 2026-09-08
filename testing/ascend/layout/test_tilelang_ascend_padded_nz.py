"""Inserted scatter and direct padded-NZ dual_copy must land identical data.

Both paths pack an ND tile into the NZ fractal layout an L1 Cube operand
expects. Multiplying by the identity recovers the original tile, so a correct
pack yields ``output == x`` -- and the two packing paths must agree.
"""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit
from tilelang.layout import make_ascend_compact_nz_layout

ROWS = 64
PADDED_ROWS = ROWS + 1
COLS = 128
C0 = 16


def _matmul_auto_scatter():
    """Dense ND UB -> L1: InsertNd2Nz auto-emits scatter + post_copy."""

    @T.prim_func
    def main(
        identity: T.Buffer((COLS, COLS), "bfloat16"),
        x: T.Buffer((ROWS * 2, COLS), "bfloat16"),
        output: T.Buffer((ROWS * 2, COLS), "float32"),
    ):
        with T.Kernel(1):
            identity_l1 = T.alloc_l1((COLS, COLS), "bfloat16")
            x_l1 = T.alloc_l1((ROWS * 2, COLS), "bfloat16")
            output_l0c = T.alloc_l0c((ROWS * 2, COLS), "float32")
            x_nd = T.alloc_shared((ROWS, COLS), "bfloat16")

            T.dual_copy(x, x_nd)
            T.dual_copy(x_nd, x_l1)
            T.copy(identity, identity_l1)
            T.gemm(x_l1, identity_l1, output_l0c, transpose_B=True, clear_accum=True)
            T.copy(output_l0c, output)

    return main


def _matmul_direct_compact_dual_copy():
    """Pre-packed compact padded-NZ UB -> L1: emit post_copy only."""

    @T.prim_func
    def main(
        identity: T.Buffer((COLS, COLS), "bfloat16"),
        x_nz: T.Buffer((PADDED_ROWS * 2, COLS), "bfloat16"),
        output: T.Buffer((ROWS * 2, COLS), "float32"),
    ):
        with T.Kernel(1):
            identity_l1 = T.alloc_l1((COLS, COLS), "bfloat16")
            x_l1 = T.alloc_l1((ROWS * 2, COLS), "bfloat16")
            output_l0c = T.alloc_l0c((ROWS * 2, COLS), "float32")
            x_nz_ub = T.alloc_shared((PADDED_ROWS, COLS), "bfloat16")
            T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})

            T.dual_copy(x_nz, x_nz_ub)
            T.dual_copy(x_nz_ub[0:ROWS, 0:COLS], x_l1)
            T.copy(identity, identity_l1)
            T.gemm(x_l1, identity_l1, output_l0c, transpose_B=True, clear_accum=True)
            T.copy(output_l0c, output)

    return main


def _matmul_explicit_ub_scatter():
    """ND UB -> compact-NZ UB via T.copy, followed by the existing post-copy."""

    @T.prim_func
    def main(
        identity: T.Buffer((COLS, COLS), "bfloat16"),
        x: T.Buffer((ROWS * 2, COLS), "bfloat16"),
        output: T.Buffer((ROWS * 2, COLS), "float32"),
    ):
        with T.Kernel(1):
            identity_l1 = T.alloc_l1((COLS, COLS), "bfloat16")
            x_l1 = T.alloc_l1((ROWS * 2, COLS), "bfloat16")
            output_l0c = T.alloc_l0c((ROWS * 2, COLS), "float32")
            x_nd_ub = T.alloc_shared((ROWS, COLS), "bfloat16")
            x_nz_ub = T.alloc_shared((PADDED_ROWS, COLS), "bfloat16")
            T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})

            T.dual_copy(x, x_nd_ub)
            T.copy(x_nd_ub, x_nz_ub[0:ROWS, 0:COLS])
            T.dual_copy(x_nz_ub[0:ROWS, 0:COLS], x_l1)
            T.copy(identity, identity_l1)
            T.gemm(x_l1, identity_l1, output_l0c, transpose_B=True, clear_accum=True)
            T.copy(output_l0c, output)

    return main


def test_compact_nz_layout_keeps_stages_tightly_packed():
    buffer = tirx.decl_buffer((2, PADDED_ROWS, 256), "float32", name="A", scope="shared")
    layout = make_ascend_compact_nz_layout(buffer)
    mapped = layout.map_forward_index([tirx.IntImm("int32", value) for value in (1, PADDED_ROWS - 1, 255)])

    assert [int(value) for value in layout.get_output_shape()] == [64, PADDED_ROWS, 8]
    assert [int(value) for value in mapped] == [63, PADDED_ROWS - 1, 7]


def test_auto_scatter_synthesizes_one_reusable_sid():
    estimated_costs = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureEstimatedCosts:
        def run_after_pass(self, mod, info):
            if info.name != "tl.EstimateLatency":
                return

            def visit(node):
                if not isinstance(node, tirx.AttrStmt) or node.attr_key != "tl.ascend_task":
                    return
                body = str(node.body)
                if "ascend_nd2nz_post_copy" in body:
                    estimated_costs["post_copy"] = (int(node.node["latency"]), int(node.node["ii"]))
                elif "x_nd" in body and "T.copy" in body:
                    estimated_costs["gm_to_ub"] = (int(node.node["latency"]), int(node.node["ii"]))

            post_order_visit(mod["main"].body, visit)

    with tvm.transform.PassContext(instruments=[CaptureEstimatedCosts()]):
        source = tilelang.lower(_matmul_auto_scatter(), target="ascend").kernel_source

    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert source.count("asc_copy_gm2ub_align") == 1
    assert source.count("asc_copy_ub2l1") == 1
    assert estimated_costs == {"gm_to_ub": (418, 328), "post_copy": (171, 128)}


def test_scatter_matches_direct_compact_nz_dual_copy():
    import torch

    identity = torch.eye(COLS, dtype=torch.bfloat16, device="npu")
    x = torch.arange(ROWS * 2 * COLS, dtype=torch.float32, device="npu").reshape(ROWS * 2, COLS).to(torch.bfloat16)
    x_nz_compact = x.reshape(2, ROWS, COLS // C0, C0).permute(0, 2, 1, 3).contiguous()
    x_nz = torch.zeros((2, COLS // C0, PADDED_ROWS, C0), dtype=x.dtype, device=x.device)
    x_nz[:, :, :ROWS, :] = x_nz_compact
    x_nz = x_nz.reshape(PADDED_ROWS * 2, COLS)

    out_auto = tilelang.compile(_matmul_auto_scatter(), out_idx=-1)(identity, x)
    out_explicit = tilelang.compile(_matmul_explicit_ub_scatter(), out_idx=-1)(identity, x)
    out_direct = tilelang.compile(_matmul_direct_compact_dual_copy(), out_idx=-1)(identity, x_nz)
    torch.npu.synchronize()

    torch.testing.assert_close(out_auto, out_explicit, rtol=0, atol=0)
    torch.testing.assert_close(out_auto, out_direct, rtol=0, atol=0)
    torch.testing.assert_close(out_direct, x.float(), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
