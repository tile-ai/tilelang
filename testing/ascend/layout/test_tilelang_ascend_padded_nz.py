"""Inserted, explicit and prepacked NZ copies deliver the same L1 matrix."""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T
from tvm import tirx
from tilelang.layout import make_ascend_compact_nz_layout

ROWS, COLS, C0 = 16, 128, 16
PADDED_ROWS = ROWS + 1


def _copy_to_l1(mode):
    input_rows = PADDED_ROWS if mode == "prepacked" else ROWS

    @T.prim_func
    def kernel(
        identity: T.Tensor((COLS, COLS), "bfloat16"),
        x: T.Tensor((input_rows * 2, COLS), "bfloat16"),
        out: T.Tensor((ROWS * 2, COLS), "float32"),
    ):
        with T.Kernel(1):
            identity_l1 = T.alloc_l1((COLS, COLS), "bfloat16")
            x_l1 = T.alloc_l1((ROWS * 2, COLS), "bfloat16")
            result = T.alloc_l0c((ROWS * 2, COLS), "float32")
            src = T.alloc_shared((input_rows, COLS), "bfloat16")
            if mode == "prepacked":
                T.annotate_layout({src: make_ascend_compact_nz_layout(src)})
            T.dual_copy(x, src)
            if mode == "explicit":
                packed = T.alloc_shared((PADDED_ROWS, COLS), "bfloat16")
                T.annotate_layout({packed: make_ascend_compact_nz_layout(packed)})
                T.copy(src, packed[:ROWS, :])
                T.dual_copy(packed[:ROWS, :], x_l1)
            else:
                T.dual_copy(src[:ROWS, :], x_l1)
            T.copy(identity, identity_l1)
            T.gemm(x_l1, identity_l1, result, transpose_B=True, clear_accum=True)
            T.copy(result, out)

    return kernel


def test_compact_nz_layout_keeps_stages_tightly_packed():
    buffer = tirx.decl_buffer((2, PADDED_ROWS, 256), "float32", name="A", scope="shared")
    layout = make_ascend_compact_nz_layout(buffer)
    mapped = layout.map_forward_index([tirx.IntImm("int32", value) for value in (1, PADDED_ROWS - 1, 255)])

    assert [int(value) for value in layout.get_output_shape()] == [64, PADDED_ROWS, 8]
    assert [int(value) for value in mapped] == [63, PADDED_ROWS - 1, 7]


@pytest.mark.parametrize("mode", ["inserted", "explicit", "prepacked"])
def test_nz_copy_preserves_logical_elements(mode):
    identity = torch.eye(COLS, dtype=torch.bfloat16, device="npu")
    x = torch.arange(ROWS * 2 * COLS, device="npu").reshape(ROWS * 2, COLS).to(torch.bfloat16)
    source = x
    if mode == "prepacked":
        packed = torch.zeros((2, COLS // C0, PADDED_ROWS, C0), dtype=x.dtype, device=x.device)
        packed[:, :, :ROWS, :] = x.reshape(2, ROWS, COLS // C0, C0).permute(0, 2, 1, 3)
        source = packed.reshape(PADDED_ROWS * 2, COLS)
    kernel = tilelang.compile(_copy_to_l1(mode), target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(identity, source), x.float(), rtol=0, atol=0)
