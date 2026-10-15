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


def _partial_copy_to_l1(padding, parts, physical_view=False, staged=False):
    rows, cols = (32 if physical_view else 64), 64
    input_shape = (2 * 4096,) if physical_view else (64, 64)
    chunk = rows // parts

    @T.prim_func
    def kernel(identity: T.Tensor((64, 64), "bfloat16"), x: T.Tensor(input_shape, "bfloat16"), out: T.Tensor((64, 64), "float32")):
        with T.MixedKernel(1) as (_, sid):
            identity_l1 = T.alloc_l1((64, 64), "bfloat16")
            x_l1 = T.alloc_l1((64, 64), "bfloat16")
            result = T.alloc_l0c((64, 64), "float32")
            if physical_view:
                owner = T.alloc_shared((32, 64), "float32")
                view = T.view(owner, shape=(4096,), dtype="bfloat16")
                T.copy(x[sid * 4096 : (sid + 1) * 4096], view)
                for part in T.unroll(parts):
                    T.ascend_nd2nz_post_copy(
                        x_l1[sid * rows + part * chunk, 0],
                        view[part * chunk * 16],
                        chunk,
                        cols,
                        src_pitch=64,
                        dst_pitch=64,
                        dtype="bfloat16",
                    )
            else:
                nd = T.alloc_shared((rows, cols), "bfloat16")
                nz = T.alloc_shared((2, rows + padding, cols) if staged else (rows + padding, cols), "bfloat16")
                T.annotate_layout({nz: make_ascend_compact_nz_layout(nz)})
                if sid == 0:
                    T.copy(x, nd)
                    if staged:
                        T.copy(nd, nz[1, :rows, :])
                        for part in T.unroll(parts):
                            T.copy(nz[1, part * chunk : (part + 1) * chunk, :], x_l1[part * chunk : (part + 1) * chunk, :])
                    else:
                        # Scatter partial row regions too: they retain rows+padding.
                        for half in T.unroll(2):
                            T.copy(nd[half * 32 : (half + 1) * 32, :], nz[half * 32 : (half + 1) * 32, :])
                        for part in T.unroll(parts):
                            T.copy(nz[part * chunk : (part + 1) * chunk, :], x_l1[part * chunk : (part + 1) * chunk, :])
            T.copy(identity, identity_l1)
            T.gemm(x_l1, identity_l1, result, transpose_B=True, clear_accum=True)
            T.copy(result, out)

    return kernel


@pytest.mark.parametrize("padding,parts,staged", [(0, 2, False), (1, 2, False), (7, 4, False), (1, 8, False), (3, 4, True)])
def test_partial_copies_keep_full_buffer_pitch(padding, parts, staged):
    identity = torch.eye(64, dtype=torch.bfloat16, device="npu")
    x = (torch.arange(4096, device="npu") % 127).reshape(64, 64).to(torch.bfloat16)
    kernel = tilelang.compile(_partial_copy_to_l1(padding, parts, staged=staged), target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(identity, x), x.float(), rtol=0, atol=0)


@pytest.mark.parametrize("parts", [1, 2, 4])
def test_existing_bf16_physical_view_keeps_owner_and_pitch(parts):
    identity = torch.eye(64, dtype=torch.bfloat16, device="npu")
    x = (torch.arange(4096, device="npu") % 127).reshape(64, 64).to(torch.bfloat16)
    packed = torch.full((2, 4, 64, 16), -13, dtype=x.dtype, device=x.device)
    packed[:, :, :32, :] = x.reshape(2, 32, 4, 16).permute(0, 2, 1, 3)
    kernel = tilelang.compile(_partial_copy_to_l1(0, parts, physical_view=True), target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(identity, packed.flatten()), x.float(), rtol=0, atol=0)


def test_post_copy_from_offset_physical_view():
    offset, size = 16, 32 * 64

    @T.prim_func
    def main(identity: T.Tensor((64, 64), "bfloat16"), x: T.Tensor((2, offset + size), "bfloat16"), out: T.Tensor((64, 64), "float32")):
        with T.MixedKernel(1) as (_, sid):
            owner = T.alloc_shared((offset + size,), "bfloat16")
            view = tirx.decl_buffer((size,), "bfloat16", data=owner.data, elem_offset=offset, scope="shared.dyn")
            x_l1 = T.alloc_l1((64, 64), "bfloat16")
            identity_l1 = T.alloc_l1((64, 64), "bfloat16")
            result = T.alloc_l0c((64, 64), "float32")
            T.copy(x[sid, :], owner)
            T.ascend_nd2nz_post_copy(x_l1[sid * 32, 0], view[0], 32, 64, src_pitch=32, dst_pitch=64, dtype="bfloat16")
            T.copy(identity, identity_l1)
            T.gemm(x_l1, identity_l1, result, transpose_B=True, clear_accum=True)
            T.copy(result, out)

    identity = torch.eye(64, dtype=torch.bfloat16, device="npu")
    x = (torch.arange(4096, device="npu") % 127).reshape(64, 64).to(torch.bfloat16)
    packed = torch.full((2, offset + size), -13, dtype=x.dtype, device=x.device)
    packed[:, offset:] = x.reshape(2, 32, 4, 16).permute(0, 2, 1, 3).reshape(2, size)
    kernel = tilelang.compile(main, target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(identity, packed), x.float(), rtol=0, atol=0)
