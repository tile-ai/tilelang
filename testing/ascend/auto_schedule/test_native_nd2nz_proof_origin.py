"""Explicit DMA pointers do not acquire a generated copy's source contract."""

import pytest
import tilelang.ascend
import tvm_ffi
from tilelang.ascend import transform
from tilelang.layout import make_ascend_compact_nz_layout, make_ascend_nz_layout
from testing.ascend._ir import calls, copy, kernel, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm_tirx import ir


def _pointer(buffer, indices, mask):
    return ir.Call(
        "tl.access_ptr",
        [ir.TensorLoad(buffer, indices), ir.const(1024), ir.const(mask)],
        ty=ir.PtrType(buffer.ty.dtype, buffer.ty.scope),
    )


@pytest.mark.parametrize("origin,expected", [("generated", 3), ("explicit", 1), ("stale_source", 1)])
def test_nd2nz_source_proof_requires_producer_contract(origin, expected):
    tilelang.ascend.register_backend()
    src = tensor((65, 128), "bfloat16", name="src", scope="shared.dyn", layout=None)
    dst = tensor((64, 16), "bfloat16", name="dst", scope="shared.l1", layout=None)
    if origin == "generated":
        operation = copy(src, dst, src_indices=[0, 16], src_shape=[64, 16])
    else:
        metadata = {"nd2nz_logical_regions": ir.const(2)} if origin == "stale_source" else {}
        operation = ir.Evaluate(
            tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
                ir.PrimType("void"),
                ir.Op.get("tl.ascend_nd2nz_post_copy"),
                [
                    _pointer(dst, [0, 0], 2),
                    _pointer(src, [0, 16], 1),
                    ir.const(64),
                    ir.const(16),
                    ir.const(64 if origin == "stale_source" else 65),
                    ir.const(64),
                    ir.StringImm("bfloat16_t"),
                ],
                metadata,
                ir.UnknownLoc(),
            )
        )
    before = kernel(
        unit(operation),
        buffers=[src, dst],
        annotations={
            "layout_map": {
                src: make_ascend_compact_nz_layout(src),
                dst: make_ascend_nz_layout(dst),
            }
        },
        allocation_stage=0,
    )
    first = transform.InsertNd2Nz()(before)
    second = transform.InsertNd2Nz()(first)
    for program in (first, second):
        (post,) = calls(program, "tl.ascend_nd2nz_post_copy")
        assert len(post.args) == 7
        assert int(post.attrs["nd2nz_logical_regions"]) == expected
    ir.assert_structural_equal(first, second)
