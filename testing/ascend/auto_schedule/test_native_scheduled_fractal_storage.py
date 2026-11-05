"""The scheduled value carrier preserves Cube transfer normalization."""

import pytest
import tilelang.ascend
from tilelang.ascend import transform
from tilelang.layout import make_ascend_nz_layout
from testing.ascend._ir import calls, copy, kernel, scheduled_fields, tensor, nodes
from testing.ascend.auto_schedule._scheduled_ir import unit
from tvm_tirx import ir


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("scope", ["shared.l0a", "shared.l0b"])
def test_fractal_tail_transfer_from_native_allocation(scheduled, scope):
    tilelang.ascend.register_backend()
    src = tensor((48, 16), "bfloat16", name="src", scope="shared.l1")
    dst = tensor((32, 16), "bfloat16", name="dst", scope=scope)
    before = kernel(
        unit(copy(src, dst, src_shape=[24, 16], dst_shape=[24, 16]), core=2),
        buffers=[src, dst],
        allocation_stage=0 if scheduled else None,
        annotations={"layout_map": {src: make_ascend_nz_layout(src), dst: make_ascend_nz_layout(dst)}},
    )
    after = transform.NormalizeAscendFractalStorage()(before)
    transfers = calls(after, "tl.tileop.ascend_copy")
    assert len(transfers) == 1
    for operand in transfers[0].args[:2]:
        assert isinstance(operand, ir.TensorRegion)
        assert int(operand.region[0].extent) == 32
        assert int(operand.region[1].extent) == 16
    # Normalization keeps the definition's stage and core metadata intact.
    markers = [scheduled_fields(binding) for binding in nodes(after, ir.Bind)]
    markers = [marker for marker in markers if marker is not None]
    assert len(markers) == (2 if scheduled else 0)
    for marker in markers:
        assert int(marker["stage"]) == 0
        assert int(marker["task"]["core_mask"]) == 7
