"""Python consumers query the native operator's actual effect regions."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from test_native_reducer_contracts import region, tensor, tileop


def query(call, annotations=None):
    api = tvm_ffi.get_global_func("tl.GetTileOpAccessRegions")
    return api(call) if annotations is None else api(call, annotations)


def test_copy_reports_exact_source_and_destination_regions():
    source, destination = tensor("A", [8, 16]), tensor("B", [2, 4])
    selected = ir.TensorRegion(source, [ir.Range(3, 5), ir.Range(7, 11)], ir.TensorRegionType())
    call = tileop("copy", selected, region(destination)).value
    result = query(call)
    assert len(result["reads"]) == len(result["writes"]) == 1
    assert tvm_ffi.structural_equal(result["reads"][0], selected)
    assert result["reads"][0].source.same_as(source)
    assert result["writes"][0].source.same_as(destination)


def test_fill_has_no_read_effect():
    destination = tensor("B", [4])
    result = query(tileop("fill", region(destination), ir.const(3.0)).value)
    assert len(result["reads"]) == 0
    assert len(result["writes"]) == 1
    assert result["writes"][0].source.same_as(destination)


@pytest.mark.parametrize("mask,reads,writes", [(1, 1, 0), (2, 0, 1), (3, 1, 1)])
def test_custom_registered_operator_and_lexical_handler(mask, reads, writes):
    source = tensor("custom", [8])
    selected = ir.TensorRegion(source, [ir.Range(2, 6)], ir.TensorRegionType())
    call = tvm_ffi.get_global_func("tl.MakeAnnotatedCall")(
        ir.PrimType("void"), ir.Op.get("tl.testing.access_probe"), [selected], {}, ir.UnknownLoc()
    )
    # The test operator refuses layout inference and lowering. It is also not
    # a built-in tl.tileop.* spelling, so no consumer-side operand table works.
    result = query(call, {"testing.access_mask": mask})
    assert len(result["reads"]) == reads and len(result["writes"]) == writes
    for value in [*result["reads"], *result["writes"]]:
        assert value.source.same_as(source)
        assert tvm_ffi.structural_equal(value, selected)


def test_query_rejects_non_tile_operator():
    call = tirx.call_extern("unrelated", ty="void")
    with pytest.raises(ValueError, match="registered TileLang tile operator"):
        query(call)
