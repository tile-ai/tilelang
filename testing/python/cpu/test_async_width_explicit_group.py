"""Keep explicit logical instruction groups separate from automatic vector loads."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from testing.python.cpu.test_async_width_contract import CopyProgram, cp_calls, integer, nodes


def explicit_group(dtype, lanes, *, source_offset=0, source_stride=1, count=1, alignment=1):
    program = CopyProgram(dtype, source_alignment=alignment)
    source = ir.TensorLoad(program.source, [program.lane * source_stride + source_offset])
    shared = ir.TensorLoad(program.shared, [program.lane])
    src_ptr = ir.Call("tirx.address_of", [source])
    dst_ptr = ir.Call("tirx.address_of", [shared])
    call = ir.Call("tl.ptx_cp_async", [dst_ptr, src_ptr, integer(count)], ty_args=[ir.PrimType("void")])
    function = program.function([program.loop([ir.Evaluate(call)], lanes=lanes)])
    return function.with_attr("target", Target({"kind": "cuda", "arch": "sm_80"}))


@pytest.mark.parametrize("dtype,lanes", [("float16", 4), ("int4", 32), ("uint8", 16), ("int2", 64), ("int1", 128)])
def test_complete_explicit_group_keeps_logical_bits_and_count(dtype, lanes, tmp_path):
    function = explicit_group(dtype, lanes)
    before = ir.save_json(function)
    result = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": function}))["main"]
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(result)
    calls = cp_calls(result)
    assert len(calls) == 1 and int(calls[0].args[2]) == lanes
    assert not nodes(result, ir.For)
    assert int(function.params[0].ty.data_alignment) == 1
    (tmp_path / "after.py").write_text(result.script())


@pytest.mark.parametrize("boundary", ["unaligned_offset", "noncontiguous", "legal_scalar_alignment"])
def test_explicit_group_does_not_invent_an_address_or_widen_legal_unaligned_calls(boundary, tmp_path):
    if boundary == "unaligned_offset":
        function = explicit_group("float16", 4, source_offset=1)
    elif boundary == "noncontiguous":
        function = explicit_group("float16", 4, source_stride=2)
    else:
        function = explicit_group("float32", 4)
    result = tvm_ffi.get_global_func("tl.transform.VectorizeLoop")(True)(ir.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    assert nodes(result, ir.For)
    assert all(int(call.args[2]) == 1 for call in cp_calls(result))
    (tmp_path / "after.py").write_text(result.script())
