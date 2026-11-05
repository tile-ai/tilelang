"""SM120 MMA operands use the byte-packed backing of native local FP4 tensors."""

import re

import pytest
import tvm_ffi
from tvm_tirx import ir, target, tirx

import tilelang.testing
from tilelang.language._native import call_intrin


CUDA = target.Target({"kind": "cuda", "arch": "sm_120"})
FP4 = "float4_e2m1fn"


def _program(mode="data", offsets=(32, 16), names=("a", "b"), config=None):
    parameters, statements = [], []

    def allocate(name, dtype, length):
        ty = tirx.TensorType((length,), dtype, scope="local")
        value = ir.Var(name, ty)
        statements.append(ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[ty])))
        return value

    # A same-named external pointer must not inherit a local tensor's entry.
    external = None
    if mode in ("external", "byte_offset"):
        external = ir.Var(names[0], ir.PtrType(FP4, "global"))
        parameters.append(external)
    a, b = allocate(names[0], FP4, 128), allocate(names[1], FP4, 128)
    c = allocate("accumulator", "float32", 16)
    sa, sb = allocate("scale_a", "uint32", 1), allocate("scale_b", "uint32", 1)
    a_ptr, b_ptr = a.data, b.data
    if mode == "bare_tensor":
        # Keep the previous direct-Var emitter path as an explicit control.
        a_ptr, b_ptr = a, b
    elif mode == "external":
        a_ptr = external
    elif mode == "byte_offset":
        a_ptr = tirx.ptr_byte_offset(external, ir.const(7, "int64"), ty=external.ty)
    elif mode != "data":
        raise AssertionError(mode)

    settings = ["float32", "m16n8k64", "row", "col", "mxf4nvf4", 4, "e2m1", "e2m1", "ue4m3"]
    for index, value in (config or {}).items():
        settings[index] = value
    operation = call_intrin(
        "float32",
        "tl.ptx_mma_block_scale",
        *settings,
        a_ptr,
        offsets[0],
        b_ptr,
        offsets[1],
        c.data,
        4,
        sa.data,
        sb.data,
        0,
        0,
        0,
        0,
    )
    statements.append(ir.Evaluate(operation))
    function = tirx.Function(parameters, statements, ret_type=ir.PrimType("void"))
    return function.with_attrs({"global_symbol": "packed_mma", "calling_conv": ir.CallingConv.DEVICE_KERNEL_LAUNCH})


def _emit(value, tmp_path):
    before = ir.save_json(value)
    assert tirx.analysis.check_well_formed(value)
    module = ir.IRModule({"packed_mma": value})
    result = tvm_ffi.get_global_func("target.build.tilelang_cuda_without_compile")(module, CUDA)
    source = result.inspect_source("cuda")
    assert ir.save_json(value) == before
    (tmp_path / "input.json").write_text(before)
    (tmp_path / "kernel.cu").write_text(source)
    return source


def _mma_line(source):
    lines = [line for line in source.splitlines() if "tl::sm120_mma_sync_blockscaled<" in line]
    assert len(lines) == 1
    return lines[0]


@pytest.mark.parametrize("mode", ["data", "bare_tensor"])
@pytest.mark.parametrize("offsets", [(0, 0), (16, 32), (64, 48)])
def test_local_fp4_operands_use_declared_packed_storage(mode, offsets, tmp_path):
    source = _emit(_program(mode, offsets), tmp_path)
    assert "fp4_e2_2_t a_packed[64]" in source
    assert "fp4_e2_2_t b_packed[64]" in source
    line = _mma_line(source)
    assert f"a_packed + ({offsets[0]}) / 2" in line
    assert f"b_packed + ({offsets[1]}) / 2" in line
    assert "accumulator + 4" in line
    assert line.count(" / 2") == 2
    assert not re.search(r"\b[ab]\b", line)


def test_same_named_native_tensors_keep_distinct_backings(tmp_path):
    source = _emit(_program(names=("operand", "operand")), tmp_path)
    declarations = re.findall(r"fp4_e2_2_t (operand\w*_packed)\[64\]", source)
    assert len(declarations) == 2 and len(set(declarations)) == 2
    line = _mma_line(source)
    assert f"{declarations[0]} + (32) / 2" in line
    assert f"{declarations[1]} + (16) / 2" in line


@pytest.mark.parametrize("mode", ["external", "byte_offset"])
def test_unmapped_pointer_keeps_identity_and_displacement_units(mode, tmp_path):
    source = _emit(_program(mode, names=("a", "b")), tmp_path)
    line = _mma_line(source)
    assert "a_1_packed" not in line
    assert "b_packed + (16) / 2" in line
    assert line.count(" / 2") == 1
    if mode == "external":
        assert "a + 32" in line
    else:
        assert "((char*)a) + (int64_t)7" in line
        assert "+ 32" in line


@pytest.mark.parametrize("change", [{1: "m16n8k32"}, {5: 2}, {8: "ue8m0"}])
def test_unsupported_blockscale_configuration_stays_rejected(change, tmp_path):
    with pytest.raises(RuntimeError, match="Unsupported ptx_mma_block_scale configuration"):
        _emit(_program(config=change), tmp_path)


if __name__ == "__main__":
    tilelang.testing.main()
