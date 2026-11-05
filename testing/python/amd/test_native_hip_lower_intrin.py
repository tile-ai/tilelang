"""Keep the HIP source target distinct from TIRx's LLVM ROCm lowering."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx.ir import prim as P

from testing.python.amd.test_native_codegen_hip import _tensor_arg

I = tvm_tirx.ir


def _lower(function):
    target = tvm_tirx.target.Target({"kind": "hip", "mcpu": "gfx90a"})
    function = function.with_attr("target", target).with_attr("global_symbol", "main").with_attr("calling_conv", 2)
    module = tvm_ffi.get_global_func("tl.transform.LowerIntrin")()(tvm_tirx.IRModule({"main": function}))
    source = tvm_ffi.get_global_func("target.build.tilelang_hip_without_compile")(module, target).inspect_source("hip")
    return module, source


@pytest.mark.parametrize("lanes", [1, 4, 8])
@pytest.mark.parametrize("op,extern", [("round", "nearbyintf"), ("exp2", "exp2f"), ("exp", "__expf"), ("tan", "tanf")])
def test_hip_math_rules_scalarize_vector_carriers(lanes, op, extern, tmp_path):
    declarations = [_tensor_arg(name, "float32", lanes) for name in ("input", "output")]
    (_, source, _), (_, output, _) = declarations
    index = I.const(0) if lanes == 1 else P.Ramp(0, 1, lanes)
    result = I.Call("prim." + op, [I.TensorLoad(source, [index])])
    function = tvm_tirx.tirx.Function(
        [param for param, _, _ in declarations],
        [*[bind for _, _, bind in declarations], I.TensorStore(output, [index], result)],
        ret_type=I.PrimType("void"),
    )
    lowered, source = _lower(function)
    (tmp_path / "lowered.py").write_text(lowered.script())
    (tmp_path / "kernel.hip").write_text(source)
    assert source.count(extern + "(") == lanes
    assert "__ocml_" not in source and "llvm." not in lowered.script()
    if op == "round":
        assert "roundf(" not in source


@pytest.mark.parametrize(
    "suffix,expected", [("", "__shfl_sync"), ("_up", "__shfl_up_sync"), ("_down", "__shfl_down_sync"), ("_xor", "__shfl_xor_sync")]
)
def test_hip_shuffle_uses_four_explicit_operands(suffix, expected, tmp_path):
    value = I.Var("value", "float32")
    pointer, tensor, bind = _tensor_arg("output", "float32", 1)
    shuffle = I.Call("tirx.gpu_warp_shuffle" + suffix, [I.const(-1), value, I.const(1), I.const(32), I.const(64)])
    function = tvm_tirx.tirx.Function([value, pointer], [bind, I.TensorStore(tensor, [0], shuffle)], ret_type=I.PrimType("void"))
    lowered, source = _lower(function)
    (tmp_path / "lowered.py").write_text(lowered.script())
    (tmp_path / "kernel.hip").write_text(source)
    store = lowered["main"].body.seq[-1]
    assert store.value.op.name == "tirx.hip." + expected
    assert len(store.value.args) == 4
    assert expected + "(" in source


def test_native_hip_target_options_and_device_identity():
    target = tvm_tirx.target.Target({"kind": "hip", "mcpu": "gfx1100", "thread_warp_size": 32})
    assert target.kind.name == "hip"
    assert target.kind.default_device_type == 10
    assert list(target.keys) == ["hip", "gpu"]
    assert target.mcpu == "gfx1100" and target.thread_warp_size == 32
    assert target.mtriple == "amdgcn-amd-amdhsa-hcc"
    with pytest.raises(ValueError, match="invalid GFX version"):
        tvm_tirx.target.Target({"kind": "hip", "mcpu": "invalid"})
