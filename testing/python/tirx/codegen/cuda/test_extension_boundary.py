import pytest
import tvm_ffi

from tvm_tirx import ir, tirx
from tvm_tirx.backend.cuda import op as cuda
from tvm_tirx.backend.cuda.ptx import PTXNamespace
from tvm_tirx.ir import CallEffectKind
from tvm_tirx.target import Target

from tilelang._compat.cuda_registry import get_codegen


def test_cuda_host_has_one_ffi_only_prelude():
    function = tirx.Function([], ir.Return(ir.const(0)), ret_type=ir.PrimType("int32"))
    function = function.with_attr("global_symbol", "call")
    source = tvm_ffi.get_global_func("tl.codegen.build_cuda_host")(ir.IRModule({"call": function}), Target("cuda_host")).inspect_source()
    assert source.count("void* __tvm_ffi__library_ctx =") == 1
    assert "tvm/runtime/" not in source
    assert source.index("#include <tvm/ffi/c_api.h>") < source.index("int32_t call(")


def test_shared_address_conversion_keeps_native_effect_contract():
    pointer = ir.Var("pointer", ir.PtrType("uint64", "shared"))
    converted = cuda.cvta_generic_to_shared(pointer)
    assert converted.op.get_attr("TCallEffectKind") == CallEffectKind.Opaque
    assert converted.ty == ir.PrimType("uint32")
    assert converted.args[0].same_as(pointer)
    lowered, tags = get_codegen(converted.op.name)(list(converted.args))
    assert lowered is not None


@pytest.mark.parametrize("name", ["cp.async.commit_group", "cp.async.wait_group", "mbarrier.init.shared.b64"])
@pytest.mark.parametrize("with_metadata", [False, True])
def test_ptx_codegen_accepts_attributes_without_changing_native_registry(name, with_metadata):
    operands = []
    if name == "cp.async.wait_group":
        operands = [0]
    elif name == "mbarrier.init.shared.b64":
        operands = [ir.Var("barrier", ir.PtrType("uint64", "shared")), 32]
    call = PTXNamespace()[name](*operands)
    lower = get_codegen(call.op.name)
    expected, expected_tags = lower(list(call.args))
    attributes = ir.DictAttrs({"tl.access_mask": ir.const(3)} if with_metadata else {})
    actual, actual_tags = lower(list(call.args), attributes)
    ir.assert_structural_equal(actual, expected)
    assert actual_tags == expected_tags
