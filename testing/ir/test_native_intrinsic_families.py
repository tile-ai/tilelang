"""Native intrinsic families own executable operands, result types, and effects."""

import pytest
import tvm_ffi
from tvm_tirx import ir

import tilelang.language as T


def _pointer(name="pointer", dtype="float32", scope="global"):
    return ir.Var(name, ir.PtrType(dtype, scope))


def _expr(value):
    if isinstance(value, str):
        return ir.StringImm(value)
    if isinstance(value, bool):
        return ir.const(value, "bool")
    if isinstance(value, int):
        return ir.const(value, "int32")
    return value


def _call(name, values, **kwargs):
    return ir.Call(name, [_expr(value) for value in values], **kwargs)


def _annotated(name, operands, result_type):
    tag = ir.Var("tag", "int32")
    attrs = {
        "schema_version": 1,
        "operand_count": len(operands),
        "annotation_schema": ["map", [["str", "tag"], ["expr", 0]]],
    }
    call = ir.Call(name, [_expr(value) for value in operands] + [ir.Tuple([tag])], attrs=attrs, ty=result_type)
    call.validate()
    restored = ir.load_json(ir.save_json(call))
    restored.validate()
    assert tvm_ffi.structural_equal(call, restored, map_free_vars=True)
    assert call.args[-1].fields[0].same_as(tag)
    return call


@pytest.mark.parametrize("dtype", ["float32", "float32x4", "float16x2"])
def test_arithmetic_infers_the_operand_type_without_a_result_hint(dtype):
    value = ir.Var("value", dtype)
    for name, operands in [
        ("tl.fmul", [value, value]),
        ("tl.fma", [value, value, value]),
        ("tl.round_ties_away_from_zero", [value]),
        ("tl.pow_of_int", [value, 3]),
    ]:
        call = _call(name, operands)
        assert call.ty == value.ty
        assert ir.reinfer_type(call) == value.ty
        call.validate()
        _annotated(name, operands, value.ty)


def test_arithmetic_cannot_adopt_a_pointer_or_unrelated_lane_type():
    value = ir.Var("value", "float32x4")
    for other, result in [(ir.Var("other", "float32"), value.ty), (value, ir.PtrType("void"))]:
        call = _call("tl.fmul", [value, other], ty=result)
        with pytest.raises(TypeError):
            call.validate()


def test_infinity_owns_its_dtype_operand():
    call = _call("tl.infinity", ["float64"])
    assert call.ty == ir.PrimType("float64")
    call.validate()
    inconsistent = _call("tl.infinity", ["float64"], ty="float32")
    with pytest.raises(TypeError, match="result type disagrees"):
        inconsistent.validate()


def test_fixed_results_and_memory_effects_are_native_contracts():
    pointer = _pointer()
    cases = [
        ("tl.any_of", [pointer, 8], "bool", ir.CallEffectKind.ReadState),
        ("tl.all_of", [pointer, 8], "bool", ir.CallEffectKind.ReadState),
        ("tl.rng_rand", [], "uint32", ir.CallEffectKind.Opaque),
        ("tl.ldg128", [pointer, True], "uint32x4", ir.CallEffectKind.ReadState),
        ("tl.lds64", [pointer], "uint32x2", ir.CallEffectKind.ReadState),
        ("tl.ds_read_tr16_b64", [pointer], "uint32x2", ir.CallEffectKind.Pure),
        ("tl.simd.pset", [32, "PAT_ALL"], "boolx256", ir.CallEffectKind.Opaque),
    ]
    for name, operands, dtype, effect in cases:
        call = _call(name, operands)
        call.validate()
        assert call.ty == ir.PrimType(dtype)
        assert call.op.get_attr("TCallEffectKind") == effect
        _annotated(name, operands, call.ty)
    assert ir.Op.get("tl.fmul").get_attr("TVectorizable")
    assert ir.Op.get("tl.fma").get_attr("TVectorizable")
    assert not ir.Op.get("tl.any_of").get_attr("TVectorizable")


def test_load_result_and_runtime_predicate_are_not_untyped_scalar_slots():
    pointer = _pointer()
    for name, operands, dtype in [
        ("tl.ldg128", [pointer], "float32x4"),
        ("tl.ldg32", [pointer, ir.const(1.0, "float32")], "uint32"),
        ("tl.lds64", [ir.const(1, "int64")], "uint32x2"),
    ]:
        call = _call(name, operands, ty=dtype)
        with pytest.raises(TypeError):
            call.validate()


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_tensor_map_rank_controls_its_four_operand_vectors(rank):
    operands = [0, rank, _pointer()] + [16] * rank + [4] * rank + [16] * rank + [1] * rank + [0, 0, 0, 0]
    call = _call("tl.create_tma_descriptor", operands)
    assert isinstance(call.ty, ir.PtrType)
    call.validate()
    _annotated("tl.create_tma_descriptor", operands, ir.PtrType("void"))
    truncated = _call("tl.create_tma_descriptor", operands[:-1], ty=ir.PtrType("void"))
    with pytest.raises(ValueError, match="runtime operands"):
        truncated.validate()


@pytest.mark.parametrize("datapaths", [None, 16, 32])
@pytest.mark.parametrize("name", ["tl.tcgen05_ld", "tl.tcgen05_st"])
def test_tensor_memory_transfer_preserves_both_instruction_forms(name, datapaths):
    operands = [32, 1, False, ir.Var("tmem_column", "uint32"), 0]
    if datapaths is not None:
        operands.append(datapaths)
    operands.append(_pointer(scope="local"))
    _annotated(name, operands, "void")
    call = _call(name, operands[:-1], ty="void")
    with pytest.raises((ValueError, TypeError)):
        call.validate()


def _matrix_case(name):
    a = _pointer("a", "float16", "local")
    b = _pointer("b", "float16", "local")
    c = _pointer("c", "float32", "local")
    desc = ir.Var("descriptor", "uint64")
    instruction = ir.Var("instruction", "uint32")
    if name == "tl.ptx_wgmma_rs":
        return ["m64n64k16", True, "f16", "f16", "f32", a, 0, desc, 0, c, 0, True, 1, 1]
    if name == "tl.ptx_tcgen05_mma_ss":
        return ["f16", desc, 0, desc, 0, c, 0, instruction, True, 0, 0, 0, 0, False, False]
    if name == "tl.ptx_tcgen05_mma_ts":
        return ["f16", a, 0, desc, 0, c, 0, instruction, True, 0, 0, 0, 0, False]
    if name == "tl.ptx_mma_sm70":
        return ["m8n8k4", "row", "col", "fp16", "fp16", "fp32", a, 0, b, 0, c, 0]
    if name == "tl.ascend_gemm_l1":
        return [c, a, b, 16, 16, 16, 16, False, True, "half", 0, 0]
    if name == "tl.ascend_blockscaled_gemm_l1":
        return [
            c,
            a,
            b,
            _pointer("sfa", "uint8"),
            _pointer("sfb", "uint8"),
            16,
            32,
            16,
            32,
            False,
            True,
            "half",
            "uint8_t",
            "float",
            0,
            0,
            1,
            0,
        ]
    assert name == "tl.ascend_copy_gm_to_cbuf"
    return [a, b, 0, 32, 0, 16, 16, 0, 0, False, 1, "half"]


# These arities are the current frontend/Copy producers and emitter ABIs, not
# the stale counts formerly stored in the generic intrinsic attributes.
@pytest.mark.parametrize(
    "name,arity",
    [
        ("tl.ptx_wgmma_rs", 14),
        ("tl.ptx_tcgen05_mma_ss", 15),
        ("tl.ptx_tcgen05_mma_ts", 14),
        ("tl.ptx_mma_sm70", 12),
        ("tl.ascend_gemm_l1", 12),
        ("tl.ascend_blockscaled_gemm_l1", 18),
        ("tl.ascend_copy_gm_to_cbuf", 12),
    ],
)
def test_matrix_and_dma_contracts_match_current_producers(name, arity):
    operands = _matrix_case(name)
    assert len(operands) == arity
    _annotated(name, operands, "void")
    for malformed in [operands[:-1], operands + [0]]:
        call = _call(name, malformed, ty="void")
        with pytest.raises(ValueError, match="runtime operands"):
            call.validate()


@pytest.mark.parametrize("dtype", ["void", "float32", ir.PtrType("void")])
def test_matrix_statement_retains_historical_result_spellings(dtype):
    call = _call("tl.ptx_wgmma_rs", _matrix_case("tl.ptx_wgmma_rs"), ty=dtype)
    call.validate()
    assert call.op.get_attr("TCallEffectKind") == ir.CallEffectKind.Opaque


@pytest.mark.parametrize(
    "name,operands", [("tl.create_barriers", [0]), ("tl.ptx_wait_group", [8]), ("tl.ptx_arrive_barrier_expect_tx", [0, 16, 0])]
)
def test_legacy_barrier_checks_survive_ownership_move(name, operands):
    call = _call(name, operands, ty="void")
    with pytest.raises(ValueError):
        call.validate()


def test_legacy_async_copy_byte_width_check_survives_ownership_move():
    call = _call("tl.ptx_cp_async_bytes", [_pointer("dst"), _pointer("src"), 12], ty="void")
    with pytest.raises(ValueError, match="constant byte width"):
        call.validate()


@pytest.mark.parametrize("name", ["tl.simd.vxor", "tl.simd.vadds", "tl.simd.vabsdif"])
def test_simd_value_and_explicit_merging_destination_keep_their_types(name):
    source = ir.Var("source", "int32x64")
    rhs = ir.const(1, "int32") if name == "tl.simd.vadds" else source
    mask = ir.Var("mask", "boolx256")
    operands = [source, rhs, mask, "MODE_MERGING"]
    value = _call(name, operands)
    assert value.ty == source.ty
    value.validate()
    destination = _pointer("destination", "int32x64", "local")
    _annotated(name, [destination] + operands, "void")
    malformed = _call(name, [destination] + operands[:-1] + ["MODE_ZEROING"], ty="void")
    with pytest.raises(ValueError, match="MODE_MERGING"):
        malformed.validate()


def test_simd_widening_and_pointer_state_results_are_distinct_contracts():
    source = ir.Var("source", "int16x128")
    mask = ir.Var("mask", "boolx256")
    reduced = _call("tl.simd.vcadd", [source, mask, "MODE_ZEROING"])
    reduced.validate()
    assert reduced.ty == ir.PrimType("int32x64")
    pointer = _pointer("state", "int16", "shared")
    advanced = _call("tl.simd.vsstb", [source, pointer, 1, mask, "POST_UPDATE"])
    advanced.validate()
    assert advanced.ty == pointer.ty
    _annotated("tl.simd.vsstb", [source, pointer, 1, mask, "POST_UPDATE"], pointer.ty)
    wrong = _call("tl.simd.vsstb", [source, pointer, 1, mask, "POST_UPDATE"], ty=ir.PtrType("void"))
    with pytest.raises(TypeError, match="result type disagrees"):
        wrong.validate()
    pair = ir.Var("pair", "int16x128")
    component = _call("tl.simd.pair_get", [pair, 1], ty_args=[pointer.ty])
    component.validate()
    assert component.ty == pointer.ty


def test_frontend_call_constructor_uses_native_type_arguments_and_payloads():
    value = ir.Var("value", "float32")
    tag = ir.Var("tag", "int32")
    call = T.call_intrin("float32", "tl.fmul", value, value, annotations={"tag": tag})
    call.validate()
    assert call.ty == value.ty
    assert call.args[-1].fields[0].same_as(tag)
