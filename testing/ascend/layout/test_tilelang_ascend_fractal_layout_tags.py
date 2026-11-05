import pytest
import tvm_ffi

import tilelang
import tilelang.testing
from tvm_tirx import ir, tirx
from tilelang.layout import (
    Layout,
    make_ascend_l0c_layout,
    make_ascend_major_k_layout,
    make_ascend_major_mn_layout,
    make_ascend_sf_layout,
    try_extract_fractal_layout,
)


@pytest.mark.parametrize(
    "factory,shape,dtype,expected",
    [
        (make_ascend_major_k_layout, (16, 32), "float8_e4m3fn", (0, 1, 0)),
        (make_ascend_major_mn_layout, (32, 16), "float8_e4m3fn", (3, 0, 1)),
        (make_ascend_l0c_layout, (16, 16), "float32", (1, 1, 0)),
        (make_ascend_sf_layout, (16, 1), "uint16", (2, 1, 0)),
        (make_ascend_sf_layout, (32, 1), "uint16", (2, 1, 0)),
        # The same small shape has different semantics despite identical indices.
        (make_ascend_major_k_layout, (16, 1), "uint16", (0, 1, 0)),
    ],
)
def test_canonical_layout_tags(factory, shape, dtype, expected):
    buf = ir.Var("A", tirx.TensorType(shape, dtype, scope="shared"))
    info = try_extract_fractal_layout(factory(buf), buf)

    assert (info.kind, info.c0_axis, info.row16_axis) == expected


def test_expand_preserves_sf_tag():
    base_buf = ir.Var("SF2", tirx.TensorType((16, 1), "uint16", scope="shared"))
    expanded_buf = ir.Var("SF3", tirx.TensorType((2, 16, 1), "uint16", scope="shared"))

    layout = make_ascend_sf_layout(base_buf).expand([2])
    info = try_extract_fractal_layout(layout, expanded_buf)

    assert info.kind == 2
    assert list(layout.get_input_shape()) == [2, 16, 1]


@pytest.mark.parametrize(
    "factory,dtype,kind",
    [
        (make_ascend_major_k_layout, "float16", 0),
        (make_ascend_l0c_layout, "float32", 1),
        (make_ascend_sf_layout, "uint8", 2),
        (make_ascend_major_mn_layout, "float16", 3),
    ],
)
@pytest.mark.parametrize("python_encoder", [False, True])
def test_annotation_json_and_renew_def_preserve_fractal_kind_and_captures(factory, dtype, kind, python_encoder):
    from tilelang.language._native import encode_annotations

    n = ir.Var("planes", "int32")
    buf = ir.Var("A", tirx.TensorType((n, 16, 32), dtype, scope="shared.l1"))
    layout = factory(buf)
    direct = ir.load_json(ir.save_json(layout))
    assert try_extract_fractal_layout(direct, buf).kind == kind
    encoder = encode_annotations if python_encoder else tvm_ffi.get_global_func("tl.EncodeAnnotations")
    schema, payload = encoder({"layout_map": {buf: layout}})
    note = ir.Evaluate(ir.Call("tl.annotation", [payload], attrs={"schema_version": 1, "annotation_schema": schema}))
    value = tirx.Function([n, buf], [note], ir.PrimType("void"))
    assert tirx.analysis.check_well_formed(value)
    restored = tirx.renew_def(ir.load_json(ir.save_json(value)))
    assert tirx.analysis.check_well_formed(restored)
    assert not tirx.analysis.undefined_vars(restored)
    assert not restored.params[0].same_as(n)
    call = restored.body.seq[0].value
    decoded = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[0], call.attrs["annotation_schema"])
    new_buf = restored.params[1]
    new_layout = decoded["layout_map"][new_buf]
    assert try_extract_fractal_layout(new_layout, new_buf).kind == kind
    assert new_layout.get_input_shape()[0].same_as(restored.params[0])
    ir.assert_structural_equal(value, restored)


@pytest.mark.parametrize("bad_kind", [ir.IntImm("int32", -1), ir.IntImm("int32", 4), ir.IntImm("int64", 1), ir.FloatImm("float32", 1)])
def test_fractal_payload_rejects_invalid_kind_fields(bad_kind):
    buf = ir.Var("A", tirx.TensorType((16, 32), "float16"))
    payload = tvm_ffi.get_global_func("tl.EncodeLayoutPayload")(make_ascend_major_k_layout(buf))
    invalid = ir.Tuple([payload.fields[0], bad_kind])
    with pytest.raises((TypeError, ValueError), match="fractal kind"):
        tvm_ffi.get_global_func("tl.DecodeLayoutPayload")(invalid, "ascend.fractal", 1)


def test_fractal_payload_rejects_missing_fields_wrong_rank_and_unknown_codec():
    encode = tvm_ffi.get_global_func("tl.EncodeLayoutPayload")
    decode = tvm_ffi.get_global_func("tl.DecodeLayoutPayload")
    plain = encode(Layout((16, 32), lambda i, j: [i, j]))
    with pytest.raises(ValueError, match="coordinates and kind"):
        decode(ir.Tuple([plain]), "ascend.fractal", 1)
    with pytest.raises(ValueError, match="four matrix coordinates"):
        decode(ir.Tuple([plain, ir.IntImm("int32", 0)]), "ascend.fractal", 1)
    with pytest.raises(ValueError, match="Unknown layout payload kind"):
        decode(plain, "unregistered.backend", 1)
    with pytest.raises(ValueError, match="version"):
        decode(plain, "ascend.fractal", 2)


if __name__ == "__main__":
    tilelang.testing.main()
