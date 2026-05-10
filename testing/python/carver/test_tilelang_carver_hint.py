from tilelang.carver.roller.hint import IntrinInfo


def test_intrin_info_repr():
    info = IntrinInfo(
        in_dtype="float16",
        out_dtype="float32",
        trans_b=True,
        input_transform_kind=1,
        weight_transform_kind=2,
    )

    text = repr(info)
    assert "float16" in text
    assert "float32" in text
    assert "input_transform_kind=1" in text
    assert "weight_transform_kind=2" in text
