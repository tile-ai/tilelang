from __future__ import annotations

import pytest

import tilelang.language as T
from tilelang.language.fp8_op import _normalize_block_scale_layout


def test_blockscaled_layout_canonical_metadata() -> None:
    layout = T.BlockScaledLayout.e8m0_k32()
    assert layout.scale_dtype == "e8m0"
    assert layout.scale_format == "e8m0_block_k32"
    assert layout.scale_axis == "contracted_k"
    assert layout.block_size == 32
    assert layout.layout == "logical_unswizzled_k_axis_blocks"
    assert layout.a_scale_shape(64) == (2,)
    assert layout.b_scale_shape(16, 64) == (16, 2)
    assert layout.broadcast_b_scale_shape(64) == (2,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scale_dtype": "fp32"}, "scale_dtype='e8m0'"),
        ({"axis": "row"}, "axis='contracted_k'"),
        ({"block_size": 16}, "block_size=32"),
        ({"layout": "swizzled"}, "logical_unswizzled_k_axis_blocks"),
    ],
)
def test_blockscaled_layout_rejects_non_e8m0_k32(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        T.BlockScaledLayout(**kwargs)


def test_blockscaled_layout_rejects_non_k32_extent() -> None:
    layout = T.BlockScaledLayout.e8m0_k32()
    with pytest.raises(ValueError, match="K divisible by 32"):
        layout.scale_blocks(48)


def test_blockscaled_layout_validates_qk_scale_shapes() -> None:
    layout = T.BlockScaledLayout.e8m0_k32()
    layout.validate_scale_shapes(
        k_extent=64,
        a_scale_shape=(2,),
        b_scale_shape=(16, 2),
        n_extent=16,
    )
    layout.validate_scale_shapes(
        k_extent=64,
        a_scale_shape=(2,),
        b_scale_shape=(2,),
        n_extent=16,
    )
    with pytest.raises(ValueError, match=r"A_scale.*K / 32"):
        layout.validate_scale_shapes(
            k_extent=64,
            a_scale_shape=(1,),
            b_scale_shape=(16, 2),
            n_extent=16,
        )
    with pytest.raises(ValueError, match=r"B_scale.*N, K / 32"):
        layout.validate_scale_shapes(
            k_extent=64,
            a_scale_shape=(2,),
            b_scale_shape=(16, 1),
            n_extent=16,
        )


def test_fp8_scaled_matmul_accepts_e8m0_layout_lowering_contract() -> None:
    layout = T.BlockScaledLayout.e8m0_k32()

    @T.prim_func
    def blockscaled_qk(
        A_fp8: T.Tensor((1, 64), "float8_e4m3"),
        A_scale: T.Tensor((2,), "uint8"),
        B_fp8: T.Tensor((16, 64), "float8_e4m3"),
        B_scale: T.Tensor((16, 2), "uint8"),
        C: T.Tensor((1, 16), "float32"),
    ):
        with T.Kernel(1, threads=128):
            T.fp8_scaled_matmul(
                A_fp8,
                A_scale,
                B_fp8,
                B_scale,
                C,
                transpose_B=True,
                block_scale_layout=layout,
            )

    src = str(blockscaled_qk)
    assert "blockscaled_qk" in src
    assert layout.scale_format == "e8m0_block_k32"
    assert layout.scale_index(63) == 1


def test_fp8_scaled_matmul_accepts_legacy_metadata_spelling() -> None:
    @T.prim_func
    def blockscaled_qk_legacy_attrs(
        A_fp8: T.Tensor((1, 64), "float8_e4m3"),
        A_scale: T.Tensor((2,), "uint8"),
        B_fp8: T.Tensor((16, 64), "float8_e4m3"),
        B_scale: T.Tensor((2,), "uint8"),
        C: T.Tensor((1, 16), "float32"),
    ):
        with T.Kernel(1, threads=128):
            T.fp8_scaled_matmul(
                A_fp8,
                A_scale,
                B_fp8,
                B_scale,
                C,
                transpose_B=True,
                scale_format="e8m0_block_k32",
                scale_block_size=32,
            )

    assert "blockscaled_qk_legacy_attrs" in str(blockscaled_qk_legacy_attrs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"scale_format": "e8m0_block_k32", "scale_block_size": None},
            "scale_block_size=32",
        ),
        (
            {"scale_format": None, "scale_block_size": 32},
            "scale_format='e8m0_block_k32'",
        ),
        (
            {"scale_format": "other_block_k32", "scale_block_size": 32},
            "scale_format='e8m0_block_k32'",
        ),
        (
            {"scale_format": "e8m0_block_k32", "scale_block_size": 16},
            "scale_block_size=32",
        ),
    ],
)
def test_fp8_scaled_matmul_rejects_partial_or_inconsistent_e8m0_metadata(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _normalize_block_scale_layout(None, **kwargs)


def test_e8m0_decode_semantics_document_sentinels() -> None:
    src = T.e8m0_to_float
    text = getattr(src, "__doc__", "") or ""
    assert "byte == 0" in text
    assert "byte == 0xFF" in text
    assert "byte - 127" in text
