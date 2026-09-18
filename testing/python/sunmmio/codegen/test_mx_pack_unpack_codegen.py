import os

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing

from tilelang.layout import (
    get_mx_scale_shape,
    make_aligned_row_major,
    make_mx_row_major_layout,
    make_mxznz_layout,
    make_mxzz_layout,
    make_row_major,
)

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    lower_sunmmio_kernel_to_device_tir,
    validate_sunmmio_codegen_with_npuir_opt,
)


tilelang.env.disable_cache()
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
os.environ.setdefault("SUNMMIO_TEST_LOG_IR", "1")

MX_DTYPE_CASES = (
    pytest.param(T.mxfp8, T.float8_e4m3fn, "!suvm.mxfp8", id="mxfp8"),
    pytest.param(T.mxfp4, T.float4_e2m1fn, "!suvm.mxfp4", id="mxfp4"),
)

LAYOUT_KINDS = (
    pytest.param(
        "row_major",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "mx_row_major scale alias currently exposes logical width N/32 "
                "without a padded 64B scale extent; row-major T.mx_pack/unpack "
                "scale copy is deferred until suvm.unpack exposes that layout"
            ),
        ),
        id="row-major",
    ),
    pytest.param("mxzz", id="mxzz"),
    pytest.param("mxznz", id="mxznz"),
)

PACK_UNPACK_CASES = (
    pytest.param("pack", id="pack"),
    pytest.param("unpack", id="unpack"),
)

LOOSE_OPT_ARGS = ("--verify-each",)


def _int_shape(shape):
    return tuple(int(x) for x in shape)


def _mx_layout(kind, shape, mx_dtype):
    if kind == "row_major":
        return make_mx_row_major_layout(shape, dtype=mx_dtype)
    if kind == "mxzz":
        return make_mxzz_layout(shape, dtype=mx_dtype)
    if kind == "mxznz":
        return make_mxznz_layout(shape, dtype=mx_dtype)
    raise ValueError(f"unsupported MX layout kind: {kind}")


def _shape_for_layout(kind):
    if kind == "row_major":
        return (32, 1024)
    return (64, 128)


def _mx_scale_shape(kind, shape, mx_dtype):
    return _int_shape(get_mx_scale_shape(_mx_layout(kind, shape, mx_dtype), mx_dtype))


def _mx_token_name(mx_token):
    return mx_token.replace("!suvm.", "")


def _assert_pack_unpack_mlir(src, mx_token, expect_scale_slice=True):
    assert "suvm.copy_async" not in src
    assert_source_contains(
        src,
        (
            mx_token,
            "suvm.unpack",
            "suvm.get_partitioned_tile_view",
            "suvm.tile.load",
            "suvm.tile.store",
        ),
    )
    if expect_scale_slice:
        assert_source_contains(src, ("suvm.tile.extract_slice", "suvm.tile.insert_slice"))


# ---------------------------------------------------------------------------
# Basic physical pack/unpack kernels
# ---------------------------------------------------------------------------


@target("Sunmmio")
def mx_physical_pack_kernel(mx_dtype, data_dtype, layout_kind):
    shape = _shape_for_layout(layout_kind)
    mx_layout = _mx_layout(layout_kind, shape, mx_dtype)
    scale_shape = _mx_scale_shape(layout_kind, shape, mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            T.annotate_layout({mx: mx_layout})
            T.mx_pack(data, scale, mx)

    return main


@target("Sunmmio")
def mx_physical_unpack_kernel(mx_dtype, data_dtype, layout_kind):
    shape = _shape_for_layout(layout_kind)
    mx_layout = _mx_layout(layout_kind, shape, mx_dtype)
    scale_shape = _mx_scale_shape(layout_kind, shape, mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            T.annotate_layout({mx: mx_layout})
            T.mx_unpack(mx, data, scale)

    return main


@pytest.mark.parametrize("op_name", PACK_UNPACK_CASES)
@pytest.mark.parametrize("layout_kind", LAYOUT_KINDS)
@pytest.mark.parametrize("mx_dtype,data_dtype,mx_token", MX_DTYPE_CASES)
def test_mx_pack_unpack_lowering_codegen_and_supported_device_validation(
    tmp_path,
    op_name,
    layout_kind,
    mx_dtype,
    data_dtype,
    mx_token,
):
    if op_name == "pack":
        kernel = mx_physical_pack_kernel(mx_dtype, data_dtype, layout_kind)
    else:
        kernel = mx_physical_unpack_kernel(mx_dtype, data_dtype, layout_kind)

    mlir_filename = f"mx_{op_name}_{layout_kind}_{_mx_token_name(mx_token)}_suvm.mlir"
    expected_tokens = ("suvm.unpack", "suvm.tile.load", "suvm.tile.store")
    opt_args = LOOSE_OPT_ARGS if layout_kind == "row_major" else ("-suvm-device-validate",)
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename=mlir_filename,
        expected_tokens=expected_tokens,
        opt_args=opt_args,
    )
    _assert_pack_unpack_mlir(src, mx_token, expect_scale_slice=layout_kind != "row_major")


# ---------------------------------------------------------------------------
# Pack result feeding GEMM
# ---------------------------------------------------------------------------


@target("Sunmmio")
def mx_pack_to_gemm_kernel(mx_dtype, data_dtype, operand):
    if operand == "a":
        shape = (32, 128)
        other_shape = (128, 32)
    else:
        shape = (128, 32)
        other_shape = (32, 128)
    c_shape = (32, 32)
    c_dtype = T.bfloat16

    if operand == "a":
        mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
        other_layout = make_mxznz_layout(other_shape, dtype=mx_dtype)
    else:
        mx_layout = make_mxznz_layout(shape, dtype=mx_dtype)
        other_layout = make_mxzz_layout(other_shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            mx_operand_scope = "shared.asram" if operand == "a" else "shared.wsram"
            mx_operand = T.alloc_shared(shape, mx_dtype, scope=mx_operand_scope)
            other_scope = "shared.wsram" if operand == "a" else "shared.asram"
            other_src = T.alloc_shared(other_shape, mx_dtype, scope="shared.rsram")
            other = T.alloc_shared(other_shape, mx_dtype, scope=other_scope)
            out = T.alloc_shared(c_shape, c_dtype)
            T.annotate_layout({mx: mx_layout, other_src: other_layout})
            T.mx_pack(data, scale, mx)
            T.copy(mx, mx_operand)
            T.copy(other_src, other)
            if operand == "a":
                T.gemm(mx_operand, other, out)
            else:
                T.gemm(other, mx_operand, out)

    return main


@target("Sunmmio")
def mx_pack_to_unaligned_gemm_kernel(mx_dtype, data_dtype, operand):
    shape = (32, 32)
    c_shape = (32, 32)
    c_dtype = T.bfloat16

    if operand == "a":
        mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
        other_layout = make_mxznz_layout(shape, dtype=mx_dtype)
    else:
        mx_layout = make_mxznz_layout(shape, dtype=mx_dtype)
        other_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            mx_operand_scope = "shared.asram" if operand == "a" else "shared.wsram"
            mx_operand = T.alloc_shared(shape, mx_dtype, scope=mx_operand_scope)
            other_scope = "shared.wsram" if operand == "a" else "shared.asram"
            other_src = T.alloc_shared(shape, mx_dtype, scope="shared.rsram")
            other = T.alloc_shared(shape, mx_dtype, scope=other_scope)
            out = T.alloc_shared(c_shape, c_dtype)
            T.annotate_layout({mx: mx_layout, other_src: other_layout})
            T.mx_pack(data, scale, mx)
            T.copy(mx, mx_operand)
            T.copy(other_src, other)
            if operand == "a":
                T.gemm(mx_operand, other, out)
            else:
                T.gemm(other, mx_operand, out)

    return main


@pytest.mark.parametrize("mx_dtype,data_dtype,mx_token", MX_DTYPE_CASES)
@pytest.mark.parametrize("operand", [pytest.param("a", id="a-pack"), pytest.param("b", id="b-pack")])
def test_mx_pack_result_can_feed_gemm_codegen(tmp_path, mx_dtype, data_dtype, mx_token, operand):
    src = validate_sunmmio_codegen_with_npuir_opt(
        mx_pack_to_gemm_kernel(mx_dtype, data_dtype, operand),
        tmp_path,
        mlir_filename=(f"mx_pack_to_gemm_{operand}_{_mx_token_name(mx_token)}_suvm.mlir"),
        expected_tokens=(mx_token, "suvm.unpack", "suvm.tc.mma"),
    )

    assert_source_contains(src, ("suvm.unpack", "suvm.copy_async", "suvm.tc.mma"))
    lines = src.splitlines()
    first_copy = next(i for i, line in enumerate(lines) if "suvm.copy_async" in line)
    pack_stores = [i for i, line in enumerate(lines[:first_copy]) if "suvm.tile.store" in line]
    pack_sync = next(i for i, line in enumerate(lines[:first_copy]) if "suvm.sync" in line and "vector" in line and "rsram" in line)
    assert pack_stores
    assert max(pack_stores) < pack_sync < first_copy


@pytest.mark.parametrize("mx_dtype,data_dtype,mx_token", MX_DTYPE_CASES)
@pytest.mark.parametrize("operand", [pytest.param("a", id="a-pack"), pytest.param("b", id="b-pack")])
def test_mx_pack_result_rejects_unaligned_gemm_operand(mx_dtype, data_dtype, mx_token, operand):
    with pytest.raises(Exception, match="Explicit MX operand padding is not implemented|requires K/M extent"):
        lower_sunmmio_kernel_to_device_tir(mx_pack_to_unaligned_gemm_kernel(mx_dtype, data_dtype, operand))


# ---------------------------------------------------------------------------
# User data/scale roundtrip
# ---------------------------------------------------------------------------


@target("Sunmmio")
def mx_manual_payload_roundtrip_kernel(mx_dtype, data_dtype):
    shape = (32, 1024)
    mx_layout = make_mx_row_major_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, data_dtype)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            data_after = T.alloc_shared(shape, data_dtype)
            scale_after = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            data_layout = make_row_major(shape)
            scale_layout = make_aligned_row_major(scale_shape, T.float8_e8m0fnu, 64)
            T.annotate_layout(
                {
                    mx: mx_layout,
                    data: data_layout,
                    data_after: data_layout,
                    scale: scale_layout,
                    scale_after: scale_layout,
                }
            )

            T.mx_pack(data, scale, mx)
            T.mx_unpack(mx, data_after, scale_after)

    return main


@pytest.mark.parametrize("mx_dtype,data_dtype,mx_token", MX_DTYPE_CASES)
@pytest.mark.xfail(
    strict=True,
    reason=(
        "mx_row_major scale alias currently exposes logical width N/32 without "
        "a padded 64B scale extent; row-major T.mx_pack/unpack scale copy is "
        "deferred until suvm.unpack exposes that layout"
    ),
)
def test_user_kernel_can_write_and_read_data_scale_around_mx_pack_unpack_codegen(tmp_path, mx_dtype, data_dtype, mx_token):
    kernel = mx_manual_payload_roundtrip_kernel(mx_dtype, data_dtype)
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        expected_tokens=(mx_token, "suvm.unpack", "suvm.tile.load", "suvm.tile.store"),
        mlir_filename=f"mx_payload_roundtrip_{_mx_token_name(mx_token)}_suvm.mlir",
        opt_args=LOOSE_OPT_ARGS,
    )

    assert "suvm.copy_async" not in src
    assert_source_contains(src, ("suvm.tile.load", "suvm.tile.store", "suvm.unpack"))


# ---------------------------------------------------------------------------
# Invalid contracts
# ---------------------------------------------------------------------------


@target("Sunmmio")
def mx_pack_bad_data_dtype_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, T.bfloat16)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            T.annotate_layout({mx: mx_layout})
            T.mx_pack(data, scale, mx)

    return main


@target("Sunmmio")
def mx_pack_bad_scale_shape_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, T.float8_e4m3fn)
            scale = T.alloc_shared((1, 32), T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            T.annotate_layout({mx: mx_layout})
            T.mx_pack(data, scale, mx)

    return main


@target("Sunmmio")
def mx_pack_bad_scope_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_local(shape, T.float8_e4m3fn)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            T.annotate_layout({mx: mx_layout})
            T.mx_pack(data, scale, mx)

    return main


@target("Sunmmio")
def mx_pack_bad_layout_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8

    @T.prim_func
    def main():
        with T.Kernel():
            data = T.alloc_shared(shape, T.float8_e4m3fn)
            scale = T.alloc_shared((8, 32), T.float8_e8m0fnu)
            mx = T.alloc_shared(shape, mx_dtype)
            T.annotate_layout({mx: make_row_major(shape)})
            T.mx_pack(data, scale, mx)

    return main


@target("Sunmmio")
def mx_unpack_bad_data_dtype_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_shared(shape, T.bfloat16)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            T.annotate_layout({mx: mx_layout})
            T.mx_unpack(mx, data, scale)

    return main


@target("Sunmmio")
def mx_unpack_bad_scale_shape_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_shared(shape, T.float8_e4m3fn)
            scale = T.alloc_shared((1, 32), T.float8_e8m0fnu)
            T.annotate_layout({mx: mx_layout})
            T.mx_unpack(mx, data, scale)

    return main


@target("Sunmmio")
def mx_unpack_bad_scope_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8
    mx_layout = make_mxzz_layout(shape, dtype=mx_dtype)
    scale_shape = _int_shape(get_mx_scale_shape(mx_layout, mx_dtype))

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_local(shape, T.float8_e4m3fn)
            scale = T.alloc_shared(scale_shape, T.float8_e8m0fnu)
            T.annotate_layout({mx: mx_layout})
            T.mx_unpack(mx, data, scale)

    return main


@target("Sunmmio")
def mx_unpack_bad_layout_kernel():
    shape = (64, 128)
    mx_dtype = T.mxfp8

    @T.prim_func
    def main():
        with T.Kernel():
            mx = T.alloc_shared(shape, mx_dtype)
            data = T.alloc_shared(shape, T.float8_e4m3fn)
            scale = T.alloc_shared((8, 32), T.float8_e8m0fnu)
            T.annotate_layout({mx: make_row_major(shape)})
            T.mx_unpack(mx, data, scale)

    return main


@pytest.mark.parametrize(
    "kernel,match",
    [
        (mx_pack_bad_data_dtype_kernel, "data dtype must be"),
        (mx_pack_bad_scale_shape_kernel, "scale.shape"),
        (mx_pack_bad_scope_kernel, "shared.rsram"),
        (mx_pack_bad_layout_kernel, "support only MX row-major, MXZZ, and MXZNZ"),
    ],
)
def test_mx_pack_rejects_invalid_contracts(kernel, match):
    with pytest.raises(Exception, match=match):
        lower_sunmmio_kernel_to_device_tir(kernel())


@pytest.mark.parametrize(
    "kernel,match",
    [
        (mx_unpack_bad_data_dtype_kernel, "data dtype must be"),
        (mx_unpack_bad_scale_shape_kernel, "scale.shape"),
        (mx_unpack_bad_scope_kernel, "shared.rsram"),
        (mx_unpack_bad_layout_kernel, "support only MX row-major, MXZZ, and MXZNZ"),
    ],
)
def test_mx_unpack_rejects_invalid_contracts(kernel, match):
    with pytest.raises(Exception, match=match):
        lower_sunmmio_kernel_to_device_tir(kernel())


if __name__ == "__main__":
    tilelang.testing.main()
