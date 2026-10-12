import pytest

from examples.sunmmio.elementwise.mixed_precision_mul import mixed_precision_mul
from testing.python.sunmmio.common.codegen_validation import validate_sunmmio_codegen_with_npuir_opt


@pytest.mark.parametrize(
    "a_dtype,b_dtype,out_dtype",
    [
        ("bfloat16", "bfloat16", "float32"),
        ("float32", "float32", "bfloat16"),
        ("bfloat16", "float32", "bfloat16"),
        ("bfloat16", "float32", "float32"),
    ],
)
def test_mixed_precision_tiles_preserve_multiply_and_cast_dtypes(tmp_path, a_dtype, b_dtype, out_dtype):
    kernel = mixed_precision_mul.get_tir(a_dtype=a_dtype, b_dtype=b_dtype, out_dtype=out_dtype)
    src = validate_sunmmio_codegen_with_npuir_opt(
        kernel,
        tmp_path,
        mlir_filename="mixed_precision_mul.mlir",
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    product_dtype = "bf16" if a_dtype == b_dtype == "bfloat16" else "f32"
    output_dtype = "bf16" if out_dtype == "bfloat16" else "f32"
    (multiply,) = [line for line in src.splitlines() if "suvm.tile.mulf" in line]
    assert multiply.strip().endswith(f"-> !suvm.tile<32x32x{product_dtype}>")
    (store,) = [line for line in src.splitlines() if "suvm.tile.store" in line]
    assert f"!suvm.tile<32x32x{output_dtype}>" in store
