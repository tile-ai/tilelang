"""Compile matrix over dtype x op x vector width for vectorized global
memory patterns.

Every cell used to be an independent hand-written case in the codegen
(make_<type> constructors, operator= from ulonglong4, pack helpers), so a
new dtype or a wider vector width silently created nvcc-only failures:
fp4 x 64 lanes, int4 x 64 lanes, and fp16 x 16 lanes all broke when 256-bit
vectorization was enabled on sm_100+. This matrix compiles every
combination so the next gap fails in CI instead of on a user's machine.

On targets without 256-bit vectorization (< sm_100) the "256" width params
compile at 128 bits, which keeps the test portable and still covers the
128-bit half of the matrix.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T

MATRIX_DTYPES = [
    "float4_e2m1fn",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0fnu",
    "float16",
    "bfloat16",
    "int8",
    "uint8",
    "int4",
    "uint4",
    "int32",
    "float32",
]

# copy: pure vector load/store, no broadcast.
# pred_copy: loop-invariant predicate materializes a typed vector temporary
#   (`condval = tl::load_global_...` assignment) and broadcasts the zero
#   else-value -- the shape that exposed the fp4/fp16/int4 gaps.
# fill_one: broadcast of a non-zero constant (exercises packed-nibble
#   constant folding for 4-bit dtypes).
MATRIX_OPS = ["copy", "pred_copy", "fill_one"]


def _matrix_kernel(dtype, op, disable_256):
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: disable_256})
    def factory():
        @T.prim_func
        def kern(
            src: T.Tensor((128, 128), dtype),
            predicate: T.Tensor((1,), T.int32),
            dst: T.Tensor((128, 128), dtype),
        ):
            with T.Kernel(1, threads=256):
                for row, col in T.Parallel(128, 128):
                    if op == "copy":
                        dst[row, col] = src[row, col]
                    elif op == "pred_copy":
                        dst[row, col] = src[row, col] if predicate[0] != 0 else 0
                    else:
                        dst[row, col] = 1

        return kern

    return factory()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
@pytest.mark.parametrize("op", MATRIX_OPS)
@pytest.mark.parametrize("width", ["128", "256"])
def test_vectorize_matrix_compiles(dtype, op, width):
    _matrix_kernel(dtype, op, disable_256=(width == "128"))


def _pred_copy_kernel(dtype):
    @tilelang.jit
    def factory():
        @T.prim_func
        def kern(
            src: T.Tensor((128, 128), dtype),
            predicate: T.Tensor((1,), T.int32),
            dst: T.Tensor((128, 128), dtype),
        ):
            with T.Kernel(1, threads=256):
                for row, col in T.Parallel(128, 128):
                    dst[row, col] = src[row, col] if predicate[0] != 0 else 0

        return kern

    return factory()


def _check_pred_copy_packed(kernel):
    """Run a predicated copy on byte-backed packed storage and check both
    predicate branches. src/dst are uint8 views so equality is bitwise."""
    src = torch.randint(0, 256, (128, 64), dtype=torch.uint8, device="cuda")
    dst = torch.full((128, 64), 0xEE, dtype=torch.uint8, device="cuda")
    packed = torch.float4_e2m1fn_x2
    pred1 = torch.tensor([1], dtype=torch.int32, device="cuda")
    pred0 = torch.tensor([0], dtype=torch.int32, device="cuda")

    kernel(src.view(packed), pred1, dst.view(packed))
    torch.cuda.synchronize()
    assert torch.equal(dst, src)

    kernel(src.view(packed), pred0, dst.view(packed))
    torch.cuda.synchronize()
    assert (dst == 0).all()


@tilelang.testing.requires_cuda
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_pred_copy_fp4_numeric():
    """fp4 predicated copy: on sm_100+ this vectorizes to 64 lanes (256 bits)
    and used to fail to compile (undefined make_fp4_e2_64_t, missing
    ulonglong4 assignment)."""
    _check_pred_copy_packed(_pred_copy_kernel("float4_e2m1fn"))


@tilelang.testing.requires_cuda
def test_pred_copy_int4_numeric():
    """int4 predicated copy: on sm_100+ this vectorizes to 64 lanes and used
    to emit the nonexistent CUDA type `int8` plus a 64-arg make_int8 call."""
    kernel = _pred_copy_kernel("int4")
    src = torch.randint(0, 256, (128, 64), dtype=torch.uint8, device="cuda")
    dst = torch.full((128, 64), 0xEE, dtype=torch.uint8, device="cuda")
    pred1 = torch.tensor([1], dtype=torch.int32, device="cuda")
    pred0 = torch.tensor([0], dtype=torch.int32, device="cuda")

    kernel(src, pred1, dst)
    torch.cuda.synchronize()
    assert torch.equal(dst, src)

    kernel(src, pred0, dst)
    torch.cuda.synchronize()
    assert (dst == 0).all()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("fill_value", [1, 5])
def test_fill_int4_nonzero_numeric(fill_value):
    """int4 constant fill: checks the packed-nibble constant path (every
    byte of the destination must hold the value in both nibbles)."""

    @tilelang.jit
    def factory():
        @T.prim_func
        def kern(dst: T.Tensor((128, 128), "int4")):
            with T.Kernel(1, threads=256):
                for row, col in T.Parallel(128, 128):
                    dst[row, col] = fill_value

        return kern

    kernel = factory()
    dst = torch.full((128, 64), 0xEE, dtype=torch.uint8, device="cuda")
    kernel(dst)
    torch.cuda.synchronize()
    expected = (fill_value & 0xF) | ((fill_value & 0xF) << 4)
    assert (dst == expected).all()


def _fill_var_kernel(dtype):
    @tilelang.jit
    def factory():
        @T.prim_func
        def kern(scalar: T.Tensor((2,), dtype), dst: T.Tensor((128, 128), dtype)):
            with T.Kernel(1, threads=256):
                for row, col in T.Parallel(128, 128):
                    dst[row, col] = scalar[0]

        return kern

    return factory()


@tilelang.testing.requires_cuda
def test_fill_scalar_var_fp8_numeric():
    """fp8 fill from a runtime scalar: a non-constant broadcast replicated
    across the 256-bit carrier (tl::broadcast byte replication)."""
    kernel = _fill_var_kernel("float8_e4m3fn")
    scalar = torch.full((2,), 1.5, device="cuda").to(torch.float8_e4m3fn)
    dst = torch.zeros((128, 128), device="cuda").to(torch.float8_e4m3fn)
    kernel(scalar, dst)
    torch.cuda.synchronize()
    assert (dst.view(torch.uint8) == scalar.view(torch.uint8)[0]).all()


@tilelang.testing.requires_cuda
def test_fill_scalar_var_int8_numeric():
    """int8 fill from a runtime scalar (non-constant byte broadcast)."""
    kernel = _fill_var_kernel("int8")
    scalar = torch.full((2,), -42, dtype=torch.int8, device="cuda")
    dst = torch.zeros((128, 128), dtype=torch.int8, device="cuda")
    kernel(scalar, dst)
    torch.cuda.synchronize()
    assert (dst == -42).all()


@tilelang.testing.requires_cuda
@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_fill_scalar_var_fp4_numeric():
    """fp4 fill from a runtime scalar: packed nibble read, then a
    non-constant nibble-pair broadcast."""
    kernel = _fill_var_kernel("float4_e2m1fn")
    # Both nibbles hold 0x4 (fp4 e2m1 value 2.0), so every destination byte
    # must equal 0x44.
    scalar = torch.full((1,), 0x44, dtype=torch.uint8, device="cuda")
    dst = torch.zeros((128, 64), dtype=torch.uint8, device="cuda")
    packed = torch.float4_e2m1fn_x2
    kernel(scalar.view(packed), dst.view(packed))
    torch.cuda.synchronize()
    assert (dst == 0x44).all()


@tilelang.testing.requires_cuda
def test_fill_fp16_numeric():
    """fp16 constant fill: at 16 lanes (256 bits) the broadcast goes through
    tl::pack_float16x4, which used to reject cutlass::half_t arguments."""

    @tilelang.jit
    def factory():
        @T.prim_func
        def kern(dst: T.Tensor((128, 128), "float16")):
            with T.Kernel(1, threads=256):
                for row, col in T.Parallel(128, 128):
                    dst[row, col] = 1.5

        return kern

    kernel = factory()
    dst = torch.zeros((128, 128), dtype=torch.float16, device="cuda")
    kernel(dst)
    torch.cuda.synchronize()
    assert (dst == 1.5).all()


if __name__ == "__main__":
    tilelang.testing.main()
