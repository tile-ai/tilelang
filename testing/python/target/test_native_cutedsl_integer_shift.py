"""CuTe native shifts retain TileLang's existing widening-cast contract."""

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_shift(source_dtype, output_dtype, direction, amount):
    @T.prim_func
    def main(A: T.Tensor((32,), source_dtype), B: T.Tensor((32,), output_dtype)):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding()
            local = T.alloc_local((1,), source_dtype)
            local[0] = A[i]
            if direction == "left":
                B[i] = T.cast(local[0] << amount, output_dtype)
            else:
                B[i] = T.cast(local[0] >> amount, output_dtype)

    return main


@pytest.mark.parametrize(
    "source_dtype,output_dtype,direction,amount,expected,forbidden",
    [
        ("uint16", "uint32", "left", 20, "cutlass.Uint32(local[0]) <<", "cutlass.Uint32(cutlass.Uint16(("),
        ("int8", "int16", "right", 1, "cutlass.Int16(local[0]) >>", "cutlass.Int16(cutlass.Int8(("),
        ("uint8", "uint16", "right", 1, "cutlass.Uint16(local[0]) >>", "cutlass.Uint16(cutlass.Uint8(("),
        ("int8", "uint16", "right", 1, "tl.cast_tensor((local[0] >> cutlass.Int8(1)), cutlass.Int8)", "cutlass.Uint16(local[0]) >>"),
    ],
)
def test_native_shift_dispatch_preserves_promotion_and_signedness(source_dtype, output_dtype, direction, amount, expected, forbidden):
    artifact = tilelang.lower(
        _make_shift(source_dtype, output_dtype, direction, amount),
        target={"kind": "cutedsl", "arch": "sm_100a"},
        target_host={"kind": "llvm", "mcpu": "generic"},
    )
    assert expected in artifact.kernel_source
    assert forbidden not in artifact.kernel_source


@pytest.mark.parametrize(
    "source_dtype,output_dtype,direction,amount,values",
    [
        ("uint16", "uint32", "left", 20, [0, 1, 7, 255, 4095, 4096, 32768, 65535]),
        ("int8", "int16", "right", 1, [-128, -127, -7, -1, 0, 1, 63, 127]),
        pytest.param(
            "int8",
            "uint16",
            "right",
            1,
            [-128, -127, -7, -1, 0, 1, 63, 127],
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    "Existing CuTeDSL 4.7.0 cross-signed widening zero extension; "
                    "old baceed66 and native stacks have the same 16/32 wrong values. "
                    "Keep the signed mathematical oracle until this is fixed."
                ),
            ),
        ),
    ],
)
@tilelang.testing.requires_cuda_compute_version_eq(10, 0)
def test_native_shift_cuda_numeric(source_dtype, output_dtype, direction, amount, values, monkeypatch):
    pytest.importorskip("cutlass")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CuTe numeric kernel")
    monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", "cpp")
    kernel = tilelang.compile(
        _make_shift(source_dtype, output_dtype, direction, amount),
        target={"kind": "cutedsl", "arch": "sm_100a"},
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="cutedsl",
        out_idx=[1],
        pass_configs={"tl.disable_warp_specialized": True},
    )
    data = np.tile(np.array(values, dtype=source_dtype), 4)
    shifted = [(int(value) << amount if direction == "left" else int(value) >> amount) for value in data]
    width = np.dtype(output_dtype).itemsize * 8
    mask = (1 << width) - 1
    expected_bits = np.array([value & mask for value in shifted], dtype=f"uint{width}")
    actual = kernel(torch.from_numpy(data).cuda()).cpu().numpy()
    assert actual.dtype == np.dtype(output_dtype)
    np.testing.assert_array_equal(actual.view(f"uint{width}"), expected_bits)
