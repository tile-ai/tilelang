"""TF32 copies preserve source bounds and the dtype's conversion semantics."""

import numpy as np
import pytest
import torch
import tvm_ffi

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm_tirx import ir
from tvm_tirx import tirx


def _copy(extent, destination_dtype, safe_value=None):
    @T.prim_func
    def padded(
        A: T.Tensor((extent,), T.tfloat32),
        B: T.Tensor((64,), destination_dtype),
    ):
        with T.Kernel(2, threads=32) as block:
            if safe_value is not None:
                T.annotate_safe_value({A: safe_value})
            T.copy(A[block * 32 : (block + 1) * 32], B[block * 32 : (block + 1) * 32])

    return padded


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("extent", [0, 7, 33])
@pytest.mark.parametrize("destination_dtype", [T.tfloat32, T.float32])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_tf32_copy_padding_preserves_storage_and_conversion(extent, destination_dtype, roundtrip):
    function = _copy(extent, destination_dtype)
    if roundtrip:
        function = ir.load_json(ir.save_json(function))
    assert function.params[0].ty.dtype == ir.PrimType(T.tfloat32)
    kernel = tilelang.compile(function, out_idx=[1], target="cuda", target_host="c", execution_backend="tvm_ffi")
    # A same-dtype copy retains all input bits, including signed zero and NaN
    # payloads. Conversion to float32 observes TF32's ten-bit mantissa.
    pattern = np.array(
        [0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000, 0x3F800001, 0xBF800FFF, 0x00000001],
        dtype="uint32",
    )
    bits = np.resize(pattern, extent)
    source = torch.from_numpy(bits).cuda().view(torch.float32)
    actual = kernel(source)
    torch.cuda.synchronize()
    assert actual.dtype == torch.float32
    actual_bits = actual.view(torch.uint32).cpu().numpy()
    tf32_mask = np.uint32(0xFFFFE000)
    expected_input = bits if destination_dtype == T.tfloat32 else bits & tf32_mask
    np.testing.assert_array_equal(actual_bits[:extent], expected_input)
    # The CUTLASS TF32 constructor may populate the unused low thirteen bits.
    # Its padding still represents positive zero, with no speculative read.
    padding = actual_bits[extent:]
    if destination_dtype == T.tfloat32:
        padding = padding & tf32_mask
    np.testing.assert_array_equal(padding, np.zeros(64 - extent, dtype="uint32"))
    with pytest.raises(RuntimeError, match="dtype"):
        kernel(torch.zeros(extent, dtype=torch.float16, device="cuda"))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("safe_value", [-3.5, -0.0])
def test_tf32_copy_explicit_padding_value_keeps_its_meaning(safe_value):
    function = _copy(7, T.tfloat32, safe_value=safe_value)
    kernel = tilelang.compile(function, out_idx=[1], target="cuda", target_host="c", execution_backend="tvm_ffi")
    source = torch.arange(7, dtype=torch.float32, device="cuda")
    actual = kernel(source)
    expected = torch.full((64,), safe_value, dtype=torch.float32, device="cuda")
    expected[:7] = source
    torch.testing.assert_close(actual[:7].view(torch.int32), source.view(torch.int32), rtol=0, atol=0)
    # Check TF32's represented value, including the sign of negative zero;
    # its unused mantissa bits are not an IEEE float32 value contract.
    logical_bits = actual.view(torch.int32).bitwise_and(-8192)
    torch.testing.assert_close(logical_bits, expected.view(torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("safe_value", [None, -3.5])
def test_tf32_safe_memory_pass_preserves_the_padding_contract(safe_value):
    extent = T.dynamic("extent")

    @T.prim_func
    def padded(A: T.Tensor((extent,), T.tfloat32), B: T.Tensor((1,), T.tfloat32)):
        with T.sblock("root"):
            if safe_value is not None:
                T.annotate_safe_value({A: safe_value})
            B[0] = A[extent]

    lowered = tilelang.transform.LegalizeSafeMemoryAccess()(ir.IRModule({"main": padded}))
    assert tirx.analysis.check_well_formed(lowered)
    constants = []
    tvm_ffi.structural_walk(lowered["main"].body, (ir.FloatImm, constants.append), order="pre")
    expected = 0.0 if safe_value is None else safe_value
    assert any(value.ty.dtype == T.tfloat32 and value.value == expected for value in constants)
