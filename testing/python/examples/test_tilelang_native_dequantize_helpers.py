"""Numeric and ownership contracts of the examples' native IR helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from tvm_tirx import tirx

import tilelang
import tilelang.language as T
import tilelang.testing


def _example(name, monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "examples" / "dequantize_gemm"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("_native_example_" + name, directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("datatype,tile_k", [("float16", 16), ("int8", 32)])
def test_ladder_stage_three_preserves_mma_lane_ownership(datatype, tile_k, monkeypatch):
    module = _example("example_dequant_gemm_fine_grained", monkeypatch)
    values = torch.arange(32 * 64, dtype=torch.int16).to(torch.int8).reshape(32, 64)
    actual = module._ladder_permutate_int4(values, 3, datatype)
    # Each thread owns tile_k/4 packed bytes. Recover their logical B
    # coordinates from the MMA register layout, without applying the three
    # stages used by the host permutation implementation.
    expected = torch.empty_like(actual)
    for block_row in range(2):
        for block_k in range(128 // tile_k):
            for lane in range(32):
                for local_byte in range(tile_k // 4):
                    logical_row = (local_byte // (tile_k // 8)) * 8 + lane // 4
                    logical_column = (
                        (tile_k // 2) * ((local_byte // (tile_k // 16)) % 2) + (lane % 4) * (tile_k // 8) + (local_byte * 2) % (tile_k // 8)
                    ) // 2
                    flat = lane * (tile_k // 4) + local_byte
                    expected[block_row, block_k, flat // (tile_k // 2), flat % (tile_k // 2)] = values[
                        block_row * 16 + logical_row, block_k * (tile_k // 2) + logical_column
                    ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("datatype,expected", [("float16", 0x75316420), ("int8", 0x73625140)])
def test_lop3_int4_places_each_nibble_in_its_decode_lane(datatype, expected, monkeypatch):
    module = _example("example_dequant_gemm_fine_grained", monkeypatch)
    words = torch.tensor([0x76543210, -1, 0, -0x80000000], dtype=torch.int32)
    actual = module._lop3_permutate_int4(words.view(torch.int8), datatype).view(torch.int32)
    assert int(actual[0]) == expected
    assert int(actual[1]) == -1
    assert int(actual[2]) == 0
    assert int(actual[3]) == -0x80000000


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_ladder_moves_each_byte_exactly_once_and_handles_noncontiguous_input(stage, monkeypatch):
    module = _example("example_dequant_gemm_fine_grained", monkeypatch)
    storage = torch.arange(32 * 64, dtype=torch.int16).to(torch.int8).reshape(32, 64)
    values = storage[:, ::2]
    actual = module._ladder_permutate_int4(values, stage)
    assert actual.shape == (2, 4, 16, 8)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, module._ladder_permutate_int4(values.contiguous(), stage), rtol=0, atol=0)
    torch.testing.assert_close(actual.flatten().sort().values, values.flatten().sort().values, rtol=0, atol=0)


def test_ladder_rejects_partial_tiles_and_unsupported_storage(monkeypatch):
    module = _example("example_dequant_gemm_fine_grained", monkeypatch)
    with pytest.raises(ValueError, match="complete"):
        module._ladder_permutate_int4(torch.zeros((17, 8), dtype=torch.int8), 3)
    with pytest.raises(ValueError, match="int8"):
        module._ladder_permutate_int4(torch.zeros((16, 8), dtype=torch.float16), 3)
    with pytest.raises(ValueError, match="stages"):
        module._ladder_permutate_int4(torch.zeros((16, 8), dtype=torch.int8), 0)


def _compile_cpu(function):
    assert isinstance(function, tirx.Function)
    assert tirx.analysis.check_well_formed(function)
    target = {"kind": "llvm", "mcpu": "generic"}
    return tilelang.compile(function, target=target, target_host=target, execution_backend="tvm_ffi")


@tilelang.testing.requires_llvm
@pytest.mark.parametrize(
    "name,output_dtype,bias,mantissa_bit",
    [
        ("example_dequant_gemm_fp4_hopper", "float16", 14, 9),
        ("example_dequant_gemm_bf16_mxfp4_hopper", "bfloat16", 126, 6),
        ("example_dequant_gemm_bf16_mxfp4_cdna4", "bfloat16", 126, 6),
    ],
)
def test_example_fp4_helpers_execute_every_packed_byte(name, output_dtype, bias, mantissa_bit, monkeypatch):
    module = _example(name, monkeypatch)
    if output_dtype == "float16":
        helper = module._tir_u8_to_f4_to_f16

        def decode(value, pos):
            return helper(4, value, pos, output_dtype)
    else:
        helper = module._tir_u8_to_f4_to_bf16

        def decode(value, pos):
            return helper(4, value, pos, T.uint16(0), output_dtype)

    @T.prim_func
    def convert(A: T.Tensor((256,), "uint8"), B: T.Tensor((512,), "uint16")):
        for i in T.serial(512):
            B[i] = T.reinterpret(decode(A[i // 2], i % 2), "uint16")

    values = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    result = torch.empty((512,), dtype=torch.uint16)
    _compile_cpu(convert)(values, result)
    nibble = np.stack((np.arange(256) & 15, np.arange(256) >> 4), axis=-1).reshape(-1)
    # These examples intentionally use a biased normal encoding for all
    # nibbles. The rebase preserves it, including nibble zero -> 0.5.
    exponent_shift = 10 if output_dtype == "float16" else 7
    expected = ((nibble >> 3) << 15) | ((((nibble & 6) >> 1) + bias) << exponent_shift) | ((nibble & 1) << mantissa_bit)
    np.testing.assert_array_equal(result.numpy(), expected.astype(np.uint16))


@tilelang.testing.requires_llvm
def test_example_signed_int4_executes_every_packed_byte(monkeypatch):
    module = _example("example_dequant_gemm_w4a8", monkeypatch)
    helper = module._tir_u8_to_i4_to_i8

    @T.prim_func
    def convert(A: T.Tensor((256,), "uint8"), B: T.Tensor((512,), "int8")):
        for i in T.serial(512):
            B[i] = helper(4, A[i // 2], i % 2, "int8")

    values = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    result = torch.empty((512,), dtype=torch.int8)
    _compile_cpu(convert)(values, result)
    nibble = np.stack((np.arange(256) & 15, np.arange(256) >> 4), axis=-1).reshape(-1)
    expected = np.where(nibble < 8, nibble, nibble - 16).astype(np.int8)
    np.testing.assert_array_equal(result.numpy(), expected)


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("helper_name", ["_tir_u8_to_f8_e4m3_to_f16", "_tir_u8_to_f8_e4m3_to_f16_naive", "_tir_u8_to_f8_e5m2_to_f16"])
def test_common_fp8_helpers_match_torch_for_every_encoding(helper_name, monkeypatch):
    directory = Path(__file__).resolve().parents[3] / "examples" / "dequantize_gemm"
    monkeypatch.syspath_prepend(str(directory))
    from quantize import quantization

    helper = getattr(quantization, helper_name)

    @T.prim_func
    def convert(A: T.Tensor((256,), "uint8"), B: T.Tensor((256,), "float16")):
        for i in T.serial(256):
            B[i] = helper(8, A[i], "float16")

    values = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    result = torch.empty((256,), dtype=torch.float16)
    _compile_cpu(convert)(values, result)
    dtype = torch.float8_e5m2 if "e5m2" in helper_name else torch.float8_e4m3fn
    expected = values.view(dtype).to(torch.float16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
    finite = ~expected.isnan()
    torch.testing.assert_close(result.view(torch.int16)[finite], expected.view(torch.int16)[finite], rtol=0, atol=0)
