from tvm_tirx import ir
import tvm_ffi

"""Tests for explicit SimtVF/SimdVF mode handling in Ascend codegen."""


import re

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import tvm_tirx
from tilelang.engine.lower import lower

tirx = tvm_tirx.tirx


def _simd_register_kernel(dtype):
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimdVF():
            value = T.alloc_var(dtype)
            T.evaluate(tirx.call_extern("consume", value, ty="int32"))

    return main


def _simt_broadcast_kernel(dtype):
    vector_dtype = tvm_ffi.dtype(dtype)
    element_dtype = dtype.rsplit("x", 1)[0]

    @T.prim_func
    def main():
        with T.Kernel(1), T.SimtVF(threads=1):
            value = ir.prim.Broadcast(ir.const(1, element_dtype), vector_dtype.lanes)
            T.evaluate(tirx.call_extern("consume", value, ty="int32"))

    return main


def test_vf_mode_is_restored_between_helpers():
    @T.prim_func
    def main():
        with T.Kernel(1):
            with T.SimtVF(threads=1):
                value = T.Broadcast(T.float16(1), 8)
                T.evaluate(tirx.call_extern("consume", value, ty="int32"))

            with T.SimdVF():
                mask = T.simd.pset(32)
                T.evaluate(T.simd.vdup(T.float32(2), "float32", mask))

            with T.SimtVF(threads=1):
                condition = T.Broadcast(T.bool(True), 8)
                value = T.Select(
                    condition,
                    T.Broadcast(T.float16(3), 8),
                    T.Broadcast(T.float16(4), 8),
                )
                T.evaluate(tirx.call_extern("consume", value, ty="int32"))

    source = lower(main, target="ascend").kernel_source

    assert len(re.findall(r"__simt_vf__ .*_simt_vf_", source)) == 2
    assert len(re.findall(r"__simd_vf__ .*_simd_vf_", source)) == 1
    assert "simd_inst::vdup" in source
    assert "vector_bool mask" in source
    assert source.count("uint4") >= 2


def test_simdvf_accepts_full_register_types():
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimdVF():
            f32 = T.simd.alloc_var("float32")
            f16 = T.simd.alloc_var("float16")
            bf16 = T.simd.alloc_var("bfloat16")
            s8 = T.simd.alloc_var("int8")
            u16 = T.simd.alloc_var("uint16")
            s64 = T.simd.alloc_var("int64")
            fp8 = T.simd.alloc_var("float8_e4m3fn")
            fp4 = T.simd.alloc_var("float4_e2m1fn")
            mask = T.simd.pset(32)
            T.evaluate(tirx.call_extern("consume", f32, f16, bf16, s8, u16, s64, fp8, fp4, mask, ty="int32"))

    source = lower(main, target="ascend").kernel_source

    for vector_type in (
        "vector_f32",
        "vector_f16",
        "vector_bf16",
        "vector_s8",
        "vector_u16",
        "vector_s64",
        "vector_f8e4m3",
        "vector_f4e2m1x2",
        "vector_bool",
    ):
        assert vector_type in source


@pytest.mark.parametrize(
    ("dtype", "expected_lanes"),
    [("float32x32", 64), ("float16x64", 128), ("int8x128", 256)],
)
def test_simdvf_rejects_partial_data_registers(dtype, expected_lanes):
    with pytest.raises(
        RuntimeError,
        match=rf"invalid SIMD register type {re.escape(repr(ir.PrimType(dtype)))}.*2048-bit register.*x{expected_lanes}",
    ):
        lower(_simd_register_kernel(dtype), target="ascend")


def test_simdvf_rejects_small_predicate_carrier():
    with pytest.raises(RuntimeError, match=r"predicates must be boolx256"):
        lower(_simd_register_kernel("boolx8"), target="ascend")


def test_simdvf_rejects_generic_vector_broadcast():
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimdVF():
            value = T.Broadcast(T.float32(1), 8)
            T.evaluate(tirx.call_extern("consume", value, ty="int32"))

    with pytest.raises(RuntimeError, match=r"Generic vector Broadcast.*inside SimdVF"):
        lower(main, target="ascend")


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("int8x4", "(int)"),
        ("uint8x4", "(uint)"),
        ("float32x8", "make_ulonglong4("),
        ("int4x4", "(int16_t)"),
        ("uint4x8", "(uint)"),
    ],
)
def test_simtvf_preserves_legacy_broadcast_carriers(dtype, expected):
    source = lower(_simt_broadcast_kernel(dtype), target="ascend").kernel_source

    assert expected in source


def test_simtvf_capture_signature_uses_kernel_abi_type():
    @T.prim_func
    def main(data: T.Tensor((1,), "float8_e4m3fn")):
        with T.Kernel(1):
            captured = T.alloc_var("float8_e4m3fn")
            captured = data[0]
            with T.SimtVF(threads=1):
                T.evaluate(tirx.call_extern("consume", captured, ty="int32"))

    source = lower(main, target="ascend").kernel_source

    assert re.search(r"__simt_vf__ .*\(fp8_e4_t captured\)", source)
    assert "tl::float_e4m3_t captured" not in source


def test_simdvf_local_register_address_uses_register_data():
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimdVF():
            mask = T.simd.pset(8)
            src = T.simd.alloc_var("int8")
            dst = T.simd.alloc_local((1,), "int8")
            dst[0] = src
            T.simd.vmula(dst[0], src, src, mask)

    source = lower(main, target="ascend").kernel_source

    # Native pointer lowering may name the array after its data pointer. Check
    # the same physical register array is initialized and passed by address.
    destinations = re.findall(r"vector_s8 (\w+)\[1\];", source)
    assert len(destinations) == 1
    destination = destinations[0]
    assert f"{destination}[0] = src;" in source
    assert f"simd_inst::vmula((&({destination}[0])), src, src, mask, MODE_ZEROING)" in source


def test_simd_intrinsic_requires_simdvf_context():
    @T.prim_func
    def main():
        with T.Kernel(1):
            T.evaluate(T.simd.pset(32))

    with pytest.raises(RuntimeError, match=r"tl\.simd\.pset.*inside SimdVF"):
        lower(main, target="ascend")


def test_simd_intrinsic_accepts_scalar_operand():
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimdVF():
            mask = T.simd.pset(32)
            src = T.simd.vdup(T.float32(1), "float32", mask)
            T.evaluate(T.simd.vmuls(src, T.float32(2), mask))

    assert "simd_inst::vmuls(" in lower(main, target="ascend").kernel_source


def test_nested_vf_blocks_are_rejected():
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimtVF(threads=1), T.SimdVF():
            T.evaluate(0)

    with pytest.raises(RuntimeError, match=r"Nested Ascend VF blocks"):
        lower(main, target="ascend")


def test_simdvf_captures_block_index_used_by_control_flow():
    @T.prim_func
    def main(A: T.Tensor((2, 64), "float32"), C: T.Tensor((2, 64), "float32")):
        with T.Kernel(2) as block:
            ub = T.alloc_shared((64,), "float32")
            T.copy(A[block, :], ub)
            with T.SimdVF():
                for _ in T.serial(T.min(1, block)):
                    T.simd.vsts(ub[0], T.simd.vadds(T.simd.vld(ub[0]), 1.0))
            T.copy(ub, C[block, :])

    source = lower(main, target="ascend").kernel_source
    helper = re.search(r"__simd_vf__[^\n]*\(([^\n]*)\)\s*\{", source)
    assert helper is not None
    assert "block_idx" in helper.group(1)


def test_vf_helpers_are_unique_after_explicit_unroll():
    @T.prim_func
    def main():
        with T.Kernel(1):
            for i in T.Unroll(0, 2, explicit=True):
                with T.SimdVF():
                    T.evaluate(i)
                with T.SimtVF(threads=1):
                    T.evaluate(i)

    source = lower(main, target="ascend").kernel_source

    simd_names = re.findall(r"__simd_vf__ inline void (main_kernel_simd_vf_\w+)\(", source)
    simt_names = re.findall(r"__simt_vf__ __launch_bounds__\(1\) inline void (main_kernel_simt_vf_\w+)\(", source)
    assert len(simd_names) == len(set(simd_names)) == 2
    assert len(simt_names) == len(set(simt_names)) == 2
    for name in simd_names:
        assert source.count(f"{name}();") == 1
    for name in simt_names:
        assert source.count(f"{name}>(cce::dim3(1));") == 1


@tilelang.testing.requires_ascend(support_required="compile-only")
def test_imported_helper_precedes_vf_definitions():
    imported_source = "__aicore__ inline int imported_helper(int x) { return x + 1; } // trailing comment \\"

    @T.prim_func
    def main():
        with T.Kernel(1):
            T.import_source(imported_source)
            with T.SimdVF():
                T.evaluate(tirx.call_extern("imported_helper", 1, ty="int32"))
            with T.SimtVF(threads=1):
                T.evaluate(tirx.call_extern("imported_helper", 2, ty="int32"))

    source = lower(main, target="ascend").kernel_source

    # SIMD headers are discovered while generating the VF body, after the
    # import was visited. They still belong before imported definitions.
    assert source.index("#include <tl_templates/ascend/simd_inst.h>") < source.index(imported_source)
    for vf_mode, argument in [("simd", 1), ("simt", 2)]:
        helper = re.search(rf"__{vf_mode}_vf__[^{{]+{{([^}}]*)}}", source)
        assert helper is not None, source
        assert source.index(imported_source) < helper.start(), source
        assert f"imported_helper({argument})" in helper.group(1), source


if __name__ == "__main__":
    tilelang.testing.main()
