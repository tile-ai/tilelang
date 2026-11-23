"""AscendC type names must not be usable as buffer or parameter identifiers.

The emitted kernel includes the ``tl_templates/ascend`` headers, which typedef
``half``, ``bfloat16_t`` and the fixed-width integer types, and the C++
``bool`` keyword. ``CodeGenC::ReserveKeywordsAsUnique`` only reserves the
standard C++ keywords, so a buffer or parameter named ``half`` used to be
printed verbatim and shadowed the type:

    extern "C" __global__ __vector__ void f_kernel(__gm__ half* C, __gm__ half* half) {
      __ubuf__ half *half = (__ubuf__ half *)0;   // `half` is now the parameter

which fails device compilation with ``unknown type name 'half'``.

These tests are source-only: they lower to AscendC and inspect the generated
signature, so they run without an NPU.
"""

import re

import pytest

import tilelang
from tilelang.ascend import language as T

M, N = 16, 16


def _signature_identifiers(source):
    """Parameter identifiers of the generated kernel entry point."""
    match = re.search(r"void \w+_kernel\(([^)]*)\)", source)
    assert match is not None, source
    identifiers = []
    for param in match.group(1).split(","):
        # "__gm__ half* C" -> "C"
        identifiers.append(param.split("*")[-1].strip())
    return identifiers


# --- parameters named after an AscendC type ---------------------------------


def _param_half():
    @T.prim_func
    def kernel(half: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            buf = T.alloc_shared((M, N), "float16")
            T.copy(half, buf)
            T.copy(buf, C)

    return kernel


def _param_bfloat16_t():
    @T.prim_func
    def kernel(bfloat16_t: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            buf = T.alloc_shared((M, N), "float16")
            T.copy(bfloat16_t, buf)
            T.copy(buf, C)

    return kernel


def _param_uint8_t():
    @T.prim_func
    def kernel(uint8_t: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            buf = T.alloc_shared((M, N), "float16")
            T.copy(uint8_t, buf)
            T.copy(buf, C)

    return kernel


# --- buffers named after an AscendC type ------------------------------------


def _buffer_half():
    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            half = T.alloc_shared((M, N), "float16")
            T.copy(A, half)
            T.copy(half, C)

    return kernel


def _buffer_bfloat16_t():
    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            bfloat16_t = T.alloc_shared((M, N), "float16")
            T.copy(A, bfloat16_t)
            T.copy(bfloat16_t, C)

    return kernel


def _buffer_uint8_t():
    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            uint8_t = T.alloc_shared((M, N), "float16")
            T.copy(A, uint8_t)
            T.copy(uint8_t, C)

    return kernel


# --- a control name that must stay untouched --------------------------------


def _param_plain():
    @T.prim_func
    def kernel(A: T.Tensor((M, N), "float16"), C: T.Tensor((M, N), "float16")):
        with T.Kernel(1):
            buf = T.alloc_shared((M, N), "float16")
            T.copy(A, buf)
            T.copy(buf, C)

    return kernel


@pytest.mark.parametrize(
    "build, reserved",
    [
        (_param_half, "half"),
        (_param_bfloat16_t, "bfloat16_t"),
        (_param_uint8_t, "uint8_t"),
    ],
    ids=["param-half", "param-bfloat16_t", "param-uint8_t"],
)
def test_parameter_named_after_type_is_uniquified(build, reserved):
    source = tilelang.lower(build(), target="ascend").kernel_source
    identifiers = _signature_identifiers(source)
    assert reserved not in identifiers, f"{reserved!r} leaked into the signature: {identifiers}"
    assert any(identifier.startswith(reserved + "_") for identifier in identifiers), (
        f"expected {reserved!r} to be uniquified, got {identifiers}"
    )


@pytest.mark.parametrize(
    "build, reserved",
    [
        (_buffer_half, "half"),
        (_buffer_bfloat16_t, "bfloat16_t"),
        (_buffer_uint8_t, "uint8_t"),
    ],
    ids=["buffer-half", "buffer-bfloat16_t", "buffer-uint8_t"],
)
def test_buffer_named_after_type_does_not_shadow_the_typedef(build, reserved):
    source = tilelang.lower(build(), target="ascend").kernel_source
    # The typedef must still be usable as a type: an allocation may print
    # "__ubuf__ half *half_1", but never "__ubuf__ half *half".
    shadowing = re.search(rf"__ubuf__\s+{reserved}\s*\*\s*{reserved}\b", source)
    assert shadowing is None, f"{reserved!r} is both the type and the variable:\n{source}"
    # And the uniquified name must actually be what the body uses.
    assert f"{reserved}_" in source, source


def test_ordinary_parameter_name_is_preserved():
    """The reservation must not rename unrelated identifiers."""
    source = tilelang.lower(_param_plain(), target="ascend").kernel_source
    identifiers = _signature_identifiers(source)
    assert identifiers == ["A", "C"], identifiers
