"""Tests for the FP6 unpacked shared-memory dtypes (16U6_ALIGN16B storage).

Mirrors the fp4 unpacked dtype tests: these are 8-bit-container storage tags
(CUTLASS ``float_e2m3_unpacksmem_t`` / ``float_e3m2_unpacksmem_t``) for the
f8f6f4 / mxf8f6f4 smem forms, not general register dtypes.
"""

import tilelang
import tilelang.language as T
import tilelang.testing


def test_float6_unpacked_dtype_properties():
    e2m3 = T.float6_e2m3fn_unpacked
    e3m2 = T.float6_e3m2fn_unpacked
    assert e2m3.bits == 8
    assert e3m2.bits == 8
    assert int(e2m3.type_code) == 132
    assert int(e3m2.type_code) == 133
    assert str(e2m3) == "custom[float6_e2m3fn_unpacked]8"
    assert str(e3m2) == "custom[float6_e3m2fn_unpacked]8"
    assert e2m3.lanes == 1 and e3m2.lanes == 1
    assert e2m3 != e3m2


def test_float6_unpacked_distinct_from_packed():
    assert T.float6_e2m3fn.bits == 6
    assert T.float6_e3m2fn.bits == 6
    assert T.float6_e2m3fn != T.float6_e2m3fn_unpacked
    assert T.float6_e3m2fn != T.float6_e3m2fn_unpacked


def test_float6_unpacked_dtype_helpers():
    from tilelang.language.dtypes import is_float6_unpacked

    e2m3 = T.float6_e2m3fn_unpacked
    e3m2 = T.float6_e3m2fn_unpacked
    assert e2m3.is_float6_e2m3fn_unpacked()
    assert not e2m3.is_float6_e3m2fn_unpacked()
    assert e2m3.is_float6_unpacked()
    assert e3m2.is_float6_e3m2fn_unpacked()
    assert e3m2.is_float6_unpacked()
    assert not T.float6_e2m3fn.is_float6_unpacked()
    assert is_float6_unpacked("custom[float6_e2m3fn_unpacked]8")
    assert not is_float6_unpacked(T.float16)


def test_float6_unpacked_in_f8f6f4_family():
    """The family predicate must see the custom-registered names.

    This was a latent bug: the predicate was string-prefix based and
    ``custom[...]`` names never match ``float6``/``float8`` prefixes.
    """
    from tilelang.language.dtypes import is_f8f6f4_family

    assert is_f8f6f4_family(T.float6_e2m3fn_unpacked)
    assert is_f8f6f4_family(T.float6_e3m2fn_unpacked)
    assert is_f8f6f4_family(T.float4_e2m1_unpacked)
    assert is_f8f6f4_family(T.float8_e4m3fn)
    assert not is_f8f6f4_family(T.bfloat16)
    assert not is_f8f6f4_family(T.int8)


def test_float6_unpacked_tir_dtype_tag():
    @T.prim_func
    def main():
        T.evaluate(T.float6_e2m3fn_unpacked(0.0))

    assert main.body.value.dtype.is_float6_e2m3fn_unpacked()


def test_float6_unpacked_mma_abbrev():
    from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

    abbrv = TensorCoreIntrinEmitter.dtype_abbrv
    assert abbrv["custom[float6_e2m3fn_unpacked]8"] == "e2m3"
    assert abbrv["custom[float6_e3m2fn_unpacked]8"] == "e3m2"


@tilelang.testing.requires_cuda
def test_float6_unpacked_shared_alloc_prints_bytes():
    """An unpacked-fp6 shared buffer must lower to plain byte storage."""

    @T.prim_func
    def main(OUT: T.Tensor((64,), T.uint8)):
        with T.Kernel(1, threads=64) as _:
            tx = T.get_thread_binding()
            smem = T.alloc_shared((64,), T.float6_e3m2fn_unpacked, scope="shared.dyn")
            smem[tx] = T.reinterpret(T.float6_e3m2fn_unpacked, T.uint8(0))
            OUT[tx] = T.reinterpret(T.uint8, smem[tx])

    kernel = tilelang.compile(main, target="cuda", out_idx=[0])
    src = kernel.get_kernel_source()
    assert "__nv_fp6" not in src  # never the packed register type


if __name__ == "__main__":
    tilelang.testing.main()
