import re

import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl(program):
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    assert target is not None

    with target:
        return lower(program.with_attr("global_symbol", "main"), target=target)


def _calls(source, name):
    return [line for line in source.splitlines() if f"tl.{name}(" in line]


def test_cutedsl_codegen_explicit_width_global_loads_and_stores():
    @T.prim_func
    def prog(X: T.Tensor((256,), "float32"), P: T.Tensor((32,), "int32"), Y: T.Tensor((256,), "float32")):
        with T.Kernel(32, threads=32) as pid:
            Y[pid] = T.reinterpret(T.ldg32(X[pid], pred=P[pid] != 0), "float32")
            v64 = T.ldg64(X[pid * 2 : pid * 2 + 2])
            T.stg64(Y[pid * 2 : pid * 2 + 2], v64, pred=P[pid] != 0)
            v128 = T.ldg128(X[pid * 4 : pid * 4 + 4], pred=P[pid] != 0)
            T.stg128(Y[pid * 4 : pid * 4 + 4], v128)
            v256 = T.ldg256(X[pid * 8 : pid * 8 + 8])
            T.stg256(Y[pid * 8 : pid * 8 + 8], v256, pred=P[pid] != 0)
            T.stg32(Y[pid], T.reinterpret(X[pid], "uint32"))

    source = _lower_cutedsl(prog).kernel_source

    for name in ("ldg32", "ldg64", "ldg128", "ldg256", "stg32", "stg64", "stg128", "stg256"):
        assert len(_calls(source, name)) == 1, f"{name}:\n{source}"
    # Each call takes a typed pointer into its buffer; the predicate is passed
    # only when one was given.
    for name in ("ldg32", "ldg128", "stg64", "stg256"):
        (line,) = _calls(source, name)
        assert re.search(rf"tl\.{name}\(\([XY]\.iterator \+ .*P\.iterator", line), line
    for name in ("ldg64", "ldg256", "stg128", "stg32"):
        (line,) = _calls(source, name)
        assert "P.iterator" not in line, line
    # Vector results are materialized as uint32 register tensors.
    for name, lanes in (("ldg64", 2), ("ldg128", 4), ("ldg256", 8)):
        assert f"tl.make_rmem_tensor(({lanes},), cutlass.Uint32)" in source, source
        assert f".store(tl.{name}(" in source, source
    # Stores are statements, not expressions.
    for name in ("stg32", "stg64", "stg128", "stg256"):
        (line,) = _calls(source, name)
        assert line.lstrip().startswith(f"tl.{name}("), line


def test_cutedsl_codegen_explicit_width_shared_loads_and_stores():
    @T.prim_func
    def prog(X: T.Tensor((128,), "float32"), Y: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((128,), "float32")
            T.sts32(S[tx], T.reinterpret(X[tx], "uint32"))
            T.sts64(S[tx * 2 : tx * 2 + 2], T.ldg64(X[tx * 2 : tx * 2 + 2]))
            T.sts128(S[tx * 4 : tx * 4 + 4], T.ldg128(X[tx * 4 : tx * 4 + 4]))
            T.sync_threads()
            T.stg32(Y[tx], T.lds32(S[tx]))
            T.stg64(Y[tx * 2 : tx * 2 + 2], T.lds64(S[tx * 2 : tx * 2 + 2]))
            T.stg128(Y[tx * 4 : tx * 4 + 4], T.lds128(S[tx * 4 : tx * 4 + 4]))

    source = _lower_cutedsl(prog).kernel_source

    for name in ("lds32", "lds64", "lds128", "sts32", "sts64", "sts128"):
        assert len(_calls(source, name)) == 1, f"{name}:\n{source}"
    for name in ("sts32", "sts64", "sts128"):
        (line,) = _calls(source, name)
        assert line.lstrip().startswith(f"tl.{name}((S.iterator + "), line
    assert ".store(tl.lds64((S.iterator + " in source, source
    assert ".store(tl.lds128((S.iterator + " in source, source
    (line,) = _calls(source, "lds32")
    assert "tl.stg32(" in line, line


def test_cutedsl_codegen_stg32_converts_value_to_uint32():
    @T.prim_func
    def prog(A: T.Tensor((32,), "int64"), Y: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.stg32(Y[tx], A[tx])

    source = _lower_cutedsl(prog).kernel_source

    (line,) = _calls(source, "stg32")
    assert "cutlass.Uint32(" in line, line


@pytest.mark.parametrize(
    "dtype, cutedsl_type",
    [
        ("float32", "cutlass.Float32"),
        ("float16", "cutlass.Float16"),
        ("bfloat16", "cutlass.BFloat16"),
        ("uint8", "cutlass.Uint8"),
        ("int16", "cutlass.Int16"),
        ("uint64", "cutlass.Uint64"),
    ],
)
def test_cutedsl_codegen_ldg_passes_element_type(dtype, cutedsl_type):
    @T.prim_func
    def prog(x: T.Tensor((32,), dtype), y: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            y[tx] = T.__ldg(x[tx])

    source = _lower_cutedsl(prog).kernel_source

    (line,) = _calls(source, "__ldg")
    # The element type selects between the signed and unsigned ld.global.nc
    # forms, which a CuTe pointer cannot tell apart.
    assert re.search(rf"tl\.__ldg\(\(x\.iterator \+ .*\), {re.escape(cutedsl_type)}\)", line), line


def test_cutedsl_codegen_ldst_rejects_values_cuda_rejects():
    @T.prim_func
    def stg64_int(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32")):
        with T.Kernel(32, threads=32) as pid:
            T.stg64(Y[pid * 2 : pid * 2 + 2], T.reinterpret(T.ldg64(X[pid * 2 : pid * 2 + 2]), "int32x2"))

    @T.prim_func
    def stg32_vector(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32")):
        with T.Kernel(32, threads=32) as pid:
            T.stg32(Y[pid], T.ldg64(X[pid * 2 : pid * 2 + 2]))

    @T.prim_func
    def sts128_float(X: T.Tensor((128,), "float32"), Y: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((128,), "float32")
            T.sts128(S[tx * 4 : tx * 4 + 4], T.reinterpret(T.ldg128(X[tx * 4 : tx * 4 + 4]), "float32x4"))
            T.sync_threads()
            T.stg128(Y[tx * 4 : tx * 4 + 4], T.lds128(S[tx * 4 : tx * 4 + 4]))

    with pytest.raises(Exception, match="T.stg64 expects a uint32x2 value, but got int32x2"):
        _lower_cutedsl(stg64_int)
    with pytest.raises(Exception, match="T.stg32 expects a scalar value convertible to uint32"):
        _lower_cutedsl(stg32_vector)
    with pytest.raises(Exception, match="T.sts128 expects a uint32x4 value, but got float32x4"):
        _lower_cutedsl(sts128_float)


def test_cutedsl_codegen_ldg_rejects_types_without_cuda_overload():
    @T.prim_func
    def prog(x: T.Tensor((32,), "float8_e4m3fn"), y: T.Tensor((32,), "float8_e4m3fn")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            y[tx] = T.__ldg(x[tx])

    with pytest.raises(Exception, match="T.__ldg has no CUDA __ldg overload for float8_e4m3fn"):
        _lower_cutedsl(prog)


if __name__ == "__main__":
    tilelang.testing.main()
