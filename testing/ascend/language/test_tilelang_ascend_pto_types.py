"""PTO SIMD source lowering tests."""

import pytest

import tilelang.ascend.language as T
from tilelang.engine.lower import lower


def _indent(line):
    return len(line) - len(line.lstrip())


@pytest.mark.parametrize("op_name,combine", [("fmax", T.max), ("fmin", T.min)])
@pytest.mark.pto
def test_pto_float32x2_minmax_codegen(op_name, combine):
    @T.prim_func
    def func(
        A: T.Tensor((2,), "float32"),
        B: T.Tensor((2,), "float32"),
        C: T.Tensor((2,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), "float32")
            b_ub = T.alloc_shared((2,), "float32")
            c_ub = T.alloc_shared((2,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(2):
                    c_ub[i] = combine(a_ub[i], b_ub[i])
            T.copy(c_ub, C)

    source = lower(func, target="pto").kernel_source
    assert f"tl.vectorize_binary_f32x2(pto.{op_name}," in source
    compile(source, "<pto-float32x2-minmax>", "exec")


@pytest.mark.parametrize(
    "scalar_op,unary",
    [
        ("pto.exp", T.exp),
        ("pto.log", T.log),
        ("pto.sqrt", T.sqrt),
    ],
)
@pytest.mark.pto
def test_pto_float32x2_unary_math_codegen(scalar_op, unary):
    @T.prim_func
    def func(A: T.Tensor((2,), "float32"), B: T.Tensor((2,), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), "float32")
            b_ub = T.alloc_shared((2,), "float32")
            T.copy(A, a_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(2):
                    b_ub[i] = unary(a_ub[i])
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    assert f"tl.vectorize_unary_f32x2({scalar_op}," in source
    compile(source, "<pto-float32x2-unary-math>", "exec")


@pytest.mark.pto
def test_pto_float32x2_div_codegen():
    @T.prim_func
    def func(
        A: T.Tensor((2,), "float32"),
        B: T.Tensor((2,), "float32"),
        C: T.Tensor((2,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), "float32")
            b_ub = T.alloc_shared((2,), "float32")
            c_ub = T.alloc_shared((2,), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=1):
                for i in T.vectorized(2):
                    c_ub[i] = a_ub[i] / b_ub[i]
            T.copy(c_ub, C)

    source = lower(func, target="pto").kernel_source
    assert "import tilelang.contrib.ptodsl as tl" in source
    assert "_tl_vectorize_binary_f32x2" not in source
    assert "tl.vectorize_binary_f32x2(tl.scalar_div," in source
    compile(source, "<pto-float32x2-div>", "exec")


@pytest.mark.parametrize(
    "src_dtype,dst_dtype,dst_pto_type",
    [
        ("float32", "float16", "pto.f16x2"),
        ("float32", "bfloat16", "pto.bf16x2"),
        ("float16", "float32", "pto.f32x2"),
        ("bfloat16", "float32", "pto.f32x2"),
    ],
)
@pytest.mark.pto
def test_pto_packed_float_cast_and_local_fragment_codegen(src_dtype, dst_dtype, dst_pto_type):
    @T.prim_func
    def func(A: T.Tensor((2,), src_dtype), B: T.Tensor((2,), dst_dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), src_dtype)
            b_ub = T.alloc_shared((2,), dst_dtype)
            T.copy(A, a_ub)
            with T.SimtVF(threads=1):
                local = T.alloc_fragment((2,), src_dtype)
                T.copy(a_ub, local)
                T.copy(local, b_ub)
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    assert f', {dst_pto_type}, rounding="r", saturation="nosat")' in source
    assert "pto.alloc_buffer((2,)," in source
    compile(source, "<pto-packed-float-cast>", "exec")


@pytest.mark.pto
def test_simdvf_pto_codegen_still_emits_existing_simd_source():
    @T.prim_func
    def func(A: T.Buffer((64,), "float32"), B: T.Buffer((64,), "float32")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((64,), "float32")
            b_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_ub)
            with T.SimdVF():
                x = T.simd.vld(a_ub[0])
                y = T.simd.vadd(x, x)
                T.simd.vsts(b_ub[0], y)
            T.copy(b_ub, B)

    source = lower(func, target="pto").kernel_source
    compile(source, "<pto-source>", "exec")
    vecscope_lines = [line for line in source.splitlines() if "with pto.vecscope():" in line]
    assert len(vecscope_lines) == 1
    vecscope_indent = _indent(vecscope_lines[0])
    vector_lines = [line for line in source.splitlines() if any(op in line for op in ("pto.vlds(", "pto.vadd(", "pto.vsts("))]
    assert vector_lines
    assert all(_indent(line) > vecscope_indent for line in vector_lines)
    assert "pto.vlds(" in source
    assert "pto.vadd(" in source
    assert "pto.vsts(" in source


@pytest.mark.pto
def test_empty_simdvf_pto_codegen_emits_valid_python():
    @T.prim_func
    def func():
        with T.Kernel(1) as _, T.SimdVF():
            pass

    source = lower(func, target="pto").kernel_source
    compile(source, "<pto-empty-simdvf>", "exec")

    lines = source.splitlines()
    vecscope_lines = [(index, line) for index, line in enumerate(lines) if "with pto.vecscope():" in line]
    assert len(vecscope_lines) == 1
    vecscope_index, vecscope_line = vecscope_lines[0]
    body_lines = [line for line in lines[vecscope_index + 1 :] if line.strip()]
    assert body_lines
    assert body_lines[0].strip() == "pass"
    assert _indent(body_lines[0]) > _indent(vecscope_line)
