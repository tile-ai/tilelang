import tilelang.language as T
import tilelang.testing
import torch
import operator
import pytest
from tvm import arith, tirx
from tvm.error import DiagnosticError
from tvm.tirx.script import prim_func as classic_prim_func


_DIVMOD_OPS = [operator.mod, operator.floordiv, T.floormod, T.floordiv, T.ceildiv, T.truncdiv]


@pytest.mark.parametrize("op", _DIVMOD_OPS, ids=["%", "//", "floormod", "floordiv", "ceildiv", "truncdiv"])
@pytest.mark.parametrize(
    "lhs, rhs",
    [
        (tirx.Var("x", "float32"), tirx.Var("y", "float32")),
        (tirx.Var("x", "float32"), 2),
        (2, tirx.Var("x", "float32")),
        (tirx.Var("x", "int32"), 2.5),
        (2.5, tirx.Var("x", "int32")),
        (tirx.Var("x", "float32"), tirx.Var("y", "int32")),
        (tirx.Var("x", "int32"), tirx.Var("y", "float32")),
        (T.float32(1.5), T.float32(2.5)),
    ],
)
def test_divmod_float_diagnostic(op, lhs, rhs):
    with pytest.raises(TypeError, match="requires integer operands.*floating-point"):
        op(lhs, rhs)


@pytest.mark.parametrize("dtype", ["float16", "float64", "bfloat16", "float32x4"])
def test_divmod_float_dtype_diagnostic(dtype):
    with pytest.raises(TypeError, match="requires integer operands"):
        T.floormod(tirx.Var("x", dtype), 2)


@pytest.mark.parametrize("op, expected", list(zip(_DIVMOD_OPS, [2, -3, 2, -3, -2, -2])))
def test_divmod_integer_semantics(op, expected):
    assert int(op(T.int32(-7), T.int32(3))) == expected
    x = tirx.Var("x", "int32")
    analyzer = arith.Analyzer()
    assert int(analyzer.simplify(tirx.stmt_functor.substitute(op(x, 3), {x: T.int32(-7)}))) == expected
    assert int(analyzer.simplify(tirx.stmt_functor.substitute(op(-7, x), {x: T.int32(3)}))) == expected


@pytest.mark.parametrize("classic", [False, True], ids=["eager", "classic"])
@pytest.mark.parametrize("op", _DIVMOD_OPS)
def test_divmod_parser_float_diagnostic(classic, op, capfd):
    with pytest.raises((TypeError, DiagnosticError)) as exc:
        if classic:

            @classic_prim_func
            def main(A: T.Tensor((1,), "float32")):
                T.evaluate(op(A[0], 2))
        else:

            @T.prim_func
            def main(A: T.Tensor((1,), "float32")):
                T.evaluate(op(A[0], 2))

    assert "requires integer operands" in str(exc.value) + capfd.readouterr().err


@pytest.mark.parametrize("classic", [False, True], ids=["eager", "classic"])
@pytest.mark.parametrize("kind", ["mod", "floordiv"])
def test_divmod_parser_syntax(classic, kind, capfd):
    with pytest.raises((TypeError, DiagnosticError)) as exc:
        if classic:

            @classic_prim_func
            def main(A: T.Tensor((1,), "float32")):
                if kind == "mod":
                    T.evaluate(2 % A[0])
                else:
                    T.evaluate(A[0] // 2.5)
        else:

            @T.prim_func
            def main(A: T.Tensor((1,), "float32")):
                if kind == "mod":
                    T.evaluate(2 % A[0])
                else:
                    T.evaluate(A[0] // 2.5)

    assert "requires integer operands" in str(exc.value) + capfd.readouterr().err


@pytest.mark.parametrize("classic", [False, True], ids=["eager", "classic"])
def test_divmod_parser_integer(classic):
    if classic:

        @classic_prim_func
        def main(A: T.Tensor((1,), "int32"), B: T.Tensor((6,), "int32")):
            B[0] = A[0] % 3
            B[1] = -7 // A[0]
            B[2] = T.floormod(A[0], 3)
            B[3] = T.floordiv(A[0], 3)
            B[4] = T.ceildiv(A[0], 3)
            B[5] = T.truncdiv(A[0], 3)
    else:

        @T.prim_func
        def main(A: T.Tensor((1,), "int32"), B: T.Tensor((6,), "int32")):
            B[0] = A[0] % 3
            B[1] = -7 // A[0]
            B[2] = T.floormod(A[0], 3)
            B[3] = T.floordiv(A[0], 3)
            B[4] = T.ceildiv(A[0], 3)
            B[5] = T.truncdiv(A[0], 3)

    assert isinstance(main, tirx.PrimFunc)


@tilelang.jit(out_idx=[-1])
def _ceildiv_kernel(a: int, b: int):
    @T.prim_func
    def ceildiv_kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.ceildiv(T.int32(a), T.int32(b))

    return ceildiv_kernel


@tilelang.jit(out_idx=[-1])
def _align_up_kernel(a: int, b: int):
    @T.prim_func
    def align_up_kernel(A: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.align_up(T.int32(a), T.int32(b))

    return align_up_kernel


def run_ceildiv(a=128, b=32):
    kernel = _ceildiv_kernel(a, b)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)


def test_ceildiv():
    run_ceildiv(a=128, b=32)
    run_ceildiv(a=1, b=32)
    run_ceildiv(a=-1, b=32)
    run_ceildiv(a=-2, b=32)


def run_align_up(a=128, b=32):
    kernel = _align_up_kernel(a, b)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)


def test_align_up():
    run_align_up(a=128, b=32)
    run_align_up(a=1, b=32)
    run_align_up(a=-1, b=32)
    run_align_up(a=-2, b=32)


@tilelang.jit
def _ceildiv_kernel_dyn(b: int):
    @T.prim_func
    def ceildiv_kernel(A: T.Tensor((1,), T.int32), a: T.int32):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.ceildiv(T.int32(a), T.int32(b))

    return ceildiv_kernel


@tilelang.jit
def _align_up_kernel_dyn(b: int):
    @T.prim_func
    def align_up_kernel(A: T.Tensor((1,), T.int32), a: T.int32):
        with T.Kernel(1, threads=1) as _:
            A[0] = T.align_up(T.int32(a), T.int32(b))

    return align_up_kernel


def run_ceildiv_dyn(a=128, b=32):
    kernel = _ceildiv_kernel_dyn(b)
    A = torch.empty((1,), dtype=torch.int32, device="cuda")
    kernel(A, a)
    print(kernel.get_kernel_source())
    print(A)


@tilelang.testing.requires_cuda
def test_ceildiv_dyn():
    run_ceildiv_dyn(a=128, b=32)
    run_ceildiv_dyn(a=1, b=32)
    run_ceildiv_dyn(a=-1, b=32)
    run_ceildiv_dyn(a=-2, b=32)


def run_align_up_dyn(a=128, b=32):
    kernel = _align_up_kernel_dyn(b)
    A = torch.empty((1,), dtype=torch.int32, device="cuda")
    kernel(A, a)
    print(kernel.get_kernel_source())
    print(A)


@tilelang.testing.requires_cuda
def test_align_up_dyn():
    run_align_up_dyn(a=128, b=32)
    run_align_up_dyn(a=1, b=32)
    run_align_up_dyn(a=-1, b=32)
    run_align_up_dyn(a=-2, b=32)


if __name__ == "__main__":
    tilelang.testing.main()
