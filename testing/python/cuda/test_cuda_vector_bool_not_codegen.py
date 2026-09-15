import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

from tilelang import tvm
from tvm import tirx


def _make_vector_not_module(lanes, bitwise=False):
    value = tirx.Var("value", f"boolx{lanes}")
    negated = ~value if bitwise else tirx.Not(value)
    func = tirx.PrimFunc([value], tirx.Evaluate(negated))
    func = func.with_attr("global_symbol", "vector_not")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm.IRModule({"vector_not": func})


def _build_vector_not_source(lanes, bitwise=False):
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the CUDA code generator")
    return build(_make_vector_not_module(lanes, bitwise), tvm.target.Target("cuda")).inspect_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("lanes", "carrier_type"),
    [(2, "ushort2"), (3, "ushort3"), (4, "ushort4")],
)
@pytest.mark.parametrize("bitwise", [False, True])
def test_vector_not_is_scalarized(lanes, carrier_type, bitwise):
    source = _build_vector_not_source(lanes, bitwise)

    declarations = [line for line in source.splitlines() if line.startswith(f"  {carrier_type} __")]
    assert len(declarations) == 1
    assignments = [line for line in source.splitlines() if "!bool(" in line]
    assert len(assignments) == lanes
    assert all(f".{member}" in assignment for member, assignment in zip("xyzw", assignments))
    # The base CodeGenC path negates the whole carrier, which is invalid.
    assert "!__" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("lanes", [2, 3, 4])
@pytest.mark.parametrize("bitwise", [False, True])
def test_vector_not_compiles(lanes, bitwise):
    build = tvm.get_global_func("target.build.tilelang_cuda")
    build(_make_vector_not_module(lanes, bitwise), tvm.target.Target("cuda"))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int32"])
@pytest.mark.parametrize("explicit_intrinsic", [False, True])
def test_bitwise_not_values(dtype, explicit_intrinsic):
    @T.prim_func
    def main(A: T.Tensor((4,), dtype), B: T.Tensor((4,), dtype), C: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for i in T.serial(4):
                value = T.bitwise_not(A[i]) if explicit_intrinsic else ~A[i]
                B[i] = value
                C[i] = T.Cast("int32", value)

    kernel = tilelang.compile(main, out_idx=[1, 2], target="cuda")
    values = [True, False, True, False] if dtype == "bool" else [0, 1, 5, 127]
    a = torch.tensor(values, dtype=getattr(torch, dtype), device="cuda")
    b, c = kernel(a)
    expected = ~a
    torch.testing.assert_close(b, expected)
    torch.testing.assert_close(c, expected.to(torch.int32))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [f"{sign}{bits}" for sign in ("int", "uint") for bits in (8, 16, 32, 64)])
@pytest.mark.parametrize("elements_per_thread", [1, 2, 4, 8])
@pytest.mark.parametrize("explicit_intrinsic", [False, True])
def test_integer_vector_bitwise_not_values(dtype, elements_per_thread, explicit_intrinsic):
    n = 32 * elements_per_thread

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                B[i] = T.bitwise_not(A[i]) if explicit_intrinsic else ~A[i]

    bits = torch.iinfo(getattr(torch, dtype)).bits
    mask = (1 << bits) - 1

    def as_value(value):
        return value - (1 << bits) if dtype.startswith("int") and value >= (1 << (bits - 1)) else value

    patterns = [0, mask, 1, 1 << (bits - 1), (1 << (bits - 1)) - 1, mask // 3, 2 * (mask // 3), mask - 1]
    values = patterns * (n // len(patterns))
    a = torch.tensor([as_value(x) for x in values], dtype=getattr(torch, dtype), device="cuda")
    kernel = tilelang.compile(main, out_idx=[1], target="cuda", execution_backend="tvm_ffi")
    assert kernel(a).cpu().tolist() == [as_value(x ^ mask) for x in values]


if __name__ == "__main__":
    tilelang.testing.main()
