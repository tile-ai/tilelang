import pytest
import tilelang
import tilelang.testing

from tilelang import tvm
from tvm import tirx


def _make_vector_not_module(lanes):
    value = tirx.Var("value", f"boolx{lanes}")
    negated = tirx.Not(value)
    func = tirx.PrimFunc([value], tirx.Evaluate(negated))
    func = func.with_attr("global_symbol", "vector_not")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm.IRModule({"vector_not": func})


def _build_vector_not_source(lanes):
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the CUDA code generator")
    return build(_make_vector_not_module(lanes), tvm.target.Target("cuda")).inspect_source()


@pytest.mark.parametrize(
    ("lanes", "carrier_type"),
    [(2, "ushort2"), (3, "ushort3"), (4, "ushort4")],
)
def test_vector_not_is_scalarized(lanes, carrier_type):
    source = _build_vector_not_source(lanes)

    declarations = [line for line in source.splitlines() if line.startswith(f"  {carrier_type} __")]
    assert len(declarations) == 1
    assignments = [line for line in source.splitlines() if "!bool(" in line]
    assert len(assignments) == lanes
    assert all(f".{member}" in assignment for member, assignment in zip("xyzw", assignments))
    # The base CodeGenC path negates the whole carrier, which is invalid.
    assert "!__" not in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("lanes", [2, 3, 4])
def test_vector_not_compiles(lanes):
    build = tvm.get_global_func("target.build.tilelang_cuda")
    build(_make_vector_not_module(lanes), tvm.target.Target("cuda"))


if __name__ == "__main__":
    tilelang.testing.main()
