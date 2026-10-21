import pytest
import tilelang
import tilelang.testing

from tilelang import tvm
from tvm import tirx


def _make_vector_not_module(lanes):
    value = tirx.Var("value", f"boolx{lanes}")
    func = tirx.PrimFunc([value], tirx.Evaluate(tirx.Not(value)))
    func = func.with_attr("global_symbol", "vector_not")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return tvm.IRModule({"vector_not": func})


def _build_vector_not_source(lanes):
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(_make_vector_not_module(lanes), tvm.target.Target("rocm")).inspect_source()


@pytest.mark.parametrize("lanes", [2, 3, 4])
def test_vector_not_is_scalarized(lanes):
    source = _build_vector_not_source(lanes)

    assignments = [line for line in source.splitlines() if "!bool(" in line]
    assert len(assignments) == lanes


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("lanes", [2, 3, 4])
def test_vector_not_compiles(lanes):
    build = tvm.get_global_func("target.build.tilelang_hip")
    build(_make_vector_not_module(lanes), tvm.target.Target("rocm"))


if __name__ == "__main__":
    tilelang.testing.main()
