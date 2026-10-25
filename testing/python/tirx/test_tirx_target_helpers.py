import pytest
import tvm_ffi
from tvm_tirx import ir
from tvm_tirx.target import Target

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def native(name, *arguments):
    return tvm_ffi.get_global_func(f"tl.{name}")(*arguments)


@pytest.mark.parametrize(
    "arch,family",
    [("sm_70", "Volta"), ("sm_75", "Turing"), ("sm_80", "Ampere"), ("sm_89", "Ampere"), ("sm_90a", "Hopper"), ("sm_120", "SM120")],
)
def test_native_cuda_architecture_helpers(arch, family):
    target = Target({"kind": "cuda", "arch": arch})
    for candidate in ["Volta", "Turing", "Ampere", "Hopper", "SM120"]:
        assert native(f"TargetIs{candidate}", target) == (candidate == family)
    assert native("TargetCudaGetWarpSize", target) == 32


@pytest.mark.parametrize("arch,basic,extended", [("sm_80", False, False), ("sm_90", True, False), ("sm_100", True, True)])
def test_native_matrix_store_capabilities(arch, basic, extended):
    target = Target({"kind": "cuda", "arch": arch})
    assert native("TargetHasStmatrix", target) == basic
    assert native("TargetHasStmatrix", target, True) == extended


def test_256_bit_planner_cap_uses_native_boolean_config():
    target = Target({"kind": "cuda", "arch": "sm_100"})
    assert native("MaxVectorLoadBits", target, True) == 256
    assert native("MaxVectorLoadBits", target, False) == 128
    with ir.transform.PassContext(config={"tl.disable_vectorize_256": True}):
        assert native("MaxVectorLoadBits", target, True) == 128
    with ir.transform.PassContext(config={"tl.disable_vectorize_256": False}):
        assert native("MaxVectorLoadBits", target, True) == 256


@pytest.mark.parametrize(
    "source,destination,expected",
    [
        ("float32", "float16", True),
        ("bfloat16", "float32", True),
        ("float8_e5m2", "float16", True),
        ("float32", "int32", False),
        ("float64", "float16", False),
    ],
)
def test_native_dtype_queries(source, destination, expected):
    assert native("IsCudaVectorizableCast", ir.PrimType(source), ir.PrimType(destination)) == expected


@pytest.mark.parametrize(
    "architecture,cdna,warp,generation",
    [("gfx942", True, 64, 0), ("gfx950", True, 64, 0), ("gfx1100", False, 32, 11), ("gfx1200", False, 32, 12)],
)
def test_native_rocm_architecture_helpers(architecture, cdna, warp, generation):
    target = Target({"kind": "rocm", "mcpu": architecture})
    assert native("TargetIsCDNA", target) == cdna
    assert native("TargetIsRDNA", target) == (not cdna)
    assert native("TargetRocmGetWarpSize", target) == warp
    assert native("TargetGetRDNAGeneration", target) == generation
