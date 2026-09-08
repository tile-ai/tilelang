import inspect

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.engine.lower import lower
from tvm import tirx


def _predicate_load_store_kernel():
    @T.prim_func
    def func(
        P: T.Buffer((128,), "uint8"),
        runtime_offset: T.int32,
    ):
        with T.Kernel(1) as _:
            p_ub = T.alloc_shared((128,), "uint8")
            T.copy(P, p_ub)
            with T.SimdVF():
                dynamic_mask = T.simd.pld(p_ub[runtime_offset], dist="US")
                default_mask = T.simd.pld(p_ub[0], dist="DS")
                T.simd.pst(p_ub[runtime_offset], dynamic_mask, dist="PK")
                T.simd.pst(p_ub[0], default_mask)

    return func


def test_predicate_load_store_ascend_codegen():
    source = lower(_predicate_load_store_kernel(), target="ascend").kernel_source

    assert "simd_inst::plds_upsample(" in source
    assert "simd_inst::plds_downsample(" in source
    assert "simd_inst::psts_pack(" in source
    assert "simd_inst::psts_norm(" in source
    assert "simd_inst::pldi(" not in source
    assert "simd_inst::psti(" not in source
    predicate_lines = [line for line in source.splitlines() if "simd_inst::p" in line]
    assert len(predicate_lines) == 4
    assert all(", 0);" in line for line in predicate_lines)


def test_predicate_public_apis_share_internal_ops():
    script = _predicate_load_store_kernel().script()

    assert script.count("T.tl.simd.pld(") == 2
    assert script.count("T.tl.simd.pst(") == 2
    assert script.count("T.access_ptr(p_ub[runtime_offset]") == 2


@pytest.mark.parametrize(
    ("op_name", "num_inputs"),
    [("vld", 2), ("vsts", 4), ("pld", 2), ("pst", 3)],
)
def test_scalar_offset_parameter_is_not_exposed(op_name, num_inputs):
    assert "off" not in inspect.signature(getattr(T.simd, op_name)).parameters
    assert tirx.op.Op.get(f"tl.simd.{op_name}").num_inputs == num_inputs


@pytest.mark.parametrize("op_name", ["pld", "pst"])
def test_predicate_load_store_builtins_are_opaque(op_name):
    op = tirx.op.Op.get(f"tl.simd.{op_name}")
    assert op.get_attr("TCallEffectKind") == tirx.CallEffectKind.Opaque


@pytest.mark.parametrize("op_name", ["init_align", "plds", "pldi", "psts", "pstu", "psti"])
def test_predicate_load_store_does_not_register_per_instruction_ops(op_name):
    with pytest.raises(AttributeError):
        tirx.op.Op.get(f"tl.simd.{op_name}")


@pytest.mark.parametrize("op_name", ["init_align", "plds", "pldi", "psts", "pstu", "psti"])
def test_predicate_instruction_specific_apis_are_not_exposed(op_name):
    assert not hasattr(T.simd, op_name)


@pytest.mark.parametrize(
    ("op_name", "dist"),
    [("pld", "PK"), ("pst", "DS"), ("pst", "US")],
)
def test_predicate_load_store_rejects_invalid_dist(op_name, dist):
    with pytest.raises(ValueError, match="dist.*must be one of"):

        @T.prim_func
        def func():
            with T.Kernel(1) as _:
                p_ub = T.alloc_shared((64,), "uint8")
                with T.SimdVF():
                    mask = T.simd.pset(8)
                    if op_name == "pld":
                        T.evaluate(T.simd.pld(p_ub[0], dist=dist))
                    else:
                        T.simd.pst(p_ub[0], mask, dist=dist)


if __name__ == "__main__":
    tilelang.testing.main()
