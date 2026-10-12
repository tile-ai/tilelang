"""PTO target selection reaches the PTO device code generator."""

import pytest

import tilelang
from tilelang.ascend import language as T
from tilelang.backend import create_backend_context


@T.prim_func
def copy_kernel(src: T.Tensor((32,), "float32"), dst: T.Tensor((32,), "float32")):
    with T.Kernel(1):
        buf = T.alloc_shared((32,), "float32")
        T.copy(src, buf)
        T.copy(buf, dst)


def test_pto_target_selects_codegen_and_execution_backend():
    context = create_backend_context("pto", "c", "auto")
    assert context.target.kind.name == "ascend"
    assert "pto" in context.target.keys
    assert context.module.name == "pto"
    assert context.module.get_device_codegen(context.target).name == "pto"
    assert context.execution_backend.name == "pto"

    with pytest.raises(ValueError, match="Invalid execution backend"):
        create_backend_context("ascend", "c", "pto")


def test_pto_lower_emits_ptodsl_source():
    ascend_source = tilelang.lower(copy_kernel, target="ascend").kernel_source
    pto_source = tilelang.lower(copy_kernel, target="pto").kernel_source
    pto_source_with_compile = tilelang.lower(copy_kernel, target="pto", enable_device_compile=True).kernel_source

    assert "from ptodsl import pto" not in ascend_source
    assert "from ptodsl import pto" in pto_source
    assert "@pto.jit(" in pto_source
    assert pto_source_with_compile == pto_source


def test_pto_single_trip_loop_does_not_create_mad_backedge():
    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "bfloat16"),
        B: T.Tensor((16, 16), "bfloat16"),
        C: T.Tensor((2, 16, 16), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), "bfloat16")
            b_l1 = T.alloc_l1((16, 16), "bfloat16")
            a_l0 = T.alloc_l0a((16, 16), "bfloat16")
            b_l0 = T.alloc_l0b((16, 16), "bfloat16")
            c0 = T.alloc_l0c((16, 16), "float32")
            c1 = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.set_mmad_direction("m")
            for _ in T.serial(1):
                T.gemm(a_l0, b_l0, c0, transpose_B=True, clear_accum=True)
                T.set_mmad_direction("n")
            T.gemm(a_l0, b_l0, c1, transpose_B=True, clear_accum=True)
            T.copy(c0, ub)
            T.copy(ub, C[0, :, :])
            T.copy(c1, ub)
            T.copy(ub, C[1, :, :])

    source = tilelang.lower(main, target="pto").kernel_source
    mad_calls = [line.strip() for line in source.splitlines() if line.strip().startswith("pto.mad(")]

    assert len(mad_calls) == 2
    assert "n_dir=True" not in mad_calls[0]
    assert "n_dir=True" in mad_calls[1]
