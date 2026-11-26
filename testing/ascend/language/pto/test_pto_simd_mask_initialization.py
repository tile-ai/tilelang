"""Compile mutable SIMD masks with the physical width set before loop use."""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tilelang.contrib.pto_toolchain import compile_ptodsl_to_pto, parse_pto_kernel_names, run_ptoas


@pytest.mark.pto
@pytest.mark.parametrize("bits", [16, 32])
def test_simd_mask_initialized_before_loop(bits, tmp_path):
    dtype = f"uint{bits}"
    lanes = 2048 // bits

    @T.prim_func
    def main(output: T.Tensor((lanes,), dtype), count: T.int32):
        with T.Kernel(1):
            output_ub = T.alloc_shared((lanes,), dtype)
            with T.SimdVF():
                full = S.pset(bits)
                selected = S.alloc_var("bool")
                selected = S.pnot(full, full)
                coord = S.vci(0, f"int{bits}")
                for i in T.serial(count):
                    current = S.vcmps(coord, T.cast(i, f"int{bits}"), full, "eq")
                    selected = S.por(selected, S.pand(current, full, full), full)
                result = S.vsel(S.vdup(1, dtype), S.vdup(0, dtype), selected)
                S.vsts(output_ub[0], result, full, dist=f"NORM_B{bits}")
            T.copy(output_ub, output)

    source = tilelang.lower(main, target="pto", enable_device_compile=False).kernel_source
    pto_path = tmp_path / "kernel.pto"
    object_path = tmp_path / "kernel.o"
    compile_ptodsl_to_pto(source, parse_pto_kernel_names(source), tmp_path / "kernel.py", pto_path)
    run_ptoas(pto_path, object_path, "a5", ["--pto-backend=vpto"])
    assert object_path.stat().st_size > 0
