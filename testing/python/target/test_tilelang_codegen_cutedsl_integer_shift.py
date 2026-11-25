import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl(program):
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_80"})
    assert target is not None

    with target:
        return lower(program.with_attr("global_symbol", "main"), target=target)


@tilelang.testing.requires_cutedsl(support_required="compile-only")
def test_cutedsl_codegen_promotes_narrow_shift_before_wide_cast():
    @T.prim_func
    def prog(A: T.Tensor((1,), "uint16"), B: T.Tensor((1,), "uint32")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "uint16")
            local[0] = A[0]
            B[0] = T.cast(local[0] << 20, T.uint32)

    artifact = _lower_cutedsl(prog)

    assert "cutlass.Uint32(local[0]) << cutlass.Uint16(20)" in artifact.kernel_source
    assert "cutlass.Uint32(cutlass.Uint16((local[0] << cutlass.Uint16(20))))" not in artifact.kernel_source


@tilelang.testing.requires_cutedsl(support_required="compile-only")
def test_cutedsl_codegen_does_not_promote_shift_across_signedness():
    @T.prim_func
    def prog(A: T.Tensor((1,), "int8"), B: T.Tensor((1,), "uint16")):
        with T.Kernel(1, threads=1):
            local = T.alloc_local((1,), "int8")
            local[0] = A[0]
            B[0] = T.cast(local[0] >> 1, T.uint16)

    artifact = _lower_cutedsl(prog)

    assert "cutlass.Uint16(local[0]) >>" not in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
