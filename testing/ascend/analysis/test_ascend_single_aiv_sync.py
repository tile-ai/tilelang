import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.engine.lower import lower


TILE = 16
STEPS = 4


def _make_epilogue_program(sids: int):

    @T.prim_func
    def main(
        A: T.Buffer((STEPS, TILE, TILE), "bfloat16"),
        B: T.Buffer((STEPS, TILE, TILE), "bfloat16"),
        C: T.Buffer((STEPS, TILE, TILE), "bfloat16"),
    ):
        with T.MixedKernel(1, sids=sids):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            output = T.alloc_shared((TILE, TILE), "bfloat16")
            T.annotate_buffer_versions({output: 2})

            for step in T.Pipelined(STEPS, num_stages=2):
                T.copy(A[step, :, :], a_l1)
                T.copy(B[step, :, :], b_l1)
                T.gemm(
                    a_l1,
                    b_l1,
                    accum,
                    transpose_B=True,
                    clear_accum=True,
                    unit_flag_ctrl=3,
                )
                T.copy(accum, output, unit_flag_ctrl=3)
                T.copy(output, C[step, :, :])

    return main


def _lower_with_snapshots(program):
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in {"tl.InsertSync", "tl.LowerScheduledTIR"}:
                snapshots[info.name] = mod.script()

    with tvm.transform.PassContext(opt_level=3, instruments=[Capture()]):
        artifact = lower(program, target="ascend")
    return artifact.kernel_source, snapshots


def _aic_branch(source: str) -> str:
    return source.split("if ASCEND_IS_AIC {", maxsplit=1)[1].split("if ASCEND_IS_AIV {", maxsplit=1)[0]


def test_mixed_kernel_single_aiv_lowers_one_subcore_protocol():
    source, snapshots = _lower_with_snapshots(_make_epilogue_program(1))

    assert "__global__ __mix__(1, 2)" in source
    assert "if (get_subblockid() == 0)" in source
    assert "get_subblockid()" in source
    assert '"vector_count": 1' in snapshots["tl.LowerScheduledTIR"]
    assert "CrossCoreSetFlag<4," in source
    assert "CrossCoreWaitFlag<4," in source
    assert "CrossCoreSetFlag<2," not in source
    assert "CrossCoreWaitFlag<2," not in source
    aic_cross_core = [line for line in _aic_branch(source).splitlines() if "CrossCore" in line]
    assert aic_cross_core
    assert not any("+ 16" in line for line in aic_cross_core)


def test_mixed_kernel_default_keeps_two_subcores():
    source, snapshots = _lower_with_snapshots(_make_epilogue_program(2))

    assert "__global__ __mix__(1, 2)" in source
    assert "get_subblockid()" in source
    assert '"vector_count": 2' in snapshots["tl.LowerScheduledTIR"]
    assert "CrossCoreSetFlag<4," in source
    assert "CrossCoreWaitFlag<4," in source
    aic_cross_core = [line for line in _aic_branch(source).splitlines() if "CrossCore" in line]
    assert any("+ 16" in line for line in aic_cross_core)


if __name__ == "__main__":
    tilelang.testing.main()
