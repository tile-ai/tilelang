import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget, PreLowerSemanticCheck
from tilelang.backend.target import determine_target

from testing.python.language.simtvf._inspect_utils import (
    collect_simtvf_blocks,
    collect_thread_extent_tags,
    collect_thread_extent_tags_outside_simtvf,
    print_stage,
)


def main():
    target = determine_target("ascend", return_object=True)
    with target:

        @T.prim_func
        def simtvf_two_threads_demo(
            A: T.Buffer((256,), "float32"),
            B: T.Buffer((256,), "float32"),
            C: T.Buffer((256,), "float32"),
        ):
            with T.Kernel(1):
                with T.SimtVF(threads=128):
                    for i in T.Parallel(256):
                        B[i] = A[i] + T.float32(1)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(256):
                        C[i] = A[i] + T.float32(2)

    mod = tvm.IRModule({"simtvf_two_threads_demo": simtvf_two_threads_demo})

    print_stage("Raw PrimFunc", mod, "simtvf_two_threads_demo")
    raw_blocks = collect_simtvf_blocks(mod["simtvf_two_threads_demo"])
    assert len(raw_blocks) == 2, "raw stage should contain exactly two SimtVF blocks"
    raw_tags = collect_thread_extent_tags_outside_simtvf(mod["simtvf_two_threads_demo"].body)
    assert "blockIdx.x" in raw_tags, "Raw stage should keep blockIdx launch binding"
    assert not any(tag.startswith("threadIdx.") for tag in raw_tags), "Raw stage should not have outer threadIdx launch bindings on ascend"

    with target:
        PreLowerSemanticCheck(mod)
        lowered = LowerAndLegalize(mod, target)
        print_stage("After LowerAndLegalize", lowered, "simtvf_two_threads_demo")
        lowered_blocks = collect_simtvf_blocks(lowered["simtvf_two_threads_demo"])
        assert len(lowered_blocks) == 2, "LowerAndLegalize should keep two SimtVF blocks"

        optimized = OptimizeForTarget(lowered, target)
        print_stage("After OptimizeForTarget", optimized, "simtvf_two_threads_demo")
        optimized_blocks = collect_simtvf_blocks(optimized["simtvf_two_threads_demo"])
        assert len(optimized_blocks) == 2, "OptimizeForTarget should keep two SimtVF blocks"

        assert all(block.name_hint == "SIMT_VF" for block in optimized_blocks), "OptimizeForTarget should preserve SIMT_VF name_hint"
        opt_func = optimized["simtvf_two_threads_demo"]
        opt_tags = collect_thread_extent_tags_outside_simtvf(opt_func.body)
        assert "blockIdx.x" in opt_tags, "OptimizeForTarget should keep blockIdx launch binding"
        thread_extent = opt_func.attrs.get("thread_extent") if opt_func.attrs else None
        if thread_extent:
            assert not any(str(tag).startswith("threadIdx.") for tag in thread_extent.keys()), (
                "OptimizeForTarget should not leave SIMT thread extent in func_attr on ascend"
            )
        for block in optimized_blocks:
            block_tags = collect_thread_extent_tags(block.body)
            assert "threadIdx.x" in block_tags, "Each SimtVF region should contain local threadIdx.x thread_extent"

    print("\n[ok] Dual-SimtVF (128/256) IR inspection finished.")


if __name__ == "__main__":
    main()
