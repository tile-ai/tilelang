import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def test_ascend_copy_gm_to_ub_outside_simtvf():
    @T.prim_func
    def gm_to_ub_kernel(A: T.Buffer((256,), "float32")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            T.copy(A[:256], temp)

    artifact = lower(gm_to_ub_kernel, target="ascend")
    source = artifact.kernel_source
    print("=== test_ascend_copy_gm_to_ub_outside_simtvf ===")
    print(source)

    assert "asc_copy_gm2ub_align" in source, "Outside-SimtVF GM->UB copy should generate the aligned C API DMA call"
    assert "__simt_vf__" not in source, "No SimtVF region in this kernel — should not emit __simt_vf__"
    print("[PASS] test_ascend_copy_gm_to_ub_outside_simtvf\n")


def test_ascend_copy_ub_to_gm_outside_simtvf():
    @T.prim_func
    def ub_to_gm_kernel(
        A: T.Buffer((256,), "float32"),
        B: T.Buffer((256,), "float32"),
    ):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            T.copy(A[:256], temp)
            T.copy(temp, B[:256])

    artifact = lower(ub_to_gm_kernel, target="ascend")
    source = artifact.kernel_source
    print("=== test_ascend_copy_ub_to_gm_outside_simtvf ===")
    print(source)

    assert "asc_copy_gm2ub_align" in source, "Should have GM->UB DMA for loading A into temp"
    assert "asc_copy_ub2gm_align" in source, "Outside-SimtVF UB->GM copy should generate the aligned C API DMA call"
    print("[PASS] test_ascend_copy_ub_to_gm_outside_simtvf\n")


def test_ascend_copy_inside_simtvf():
    @T.prim_func
    def inside_simtvf_kernel(A: T.Buffer((256,), "float32")):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            with T.SimtVF(threads=128):
                T.copy(A[:256], temp)

    artifact = lower(inside_simtvf_kernel, target="ascend")
    source = artifact.kernel_source
    print("=== test_ascend_copy_inside_simtvf ===")
    print(source)

    assert "__simt_vf__" in source, "Should emit __simt_vf__ helper for the SimtVF region"
    # simt_vf_0 helper must use per-thread access, not DMA
    vf_start = source.find("simt_vf_0")
    assert vf_start >= 0, "Should have simt_vf_0 helper function"
    vf_body = source[vf_start : source.find("\nextern", vf_start)]
    assert "asc_copy_gm2ub_align" not in vf_body, "Inside-SimtVF copy should use per-thread access rather than DMA"
    assert "threadIdx.x" in source, "Inside-SimtVF copy should use threadIdx.x for per-thread indexing"
    print("[PASS] test_ascend_copy_inside_simtvf\n")


def test_ascend_copy_mixed_outside_dma_inside_simt():
    @T.prim_func
    def mixed_copy_kernel(
        A: T.Buffer((256,), "float32"),
        B: T.Buffer((256,), "float32"),
    ):
        with T.Kernel(1) as _:
            temp = T.alloc_shared((256,), "float32")
            T.copy(A[:256], temp)
            with T.SimtVF(threads=128):
                T.copy(A[:256], temp)
                for i in T.Parallel(256):
                    temp[i] = A[i] * B[i]

    artifact = lower(mixed_copy_kernel, target="ascend")
    source = artifact.kernel_source
    print("=== test_ascend_copy_mixed_outside_dma_inside_simt ===")
    print(source)

    assert "asc_copy_gm2ub_align" in source, "Outside-SimtVF copy should generate the aligned C API DMA call"
    assert "__simt_vf__" in source, "Should emit __simt_vf__ for the SimtVF region"
    assert "asc_vf_call" in source, "Should emit asc_vf_call to invoke the SimtVF helper"
    print("[PASS] test_ascend_copy_mixed_outside_dma_inside_simt\n")


if __name__ == "__main__":
    tilelang.testing.main()
