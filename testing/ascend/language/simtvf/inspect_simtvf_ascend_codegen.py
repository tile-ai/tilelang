import tilelang.language as T
from tilelang.engine.lower import lower


def main():
    @T.prim_func
    def simtvf_codegen_demo(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1) as _, T.SimtVF(threads=128):
            for i in T.Parallel(16):
                B[i] = A[i] + T.float32(1)
                T.tvm_storage_sync("shared")

    artifact = lower(simtvf_codegen_demo, target="ascend")
    source = artifact.kernel_source
    print(source)

    assert "__simt_vf__" in source, "Ascend source should mark SimtVF region"
    assert "__global__ __vector__ void simtvf_codegen_demo_kernel" in source, (
        "Ascend source should use '__global__ __vector__ <ret> <name>' signature order"
    )
    assert "asc_vf_call" in source, "Ascend source should emit asc_vf_call marker"
    assert "asc_syncthreads();" in source, "Ascend source should map tvm_storage_sync to asc_syncthreads()"

    @T.prim_func
    def simtvf_codegen_two_vf(
        A: T.Buffer((16,), "float32"),
        B: T.Buffer((16,), "float32"),
        C: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1) as _:
            with T.SimtVF(threads=64):
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)
            with T.SimtVF(threads=128):
                for i in T.Parallel(16):
                    C[i] = B[i] * T.float32(2)

    artifact_two = lower(simtvf_codegen_two_vf, target="ascend")
    source_two = artifact_two.kernel_source
    print(source_two)

    assert "__global__ __vector__ void simtvf_codegen_two_vf_kernel" in source_two, (
        "Two-VF kernel should keep '__global__ __vector__ <ret> <name>' signature order"
    )
    assert source_two.count("__simt_vf__ __launch_bounds__(") >= 2, "Two-VF source should emit two SimtVF helper functions"
    assert source_two.count("_simt_vf_") >= 4, "Two-VF source should emit namespaced helpers and asc_vf_call invocations"
    assert "__launch_bounds__(64)" in source_two and "__launch_bounds__(128)" in source_two, (
        "Two-VF source should preserve per-region thread bounds"
    )

    print("[ok] Ascend codegen smoke check passed.")


if __name__ == "__main__":
    main()
