import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_language_ldg_codegen():
    N = 128

    @T.prim_func
    def main(
        x: T.Tensor((N,), T.float32),
        y: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N, threads=32) as pid:
            # Explicitly request read-only cache load for x[pid]
            y[pid] = T.__ldg(x[pid]) + 1.0

    # Compile for CUDA and retrieve generated CUDA source
    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print(src)
    # Assert that codegen uses __ldg on CUDA backend
    # We look for the intrinsic call with address-of argument
    assert "__ldg(" in src, "Expected __ldg call in generated CUDA source"
    assert "__ldg(&" in src or "__ldg(&(" in src, "Expected address-of form in __ldg call"


if __name__ == "__main__":
    tilelang.testing.main()
