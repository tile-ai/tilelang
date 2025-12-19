import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit
def get_kernel(scope, sem, na):

    @T.prim_func
    def main(x: T.Tensor((32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.st(x[tx], tx, scope=scope, sem=sem, na=na)

    return main


def test_st_options(scope, sem, na):
    kernel = get_kernel(scope, sem, na)
    x = torch.randint(0, 100, (32,), device="cuda", dtype=torch.int32)
    kernel(x)
    assert x.equal(torch.arange(32, device="cuda"))
    print(f'check passed for {scope=}.{sem=}.{na=} ✅')


if __name__ == "__main__":
    # from DeepEP all st instructions

    # st.relaxed.sys.global.s32
    test_st_options("sys", "relaxed", False)

    # # st.release.sys.global.s32
    test_st_options("sys", "release", False)

    # st.release.cta.s32
    test_st_options("cta", "release", False)

    # st.relaxed.gpu.global.L1::no_allocate.b*
    test_st_options("gpu", "relaxed", True)

    # st.release.gpu.global.L1::no_allocate.b*
    test_st_options("gpu", "release", True)

    # test_st_options("gpu", "weak", False)
    test_st_options("gpu", "weak", False)
    test_st_options("gpu", "weak", True)
