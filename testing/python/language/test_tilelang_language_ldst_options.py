import tilelang
import tilelang.language as T
import torch
import tilelang.testing


@tilelang.jit
def get_ld_kernel(scope, sem, na, nc):

    @T.prim_func
    def main(x: T.Tensor((32), "int32"), y: T.Tensor((32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            reg = T.alloc_var('int32')
            T.ld(x[tx], reg, scope=scope, sem=sem, na=na, nc=nc)
            y[tx] = reg

    return main


@tilelang.jit
def get_st_kernel(scope, sem, na):

    @T.prim_func
    def main(x: T.Tensor((32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.st(x[tx], tx, scope=scope, sem=sem, na=na)

    return main


def _test_ld_options(scope, sem, na, nc):
    kernel = get_ld_kernel(scope, sem, na, nc)
    x = torch.randint(0, 100, (32,), device="cuda", dtype=torch.int32)
    y = torch.zeros_like(x)
    kernel(x, y)
    assert torch.equal(x, y)


@tilelang.testing.requires_cuda
def test_ld_options():
    # ld.acquire.sys.global.s32 / u64
    _test_ld_options(scope="sys", sem="acquire", na=False, nc=False)

    # ld.acquire.gpu.global.s32
    _test_ld_options(scope="gpu", sem="acquire", na=False, nc=False)

    # ld.acquire.cta.s32
    _test_ld_options(scope="cta", sem="acquire", na=False, nc=False)

    # ld.relaxed.gpu.global.L1::no_allocate.b8/b16/b32/b64
    _test_ld_options(scope="gpu", sem="relaxed", na=True, nc=False)

    # ld.volatile.global.s32/f32/s64/u64
    _test_ld_options(scope="gpu", sem="volatile", na=False, nc=False)

    # ld.global.nc.L1::no_allocate.L2::256B (or ld.volatile.global when DISABLE_AGGRESSIVE_PTX_INSTRS)
    _test_ld_options(scope="gpu", sem="weak", na=True, nc=True)


def _test_st_options(scope, sem, na):
    kernel = get_st_kernel(scope, sem, na)
    x = torch.randint(0, 100, (32,), device="cuda", dtype=torch.int32)
    kernel(x)
    assert x.equal(torch.arange(32, device="cuda"))


@tilelang.testing.requires_cuda
def test_st_options():
    # st.relaxed.sys.global.s32
    _test_st_options("sys", "relaxed", False)

    # st.release.sys.global.s32
    _test_st_options("sys", "release", False)

    # st.release.cta.s32
    _test_st_options("cta", "release", False)

    # st.relaxed.gpu.global.L1::no_allocate.b*
    _test_st_options("gpu", "relaxed", True)

    # st.release.gpu.global.L1::no_allocate.b*
    _test_st_options("gpu", "release", True)

    _test_st_options("gpu", "weak", False)
    _test_st_options("gpu", "weak", True)


if __name__ == "__main__":
    tilelang.testing.main()
