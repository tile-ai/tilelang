import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit
def get_kernel(scope, sem, na, nc):

    @T.prim_func
    def main(x: T.Tensor((32), "int32"), y: T.Tensor((32), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            reg = T.alloc_var('int32')
            T.ld(x[tx], reg, scope=scope, sem=sem, na=na, nc=nc)
            y[tx] = reg

    return main


def test_ld_options(scope, sem, na, nc):
    kernel = get_kernel(scope, sem, na, nc)
    x = torch.randint(0, 100, (32,), device="cuda", dtype=torch.int32)
    y = torch.zeros_like(x)
    kernel(x, y)
    assert torch.equal(x, y)
    print(f'check passed for {scope=}.{sem=}.{na=}.{nc=} ✅')


if __name__ == "__main__":
    # from DeepEP all ld instructions

    # ld.acquire.sys.global.s32 / u64
    test_ld_options(scope="sys", sem="acquire", na=False, nc=False)

    # ld.acquire.gpu.global.s32
    test_ld_options(scope="gpu", sem="acquire", na=False, nc=False)

    # ld.acquire.cta.s32
    test_ld_options(scope="cta", sem="acquire", na=False, nc=False)

    # ld.relaxed.gpu.global.L1::no_allocate.b8/b16/b32/b64
    test_ld_options(scope="gpu", sem="relaxed", na=True, nc=False)

    # ld.volatile.global.s32/f32/s64/u64
    test_ld_options(scope="gpu", sem="volatile", na=False, nc=False)

    # ld.global.nc.L1::no_allocate.L2::256B (or ld.volatile.global when DISABLE_AGGRESSIVE_PTX_INSTRS)
    test_ld_options(scope="gpu", sem="weak", na=True, nc=True)
