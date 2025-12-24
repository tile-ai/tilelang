import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.jit
def _compile_kernel_without_inplace():
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float]):
        with T.Kernel(num_tokens, threads=32) as pid:
            read = T.alloc_var(T.int)
            read = x[pid]

            write = T.alloc_var(T.int)
            write = read * 2
            x[pid] = write

    return buggy_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_STORAGE_REWRITE_DETECT_INPLACE: True,
    },
)
def _compile_kernel_with_inplace():
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float]):
        with T.Kernel(num_tokens, threads=32) as pid:
            read = T.alloc_var(T.int)
            read = x[pid]

            write = T.alloc_var(T.int)
            write = read * 2
            x[pid] = write

    return buggy_kernel


def _get_device_kernel_script(detect_inplace: bool) -> str:
    if detect_inplace:
        kernel = _compile_kernel_with_inplace()
    else:
        kernel = _compile_kernel_without_inplace()
    source = kernel.get_kernel_source()
    return source


def test_storage_rewrite_detect_inplace_toggle():
    script_off = _get_device_kernel_script(detect_inplace=False)
    script_on = _get_device_kernel_script(detect_inplace=True)

    assert script_off.count("read = (read * 2);") == 0
    assert script_on.count("read = (read * 2);") > 0


if __name__ == "__main__":
    tilelang.testing.main()
