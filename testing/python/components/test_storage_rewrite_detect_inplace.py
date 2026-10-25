import tilelang
import tilelang.testing
from tilelang import language as T, tvm


@tilelang.jit
def _compile_kernel_without_inplace():
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float]):
        with T.Kernel(num_tokens, threads=32) as pid:
            read = T.alloc_local((1,), T.int)
            read[0] = x[pid]

            write = T.alloc_local((1,), T.int)
            write[0] = read[0] * 2
            x[pid] = write[0]

    return buggy_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_STORAGE_REWRITE_DETECT_INPLACE: True,
    },
)
def _compile_kernel_with_inplace():
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float]):
        with T.Kernel(num_tokens, threads=32) as pid:
            read = T.alloc_local((1,), T.int)
            read[0] = x[pid]

            write = T.alloc_local((1,), T.int)
            write[0] = read[0] * 2
            x[pid] = write[0]

    return buggy_kernel


def _get_device_kernel_script(detect_inplace: bool) -> str:
    kernel = _compile_kernel_with_inplace if detect_inplace else _compile_kernel_without_inplace
    pass_configs = {}
    if detect_inplace:
        pass_configs[tilelang.PassConfigKey.TL_STORAGE_REWRITE_DETECT_INPLACE] = True
    with tvm.transform.PassContext(config=pass_configs):
        artifact = tilelang.lower(kernel.get_tir(), target="cuda")
    return artifact.kernel_source


@tilelang.testing.requires_cuda
def test_storage_rewrite_detect_inplace_toggle():
    script_off = _get_device_kernel_script(detect_inplace=False)
    script_on = _get_device_kernel_script(detect_inplace=True)

    pattern_on = "read_ptr[0] = (read_ptr[0] * 2);"
    pattern_off = "write_ptr[0] = (read_ptr[0] * 2);"
    assert script_off.count(pattern_on) == 0, f"inplace pattern found when disabled:\n{script_off}"
    assert script_on.count(pattern_on) > 0, f"inplace pattern not found when enabled:\n{script_on}"
    assert script_off.count(pattern_off) > 0, f"separate-write pattern not found when disabled:\n{script_off}"


if __name__ == "__main__":
    tilelang.testing.main()
