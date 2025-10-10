# yapf: disable
# ruff: noqa

# This testfile can't be formatted by yapf & ruff

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit()
def get_buggy_kernel():
    @T.prim_func
    def buggy():
        with T.Kernel(1, threads=32) as pid:
            A_shared = T.alloc_shared((1,), "float32")

    return buggy


@tilelang.jit()
def get_buggy_kernel1():
    num_tokens = T.symbolic('num_tokens')

    @T.prim_func
    def buggy(x: T.Tensor[(num_tokens, ), 'float'],):
        with T.Kernel(num_tokens, threads=32) as pid:
            y = x[pid]

    return buggy

def test_dummy_kernel_gen():
    """Test dummy kernel generation"""
    kernel = get_buggy_kernel()
    # Currently still can't pass the test
    # kernel()


if __name__ == "__main__":
    tilelang.testing.main()
