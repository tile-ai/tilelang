import os

import tilelang
import tilelang.testing
import tilelang.language as T

_PRINT = os.environ.get("TILELANG_REDUCE_MAXMIN_NAN_PRINT") in ("1", "true", "yes")


def _compile_cuda(prim_func, *, pass_configs=None):
    return tilelang.compile(
        prim_func,
        out_idx=-1,
        target="cuda",
        pass_configs=pass_configs or {},
    )


# --- reduce_max ---


def _kernel_reduce_max(length: int, dtype):
    @T.prim_func
    def kernel(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            T.reduce_max(frag, out_frag)
            T.copy(out_frag, out)

    return kernel


def test_reduce_max_fp16_nan_propagate_default():
    k = _compile_cuda(_kernel_reduce_max(64, T.float16))
    src = k.get_kernel_source()
    if _PRINT:
        print(k.adapter.prim_func.script())
        print(src)
    assert "tl::MaxOp" in src
    assert "MaxOpNan" not in src


def test_reduce_max_fp16_nan_propagate_false():
    k = _compile_cuda(
        _kernel_reduce_max(64, T.float16),
        pass_configs={tilelang.PassConfigKey.TL_REDUCE_MAXMIN_NAN_PROPAGATE: False},
    )
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src
    assert "__hmax_nan" in src


def test_reduce_max_bf16_nan_propagate_default():
    k = _compile_cuda(_kernel_reduce_max(64, T.bfloat16))
    src = k.get_kernel_source()
    assert "tl::MaxOp" in src
    assert "__hmax(" in src


def test_reduce_max_bf16_nan_propagate_false():
    k = _compile_cuda(
        _kernel_reduce_max(64, T.bfloat16),
        pass_configs={tilelang.PassConfigKey.TL_REDUCE_MAXMIN_NAN_PROPAGATE: False},
    )
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src
    assert "__hmax_nan" in src


# --- reduce_min ---


def _kernel_reduce_min(length: int, dtype):
    @T.prim_func
    def kernel(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            T.reduce_min(frag, out_frag)
            T.copy(out_frag, out)

    return kernel


def test_reduce_min_fp16_nan_propagate_default():
    k = _compile_cuda(_kernel_reduce_min(64, T.float16))
    src = k.get_kernel_source()
    assert "tl::MinOp" in src
    assert "MinOpNan" not in src


def test_reduce_min_fp16_nan_propagate_false():
    k = _compile_cuda(
        _kernel_reduce_min(64, T.float16),
        pass_configs={tilelang.PassConfigKey.TL_REDUCE_MAXMIN_NAN_PROPAGATE: False},
    )
    src = k.get_kernel_source()
    assert "tl::MinOpNan" in src
    assert "__hmin_nan" in src


def test_reduce_min_bf16_nan_propagate_false():
    k = _compile_cuda(
        _kernel_reduce_min(64, T.bfloat16),
        pass_configs={tilelang.PassConfigKey.TL_REDUCE_MAXMIN_NAN_PROPAGATE: False},
    )
    src = k.get_kernel_source()
    assert "tl::MinOpNan" in src
    assert "__hmin_nan" in src


# --- reduce_absmax (uses MaxOp / MaxOpNan like reduce_max) ---


def _kernel_reduce_absmax(length: int, dtype):
    @T.prim_func
    def kernel(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            T.reduce_absmax(frag, out_frag)
            T.copy(out_frag, out)

    return kernel


def test_reduce_absmax_fp16_nan_propagate_default():
    k = _compile_cuda(_kernel_reduce_absmax(64, T.float16))
    src = k.get_kernel_source()
    assert "tl::MaxOp" in src


def test_reduce_absmax_fp16_nan_propagate_false():
    k = _compile_cuda(
        _kernel_reduce_absmax(64, T.float16),
        pass_configs={tilelang.PassConfigKey.TL_REDUCE_MAXMIN_NAN_PROPAGATE: False},
    )
    src = k.get_kernel_source()
    assert "tl::MaxOpNan" in src
    assert "__hmax_nan" in src


if __name__ == "__main__":
    tilelang.testing.main()
