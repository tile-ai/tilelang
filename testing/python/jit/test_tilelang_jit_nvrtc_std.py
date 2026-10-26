import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("standard", ["c++17", "c++20"])
def test_nvrtc_is_integral(standard):
    pytest.importorskip("cuda.bindings.nvrtc")
    from tilelang.contrib.nvrtc import compile_cuda
    from tilelang.env import TILELANG_TEMPLATE_PATH

    integral_types = [
        "bool",
        "char",
        "signed char",
        "unsigned char",
        "wchar_t",
        "char16_t",
        "char32_t",
        "short",
        "unsigned short",
        "int",
        "unsigned int",
        "long",
        "unsigned long",
        "long long",
        "unsigned long long",
    ]
    if standard == "c++20":
        integral_types.append("char8_t")
    assertions = []
    for dtype in integral_types:
        for cv in ("", "const ", "volatile ", "const volatile "):
            assertions.append(f"static_assert(std::is_integral<{cv}{dtype}>::value);")
            assertions.append(f"static_assert(std::is_integral_v<{cv}{dtype}>);")
    for dtype in ("float", "double", "const float", "void", "int*", "int&", "int&&", "int[2]", "Enum", "Struct"):
        assertions.append(f"static_assert(!std::is_integral<{dtype}>::value);")
        assertions.append(f"static_assert(!std::is_integral_v<{dtype}>);")
    source = "enum Enum { value }; struct Struct {};\n" + "\n".join(assertions)
    source += '\nextern "C" __global__ void main_kernel() {}'
    assert compile_cuda(source, arch=80, options=[f"-std={standard}", f"-I{TILELANG_TEMPLATE_PATH}"], verbose=True)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("dtype", ["int16", "int32", "uint32", "float32"])
@pytest.mark.parametrize("op", ["sum", "max", "min"])
def test_nvrtc_warp_reduce(dtype, op):
    pytest.importorskip("cuda.bindings.nvrtc")
    reduce = getattr(T, f"warp_reduce_{op}")

    @T.prim_func
    def main(A: T.Tensor((32,), dtype), O: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            tid = T.get_thread_binding()
            O[tid] = reduce(A[tid])

    kernel = tilelang.compile(main, out_idx=[1], execution_backend="nvrtc")
    values = torch.arange(32, device="cuda", dtype=torch.int32)
    a = values.to(getattr(torch, dtype))
    expected = getattr(values, op)().to(a.dtype).expand_as(a)
    out = kernel(a, stream=torch.cuda.current_stream().cuda_stream)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
