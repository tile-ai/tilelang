import pytest
import torch

import tilelang
import tilelang.language as language
import tilelang.testing
from tilelang import tvm
from tilelang.contrib import nvcc


_REDUCERS = {
    "sum": language.reduce_sum,
    "max": language.reduce_max,
    "min": language.reduce_min,
    "bitand": language.reduce_bitand,
    "bitor": language.reduce_bitor,
    "bitxor": language.reduce_bitxor,
}


def _make_kernel(operation, dtype, width, *, scale=1, batch=1, clear=True, thread_offset=0):
    rows = 4
    threads = max(32, width * scale + thread_offset)
    reduce_fn = _REDUCERS[operation]

    @language.prim_func
    def kernel(
        inputs: language.Tensor((rows, width, scale), dtype),
        outputs: language.Tensor((rows, scale), dtype),
    ):
        with language.Kernel(1, threads=threads):
            source = language.alloc_fragment((rows, width, scale), dtype)
            result = language.alloc_fragment((rows, scale), dtype)
            language.annotate_layout(
                {
                    source: language.Fragment(
                        source.shape,
                        forward_fn=lambda row, column, subgroup: (thread_offset + column * scale + subgroup, row),
                    ),
                    result: language.Fragment(
                        result.shape,
                        forward_fn=lambda row, subgroup, replica: (thread_offset + replica * scale + subgroup, row),
                        replicate=width,
                    ),
                }
            )
            language.copy(inputs, source)
            if not clear:
                language.fill(result, 1)
            reduce_fn(source, result, dim=1, clear=clear, batch=batch)
            language.copy(result, outputs)

    return kernel


def _compile_ptx(program, arch):
    if arch.startswith("sm_100") and nvcc.get_cuda_version() < (12, 8):
        pytest.skip("SM100 compilation requires CUDA 12.8+")
    if arch == "sm_100f" and nvcc.get_cuda_version() < (12, 9):
        pytest.skip("SM100f compilation requires CUDA 12.9+")
    if arch.startswith("sm_103") and nvcc.get_cuda_version() < (12, 9):
        pytest.skip("SM103 compilation requires CUDA 12.9+")
    target = tvm.target.Target({"kind": "cuda", "arch": arch})
    with tvm.transform.PassContext(), target:
        artifact = tilelang.lower(program, target=target)
    ptx = nvcc.compile_cuda(
        artifact.kernel_source,
        target_format="ptx",
        arch=arch,
        options=nvcc.default_compile_options(),
    )
    return artifact.kernel_source, ptx.decode()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("operation", "dtype", "width", "batch", "arch", "instruction"),
    [
        ("sum", "int32", 32, 1, "sm_80", "redux.sync.add"),
        ("max", "uint32", 64, 1, "sm_80", "redux.sync.max.u32"),
        ("min", "int32", 128, 2, "sm_80", "redux.sync.min.s32"),
        ("bitand", "int32", 32, 1, "sm_80", "redux.sync.and.b32"),
        ("bitor", "uint32", 32, 2, "sm_80", "redux.sync.or.b32"),
        ("bitxor", "int32", 64, 2, "sm_80", "redux.sync.xor.b32"),
        ("max", "float32", 32, 1, "sm_100a", "redux.sync.max.f32"),
        ("min", "float32", 128, 1, "sm_100a", "redux.sync.min.f32"),
        ("max", "float32", 64, 1, "sm_100f", "redux.sync.max.f32"),
        ("max", "float32", 32, 1, "sm_103a", "redux.sync.max.f32"),
        ("min", "float32", 128, 1, "sm_103a", "redux.sync.min.f32"),
        ("max", "float32", 64, 1, "sm_103f", "redux.sync.max.f32"),
    ],
)
def test_reduce_redux_ptx(operation, dtype, width, batch, arch, instruction):
    source, ptx = _compile_ptx(_make_kernel(operation, dtype, width, batch=batch), arch)
    assert "tl::AllReduce<" in source
    assert instruction in ptx
    assert "shfl.sync.bfly" not in ptx
    if width > 32:
        assert "bar.sync" in ptx


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("operation", "dtype", "width", "scale", "batch", "arch"),
    [
        ("sum", "int32", 32, 1, 1, "sm_75"),
        ("sum", "int32", 16, 1, 1, "sm_80"),
        ("sum", "int32", 32, 2, 1, "sm_80"),
        ("max", "float32", 32, 1, 1, "sm_90"),
        ("max", "float32", 32, 1, 1, "sm_100"),
        ("max", "float32", 32, 1, 1, "sm_103"),
        ("sum", "float32", 32, 1, 1, "sm_100a"),
        ("max", "int64", 32, 1, 1, "sm_100a"),
        ("max", "float64", 32, 1, 1, "sm_100a"),
        ("max", "float64", 32, 1, 1, "sm_103a"),
        ("max", "float16", 32, 1, 1, "sm_100a"),
        ("max", "float32", 32, 1, 2, "sm_100a"),
        ("max", "float16", 32, 1, 2, "sm_100a"),
        ("min", "bfloat16", 32, 1, 2, "sm_100a"),
    ],
)
def test_reduce_redux_fallback_ptx(operation, dtype, width, scale, batch, arch):
    _, ptx = _compile_ptx(_make_kernel(operation, dtype, width, scale=scale, batch=batch), arch)
    assert "redux.sync" not in ptx
    assert "shfl.sync.bfly" in ptx


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("thread_offset", [0, 16, 32])
@pytest.mark.parametrize("batch", [1, 2])
def test_reduce_redux_thread_offset_ptx(thread_offset, batch):
    source = f"""
#include <tl_templates/cuda/reduce.h>

extern "C" __global__ void reduce_offset(const int32_t* inputs, int32_t* outputs) {{
    if (threadIdx.x < {thread_offset} || threadIdx.x >= {thread_offset} + 32) {{
        return;
    }}
    using Reduction = tl::AllReduce<tl::SumOp, 32, 1, {thread_offset},
                                    tl::SyncThreadsBarrier, {batch}>;
    int32_t values[{batch}];
    #pragma unroll
    for (int index = 0; index < {batch}; ++index) {{
        values[index] = inputs[index * blockDim.x + threadIdx.x];
    }}
    if constexpr ({batch} == 1) {{
        values[0] = Reduction::run(values[0]);
    }} else {{
        Reduction::run_batch(values);
    }}
    #pragma unroll
    for (int index = 0; index < {batch}; ++index) {{
        outputs[index * blockDim.x + threadIdx.x] = values[index];
    }}
}}
"""
    ptx = nvcc.compile_cuda(source, target_format="ptx", arch="sm_80", options=nvcc.default_compile_options()).decode()
    if thread_offset % 32 == 0:
        assert "redux.sync.add" in ptx
        assert "shfl.sync.bfly" not in ptx
    else:
        assert "redux.sync" not in ptx
        assert "shfl.sync.bfly" in ptx


def _reference(inputs, operation, clear):
    values = inputs.cpu().to(torch.int64)
    if operation == "sum":
        result = values.sum(dim=1)
    elif operation == "max":
        result = values.amax(dim=1)
    elif operation == "min":
        result = values.amin(dim=1)
    else:
        combine = {"bitand": torch.bitwise_and, "bitor": torch.bitwise_or, "bitxor": torch.bitwise_xor}[operation]
        result = values[:, 0].clone()
        for column in range(1, values.shape[1]):
            result = combine(result, values[:, column])
    if not clear:
        initial = torch.ones_like(result)
        combine = {
            "sum": torch.add,
            "max": torch.maximum,
            "min": torch.minimum,
            "bitand": torch.bitwise_and,
            "bitor": torch.bitwise_or,
            "bitxor": torch.bitwise_xor,
        }[operation]
        result = combine(result, initial)
    return result.to(inputs.dtype).to(torch.int64)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("operation", _REDUCERS)
@pytest.mark.parametrize("dtype", ["int32", "uint32"])
@pytest.mark.parametrize(
    ("width", "batch", "clear", "execution_backend"),
    [(32, 1, True, "auto"), (64, 1, False, "auto"), (128, 2, True, "auto"), (64, 2, False, "auto"), (64, 2, False, "nvrtc")],
)
def test_reduce_redux_integer_runtime(operation, dtype, width, batch, clear, execution_backend):
    generator = torch.Generator().manual_seed(42)
    lower = -(2**31) if dtype == "int32" else 0
    upper = 2**31 if dtype == "int32" else 2**32
    inputs = torch.randint(lower, upper, (4, width, 1), generator=generator, dtype=torch.int64)
    inputs[:, 0, 0] = lower
    inputs[:, 1, 0] = upper - 1
    inputs = inputs.to(getattr(torch, dtype)).cuda()
    compiled = tilelang.compile(
        _make_kernel(operation, dtype, width, batch=batch, clear=clear), out_idx=-1, execution_backend=execution_backend
    )
    actual = compiled(inputs)
    torch.testing.assert_close(actual.cpu().to(torch.int64), _reference(inputs, operation, clear), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(("width", "scale", "thread_offset"), [(16, 1, 0), (32, 2, 0), (64, 1, 32)])
def test_reduce_redux_layout_runtime(width, scale, thread_offset):
    inputs = torch.arange(4 * width * scale, dtype=torch.int32).reshape(4, width, scale).cuda()
    compiled = tilelang.compile(_make_kernel("sum", "int32", width, scale=scale, thread_offset=thread_offset), out_idx=-1)
    torch.testing.assert_close(compiled(inputs).cpu().to(torch.int64), _reference(inputs, "sum", True), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("operation", ["max", "min"])
@pytest.mark.parametrize(("width", "execution_backend"), [(32, "auto"), (128, "auto"), (32, "nvrtc")])
def test_reduce_redux_float_runtime(operation, width, execution_backend):
    inputs = torch.zeros((4, width, 1), device="cuda")
    inputs[0, :, 0] = float("nan")
    inputs[0, 1, 0] = 3.0
    inputs[1, 0, 0] = float("inf")
    inputs[1, 1, 0] = -float("inf")
    inputs[2, :, 0] = -0.0
    inputs[3, :, 0] = -0.0
    inputs[3, 0, 0] = 0.0
    compiled = tilelang.compile(_make_kernel(operation, "float32", width), out_idx=-1, execution_backend=execution_backend)
    actual = compiled(inputs)
    expected = torch.tensor([[3.0], [float("inf") if operation == "max" else -float("inf")], [-0.0], [0.0]], device="cuda")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.signbit(actual[2]).all()


if __name__ == "__main__":
    tilelang.testing.main()
