import re

import numpy as np
import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm


def _matmul(rows, columns, reduction, dtype):
    @T.prim_func
    def main(
        input_left: T.Tensor((rows, reduction), dtype),
        input_right: T.Tensor((reduction, columns), dtype),
        output: T.Tensor((rows, columns), dtype),
    ):
        with T.Kernel(T.ceildiv(columns, 16), T.ceildiv(rows, 16), threads=128) as (block_x, block_y):
            left_shared = T.alloc_shared((16, 16), dtype)
            right_shared = T.alloc_shared((16, 16), dtype)
            accumulator = T.alloc_fragment((16, 16), T.float32)
            T.clear(accumulator)
            for outer in T.Pipelined(T.ceildiv(reduction, 16), num_stages=0):
                T.copy(input_left[block_y * 16, outer * 16], left_shared, coalesced_width=2)
                T.copy(input_right[outer * 16, block_x * 16], right_shared, coalesced_width=2)
                for row, column in T.Parallel(16, 16):
                    for inner in T.serial(16):
                        accumulator[row, column] += T.cast(left_shared[row, inner], T.float32) * T.cast(
                            right_shared[inner, column], T.float32
                        )
            T.copy(accumulator, output[block_y * 16, block_x * 16], coalesced_width=2)

    return main


@pytest.fixture(scope="module")
def webgpu_device():
    wgpu = pytest.importorskip("wgpu")
    adapters = [
        adapter
        for adapter in wgpu.gpu.enumerate_adapters_sync()
        if adapter.info["backend_type"] == "Vulkan" and "shader-f16" in adapter.features
    ]
    if not adapters:
        pytest.skip("A Vulkan WebGPU adapter with shader-f16 is unavailable")
    discrete = [adapter for adapter in adapters if adapter.info["adapter_type"] == "DiscreteGPU"]
    adapter = discrete[min(5, len(discrete) - 1)] if discrete else adapters[0]
    return adapter.request_device_sync(required_features=["shader-f16"])


def _execute(device, source, left, right, shape):
    import wgpu

    shader = device.create_shader_module(code=source)
    entry = re.search(r"\bfn\s+(\w+)\(", source).group(1)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": shader, "entry_point": entry})
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST

    def make_buffer(values):
        alignment = 4 // values.dtype.itemsize
        values = np.pad(values.reshape(-1), (0, (-values.size) % alignment))
        return device.create_buffer_with_data(data=values, usage=usage)

    buffers = {
        "input_left_ptr": make_buffer(left),
        "input_right_ptr": make_buffer(right),
        "output_ptr": make_buffer(np.zeros(shape, dtype=left.dtype)),
    }
    declarations = re.findall(r"@binding\((\d+)\) var<storage[^>]*> (\w+)", source)
    assert {name for _, name in declarations} == set(buffers)
    entries = [{"binding": int(binding), "resource": {"buffer": buffers[name]}} for binding, name in declarations]
    grid_x = (shape[1] + 15) // 16
    uniform = device.create_buffer_with_data(data=np.array([grid_x, 0, 0, 0], dtype="uint32"), usage=wgpu.BufferUsage.UNIFORM)
    uniform_binding = int(re.search(r"@binding\((\d+)\) var<uniform>", source).group(1))
    entries.append({"binding": uniform_binding, "resource": {"buffer": uniform}})
    group = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
    encoder = device.create_command_encoder()
    compute = encoder.begin_compute_pass()
    compute.set_pipeline(pipeline)
    compute.set_bind_group(0, group)
    compute.dispatch_workgroups(grid_x, (shape[0] + 15) // 16)
    compute.end()
    device.queue.submit([encoder.finish()])
    return np.frombuffer(device.queue.read_buffer(buffers["output_ptr"]), dtype=left.dtype, count=shape[0] * shape[1]).reshape(shape).copy()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(16, 16, 16), (32, 48, 64), (19, 23, 21)])
def test_parallel_gemm_numerically_matches_numpy(webgpu_device, dtype, shape):
    rows, columns, reduction = shape
    function = _matmul(rows, columns, reduction, dtype)
    with tvm.transform.PassContext(), tvm.target.Target("webgpu"):
        artifact = tilelang.lower(function, target="webgpu")
    generator = np.random.default_rng(37)
    left = generator.integers(-2, 3, size=(rows, reduction)).astype(dtype)
    right = generator.integers(-2, 3, size=(reduction, columns)).astype(dtype)
    expected = (left.astype("float32") @ right.astype("float32")).astype(dtype)
    actual = _execute(webgpu_device, artifact.kernel_source, left, right, expected.shape)
    np.testing.assert_array_equal(actual, expected)
