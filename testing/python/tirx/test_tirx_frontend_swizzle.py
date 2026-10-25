import pytest
import torch

import tilelang
import tilelang.language as T


@pytest.mark.parametrize("order", ["row", "column"])
@pytest.mark.parametrize("panel", [1, 3, 10])
@pytest.mark.parametrize("cluster", [None, (2, 1, 1), (1, 2, 1)])
def test_native_threadblock_swizzle_preserves_grid_and_cluster(order, panel, cluster):
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")
    if cluster is not None and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Threadblock clusters require SM90 or newer")
    cluster_x, cluster_y, _ = cluster or (1, 1, 1)
    columns, rows, depth = 7 * cluster_x, 5 * cluster_y, 2
    launch_options = {} if cluster is None else {"cluster_dims": cluster}

    @T.prim_func
    def kernel(output: T.Tensor((depth, rows, columns), "int32")):
        with T.Kernel(columns, rows, depth, threads=32, **launch_options) as (block_x, block_y, block_z):
            T.import_source(
                'extern "C" __device__ int physical_block_x() { return blockIdx.x; }\n'
                'extern "C" __device__ int physical_block_y() { return blockIdx.y; }\n'
            )
            T.use_swizzle(panel, order=order)
            if T.get_thread_binding() == 0:
                output[block_z, T.call_extern("int32", "physical_block_y"), T.call_extern("int32", "physical_block_x")] = (
                    block_x + block_y * columns
                )

    compiled = tilelang.compile(kernel, target="cuda")
    assert "swizzled_block_" in compiled.get_kernel_source()
    output = torch.full((depth, rows, columns), -1, device="cuda", dtype=torch.int32)
    compiled(output)
    expected = torch.empty((depth, rows, columns), dtype=torch.int32)
    for block_y in range(rows):
        for block_x in range(columns):
            linear = block_x // cluster_x + block_y // cluster_y * 7
            panel_index, panel_offset = divmod(linear, panel * (5 if order == "column" else 7))
            stride = min(panel, (7 if order == "column" else 5) - panel_index * panel)
            major, minor = divmod(panel_offset, stride)
            if panel_index % 2:
                major = (5 if order == "column" else 7) - 1 - major
            minor += panel_index * panel
            logical_x, logical_y = (minor, major) if order == "column" else (major, minor)
            logical_x = logical_x * cluster_x + block_x % cluster_x
            logical_y = logical_y * cluster_y + block_y % cluster_y
            expected[:, block_y, block_x] = logical_x + logical_y * columns
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    for plane in output.cpu():
        torch.testing.assert_close(plane.flatten().sort().values, torch.arange(rows * columns, dtype=torch.int32))


@pytest.mark.parametrize("order,panel,diagnostic", [("mlx", 3, "Unsupported CUDA"), ("row", 0, "positive constant")])
def test_native_swizzle_rejects_unsupported_configuration(order, panel, diagnostic):
    @T.prim_func
    def kernel(output: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=32):
            T.use_swizzle(panel, order=order)
            output[0] = 1

    with pytest.raises(ValueError, match=diagnostic):
        tilelang.lower(kernel, target="cuda -arch=sm_90")
