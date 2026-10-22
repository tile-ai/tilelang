import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
def test_host_clz_cuda_graph_capture(dtype):
    bits = int(dtype.lstrip("uint"))

    @T.prim_func
    def main(B: T.Tensor((65,), "int32"), d: T.int64):
        with T.Kernel(T.clz(T.cast(d, dtype)) + 1, threads=32) as bx:
            B[bx] = bx

    kernel = tilelang.compile(main, target="cuda", target_host="c", execution_backend="tvm_ffi")
    assert f"tl_clz{bits}" in kernel.get_host_source()
    out = torch.empty(65, device="cuda", dtype=torch.int32)
    values = [0, 1, 2, 3, (1 << (bits - 1)) - 1]
    values += [-1, -(1 << (bits - 1))] if dtype.startswith("int") else [1 << (bits - 1), (1 << bits) - 1]
    graphs = []
    for value in values:
        count = bits - (value % (1 << bits)).bit_length() + 1
        expected = torch.full_like(out, -1)
        expected[:count] = torch.arange(count, device="cuda", dtype=torch.int32)
        argument = value if value < (1 << 63) else value - (1 << 64)
        out.fill_(-1)
        kernel(out, argument)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel(out, argument)
        graphs.append((graph, expected))
    # Each graph keeps its capture-time scalar and launch extent, even after
    # ordinary calls and captures with other values. No host code runs on replay.
    for graph, expected in reversed(graphs):
        kernel(out, 0)
        out.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
