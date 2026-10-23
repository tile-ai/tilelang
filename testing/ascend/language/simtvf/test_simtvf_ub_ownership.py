"""Ownership selection across static GM/UB/fragment paths and native fallbacks."""

import pytest
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def elementwise_kernel(n=1048576, tile=8192, threads=2048, blocks=64, affine=False, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(blocks) as block:
            a = T.alloc_shared((tile,), dtype)
            b = T.alloc_shared((tile,), dtype)
            c = T.alloc_shared((tile,), dtype)
            for k in T.serial(n // (blocks * tile)):
                offset = (k * blocks + block) * tile
                T.copy(A[offset : offset + tile], a)
                T.copy(B[offset : offset + tile], b)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile):
                        if affine:
                            c[i] = a[i] * T.float32(2.0) + b[i]
                        else:
                            c[i] = a[i] + b[i]
                T.copy(c, C[offset : offset + tile])

    return main


def fragment_elementwise_kernel(tile=8192, threads=128, blocks=1, dtype="float16", iterations=1, num_stages=0):
    n = tile * blocks * iterations

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(blocks) as block:
            a = T.alloc_shared((tile,), dtype)
            b = T.alloc_shared((tile,), dtype)
            c = T.alloc_shared((tile,), dtype)
            for step in T.Pipelined(iterations, num_stages=num_stages):
                offset = (step * blocks + block) * tile
                T.copy(A[offset : offset + tile], a)
                T.copy(B[offset : offset + tile], b)
                with T.SimtVF(threads=threads):
                    af = T.alloc_fragment((tile,), dtype)
                    bf = T.alloc_fragment((tile,), dtype)
                    cf = T.alloc_fragment((tile,), dtype)
                    T.copy(a, af)
                    T.copy(b, bf)
                    for i in T.Parallel(tile):
                        cf[i] = af[i] + bf[i]
                    T.copy(cf, c)
                T.copy(c, C[offset : offset + tile])

    return main


@pytest.mark.parametrize("blocks,threads,iterations,num_stages", [(1, 128, 1, 0), (32, 128, 3, 0), (64, 2048, 4, 2)])
def test_ownership_gm_ub_fragment_full_lowering(blocks, threads, iterations, num_stages):
    snapshots = {}

    @tilelang.tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLayoutInference":
                snapshots["inferred"] = mod

    with tilelang.tvm.target.Target("ascend"), tilelang.transform.PassContext(instruments=[Capture()]):
        lowered = tilelang.lower(
            fragment_elementwise_kernel(threads=threads, blocks=blocks, iterations=iterations, num_stages=num_stages), target="ascend"
        )
    assert "__simt_vf__" in lowered.kernel_source
    from testing.ascend._ir import nodes
    from tvm import tirx

    layouts = [
        n.annotations["parallel_loop_layout"] for n in nodes(snapshots["inferred"], tirx.For) if "parallel_loop_layout" in n.annotations
    ]
    assert len(layouts) == 1
    analyzer = tilelang.tvm.arith.Analyzer()
    i = tirx.Var("i", "int32")
    analyzer.bind(i, tilelang.tvm.ir.Range(0, 8192))
    assert analyzer.can_prove_equal(layouts[0].map_forward_thread([i])[0], (i // 2) % threads)


def test_ownership_pure_ub_full_lowering():
    @T.prim_func
    def main():
        with T.Kernel(1):
            a = T.alloc_shared((1024,), "float16")
            c = T.alloc_shared((1024,), "float16")
            with T.SimtVF(threads=128):
                for i in T.Parallel(1024):
                    a[i] = T.float16(i)
                for i in T.Parallel(1024):
                    c[i] = a[i] + T.float16(1)

    layouts = []

    @tilelang.tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLayoutInference":
                from testing.ascend._ir import nodes
                from tvm import tirx

                layouts.extend(
                    n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For) if "parallel_loop_layout" in n.annotations
                )

    with tilelang.tvm.target.Target("ascend"), tilelang.transform.PassContext(instruments=[Capture()]):
        lowered = tilelang.lower(main, target="ascend")
    assert "__simt_vf__" in lowered.kernel_source
    assert len(layouts) == 2
    analyzer = tilelang.tvm.arith.Analyzer()
    for layout in layouts:
        for i in range(1024):
            assert int(analyzer.simplify(layout.map_forward_thread([i])[0])) == (i // 2) % 128


def test_ownership_mixed_kernel_full_lowering():
    with tilelang.tvm.target.Target("ascend"):
        lowered = tilelang.lower(elementwise_kernel(), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "int32"])
@pytest.mark.parametrize("tile,threads", [(258, 128), (1024, 33)])
def test_ownership_types_and_tails_full_lowering(dtype, tile, threads):
    snapshots = {}

    @tilelang.tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLayoutInference":
                snapshots["inferred"] = mod

    with (
        tilelang.tvm.target.Target("ascend"),
        tilelang.transform.PassContext(instruments=[Capture()]),
    ):
        lowered = tilelang.lower(elementwise_kernel(n=tile, tile=tile, threads=threads, blocks=1, dtype=dtype), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source
    assert "if (" in lowered.kernel_source
    from testing.ascend._ir import nodes
    from tilelang import tvm
    from tvm import tirx

    layouts = [
        loop.annotations["parallel_loop_layout"]
        for loop in nodes(snapshots["inferred"], tirx.For)
        if "parallel_loop_layout" in loop.annotations
    ]
    assert len(layouts) == 1
    analyzer = tvm.arith.Analyzer()
    # Width is inferred, including packed f16/bf16 tails. Every logical
    # element must still have a distinct valid (thread, slot) assignment.
    assignments = set()
    for index in range(tile):
        thread = int(analyzer.simplify(layouts[0].map_forward_thread([index])[0]))
        slots = tuple(int(analyzer.simplify(x)) for x in layouts[0].map_forward_index([index]))
        assert 0 <= thread < threads
        assignments.add((thread, slots))
    assert len(assignments) == tile


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("blocks,threads,iterations,num_stages", [(1, 128, 1, 0), (32, 128, 3, 0), (64, 2048, 4, 2)])
def test_ownership_gm_ub_fragment_numerical(blocks, threads, iterations, num_stages):
    import torch

    n = 8192 * blocks * iterations
    compiled = tilelang.compile(
        fragment_elementwise_kernel(blocks=blocks, threads=threads, iterations=iterations, num_stages=num_stages),
        target="ascend",
        out_idx=[2],
    )
    a = ((torch.arange(n) % 31) - 15).to(torch.float16).npu()
    b = ((torch.arange(n) % 17) - 8).to(torch.float16).npu()
    torch.testing.assert_close(compiled(a, b).cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("threads", [128, 2048])
def test_ownership_numerical(threads):
    import torch

    n = 8192
    compiled = tilelang.compile(
        elementwise_kernel(n=n, blocks=1, threads=threads),
        target="ascend",
        out_idx=[2],
    )
    a = (torch.arange(n, dtype=torch.float32) % 251 - 125).npu()
    b = (torch.arange(n, dtype=torch.float32) % 67 - 33).npu()
    torch.testing.assert_close(compiled(a, b).cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)
