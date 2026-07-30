"""End-to-end regressions for software-pipeline dependency planning."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_issue_2595_atomic_write_is_not_reordered_after_dependent_read():
    """An atomic read/write access must constrain a later pipeline read."""

    n = 8

    @T.prim_func
    def main(
        data: T.Tensor((n + 1,), T.int32),
        output: T.Tensor((n,), T.int32),
    ):
        with T.Kernel(1, threads=1):
            staged = T.alloc_shared((1,), T.int32)
            for k in T.Pipelined(n, num_stages=2):
                T.atomic_add(data[k + 1], 100)
                staged[0] = data[k]
                output[k] = staged[0]

    kernel = tilelang.compile(main, out_idx=[1])
    data = torch.arange(n + 1, dtype=torch.int32, device="cuda")
    output = kernel(data)
    torch.cuda.synchronize()

    expected = torch.arange(n, dtype=torch.int32, device="cuda")
    expected[1:] += 100
    torch.testing.assert_close(output, expected)


@tilelang.testing.requires_cuda
def test_issue_2668_read_before_shared_overwrite_preserves_serial_result():
    """A loop-carried WAR lifecycle must not become an unawaited async copy."""

    width = 32
    num_tiles = 8

    @T.prim_func
    def main(
        source: T.Tensor((num_tiles, width), T.int32),
        initial: T.Tensor((width,), T.int32),
        output: T.Tensor((width,), T.int32),
    ):
        with T.Kernel(1, threads=width):
            shared = T.alloc_shared((width,), T.int32)
            accum = T.alloc_fragment((width,), T.int32)
            T.clear(accum)
            T.copy(initial, shared)
            for k in T.Pipelined(num_tiles, num_stages=2):
                for i in T.Parallel(width):
                    accum[i] += shared[i]
                T.copy(source[k, :], shared)
            T.copy(accum, output)

    kernel = tilelang.compile(
        main,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    torch.manual_seed(1)
    source = torch.randint(0, 5, (num_tiles, width), dtype=torch.int32, device="cuda")
    initial = torch.randint(100, 105, (width,), dtype=torch.int32, device="cuda")
    output = kernel(source, initial)
    torch.cuda.synchronize()

    expected = initial.to(torch.int64) + source[:-1].to(torch.int64).sum(0)
    torch.testing.assert_close(output.to(torch.int64), expected)


if __name__ == "__main__":
    tilelang.testing.main()
