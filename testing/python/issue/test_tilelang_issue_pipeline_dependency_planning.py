"""End-to-end regressions for software-pipeline dependency planning."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_ISSUE_2595_N = 8
_ISSUE_2668_WIDTH = 32
_ISSUE_2668_NUM_TILES = 8


@T.prim_func
def _issue_2595_kernel(
    data: T.Tensor((_ISSUE_2595_N + 1,), T.int32),
    output: T.Tensor((_ISSUE_2595_N,), T.int32),
):
    with T.Kernel(1, threads=1):
        staged = T.alloc_shared((1,), T.int32)
        for k in T.Pipelined(_ISSUE_2595_N, num_stages=2):
            T.atomic_add(data[k + 1], 100)
            staged[0] = data[k]
            output[k] = staged[0]


@T.prim_func
def _issue_2668_kernel(
    source: T.Tensor((_ISSUE_2668_NUM_TILES, _ISSUE_2668_WIDTH), T.int32),
    initial: T.Tensor((_ISSUE_2668_WIDTH,), T.int32),
    output: T.Tensor((_ISSUE_2668_WIDTH,), T.int32),
):
    with T.Kernel(1, threads=_ISSUE_2668_WIDTH):
        shared = T.alloc_shared((_ISSUE_2668_WIDTH,), T.int32)
        accum = T.alloc_fragment((_ISSUE_2668_WIDTH,), T.int32)
        T.clear(accum)
        T.copy(initial, shared)
        for k in T.Pipelined(_ISSUE_2668_NUM_TILES, num_stages=2):
            for i in T.Parallel(_ISSUE_2668_WIDTH):
                accum[i] += shared[i]
            T.copy(source[k, :], shared)
        T.copy(accum, output)


@tilelang.testing.requires_cuda
def test_issue_2595_atomic_write_is_not_reordered_after_dependent_read():
    """An atomic read/write access must constrain a later pipeline read."""

    kernel = tilelang.compile(_issue_2595_kernel, out_idx=[1])
    data = torch.arange(_ISSUE_2595_N + 1, dtype=torch.int32, device="cuda")
    output = kernel(data)
    torch.cuda.synchronize()

    expected = torch.arange(_ISSUE_2595_N, dtype=torch.int32, device="cuda")
    expected[1:] += 100
    torch.testing.assert_close(output, expected)


@tilelang.testing.requires_cuda
def test_issue_2668_read_before_shared_overwrite_preserves_serial_result():
    """A loop-carried WAR lifecycle must not become an unawaited async copy."""

    kernel = tilelang.compile(
        _issue_2668_kernel,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    torch.manual_seed(1)
    source = torch.randint(
        0,
        5,
        (_ISSUE_2668_NUM_TILES, _ISSUE_2668_WIDTH),
        dtype=torch.int32,
        device="cuda",
    )
    initial = torch.randint(100, 105, (_ISSUE_2668_WIDTH,), dtype=torch.int32, device="cuda")
    output = kernel(source, initial)
    torch.cuda.synchronize()

    expected = initial.to(torch.int64) + source[:-1].to(torch.int64).sum(0)
    torch.testing.assert_close(output.to(torch.int64), expected)


if __name__ == "__main__":
    tilelang.testing.main()
