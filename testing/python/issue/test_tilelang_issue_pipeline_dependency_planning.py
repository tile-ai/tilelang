"""End-to-end regressions for software-pipeline dependency planning."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_ISSUE_2595_N = 8
_ISSUE_2668_WIDTH = 32
_ISSUE_2668_NUM_TILES = 8
_EXPLICIT_PTX_BYTES = 16
_EXPLICIT_PTX_TILES = 4
_SHIFTED_WAR_EXTENT = 4
_SHIFTED_RAW_EXTENT = 8
_IMPLICIT_RNG_STATE_EXTENT = 8


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


@T.prim_func
def _explicit_ptx_lightweight_consumer_kernel(
    meta: T.Tensor((_EXPLICIT_PTX_TILES * _EXPLICIT_PTX_BYTES,), T.uint8),
    data: T.Tensor((_EXPLICIT_PTX_TILES * _EXPLICIT_PTX_BYTES,), T.uint8),
    meta_out: T.Tensor((_EXPLICIT_PTX_TILES,), T.uint8),
    data_out: T.Tensor((_EXPLICIT_PTX_TILES * _EXPLICIT_PTX_BYTES,), T.uint8),
):
    with T.Kernel(1, threads=32):
        meta_shared = T.alloc_shared((_EXPLICIT_PTX_BYTES,), T.uint8)
        data_shared = T.alloc_shared((_EXPLICIT_PTX_BYTES,), T.uint8)
        for k in T.Pipelined(_EXPLICIT_PTX_TILES, num_stages=3):
            T.ptx_cp_async(
                T.access_ptr(meta_shared[0], "w", _EXPLICIT_PTX_BYTES),
                T.access_ptr(meta[k * _EXPLICIT_PTX_BYTES], "r", _EXPLICIT_PTX_BYTES),
                _EXPLICIT_PTX_BYTES,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            meta_out[k] = meta_shared[0]
            T.copy(data[k * _EXPLICIT_PTX_BYTES], data_shared)
            T.copy(data_shared, data_out[k * _EXPLICIT_PTX_BYTES])


@T.prim_func
def _shifted_loop_carried_war_kernel(
    initial: T.Tensor((_SHIFTED_WAR_EXTENT + 1,), T.int32),
    source: T.Tensor((_SHIFTED_WAR_EXTENT,), T.int32),
    prev_out: T.Tensor((_SHIFTED_WAR_EXTENT,), T.int32),
    next_out: T.Tensor((_SHIFTED_WAR_EXTENT,), T.int32),
):
    with T.Kernel(1, threads=1):
        shared = T.alloc_shared((_SHIFTED_WAR_EXTENT + 1,), T.int32)
        T.copy(initial, shared)
        for k in T.Pipelined(_SHIFTED_WAR_EXTENT, num_stages=3):
            prev_out[k] = shared[k + 1]
            shared[k] = source[k]
            next_out[k] = shared[k]


@T.prim_func
def _shifted_loop_carried_raw_kernel(
    source: T.Tensor((_SHIFTED_RAW_EXTENT,), T.int32),
    current: T.Tensor((_SHIFTED_RAW_EXTENT,), T.int32),
    previous: T.Tensor((_SHIFTED_RAW_EXTENT,), T.int32),
):
    with T.Kernel(1, threads=1):
        shared = T.alloc_shared((_SHIFTED_RAW_EXTENT,), T.int32)
        for k in T.Pipelined(_SHIFTED_RAW_EXTENT, num_stages=3):
            shared[k] = source[k]
            current[k] = shared[k]
            if k > 0:
                previous[k] = shared[k - 1]


@T.prim_func
def _implicit_rng_state_pipeline_kernel(
    source: T.Tensor((_IMPLICIT_RNG_STATE_EXTENT,), T.int32),
    random_output: T.Tensor((2 * _IMPLICIT_RNG_STATE_EXTENT,), T.uint32),
    copy_output: T.Tensor((_IMPLICIT_RNG_STATE_EXTENT,), T.int32),
):
    with T.Kernel(1, threads=1):
        shared = T.alloc_shared((1,), T.int32)
        T.rng_init(42, 0, 0)
        for k in T.Pipelined(_IMPLICIT_RNG_STATE_EXTENT, num_stages=3):
            random_output[2 * k] = T.rng_rand()
            T.copy(source[k], shared)
            random_output[2 * k + 1] = T.rng_rand()
            copy_output[k] = shared[0]


@T.prim_func
def _implicit_rng_state_serial_kernel(
    source: T.Tensor((_IMPLICIT_RNG_STATE_EXTENT,), T.int32),
    random_output: T.Tensor((2 * _IMPLICIT_RNG_STATE_EXTENT,), T.uint32),
    copy_output: T.Tensor((_IMPLICIT_RNG_STATE_EXTENT,), T.int32),
):
    with T.Kernel(1, threads=1):
        shared = T.alloc_shared((1,), T.int32)
        T.rng_init(42, 0, 0)
        for k in T.serial(_IMPLICIT_RNG_STATE_EXTENT):
            random_output[2 * k] = T.rng_rand()
            T.copy(source[k], shared)
            random_output[2 * k + 1] = T.rng_rand()
            copy_output[k] = shared[0]


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


@tilelang.testing.requires_cuda_compute_version(8, 0)
def test_explicit_ptx_lightweight_consumer_preserves_serial_values():
    """An explicit PTX chain must not overwrite its unversioned shared slot."""

    kernel = tilelang.compile(
        _explicit_ptx_lightweight_consumer_kernel,
        out_idx=[2, 3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    size = _EXPLICIT_PTX_TILES * _EXPLICIT_PTX_BYTES
    meta = torch.arange(size, dtype=torch.uint8, device="cuda")
    data = torch.arange(size, dtype=torch.uint8, device="cuda")
    meta_out, data_out = kernel(meta, data)
    torch.cuda.synchronize()

    torch.testing.assert_close(meta_out, meta[::_EXPLICIT_PTX_BYTES])
    torch.testing.assert_close(data_out, data)


@tilelang.testing.requires_cuda
def test_shifted_loop_carried_war_preserves_live_in_values():
    """A read of shared[k + 1] must precede the next iteration's write."""

    kernel = tilelang.compile(
        _shifted_loop_carried_war_kernel,
        out_idx=[2, 3],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    initial = torch.arange(100, 100 + _SHIFTED_WAR_EXTENT + 1, dtype=torch.int32, device="cuda")
    source = torch.arange(10, 10 + _SHIFTED_WAR_EXTENT, dtype=torch.int32, device="cuda")
    prev_out, next_out = kernel(initial, source)
    torch.cuda.synchronize()

    torch.testing.assert_close(prev_out, initial[1:])
    torch.testing.assert_close(next_out, source)


@tilelang.testing.requires_cuda_compute_version(8, 0)
def test_shifted_loop_carried_raw_reads_previous_iteration_value():
    """A read of shared[k - 1] must use the preceding iteration's write."""

    kernel = tilelang.compile(
        _shifted_loop_carried_raw_kernel,
        out_idx=[1, 2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    source = torch.arange(
        10,
        10 + _SHIFTED_RAW_EXTENT,
        dtype=torch.int32,
        device="cuda",
    )
    current, previous = kernel(source)
    torch.cuda.synchronize()

    torch.testing.assert_close(current, source)
    torch.testing.assert_close(previous[1:], source[:-1])


@tilelang.testing.requires_cuda
def test_implicit_rng_state_preserves_serial_call_order():
    """Pipeline skew must not interleave calls that mutate implicit RNG state."""

    pass_configs = {tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}
    pipeline = tilelang.compile(
        _implicit_rng_state_pipeline_kernel,
        out_idx=[1, 2],
        pass_configs=pass_configs,
    )
    serial = tilelang.compile(
        _implicit_rng_state_serial_kernel,
        out_idx=[1, 2],
        pass_configs=pass_configs,
    )
    source = torch.arange(_IMPLICIT_RNG_STATE_EXTENT, dtype=torch.int32, device="cuda")
    actual_random, actual_copy = pipeline(source)
    expected_random, expected_copy = serial(source)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_copy, expected_copy, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_random.to(torch.int64),
        expected_random.to(torch.int64),
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    tilelang.testing.main()
