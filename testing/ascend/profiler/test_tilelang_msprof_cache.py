import ctypes
import struct
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import torch

import tilelang.testing
from tilelang.profiler.msprof import _get_cache_read_kernel, bench_with_msprof
from tilelang.profiler.torch_bench import _allocate_cache, _get_npu_l2_cache_flush


def _task_intervals(path):
    intervals = defaultdict(list)
    # Each 128-byte A5 PMU record has the full task ID at byte 4, the cycle
    # count at byte 8 and its device start/end timestamps at bytes 112/120.
    record = struct.Struct("<4xIQ96xQQ")
    for part in path.rglob("ffts_profile*"):
        if part.is_file() and not part.name.endswith(".done"):
            for task, cycles, start, end in record.iter_unpack(part.read_bytes()):
                if cycles:
                    intervals[task].append((start, end))
    return intervals


@tilelang.testing.requires_ascend
def test_msprof_uses_native_flush_with_caller_buffer(monkeypatch):
    native_flush = _get_npu_l2_cache_flush()
    if native_flush is None:
        pytest.skip("Requires the native L2 flush")
    cache = torch.zeros(8192, dtype=torch.int32, device="npu")
    flush = Mock(wraps=native_flush)
    monkeypatch.setattr("tilelang.profiler.torch_bench._get_npu_l2_cache_flush", lambda: flush)
    assert bench_with_msprof(lambda: cache.add_(1), cache, n_repeat=2) > 0
    assert flush.call_args_list == [call(None), call(None)]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8])
def test_msprof_with_2d_caller_buffer(dtype):
    cache = torch.empty((3, 2731), dtype=dtype, device="npu")
    work = torch.zeros(1, device="npu")
    assert bench_with_msprof(lambda: work.add_(1), cache, n_repeat=1) > 0


@tilelang.testing.requires_ascend
def test_msprof_keeps_user_launches_of_the_cache_kernel(monkeypatch):
    import torch_npu.profiler as profiler

    device = torch.device("npu", torch.npu.current_device())
    cache = _allocate_cache(device, 4)
    if not cache.numel():
        pytest.skip("The native L2 flush has no scratch kernel to exclude")
    work = torch.empty_like(cache)
    read = _get_cache_read_kernel(work.numel())
    streams = (torch.npu.current_stream(device), torch.npu.Stream(device=device))

    with pytest.raises(RuntimeError, match="No profiled kernels were recorded"):
        bench_with_msprof(lambda: None, cache, n_repeat=3)
    for stream in streams:
        with torch.npu.stream(stream):
            read(work)
    torch.npu.synchronize()

    real_profile = profiler.profile
    intervals = {}

    @contextmanager
    def capture(*args, **kwargs):
        with real_profile(*args, **kwargs) as prof:
            yield prof
        intervals.update(_task_intervals(Path(prof.prof_if.prof_path)))

    monkeypatch.setattr(profiler, "profile", capture)
    last_task = ctypes.CDLL("libascendcl.so").aclrtGetThreadLastTaskId
    last_task.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    user_tasks = []

    # Both user launches have the same name as the profiler's own cache read.
    def workload():
        for stream in streams:
            with torch.npu.stream(stream):
                read(work)
                task = ctypes.c_uint32()
                assert last_task(ctypes.byref(task)) == 0
                user_tasks.append(task.value)

    repeats = 3
    profile = bench_with_msprof(workload, cache, n_repeat=repeats, detailed=True)
    assert len(set(user_tasks)) == 2 * repeats
    assert all(task in intervals for task in user_tasks)
    expected = sum(end - start for task in user_tasks for start, end in intervals[task])
    assert all(end > start for task in user_tasks for start, end in intervals[task])
    assert profile.dur_ns == pytest.approx(expected / repeats, rel=0, abs=1e-6)

    # Only our two user launches and one scratch read are profiled per round.
    scratch_spans = (intervals[task] for task in intervals.keys() - set(user_tasks))
    flushes = sorted((min(start for start, _ in spans), max(end for _, end in spans)) for spans in scratch_spans)
    assert len(flushes) == repeats
    for i, (flush_start, flush_end) in enumerate(flushes):
        work_spans = [span for task in user_tasks[2 * i : 2 * i + 2] for span in intervals[task]]
        assert flush_start < flush_end <= min(start for start, _ in work_spans)
        if i + 1 < repeats:
            assert max(end for _, end in work_spans) <= flushes[i + 1][0]
