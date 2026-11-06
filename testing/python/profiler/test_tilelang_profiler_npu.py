import ctypes
import os
from pathlib import Path
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tilelang.profiler import msprof, npu, torch_bench


@pytest.fixture(autouse=True)
def clear_binding_cache():
    binding = npu._get_device_l2_cache_flush
    binding.cache_clear()
    yield
    binding.cache_clear()


def test_flush_binding_is_cached_and_synchronizes_before_each_call(monkeypatch):
    calls = []
    loads = []

    class Flush:
        def __call__(self, reserved):
            calls.append(("flush", reserved))
            return 0

    flush = Flush()

    def load(name):
        loads.append(name)
        return SimpleNamespace(aclrtDeviceL2CacheFlush=flush)

    monkeypatch.setattr(npu.ctypes, "CDLL", load)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: calls.append("sync")), raising=False)
    npu.flush_l2_cache()
    npu.flush_l2_cache()

    assert loads == ["libascendcl.so"]
    assert flush.argtypes == [ctypes.c_void_p]
    assert flush.restype is ctypes.c_int
    assert calls == ["sync", ("flush", None), "sync", ("flush", None)]


@pytest.mark.parametrize("missing", ["library", "symbol"])
def test_missing_flush_api_is_reported(monkeypatch, missing):
    def load(name):
        if missing == "library":
            raise OSError("CANN is not installed")
        return SimpleNamespace()

    monkeypatch.setattr(npu.ctypes, "CDLL", load)
    with pytest.raises(RuntimeError, match="requires a CANN runtime providing aclrtDeviceL2CacheFlush"):
        npu.flush_l2_cache()


def test_flush_error_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(npu, "_get_device_l2_cache_flush", lambda: lambda reserved: 207000)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    with pytest.raises(RuntimeError, match="aclrtDeviceL2CacheFlush failed with error code 207000"):
        npu.flush_l2_cache()


def test_event_timing_excludes_synchronous_flush(monkeypatch):
    now = 0.0
    flushes = []

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            self.time = now

        def elapsed_time(self, end):
            return end.time - self.time

    def flush():
        nonlocal now
        flushes.append(now)
        now += 100.0

    def function():
        nonlocal now
        now += 2.0

    monkeypatch.setattr(npu, "flush_l2_cache", flush)
    monkeypatch.setattr(torch_bench, "Event", Event)
    monkeypatch.setattr(torch_bench, "_cuda_synchronize", lambda device: None)
    assert torch_bench.bench_with_cuda_events(function, None, 3, None, "mean", None) == 2.0
    assert len(flushes) == 3


def test_msprof_flushes_before_each_target_and_reports_only_kernel_time(monkeypatch):
    calls = []

    class Profile:
        def __init__(self, **kwargs):
            path = Path(os.environ["ASCEND_WORK_PATH"]) / "PROF_fake"
            path.mkdir()
            (path / "hash_dic.slice_0").write_text("123:target_kernel\n")
            tasks, records = [], []
            for seq in (1, 2):
                task = [0] * 8
                task[3], task[5] = seq << 32, 123
                tasks.append(struct.pack("<8q", *task))
                record = [0] * 16
                record[0], record[1], record[2] = seq << 32, 100, 1 << 31
                record[14], record[15] = 1000, 11000
                records.append(struct.pack("<16q", *record))
            (path / "task_track.slice_0").write_bytes(b"".join(tasks))
            (path / "ffts_profile.slice_0").write_bytes(b"".join(records))
            self.prof_if = SimpleNamespace(prof_path=str(path))

        def __enter__(self):
            calls.append("profile_enter")
            return self

        def __exit__(self, *args):
            calls.append("profile_exit")

        def step(self):
            calls.append("step")

    profiler = ModuleType("torch_npu.profiler")
    profiler.profile = Profile
    profiler.ProfilerActivity = SimpleNamespace(NPU="NPU")
    profiler.ProfilerLevel = SimpleNamespace(Level1=1)
    profiler.AiCMetrics = SimpleNamespace(PipeUtilization=1)
    profiler.schedule = lambda **kwargs: None
    profiler._ExperimentalConfig = lambda **kwargs: None
    package = ModuleType("torch_npu")
    package.profiler = profiler
    monkeypatch.setitem(sys.modules, "torch_npu", package)
    monkeypatch.setitem(sys.modules, "torch_npu.profiler", profiler)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: calls.append("sync")), raising=False)
    monkeypatch.setattr(msprof, "flush_l2_cache", lambda: calls.append("flush"))

    assert msprof.bench_with_msprof(lambda: calls.append("target"), None, 2) == pytest.approx(0.01)
    assert calls == ["flush", "profile_enter", "flush", "target", "flush", "target", "sync", "step", "profile_exit"]
