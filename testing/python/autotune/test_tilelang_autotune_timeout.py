"""Exercise real subprocesses; never launch a permanently stuck GPU kernel."""

import concurrent.futures
import json
import os
from pathlib import Path
import queue
import threading
import time

import cloudpickle
import pytest

from tilelang.autotuner import AutoTuner
from tilelang.autotuner._process import ProcessWorker, WorkerError
from tilelang.autotuner.param import CompileArgs
from tilelang.autotuner.tuner import _BenchmarkWorkerState
import tilelang.autotuner.tuner as tuner_module


def _factory(events, actions, ignore_term=False, startup_error=False, fail_restart=False):
    # Local functions are serialized by value, without importing this test module
    # or TileLang in the child. This also tests notebook-style callable transport.
    def create():
        import json
        import os
        import signal
        import time

        if startup_error:
            raise RuntimeError("startup failed")
        if fail_restart and Path(events).exists():
            raise RuntimeError("recovery unavailable")
        if ignore_term:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        count = 0

        def run(request):
            nonlocal count
            count += 1
            idx = int(Path(request["path"]).name)
            action = actions[idx]
            with open(events, "a") as f:
                f.write(json.dumps([idx, os.getpid(), count]) + "\n")
            # Both Python and native stdout must be safe for user callbacks.
            print("callback output")
            os.write(1, b"native callback output\n")
            if action == "timeout":
                while True:
                    time.sleep(1)
            if action == "crash":
                os._exit(17)
            if action == "error":
                raise RuntimeError("benchmark failed")
            return float(count), None

        return run

    return cloudpickle.dumps(create)


@pytest.fixture
def processes(monkeypatch):
    import tilelang.autotuner._process as process_module

    children = []
    popen = process_module.subprocess.Popen

    def track(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(process_module.subprocess, "Popen", track)
    monkeypatch.setattr(ProcessWorker, "SHUTDOWN_TIMEOUT", 0.2)
    yield children
    live = [child for child in children if child.poll() is None]
    for child in live:
        child.kill()
        child.wait(timeout=3)
    assert not live, "autotuning left a live benchmark process"


def _start_supervisor(tmp_path, actions, **factory_kwargs):
    tasks, results = queue.Queue(), queue.Queue()
    start, stop = threading.Event(), threading.Event()
    events = tmp_path / "events.jsonl"
    for idx in range(len(actions)):
        tasks.put((object(), {}, idx, tmp_path / str(idx)))
    tasks.put(None)
    start.set()
    worker = threading.Thread(
        target=AutoTuner(lambda: None, [])._benchmark_worker_loop,
        args=(0, tasks, results, start, "c", None, 0.15, _BenchmarkWorkerState(), stop, _factory(str(events), actions, **factory_kwargs)),
    )
    worker.start()
    return worker, results, stop, events


@pytest.mark.parametrize("actions", [("timeout", "ok"), ("ok", "timeout"), ("crash", "ok"), ("error", "ok"), ("ok", "ok")])
def test_worker_recovery_and_last_trial_cleanup(tmp_path, processes, actions):
    worker, results, stop, events = _start_supervisor(tmp_path, actions)
    try:
        for action in actions:
            result = results.get(timeout=10)
            assert result[5] == ({"ok": None, "timeout": "timeout"}.get(action, "error"))
            if action != "ok":
                assert processes[0].poll() is not None
        worker.join(timeout=3)
        assert not worker.is_alive()
        records = [json.loads(line) for line in events.read_text().splitlines()]
        if actions[0] == "ok":
            assert records[0][1] == records[1][1]
            assert records[1][2] == 2
        else:
            assert records[0][1] != records[1][1]
            assert records[1][2] == 1  # Fresh input/reference state after recovery.
    finally:
        stop.set()
        worker.join(timeout=3)


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM escalation is POSIX-specific")
def test_timeout_escalates_to_kill(tmp_path, processes):
    before = time.monotonic()
    worker, results, stop, _ = _start_supervisor(tmp_path, ["timeout"], ignore_term=True)
    try:
        assert results.get(timeout=10)[5] == "timeout"
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert processes[0].returncode == -9
        assert time.monotonic() - before < 5
    finally:
        stop.set()
        worker.join(timeout=3)


def test_parent_cancellation_reaps_active_call(tmp_path, processes):
    worker, _, stop, events = _start_supervisor(tmp_path, ["timeout"])
    try:
        deadline = time.monotonic() + 10
        while not events.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert events.exists()
        stop.set()
        worker.join(timeout=3)
        assert not worker.is_alive()
    finally:
        stop.set()
        worker.join(timeout=3)


def test_startup_failure_is_fatal(tmp_path, processes):
    worker, results, stop, _ = _start_supervisor(tmp_path, ["ok"], startup_error=True)
    try:
        result = results.get(timeout=10)
        assert result[5] == "fatal"
        assert "startup failed" in result[6]
        worker.join(timeout=3)
        assert stop.is_set()
    finally:
        stop.set()
        worker.join(timeout=3)


def test_startup_has_a_bounded_deadline(tmp_path, monkeypatch, processes):
    def create():
        import time

        time.sleep(60)

    monkeypatch.setattr(ProcessWorker, "STARTUP_TIMEOUT", 0.15)
    worker = ProcessWorker(cloudpickle.dumps(create), threading.Event())
    with pytest.raises(TimeoutError):
        worker.start()
    assert worker._process is None


class _FakeKernel:
    prim_func = None
    execution_backend = "tvm_ffi"

    def get_kernel_source(self):
        return ""

    def update_tuner_result(self, **kwargs):
        return self


def _kernel(threads):
    pass


def _fake_tuner(monkeypatch, tmp_path, actions, **factory_kwargs):
    tuner = AutoTuner(_kernel, [{"threads": i} for i in range(len(actions))])
    tuner.compile_args = CompileArgs(execution_backend="tvm_ffi", target="c")
    monkeypatch.setattr(tuner, "generate_cache_key", lambda *args: None)
    monkeypatch.setattr(tuner, "_ensure_jit_functions", lambda: (None, lambda **kwargs: None))
    monkeypatch.setattr(tuner, "_validate_input_supply_requirements", lambda *args: None)
    monkeypatch.setattr(tuner, "_resolve_grouped_compile_mode", lambda **kwargs: ("c", None, False, None))
    monkeypatch.setattr(tuner, "_resolve_benchmark_devices", lambda **kwargs: (False, [0]))

    def prepare(**kwargs):
        pool = concurrent.futures.ThreadPoolExecutor(1)
        items = list(enumerate(tuner.configs))
        future = pool.submit(lambda: [(idx, config, _FakeKernel(), None) for idx, config in items])
        return pool, [future], {future: items}, "compile"

    monkeypatch.setattr(tuner, "_prepare_compile_execution", prepare)
    monkeypatch.setattr(tuner_module, "export_kernel", lambda *args: None)
    factory = _factory(str(tmp_path / "events.jsonl"), actions, **factory_kwargs)
    monkeypatch.setattr(tuner_module, "make_worker_factory", lambda *args: factory)
    return tuner


@pytest.mark.parametrize("pipeline", [False, True])
def test_run_finishes_after_last_timeout(tmp_path, monkeypatch, processes, pipeline):
    tuner = _fake_tuner(monkeypatch, tmp_path, ["ok", "timeout"])
    trials = []
    result = tuner.run(timeout=0.15, use_pipeline=pipeline, on_trial=trials.append)
    assert result.config == {"threads": 0}
    assert [trial["status"] for trial in trials] == ["ok", "timeout"]


def test_all_timeouts_fail_without_hanging(tmp_path, monkeypatch, processes):
    tuner = _fake_tuner(monkeypatch, tmp_path, ["timeout"])
    with pytest.raises(RuntimeError, match="No configuration"):
        tuner.run(timeout=0.15)


def test_failed_recovery_aborts_even_with_a_best_result(tmp_path, monkeypatch, processes):
    tuner = _fake_tuner(monkeypatch, tmp_path, ["ok", "timeout", "ok"], fail_restart=True)
    with pytest.raises(WorkerError, match="recovery unavailable"):
        tuner.run(timeout=0.15)


def test_cleanup_failure_aborts_run(tmp_path, monkeypatch, processes):
    tuner = _fake_tuner(monkeypatch, tmp_path, ["ok", "timeout"])
    close = ProcessWorker.close

    def fail_cleanup(self, *args, **kwargs):
        close(self, *args, **kwargs)
        raise WorkerError("simulated cleanup failure")

    monkeypatch.setattr(ProcessWorker, "close", fail_cleanup)
    with pytest.raises(WorkerError, match="simulated cleanup failure"):
        tuner.run(timeout=0.15)


def test_parent_exception_closes_workers(tmp_path, monkeypatch, processes):
    tuner = _fake_tuner(monkeypatch, tmp_path, ["ok", "timeout"])

    def interrupt(_record):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        tuner.run(timeout=10, on_trial=interrupt)
