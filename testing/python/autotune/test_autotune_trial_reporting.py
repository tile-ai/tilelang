"""CPU contract tests for the opt-in AutoTuner trial observer.

Compilation and benchmarking are injected; these tests do not exercise a device.
"""

import concurrent.futures
import inspect
import json
import logging
import threading
from types import SimpleNamespace

import pytest

from tilelang.autotuner import tuner as tuner_module
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.report import TrialReporter


def _kernel(block=1):
    return block


class _CompiledKernel:
    prim_func = None

    def __init__(self, block):
        self.block = block

    def update_tuner_result(self, **_kwargs):
        return self

    def get_kernel_source(self):
        return "injected kernel"


class _Progress:
    def __init__(self, total, desc):
        self.n = 0
        self.total = total

    def update(self, count):
        self.n += count

    def set_postfix(self, _value):
        pass

    def refresh(self):
        pass

    def close(self):
        pass

    @staticmethod
    def write(_value):
        pass


class _Tuner(AutoTuner):
    def __init__(self, statuses, *, grouped_failure=False):
        super().__init__(_kernel, [{"block": i + 1} for i in range(len(statuses))])
        self.statuses = statuses
        self.grouped_failure = grouped_failure
        self._lock = threading.Lock()
        self._memory_cache = {}
        self.compile_args = SimpleNamespace(out_idx=None, execution_backend="torch")
        self.profile_args = SimpleNamespace(skip_check=True, ref_prog=None)
        self.cache_key = None
        self.disk_result = None
        self.pool_closed = False
        self.workers = []

    def generate_cache_key(self, _parameters, _extra_parameters):
        return self.cache_key

    def _load_result_from_disk(self, _key):
        return self.disk_result

    def _ensure_jit_functions(self):
        return lambda **kwargs: _CompiledKernel(kwargs.get("block", 1)), lambda **_kwargs: None

    def _validate_input_supply_requirements(self, *_args):
        pass

    def _resolve_grouped_compile_mode(self, **_kwargs):
        return "llvm", "torch", self.grouped_failure, ""

    def _resolve_benchmark_devices(self, **kwargs):
        return kwargs["benchmark_multi_gpu"], [0, 1] if kwargs["benchmark_multi_gpu"] else [0]

    def _prepare_compile_execution(self, config_args, **_kwargs):
        futures = []
        mapping = {}
        groups = (
            [list(enumerate(config_args))] if self.grouped_failure else [[(i, config_args[i])] for i in reversed(range(len(config_args)))]
        )
        for group in groups:
            future = concurrent.futures.Future()
            if self.grouped_failure:
                future.set_exception(ValueError("group failed"))
            else:
                idx, config = group[0]
                error = ValueError("compile failed") if self.statuses[idx] == "compile_error" else None
                future.set_result([(idx, config, None if error else _CompiledKernel(config["block"]), error)])
            futures.append(future)
            mapping[future] = group
        pool = SimpleNamespace(shutdown=lambda: setattr(self, "pool_closed", True))
        return pool, futures, mapping, "injected compile"

    def _benchmark_worker_loop(self, _device, tasks, results, start, *_args):
        self.workers.append(threading.current_thread())
        start.wait()
        while (task := tasks.get()) is not None:
            kernel, config, idx = task
            status = self.statuses[idx]
            results.put(
                (
                    idx,
                    config,
                    kernel,
                    float(idx + 1) if status == "ok" else None,
                    None,
                    None if status == "ok" else "error" if status == "benchmark_error" else "timeout",
                    "injected benchmark error" if status == "benchmark_error" else "",
                )
            )


@pytest.fixture(autouse=True)
def _quiet_run(monkeypatch):
    monkeypatch.setattr(tuner_module, "_init_logger_handlers", lambda: None)
    monkeypatch.setattr(tuner_module, "tqdm", _Progress)


@pytest.mark.parametrize("pipeline,multi_gpu", [(False, False), (True, False), (False, True), (True, True)])
def test_each_outcome_once_on_aggregation_thread(pipeline, multi_gpu):
    tuner = _Tuner(["ok", "compile_error", "benchmark_error", "timeout", "ok"])
    records = []
    owner = threading.get_ident()

    def sink(record):
        assert threading.get_ident() == owner
        records.append(json.loads(json.dumps(record)))

    result = tuner.run(on_trial=sink, timeout=0, use_pipeline=pipeline, benchmark_multi_gpu=multi_gpu)
    assert (result.config, result.latency) == ({"block": 1}, 1.0)
    assert len(records) == len(tuner.configs)
    assert {record["index"] for record in records} == set(range(len(tuner.configs)))
    assert {record["index"]: record["status"] for record in records} == {
        0: "ok",
        1: "compile_error",
        2: "benchmark_error",
        3: "timeout",
        4: "ok",
    }
    assert all(set(record) == {"status", "index", "config", "latency_ms", "validation", "error"} for record in records)
    assert tuner.pool_closed
    assert all(not worker.is_alive() for worker in tuner.workers)


def test_detached_nested_config_and_bounded_error():
    original = {"nested": {"values": [1]}}
    records = []
    reporter = TrialReporter(records.append, logging.getLogger(__name__))
    reporter.emit("compile_error", 0, original, error="x" * 4096)
    records[0]["config"]["nested"]["values"].append(2)
    assert original == {"nested": {"values": [1]}}
    assert len(records[0]["error"]) == 2048


def test_sink_failure_keeps_winner_and_disables_later_calls():
    baseline = _Tuner(["ok", "ok"]).run(timeout=0)
    calls = []

    def broken(record):
        calls.append(record)
        raise OSError("sink failed")

    result = _Tuner(["ok", "ok"]).run(on_trial=broken, timeout=0)
    assert (result.config, result.latency) == (baseline.config, baseline.latency)
    assert len(calls) == 1


@pytest.mark.parametrize("broken_error", [False, True])
def test_serialization_or_error_conversion_disables_reporting(broken_error):
    if broken_error:

        class BrokenError:
            def __str__(self):
                raise ValueError("conversion failed")

        error = BrokenError()
        config = {"value": 1}
    else:
        error = None
        config = {"value": float("nan")}
    records = []
    reporter = TrialReporter(records.append, logging.getLogger(__name__))
    reporter.emit("compile_error", 0, config, error=error)
    reporter.emit("ok", 1, {"value": 1})
    assert records == []
    assert reporter.sink is None


def test_keyboard_interrupt_propagates_through_cleanup():
    tuner = _Tuner(["compile_error"])

    def interrupted(_record):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        tuner.run(on_trial=interrupted, timeout=0)
    assert tuner.pool_closed
    assert all(not worker.is_alive() for worker in tuner.workers)


def test_unsupported_config_object_disables_reporting_without_changing_result():
    shared = object()
    tuner = _Tuner(["ok"])
    tuner.configs[0]["pass_configs"] = {"custom": shared}
    records = []
    result = tuner.run(on_trial=records.append, timeout=0)
    assert result.config["pass_configs"]["custom"] is shared
    assert records == []


@pytest.mark.parametrize("disk", [False, True])
def test_cache_hit_is_marker_outside_lock_without_trial_history(disk):
    tuner = _Tuner(["ok"])
    tuner.cache_key = "fixed"
    cached = SimpleNamespace(func=None)
    if disk:
        tuner.disk_result = cached
    else:
        tuner._memory_cache["fixed"] = cached
    records = []

    def sink(record):
        assert not tuner._lock.locked()
        records.append(record)

    assert tuner.run(on_trial=sink) is cached
    assert records == [
        {
            "status": "cache_hit",
            "index": None,
            "config": None,
            "latency_ms": None,
            "validation": "not_run",
            "error": None,
        }
    ]
    assert tuner.workers == []


def test_direct_jit_is_marker_without_trial_history():
    tuner = _Tuner(["ok"])
    tuner.set_kernel_parameters(((1,), ()), inspect.signature(_kernel).parameters)
    records = []
    tuner.run(on_trial=records.append)
    assert [record["status"] for record in records] == ["direct_jit"]
    assert records[0]["index"] is None
    assert tuner.workers == []


@pytest.mark.parametrize(
    "skip,reference,expected",
    [(True, _kernel, "not_run"), (False, None, "not_run"), (False, _kernel, "passed")],
)
def test_validation_label(skip, reference, expected):
    tuner = _Tuner(["ok"])
    tuner.profile_args = SimpleNamespace(skip_check=skip, ref_prog=reference)
    records = []
    tuner.run(on_trial=records.append, timeout=0)
    assert records[0]["validation"] == expected


def test_grouped_compile_failure_reports_each_index_and_cleans_up():
    tuner = _Tuner(["compile_error", "compile_error"], grouped_failure=True)
    records = []
    with pytest.raises(RuntimeError, match="No configuration"):
        tuner.run(on_trial=records.append, timeout=0, enable_grouped_compile=True)
    assert {record["index"] for record in records} == {0, 1}
    assert all(record["status"] == "compile_error" for record in records)
    assert tuner.pool_closed
    assert all(not worker.is_alive() for worker in tuner.workers)


def test_early_stop_with_observer_is_rejected():
    with pytest.raises(ValueError, match="early_stop"):
        _Tuner(["ok"]).run(on_trial=lambda _record: None, early_stop=True)
    assert _Tuner(["ok"]).run(early_stop=True, timeout=0).latency == 1.0
