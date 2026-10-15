import queue
import threading

from tilelang.autotuner import AutoTuner
from tilelang.autotuner.tuner import _BenchmarkWorkerState


def test_benchmark_worker_reports_timeout_without_signals():
    tuner = AutoTuner(lambda: None, configs=[{}])
    worker_queue = queue.Queue()
    result_queue = queue.Queue()
    start_event = threading.Event()
    release_benchmark = threading.Event()

    def benchmark_target(**_kwargs):
        release_benchmark.wait(timeout=1)
        return 1.0, None

    kernel = object()
    worker_queue.put((kernel, {}, 0))
    worker_queue.put(None)
    start_event.set()

    try:
        tuner._benchmark_worker_loop(
            worker_device=0,
            worker_queue=worker_queue,
            result_queue=result_queue,
            start_event=start_event,
            target_kind="c",
            benchmark_target=benchmark_target,
            timeout=0.01,
            worker_state=_BenchmarkWorkerState(),
        )
    finally:
        release_benchmark.set()

    idx, config, result_kernel, latency, ref_latency, status, error_text = result_queue.get_nowait()
    assert (idx, config) == (0, {})
    assert result_kernel is kernel
    assert latency is None
    assert ref_latency is None
    assert status == "timeout"
    assert error_text == ""


def test_benchmark_worker_discards_inputs_after_timeout():
    tuner = AutoTuner(lambda: None, configs=[{}])
    worker_queue = queue.Queue()
    result_queue = queue.Queue()
    start_event = threading.Event()
    release_benchmark = threading.Event()
    slow_finished = threading.Event()
    cached_inputs = [object()]
    worker_state = _BenchmarkWorkerState(jit_input_tensors=cached_inputs)
    next_inputs = []

    def benchmark_target(*, jit_kernel, benchmark_state, benchmark_device):
        if jit_kernel == "slow":
            try:
                assert benchmark_state.jit_input_tensors is cached_inputs
                assert release_benchmark.wait(timeout=3)
                benchmark_state.jit_input_tensors = cached_inputs
            finally:
                slow_finished.set()
        else:
            next_inputs.append(benchmark_state.jit_input_tensors)
        return 1.0, None

    worker_queue.put(("slow", {}, 0))
    worker_queue.put(("next", {}, 1))
    worker_queue.put(None)
    start_event.set()
    worker = threading.Thread(
        target=tuner._benchmark_worker_loop,
        kwargs=dict(
            worker_device=0,
            worker_queue=worker_queue,
            result_queue=result_queue,
            start_event=start_event,
            target_kind="c",
            benchmark_target=benchmark_target,
            timeout=0.01,
            worker_state=worker_state,
        ),
    )
    worker.start()
    try:
        assert result_queue.get(timeout=3)[5] == "timeout"
        assert result_queue.get(timeout=3)[5] is None
        assert next_inputs == [None]
    finally:
        release_benchmark.set()
        worker.join(timeout=3)
        assert slow_finished.wait(timeout=3)

    assert not worker.is_alive()
    assert worker_state.jit_input_tensors is None
